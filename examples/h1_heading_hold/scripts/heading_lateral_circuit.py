"""Write a heading-hold + lateral-hold circuit into freed hidden units of the H1_2 squat policy.

All quantities come from the newest observation sample, dt = 0.02 s.

Heading hold (physics only: gyro + projected gravity):
    rate   = -(w_y g_y + w_z g_z) (1 + g_x^2)          Euler yaw rate
    act    = 1 if the gait clock runs and |wz_cmd| ~ 0, else 0
    theta' = clamp(theta + dt (wz_cmd - rate), +-L_theta act)     heading error      -> action 25
    I'     = clamp(I + dt theta,              +-L_I act)          its integral       -> action 26
    wz seen by the original network = wz_cmd + K_theta theta + K_I I

Lateral hold (needs an estimator fitted in simulation):
    v_hat  = w . h1 + b                 linear readout of the first hidden layer: lateral velocity
    act_l  = act and |vy_cmd| ~ 0
    p'     = clamp(p + dt v_hat, +-L_p act_l)                     lateral displacement -> action 21
    vy seen by the original network = vy_cmd - K_p p

The registers live in action channels (left/right wrist yaw, left wrist roll) and are read back
through last_action.
"""
import numpy as np

GYRO, GRAV, CMD, LA, GAIT = 12, 27, 42, 423, 458      # newest-sample column offsets
VY_COLS = [31, 34, 37, 40, 43]
WZ_COLS = [32, 35, 38, 41, 44]
la_cols = lambda j: [315 + 27 * k + j for k in range(5)]
R1, R2, R3 = 25, 26, 21                                # left / right wrist yaw, left wrist roll

P = dict(Kth=3.0, KI=3.0, s1=2.0, s2=2.0, Lth=0.35, LI=0.3, dt=0.02, m25=0.027, m26=0.067,
         Kp=2.0, s3=4.0, Lp=0.15, m21=-0.056,
         S=200.0,     # sharpness of the clamps
         Sg=2000.0,   # sharpness of the |.| units of the gates
         B=1.0,       # bias of the linear pass-through units
         alpha=2.0, beta=0.4, beta2=0.1, sn=10.0, sx=5.0, gq=0.15)

N_UNITS = (23, 13, 8)   # circuit units used in hidden layers 1..3


def build(L_pruned, freed, p=P, lateral=None):
    """L_pruned: 4 (W,b) with freed units zeroed; freed: [idx_l1, idx_l2, idx_l3];
    lateral: None (register 21 stays at 0) or dict(w=(512,), b=float) of the velocity readout."""
    (W0, b0), (W1, b1), (W2, b2), (W3, b3) = [(W.astype(np.float64).copy(), b.astype(np.float64).copy()) for W, b in L_pruned]
    f1, f2, f3 = [list(f) for f in freed]
    a, be, be2, sn, sx, gq, S, Sg, B = p["alpha"], p["beta"], p["beta2"], p["sn"], p["sx"], p["gq"], p["S"], p["Sg"], p["B"]
    s1, s2, s3, dt = p["s1"], p["s2"], p["s3"], p["dt"]

    # ---- first layer: constant-fold the original use of the register channels, add the folds ----
    for j, m in ((R1, p["m25"]), (R2, p["m26"]), (R3, p["m21"])):
        cols = la_cols(j)
        b0 += W0[:, cols].sum(1) * m
        W0[:, cols] = 0
    for k in range(5):
        W0[:, la_cols(R1)[k]] += p["Kth"] / s1 * W0[:, WZ_COLS[k]]
        W0[:, la_cols(R2)[k]] += p["KI"] / s2 * W0[:, WZ_COLS[k]]
        W0[:, la_cols(R3)[k]] += -p["Kp"] / s3 * W0[:, VY_COLS[k]]

    class Lin(dict):
        const = 0.0
        def add(self, other, k=1.0):
            for u, c in other.items(): self[u] = self.get(u, 0.0) + k * c
            self.const += k * other.const
            return self
    def lin(pairs=None, const=0.0):
        e = Lin(pairs or {}); e.const = const; return e

    # ---- layer-1 circuit units ------------------------------------------------------------------
    it = iter(f1)
    def unit1(cols_w, bias):
        u = next(it); assert not W0[u].any() and not W1[:, u].any()
        for c, w in cols_w: W0[u, c] = w
        b0[u] = bias
        return u
    def absval(col):                                   # |x| ~ (elu(Sg x) + elu(-Sg x)) / Sg
        e = lin()
        for sgn in (1, -1): e.add(lin({unit1([(col, sgn * Sg)], 0.0): 1.0 / Sg}))
        return e
    u = unit1([(LA + R1, 1.0)], B); r1 = lin({u: 1.0}, -B)
    u = unit1([(LA + R2, 1.0)], B); r2 = lin({u: 1.0}, -B)
    u = unit1([(LA + R3, 1.0)], B); r3 = lin({u: 1.0}, -B)
    absum = lin().add(absval(GAIT)).add(absval(GAIT + 1))
    abswz = absval(CMD + 2)
    absvy = absval(CMD + 1)
    u = unit1([(CMD + 2, 0.5), (GYRO + 2, -0.5 / 0.2)], 4.0); e_lin = lin({u: 2.0}, -8.0)   # wz - w_z
    u = unit1([(GYRO + 2, 0.5 / 0.2)], 4.0); wz_gyro = lin({u: 2.0}, -8.0)                  # w_z
    def square_units(cols_w, const):                   # elu(x)+elu(-x) ~ x^2/2
        e = lin()
        for sgn in (1, -1):
            e.add(lin({unit1([(c, sgn * w) for c, w in cols_w], sgn * const): 1.0}))
        return e
    def product(colsA, cA, colsB, cB, scale):
        plus = square_units([(c, w / sn) for c, w in colsA + colsB], (cA + cB) / sn)
        minus = square_units([(c, w / sn) for c, w in colsA] + [(c, -w / sn) for c, w in colsB], (cA - cB) / sn)
        return lin().add(plus, scale * sn * sn / 2).add(minus, -scale * sn * sn / 2)
    wy_gy = product([(GYRO + 1, 1 / 0.2 / a)], 0.0, [(GRAV + 1, 1 / be)], 0.0, a * be)
    wz_1gz = product([(GYRO + 2, 1 / 0.2 / a)], 0.0, [(GRAV + 2, 1 / be2)], 1 / be2, a * be2)
    gx2 = lin().add(square_units([(GRAV, 1 / sx)], 0.0), 2 * sx * sx)
    r0 = lin().add(wz_gyro).add(wz_1gz, -1.0).add(wy_gy, -1.0)
    E0 = lin().add(e_lin).add(wz_1gz, 1.0).add(wy_gy, 1.0)
    assert next(it, None) is None, "layer-1 unit count"

    # ---- layer-2 circuit units ------------------------------------------------------------------
    it2 = iter(f2)
    def unit2(e, k=1.0, bias=0.0):
        u = next(it2); assert not W1[u].any() and not W2[:, u].any()
        for v, c in e.items(): W1[u, v] = k * c
        b1[u] = k * e.const + bias
        return u
    u = unit2(r1, 1.0, B); r1_2 = lin({u: 1.0}, -B)
    u = unit2(r2, 1.0, B); r2_2 = lin({u: 1.0}, -B)
    u = unit2(r3, 1.0, B); r3_2 = lin({u: 1.0}, -B)
    z = lin(None, -4.5).add(absum, 6.0).add(abswz, -120.0)       # >= 1 when active, <= 0 otherwise
    ua = unit2(z, S); ub = unit2(z, S, -S)
    act = lin({ua: 1.0 / S, ub: -1.0 / S})
    zl = lin().add(z).add(absvy, -120.0)
    ua = unit2(zl, S); ub = unit2(zl, S, -S)
    act_l = lin({ua: 1.0 / S, ub: -1.0 / S})
    u = unit2(E0, 0.5, 4.0); E0_2 = lin({u: 2.0}, -8.0)
    vhat = lin()
    if lateral is not None:
        w = np.asarray(lateral["w"], np.float64); assert not w[f1].any()
        vhat = lin({int(i): float(w[i]) for i in np.nonzero(w)[0]}, float(lateral["b"]))
    u = unit2(vhat, 0.5, 4.0); vhat_2 = lin({u: 2.0}, -8.0)
    def sq2(e):
        out = lin()
        for sgn in (1, -1): out.add(lin({unit2(e, sgn): 1.0}))
        return out
    A_ = lin().add(r0, 1 / a / sn); Bq = lin().add(gx2, 1 / gq / sn)
    plus = sq2(lin().add(A_).add(Bq)); minus = sq2(lin().add(A_).add(Bq, -1.0))
    r0q = lin().add(plus, a * gq * sn * sn / 2).add(minus, -a * gq * sn * sn / 2)
    e_2 = lin().add(E0_2).add(r0q, -1.0)                         # wz - rate
    assert next(it2, None) is None, "layer-2 unit count"

    # ---- layer-3 circuit units and output rows ---------------------------------------------------
    it3 = iter(f3)
    def unit3(e, k=1.0, bias=0.0):
        u = next(it3); assert not W2[u].any() and not W3[:, u].any()
        for v, c in e.items(): W2[u, v] = k * c
        b2[u] = k * e.const + bias
        return u
    ua3 = unit3(act, 1.0, 0.5); ul3 = unit3(act_l, 1.0, 0.5)
    for row in (R1, R2, R3): W3[row] = 0; b3[row] = 0
    regs = [(R1, lin().add(r1_2).add(e_2, s1 * dt), s1 * p["Lth"], act, ua3),
            (R2, lin().add(r2_2).add(r1_2, s2 * dt / s1), s2 * p["LI"], act, ua3),
            (R3, lin().add(r3_2).add(vhat_2, s3 * dt), s3 * p["Lp"], act_l, ul3)]
    for row, x, Lreg, g, ug in regs:
        lo = unit3(lin().add(x).add(g, Lreg), S)                 # elu(S (x + L act))
        hi = unit3(lin().add(x).add(g, -Lreg), S)                # elu(S (x - L act))
        if row == R3 and lateral is None: continue
        W3[row, lo] = 1 / S; W3[row, hi] = -1 / S
        W3[row, ug] = -(Lreg + 1 / S); b3[row] = 0.5 * (Lreg + 1 / S)
    assert next(it3, None) is None, "layer-3 unit count"
    return [(W.astype(np.float32), b.astype(np.float32)) for W, b in ((W0, b0), (W1, b1), (W2, b2), (W3, b3))]


def gates(o):
    """Exact activation gates for one observation: (act, act_l)."""
    z = 6 * (abs(o[GAIT]) + abs(o[GAIT + 1])) - 4.5 - 120 * abs(o[CMD + 2])
    zl = z - 120 * abs(o[CMD + 1])
    return min(1.0, max(0.0, z)), min(1.0, max(0.0, zl))


def reference(o, p=P, vhat=None):
    """Exact register update for one observation; returns (a25, a26[, a21])."""
    cl = lambda x, l: min(max(x, -l), l)
    th = o[LA + R1] / p["s1"]; I = o[LA + R2] / p["s2"]; wz = o[CMD + 2]
    act, act_l = gates(o)
    w = o[GYRO:GYRO + 3] / 0.2; g = o[GRAV:GRAV + 3]
    rate = -(w[1] * g[1] + w[2] * g[2]) * (1 + g[0] ** 2)
    out = [p["s1"] * cl(th + p["dt"] * (wz - rate), p["Lth"] * act), p["s2"] * cl(I + p["dt"] * th, p["LI"] * act)]
    if vhat is not None:
        out.append(p["s3"] * cl(o[LA + R3] / p["s3"] + p["dt"] * vhat, p["Lp"] * act_l))
    return tuple(out)
