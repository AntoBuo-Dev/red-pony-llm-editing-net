"""Write a heading-hold circuit into freed hidden units of the H1_2 squat policy (no training).

Circuit (all quantities from the newest observation sample, dt = 0.02 s):
    rate   = -(w_y g_y + w_z g_z) (1 + g_x^2)          Euler yaw rate from gyro + projected gravity
    act    = 1 if the gait clock runs and |wz_cmd| ~ 0, else 0
    theta' = clamp(theta + dt (wz_cmd - rate), +-L_theta act)     heading error      -> action 25
    I'     = clamp(I + dt theta,              +-L_I act)          its integral       -> action 26
    wz seen by the original network = wz_cmd + K_theta theta + K_I I   (fold in the first layer)
The two registers live in the left/right wrist-yaw action channels, read back through last_action.
"""
import numpy as np

GYRO, GRAV, CMD, LA, GAIT = 12, 27, 42, 423, 458      # newest-sample column offsets
WZ_COLS = [32, 35, 38, 41, 44]
la_cols = lambda j: [315 + 27 * k + j for k in range(5)]
R1, R2 = 25, 26                                        # left / right wrist yaw

P = dict(Kth=3.0, KI=3.0, s1=2.0, s2=2.0, Lth=0.35, LI=0.3, dt=0.02, m25=0.027, m26=0.067,
         S=200.0,      # sharpness of the clamps
         Sg=2000.0,    # sharpness of the |.| units of the gate
         B=1.0,       # bias of the linear pass-through units
         alpha=2.0, beta=0.4, beta2=0.1, sn=10.0, sx=5.0, gq=0.15)


def build(L_pruned, freed, p=P):
    """L_pruned: 4 (W,b) float arrays with freed units zeroed; freed: [idx_l1, idx_l2, idx_l3]."""
    (W0, b0), (W1, b1), (W2, b2), (W3, b3) = [(W.astype(np.float64).copy(), b.astype(np.float64).copy()) for W, b in L_pruned]
    f1, f2, f3 = [list(f) for f in freed]
    a, be, be2, sn, sx, gq, S, Sg, B = p["alpha"], p["beta"], p["beta2"], p["sn"], p["sx"], p["gq"], p["S"], p["Sg"], p["B"]
    s1, s2, dt = p["s1"], p["s2"], p["dt"]

    # ---- first layer: constant-fold the original use of last_action[25/26], then add the wz fold ----
    for j, m in ((R1, p["m25"]), (R2, p["m26"])):
        cols = la_cols(j)
        b0 += W0[:, cols].sum(1) * m
        W0[:, cols] = 0
    for k in range(5):
        W0[:, la_cols(R1)[k]] += p["Kth"] / s1 * W0[:, WZ_COLS[k]]
        W0[:, la_cols(R2)[k]] += p["KI"] / s2 * W0[:, WZ_COLS[k]]

    # ---- layer-1 circuit units ------------------------------------------------------------------
    it = iter(f1)
    def unit1(cols_w, bias):
        u = next(it); assert not W0[u].any() and not W1[:, u].any()
        for c, w in cols_w: W0[u, c] = w
        b0[u] = bias
        return u
    # linear expression over layer-1 units: dict unit->coef plus constant
    class Lin(dict):
        const = 0.0
        def add(self, other, k=1.0):
            for u, c in other.items(): self[u] = self.get(u, 0.0) + k * c
            self.const += k * other.const
            return self
    def lin(pairs, const=0.0):
        e = Lin(pairs); e.const = const; return e

    u = unit1([(LA + R1, 1.0)], B); r1 = lin({u: 1.0}, -B)                    # register theta (scaled s1)
    u = unit1([(LA + R2, 1.0)], B); r2 = lin({u: 1.0}, -B)                    # register I (scaled s2)
    # |sin|+|cos| and |wz|: relu(x) ~ elu(Sg x)/Sg
    absum = lin({})
    for c in (GAIT, GAIT + 1):
        for sgn in (1, -1): absum.add(lin({unit1([(c, sgn * Sg)], 0.0): 1.0 / Sg}))
    abswz = lin({})
    for sgn in (1, -1): abswz.add(lin({unit1([(CMD + 2, sgn * Sg)], 0.0): 1.0 / Sg}))
    # e_lin = wz - w_z   (unit holds 0.5*(wz - w_z) + 4)
    u = unit1([(CMD + 2, 0.5), (GYRO + 2, -0.5 / 0.2)], 4.0); e_lin = lin({u: 2.0}, -8.0)
    # w_z alone is needed for r0: w_z = wz - e_lin, wz enters linearly only via e_lin -> use a unit
    u = unit1([(GYRO + 2, 0.5 / 0.2)], 4.0); wz_gyro = lin({u: 2.0}, -8.0)   # w_z [rad/s]
    def square_units(cols_w, const):
        """q(x) = elu(x)+elu(-x) ~ x^2/2 for the affine input x = sum w*obs + const"""
        e = lin({})
        for sgn in (1, -1):
            e.add(lin({unit1([(c, sgn * w) for c, w in cols_w], sgn * const): 1.0}))
        return e
    def product(colsA, cA, colsB, cB, scale):
        """A*B with A, B affine in obs (already normalised); returns Lin = scale * A * B"""
        plus = square_units([(c, w / sn) for c, w in colsA + colsB], (cA + cB) / sn)
        minus = square_units([(c, w / sn) for c, w in colsA] + [(c, -w / sn) for c, w in colsB], (cA - cB) / sn)
        return lin({}).add(plus, scale * sn * sn / 2).add(minus, -scale * sn * sn / 2)
    wy_gy = product([(GYRO + 1, 1 / 0.2 / a)], 0.0, [(GRAV + 1, 1 / be)], 0.0, a * be)             # w_y g_y
    wz_1gz = product([(GYRO + 2, 1 / 0.2 / a)], 0.0, [(GRAV + 2, 1 / be2)], 1 / be2, a * be2)       # w_z (1+g_z)
    gx2 = lin({}).add(square_units([(GRAV, 1 / sx)], 0.0), 2 * sx * sx)                             # g_x^2
    # r0 = -(w_y g_y + w_z g_z) = w_z - w_z(1+g_z) - w_y g_y ;  E0 = wz - r0
    r0 = lin({}).add(wz_gyro).add(wz_1gz, -1.0).add(wy_gy, -1.0)
    E0 = lin({}).add(e_lin).add(wz_1gz, 1.0).add(wy_gy, 1.0)
    n1 = len(f1) - len(list(it))

    # ---- layer-2 circuit units: pre-activation = Lin over layer-1 units --------------------------
    it2 = iter(f2)
    def unit2(e, k=1.0, bias=0.0):
        u = next(it2); assert not W1[u].any() and not W2[:, u].any()
        for v, c in e.items(): W1[u, v] = k * c
        b1[u] = k * e.const + bias
        return u
    u = unit2(r1, 1.0, B); r1_2 = lin({u: 1.0}, -B)
    u = unit2(r2, 1.0, B); r2_2 = lin({u: 1.0}, -B)
    z = lin({}, -4.5).add(absum, 6.0).add(abswz, -120.0)       # >= 1 when active, <= 0 otherwise
    ua = unit2(z, S); ub = unit2(z, S, -S)                      # act = (elu(S z) - elu(S (z-1)))/S
    act = lin({ua: 1.0 / S, ub: -1.0 / S})
    u = unit2(E0, 0.5, 4.0); E0_2 = lin({u: 2.0}, -8.0)
    # r0 * g_x^2 : a = r0/alpha, b = gx2/gq
    def sq2(e):
        out = lin({})
        for sgn in (1, -1): out.add(lin({unit2(e, sgn): 1.0}))
        return out
    A_ = lin({}).add(r0, 1 / a / sn); Bq = lin({}).add(gx2, 1 / gq / sn)
    plus = sq2(lin({}).add(A_).add(Bq)); minus = sq2(lin({}).add(A_).add(Bq, -1.0))
    r0q = lin({}).add(plus, a * gq * sn * sn / 2).add(minus, -a * gq * sn * sn / 2)
    e_2 = lin({}).add(E0_2).add(r0q, -1.0)                      # wz - rate
    n2 = len(f2) - len(list(it2))

    # ---- layer-3 circuit units and output rows ---------------------------------------------------
    it3 = iter(f3)
    def unit3(e, k=1.0, bias=0.0):
        u = next(it3); assert not W2[u].any() and not W3[:, u].any()
        for v, c in e.items(): W2[u, v] = k * c
        b2[u] = k * e.const + bias
        return u
    ua3 = unit3(act, 1.0, 0.5)                                  # act pass-through (act + 0.5 > 0)
    W3[R1] = 0; b3[R1] = 0; W3[R2] = 0; b3[R2] = 0
    for row, x, Lreg in ((R1, lin({}).add(r1_2).add(e_2, s1 * dt), s1 * p["Lth"]),
                         (R2, lin({}).add(r2_2).add(r1_2, s2 * dt / s1), s2 * p["LI"])):
        lo = unit3(lin({}).add(x).add(act, Lreg), S)            # elu(S (x + L act))
        hi = unit3(lin({}).add(x).add(act, -Lreg), S)           # elu(S (x - L act))
        W3[row, lo] = 1 / S; W3[row, hi] = -1 / S
        W3[row, ua3] = -(Lreg + 1 / S); b3[row] = 0.5 * (Lreg + 1 / S)
    n3 = len(f3) - len(list(it3))
    L = [(W.astype(np.float32), b.astype(np.float32)) for W, b in ((W0, b0), (W1, b1), (W2, b2), (W3, b3))]
    return L, (n1, n2, n3)

N_UNITS = (20, 9, 5)   # circuit units used in hidden layers 1..3


def reference(o, p=P):
    """Exact (non-approximated) register update for one observation; returns (a25, a26)."""
    cl = lambda x, l: min(max(x, -l), l)
    th = o[LA + R1] / p["s1"]; I = o[LA + R2] / p["s2"]
    gait = min(1.0, abs(o[GAIT]) + abs(o[GAIT + 1])); wz = o[CMD + 2]
    act = min(1.0, max(0.0, 6 * (abs(o[GAIT]) + abs(o[GAIT + 1])) - 4.5 - 120 * abs(wz)))
    w = o[GYRO:GYRO + 3] / 0.2; g = o[GRAV:GRAV + 3]
    rate = -(w[1] * g[1] + w[2] * g[2]) * (1 + g[0] ** 2)
    th2 = cl(th + p["dt"] * (wz - rate), p["Lth"] * act)
    I2 = cl(I + p["dt"] * th, p["LI"] * act)
    return p["s1"] * th2, p["s2"] * I2
