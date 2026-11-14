"""v2: heading hold for the three-leg edit (port of the H1 heading-hold circuit), written into freed units.

With the front-left leg broken (fault flag F = 1) the three FL action channels are dead outputs. They become
registers, read back through last_action (obs 33..35):

    rate   = -(w_y g_y + w_z g_z)(1 + g_x^2)                          Euler yaw rate from gyro + gravity
    act    = clamp(6(|sin|+|cos|) - 4.5 - 120|wz_cmd| + 10(F - 1), 0, 1)
    theta' = clamp(theta + dt (wz_cmd - rate), +-L_th act)            register along e_1 (scale s1)
    I'     = clamp(I + dt theta,               +-L_I  act)            register along e_2 (scale s2)
    y'     = K_th theta' + K_I I'                                      register along e_v (scale s_y, offset m_v F)
    wz seen by the original units = ... + (e_v . last_action[FL] - m_v) / s_y

Two problems of using the FL channels on four legs are solved as follows:
  (a) the original units' own reading of last_action[FL] is re-wired to a linear predictor of it from the
      other 44 inputs (fitted on four-leg and three-leg rollouts of v1, declared); the FL last-action
      columns of the original units then carry only the yaw injection;
  (b) on four legs the FL outputs must stay the FL commands, with the leg broken they must be exactly the
      registers: the FL output rows are replaced by a ridge readout of the FL command from layer 2
      (fitted, declared), gated by (1 - F) with sharp ELU differences, plus the register clamps gated by
      act (which contains F).
On four legs the injection sees (e_v . a_FL - m_v)/s_y: e_v is the direction of least variance of the FL
command and s_y = 10, so this is a small zero-mean yaw-command noise (std 0.02 rad/s).
"""
import numpy as np

P2 = dict(K_th=3.0, K_I=1.5, L_th=0.2, L_I=0.34, s1=1.0, s2=1.0, s_y=4.0, dt=0.02,
          S=200.0, Sg=2000.0, B=1.0, alpha=2.0, beta=0.4, beta2=0.1, sn=10.0, sx=5.0, gq=0.15,
          B_R=6.0, M_R=40.0,
          th_b=-0.1)      # heading offset held by the integrator [rad] (crab-angle compensation, see README)
GYRO, GRAV, CMD, LA_FL = 0, 3, 6, [33, 34, 35]
GS = 0.25                                        # gyro / wz observation scale
N_UNITS = (16, 9, 9)


class Lin(dict):
    """linear expression over hidden units of one layer (unit -> coefficient) plus a constant."""
    const = 0.0

    def add(self, other, k=1.0):
        for u, c in other.items():
            self[u] = self.get(u, 0.0) + k * c
        self.const += k * other.const
        return self


def lin(pairs=None, const=0.0):
    e = Lin(pairs or {})
    e.const = const
    return e


def rewire_fl_last_action(L, pred_cols, M, e_v, m_v, s_y):
    """(a) + injection, on the original units (call after the v1 input edits, before freeing).
    M: (len(pred_cols)+1, 3) predictor of last_action[FL]; W0 columns of the ORIGINAL net are used."""
    (W0, b0) = L[0]
    W0 = W0.copy(); b0 = b0.copy()
    Wla = W0[:, LA_FL].copy()                    # v1 did not modify these columns: original weights
    W0[:, pred_cols] += Wla @ M[:-1].T
    b0 += Wla @ M[-1]
    W0[:, LA_FL] = 0.0
    return [(W0, b0)] + list(L[1:]), Wla


def add_injection(L, W0_orig_wz, e_v, m_v, s_y):
    """original units: wz seen += (e_v . la_FL - m_v)/s_y  (obs scale 0.25 on wz)."""
    (W0, b0) = L[0]
    W0 = W0.copy(); b0 = b0.copy()
    k = GS / s_y
    W0[:, LA_FL] += k * np.outer(W0_orig_wz, e_v)
    b0 += -k * m_v * W0_orig_wz
    return [(W0, b0)] + list(L[1:])


def write_heading(L, freed, v1_used, basis, m_v, readout, orig_l2, p=P2):
    """L: layers with v1 circuit already written; freed: remaining freed units (3 lists); v1_used: units of
    the v1 circuit (flag, abs, flag pass-throughs); basis (3,3) columns e_v, e_1, e_2; readout: (W (k,3), b (3,))
    over layer-2 units orig_l2 (the FL command readout)."""
    (W0, b0), (W1, b1), (W2, b2), (W3, b3) = [(W.copy(), b.copy()) for W, b in L]
    fr1, fr2, fr3 = [list(f) for f in freed]
    it1, it2, it3 = iter(fr1), iter(fr2), iter(fr3)
    a, be, be2, sn, sx, gq, S, Sg, B = (p[k] for k in ("alpha", "beta", "beta2", "sn", "sx", "gq", "S", "Sg", "B"))
    s1, s2, sy, dt = p["s1"], p["s2"], p["s_y"], p["dt"]
    e_v, e_1, e_2 = basis[:, 0], basis[:, 1], basis[:, 2]
    used = dict(layer1={}, layer2={}, layer3={})

    def unit1(cols_w, bias):
        u = next(it1); assert not W0[u].any() and not W1[:, u].any()
        for c, wv in cols_w:
            W0[u, c] += wv
        b0[u] = bias
        return u

    # ---- layer 1 ------------------------------------------------------------------------------
    u = unit1([(LA_FL[k], e_1[k] / s1) for k in range(3)], B); r1 = lin({u: 1.0}, -B)     # theta
    used["layer1"]["theta_read"] = u
    u = unit1([(LA_FL[k], e_2[k] / s2) for k in range(3)], B); r2 = lin({u: 1.0}, -B)     # I
    used["layer1"]["I_read"] = u
    abswz = lin()                                                                           # |wz| [rad/s]
    for sgn in (1.0, -1.0):
        u = unit1([(CMD + 2, sgn * Sg)], 0.0); abswz.add(lin({u: 1.0 / (Sg * GS)}))
        used["layer1"].setdefault("abs_wz", []).append(u)
    # e_lin = wz - w_z  (unit = 0.5 (wz - w_z) + 4)
    u = unit1([(CMD + 2, 0.5 / GS), (GYRO + 2, -0.5 / GS)], 4.0); e_lin = lin({u: 2.0}, -8.0)
    used["layer1"]["e_lin"] = u
    u = unit1([(GYRO + 2, 0.5 / GS)], 4.0); wz_gyro = lin({u: 2.0}, -8.0)
    used["layer1"]["w_z"] = u

    def square_units(cols_w, const, tag):
        e = lin()
        for sgn in (1.0, -1.0):
            uu = unit1([(c, sgn * wv) for c, wv in cols_w], sgn * const)
            used["layer1"].setdefault(tag, []).append(uu)
            e.add(lin({uu: 1.0}))
        return e

    def product(colsA, cA, colsB, cB, scale, tag):
        plus = square_units([(c, wv / sn) for c, wv in colsA + colsB], (cA + cB) / sn, tag)
        minus = square_units([(c, wv / sn) for c, wv in colsA] + [(c, -wv / sn) for c, wv in colsB], (cA - cB) / sn, tag)
        return lin().add(plus, scale * sn * sn / 2).add(minus, -scale * sn * sn / 2)

    wy_gy = product([(GYRO + 1, 1 / GS / a)], 0.0, [(GRAV + 1, 1 / be)], 0.0, a * be, "wy_gy")
    wz_1gz = product([(GYRO + 2, 1 / GS / a)], 0.0, [(GRAV + 2, 1 / be2)], 1 / be2, a * be2, "wz_1gz")
    gx2 = lin().add(square_units([(GRAV, 1 / sx)], 0.0, "gx2"), 2 * sx * sx)
    r0 = lin().add(wz_gyro).add(wz_1gz, -1.0).add(wy_gy, -1.0)          # -(w_y g_y + w_z g_z)
    E0 = lin().add(e_lin).add(wz_1gz, 1.0).add(wy_gy, 1.0)              # wz - r0

    # v1 units reused: fault flag F and |sin| + |cos|
    from three_leg_circuit import S_FLAG, S_ABS
    fa, fb = v1_used["layer1"]["flag"]
    F1 = lin({fa: 1.0 / S_FLAG, fb: -1.0 / S_FLAG})
    absum = lin({u_: 1.0 / S_ABS for u_ in v1_used["layer1"]["abs"]})

    # ---- layer 2 ------------------------------------------------------------------------------
    def unit2(e, k=1.0, bias=0.0, tag=None):
        uu = next(it2); assert not W1[uu].any() and not W2[:, uu].any()
        for v, c in e.items():
            W1[uu, v] += k * c
        b1[uu] = k * e.const + bias
        if tag:
            used["layer2"].setdefault(tag, []).append(uu)
        return uu

    u = unit2(r1, 1.0, B, "theta_pass"); r1_2 = lin({u: 1.0}, -B)
    u = unit2(r2, 1.0, B, "I_pass"); r2_2 = lin({u: 1.0}, -B)
    z = lin({}, -4.5).add(absum, 6.0).add(abswz, -120.0).add(F1, 10.0).add(lin({}, -10.0))
    ua = unit2(z, S, 0.0, "act"); ub = unit2(z, S, -S, "act")
    act = lin({ua: 1.0 / S, ub: -1.0 / S})
    u = unit2(E0, 0.5, 4.0, "E0_pass"); E0_2 = lin({u: 2.0}, -8.0)

    def sq2(e):
        out = lin()
        for sgn in (1.0, -1.0):
            out.add(lin({unit2(e, sgn, 0.0, "r0_gx2"): 1.0}))
        return out
    A_ = lin().add(r0, 1 / a / sn); Bq = lin().add(gx2, 1 / gq / sn)
    plus = sq2(lin().add(A_).add(Bq)); minus = sq2(lin().add(A_).add(Bq, -1.0))
    r0q = lin().add(plus, a * gq * sn * sn / 2).add(minus, -a * gq * sn * sn / 2)
    werr = lin().add(E0_2).add(r0q, -1.0)                                # wz - rate

    # ---- layer 3 ------------------------------------------------------------------------------
    def unit3(e, k=1.0, bias=0.0, tag=None):
        uu = next(it3); assert not W2[uu].any() and not W3[:, uu].any()
        for v, c in e.items():
            W2[uu, v] += k * c
        b2[uu] = k * e.const + bias
        if tag:
            used["layer3"].setdefault(tag, []).append(uu)
        return uu

    ua3 = unit3(act, 1.0, 0.5, "act_pass")
    regs = {}
    for name, x, Lr in (("theta", lin().add(r1_2).add(werr, dt), p["L_th"]),
                        ("I", lin().add(r2_2).add(r1_2, dt).add(lin({}, -dt * p["th_b"])), p["L_I"])):
        lo = unit3(lin().add(x).add(act, Lr), S, 0.0, name + "_clamp")
        hi = unit3(lin().add(x).add(act, -Lr), S, 0.0, name + "_clamp")
        # value = (lo - hi)/S - (Lr + 1/S)(ua3 - 0.5)
        regs[name] = ({lo: 1 / S, hi: -1 / S, ua3: -(Lr + 1 / S)}, 0.5 * (Lr + 1 / S))

    # (b) FL command readout from layer 2, gated by (1 - F)
    fp = v1_used["layer2"]["flag_pass"][0]                                  # = F + 0.5
    MR, BR = p["M_R"], p["B_R"]
    Wr, br = readout
    ro_units = []
    for j in range(3):
        uu = unit3(lin({}, 0.0), 1.0, 0.0, "fl_readout")
        for k_, v in enumerate(orig_l2):
            W2[uu, v] += Wr[k_, j]
        W2[uu, fp] += -MR
        b2[uu] = br[j] + BR + 0.5 * MR
        ro_units.append(uu)
    ref = unit3(lin({fp: -MR}), 1.0, BR + 0.5 * MR, "fl_readout_ref")

    # ---- output rows of the FL channels ---------------------------------------------------------
    f3 = v1_used["layer3"]["flag_pass"][0]                                  # = F + 0.5
    W3[0:3, :] = 0.0
    b3[0:3] = 0.0
    for k_ in range(3):
        W3[k_, ro_units[k_]] += 1.0
        W3[k_, ref] += -1.0
        coef = {"theta": s1 * e_1[k_] + sy * p["K_th"] * e_v[k_], "I": s2 * e_2[k_] + sy * p["K_I"] * e_v[k_]}
        for name in ("theta", "I"):
            terms, const = regs[name]
            for uu, c in terms.items():
                W3[k_, uu] += coef[name] * c
            b3[k_] += coef[name] * const
        W3[k_, f3] += m_v * e_v[k_]
        b3[k_] += -0.5 * m_v * e_v[k_]
    n_used = (len(fr1) - len(list(it1)), len(fr2) - len(list(it2)), len(fr3) - len(list(it3)))
    return [(W0, b0), (W1, b1), (W2, b2), (W3, b3)], used, n_used


def reference_registers(obs, F, p=P2, basis=None, m_v=0.0):
    """exact register update (theta', I', y') for checks."""
    e_v, e_1, e_2 = basis[:, 0], basis[:, 1], basis[:, 2]
    la = obs[:, 33:36]
    th = la @ e_1 / p["s1"]; I = la @ e_2 / p["s2"]
    s, c = obs[:, 45], obs[:, 46]
    wz = obs[:, 8] / GS
    act = np.clip(6 * (np.abs(s) + np.abs(c)) - 4.5 - 120 * np.abs(wz) + 10 * (F - 1), 0, 1)
    w = obs[:, 0:3] / GS; g = obs[:, 3:6]
    rate = -(w[:, 1] * g[:, 1] + w[:, 2] * g[:, 2]) * (1 + g[:, 0] ** 2)
    th2 = np.clip(th + p["dt"] * (wz - rate), -p["L_th"] * act, p["L_th"] * act)
    I2 = np.clip(I + p["dt"] * (th - p["th_b"]), -p["L_I"] * act, p["L_I"] * act)
    return th2, I2, p["K_th"] * th2 + p["K_I"] * I2, act
