"""Write the three-legs edit into the weights of the SOLO-12 walking MLP (47 -> 512 -> 256 -> 128 -> 12, ELU).

Observation columns used (see policies/original/config.yaml):
    2  gyro z * 0.25        6,7,8  command (vx, vy, wz) * (2, 2, 0.25)
    9:21  joint pos - default (FL = 9,10,11)      21:33  joint vel * 0.05 (FL = 21,22,23)
    33:45 last action       45,46  clock sin, cos (both 0 when standing)

Edit (all parameters in DESIGN / weight_edit.json):

  E1  yaw-rate loop, always on       wz seen = wz + K (wz - gyro_z)                (first-layer columns 2, 8)
  E2  broken-leg constant shifts      the original first layer additionally receives  c * g(x)
      g(x) = w . [q_FL, dq_FL]        (= 1 when the front-left leg is locked tucked, rms 0.09 on four legs)
      c    = W0 [ -0.3 A (joint pos) ; -A (last action) ; trim (command) ]          (rank-1 update of the
      FL position / velocity columns: hides the posture offset A from the network and trims its command)
  E3  fault flag    F = clamp((g - 0.55) / 0.3, 0, 1)                               (2 layer-1 units)
  E4  clock on      |sin| + |cos|                                                    (4 layer-1 units)
      gate          G = clamp(6 (|sin|+|cos|) - 5 + 10 (F - 1), 0, 1)                (2 layer-2 units)
  E5  FR re-timing  D(phase) = sum_i C_i bell_i(sin, cos) + C_0                      (40 layer-1 bells,
                    3 layer-2 pass-through units, 3+1 layer-3 gated units)
  E6  outputs       a_FR += G * D(phase);   a_{FR,HL,HR} += F * A                    (1+1 pass-through units)
"""
import numpy as np

S_FLAG, S_GATE, S_ABS = 50.0, 50.0, 2000.0
B_L, B_G, M_G = 6.0, 6.0, 40.0
FL_COLS = [9, 10, 11, 21, 22, 23]


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def apply_input_edits(L, A, trim_scaled, K, w):
    """E1 + E2 on the first layer of the ORIGINAL units (before freeing). Returns new layers and c."""
    (W0, b0) = L[0]
    W0 = W0.astype(np.float64).copy(); b0 = b0.astype(np.float64).copy()
    W0orig = W0.copy()
    # E1: wz seen = wz + K (wz - gyro_z), both columns scaled by 0.25
    W0[:, 8] = (1.0 + K) * W0orig[:, 8]
    W0[:, 2] = W0orig[:, 2] - K * W0orig[:, 8]
    # E2: constant shift c, gated by the linear fault signal g(x) = w . x[FL_COLS]
    c = -W0orig[:, 9:21] @ (0.3 * A) - W0orig[:, 33:45] @ A + W0orig[:, 6:9] @ trim_scaled
    W0[:, FL_COLS] += np.outer(c, w)
    return [(W0, b0)] + [(W.astype(np.float64).copy(), b.astype(np.float64).copy()) for W, b in L[1:]], c


def write_circuit(L, freed, A, w, bell_dirs, bell_bias, C):
    """L: layers after apply_input_edits + free_units (freed rows/cols are zero).
    bell_dirs (K,2): weights on (sin, cos) of each bell (already multiplied by beta); bell_bias (K,);
    C (K+1, 3): bell coefficients for the three FR action channels (last row = constant)."""
    (W0, b0), (W1, b1), (W2, b2), (W3, b3) = [(W.copy(), b.copy()) for W, b in L]
    fr1, fr2, fr3 = [list(f) for f in freed]
    it1, it2, it3 = iter(fr1), iter(fr2), iter(fr3)
    used = {}

    def u1(cols_w, bias):
        u = next(it1); assert not W0[u].any() and not W1[:, u].any()
        for col, val in cols_w:
            W0[u, col] = val
        b0[u] = bias
        return u

    # ---- layer 1 -------------------------------------------------------------------------------
    # E3 fault flag: zF = (g - 0.55)/0.3 ; F = (elu(S zF) - elu(S (zF - 1))) / S
    zF = [(col, wk / 0.3) for col, wk in zip(FL_COLS, w)]
    fa = u1([(col, S_FLAG * v) for col, v in zF], -S_FLAG * 0.55 / 0.3)
    fb = u1([(col, S_FLAG * v) for col, v in zF], -S_FLAG * 0.55 / 0.3 - S_FLAG)
    # E4 |sin|, |cos| ~ (elu(S x) + elu(-S x)) / S
    abs_units = [u1([(col, sg * S_ABS)], 0.0) for col in (45, 46) for sg in (1.0, -1.0)]
    # E5 bells
    bell_units = [u1([(45, d[0]), (46, d[1])], bb) for d, bb in zip(bell_dirs, bell_bias)]
    used["layer1"] = dict(flag=[fa, fb], abs=abs_units, bells=bell_units)

    def u2(terms, bias):
        """terms: dict layer1-unit -> coef"""
        u = next(it2); assert not W1[u].any() and not W2[:, u].any()
        for v, cf in terms.items():
            W1[u, v] += cf
        b1[u] = bias
        return u

    # ---- layer 2 -------------------------------------------------------------------------------
    F_terms = {fa: 1.0 / S_FLAG, fb: -1.0 / S_FLAG}
    absum = {a: 1.0 / S_ABS for a in abs_units}
    zG = {}
    for d_ in (absum,):
        for k, v in d_.items():
            zG[k] = zG.get(k, 0.0) + 6.0 * v
    for k, v in F_terms.items():
        zG[k] = zG.get(k, 0.0) + 10.0 * v
    zG_const = -5.0 - 10.0
    ga = u2({k: S_GATE * v for k, v in zG.items()}, S_GATE * zG_const)
    gb = u2({k: S_GATE * v for k, v in zG.items()}, S_GATE * (zG_const - 1.0))
    Lu = [u2({b: C[i, j] for i, b in enumerate(bell_units)}, C[-1, j] + B_L) for j in range(3)]
    fp = u2(F_terms, 0.5)
    used["layer2"] = dict(gate=[ga, gb], fr_signal=Lu, flag_pass=[fp])

    def u3(terms, bias):
        u = next(it3); assert not W2[u].any() and not W3[:, u].any()
        for v, cf in terms.items():
            W2[u, v] += cf
        b2[u] = bias
        return u

    # ---- layer 3 -------------------------------------------------------------------------------
    MG = {ga: M_G / S_GATE, gb: -M_G / S_GATE}                    # M * G
    v_units = []
    for j in range(3):
        t = dict(MG); t[Lu[j]] = t.get(Lu[j], 0.0) + 1.0
        v_units.append(u3(t, -B_L + B_G - M_G))                   # elu(L_j + B_G - M + M G)
    r = u3(dict(MG), B_G - M_G)                                   # elu(B_G - M + M G)
    f3 = u3({fp: 1.0}, 0.0)                                       # = F + 0.5
    used["layer3"] = dict(fr_gated=v_units, ref=[r], flag_pass=[f3])

    # ---- output --------------------------------------------------------------------------------
    for j in range(3):
        W3[3 + j, v_units[j]] += 1.0
        W3[3 + j, r] += -1.0
    for k in range(3, 12):
        W3[k, f3] += A[k]
        b3[k] += -0.5 * A[k]
    n_used = (len(fr1) - len(list(it1)), len(fr2) - len(list(it2)), len(fr3) - len(list(it3)))
    return [(W0, b0), (W1, b1), (W2, b2), (W3, b3)], used, n_used


N_UNITS = (46, 6, 5)
