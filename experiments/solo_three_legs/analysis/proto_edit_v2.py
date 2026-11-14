"""Prototype of the v2 edit (heading hold), evaluated OUTSIDE the weights: a wrapper around the v1 network.

v2 = v1 + a heading-hold integrator (ported from the H1 heading-hold circuit) whose two registers live in
the three front-left (FL) action channels when the leg is broken, read back through last_action:

  fault flag      F = clamp((g - 0.55)/0.3, 0, 1)          (v1)
  act             = clamp(6(|sin|+|cos|) - 4.5 - 120 |wz_cmd| + 10 (F - 1), 0, 1)
  rate            = -(w_y g_y + w_z g_z)(1 + g_x^2)          Euler yaw rate (gyro + projected gravity)
  theta'          = clamp(theta + dt (wz_cmd - rate), +-L_th act)
  I'              = clamp(I + dt theta,               +-L_I act)
  y'              = K_th theta' + K_I I'                     yaw-rate correction
  wz seen by the original units = wz + K (wz - gyro_z) + trim + y        (y read back from last_action)

FL channels (basis e_v, e_1, e_2 = eigenvectors of the FL action covariance on four legs, e_v = smallest):
  four legs (F=0):  a_FL = P_perp a_FL^v1 + e_v h(phase_next)       (component along e_v replaced by its
                                                                     first clock harmonic h, fitted)
  leg broken (F=1): a_FL = s1 theta' e_1 + s2 I' e_2 + (s_y y' + h(phase_next)) e_v
  read back:        theta = e_1.la/s1, I = e_2.la/s2, y = (e_v.la - h(phase))/s_y
The original units' own reading of last_action[FL] is constant-folded (they see its four-leg mean).
"""
import numpy as np

DT = 0.02
DPHI = 2 * np.pi * 2.5 * DT          # clock advance per policy step

V2 = dict(K_th=3.0, K_I=3.0, L_th=0.35, L_I=0.3, s1=1.0, s2=1.0, s_y=10.0)


def clockfeat(obs):
    return obs[:, 45], obs[:, 46]


def next_clock(s, c):
    """(sin, cos) of the phase one policy step later (exact rotation; zeros stay zeros)."""
    return s * np.cos(DPHI) + c * np.sin(DPHI), c * np.cos(DPHI) - s * np.sin(DPHI)


PRED_COLS = list(range(36, 45)) + [45, 46, 6, 7, 8]      # last action of the 9 working joints, clock, command


class ProtoEditV2:
    """consts: w (fault signal), pred (len(PRED_COLS)+1, 3) linear predictor of last_action[FL],
    basis (3,3) columns [e_v, e_1, e_2], m_v (four-leg mean of e_v . a_FL)."""

    def __init__(self, v1_net, consts, p=V2, heading=True):
        self.net, self.c, self.p, self.heading = v1_net, consts, dict(p), heading
        self.E = np.asarray(consts["basis"])

    def __call__(self, obs):
        obs = np.asarray(obs, np.float64)
        p, E = self.p, self.E
        ev, e1, e2 = E[:, 0], E[:, 1], E[:, 2]
        la = obs[:, 33:36]
        s, c = obs[:, 45], obs[:, 46]
        g = obs[:, [9, 10, 11, 21, 22, 23]] @ self.c["w"]
        F = np.clip((g - 0.55) / 0.3, 0.0, 1.0)
        wz = obs[:, 8] / 0.25
        act = np.clip(6 * (np.abs(s) + np.abs(c)) - 4.5 - 120 * np.abs(wz) + 10 * (F - 1), 0.0, 1.0)
        if not self.heading:
            act = act * 0.0
        th = la @ e1 / p["s1"]
        I = la @ e2 / p["s2"]
        y = (la @ ev - self.c["m_v"]) / p["s_y"]               # = register in faulty mode, small noise on four legs
        o = obs.copy()
        pc = self.c.get("pred_cols", PRED_COLS)
        o[:, 33:36] = np.c_[obs[:, pc], np.ones(len(obs))] @ self.c["pred"]   # re-wired FL last-action
        o[:, 8] = o[:, 8] + 0.25 * y / 1.5                         # v1 weights scale column 8 by 1.5 (yaw loop)
        if p.get("extra_trim") is not None:                         # extra constant command trim (leg broken only)
            o[:, 6:9] += np.outer(g, np.asarray(p["extra_trim"]) * np.array([2.0, 2.0, 0.25 / 1.5]))
        a, hs = self.net(o, return_hidden=True)
        self.last_h1 = hs[0]
        w = obs[:, 0:3] / 0.25
        gv = obs[:, 3:6]
        rate = -(w[:, 1] * gv[:, 1] + w[:, 2] * gv[:, 2]) * (1 + gv[:, 0] ** 2)
        th2 = np.clip(th + DT * (wz - rate), -p["L_th"] * act, p["L_th"] * act)
        lat = self.c.get("lat")
        if lat is not None and p.get("K_lat", 0.0):
            # path hold: integrate heading error minus (lateral velocity estimate / nominal speed)
            vy_hat = hs[0] @ lat["W"] + lat["b"]
            I2 = np.clip(I + DT * (th - p["K_lat"] * vy_hat / 0.2), -p["L_I"] * act, p["L_I"] * act)
        else:
            I2 = np.clip(I + DT * (th - p.get("th_b", 0.0)), -p["L_I"] * act, p["L_I"] * act)
        y2 = p["K_th"] * th2 + p["K_I"] * I2
        faulty = np.outer(p["s1"] * th2, e1) + np.outer(p["s2"] * I2, e2) + np.outer(p["s_y"] * y2 + self.c["m_v"], ev)
        a[:, 0:3] = (1 - F)[:, None] * a[:, 0:3] + F[:, None] * faulty
        return a


def fit_constants(v1_net, obs_four_legs, w):
    """constants from four-leg observations of the v1 network (declared: fitted to data)."""
    A = v1_net(obs_four_legs)[:, 0:3]
    X = np.c_[obs_four_legs[:, PRED_COLS], np.ones(len(obs_four_legs))]
    pred = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ obs_four_legs[:, 33:36])
    E = X @ pred - obs_four_legs[:, 33:36]
    ev_, evec = np.linalg.eigh(np.cov(A.T))
    basis = evec[:, [0, 1, 2]]                                   # smallest variance first
    m_v = float((A @ basis[:, 0]).mean())
    return dict(w=np.asarray(w), pred=pred, basis=basis, m_v=m_v,
                stats=dict(var_along=ev_.tolist(), pred_rms=np.sqrt((E ** 2).mean(0)).tolist(),
                           ev_component_std=float((A @ basis[:, 0]).std())))
