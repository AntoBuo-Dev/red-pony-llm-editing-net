"""Prototype of the three-legs edit, evaluated OUTSIDE the weights (policy wrapper).

The weight edit (edit/build_edit.py) implements the same map inside the MLP; check_edit compares the two.
In the broken-leg condition (front-left leg locked) the prototype is

    obs' = obs  with  q_rel[9:21] -= 0.3 * A,  last_action[33:45] -= A        (hide the posture change)
                and   cmd[6:9]    += trim * (2, 2, 0.25)                       (command trim)
    act  = net(obs') + A + D(phase)                                             (posture + FR re-timing)

A      constant action offset that moves the three working feet (inverse kinematics of a foot shift);
D      periodic offset on the front-right leg: FR replays its own stance twice per gait cycle, with two
       short swings (hops) instead of one long swing; D = N_FR(psi(phase)) - N_FR(phase), where N_FR is
       the original network's phase-averaged FR action on four legs (measured) and psi a phase warp.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_lib import rollout, phase_of, Wrapped, NumpyPolicy, load_onnx_layers  # noqa: E402
from leg_kinematics import foot_pos, ik, LEGS  # noqa: E402

Q0 = np.array([0.1, 0.8, -1.6, -0.1, 0.8, -1.6, 0.1, -0.8, 1.6, -0.1, -0.8, 1.6])
NBIN = 20                                            # 20 policy steps per gait cycle (2.5 Hz, 50 Hz)

# design parameters (chosen by the manual scans logged in README / results/edit_log.md)
DESIGN = dict(
    foot_shift={"FR": (0.06, 0.11, -0.02), "HL": (0.06, 0.06, -0.02), "HR": (0.06, 0.06, -0.02)},  # [m], base frame
    fr_swings=[(0.4, 0.1), (0.9, 0.1)],             # new FR swing windows (start, width) in gait phase
    fr_nominal_swing=(0.5, 0.45),                   # FR swing window of the original trot (from foot contacts)
    cmd_trim=(0.7, 0.2, 0.2),                       # added to (vx, vy, wz) seen by the network
    yaw_K=0.5,                                      # wz seen = wz + K (wz - gyro_z)   (all conditions)
    n_bells=40, bell_halfwidth=1.5,                 # D(phase) is synthesised from 40 "bell" units
)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def bell_params(design=DESIGN):
    K = design["n_bells"]
    alpha = 2 * np.pi * design["bell_halfwidth"] / K
    beta = 3.0 / (1.0 - np.cos(alpha))
    nodes = np.arange(K) / K
    return nodes, alpha, beta


def bells(s, c, design=DESIGN):
    """bell activations from the clock observation (s = sin, c = cos of 2 pi phase); shape (n, K)."""
    nodes, alpha, beta = bell_params(design)
    z = beta * (np.outer(s, np.sin(2 * np.pi * nodes)) + np.outer(c, np.cos(2 * np.pi * nodes)) - np.cos(alpha))
    return elu(z)


def fit_bells(N, design=DESIGN, ngrid=4000):
    """least-squares coefficients C (K+1, 3) so that bells(phase) @ C[:K] + C[K] ~ D(phase) on a fine grid."""
    ph = np.arange(ngrid) / ngrid
    D = fr_delta(N, ph, design)
    X = np.c_[bells(np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), design), np.ones(ngrid)]
    C, *_ = np.linalg.lstsq(X, D, rcond=None)
    E = X @ C - D
    return C, dict(rms=np.sqrt((E ** 2).mean(0)).tolist(), max=np.abs(E).max(0).tolist())


def posture_offset(foot_shift):
    """action offset A (12,) that moves each listed foot by the given base-frame displacement."""
    a = np.zeros(12)
    for leg, dp in foot_shift.items():
        i = LEGS.index(leg)
        qn = Q0[3 * i:3 * i + 3]
        q = ik(leg, foot_pos(leg, qn) + np.asarray(dp), qn)
        a[3 * i:3 * i + 3] = (q - qn) / 0.3
    return a


def nominal_pattern(net, cfg, cmd=(0.3, 0.0, 0.0), T=12.0, seeds=(0, 1, 2)):
    """phase-binned mean action of the original policy on four legs (NBIN x 12)."""
    acts, bins = [], []
    for s in seeds:
        r = rollout(net, cfg, cmd, T=T, seed=s, phase0=0.0)
        k = 100
        acts.append(r["act"][k:]); bins.append(np.round(r["phase"][k:] * NBIN).astype(int) % NBIN)
    acts = np.concatenate(acts); bins = np.concatenate(bins)
    return np.array([acts[bins == i].mean(0) for i in range(NBIN)])


def interp_periodic(N, ph):
    x = ph * NBIN
    i0 = np.floor(x).astype(int) % NBIN
    f = x - np.floor(x)
    return N[i0] * (1 - f)[..., None] + N[(i0 + 1) % NBIN] * f[..., None]


def warp_windows(ph, windows, nom):
    """phase warp psi: each new swing window maps onto the nominal swing window, each stance gap
    between new swings maps onto the full nominal stance."""
    s0, w0 = nom
    out = np.zeros_like(ph)
    ws = sorted(windows)
    n = len(ws)
    for i, (a, w) in enumerate(ws):
        nxt = ws[(i + 1) % n][0] + (1.0 if i == n - 1 else 0.0)
        gap = nxt - (a + w)
        u = (ph - a) % 1.0
        out = np.where(u < w, s0 + w0 * u / w, out)
        out = np.where((u >= w) & (u < w + gap), s0 + w0 + (1 - w0) * (u - w) / gap, out)
    return out % 1.0


def fr_delta(N, ph, design=DESIGN):
    """D(phase) for the three FR action channels, shape (len(ph), 3)."""
    psi = warp_windows(ph, design["fr_swings"], design["fr_nominal_swing"])
    return interp_periodic(N, psi)[:, 3:6] - interp_periodic(N, ph)[:, 3:6]


class ProtoEdit:
    """policy wrapper; `faulty` selects the broken-leg branch (the weight edit detects it itself).
    bellC: if given, D(phase) is the bell synthesis (exactly what the weights compute) instead of the
    piecewise-linear design function."""

    def __init__(self, net, N, design=DESIGN, faulty=True, bellC=None, w=None):
        """faulty: True / False selects the branch by hand; None = detect it from the observation like the
        weight edit does (needs w: g = w . obs[9,10,11,21,22,23], flag F = clamp((g-0.55)/0.3, 0, 1),
        and the constant shifts are multiplied by g, including its small value on four legs)."""
        self.net, self.N, self.d, self.faulty, self.C, self.w = net, N, design, faulty, bellC, w
        self.A = posture_offset(design["foot_shift"])
        self.trim = np.asarray(design["cmd_trim"]) * np.array([2.0, 2.0, 0.25])
        self.K = design.get("yaw_K", 0.0)
        self.tiltk = design.get("tilt_gain", 0.0)
        if design.get("getup"):
            ex = design["getup"]["extra_shift"]
            ex = ex if isinstance(ex, dict) else {l: ex for l in design["foot_shift"]}
            fs = {l: tuple(np.asarray(v) + np.asarray(ex.get(l, (0, 0, 0)))) for l, v in design["foot_shift"].items()}
            self.Bgu = posture_offset(fs) - self.A
        if self.tiltk:
            self.Jt = tilt_jacobian(design)

    def __call__(self, obs):
        obs = np.asarray(obs, np.float64)
        o = obs.copy()
        o[:, 8] += self.K * (obs[:, 8] - obs[:, 2])          # yaw-rate loop (always on)
        if self.faulty is None:
            g = obs[:, [9, 10, 11, 21, 22, 23]] @ self.w
            F = np.clip((g - 0.55) / 0.3, 0.0, 1.0)
        else:
            if not self.faulty:
                return self.net(o)
            g = F = np.ones(len(obs))
        o[:, 9:21] -= 0.3 * np.outer(g, self.A)
        o[:, 33:45] -= np.outer(g, self.A)
        o[:, 6:9] += np.outer(g, self.trim)
        a = self.net(o) + np.outer(F, self.A)
        if self.tiltk:
            a += F[:, None] * (obs[:, 3:5] @ self.Jt.T) * self.tiltk
        gu = self.d.get("getup")
        if gu:   # get-up reflex: extra posture offset when the body leans onto the broken corner
            gw = np.asarray(gu.get("gw", (1.0, 1.0)))         # detector on gw . (g_x, g_y)
            Cr = np.clip((obs[:, 3:5] @ gw - gu["theta"]) / gu["delta"], 0.0, 1.0)
            H = np.clip(F + Cr - 1.0, 0.0, 1.0)
            a += H[:, None] * self.Bgu[None]
        s, c = obs[:, 45], obs[:, 46]
        G = np.clip(6 * (np.abs(s) + np.abs(c)) - 5 + 10 * (F - 1), 0.0, 1.0)
        if self.C is None:
            on = G > 0
            if on.any():
                a[on, 3:6] += G[on, None] * fr_delta(self.N, phase_of(obs)[on], self.d)
        else:
            a[:, 3:6] += G[:, None] * (bells(s, c, self.d) @ self.C[:-1] + self.C[-1])
        return a


def tilt_jacobian(design=DESIGN, eps=0.01):
    """(12, 2) action change per unit of (g_x, g_y) projected gravity: all three working feet move by
    (g_x, g_y) * 1 m in the base frame (forward / left), i.e. the body moves back / right."""
    base = posture_offset(design["foot_shift"])
    J = np.zeros((12, 2))
    for k in range(2):
        fs = {l: tuple(np.asarray(v) + eps * np.eye(3)[k]) for l, v in design["foot_shift"].items()}
        J[:, k] = (posture_offset(fs) - base) / eps
    return J
