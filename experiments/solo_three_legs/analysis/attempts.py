#!/usr/bin/env python3
"""Re-run the main attempts of the edit (prototypes evaluated outside the weights, see README) on one small
fixed protocol and write results/attempts.json.

    python analysis/attempts.py

Protocol: front-left leg locked from the start, commands stand / forward 0.3 m/s / turn left (0.3, 0, 0.5),
3 seeds, 12 s. Each attempt is a policy wrapper around the ORIGINAL network (or, for "PD only", no network).
"""
import copy
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
from proto_edit import (DESIGN, ProtoEdit, posture_offset, interp_periodic, warp_windows, phase_of,  # noqa: E402
                        nominal_pattern)
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402
from sim_harness import run_episode  # noqa: E402

P = os.path.join(ROOT, "policies", "original")
CFG = os.path.join(P, "config.yaml")
CMDS = {"stand": (0.0, 0.0, 0.0), "fwd_0.3": (0.3, 0.0, 0.0), "turn_left": (0.3, 0.0, 0.5)}
NOM_SW = {1: (0.5, 0.45), 2: (0.5, 0.45), 3: (0.0, 0.45)}   # nominal swing windows FR, HL, HR (foot contacts)
_net = _N = None


class Variant:
    """generic prototype: obs shifts (hidden posture), command trim, output posture offset, per-leg phase
    warps of the original's nominal pattern, optional yaw loop, optional input lies."""

    def __init__(self, net, N, foot_shift=None, hide=False, warps=None, trim=(0, 0, 0), yaw_K=0.0, q_lie=False,
                 grav_lie=None, no_net=False):
        self.net, self.N = net, N
        self.A = posture_offset(foot_shift) if foot_shift else np.zeros(12)
        self.hide, self.warps, self.K, self.q_lie, self.no_net = hide, warps or {}, yaw_K, q_lie, no_net
        self.trim = np.asarray(trim) * np.array([2.0, 2.0, 0.25])
        self.grav = np.zeros(3) if grav_lie is None else np.asarray(grav_lie)

    def __call__(self, obs):
        obs = np.asarray(obs, np.float64)
        if self.no_net:
            return np.tile(self.A, (len(obs), 1))
        o = obs.copy()
        o[:, 8] += self.K * (obs[:, 8] - obs[:, 2])
        o[:, 6:9] += self.trim
        o[:, 3:6] += self.grav
        if self.q_lie:                      # posture through the network itself: lie about joint positions
            o[:, 9:21] -= 0.3 * self.A
            return self.net(o)
        if self.hide:
            o[:, 9:21] -= 0.3 * self.A
            o[:, 33:45] -= self.A
        a = self.net(o) + self.A
        ph = phase_of(obs)
        on = ~np.isnan(ph)
        if on.any():
            base = interp_periodic(self.N, ph[on])
            for leg, wins in self.warps.items():
                sl = slice(3 * leg, 3 * leg + 3)
                a[on, sl] += interp_periodic(self.N, warp_windows(ph[on], wins, NOM_SW[leg]))[:, sl] - base[:, sl]
        return a


S6 = {"FR": (0.06, 0.11, -0.02), "HL": (0.06, 0.06, -0.02), "HR": (0.06, 0.06, -0.02)}
FRDBL = {1: [(0.4, 0.1), (0.9, 0.1)]}
ATTEMPTS = [
    ("original network", "the trained trot, unchanged", dict()),
    ("FR never swings", "front-right replays its mean stance pattern in its swing half", dict(warps={1: [(0.0, 0.0001)]})),
    ("FR short swing", "front-right swing compressed into 0.15 of the cycle", dict(warps={1: [(0.45, 0.15)]})),
    ("3-beat gait", "front-right and hind-left swing one after the other in the second half", dict(warps={1: [(0.5, 0.2)], 2: [(0.72, 0.23)]})),
    ("posture only, network sees it", "feet shifted forward-left (body back-right), offset added to the outputs", dict(foot_shift=S6)),
    ("posture only, PD without network", "same foot shift, fixed joint targets, no network", dict(foot_shift=S6, no_net=True)),
    ("posture through an input lie", "network told its legs are displaced the opposite way", dict(foot_shift=S6, q_lie=True)),
    ("gravity lie", "constant +0.1 on projected-gravity y", dict(grav_lie=(0.0, 0.1, 0.0))),
    ("posture hidden from the network", "offset added to the outputs and subtracted from joint-pos / last-action inputs", dict(foot_shift=S6, hide=True)),
    ("FR-medial-back posture (LIP design)", "front-right foot 5 cm back / 12 cm in, hidden, FR double stance", dict(foot_shift={"FR": (-0.05, 0.12, -0.01), "HL": (0, 0, -0.01), "HR": (0, 0, -0.01)}, hide=True, warps=FRDBL)),
    ("hidden posture + FR double stance", "two short FR swings per cycle (0.40-0.50, 0.90-1.00)", dict(foot_shift=S6, hide=True, warps=FRDBL)),
    ("+ forward trim 0.5", "network sees vx + 0.5 m/s", dict(foot_shift=S6, hide=True, warps=FRDBL, trim=(0.5, 0, 0))),
    ("+ trims (0.7, 0.2, 0.2) + yaw loop K 0.5 (final design)", "the design written into the weights", dict(foot_shift=S6, hide=True, warps=FRDBL, trim=(0.7, 0.2, 0.2), yaw_K=0.5)),
    ("ablation: final without hiding the posture", "posture offset and FR re-timing added to the outputs, inputs not shifted", dict(foot_shift=S6, hide=False, warps=FRDBL, trim=(0.7, 0.2, 0.2), yaw_K=0.5)),
    ("ablation: final without FR re-timing", "hidden posture + trims + yaw loop, FR keeps its original swing", dict(foot_shift=S6, hide=True, trim=(0.7, 0.2, 0.2), yaw_K=0.5)),
    ("ablation: final without posture offset", "FR re-timing + trims + yaw loop only", dict(warps=FRDBL, trim=(0.7, 0.2, 0.2), yaw_K=0.5)),
    ("final design + tilt feedback (rejected)", "extra foot shift 0.1 m per unit of projected gravity x/y", dict(proto={"tilt_gain": 0.1})),
    ("final design + get-up reflex (rejected)", "extra 5 cm foot shift when g_x + g_y > 0.35", dict(proto={"getup": dict(theta=0.35, delta=0.1, extra_shift=(0.05, 0.05, 0.0))})),
]


def _job(a):
    i, cname, seed = a
    global _net, _N
    if _net is None:
        _net = NumpyPolicy(load_onnx_layers(os.path.join(P, "policy.onnx")))
        _N = np.load(os.path.join(ROOT, "policies", "edited", "edit_data.npz"))["N"]
    name, desc, kw = ATTEMPTS[i]
    if "proto" in kw:
        d = copy.deepcopy(DESIGN); d.update(kw["proto"])
        pol = ProtoEdit(_net, _N, d, faulty=True)
    else:
        pol = _net if not kw else Variant(_net, _N, **kw)
    r = run_episode(pol, CFG, CMDS[cname], T=12.0, fault="locked", seed=seed)
    return i, cname, r


def main():
    jobs = [(i, c, s) for i in range(len(ATTEMPTS)) for c in CMDS for s in range(3)]
    with Pool(max(1, os.cpu_count() - 2)) as p:
        R = p.map(_job, jobs)
    out = []
    for i, (name, desc, kw) in enumerate(ATTEMPTS):
        row = dict(order=i, name=name, description=desc, per_command={})
        for c in CMDS:
            rs = [r for (j, cc, r) in R if j == i and cc == c]
            row["per_command"][c] = {k: float(np.mean([r[k] for r in rs])) for k in
                                     ("broken_leg_ground_frac", "vx", "vy", "wz", "yaw_deg", "lat_rate_cm_s", "tilt_deg", "height_m", "survival_s")}
            row["per_command"][c]["falls"] = int(sum(r["fell"] for r in rs))
        out.append(row)
        pc = row["per_command"]
        print(f"{i:2d} {name:58s} | stand broken {pc['stand']['broken_leg_ground_frac']:.2f} tilt {pc['stand']['tilt_deg']:4.1f} f{pc['stand']['falls']}"
              f" | fwd0.3 broken {pc['fwd_0.3']['broken_leg_ground_frac']:.2f} vx {pc['fwd_0.3']['vx']:.2f} tilt {pc['fwd_0.3']['tilt_deg']:4.1f} f{pc['fwd_0.3']['falls']}"
              f" | turnL broken {pc['turn_left']['broken_leg_ground_frac']:.2f} wz {pc['turn_left']['wz']:.2f} f{pc['turn_left']['falls']}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(dict(protocol=dict(fault="locked front-left, from the start", T_s=12.0, seeds=[0, 1, 2], commands=CMDS,
                                 note="prototypes evaluated outside the weights; the last row is the design compiled into policies/edited"),
                   attempts=out), open(os.path.join(ROOT, "results", "attempts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
