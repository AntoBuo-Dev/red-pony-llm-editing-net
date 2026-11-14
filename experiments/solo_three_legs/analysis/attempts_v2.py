#!/usr/bin/env python3
"""Re-run the main v2 (heading-hold) attempts as prototypes around the v1 network -> results/attempts_v2.json.

    python analysis/attempts_v2.py

Two small protocols (design seeds 0-9, never the held-out seeds 100-109):
  four legs: forward 0.3 m/s and sideways right 0.2 m/s, 20 s -> |heading change| (did the change break normal walking?)
  three legs, goal reaching mode A (fixed command toward the goal), goals 2 m, 4 m, (3 m, 1 m) -> goals reached
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
sys.path.insert(0, os.path.join(ROOT, "edit"))
from proto_edit_v2 import ProtoEditV2  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402
from sim_harness import run_episode  # noqa: E402
from goal_reaching import run_goal, GOALS  # noqa: E402

V1 = os.path.join(ROOT, "policies", "edited")
CFG = os.path.join(V1, "config.yaml")
ED2 = dict(np.load(os.path.join(ROOT, "policies", "edited_v2", "edit_data.npz")))     # eager: workers fork
W = np.load(os.path.join(V1, "edit_data.npz"))["w"]
PRED_ALL = [i for i in range(47) if i not in (33, 34, 35)]
P1_COLS = list(range(36, 45)) + [45, 46, 6, 7, 8]
H1_GAINS = dict(K_th=3.0, K_I=3.0, L_th=0.35, L_I=0.3, s1=1.0, s2=1.0, s_y=10.0)
FINAL = dict(K_th=3.0, K_I=1.5, L_th=0.2, L_I=0.34, s1=1.0, s2=1.0, s_y=4.0, th_b=-0.1)
_net = None
_consts = {}


class Fold:
    """H1 recipe for the register channels: the original units see the four-leg mean of last_action[FL]."""

    def __init__(self, net, mean):
        self.net, self.mean = net, mean

    def __call__(self, obs):
        o = np.array(obs, np.float64); o[:, 33:36] = self.mean
        return self.net(o)


def consts(kind):
    c = dict(w=W, basis=ED2["basis"], m_v=float(ED2["m_v"]))
    if kind == "mixed_all":
        c.update(pred=ED2["pred"], pred_cols=PRED_ALL)
    else:
        c.update(pred=_consts["p1"], pred_cols=P1_COLS)
    return c


VARIANTS = [
    ("v1 (no heading hold)", "v1", None, None),
    ("fold last_action[FL] (H1 recipe), no hold", "fold", None, None),
    ("predictor from four-leg data (14 inputs), no hold", "proto", "p1", dict(FINAL, K_th=0.0, K_I=0.0)),
    ("predictor from four- and three-leg data (44 inputs), no hold", "proto", "mixed_all", dict(FINAL, K_th=0.0, K_I=0.0)),
    ("+ heading hold, H1 gains and limits, s_y = 10", "proto", "mixed_all", dict(H1_GAINS)),
    ("+ heading hold, bounded registers (K_I 3, no crab offset)", "proto", "mixed_all", dict(FINAL, K_I=3.0, L_I=0.17, th_b=0.0)),
    ("+ extra side trim 0.2 m/s", "proto", "mixed_all", dict(FINAL, K_I=3.0, L_I=0.17, th_b=0.0, extra_trim=(0.0, 0.2, 0.0))),
    ("final: K_I 1.5, crab offset 0.1 rad", "proto", "mixed_all", dict(FINAL)),
]


def make(i):
    global _net
    if _net is None:
        _net = NumpyPolicy(load_onnx_layers(os.path.join(V1, "policy.onnx")))
    name, kind, ck, p = VARIANTS[i]
    if kind == "v1":
        return _net
    if kind == "fold":
        return Fold(_net, _consts["la_mean"])
    return ProtoEditV2(_net, consts(ck), p=p)


def _four(a):
    i, cmd, seed = a
    r = run_episode(make(i), CFG, cmd, T=20.0, seed=seed)
    return i, tuple(cmd), r["yaw_deg"], r["vx"]


def _goal(a):
    i, g, seed = a
    r = run_goal(make(i), CFG, GOALS[g], mode="A", fault="locked", seed=seed)
    return i, g, r["success"]


def _init(p1, la_mean):
    _consts["p1"] = p1; _consts["la_mean"] = la_mean


def main():
    # data for the 14-input predictor and the fold mean: four-leg rollouts of v1 (same generator as build_edit_v2)
    import build_edit_v2 as B
    B.V1_PATH = V1
    with Pool(max(1, os.cpu_count() - 2)) as p:
        H = np.concatenate(p.map(B._v1_rollout, [(s, "four") for s in range(20)]))
    p1 = B.ridge(H[:, P1_COLS], H[:, 33:36], 1e-3)
    la_mean = H[:, 33:36].mean(0)
    jobs4 = [(i, c, s) for i in range(len(VARIANTS)) for c in ((0.3, 0.0, 0.0), (0.0, -0.2, 0.0)) for s in range(10)]
    jobsg = [(i, g, s) for i in range(len(VARIANTS)) for g in GOALS for s in range(10)]
    with Pool(max(1, os.cpu_count() - 2), initializer=_init, initargs=(p1, la_mean)) as p:
        R4 = p.map(_four, jobs4)
        RG = p.map(_goal, jobsg)
    out = []
    for i, (name, kind, ck, prm) in enumerate(VARIANTS):
        row = dict(order=i, name=name, parameters=prm)
        for c, key in (((0.3, 0.0, 0.0), "four_legs_fwd0.3_abs_heading_deg"), ((0.0, -0.2, 0.0), "four_legs_side_right_abs_heading_deg")):
            row[key] = float(np.mean([abs(y) for (j, cc, y, v) in R4 if j == i and cc == c]))
        for g in GOALS:
            row[f"modeA_{g}_reached"] = int(sum(ok for (j, gg, ok) in RG if j == i and gg == g))
        out.append(row)
        print(f"{i} {name:62s} 4 legs |yaw| fwd {row['four_legs_fwd0.3_abs_heading_deg']:5.1f} side {row['four_legs_side_right_abs_heading_deg']:5.1f} | "
              f"mode A reached " + " ".join(f"{g} {row[f'modeA_{g}_reached']}/10" for g in GOALS))
    json.dump(dict(protocol="design seeds 0-9; four legs: 20 s; three legs: goal reaching mode A (see harness/goal_reaching.py)",
                   attempts=out), open(os.path.join(ROOT, "results", "attempts_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
