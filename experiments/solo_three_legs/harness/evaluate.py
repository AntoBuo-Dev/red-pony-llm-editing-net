#!/usr/bin/env python3
"""Fixed evaluation protocol (numpy policy + MuJoCo, no training code).

    python harness/evaluate.py --policy original=policies/original --policy edited=policies/edited \
        --faults none locked passive --seeds 5 --T 20 --out results/results.json

Each episode: 1 s standing (zero command), then one command held for T seconds.
Commands (vx [m/s], vy [m/s], wz [rad/s]) are listed in COMMANDS. The seed sets the floor friction
(0.7-1.1), a base mass offset (-0.1..+0.2 kg), the initial joint angles (+-0.05 rad) and the gait phase.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_harness import run_episode, load_cfg  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402

COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd_0.2": (0.2, 0.0, 0.0),
    "fwd_0.3": (0.3, 0.0, 0.0),
    "fwd_0.4": (0.4, 0.0, 0.0),
    "fwd_0.5": (0.5, 0.0, 0.0),
    "turn_left": (0.3, 0.0, 0.5),
    "turn_right": (0.3, 0.0, -0.5),
    "spin_left": (0.0, 0.0, 0.5),
    "spin_right": (0.0, 0.0, -0.5),
    "side_left": (0.0, 0.2, 0.0),
    "side_right": (0.0, -0.2, 0.0),
}

_POL = {}


def get_policy(path):
    if path not in _POL:
        onnx = path if path.endswith(".onnx") else os.path.join(path, "policy.onnx")
        _POL[path] = NumpyPolicy(load_onnx_layers(onnx))
    return _POL[path]


def cfg_of(path):
    d = path if os.path.isdir(path) else os.path.dirname(path)
    return os.path.join(d, "config.yaml")


def _job(args):
    name, path, fault, cname, seed, T, fault_at, push = args
    on_step = None
    if push > 0:
        rs = np.random.RandomState(500 + seed)
        dirs = rs.uniform(0, 2 * np.pi, 100)

        def on_step(sim, i):   # a sideways/forward shove of `push` m/s every 4 s (random direction)
            if i > 0 and i % int(4.0 / sim.dt) == 0:
                a = dirs[i // int(4.0 / sim.dt)]
                sim.d.qvel[0] += push * np.cos(a); sim.d.qvel[1] += push * np.sin(a)
    r = run_episode(get_policy(path), cfg_of(path), COMMANDS[cname], T=T, fault=None if fault == "none" else fault, seed=seed,
                    sim_kw=dict(fault_at=fault_at), on_step=on_step)
    r.update(policy=name, command=cname, fault=fault)
    return r


def aggregate(R):
    keys = ["survival_s", "vx", "vy", "wz", "fwd_m", "lat_m", "yaw_deg", "lat_rate_cm_s", "yaw_rate_deg_s", "path_m", "net_m",
            "height_m", "tilt_deg", "dact_legs", "broken_leg_ground_frac"]
    out = {}
    for r in R:
        k = (r["policy"], r["fault"], r["command"])
        out.setdefault(k, []).append(r)
    agg = []
    for (p, f, c), rs in sorted(out.items()):
        a = dict(policy=p, fault=f, command=c, cmd=rs[0]["cmd"], n=len(rs), falls=int(sum(r["fell"] for r in rs)))
        for key in keys:
            v = np.array([r[key] for r in rs], float)
            a[key + "_mean"] = float(v.mean()); a[key + "_std"] = float(v.std())
        a["abs_lat_rate_cm_s_mean"] = float(np.mean([abs(r["lat_rate_cm_s"]) for r in rs]))
        a["abs_yaw_deg_mean"] = float(np.mean([abs(r["yaw_deg"]) for r in rs]))
        agg.append(a)
    return agg


def summary_rows(agg):
    """one row per (policy, fault): averages over commands (walking commands only for speed ratios)."""
    rows = {}
    for a in agg:
        rows.setdefault((a["policy"], a["fault"]), []).append(a)
    out = []
    for (p, f), al in sorted(rows.items()):
        walk = [a for a in al if a["command"].startswith("fwd")]
        straight = [a for a in al if a["cmd"][2] == 0.0 and a["command"] != "stand"]
        out.append(dict(policy=p, fault=f, episodes=int(sum(a["n"] for a in al)), falls=int(sum(a["falls"] for a in al)),
                        mean_survival_s=float(np.mean([a["survival_s_mean"] for a in al])),
                        fwd_speed_ratio=float(np.mean([a["vx_mean"] / a["cmd"][0] for a in walk])) if walk else None,
                        fwd_abs_lat_drift_cm_s=float(np.mean([a["abs_lat_rate_cm_s_mean"] for a in walk])) if walk else None,
                        straight_abs_heading_drift_deg=float(np.mean([a["abs_yaw_deg_mean"] for a in straight])) if straight else None,
                        fwd_distance_m=float(np.mean([a["fwd_m_mean"] for a in walk])) if walk else None,
                        dact_legs=float(np.mean([a["dact_legs_mean"] for a in al])),
                        broken_leg_ground_frac=float(np.mean([a["broken_leg_ground_frac_mean"] for a in al]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name=path (policy dir or .onnx)")
    ap.add_argument("--faults", nargs="+", default=["none", "locked"])
    ap.add_argument("--commands", nargs="+", default=list(COMMANDS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--fault_at", type=float, default=0.0,
                    help="seconds after reset at which the leg breaks (0 = broken from the start; the command starts at 1 s)")
    ap.add_argument("--seed_offset", type=int, default=0, help="first seed (use e.g. 100 for seeds never used while designing the edit)")
    ap.add_argument("--push", type=float, default=0.0, help="base velocity kick [m/s] every 4 s, random direction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pols = [p.split("=", 1) for p in a.policy]
    jobs = [(n, p, f, c, s, a.T, a.fault_at, a.push) for n, p in pols for f in a.faults for c in a.commands for s in range(a.seed_offset, a.seed_offset + a.seeds)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    agg = aggregate(R)
    summ = summary_rows(agg)
    print(f"{len(R)} episodes in {time.time() - t0:.0f} s")
    print(f"{'policy':10s} {'fault':8s} {'command':11s} falls  surv[s]   vx/cmd      vy     wz/cmd   lat[cm/s] yaw[deg]  path[m] dact  brokenleg")
    for x in agg:
        c = x["cmd"]
        print(f"{x['policy']:10s} {x['fault']:8s} {x['command']:11s} {x['falls']:2d}/{x['n']:<2d} {x['survival_s_mean']:6.1f} "
              f"{x['vx_mean']:6.3f}/{c[0]:.1f} {x['vy_mean']:6.3f} {x['wz_mean']:6.3f}/{c[2]:+.1f} {x['lat_rate_cm_s_mean']:8.2f} "
              f"{x['yaw_deg_mean']:8.1f} {x['path_m_mean']:7.2f} {x['dact_legs_mean']:.3f} {x['broken_leg_ground_frac_mean']:.2f}")
    print()
    for s in summ:
        print(s)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(protocol=dict(T=a.T, settle_s=1.0, seeds=a.seeds, seed_offset=a.seed_offset, fault_at_s=a.fault_at, push_m_s=a.push, commands={c: COMMANDS[c] for c in a.commands},
                                     faults=a.faults, policies=dict(pols)),
                       summary=summ, per_command=agg,
                       episodes=[{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in R]),
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
