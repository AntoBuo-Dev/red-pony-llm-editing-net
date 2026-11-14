#!/usr/bin/env python3
"""Goal-reaching protocol (v2 primary protocol). Numpy policy + MuJoCo, no training code.

    python harness/goal_reaching.py --policy edited_v2=policies/edited_v2 --fault locked --seeds 10 --seed_offset 100 \
        --out results/goal_reaching_edited_v2.json

The robot starts at the origin facing the goal (yaw = bearing of the goal), stands 1 s, then:
  mode A ("pure network"): fixed command (vx, 0, 0) for the whole run; any heading correction must come
                           from the network itself.
  mode B ("joystick"):     vx constant, wz = clip(k * bearing error, +-wz_max), recomputed every step
                           from the true base pose, like an operator steering toward the goal.
Success = base within 0.3 m (horizontal) of the goal before the time limit and without a fall.
Time limit = 2 * distance / 0.2 m/s (0.2 m/s ~ the edited policy's walking speed on three legs).
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_harness import SoloSim  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402

GOALS = {"2m": (2.0, 0.0), "4m": (4.0, 0.0), "3m_1m": (3.0, 1.0)}
VX, K_BEARING, WZ_MAX, RADIUS, REF_SPEED = 0.3, 1.0, 0.5, 0.3, 0.2


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def run_goal(policy, cfg, goal, mode="A", fault="locked", seed=0, record=False, sim_kw=None, settle=1.0):
    rs = np.random.RandomState(1000 + seed)          # same per-seed randomisation as harness/evaluate.py
    kw = dict(friction=rs.uniform(0.7, 1.1), mass_offset=rs.uniform(-0.1, 0.2), seed=seed)
    kw.update(sim_kw or {})
    sim = SoloSim(policy, cfg, fault=fault, **kw)
    gx, gy = goal
    dist0 = math.hypot(gx, gy)
    bearing0 = math.atan2(gy, gx)
    sim.d.qpos[3:7] = yaw_quat(bearing0)              # face the goal
    import mujoco
    mujoco.mj_forward(sim.m, sim.d)
    t_limit = 2.0 * dist0 / REF_SPEED
    for _ in range(int(settle / sim.dt)):
        sim.policy_step()
    x0, y0 = sim.d.qpos[0], sim.d.qpos[1]
    path, reached, t_reach, fell = [], False, None, False
    n = int(t_limit / sim.dt)
    for i in range(n):
        x, y, z, yaw, up = sim.base()
        px, py = x - x0, y - y0
        if mode == "A":
            sim.cmd = np.array([VX, 0.0, 0.0])
        else:
            err = math.atan2(gy - py, gx - px) - yaw
            err = (err + math.pi) % (2 * math.pi) - math.pi
            sim.cmd = np.array([VX, 0.0, float(np.clip(K_BEARING * err, -WZ_MAX, WZ_MAX))])
        sim.policy_step()
        x, y, z, yaw, up = sim.base()
        px, py = x - x0, y - y0
        path.append((px, py))
        if sim.fallen():
            fell = True
            break
        if math.hypot(gx - px, gy - py) <= RADIUS:
            reached, t_reach = True, (i + 1) * sim.dt
            break
    p = np.array(path)
    d = np.hypot(gx - p[:, 0], gy - p[:, 1])
    res = dict(goal=list(goal), mode=mode, fault=fault, seed=seed, success=bool(reached and not fell), fell=fell,
               time_to_goal_s=t_reach, time_limit_s=t_limit, final_miss_m=float(d[-1]), min_miss_m=float(d.min()),
               path_m=float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0,
               friction=kw["friction"])
    if record:
        res["path"] = p
    return res


_POL = {}


def _job(a):
    name, path, goal_name, mode, fault, seed = a
    if path not in _POL:
        _POL[path] = NumpyPolicy(load_onnx_layers(os.path.join(path, "policy.onnx")))
    r = run_goal(_POL[path], os.path.join(path, "config.yaml"), GOALS[goal_name], mode=mode,
                 fault=None if fault == "none" else fault, seed=seed, record=True)
    r["path"] = r["path"][::5].round(3).tolist()        # 10 Hz is enough for the plots
    r.update(policy=name, goal_name=goal_name)
    return r


def summarise(R):
    out = []
    keys = sorted({(r["policy"], r["fault"] or "none", r["mode"], r["goal_name"]) for r in R})
    for k in keys:
        rs = [r for r in R if (r["policy"], r["fault"] or "none", r["mode"], r["goal_name"]) == k]
        ok = [r for r in rs if r["success"]]
        out.append(dict(policy=k[0], fault=k[1], mode=k[2], goal=k[3], n=len(rs), successes=len(ok),
                        success_rate=len(ok) / len(rs), falls=int(sum(r["fell"] for r in rs)),
                        mean_time_to_goal_s=float(np.mean([r["time_to_goal_s"] for r in ok])) if ok else None,
                        mean_final_miss_m=float(np.mean([r["final_miss_m"] for r in rs])),
                        mean_min_miss_m=float(np.mean([r["min_miss_m"] for r in rs])),
                        mean_path_m=float(np.mean([r["path_m"] for r in rs]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--faults", nargs="+", default=["locked"])
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--goals", nargs="+", default=list(GOALS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pols = [p.split("=", 1) for p in a.policy]
    jobs = [(n, p, g, m, f, s) for n, p in pols for f in a.faults for m in a.modes for g in a.goals
            for s in range(a.seed_offset, a.seed_offset + a.seeds)]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        R = pool.map(_job, jobs, chunksize=1)
    S = summarise(R)
    for s in S:
        t = f"{s['mean_time_to_goal_s']:.1f}" if s["mean_time_to_goal_s"] else "  -"
        print(f"{s['policy']:24s} {s['fault']:7s} mode {s['mode']} {s['goal']:6s} success {s['successes']:2d}/{s['n']}  "
              f"t {t:>5s} s  final miss {s['mean_final_miss_m']:.2f} m  min miss {s['mean_min_miss_m']:.2f} m  path {s['mean_path_m']:.1f} m  falls {s['falls']}")
    if a.out:
        json.dump(dict(protocol=dict(goals=GOALS, vx=VX, k_bearing=K_BEARING, wz_max=WZ_MAX, radius_m=RADIUS,
                                     time_limit="2 * distance / 0.2 m/s", seeds=list(range(a.seed_offset, a.seed_offset + a.seeds)),
                                     settle_s=1.0, policies=dict(pols)), summary=S, runs=R),
                  open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
