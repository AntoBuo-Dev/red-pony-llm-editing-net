#!/usr/bin/env python3
"""Compact JSON exports of the figure data for the website (results/site/, each file < ~300 KB).

    python analysis/export_site.py

gait_contacts.json        foot contacts (and broken-leg contact) at 50 Hz, 2 s at 0.3 m/s, four conditions
friction_sweep.json       forward speed and broken-leg-on-floor fraction against floor friction, per policy
fault_while_walking.json  per-run outcome against the gait phase at the moment the leg locks
goal_paths.json           x, y paths (5 Hz) of goal-reaching runs (5 seeds per policy, goal and mode), goals, success
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from sim_harness import run_episode  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "site")
LEGS = ["FL", "FR", "HL", "HR"]


def r3(x, n=3):
    return [round(float(v), n) for v in x]


def gait_contacts():
    conds = [("original", None, "original, four legs"), ("original", "locked", "original, front-left leg locked"),
             ("edited", "locked", "edited v1, front-left leg locked"), ("edited_v2", "locked", "edited v2, front-left leg locked")]
    out = dict(description="ground contact (1/0) per foot and of the broken leg (thigh/shin/foot), 50 Hz, command 0.3 m/s, "
                           "window 5-7 s after the command starts, seed 0", dt_s=0.02, legs=LEGS, conditions=[])
    for p, fault, label in conds:
        path = os.path.join(ROOT, "policies", p)
        pol = NumpyPolicy(load_onnx_layers(os.path.join(path, "policy.onnx")))
        broken = []

        def on_step(sim, i):
            broken.append(int(sim.leg_ground_contact("FL")) if fault else 0)
        r = run_episode(pol, os.path.join(path, "config.yaml"), (0.3, 0.0, 0.0), T=8, fault=fault, seed=0, record=True,
                        on_step=on_step)
        c = r["contacts"].astype(int)
        k0, k1 = 250, 350
        feet = {l: c[k0:k1, j].tolist() for j, l in enumerate(LEGS)}
        if fault:
            feet["FL"] = [0] * (k1 - k0)
        out["conditions"].append(dict(policy=p, fault=fault or "none", label=label, t0_s=k0 * 0.02, contacts=feet,
                                      broken_leg=broken[k0:k1]))
    return out


def friction_sweep():
    F = json.load(open(os.path.join(RES, "eval_friction_sweep.json")))
    out = dict(description="forward 0.3 m/s for 20 s at fixed friction, 5 seeds; vx = mean forward speed [m/s]; "
                           "broken = mean fraction of time the broken leg touches the floor; leaning = episodes (of 5) with broken > 5 %",
               frictions=F["protocol"]["frictions"], series=[])
    for p in ("original", "edited", "edited_v2", "trained_3leg_reference"):
        for f in ("none", "locked"):
            a = [x for x in F["per_friction"] if x["policy"] == p and x["fault"] == f]
            if not a:
                continue
            out["series"].append(dict(policy=p, fault=f, vx=r3([x["vx_mean"] for x in a]),
                                      broken=r3([x["broken_leg_ground_frac_mean"] for x in a]) if f != "none" else None,
                                      leaning=[x["leaning_on_broken_leg_episodes"] for x in a] if f != "none" else None,
                                      falls=[x["falls"] for x in a]))
    return out


def phase_at_fault(seed, fault_at=4.0, dt=0.02, freq=2.5):
    rs = np.random.RandomState(seed)          # same draws as SoloSim.reset: joint noise, then phase
    rs.uniform(-0.05, 0.05, 12)
    ph = rs.uniform()
    t = 0.0
    while not t >= fault_at:                  # the leg locks at the first policy step with t >= fault_at
        ph = (ph + dt * freq) % 1.0
        t += dt
    return ph


def fault_while_walking():
    E = json.load(open(os.path.join(RES, "eval_fault_while_walking.json")))
    out = dict(description="leg locks at t = 4 s (3 s into the command); recovered = broken leg on the floor < 5 % of the "
                           "time after the fault; phase = gait-clock phase when the leg locks (front-left foot unloaded for "
                           "phase in ~0.95-0.45)", runs=[])
    for e in E["episodes"]:
        out["runs"].append(dict(policy=e["policy"], command=e["command"], seed=e["seed"],
                                phase_at_fault=round(phase_at_fault(e["seed"]), 3),
                                recovered=bool(e["broken_leg_ground_frac"] < 0.05 and not e["fell"]),
                                broken=round(e["broken_leg_ground_frac"], 3), fell=e["fell"]))
    return out


def goal_paths():
    G = json.load(open(os.path.join(RES, "goal_reaching.json")))
    out = dict(description="goal reaching with the front-left leg locked (and v2/original on four legs); paths at 5 Hz "
                           "relative to the start pose, seeds 100-104; success = within 0.3 m before 2*distance/0.2 m/s",
               goals=G["protocol"]["goals"], radius_m=G["protocol"]["radius_m"], vx=G["protocol"]["vx"],
               mode_B_gain=G["protocol"]["k_bearing"], mode_B_wz_max=G["protocol"]["wz_max"], runs=[],
               summary=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in G["summary"]])
    for r in G["runs"]:
        if r["seed"] >= 105:
            continue
        if r["fault"] is None and r["policy"] not in ("edited_v2", "original"):
            continue
        pth = np.array(r["path"])[::2]                       # stored at 10 Hz -> 5 Hz
        out["runs"].append(dict(policy=r["policy"], fault=r["fault"] or "none", mode=r["mode"], goal=r["goal_name"],
                                seed=r["seed"], success=r["success"],
                                time_to_goal_s=r["time_to_goal_s"], x=r3(pth[:, 0], 2), y=r3(pth[:, 1], 2)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("gait_contacts", gait_contacts), ("friction_sweep", friction_sweep),
                     ("fault_while_walking", fault_while_walking), ("goal_paths", goal_paths)):
        d = fn()
        p = os.path.join(OUT, name + ".json")
        json.dump(d, open(p, "w"), separators=(",", ":"))
        print(f"{p}: {os.path.getsize(p) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
