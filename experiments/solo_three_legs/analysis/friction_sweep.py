#!/usr/bin/env python3
"""Friction sweep: forward 0.3 m/s for 20 s at fixed floor friction, 5 seeds, every policy, four legs and
front-left leg locked. Writes results/eval_friction_sweep.json and results/friction_sweep.png.

    python analysis/friction_sweep.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from sim_harness import run_episode  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402

FRICTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]
POLICIES = ["original", "edited", "trained_3leg_reference"]
COL = {"original": "#2a78d6", "edited": "#eb6834", "trained_3leg_reference": "#1baf7a"}


def _job(a):
    p, fault, mu, seed = a
    path = os.path.join(ROOT, "policies", p)
    pol = NumpyPolicy(load_onnx_layers(os.path.join(path, "policy.onnx")))
    r = run_episode(pol, os.path.join(path, "config.yaml"), (0.3, 0.0, 0.0), T=20.0, fault=fault, seed=seed,
                    sim_kw=dict(friction=mu))
    return dict(policy=p, fault=fault or "none", friction=mu, seed=seed, fell=r["fell"], vx=r["vx"],
                broken_leg_ground_frac=r["broken_leg_ground_frac"], lat_rate_cm_s=r["lat_rate_cm_s"], yaw_deg=r["yaw_deg"],
                tilt_deg=r["tilt_deg"])


def main():
    jobs = [(p, f, mu, s) for p in POLICIES for f in (None, "locked") for mu in FRICTIONS for s in range(5)]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        E = pool.map(_job, jobs)
    agg = []
    for p in POLICIES:
        for f in ("none", "locked"):
            for mu in FRICTIONS:
                rs = [e for e in E if e["policy"] == p and e["fault"] == f and e["friction"] == mu]
                agg.append(dict(policy=p, fault=f, friction=mu, n=len(rs), falls=int(sum(e["fell"] for e in rs)),
                                vx_mean=float(np.mean([e["vx"] for e in rs])),
                                broken_leg_ground_frac_mean=float(np.mean([e["broken_leg_ground_frac"] for e in rs])) if f != "none" else None,
                                leaning_on_broken_leg_episodes=int(sum(e["broken_leg_ground_frac"] > 0.05 for e in rs)) if f != "none" else None))
    json.dump(dict(protocol=dict(command=[0.3, 0, 0], T_s=20.0, seeds=list(range(5)), frictions=FRICTIONS), per_friction=agg, episodes=E),
              open(os.path.join(ROOT, "results", "eval_friction_sweep.json"), "w"), indent=1)
    for a in agg:
        print(a)
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#e4e3df", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.labelcolor": "#52514e", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for p in POLICIES:
        for f, ls in (("none", ":"), ("locked", "-")):
            if p == "trained_3leg_reference" and f == "none":
                continue
            a = [x for x in agg if x["policy"] == p and x["fault"] == f]
            lab = f"{p.replace('trained_3leg_reference', 'RL reference')}, {'four legs' if f == 'none' else 'leg locked'}"
            axes[0].plot(FRICTIONS, [x["vx_mean"] for x in a], ls, color=COL[p], linewidth=2, marker="o", markersize=7,
                         markeredgecolor="#fcfcfb", markeredgewidth=2, label=lab)
            if f == "locked":
                axes[1].plot(FRICTIONS, [x["leaning_on_broken_leg_episodes"] for x in a], ls, color=COL[p], linewidth=2, marker="o",
                             markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2, label=lab)
    axes[0].axhline(0.3, color="#52514e", linewidth=1, linestyle=(0, (2, 3)))
    axes[0].set_xlabel("floor friction coefficient"); axes[0].set_ylabel("forward speed [m/s] (command 0.3)")
    axes[1].set_xlabel("floor friction coefficient"); axes[1].set_ylabel("episodes leaning on the broken leg (of 5)")
    for ax in axes:
        ax.grid(color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "friction_sweep.png"), dpi=130)


if __name__ == "__main__":
    main()
