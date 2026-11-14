#!/usr/bin/env python3
"""Plots for results/ (PNG). Reads results/results.json and re-runs a few recorded rollouts.

    python analysis/make_plots.py
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from sim_harness import run_episode  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402

OUT = os.path.join(ROOT, "results")
# validated categorical slots 1-3 (dataviz reference palette, light surface)
COL = {"original": "#2a78d6", "edited": "#eb6834", "edited_v2": "#6250d6", "trained_3leg_reference": "#1baf7a"}
LABEL = {"original": "original", "edited": "edited v1 (by hand)", "edited_v2": "edited v2 (+ heading hold)",
         "trained_3leg_reference": "RL fine-tuned on 3 legs (reference)"}
POLS = ["original", "edited", "edited_v2", "trained_3leg_reference"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})


def policy_path(name):
    return os.path.join(ROOT, "policies", name)


def load_pol(name):
    return NumpyPolicy(load_onnx_layers(os.path.join(policy_path(name), "policy.onnx")))


def bars(R):
    """summary bars: two panels (four legs / three legs), bars per policy."""
    ev = R.get("heldout_seeds", R["from_start"])
    summ = {(s["policy"], s["fault"]): s for s in ev["summary"]}
    pols = [p for p in POLS if any(k[0] == p for k in summ)]
    metrics = [("turn_rate_ratio", "turn rate / command (turn and spin commands)"),
               ("broken_leg_ground_frac", "broken leg touching the ground [% of time]"),
               ("fwd_speed_ratio", "forward speed / command (0.2-0.5 m/s)"),
               ("fwd_abs_lat_drift_cm_s", "|lateral drift| walking forward [cm/s]"),
               ("fwd_abs_heading_drift_deg", "|heading change| in 20 s walking forward [deg]"),
               ("mean_tilt_deg", "mean body tilt [deg]")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8))
    for ax, (key, title) in zip(axes.flat, metrics):
        xs, labels = [], []
        x = 0
        for fault, fl in (("none", "four legs"), ("locked", "front-left leg locked")):
            for p in pols:
                s = summ.get((p, fault))
                if s is None or (fault == "none" and p == "trained_3leg_reference"):
                    continue
                if key == "broken_leg_ground_frac" and fault == "none":
                    continue
                v = s[key]
                if key in ("broken_leg_ground_frac",):
                    v = 100 * v
                ax.bar(x, v, width=0.8, color=COL[p], edgecolor=SURF, linewidth=2,
                       hatch="//" if p == "trained_3leg_reference" else None)
                ax.text(x, v, f"{v:.0f}" if abs(v) >= 10 else (f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}"), ha="center", va="bottom", fontsize=8, color=INK)
                xs.append(x); labels.append(fl)
                x += 1
            x += 0.6
        # group labels
        groups = {}
        for xi, l in zip(xs, labels):
            groups.setdefault(l, []).append(xi)
        ax.set_xticks([np.mean(v) for v in groups.values()]); ax.set_xticklabels(list(groups.keys()))
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[p], hatch="//" if p == "trained_3leg_reference" else None) for p in pols]
    fig.legend(handles, [LABEL[p] for p in pols], loc="upper center", ncol=len(pols), frameon=False, fontsize=9)
    fig.suptitle(" ", fontsize=4)
    fig.text(0.01, 0.005, "MuJoCo simulation, leg broken from the start, 11 commands x 10 held-out seeds (100-109) x 20 s per condition; "
             "no falls. Passive-leg condition (not plotted): original and edited fall within 1 s in 55/55 episodes.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(os.path.join(OUT, "summary_bars.png"), dpi=130)
    plt.close(fig)


def trajectories(R):
    conds = [("original", None, "original, four legs"), ("original", "locked", "original, front-left leg locked"),
             ("edited", "locked", "edited v1, front-left leg locked"), ("edited_v2", "locked", "edited v2, front-left leg locked"),
             ("edited_v2", None, "edited v2, four legs")]
    if os.path.exists(policy_path("trained_3leg_reference")):
        conds.append(("trained_3leg_reference", "locked", "RL fine-tuned (reference), leg locked"))
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 3.9), sharex=True, sharey=True)
    for ax, (p, fault, title) in zip(axes, conds):
        pol = load_pol(p)
        for seed in range(5):
            r = run_episode(pol, os.path.join(policy_path(p), "config.yaml"), (0.3, 0.0, 0.0), T=20, fault=fault, seed=seed, record=True)
            tr = r["traj"]
            ax.plot(tr[:, 3], tr[:, 4], color=COL[p], linewidth=2, alpha=0.85)
            ax.plot(tr[-1, 3], tr[-1, 4], "o", color=COL[p], markersize=5, markeredgecolor=SURF, markeredgewidth=2)
        ax.plot([0, 6], [0, 0], color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("forward [m]"); ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.set_aspect("equal")
    axes[0].set_ylabel("left [m]")
    fig.text(0.01, 0.01, "command: 0.3 m/s straight ahead for 20 s, 5 seeds; dashed line = commanded path (6 m).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "trajectories_fwd0.3.png"), dpi=130)
    plt.close(fig)


def gait_diagrams():
    conds = [("original", None, "original, four legs"), ("original", "locked", "original, front-left leg locked"),
             ("edited", "locked", "edited v1, front-left leg locked"), ("edited_v2", "locked", "edited v2, front-left leg locked")]
    fig, axes = plt.subplots(len(conds), 1, figsize=(10, 8.0), sharex=True)
    rows = ["FL", "FR", "HL", "HR", "broken leg\n(knee/shin/foot)"]
    for ax, (p, fault, title) in zip(axes, conds):
        pol = load_pol(p)
        holder = {}

        def on_step(sim, i, holder=holder):
            holder.setdefault("b", []).append(sim.leg_ground_contact("FL") if fault else False)
        r = run_episode(pol, os.path.join(policy_path(p), "config.yaml"), (0.3, 0.0, 0.0), T=8, fault=fault, seed=0,
                        record=True, on_step=on_step)
        c = r["contacts"].astype(float)
        b = np.array(holder["b"], float)
        k0 = int(5.0 / 0.02); k1 = k0 + int(2.0 / 0.02)
        t = np.arange(k0, k1) * 0.02
        M = np.c_[c[k0:k1], b[k0:k1]].T
        if fault:
            M[0] = 0.0          # FL foot belongs to the broken leg: shown in the last row instead
        for j in range(M.shape[0]):
            on = M[j] > 0.5
            for k in np.where(on)[0]:
                ax.add_patch(plt.Rectangle((t[k], j - 0.38), 0.02, 0.76, color="#c94040" if j == 4 else COL[p], linewidth=0))
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.5)
        ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(t[0], t[-1] + 0.02)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    axes[-1].set_xlabel("time [s] (command 0.3 m/s; filled = in contact with the ground)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gait_diagrams.png"), dpi=130)
    plt.close(fig)


def speed_tracking(R):
    per = R.get("heldout_seeds", R["from_start"])["per_command"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for p, fault, ls, lab in (("original", "none", "-", "original, four legs"), ("original", "locked", "--", "original, leg locked"),
                              ("edited", "locked", "-", "edited v1, leg locked"), ("edited", "none", ":", "edited v1, four legs"),
                              ("edited_v2", "locked", "-", "edited v2, leg locked"),
                              ("trained_3leg_reference", "locked", "-.", "RL reference, leg locked")):
        pts = sorted([(a["cmd"][0], a["vx_mean"]) for a in per if a["policy"] == p and a["fault"] == fault and a["command"].startswith("fwd")])
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, ls, color=COL[p], linewidth=2, marker="o", markersize=8, markeredgecolor=SURF, markeredgewidth=2, label=lab)
    ax.plot([0.15, 0.55], [0.15, 0.55], color=INK2, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(0.5, 0.52, "speed = command", color=INK2, fontsize=8, ha="right")
    ax.set_xlabel("commanded forward speed [m/s]"); ax.set_ylabel("measured forward speed [m/s]")
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, handlelength=3.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed_tracking.png"), dpi=130)
    plt.close(fig)


def training_curve():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for run, p, lab in (("solo_walk", "original", "four-leg policy (original)"),
                        ("solo_3leg_finetune", "trained_3leg_reference", "fine-tune on three legs (reference)")):
        f = os.path.join(ROOT, "runs", run, "log.csv")
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        x = np.array([float(r["steps"]) for r in rows]) / 1e6
        y = np.array([float(r["mean_ep_ret"]) for r in rows])
        if run == "solo_3leg_finetune":
            x = x + 1200 * 2048 * 24 / 1e6
        ax.plot(x, y, color=COL[p], linewidth=2, label=lab)
    ax.set_xlabel("environment steps [millions]"); ax.set_ylabel("mean episode return")
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=130)
    plt.close(fig)


def midwalk(R):
    if "fault_midwalk" not in R:
        return
    per = R["fault_midwalk"]["per_command"]
    pols = [p for p in POLS if any(a["policy"] == p for a in per)]
    cmds = [c for c in ("stand", "fwd_0.3", "fwd_0.5", "turn_left", "turn_right") if any(a["command"] == c for a in per)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    wdt = 0.8 / len(pols)
    for i, p in enumerate(pols):
        v = [100 * next(a for a in per if a["policy"] == p and a["command"] == c)["broken_leg_ground_frac_mean"] for c in cmds]
        xs = np.arange(len(cmds)) + (i - (len(pols) - 1) / 2) * wdt
        ax.bar(xs, v, width=wdt * 0.92, color=COL[p], edgecolor=SURF, linewidth=2, label=LABEL[p],
               hatch="//" if p == "trained_3leg_reference" else None)
        for xx, vv in zip(xs, v):
            ax.text(xx, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(cmds))); ax.set_xticklabels(cmds)
    ax.set_ylabel("broken leg on the ground [% of time]")
    ax.set_title("leg locks while walking (t = 4 s), measured after the fault", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fault_while_walking.png"), dpi=130)
    plt.close(fig)


def goal_plots():
    f = os.path.join(OUT, "goal_reaching.json")
    if not os.path.exists(f):
        return
    G = json.load(open(f))
    goals = G["protocol"]["goals"]
    runs = [r for r in G["runs"] if r["fault"] == "locked"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True, sharey=True)
    for i, mode in enumerate(("A", "B")):
        for j, p in enumerate(POLS):
            ax = axes[i, j]
            for r in [r for r in runs if r["policy"] == (p if p != "edited" else "edited_v1") and r["mode"] == mode]:
                pth = np.array(r["path"])
                ax.plot(pth[:, 0], pth[:, 1], color=COL[p], linewidth=1.6, alpha=0.9 if r["success"] else 0.35)
            for gname, (gx, gy) in goals.items():
                ax.add_patch(plt.Circle((gx, gy), 0.3, fill=False, color=INK, linewidth=1.2))
            ok = sum(r["success"] for r in runs if r["policy"] == (p if p != "edited" else "edited_v1") and r["mode"] == mode)
            n = sum(1 for r in runs if r["policy"] == (p if p != "edited" else "edited_v1") and r["mode"] == mode)
            ax.set_title(f"{LABEL[p]}\nmode {mode}: {ok}/{n} goals reached", loc="left", fontsize=9, color=INK)
            ax.set_aspect("equal"); ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
            ax.set_xlim(-1.5, 6.5); ax.set_ylim(-3.5, 3.5)
            if i == 1: ax.set_xlabel("x [m]")
            if j == 0: ax.set_ylabel("y [m]")
    fig.text(0.01, 0.005, "Front-left leg locked, 10 held-out seeds x 3 goals (circles: 0.3 m success radius). Mode A: fixed command "
             "0.3 m/s toward the goal; mode B: joystick steering wz = clip(1.0 x bearing error, 0.5). Faded = goal not reached.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "goal_paths.png"), dpi=120)
    plt.close(fig)
    S = G["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    gl = list(goals)
    for ax, mode in zip(axes, ("A", "B")):
        wdt = 0.8 / len(POLS)
        for i, p in enumerate(POLS):
            pn = p if p != "edited" else "edited_v1"
            v = [100 * next((x["success_rate"] for x in S if x["policy"] == pn and x["mode"] == mode and x["goal"] == g
                             and x["fault"] == "locked"), 0) for g in gl]
            xs = np.arange(len(gl)) + (i - (len(POLS) - 1) / 2) * wdt
            ax.bar(xs, v, width=wdt * 0.92, color=COL[p], edgecolor=SURF, linewidth=2, label=LABEL[p],
                   hatch="//" if p == "trained_3leg_reference" else None)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(gl))); ax.set_xticklabels([f"goal {g}" for g in gl])
        ax.set_title(f"mode {mode}: " + ("fixed command, network alone" if mode == "A" else "joystick steering toward the goal"),
                     loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.set_ylim(0, 112)
    axes[0].set_ylabel("goal reached [% of 10 runs]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "goal_summary.png"), dpi=130)
    plt.close(fig)


def main():
    R = json.load(open(os.path.join(OUT, "results.json")))
    goal_plots(); bars(R); speed_tracking(R); midwalk(R); training_curve(); trajectories(R); gait_diagrams()
    print("plots written to", OUT)


if __name__ == "__main__":
    main()
