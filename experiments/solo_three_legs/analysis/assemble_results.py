#!/usr/bin/env python3
"""Merge the evaluation outputs into results/results.json (machine-readable, for the website).

    python analysis/assemble_results.py
inputs: results/eval_from_start.json, results/eval_fault_while_walking.json, results/attempts.json,
        policies/*/weight_edit.json, runs/*/log.csv
"""
import csv
import hashlib
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def augment(ev):
    """add falls_pct and mean tilt to the summary rows."""
    for s in ev["summary"]:
        rows = [a for a in ev["per_command"] if a["policy"] == s["policy"] and a["fault"] == s["fault"]]
        s["falls_pct"] = 100.0 * s["falls"] / max(1, s["episodes"])
        s["mean_tilt_deg"] = float(np.mean([a["tilt_deg_mean"] for a in rows]))
        s["mean_height_m"] = float(np.mean([a["height_m_mean"] for a in rows]))
        fwd = [a for a in rows if a["command"].startswith("fwd")]
        s["fwd_abs_heading_drift_deg"] = float(np.mean([a["abs_yaw_deg_mean"] for a in fwd])) if fwd else None
        turn = [a for a in rows if a["command"] in ("turn_left", "turn_right", "spin_left", "spin_right")]
        s["turn_rate_ratio"] = float(np.mean([a["wz_mean"] / a["cmd"][2] for a in turn])) if turn else None
        if s["fault"] == "none":
            s["broken_leg_ground_frac"] = None     # no broken leg (the field would be the FL foot duty factor)
    return ev


def train_info(run):
    f = os.path.join(ROOT, "runs", run, "log.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    last = rows[-1]
    cfg = json.load(open(os.path.join(ROOT, "runs", run, "train_config.json")))
    return dict(iterations_logged=int(last["iter"]) + 1, env_steps=int(float(last["steps"])), wall_time_s=float(last["time_s"]),
                final_mean_episode_return=float(last["mean_ep_ret"]), final_fall_rate=float(last["fall_rate"]),
                final_action_std=float(last["std"]), num_envs=cfg["num_envs"], horizon=cfg["horizon"],
                note="wall time includes periods where the CPU was shared with other jobs")


def main():
    out = dict(
        meta=dict(
            robot="SOLO-12 (Open Dynamic Robot Initiative), example-robot-data solo12.urdf converted to MJCF",
            simulator="MuJoCo 4.0.1, 2 ms physics step, policy at 50 Hz",
            policy="MLP 47 -> 512 -> 256 -> 128 -> 12, ELU",
            failure="front-left leg: motors off, joints locked at a tucked pose (HAA 0, HFE 1.6, KFE -2.9 rad)",
            training=dict(original=train_info("solo_walk"), trained_3leg_reference=train_info("solo_3leg_finetune")),
            onnx_sha256={p: sha(os.path.join(ROOT, "policies", p, "policy.onnx")) for p in ("original", "edited", "edited_v2", "trained_3leg_reference")},
            only_simulation=True),
    )
    for key, fn in (("from_start", "eval_from_start.json"), ("fault_midwalk", "eval_fault_while_walking.json"),
                    ("pushes", "eval_pushes.json"), ("heldout_seeds", "eval_heldout_seeds.json")):
        f = os.path.join(RES, fn)
        if os.path.exists(f):
            out[key] = augment(json.load(open(f)))
            if key in ("fault_midwalk", "heldout_seeds", "from_start"):   # recovered = broken leg on the ground < 5 % of the time after the fault
                ev = out[key]
                for a in ev["per_command"]:
                    eps = [e for e in ev["episodes"] if e["policy"] == a["policy"] and e["command"] == a["command"]]
                    a["recovered"] = int(sum(e["broken_leg_ground_frac"] < 0.05 and not e["fell"] for e in eps))
    f = os.path.join(RES, "goal_reaching.json")
    if os.path.exists(f):
        g = json.load(open(f))
        out["goal_reaching"] = dict(protocol=g["protocol"], summary=g["summary"],
                                    note="per-run paths and outcomes in results/goal_reaching.json")
    f = os.path.join(RES, "eval_friction_sweep.json")
    if os.path.exists(f):
        out["friction_sweep"] = json.load(open(f))
    f = os.path.join(RES, "attempts.json")
    if os.path.exists(f):
        out["attempts"] = json.load(open(f))
    f = os.path.join(RES, "attempts_v2.json")
    if os.path.exists(f):
        out["attempts_v2"] = json.load(open(f))
    f = os.path.join(ROOT, "policies", "edited", "weight_edit.json")
    if os.path.exists(f):
        we = json.load(open(f))
        out["edit"] = {k: we[k] for k in ("design", "checks", "freeing_action_error", "fault_signal", "bell_synthesis",
                                          "changed_entries", "fitted_to_data") if k in we}
    f = os.path.join(ROOT, "policies", "edited_v2", "weight_edit.json")
    if os.path.exists(f):
        we = json.load(open(f))
        out["edit_v2"] = {k: we[k] for k in ("v2_parameters", "checks", "freeing_action_error", "predictor_error",
                                             "fl_readout_error", "injection_noise_four_legs_rad_s", "changed_entries",
                                             "fitted_to_data", "m_v", "register_basis_columns_e_v_e1_e2") if k in we}
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=1)
    print("wrote", os.path.join(RES, "results.json"))
    for s in out.get("from_start", {}).get("summary", []):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
