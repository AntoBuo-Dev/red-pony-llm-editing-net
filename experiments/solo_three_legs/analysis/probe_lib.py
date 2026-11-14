"""Probing helpers: detailed rollouts in the harness, phase-binned statistics, policy wrappers."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from sim_harness import SoloSim, load_cfg, LEGS  # noqa: E402,F401
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402,F401

NB = 40  # phase bins


def phase_of(obs):
    """gait phase in [0,1) from the clock observation (nan when standing)."""
    s, c = obs[..., 45], obs[..., 46]
    ph = (np.arctan2(s, c) / (2 * np.pi)) % 1.0
    return np.where((np.abs(s) + np.abs(c)) < 0.5, np.nan, ph)


def rollout(policy, cfg, cmd, T=10.0, settle=1.0, fault=None, seed=0, sim_kw=None, phase0=0.0):
    """returns dict of per-step arrays during the command phase."""
    kw = dict(seed=seed, phase0=phase0)
    kw.update(sim_kw or {})
    sim = SoloSim(policy, cfg, fault=fault, **kw)
    for _ in range(int(settle / sim.dt)):
        sim.policy_step()
    sim.cmd = np.array(cmd, float)
    L = {k: [] for k in ("obs", "act", "q", "dq", "rpy", "pos", "vb", "wb", "contact", "broken", "foot_b", "tau", "fallen")}
    for i in range(int(T / sim.dt)):
        o, a = sim.policy_step()
        d = sim.d
        w, x, y, z = d.qpos[3:7]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        vb, wb = sim.body_vel()
        R = d.xmat[1].reshape(3, 3)
        fb = np.array([R.T @ (d.site(f"{l}_foot").xpos - d.xpos[1]) for l in LEGS])
        L["obs"].append(o); L["act"].append(a); L["q"].append(d.qpos[7:].copy()); L["dq"].append(d.qvel[6:].copy())
        L["rpy"].append((roll, pitch, yaw)); L["pos"].append(d.qpos[:3].copy()); L["vb"].append(vb); L["wb"].append(wb)
        L["contact"].append(sim.foot_contacts()); L["broken"].append(sim.leg_ground_contact("FL")); L["foot_b"].append(fb)
        L["tau"].append(d.actuator_force.copy()); L["fallen"].append(sim.fallen())
        if sim.fallen():
            break
    out = {k: np.array(v) for k, v in L.items()}
    out["phase"] = phase_of(out["obs"])
    return out


def phase_binned(phase, X, nb=NB):
    """mean of X (N,...) in nb phase bins."""
    b = np.floor(phase * nb).astype(int) % nb
    out = np.zeros((nb,) + X.shape[1:])
    for k in range(nb):
        m = b == k
        out[k] = X[m].mean(0) if m.any() else np.nan
    return out


class Wrapped:
    """policy(obs) = net(obs_fn(obs)) + delta_fn(obs)  (prototype of an edit, evaluated outside the weights)."""

    def __init__(self, net, delta_fn=None, obs_fn=None):
        self.net, self.delta_fn, self.obs_fn = net, delta_fn, obs_fn

    def __call__(self, obs):
        o = self.obs_fn(obs) if self.obs_fn else obs
        a = self.net(o)
        if self.delta_fn is not None:
            a = a + self.delta_fn(obs)
        return a
