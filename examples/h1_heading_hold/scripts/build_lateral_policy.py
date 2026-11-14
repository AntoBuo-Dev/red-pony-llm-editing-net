#!/usr/bin/env python3
"""Build the heading + lateral hold policy by editing the weights of an exported policy.

    python evaluation/heading_hold/build_lateral_policy.py <source policy dir> <output policy dir>

Like build_policy.py, plus the lateral hold of heading_lateral_circuit.py. Its velocity readout is a
ridge regression on the first hidden layer, fitted on closed-loop rollouts in the Python replica
(5 rounds: each round runs the loop with the previous readout and adds its data). The target is the
base velocity perpendicular to the reference heading of the heading hold. No gradient training.
"""
import argparse, json, math, os, shutil, sys
from multiprocessing import Pool
import numpy as np
import mujoco
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_harness import Sim, NumpyPolicy, load_onnx_layers
from free_units import free_units
import heading_lateral_circuit as C
import build_policy as BP

G = {}
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
clamp = lambda x, l: min(max(x, -l), l)


def torso_yaw(sim):
    R = sim.d.xmat[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


class Wrap:
    """Heading-hold network with an exploration offset on vy and the lateral loop run in Python."""
    def __init__(self, layers, est=None):
        self.net = NumpyPolicy(layers); self.est = est; self.off = 0.0; self.rec = []
    def __call__(self, obs):
        o = np.array(obs[0], np.float32); raw = o.copy()
        o[C.VY_COLS] += self.off
        a, hs = self.net(o[None], return_hidden=True); a = np.array(a[0])
        self.rec.append(o)
        if self.est is not None:
            w, b = self.est
            _, act_l = C.gates(raw)
            a[C.R3] = C.P["s3"] * clamp(raw[C.LA + C.R3] / C.P["s3"] + C.P["dt"] * float(hs[0][0] @ w + b), C.P["Lp"] * act_l)
        return a[None]


def _episode(seed):
    """72 s: a new forward/backward speed and vy offset every 8 s, a new height every 24 s."""
    rs = np.random.RandomState(5000 + seed)
    pol = Wrap(G["layers"], G["est"])
    sim = Sim(pol, G["deploy"]); h = 0.98; Y = []; Cn = []
    for k in range(int(72 / 0.02)):
        if k % 1200 == 0: h = rs.choice([0.98, rs.uniform(0.71, 0.98)])
        if k % 400 == 0:
            vx = rs.uniform(0.15, 1.0) if rs.rand() < 0.85 else rs.uniform(-0.3, -0.12)
            sim.cmd = np.array([vx, 0, 0.0]); pol.off = rs.uniform(-0.12, 0.12) * G["explore"]
        sim.set_height(h); moving = abs(sim.h_target - h) > 1e-6
        sim.policy_step()
        x, y, z, yaw, up = sim.base()
        if up < 0.5 or z < 0.35: pol.rec.pop(); break
        ref = torso_yaw(sim) + sim.action[C.R1] / C.P["s1"]        # reference heading of the hold
        v = sim.d.qvel[:2]
        Y.append(-math.sin(ref) * v[0] + math.cos(ref) * v[1]); Cn.append((seed * 100 + k // 400, k % 400, moving))
    return np.array(pol.rec, np.float32), np.array(Y), np.array(Cn)


def collect(layers, est, n, seed0, explore, deploy, workers):
    G.update(layers=layers, est=est, explore=explore, deploy=deploy)
    with Pool(workers) as p: R = p.map(_episode, range(seed0, seed0 + n))
    return [np.concatenate([r[i] for r in R]) for i in range(3)]


def fit(layers, D, freed1, lam=1e-2, wc=30.0):
    """Ridge on instantaneous samples + heavily weighted rows of per-condition means (the mean is
    what the integrator of the loop sees)."""
    keep = np.setdiff1d(np.arange(512), freed1)
    A = np.zeros((len(keep) + 1,) * 2); r = np.zeros(len(keep) + 1); FC = []; YC = []
    for X, Y, Cn in D:
        H = elu(X.astype(np.float64) @ layers[0][0].T.astype(np.float64) + layers[0][1])
        F = np.hstack([H[:, keep], np.ones((len(H), 1))]); A += F.T @ F; r += F.T @ Y
        ids, inv = np.unique(Cn[:, 0], return_inverse=True)
        for i in range(len(ids)):
            m = inv == i
            if m.sum() < 400 or Cn[m, 2].any(): continue
            m = m & (Cn[:, 1] >= 200)                                # last 5 gait periods
            FC.append(F[m].mean(0)); YC.append(Y[m].mean())
    FC = np.array(FC); YC = np.array(YC); te = np.arange(len(YC)) % 5 == 0
    def solve(m):
        return np.linalg.solve(A + wc * 200 * FC[m].T @ FC[m] + lam * np.eye(len(A)), r + wc * 200 * FC[m].T @ YC[m])
    e = FC[te] @ solve(~te) - YC[te]
    stats = dict(conditions=int(len(YC)), heldout_condition_mean_error_rms_m_s=float(np.sqrt((e ** 2).mean())),
                 heldout_condition_mean_error_max_m_s=float(np.abs(e).max()))
    wf = solve(np.ones(len(YC), bool)); w = np.zeros(512); w[keep] = wf[:-1]
    return (w, float(wf[-1])), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    a = ap.parse_args()
    src = os.path.abspath(a.src); out = os.path.abspath(a.out); deploy = src + "/params/deploy.yaml"
    os.makedirs(out + "/exported", exist_ok=True); os.makedirs(out + "/params", exist_ok=True)

    print("1) sampling observations with the source policy ...")
    BP.SRC = src
    with Pool(a.workers) as p: O = np.concatenate(p.map(BP._episode, range(56)))
    L0 = load_onnx_layers(src + "/exported/policy.onnx")
    Lc = [(W.copy(), b.copy()) for W, b in L0]
    regs = ((C.R1, C.P["m25"]), (C.R2, C.P["m26"]), (C.R3, C.P["m21"]))
    for j, m in regs:
        cols = C.la_cols(j); Lc[0][1][:] += Lc[0][0][:, cols].sum(1) * m; Lc[0][0][:, cols] = 0
    rows = [j for j, _ in regs]; Lc[3][0][rows] = 0; Lc[3][1][rows] = 0
    print("2) freeing units", C.N_UNITS)
    Lp, freed = free_units(Lc, O[::2], C.N_UNITS, ridge=1e-4)
    err = np.abs(NumpyPolicy(Lp)(O[1::2]) - NumpyPolicy(Lc)(O[1::2]))
    prune = dict(max=float(err.max()), rms=float(np.sqrt((err ** 2).mean())), p999=float(np.quantile(err, 0.999)))
    print("   action error on held-out samples:", prune)

    print("3) fitting the lateral-velocity readout in closed loop")
    LH = C.build(Lp, freed)
    D = []; est = None; rounds = []
    for rnd, ex in enumerate((1.0, 0.6, 0.3, 0.3, 0.2)):
        D.append(collect(LH, est, 90, 1000 * rnd, ex, deploy, a.workers))
        est, st = fit(LH, D, freed[0]); rounds.append(st); print("   round", rnd, st)

    print("4) writing the circuit, checks")
    Lnew = C.build(Lp, freed, lateral=dict(w=est[0], b=est[1]))
    pol = NumpyPolicy(Lnew)
    rs = np.random.RandomState(0)
    Xall = np.concatenate([O] + [d[0] for d in D[1:]])
    X = Xall[rs.choice(len(Xall), 20000, replace=False)].copy()
    for j, amp in ((C.R1, 0.6), (C.R2, 0.5), (C.R3, 0.5)):
        X[:, C.la_cols(j)] = rs.uniform(-amp, amp, (len(X), 1)) + rs.normal(0, 0.02, (len(X), 5))
    X[:len(X) // 2, 44] = 0; X[:len(X) // 4, 43] = 0
    Y, hs = pol(X, return_hidden=True)
    vh = hs[0].astype(np.float64) @ est[0] + est[1]
    ref = np.array([C.reference(x, vhat=v) for x, v in zip(X, vh)]); e = Y[:, [C.R1, C.R2, C.R3]] - ref
    z = 6 * (np.abs(X[:, 458]) + np.abs(X[:, 459])) - 4.5 - 120 * np.abs(X[:, 44]); zl = z - 120 * np.abs(X[:, 43])
    X2 = X.copy()
    for k in range(5):
        X2[:, C.WZ_COLS[k]] += C.P["Kth"] / C.P["s1"] * X[:, C.la_cols(C.R1)[k]] + C.P["KI"] / C.P["s2"] * X[:, C.la_cols(C.R2)[k]]
        X2[:, C.VY_COLS[k]] += -C.P["Kp"] / C.P["s3"] * X[:, C.la_cols(C.R3)[k]]
    d = np.abs(np.delete(Y, rows, 1) - np.delete(NumpyPolicy(Lp)(X2), rows, 1))
    checks = dict(heading_registers_max_err_active=float(np.abs(e[z >= 1, :2]).max()), heading_registers_max_err_inactive=float(np.abs(e[z <= -0.1, :2]).max()),
                  lateral_register_max_err_active=float(np.abs(e[zl >= 1, 2]).max()), lateral_register_max_err_inactive=float(np.abs(e[zl <= -0.1, 2]).max()),
                  samples_lateral_active=int((zl >= 1).sum()), other_actions_max_diff_vs_folded_commands=float(d.max()))
    print("   ", checks)
    assert checks["heading_registers_max_err_active"] < 1e-3 and checks["lateral_register_max_err_active"] < 1e-3
    assert checks["heading_registers_max_err_inactive"] < 1e-5 and checks["lateral_register_max_err_inactive"] < 1e-5 and d.max() < 1e-4

    print("5) writing files")
    m = onnx.load(src + "/exported/policy.onnx")
    names = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]
    arrs = dict(zip(names, [x for W, b in Lnew for x in (W, b)])); orig = dict(zip(names, [x for W, b in L0 for x in (W, b)]))
    for t in m.graph.initializer:
        if t.name in arrs: t.CopyFrom(numpy_helper.from_array(arrs[t.name], t.name))
    onnx.checker.check_model(m); onnx.save(m, out + "/exported/policy.onnx")
    sess = ort.InferenceSession(out + "/exported/policy.onnx", providers=["CPUExecutionProvider"])
    yo = np.concatenate([sess.run(None, {"obs": x[None]})[0] for x in X[:256]])
    checks["max_onnx_numpy_error"] = float(np.abs(yo - Y[:256]).max())
    np.savez(out + "/weights.npz", **arrs); np.savez(out + "/check_io.npz", obs=X[:256], actions=yo)
    shutil.copy(deploy, out + "/params/deploy.yaml")
    json.dump(dict(source_policy=os.path.relpath(src, out), source_onnx_sha256=BP.sha(src + "/exported/policy.onnx"),
                   onnx_sha256=BP.sha(out + "/exported/policy.onnx"),
                   method="unit freeing + hand-written heading/lateral hold circuit; lateral-velocity readout = ridge regression fitted in the Python MuJoCo replica",
                   parameters=C.P, freed_units=dict(layer1=freed[0], layer2=freed[1], layer3=freed[2]),
                   register_actions={"25": "left_wrist_yaw_joint = s1 * heading error [rad]", "26": "right_wrist_yaw_joint = s2 * integral of heading error [rad s]",
                                     "21": "left_wrist_roll_joint = s3 * estimated lateral displacement [m]"},
                   changed_entries={n: int((arrs[n] != orig[n]).sum()) for n in names}, freeing_action_error=prune,
                   readout_fit_rounds=rounds, readout_bias=est[1], readout_weight_norm=float(np.linalg.norm(est[0])), checks=checks),
              open(out + "/weight_edit.json", "w"), indent=2)
    print("   onnx vs numpy max error", checks["max_onnx_numpy_error"]); print("done:", out)


if __name__ == "__main__":
    main()
