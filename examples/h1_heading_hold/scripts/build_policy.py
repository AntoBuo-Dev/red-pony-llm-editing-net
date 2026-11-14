#!/usr/bin/env python3
"""Build the heading-hold policy by editing the weights of an exported policy (no training).

    python evaluation/heading_hold/build_policy.py <source policy dir> <output policy dir>

Steps: (1) roll out the source policy in the Python sim2sim replica to sample observations;
(2) free the most redundant hidden units (least-squares fold into the other units of the layer);
(3) write the heading-hold circuit into them (heading_hold_circuit.py); (4) check it against an
exact reference; (5) write exported/policy.onnx, weights.npz and weight_edit.json.
The TorchScript export is a separate step (needs torch): export_torchscript.py.
"""
import argparse, hashlib, json, os, shutil, sys
from multiprocessing import Pool
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_harness import Sim, NumpyPolicy, load_onnx_layers
from free_units import free_units
import heading_hold_circuit as C

SRC = None


def _episode(seed):
    """40 s of random commands, heights and pushes with the source policy."""
    pol = NumpyPolicy(load_onnx_layers(SRC + "/exported/policy.onnx"))
    sim = Sim(pol, SRC + "/params/deploy.yaml")
    rs = np.random.RandomState(seed)
    O = []; h = 0.98
    for k in range(int(40 / sim.step_dt)):
        if k % 150 == 0:
            mode = rs.randint(6)
            c = np.array([rs.uniform(-0.3, 1.0), rs.uniform(-0.3, 0.3), rs.uniform(-0.5, 0.5)])
            if mode == 0: c[:] = 0
            if mode == 1: c[1:] = 0
            if mode == 2: c[2] = 0
            if mode == 3: c[[0, 2]] = 0
            sim.cmd = c
            if rs.rand() < 0.4: h = rs.choice([0.98, 0.98, 0.85, 0.74, 0.71, 0.66, 0.60])
        if k % 200 == 100 and rs.rand() < 0.5:
            sim.d.qvel[:2] += rs.uniform(-0.4, 0.4, 2); sim.d.qvel[5] += rs.uniform(-0.5, 0.5)
        sim.set_height(h)
        o, a = sim.policy_step()
        x, y, z, yaw, up = sim.base()
        if up < 0.5 or z < 0.35: break
        O.append(o)
    return np.array(O)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=56); ap.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    a = ap.parse_args()
    SRC = os.path.abspath(a.src); out = os.path.abspath(a.out)
    os.makedirs(out + "/exported", exist_ok=True); os.makedirs(out + "/params", exist_ok=True)

    print("1) sampling observations with the source policy ...")
    with Pool(a.workers) as p: O = np.concatenate(p.map(_episode, range(a.episodes)))
    print("   ", O.shape)

    L0 = load_onnx_layers(SRC + "/exported/policy.onnx")
    # what the circuit overwrites must not count when ranking the units: the original use of
    # last_action[25/26] (constant-folded) and the output rows 25/26
    Lc = [(W.copy(), b.copy()) for W, b in L0]
    for j, m in ((C.R1, C.P["m25"]), (C.R2, C.P["m26"])):
        cols = C.la_cols(j); Lc[0][1][:] += Lc[0][0][:, cols].sum(1) * m; Lc[0][0][:, cols] = 0
    Lc[3][0][[C.R1, C.R2]] = 0; Lc[3][1][[C.R1, C.R2]] = 0
    print("2) freeing units", C.N_UNITS)
    Lp, freed = free_units(Lc, O[::2], C.N_UNITS, ridge=1e-4)
    A0 = NumpyPolicy(Lc)(O[1::2]); err = np.abs(NumpyPolicy(Lp)(O[1::2]) - A0)
    prune = dict(max=float(err.max()), rms=float(np.sqrt((err ** 2).mean())), p999=float(np.quantile(err, 0.999)))
    print("   action error on held-out samples:", prune)

    print("3) writing the circuit")
    Lnew, used = C.build(Lp, freed)
    pol = NumpyPolicy(Lnew)

    print("4) checks")
    rs = np.random.RandomState(0)
    X = O[rs.choice(len(O), 20000, replace=False)].copy()
    X[:, C.la_cols(C.R1)] = rs.uniform(-0.6, 0.6, (len(X), 1)) + rs.normal(0, 0.02, (len(X), 5))
    X[:, C.la_cols(C.R2)] = rs.uniform(-0.5, 0.5, (len(X), 1)) + rs.normal(0, 0.02, (len(X), 5))
    X[:len(X) // 2, 44] = 0
    Y = pol(X); ref = np.array([C.reference(x) for x in X]); e = Y[:, [C.R1, C.R2]] - ref
    z = 6 * (np.abs(X[:, 458]) + np.abs(X[:, 459])) - 4.5 - 120 * np.abs(X[:, 44])
    on, off = z >= 1, z <= -0.1      # the smooth |wz| shifts the activation band by up to 0.06 in z
    X2 = X.copy()
    for k in range(5):
        X2[:, C.WZ_COLS[k]] += C.P["Kth"] / C.P["s1"] * X[:, C.la_cols(C.R1)[k]] + C.P["KI"] / C.P["s2"] * X[:, C.la_cols(C.R2)[k]]
    d = np.abs(np.delete(Y, [C.R1, C.R2], 1) - np.delete(NumpyPolicy(Lp)(X2), [C.R1, C.R2], 1))
    checks = dict(register_max_err_active=float(np.abs(e[on]).max()), register_max_err_inactive=float(np.abs(e[off]).max()),
                  yaw_rate_bias_rad_s=float(e[on, 0].mean() / (C.P["s1"] * C.P["dt"])),
                  other_actions_max_diff_vs_folded_command=float(d.max()))
    print("   ", checks)
    assert checks["register_max_err_active"] < 1e-3 and checks["register_max_err_inactive"] < 1e-5 and d.max() < 1e-4

    print("5) writing files")
    m = onnx.load(SRC + "/exported/policy.onnx")
    names = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]
    arrs = dict(zip(names, [x for W, b in Lnew for x in (W, b)]))
    for t in m.graph.initializer:
        if t.name in arrs: t.CopyFrom(numpy_helper.from_array(arrs[t.name], t.name))
    onnx.checker.check_model(m); onnx.save(m, out + "/exported/policy.onnx")
    sess = ort.InferenceSession(out + "/exported/policy.onnx", providers=["CPUExecutionProvider"])
    yo = np.concatenate([sess.run(None, {"obs": x[None]})[0] for x in X[:256]])
    checks["max_onnx_numpy_error"] = float(np.abs(yo - Y[:256]).max())
    np.savez(out + "/weights.npz", **arrs)
    np.savez(out + "/check_io.npz", obs=X[:256], actions=yo)
    shutil.copy(SRC + "/params/deploy.yaml", out + "/params/deploy.yaml")
    changed = {n: int((arrs[n] != dict(zip(names, [x for W, b in L0 for x in (W, b)]))[n]).sum()) for n in names}
    json.dump(dict(source_policy=os.path.relpath(SRC, out), source_onnx_sha256=sha(SRC + "/exported/policy.onnx"),
                   onnx_sha256=sha(out + "/exported/policy.onnx"), method="unit freeing + hand-written heading-hold circuit; no training",
                   parameters=C.P, freed_units=dict(layer1=freed[0], layer2=freed[1], layer3=freed[2]),
                   register_actions={"25": "left_wrist_yaw_joint = s1 * heading error [rad]", "26": "right_wrist_yaw_joint = s2 * integral of heading error [rad s]"},
                   changed_entries=changed, freeing_action_error=prune, checks=checks, samples=int(len(O))),
              open(out + "/weight_edit.json", "w"), indent=2)
    print("   onnx vs numpy max error", checks["max_onnx_numpy_error"]); print("done:", out)


if __name__ == "__main__":
    main()
