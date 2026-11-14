#!/usr/bin/env python3
"""Build the three-legs policy by editing the weights of the original SOLO-12 policy (no training).

    python edit/build_edit.py policies/original policies/edited

Steps
  1. measure the original's phase-averaged action pattern on four legs (N, used by the FR re-timing);
  2. posture offset A (inverse kinematics of the foot shift), bell synthesis of the FR re-timing D(phase);
  3. fault signal g(x) = w . [q_FL, dq_FL]: w = least-norm-on-four-legs solution of w . x_tucked = 1
     (fitted on four-leg rollouts of the original);
  4. E1 + E2 on the first layer;
  5. sample observations (four legs: original+E1; three legs: prototype) and free 46 / 6 / 5 hidden units;
  6. write the circuit (edit/three_leg_circuit.py) into the freed units;
  7. checks against the prototype (analysis/proto_edit.py) and onnxruntime; write policy.onnx,
     config.yaml, weight_edit.json, check_io.npz.
"""
import os
# single-threaded BLAS: the least-squares fold is then bitwise reproducible (same sha256 on every run)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from onnx_mlp import load_onnx_layers, write_onnx_mlp, NumpyPolicy  # noqa: E402
from sim_harness import SoloSim  # noqa: E402
from proto_edit import (DESIGN, ProtoEdit, nominal_pattern, posture_offset, fit_bells, bell_params,  # noqa: E402
                        fr_delta)
from free_units import free_units  # noqa: E402
import three_leg_circuit as TC  # noqa: E402

SRC = None
_N = _C = None


def _episode(args):
    """12 s of random commands (changed every 3 s) with pushes; four legs (original + E1) or three legs
    (prototype). Returns the raw observations."""
    seed, mode = args
    net = NumpyPolicy(load_onnx_layers(SRC + "/policy.onnx"))
    faulty = mode == "three"
    pol = ProtoEdit(net, _N, DESIGN, faulty=faulty, bellC=_C)
    rs = np.random.RandomState(seed)
    sim = SoloSim(pol, SRC + "/config.yaml", fault="locked" if faulty else None, seed=seed,
                  friction=rs.uniform(0.6, 1.2))
    O = []
    for k in range(int(12 / sim.dt)):
        if k % 150 == 0:
            m = rs.randint(5)
            c = np.array([rs.uniform(-0.3, 0.6), rs.uniform(-0.3, 0.3), rs.uniform(-0.8, 0.8)])
            if m == 0: c[:] = 0
            if m == 1: c[1:] = 0
            if m == 2: c[1] = 0
            if np.linalg.norm(c) < 0.1: c[:] = 0
            sim.cmd = c
        if k % 200 == 100 and rs.rand() < 0.5:
            sim.d.qvel[:2] += rs.uniform(-0.3, 0.3, 2)
        o, a = sim.policy_step()
        O.append(o)
        if sim.fallen():
            break
    return np.array(O)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    global SRC, _N, _C
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    SRC = os.path.abspath(a.src); out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    L0 = load_onnx_layers(SRC + "/policy.onnx")
    net = NumpyPolicy(L0)
    cfg = SRC + "/config.yaml"

    print("1) nominal FR pattern of the original on four legs (vx = 0.3 m/s)")
    _N = nominal_pattern(net, cfg)
    print("2) posture offset and FR re-timing")
    A = posture_offset(DESIGN["foot_shift"])
    _C, bell_fit = fit_bells(_N, DESIGN)
    nodes, alpha, beta = bell_params(DESIGN)
    bell_dirs = beta * np.stack([np.sin(2 * np.pi * nodes), np.cos(2 * np.pi * nodes)], 1)
    bell_bias = -beta * np.cos(alpha) * np.ones(len(nodes))
    print("   A =", np.round(A, 3), "\n   bell fit of D:", {k: np.round(v, 3).tolist() for k, v in bell_fit.items()})

    print("3) sampling observations (four legs: original + yaw loop; three legs: prototype)")
    with Pool(a.workers) as p:
        H = p.map(_episode, [(s, "four") for s in range(a.episodes)])
        T3 = p.map(_episode, [(1000 + s, "three") for s in range(a.episodes)])
    H = np.concatenate(H); T3 = np.concatenate(T3)
    print("   samples: four legs", H.shape, " three legs", T3.shape)

    # fault signal: min E_4legs[(w.x)^2]  s.t.  w . x_tucked = 1
    x_t = np.r_[np.array(DESIGN_TUCK) - np.array([0.1, 0.8, -1.6]), np.zeros(3)]
    X = H[:, TC.FL_COLS].astype(np.float64)
    Cxx = X.T @ X / len(X) + 1e-6 * np.eye(6)
    w = np.linalg.solve(Cxx, x_t); w /= w @ x_t
    gH = X @ w; g3 = T3[:, TC.FL_COLS] @ w
    gate = dict(w=w.tolist(), four_legs_rms=float(np.sqrt((gH ** 2).mean())), four_legs_max=float(np.abs(gH).max()),
                three_legs_min=float(g3.min()), three_legs_max=float(g3.max()))
    print("   fault signal g:", {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in gate.items()})
    assert gate["four_legs_max"] < 0.55 - 0.05 and gate["three_legs_min"] > 0.85 + 0.05

    print("4) E1 + E2 on the first layer")
    trim_scaled = np.asarray(DESIGN["cmd_trim"]) * np.array([2.0, 2.0, 0.25])
    L1, c = TC.apply_input_edits(L0, A, trim_scaled, DESIGN["yaw_K"], w)

    print("5) freeing units", TC.N_UNITS)
    rs = np.random.RandomState(0)
    Xall = np.concatenate([H, T3])
    idx = rs.permutation(len(Xall)); fit_idx, test_idx = idx[: len(idx) * 2 // 3], idx[len(idx) * 2 // 3:]
    Lf, freed, resid = free_units(L1, Xall[fit_idx], TC.N_UNITS)
    Y1 = NumpyPolicy(L1)(Xall[test_idx]); Yf = NumpyPolicy(Lf)(Xall[test_idx])
    e = np.abs(Yf - Y1)
    free_err = dict(rms=float(np.sqrt((e ** 2).mean())), max=float(e.max()), p999=float(np.quantile(e, 0.999)))
    print("   action error of the fold on held-out samples:", free_err)

    print("6) writing the circuit")
    Lnew, used, n_used = TC.write_circuit(Lf, freed, A, w, bell_dirs, bell_bias, _C)
    assert n_used == TC.N_UNITS, n_used
    pol = NumpyPolicy(Lnew)

    print("7) checks")
    proto3 = ProtoEdit(net, _N, DESIGN, faulty=True, bellC=_C)
    proto4 = ProtoEdit(net, _N, DESIGN, faulty=False, bellC=_C)
    d3 = np.abs(pol(T3) - proto3(T3))
    d4 = np.abs(pol(H) - proto4(H))
    d4o = np.abs(pol(H) - net(H))
    checks = dict(three_legs_vs_prototype_max=float(d3.max()), three_legs_vs_prototype_rms=float(np.sqrt((d3 ** 2).mean())),
                  four_legs_vs_original_plus_yawloop_max=float(d4.max()), four_legs_vs_original_plus_yawloop_rms=float(np.sqrt((d4 ** 2).mean())),
                  four_legs_vs_original_max=float(d4o.max()), four_legs_vs_original_rms=float(np.sqrt((d4o ** 2).mean())))
    for k, v in checks.items():
        print(f"   {k}: {v:.4f}")

    names = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]
    arrs = dict(zip(names, [np.asarray(x, np.float32) for W, b in Lnew for x in (W, b)]))
    orig = dict(zip(names, [np.asarray(x, np.float32) for W, b in L0 for x in (W, b)]))
    onnx_path = out + "/policy.onnx"
    write_onnx_mlp([(arrs[f"mlp.{i}.weight"], arrs[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)], onnx_path,
                   doc="SOLO-12 policy edited by hand for walking with the front-left leg locked (no training)")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    Xc = np.concatenate([H[:128], T3[:128]]).astype(np.float32)
    yo = sess.run(None, {"obs": Xc})[0]
    checks["onnx_vs_numpy_max"] = float(np.abs(yo - pol(Xc)).max())
    np.savez(out + "/check_io.npz", obs=Xc, actions=yo)
    shutil.copy(cfg, out + "/config.yaml")
    np.savez(out + "/edit_data.npz", N=_N, A=A, bellC=_C, w=w, c=c)
    changed = {n: int((arrs[n] != orig[n]).sum()) for n in names}
    json.dump(dict(
        source_policy=os.path.relpath(SRC, out), source_onnx_sha256=sha(SRC + "/policy.onnx"), onnx_sha256=sha(onnx_path),
        method="hand edit, no training: unit freeing (least-squares fold) + hand-written circuits; see README",
        design={k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in DESIGN.items()},
        posture_offset_A_action_units=A.tolist(),
        fault_signal=gate, fault_flag_threshold=[0.55, 0.85],
        yaw_loop_K=DESIGN["yaw_K"], command_trim=list(DESIGN["cmd_trim"]),
        bell_synthesis=dict(n=len(nodes), beta=float(beta), alpha=float(alpha), fit_error_action_units=bell_fit),
        freed_units=dict(layer1=freed[0], layer2=freed[1], layer3=freed[2]),
        circuit_units=used, freeing_action_error=free_err,
        fitted_to_data=["N_FR: phase-averaged FR actions of the original on four legs (3 rollouts, vx 0.3 m/s)",
                        "w: fault signal weights, least-norm on four-leg observations of the original",
                        "unit freeing: least-squares fold on four-leg and three-leg observations"],
        weight_groups_changed={
            "mlp.0.weight[:, 2], [:, 8]": "E1 yaw-rate loop (all units)",
            "mlp.0.weight[:, 9:12], [:, 21:24]": "E2 rank-1 update c w^T (posture hiding + command trim, gated by g)",
            "mlp.0 rows of freed layer-1 units": "fault flag (2), |sin|,|cos| (4), 40 bells",
            "mlp.2 / mlp.4 rows+cols of freed units": "gate G, FR signal pass-through, gated FR signal, flag pass-through",
            "mlp.6 columns of the 5 layer-3 circuit units, mlp.6 bias rows 3..11": "FR re-timing and posture offset into the outputs",
            "all next-layer weights/biases of freed units": "least-squares fold"},
        changed_entries=changed, checks=checks, samples=dict(four_legs=int(len(H)), three_legs=int(len(T3))),
        build_time_s=round(time.time() - t0, 1)), open(out + "/weight_edit.json", "w"), indent=2)
    print("   onnx vs numpy max error", checks["onnx_vs_numpy_max"])
    print("done:", out, f"({time.time() - t0:.0f} s)")


DESIGN_TUCK = [0.0, 1.6, -2.9]

if __name__ == "__main__":
    main()
