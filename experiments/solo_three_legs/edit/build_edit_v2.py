#!/usr/bin/env python3
"""Build the v2 three-legs policy (v1 + heading hold in the FL action channels) from the ORIGINAL policy.

    python edit/build_edit_v2.py policies/original policies/edited policies/edited_v2

policies/edited (v1) is read only for its design data (edit_data.npz: N, A, bell coefficients, fault-signal
weights) and its ONNX (used to roll out the prototype); v2 is rebuilt from the original weights.

Steps
  1. v1 rollouts on four legs and with the front-left leg locked -> predictor of last_action[FL] from the
     other 44 inputs (ridge, fitted), FL-command covariance -> basis (e_v, e_1, e_2) and mean m_v;
  2. first layer: v1 edits (E1 yaw-rate loop, E2 rank-1 gated shift), re-wiring of the original units'
     last_action[FL] reading to the predictor, yaw injection from the FL channels;
  3. observations of the v2 prototype (analysis/proto_edit_v2.py) on four and three legs; free 62/15/14
     hidden units (least-squares fold);
  4. v1 circuit (edit/three_leg_circuit.py) and v2 heading circuit (edit/heading_circuit.py) into the
     freed units; ridge readout of the FL command from layer 2 (fitted) replaces the FL output rows;
  5. checks against the prototype and onnxruntime; write policy.onnx, config.yaml, weight_edit.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
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
from proto_edit import DESIGN, bell_params  # noqa: E402
from proto_edit_v2 import ProtoEditV2  # noqa: E402
from free_units import free_units  # noqa: E402
import three_leg_circuit as TC  # noqa: E402
import heading_circuit as HC  # noqa: E402

PRED_COLS = [i for i in range(47) if i not in (33, 34, 35)]
V1_PATH = None
_CONSTS = None


def _v1_rollout(args):
    """12 s of v1 with random commands; four legs (pushes) or front-left leg locked."""
    seed, mode = args
    v1 = NumpyPolicy(load_onnx_layers(os.path.join(V1_PATH, "policy.onnx")))
    rs = np.random.RandomState(seed)
    faulty = mode == "three"
    sim = SoloSim(v1, os.path.join(V1_PATH, "config.yaml"), fault="locked" if faulty else None, seed=seed,
                  friction=rs.uniform(0.75, 1.0) if faulty else rs.uniform(0.6, 1.2))
    O = []
    for k in range(600):
        if k % 150 == 0:
            if faulty:
                c = np.array([rs.uniform(0.0, 0.5), 0.0, rs.uniform(-0.5, 0.5)])
                if rs.rand() < 0.3: c[2] = 0
                if rs.rand() < 0.15: c[:] = 0
            else:
                c = np.array([rs.uniform(-0.3, 0.6), rs.uniform(-0.3, 0.3), rs.uniform(-0.8, 0.8)])
                if rs.rand() < 0.2: c[:] = 0
                if np.linalg.norm(c) < 0.1: c[:] = 0
            sim.cmd = c
        if not faulty and k % 200 == 100 and rs.rand() < 0.5:
            sim.d.qvel[:2] += rs.uniform(-0.3, 0.3, 2)
        o, a = sim.policy_step()
        O.append(o)
    return np.array(O)


def _proto_rollout(args):
    """v2 prototype rollouts (heading hold active) for freeing / checks."""
    seed, mode = args
    v1 = NumpyPolicy(load_onnx_layers(os.path.join(V1_PATH, "policy.onnx")))
    pol = ProtoEditV2(v1, _CONSTS, p=_CONSTS["p"])
    rs = np.random.RandomState(seed)
    faulty = mode == "three"
    sim = SoloSim(pol, os.path.join(V1_PATH, "config.yaml"), fault="locked" if faulty else None, seed=seed,
                  friction=rs.uniform(0.75, 1.05) if faulty else rs.uniform(0.6, 1.2))
    O = []
    for k in range(600):
        if k % 150 == 0:
            c = np.array([rs.uniform(0.0, 0.5), 0.0, 0.0]) if rs.rand() < 0.6 else \
                np.array([rs.uniform(-0.3, 0.5), rs.uniform(-0.2, 0.2), rs.uniform(-0.6, 0.6)])
            if rs.rand() < 0.1: c[:] = 0
            if np.linalg.norm(c) < 0.1: c[:] = 0
            sim.cmd = c
        if not faulty and k % 200 == 100 and rs.rand() < 0.5:
            sim.d.qvel[:2] += rs.uniform(-0.3, 0.3, 2)
        o, a = sim.policy_step()
        O.append(o)
        if sim.fallen():
            break
    return np.array(O)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ridge(X, Y, lam):
    X1 = np.c_[X, np.ones(len(X))]
    return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ Y)


def main():
    global V1_PATH, _CONSTS
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("v1"); ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--th_b", type=float, default=None, help="override the heading offset (crab compensation) [rad]")
    ap.add_argument("--K_I", type=float, default=None, help="override K_I (L_I is then set so that K_th L_th + K_I L_I = 1.11 rad/s)")
    a = ap.parse_args()
    if a.th_b is not None:
        HC.P2["th_b"] = a.th_b
    if a.K_I is not None:
        HC.P2["K_I"] = a.K_I
        HC.P2["L_I"] = min(HC.P2["L_I"], 1.11 - HC.P2["K_th"] * HC.P2["L_th"]) if HC.P2["K_I"] <= 0 else (1.11 - HC.P2["K_th"] * HC.P2["L_th"]) / HC.P2["K_I"]
    SRC = os.path.abspath(a.src); V1_PATH = os.path.abspath(a.v1); out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    L0 = [(W.astype(np.float64), b.astype(np.float64)) for W, b in load_onnx_layers(SRC + "/policy.onnx")]
    ed = np.load(V1_PATH + "/edit_data.npz")
    A, bellC, w = ed["A"], ed["bellC"], ed["w"]
    v1 = NumpyPolicy(load_onnx_layers(V1_PATH + "/policy.onnx"))

    print("1) v1 rollouts -> predictor of last_action[FL], FL command basis")
    with Pool(a.workers) as p:
        H = np.concatenate(p.map(_v1_rollout, [(s, "four") for s in range(a.episodes)]))
        T = np.concatenate(p.map(_v1_rollout, [(500 + s, "three") for s in range(a.episodes)]))
    Xp = np.r_[H, T][:, PRED_COLS]; Yp = np.r_[H, T][:, 33:36]
    M = ridge(Xp, Yp, 1e-3)
    eH = np.c_[H[:, PRED_COLS], np.ones(len(H))] @ M - H[:, 33:36]
    eT = np.c_[T[:, PRED_COLS], np.ones(len(T))] @ M - T[:, 33:36]
    pred_err = dict(four_legs_rms=np.sqrt((eH ** 2).mean(0)).tolist(), three_legs_rms=np.sqrt((eT ** 2).mean(0)).tolist())
    AH = v1(H)[:, 0:3]
    evals, evecs = np.linalg.eigh(np.cov(AH.T))
    basis = evecs[:, [0, 1, 2]]
    m_v = float((AH @ basis[:, 0]).mean())
    inj_noise = float((AH @ basis[:, 0]).std() / HC.P2["s_y"])
    print("   predictor rms four legs", np.round(pred_err["four_legs_rms"], 3), "three legs", np.round(pred_err["three_legs_rms"], 3))
    print("   FL command variances", np.round(evals, 3), " e_v", np.round(basis[:, 0], 3), " m_v %.3f" % m_v,
          " injection noise on four legs %.4f rad/s" % inj_noise)
    _CONSTS = dict(w=w, pred=M, pred_cols=PRED_COLS, basis=basis, m_v=m_v)
    V2P = dict(K_th=HC.P2["K_th"], K_I=HC.P2["K_I"], L_th=HC.P2["L_th"], L_I=HC.P2["L_I"], s1=HC.P2["s1"], s2=HC.P2["s2"],
               s_y=HC.P2["s_y"], th_b=HC.P2["th_b"])
    _CONSTS["p"] = V2P

    print("2) first-layer edits")
    trim_scaled = np.asarray(DESIGN["cmd_trim"]) * np.array([2.0, 2.0, 0.25])
    L1, c = TC.apply_input_edits(L0, A, trim_scaled, DESIGN["yaw_K"], w)
    L1, Wla = HC.rewire_fl_last_action(L1, PRED_COLS, M, basis[:, 0], m_v, HC.P2["s_y"])
    L1 = HC.add_injection(L1, L0[0][0][:, 8], basis[:, 0], m_v, HC.P2["s_y"])

    print("3) prototype rollouts and unit freeing")
    with Pool(a.workers) as p:
        H2 = np.concatenate(p.map(_proto_rollout, [(1000 + s, "four") for s in range(a.episodes)]))
        T2 = np.concatenate(p.map(_proto_rollout, [(2000 + s, "three") for s in range(a.episodes)]))
    counts = tuple(x + y for x, y in zip(TC.N_UNITS, HC.N_UNITS))
    Xall = np.concatenate([H2, T2])
    rs = np.random.RandomState(0)
    idx = rs.permutation(len(Xall)); fit_idx, test_idx = idx[: len(idx) * 2 // 3], idx[len(idx) * 2 // 3:]
    Lf, freed, _ = free_units(L1, Xall[fit_idx], counts)
    e = np.abs(NumpyPolicy(Lf)(Xall[test_idx]) - NumpyPolicy(L1)(Xall[test_idx]))
    free_err = dict(rms=float(np.sqrt((e ** 2).mean())), max=float(e.max()), p999=float(np.quantile(e, 0.999)))
    print("   counts", counts, " fold action error", {k: round(v, 4) for k, v in free_err.items()})

    print("4) circuits")
    nodes, alpha, beta = bell_params(DESIGN)
    bell_dirs = beta * np.stack([np.sin(2 * np.pi * nodes), np.cos(2 * np.pi * nodes)], 1)
    bell_bias = -beta * np.cos(alpha) * np.ones(len(nodes))
    fv1 = [f[:n] for f, n in zip(freed, TC.N_UNITS)]
    fv2 = [f[n:] for f, n in zip(freed, TC.N_UNITS)]
    Lc, used1, n1 = TC.write_circuit(Lf, fv1, A, w, bell_dirs, bell_bias, bellC)
    assert n1 == TC.N_UNITS
    # FL command readout from the original layer-2 units, fitted on four-leg observations
    orig_l2 = [u for u in range(Lc[1][0].shape[0]) if u not in set(freed[1])]
    yH, hs = NumpyPolicy(Lc)(H2, return_hidden=True)
    Wr = ridge(hs[1][:, orig_l2], yH[:, 0:3], 1e-2 * len(H2) / 1000)
    ro_err = np.c_[hs[1][:, orig_l2], np.ones(len(H2))] @ Wr - yH[:, 0:3]
    readout_err = dict(rms=np.sqrt((ro_err ** 2).mean(0)).tolist(), max=np.abs(ro_err).max(0).tolist())
    print("   FL readout error on four legs (action units):", {k: np.round(v, 3).tolist() for k, v in readout_err.items()})
    Lnew, used2, n2 = HC.write_heading(Lc, fv2, used1, basis, m_v, (Wr[:-1], Wr[-1]), orig_l2)
    assert n2 == HC.N_UNITS, n2
    pol = NumpyPolicy(Lnew)

    print("5) checks")
    proto = ProtoEditV2(v1, _CONSTS, p=_CONSTS["p"])
    g3 = T2[:, TC.FL_COLS] @ w
    F3 = np.clip((g3 - 0.55) / 0.3, 0, 1)
    th2, I2, y2, act = HC.reference_registers(T2, F3, p=HC.P2, basis=basis, m_v=m_v)
    Y3 = pol(T2)
    reg_ref = np.outer(th2, basis[:, 1]) + np.outer(I2, basis[:, 2]) + np.outer(HC.P2["s_y"] * y2 + m_v, basis[:, 0])
    d_reg = np.abs(Y3[:, 0:3] - reg_ref)
    on = act > 0.999
    checks = dict(
        fl_registers_max_err_act_on=float(d_reg[on].max()) if on.any() else None,
        fl_registers_max_err_all_three_leg=float(d_reg.max()),
        theta_register_max_err=float(np.abs(Y3[:, 0:3] @ basis[:, 1] - th2).max()),
        I_register_max_err=float(np.abs(Y3[:, 0:3] @ basis[:, 2] - I2).max()),
        three_legs_other_actions_vs_prototype_max=float(np.abs(Y3[:, 3:] - proto(T2)[:, 3:]).max()),
        three_legs_other_actions_vs_prototype_rms=float(np.sqrt(((Y3[:, 3:] - proto(T2)[:, 3:]) ** 2).mean())),
        four_legs_vs_prototype_max=float(np.abs(pol(H2) - proto(H2)).max()),
        four_legs_vs_prototype_rms=float(np.sqrt(((pol(H2) - proto(H2)) ** 2).mean())),
        four_legs_vs_v1_rms=float(np.sqrt(((pol(H2) - v1(H2)) ** 2).mean())),
        four_legs_vs_v1_max=float(np.abs(pol(H2) - v1(H2)).max()),
        active_fraction_three_leg_samples=float(on.mean()))
    for k, v in checks.items():
        print(f"   {k}: {v}")

    names = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]
    arrs = dict(zip(names, [np.asarray(x, np.float32) for W, b in Lnew for x in (W, b)]))
    orig = dict(zip(names, [np.asarray(x, np.float32) for W, b in L0 for x in (W, b)]))
    onnx_path = out + "/policy.onnx"
    write_onnx_mlp([(arrs[f"mlp.{i}.weight"], arrs[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)], onnx_path,
                   doc="SOLO-12 policy edited by hand: three legs (v1) + heading hold in the dead leg's action channels (v2); no training")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    Xc = np.concatenate([H2[:128], T2[:128]]).astype(np.float32)
    yo = sess.run(None, {"obs": Xc})[0]
    checks["onnx_vs_numpy_max"] = float(np.abs(yo - pol(Xc)).max())
    np.savez(out + "/check_io.npz", obs=Xc, actions=yo)
    shutil.copy(SRC + "/config.yaml", out + "/config.yaml")
    np.savez(out + "/edit_data.npz", N=ed["N"], A=A, bellC=bellC, w=w, pred=M, basis=basis, m_v=m_v, readout=Wr)
    changed = {n: int((arrs[n] != orig[n]).sum()) for n in names}
    json.dump(dict(
        source_policy=os.path.relpath(SRC, out), source_onnx_sha256=sha(SRC + "/policy.onnx"), onnx_sha256=sha(onnx_path),
        v1_policy=os.path.relpath(V1_PATH, out), v1_onnx_sha256=sha(V1_PATH + "/policy.onnx"),
        method="hand edit, no training: v1 (three-leg gait) + v2 heading hold with registers in the dead leg's action channels",
        v1_design={k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in DESIGN.items()},
        v2_parameters=HC.P2, register_basis_columns_e_v_e1_e2=basis.tolist(), m_v=m_v,
        injection_noise_four_legs_rad_s=inj_noise, predictor_error=pred_err, fl_readout_error=readout_err,
        freed_units=dict(layer1=freed[0], layer2=freed[1], layer3=freed[2]),
        circuit_units=dict(v1=used1, v2=used2), freeing_action_error=free_err,
        fitted_to_data=["v1: N_FR, fault-signal weights w, least-squares fold (see policies/edited/weight_edit.json)",
                        "predictor of last_action[FL] from the other 44 inputs (ridge on v1 rollouts, four and three legs)",
                        "FL command covariance (register basis) and mean m_v (v1 rollouts on four legs)",
                        "ridge readout of the FL command from layer 2 (v2 network, four legs)",
                        "least-squares fold (62/15/14 units) on v2-prototype observations"],
        weight_groups_changed={
            "mlp.0.weight[:, 2], [:, 8]": "E1 yaw-rate loop",
            "mlp.0.weight[:, 9:12], [:, 21:24]": "E2 rank-1 gated shift (v1)",
            "mlp.0.weight[:, all except 33:36] (original units)": "re-wired reading of last_action[FL] = W0[:, FL] M^T",
            "mlp.0.weight[:, 33:36] (original units)": "only the yaw injection (0.25/s_y) W0[:, 8] e_v^T",
            "freed units": "v1 circuit (46/6/5) + v2 heading circuit and FL readout (16/9/9)",
            "mlp.6 rows 0..2": "FL outputs = gated readout + gated registers",
            "all next-layer weights of freed units": "least-squares fold"},
        changed_entries=changed, checks=checks, samples=dict(v1_four=int(len(H)), v1_three=int(len(T)),
                                                             proto_four=int(len(H2)), proto_three=int(len(T2))),
        build_time_s=round(time.time() - t0, 1)), open(out + "/weight_edit.json", "w"), indent=2)
    print("   onnx vs numpy", checks["onnx_vs_numpy_max"])
    print("done:", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
