#!/usr/bin/env python3
"""Export a training checkpoint to a deployable policy folder:

    python train/export_policy.py runs/solo_walk/model_final.pt policies/original

writes policy.onnx (plain Gemm/Elu MLP, see harness/onnx_mlp.py), checkpoint.pt (copy),
config.yaml (everything the harness needs), check_io.npz (obs/actions for checking other runtimes).
"""
import argparse
import hashlib
import os
import shutil
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import solo_config as C  # noqa: E402
from onnx_mlp import write_onnx_mlp, load_onnx_layers, NumpyPolicy  # noqa: E402


def deploy_config():
    return dict(
        robot="SOLO-12 (example-robot-data solo12.urdf -> assets/solo12/solo12.xml)",
        joint_names=C.JOINTS, default_q=C.DEFAULT_Q.tolist(),
        dt=C.DT, decimation=C.DECIMATION, sim_dt=C.SIM_DT,
        kp=4.0, kd=0.2, torque_limit=2.7,
        action_scale=C.ACTION_SCALE, action_clip=C.ACTION_CLIP,
        obs_layout={k: [v.start, v.stop] for k, v in C.OBS_SLICES.items()}, obs_dim=C.OBS_DIM,
        obs_scale_ang_vel=C.SCALE_ANG_VEL, obs_scale_cmd=C.SCALE_CMD.tolist(), obs_scale_dof_pos=C.SCALE_DOF_POS,
        obs_scale_dof_vel=C.SCALE_DOF_VEL,
        gait_freq=C.GAIT_FREQ, leg_phase_offset=C.LEG_PHASE_OFFSET.tolist(), stand_cmd_norm=C.STAND_CMD_NORM,
        tucked_pose=C.TUCKED_FL.tolist(),
        notes="obs = [ang_vel*0.25, projected_gravity, cmd*(2,2,0.25), q-q0, dq*0.05, last_action, sin/cos(2 pi phase) "
              "(zero when |cmd|<0.1)]; q_target = q0 + 0.3*action; joint PD in MuJoCo position actuators")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["model"]
    layers = [(sd[f"actor.{i}.weight"].numpy(), sd[f"actor.{i}.bias"].numpy()) for i in (0, 2, 4, 6)]
    onnx_path = os.path.join(a.out, "policy.onnx")
    write_onnx_mlp(layers, onnx_path, doc=f"SOLO-12 walking policy exported from {os.path.basename(a.ckpt)} (iter {ck.get('iter')})")
    shutil.copy(a.ckpt, os.path.join(a.out, "checkpoint.pt"))
    cfg = deploy_config()
    cfg["source_checkpoint"] = os.path.basename(a.ckpt)
    cfg["train_iter"] = int(ck.get("iter", -1))
    yaml.safe_dump(cfg, open(os.path.join(a.out, "config.yaml"), "w"), sort_keys=False)
    # check: torch actor vs onnxruntime vs numpy
    import onnxruntime as ort
    sys.path.insert(0, HERE)
    from ppo import ActorCritic
    ac = ActorCritic(C.OBS_DIM, sd["critic.0.weight"].shape[1] - C.OBS_DIM, 12)
    ac.load_state_dict(sd)
    X = np.random.RandomState(0).normal(0, 0.5, (256, C.OBS_DIM)).astype(np.float32)
    yt = ac.actor(torch.as_tensor(X)).detach().numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    yo = sess.run(None, {"obs": X})[0]
    yn = NumpyPolicy(load_onnx_layers(onnx_path))(X)
    print("max |torch - onnx| %.2e   max |numpy - onnx| %.2e" % (np.abs(yt - yo).max(), np.abs(yn - yo).max()))
    np.savez(os.path.join(a.out, "check_io.npz"), obs=X, actions=yo)
    print("sha256", sha(onnx_path))


if __name__ == "__main__":
    main()
