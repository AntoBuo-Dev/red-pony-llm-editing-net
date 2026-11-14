#!/usr/bin/env python3
"""Side-by-side MP4 of goal-reaching runs (same start, goal and seed for every panel).

    MUJOCO_GL=egl python harness/record_goal_video.py --panel "original=policies/original" \
        --panel "edited v2=policies/edited_v2" --goal 4 0 --mode A --seed 102 --out results/videos/goal_4m.mp4

The goal is drawn as a green disc (0.3 m success radius); the run of a panel stops (frozen) once it reaches
the goal, falls, or the time limit 2 * distance / 0.2 m/s is over.
"""
import argparse
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_harness import SoloSim  # noqa: E402
from onnx_mlp import NumpyPolicy, load_onnx_layers  # noqa: E402
from goal_reaching import yaw_quat, VX, K_BEARING, WZ_MAX, RADIUS, REF_SPEED  # noqa: E402
from record_video import label_frame  # noqa: E402


def add_disc(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, 0.002, 0.0]),
                        np.array([pos[0], pos[1], 0.002]), np.eye(3).ravel(), np.array(rgba, np.float32))
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True, help="label=policy_dir[:fault]")
    ap.add_argument("--goal", nargs=2, type=float, default=[4.0, 0.0])
    ap.add_argument("--mode", default="A")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speedup", type=float, default=2.0, help="video time = sim time / speedup")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gx, gy = a.goal
    dist0 = math.hypot(gx, gy)
    bearing0 = math.atan2(gy, gx)
    t_limit = 2.0 * dist0 / REF_SPEED
    rs = np.random.RandomState(1000 + a.seed)
    fr, mo = rs.uniform(0.7, 1.1), rs.uniform(-0.1, 0.2)
    sims, labels, state = [], [], []
    for p in a.panel:
        label, rest = p.split("=", 1)
        path, fault = (rest.rsplit(":", 1) + ["locked"])[:2] if ":" in rest else (rest, "locked")
        pol = NumpyPolicy(load_onnx_layers(os.path.join(path, "policy.onnx")))
        s = SoloSim(pol, os.path.join(path, "config.yaml"), fault=None if fault == "none" else fault, seed=a.seed,
                    friction=fr, mass_offset=mo)
        s.d.qpos[3:7] = yaw_quat(bearing0)
        mujoco.mj_forward(s.m, s.d)
        sims.append(s); labels.append(label); state.append(dict(done=False, status=""))
    rend = [mujoco.Renderer(s.m, a.height, a.width) for s in sims]
    cams = []
    for s in sims:
        c = mujoco.MjvCamera(); c.distance = 2.2; c.elevation = -35; c.azimuth = -140
        cams.append(c)
    n_settle = int(1.0 / sims[0].dt)
    origin = []
    n = n_settle + int(t_limit / sims[0].dt)
    every = max(1, int(round(a.speedup / (a.fps * sims[0].dt))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
    for k in range(n):
        for i, s in enumerate(sims):
            if k == n_settle:
                origin.append((s.d.qpos[0], s.d.qpos[1]))
            st = state[i]
            if st["done"]:
                continue
            if k >= n_settle:
                x0, y0 = origin[i]
                x, y, z, yaw, up = s.base()
                px, py = x - x0, y - y0
                if a.mode == "A":
                    s.cmd = np.array([VX, 0.0, 0.0])
                else:
                    err = (math.atan2(gy - py, gx - px) - yaw + math.pi) % (2 * math.pi) - math.pi
                    s.cmd = np.array([VX, 0.0, float(np.clip(K_BEARING * err, -WZ_MAX, WZ_MAX))])
            s.policy_step()
            if s.fallen():
                st["done"], st["status"] = True, "FALLEN"
            elif k >= n_settle:
                x0, y0 = origin[i]
                if math.hypot(gx - (s.d.qpos[0] - x0), gy - (s.d.qpos[1] - y0)) <= RADIUS:
                    st["done"], st["status"] = True, "REACHED"
        if k % every == 0:
            frames = []
            for i, (s, r, c) in enumerate(zip(sims, rend, cams)):
                x0, y0 = origin[i] if origin else (s.d.qpos[0], s.d.qpos[1])
                c.lookat[:] = [(s.d.qpos[0] + x0 + gx) / 2 if False else s.d.qpos[0], s.d.qpos[1], 0.1]
                r.update_scene(s.d, c)
                add_disc(r.scene, (x0 + gx, y0 + gy) if origin else (gx, gy), RADIUS, [0.1, 0.7, 0.3, 0.6])
                img = r.render()
                t = max(0.0, (k - n_settle) * s.dt)
                dgoal = math.hypot(gx - (s.d.qpos[0] - x0), gy - (s.d.qpos[1] - y0)) if origin else dist0
                txt = f"{labels[i]}\nt = {t:4.1f} s  to goal {dgoal:.2f} m {state[i]['status']}"
                frames.append(label_frame(img, txt))
            wr.append_data(np.concatenate(frames, axis=1))
    wr.close()
    print("wrote", a.out, os.path.getsize(a.out) / 1e6, "MB", [s["status"] for s in state])


if __name__ == "__main__":
    main()
