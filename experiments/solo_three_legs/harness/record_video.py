#!/usr/bin/env python3
"""Offscreen MuJoCo videos (MP4), one or several policies side by side.

    MUJOCO_GL=egl python harness/record_video.py --out results/videos/x.mp4 \
        --panel "original, 3 legs=policies/original:locked" --panel "edited, 3 legs=policies/edited:locked" \
        --cmd 0.3 0 0 --T 12

Each panel is "label=policy_dir:fault" (fault: none | locked | passive). The camera follows the base.
"""
import argparse
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


def label_frame(img, text):
    """burn a simple text label into the frame (no PIL dependency: use matplotlib's agg)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    h, w, _ = img.shape
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    for k, line in enumerate(text.split("\n")):
        ax.text(12, 26 + 24 * k, line, color="black", fontsize=13, family="sans-serif",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2) if k == 0 else None)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.3, 0.0, 0.0])
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--fault_at", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--azimuth", type=float, default=-135.0)   # camera at the front-left (broken leg side)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sims, labels = [], []
    for p in a.panel:
        label, rest = p.split("=", 1)
        path, fault = rest.rsplit(":", 1)
        pol = NumpyPolicy(load_onnx_layers(os.path.join(path, "policy.onnx")))
        s = SoloSim(pol, os.path.join(path, "config.yaml"), fault=None if fault == "none" else fault, seed=a.seed,
                    fault_at=a.fault_at, phase0=0.0, init_noise=0.0)
        sims.append(s); labels.append(label)
    rend = [mujoco.Renderer(s.m, a.height, a.width) for s in sims]
    cams = []
    for s in sims:
        c = mujoco.MjvCamera(); c.distance = 0.95; c.elevation = -18; c.azimuth = a.azimuth
        cams.append(c)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
    every = int(round(1.0 / (a.fps * sims[0].dt)))
    n_settle = int(a.settle / sims[0].dt)
    n = n_settle + int(a.T / sims[0].dt)
    fallen = [False] * len(sims)
    for k in range(n):
        for i, s in enumerate(sims):
            if k == n_settle:
                s.cmd = np.array(a.cmd, float)
            if not fallen[i]:
                s.policy_step()
                fallen[i] = s.fallen()
        if k % every == 0:
            frames = []
            for i, (s, r, c) in enumerate(zip(sims, rend, cams)):
                c.lookat[:] = [s.d.qpos[0], s.d.qpos[1], 0.15]
                r.update_scene(s.d, c)
                img = r.render()
                t = k * s.dt
                txt = f"{labels[i]}\nt = {t:4.1f} s  cmd = ({s.cmd[0]:.1f}, {s.cmd[1]:.1f}, {s.cmd[2]:.1f})" + ("  FALLEN" if fallen[i] else "")
                frames.append(label_frame(img, txt))
            wr.append_data(np.concatenate(frames, axis=1))
    wr.close()
    print("wrote", a.out, os.path.getsize(a.out) / 1e6, "MB")


if __name__ == "__main__":
    main()
