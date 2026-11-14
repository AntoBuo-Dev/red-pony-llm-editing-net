#!/usr/bin/env python3
"""Record videos of several policies on the same command in the Python sim2sim replica.

    MUJOCO_GL=egl python evaluation/heading_hold/record_video.py [--out videos/heading_hold]

For every scenario: one clip per policy (chase camera, red line = torso heading at the start
of the command) and a side-by-side clip with a top view of the three paths.
"""
import argparse, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_harness import Sim, NumpyPolicy, load_onnx_layers, REPO

POL = REPO + "/policies/h1_2_squat/"
POLICIES = [("originale", POL + "2026-09-13_21-27-39", (230, 80, 60)),
            ("rotta", POL + "2026-11-11_heading-hold", (240, 170, 30)),
            ("rotta + laterale", POL + "2026-11-11_heading-lateral-hold", (60, 170, 90))]
SCENARIOS = [("in_piedi_0.98m_0.8ms", 0.98, 0.8, 20.0), ("accovacciato_0.74m_0.6ms", 0.74, 0.6, 20.0)]
W, H, FPS = 640, 640, 25


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()
F1, F2 = font(26), font(20)


def rollout(pdir, height, vx, T):
    """Returns frames (list of HxWx3) and the path [(t, fwd, lat, yaw_deg)] from the start of the command."""
    sim = Sim(NumpyPolicy(load_onnx_layers(pdir + "/exported/policy.onnx")), POL + "2026-09-13_21-27-39/params/deploy.yaml")
    sim.m.vis.global_.offwidth = W; sim.m.vis.global_.offheight = H
    ren = mujoco.Renderer(sim.m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = 5.0; cam.elevation = -28; cam.azimuth = 0
    for k in range(int((3 + (0.98 - height) / 0.05 + 1.5) / 0.02)):      # stand, go to height (not recorded)
        sim.set_height(height); sim.policy_step()
    tid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    def torso_yaw():                                                      # the IMU is on the torso
        R = sim.d.xmat[tid].reshape(3, 3); return math.atan2(R[1, 0], R[0, 0])
    x0, y0, _, _, _ = sim.base(); yaw0 = torso_yaw(); c, s = math.cos(yaw0), math.sin(yaw0)
    cam.azimuth = math.degrees(yaw0) + 180                                # camera behind the robot
    frames, path = [], []
    n_pre = int(1.0 / 0.02)
    for k in range(n_pre + int(T / 0.02)):
        if k <= n_pre:                                   # the reference is the torso heading when the command starts,
            x0, y0 = sim.base()[:2]; yaw0 = torso_yaw()  # which is what the network latches (standing, the robot creeps in yaw)
            c, s = math.cos(yaw0), math.sin(yaw0); cam.azimuth = math.degrees(yaw0) + 180
        if k == n_pre: sim.cmd = np.array([vx, 0.0, 0.0])
        sim.policy_step()
        x, y, z, _, up = sim.base(); yaw = torso_yaw()
        fwd = (x - x0) * c + (y - y0) * s; lat = -(x - x0) * s + (y - y0) * c
        dyaw = math.degrees((yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi)
        path.append(((k - n_pre) * 0.02, fwd, lat, dyaw))
        if k % 2 == 0:
            cam.lookat[:] = (x + 0.8 * c, y + 0.8 * s, 0.75)
            ren.update_scene(sim.d, cam)
            g = ren.scene.geoms[ren.scene.ngeom]                           # reference line on the floor
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([20.0, 0.025, 0.002]),
                                np.array([x0 + 18 * c, y0 + 18 * s, 0.003]), R.flatten(), np.array([1.0, 0.1, 0.1, 1.0], np.float32))
            ren.scene.ngeom += 1
            frames.append((ren.render().copy(), len(path) - 1))
        if up < 0.5 or z < 0.35: break
    ren.close()
    return frames, np.array(path)


def label(img, name, color, row, cmd):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 92], fill=(0, 0, 0)); d.rectangle([0, 0, 10, 92], fill=color)
    d.text((22, 8), name, font=F1, fill=(255, 255, 255))
    d.text((22, 44), cmd, font=F2, fill=(190, 190, 190))
    d.text((22, 66), f"t {max(row[0], 0):4.1f} s   laterale {row[2]:+6.2f} m   imbardata {row[3]:+6.1f}°", font=F2, fill=(255, 255, 255))
    return np.asarray(im)


def top_view(paths, idx, xmax, width=3 * W, height=440, ymax=2.6):
    im = Image.new("RGB", (width, height), (18, 18, 20)); d = ImageDraw.Draw(im)
    m = 50; sx = (width - 2 * m) / xmax; sy = (height / 2 - 30) / ymax; cy = height // 2 + 10
    px = lambda f, l: (m + f * sx, cy - l * sy)
    for gx in range(0, int(xmax) + 1, 2):
        d.line([px(gx, -ymax), px(gx, ymax)], fill=(45, 45, 50)); d.text((px(gx, -ymax)[0] - 8, height - 22), f"{gx}", font=F2, fill=(150, 150, 150))
    for gy in (-2, -1, 1, 2):
        d.line([px(0, gy), px(xmax, gy)], fill=(45, 45, 50)); d.text((6, px(0, gy)[1] - 10), f"{gy:+d}", font=F2, fill=(150, 150, 150))
    d.line([px(0, 0), px(xmax, 0)], fill=(255, 40, 40), width=2)
    d.text((m, 6), "Vista dall'alto: avanzamento [m] in orizzontale, scostamento laterale [m] in verticale (linea rossa = rotta iniziale del busto, dove sta l'IMU)", font=F2, fill=(210, 210, 210))
    for (name, _, col), p in zip(POLICIES, paths):
        q = p[:min(idx, len(p) - 1) + 1]; q = q[q[:, 0] >= 0]
        pts = [px(f, float(np.clip(l, -ymax, ymax))) for _, f, l, _ in q[::5]]
        if len(pts) > 1: d.line(pts, fill=col, width=4)
        if pts: X, Y = pts[-1]; d.ellipse([X - 7, Y - 7, X + 7, Y + 7], fill=col)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=REPO + "/videos/heading_hold"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sname, h, vx, T in SCENARIOS:
        cmd = f"altezza {h:.2f} m, comando vx {vx:.1f} m/s, vy 0, wz 0"
        runs = [rollout(p, h, vx, T) for _, p, _ in POLICIES]
        n = min(len(r[0]) for r in runs)
        clips = []
        for (name, _, col), (frames, path) in zip(POLICIES, runs):
            lab = [label(img, name, col, path[i], cmd) for img, i in frames]; clips.append(lab)
            fn = f"{a.out}/{sname}__{name.replace(' + ', '_').replace(' ', '_')}.mp4"
            imageio.mimwrite(fn, lab, fps=FPS, quality=7, macro_block_size=1); print("written", fn, "final lateral %.2f m, yaw %.1f deg, forward %.1f m" % (path[-1, 2], path[-1, 3], path[-1, 1]))
        xmax = max(float(r[1][:, 1].max()) for r in runs) + 1.0
        comp = [np.vstack([np.hstack([c[i] for c in clips]), top_view([r[1] for r in runs], runs[0][0][i][1], xmax)]) for i in range(n)]
        fn = f"{a.out}/{sname}__confronto.mp4"
        imageio.mimwrite(fn, comp, fps=FPS, quality=7, macro_block_size=1); print("written", fn)
        Image.fromarray(comp[-1]).save(f"{a.out}/{sname}__confronto_ultimo_frame.png")
        Image.fromarray(comp[n // 2]).save(f"{a.out}/{sname}__confronto_meta.png")


if __name__ == "__main__":
    main()
