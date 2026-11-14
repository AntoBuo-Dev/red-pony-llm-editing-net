#!/usr/bin/env python3
"""Analysis videos: simulator view with the reference line + live plots of the circuit state.

    MUJOCO_GL=egl python evaluation/heading_hold/record_analysis.py [--out videos/heading_hold/analisi]

One clip per policy and scenario. Left: chase camera (red line = torso heading at the start of the
command) and top view of the path. Right: true heading/lateral error against the values the network
keeps in its registers, the corrections it feeds to the original network, and the wrist joints that
carry the registers. A summary PNG per scenario overlays the three policies.
"""
import argparse, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_harness import Sim, NumpyPolicy, load_onnx_layers, REPO
from record_video import POLICIES, POL, F1, F2

# name, height, vx, duration, pushes [(t, yaw-rate kick rad/s, lateral velocity kick m/s)]
SCENARIOS = [("in_piedi_0.98m_0.8ms_con_spinte", 0.98, 0.8, 26.0, [(9.0, 1.2, 0.0), (17.0, 0.0, 0.35)]),
             ("accovacciato_0.74m_0.6ms", 0.74, 0.6, 22.0, [])]
VW, VH, PW, FPS = 800, 760, 1120, 25
KTH, KI, KP, S1, S2, S3 = 3.0, 3.0, 2.0, 2.0, 2.0, 4.0
J = {"left_wrist_roll_joint": 21, "left_wrist_yaw_joint": 25, "right_wrist_yaw_joint": 26}   # policy indices
plt.rcParams.update({"figure.facecolor": "#121214", "axes.facecolor": "#1b1b1f", "axes.edgecolor": "#666", "text.color": "#ddd",
                     "axes.labelcolor": "#ddd", "xtick.color": "#aaa", "ytick.color": "#aaa", "grid.color": "#333", "font.size": 11,
                     "legend.facecolor": "#1b1b1f", "legend.edgecolor": "#444", "legend.fontsize": 9.5})


def rollout(pdir, height, vx, T, pushes, render=True):
    sim = Sim(NumpyPolicy(load_onnx_layers(pdir + "/exported/policy.onnx")), POL + "2026-09-13_21-27-39/params/deploy.yaml")
    sim.m.vis.global_.offwidth = VW; sim.m.vis.global_.offheight = VH
    ren = mujoco.Renderer(sim.m, VH, VW) if render else None
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = 5.0; cam.elevation = -28
    for k in range(int((3 + (0.98 - height) / 0.05 + 1.5) / 0.02)):
        sim.set_height(height); sim.policy_step()
    tid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    def torso_yaw():
        R = sim.d.xmat[tid].reshape(3, 3); return math.atan2(R[1, 0], R[0, 0])
    x0, y0 = sim.base()[:2]; yaw0 = torso_yaw(); c, s = math.cos(yaw0), math.sin(yaw0)
    cam.azimuth = math.degrees(yaw0) + 180
    sdk = {n: int(sim.ids[j]) for n, j in J.items()}
    frames, log = [], []
    n_pre = int(2.0 / 0.02); yaw_prev, yaw_unw = yaw0, 0.0
    for k in range(n_pre + int(T / 0.02)):
        t = (k - n_pre) * 0.02
        if k <= n_pre:                                   # the reference is the torso heading when the command starts,
            x0, y0 = sim.base()[:2]; yaw0 = torso_yaw()  # which is what the network latches (standing, the robot creeps in yaw)
            c, s = math.cos(yaw0), math.sin(yaw0); cam.azimuth = math.degrees(yaw0) + 180
            yaw_prev, yaw_unw = yaw0, 0.0
        if k == n_pre: sim.cmd = np.array([vx, 0.0, 0.0])
        for tp, kyaw, klat in pushes:
            if abs(t - tp) < 0.01:
                sim.d.qvel[5] += kyaw; sim.d.qvel[0] += -s * klat; sim.d.qvel[1] += c * klat
        o, a = sim.policy_step()
        x, y, z, _, up = sim.base(); yaw = torso_yaw()
        yaw_unw += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi; yaw_prev = yaw
        v = sim.d.qvel[:2]
        log.append([t, (x - x0) * c + (y - y0) * s, -(x - x0) * s + (y - y0) * c, math.degrees(yaw_unw), -s * v[0] + c * v[1],
                    a[25], a[26], a[21]] + [sim.d.qpos[7 + sdk[n]] for n in J] + [o[14] / 0.2])
        if render and k % 2 == 0:
            cam.lookat[:] = (x + 0.8 * c, y + 0.8 * s, 0.75)
            ren.update_scene(sim.d, cam)
            g = ren.scene.geoms[ren.scene.ngeom]
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([25.0, 0.025, 0.002]),
                                np.array([x0 + 22 * c, y0 + 22 * s, 0.003]), R.flatten(), np.array([1.0, 0.1, 0.1, 1.0], np.float32))
            ren.scene.ngeom += 1
            frames.append((ren.render().copy(), len(log) - 1))
        if up < 0.5 or z < 0.35: break
    if ren: ren.close()
    L = np.array(log)
    return frames, dict(t=L[:, 0], fwd=L[:, 1], lat=L[:, 2], yaw=L[:, 3], vlat=L[:, 4], a25=L[:, 5], a26=L[:, 6], a21=L[:, 7],
                        q_lroll=L[:, 8], q_lyaw=L[:, 9], q_ryaw=L[:, 10], gz=L[:, 11])


def panels(axs, d, kind, pushes, title=None):
    """kind: 0 original, 1 heading hold, 2 heading + lateral hold."""
    t = d["t"]; deg = np.degrees
    for ax in axs:
        ax.grid(True); ax.axvline(0, color="#888", lw=1, ls=":")
        for tp, kyaw, klat in pushes: ax.axvline(tp, color="#c8f", lw=1, ls="--")
    ax = axs[0]
    ax.plot(t, d["yaw"], color="#4da3ff", lw=1.6, label="imbardata vera del busto [°]")
    if kind >= 1:
        ax.plot(t, -deg(d["a25"] / S1), color="#ffb000", lw=1.3, ls="--", label="imbardata stimata dalla rete = −θ (registro 25) [°]")
        ax.plot(t, deg(d["a26"] / S2), color="#7fd17f", lw=1.2, label="integrale I (registro 26) [°·s]")
    ax.set_ylabel("rotta"); ax.legend(loc="upper right", ncol=1)
    if title: ax.set_title(title, loc="left", fontsize=13)
    ax = axs[1]
    ax.plot(t, 100 * d["lat"], color="#4da3ff", lw=1.6, label="scostamento laterale vero dalla linea [cm]")
    if kind == 2:
        ax.plot(t, 100 * d["a21"] / S3, color="#ffb000", lw=1.3, ls="--", label="scostamento stimato dalla rete p (registro 21) [cm]")
    ax.set_ylabel("laterale"); ax.legend(loc="upper left")
    ax = axs[2]
    if kind >= 1:
        ax.plot(t, KTH * d["a25"] / S1 + KI * d["a26"] / S2, color="#ff6b6b", lw=1.4, label="correzione di rotazione wz = 3θ + 3I [rad/s]")
    if kind == 2:
        ax.plot(t, -KP * d["a21"] / S3, color="#c792ea", lw=1.4, label="correzione laterale vy = −2p [m/s]")
    if kind == 0: ax.text(0.5, 0.5, "rete originale: nessuna correzione interna", transform=ax.transAxes, ha="center", color="#888")
    else: ax.legend(loc="upper right")
    ax.set_ylabel("comando interno")
    ax = axs[3]
    for key, act, col, name in (("q_lyaw", "a25", "#ffb000", "polso sx yaw (25)"), ("q_ryaw", "a26", "#7fd17f", "polso dx yaw (26)"), ("q_lroll", "a21", "#c792ea", "polso sx roll (21)")):
        ax.plot(t, deg(d[key]), color=col, lw=1.5, label=name + " misurato [°]")
        ax.plot(t, deg(0.25 * d[act]), color=col, lw=0.9, ls=":", label="  bersaglio 0,25·azione")
    ax.set_ylabel("polsi"); ax.set_xlabel("tempo dal comando di cammino [s]"); ax.legend(loc="upper right", ncol=3, fontsize=8)


def make_plot(d, kind, pushes, title):
    fig, axs = plt.subplots(4, 1, figsize=(PW / 100, 10.8), dpi=100, sharex=True)
    panels(axs, d, kind, pushes, title)
    axs[0].set_xlim(d["t"][0], d["t"][-1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.055, hspace=0.12)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x_of = lambda t: axs[0].transData.transform((t, 0))[0]
    boxes = [ax.get_window_extent() for ax in axs]
    spans = [(int(img.shape[0] - b.y1), int(img.shape[0] - b.y0)) for b in boxes]
    plt.close(fig)
    return img, x_of, spans


def left_column(frame, d, i, name, color, cmd, xmax):
    im = Image.new("RGB", (VW, 1080), (18, 18, 20)); im.paste(Image.fromarray(frame), (0, 0)); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, VW, 70], fill=(0, 0, 0)); dr.rectangle([0, 0, 10, 70], fill=color)
    dr.text((22, 6), name, font=F1, fill=(255, 255, 255)); dr.text((22, 42), cmd, font=F2, fill=(190, 190, 190))
    top, hgt, m, ymax = VH, 1080 - VH, 46, 1.2
    sx = (VW - 2 * m) / xmax; sy = (hgt / 2 - 34) / ymax; cy = top + hgt // 2 + 12
    px = lambda f, l: (m + f * sx, cy - float(np.clip(l, -ymax, ymax)) * sy)
    dr.text((m, top + 6), "Vista dall'alto: avanzamento [m], scostamento laterale [m]", font=F2, fill=(210, 210, 210))
    for gy in (-1, -0.5, 0.5, 1):
        dr.line([px(0, gy), px(xmax, gy)], fill=(45, 45, 50)); dr.text((2, px(0, gy)[1] - 10), f"{gy:+.1f}", font=F2, fill=(150, 150, 150))
    for gx in range(0, int(xmax) + 1, 4): dr.text((px(gx, 0)[0] - 6, 1080 - 24), str(gx), font=F2, fill=(150, 150, 150))
    dr.line([px(0, 0), px(xmax, 0)], fill=(255, 40, 40), width=2)
    q = np.column_stack([d["fwd"][:i + 1], d["lat"][:i + 1]])[d["t"][:i + 1] >= 0]
    pts = [px(f, l) for f, l in q[::4]]
    if len(pts) > 1: dr.line(pts, fill=color, width=4)
    if pts: X, Y = pts[-1]; dr.ellipse([X - 7, Y - 7, X + 7, Y + 7], fill=color)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=REPO + "/videos/heading_hold/analisi"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sname, h, vx, T, pushes in SCENARIOS:
        cmd = f"altezza {h:.2f} m, comando vx {vx:.1f} m/s" + ("; spinte: " + ", ".join(f"{'imbardata' if ky else 'laterale'} a {tp:.0f} s" for tp, ky, kl in pushes) if pushes else "")
        data = []
        for kind, (name, pdir, col) in enumerate(POLICIES):
            frames, d = rollout(pdir, h, vx, T, pushes); data.append(d)
            plot, x_of, spans = make_plot(d, kind, pushes, f"{name} — {cmd}")
            xmax = float(d["fwd"].max()) + 1.0; out = []
            for img, i in frames:
                right = Image.fromarray(plot); dr = ImageDraw.Draw(right); X = x_of(d["t"][i])
                for y0, y1 in spans: dr.line([(X, y0), (X, y1)], fill=(255, 255, 255), width=2)
                out.append(np.hstack([left_column(img, d, i, name, col, cmd, xmax), np.asarray(right)]))
            fn = f"{a.out}/{sname}__{name.replace(' + ', '_').replace(' ', '_')}.mp4"
            imageio.mimwrite(fn, out, fps=FPS, quality=7, macro_block_size=1)
            Image.fromarray(out[-1]).save(fn.replace(".mp4", "_ultimo_frame.png"))
            print("written", fn, "| final lateral %.2f m, yaw %.1f deg, max |wrist| %.1f deg" % (d["lat"][-1], d["yaw"][-1], np.degrees(max(np.abs(d[k]).max() for k in ("q_lyaw", "q_ryaw", "q_lroll")))))
        fig, axs = plt.subplots(4, 3, figsize=(24, 12), dpi=90, sharex=True, sharey="row")
        for kind, (name, _, _) in enumerate(POLICIES): panels(axs[:, kind], data[kind], kind, pushes, name)
        for ax in axs[:, 1:].ravel(): ax.set_ylabel("")
        axs[0, 0].set_ylim(-12, 12); axs[1, 0].set_ylim(-60, 60)
        fig.suptitle(cmd + "   (assi di rotta e laterale tagliati a ±12° e ±60 cm)", fontsize=14); fig.tight_layout()
        fig.savefig(f"{a.out}/{sname}__riepilogo.png"); plt.close(fig); print("written", f"{a.out}/{sname}__riepilogo.png")


if __name__ == "__main__":
    main()
