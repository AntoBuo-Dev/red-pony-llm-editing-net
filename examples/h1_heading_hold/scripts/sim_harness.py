"""Headless Python replica of simulator + controller (sim2sim) for the H1_2 squat policy.

Mirrors deploy/include (observation layout, history, gait clock, cutoff) and the bridge PD law.
"""
import math
import numpy as np
import mujoco
import yaml
import onnxruntime as ort

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE = REPO + "/assets/h1_2/scene.xml"

TERMS = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos_rel",
         "joint_vel_rel", "last_action", "gait_phase", "base_height_command"]


class NumpyPolicy:
    """MLP with ELU evaluated in numpy from a list of (W, b)."""

    def __init__(self, layers):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in layers]

    def __call__(self, obs, return_hidden=False):
        x = np.asarray(obs, np.float32)
        hs = []
        n = len(self.layers)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < n - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
                hs.append(x)
        return (x, hs) if return_hidden else x


def load_onnx_layers(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t).copy() for t in m.graph.initializer}
    return [(init[f"mlp.{i}.weight"], init[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]


class Sim:
    def __init__(self, policy, deploy_yaml, seed=0, friction=None, gyro_bias=0.0):
        self.cfg = yaml.safe_load(open(deploy_yaml))
        self.policy = policy
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        if friction is not None:
            self.m.geom_friction[:, 0] = friction
        self.nu = self.m.nu
        self.ids = np.array(self.cfg["joint_ids_map"], int)  # policy idx -> sdk idx
        self.kp = np.zeros(self.nu); self.kd = np.zeros(self.nu)
        self.kp[:] = self.cfg["stiffness"]; self.kd[:] = self.cfg["damping"]  # SDK order
        self.default = np.array(self.cfg["default_joint_pos"], np.float64)
        self.scale = np.array(self.cfg["actions"]["JointPositionAction"]["scale"])
        self.offset = np.array(self.cfg["actions"]["JointPositionAction"]["offset"])
        self.step_dt = self.cfg["step_dt"]
        self.decim = int(round(self.step_dt / self.m.opt.timestep))
        s = lambda n: self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        self.a_quat, self.a_gyro = s("imu_quat"), s("imu_gyro")
        r = self.cfg["commands"]["base_velocity"]["ranges"]
        self.rng = (r["lin_vel_x"], r["lin_vel_y"], r["ang_vel_z"])
        self.cutoff = self.cfg["commands"]["base_velocity"].get("cutoff_height", None)
        self.hr = self.cfg["commands"]["base_height"]["ranges"]["height"]
        self.obs_scales = [np.array(self.cfg["observations"][t]["scale"], np.float32) for t in TERMS]
        self.H = 5
        self.gyro_bias = gyro_bias
        self.rs = np.random.RandomState(seed)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        q_sdk = np.zeros(self.nu); q_sdk[self.ids] = self.default
        d.qpos[7:] = q_sdk
        d.qpos[2] = 1.2
        mujoco.mj_forward(m, d)
        # drop the robot so that the lowest collision geom touches the floor
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        dmin = 1e9
        for g in range(m.ngeom):
            if g == floor or m.geom_contype[g] == 0:
                continue
            dist = mujoco.mj_geomDistance(m, d, g, floor, 2.0, None)
            dmin = min(dmin, dist)
        d.qpos[2] -= dmin - 0.001
        mujoco.mj_forward(m, d)
        self.q_target_sdk = q_sdk.copy()
        self.action = np.zeros(27, np.float32)
        self.phase = 0.0
        self.h_target = self.hr[1]
        self.cmd = np.zeros(3)
        self.hist = None
        self._terms()  # like observation_manager->reset(): fills the history with the first sample
        self.hist = [[t.copy() for _ in range(self.H)] for t in self._last_terms]

    def _terms(self):
        d = self.d
        sd = d.sensordata
        q = sd[:self.nu][self.ids]; dq = sd[self.nu:2 * self.nu][self.ids]
        gyro = sd[self.a_gyro:self.a_gyro + 3].copy(); gyro[2] += self.gyro_bias
        w, x, y, z = sd[self.a_quat:self.a_quat + 4]
        # gravity (0,0,-1) in body frame
        g = np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])
        cmd = np.array([np.clip(self.cmd[i], *self.rng[i]) for i in range(3)])
        if self.cutoff is not None and self.h_target < self.cutoff:
            cmd[:] = 0
        self.phase = math.fmod(self.phase + self.step_dt / 0.8, 1.0)
        gp = np.array([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)])
        if np.linalg.norm(cmd) <= 0.1:
            gp[:] = 0
        raw = [gyro, g, cmd, q - self.default, dq, self.action, gp, np.array([self.h_target - 0.98])]
        self._last_terms = [np.asarray(r, np.float32) * s for r, s in zip(raw, self.obs_scales)]
        return self._last_terms

    def obs(self):
        for h, t in zip(self.hist, self._terms()):
            h.pop(0); h.append(t.copy())
        return np.concatenate([np.concatenate(h) for h in self.hist]).astype(np.float32)

    def policy_step(self):
        o = self.obs()
        a = np.asarray(self.policy(o[None])[0], np.float32)
        self.action = a
        qt = self.offset + self.scale * a
        self.q_target_sdk = np.zeros(self.nu); self.q_target_sdk[self.ids] = qt
        for _ in range(self.decim):
            sd = self.d.sensordata
            self.d.ctrl[:] = self.kp * (self.q_target_sdk - sd[:self.nu]) - self.kd * sd[self.nu:2 * self.nu]
            mujoco.mj_step(self.m, self.d)
        return o, a

    # ------------------------------------------------------------------ helpers
    def base(self):
        q = self.d.qpos
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        up_z = 1 - 2 * (x * x + y * y)
        return q[0], q[1], q[2], yaw, up_z

    def set_height(self, h, rate=0.05):
        """Ramp like the controller (0.05 m/s); call every policy step."""
        if self.h_target > h:
            self.h_target = max(h, self.h_target - rate * self.step_dt)
        elif self.h_target < h:
            self.h_target = min(h, self.h_target + rate * self.step_dt)


def run_segment(policy, deploy_yaml, vx=0.0, vy=0.0, wz=0.0, height=0.98, T=20.0, settle=3.0,
                record=False, **kw):
    """Stand, go to height, then hold a constant command for T seconds.
    Returns drift metrics measured w.r.t. the heading at the start of the command."""
    sim = Sim(policy, deploy_yaml, **kw)
    n_settle = int(settle / sim.step_dt)
    log = []
    fell = False
    k = 0
    while True:
        sim.set_height(height)
        done_h = abs(sim.h_target - height) < 1e-6
        o, a = sim.policy_step(); k += 1
        if done_h and k >= n_settle:
            break
    for _ in range(int(1.5 / sim.step_dt)):
        sim.policy_step()
    x0, y0, z0, yaw0, _ = sim.base()
    c, s = math.cos(yaw0), math.sin(yaw0)
    sim.cmd = np.array([vx, vy, wz], float)
    traj = []
    yaw_prev, yaw_unw = yaw0, 0.0
    obs_log, act_log = [], []
    for i in range(int(T / sim.step_dt)):
        o, a = sim.policy_step()
        x, y, z, yaw, up = sim.base()
        dyaw = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_unw += dyaw; yaw_prev = yaw
        fwd = (x - x0) * c + (y - y0) * s
        lat = -(x - x0) * s + (y - y0) * c
        traj.append((fwd, lat, yaw_unw, z))
        if record:
            obs_log.append(o); act_log.append(a)
        if up < 0.5 or z < 0.35:
            fell = True; break
    traj = np.array(traj)
    res = dict(vx=vx, vy=vy, wz=wz, height=height, fell=fell, fwd=traj[-1, 0], lat=traj[-1, 1],
               yaw_deg=math.degrees(traj[-1, 2]), max_lat=float(np.abs(traj[:, 1]).max()),
               max_yaw_deg=float(np.degrees(np.abs(traj[:, 2])).max()), z=float(traj[-200:, 3].mean()))
    if record:
        res["traj"] = traj; res["obs"] = np.array(obs_log); res["act"] = np.array(act_log)
    return res


