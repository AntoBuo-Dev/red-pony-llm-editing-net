"""Deployment-style evaluation harness for the SOLO-12 policies: numpy (or onnxruntime) policy +
single MuJoCo instance. Independent of the training code: every constant comes from the policy's
config.yaml and from assets/solo12/solo12.xml.

Broken-leg conditions (sim options, see README):
  fault="locked"  : the leg's motors are off and its three joints are held at a fixed tucked pose
                    (joint equality constraints) -> the foot stays about 15 cm off the ground.
  fault="passive" : the leg's motors produce zero torque; the leg is free (joint damping 0.01,
                    friction loss 0.01 Nm) and folds under load.
"""
import math
import os

import numpy as np
import mujoco
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
XML = os.path.join(ROOT, "assets", "solo12", "solo12.xml")
LEGS = ["FL", "FR", "HL", "HR"]


def load_cfg(path):
    c = yaml.safe_load(open(path))
    for k in ("default_q", "obs_scale_cmd", "leg_phase_offset", "tucked_pose"):
        c[k] = np.array(c[k], float)
    return c


class SoloSim:
    def __init__(self, policy, cfg, fault=None, leg="FL", seed=0, friction=1.0, mass_offset=0.0,
                 kp_scale=1.0, obs_noise=0.0, init_noise=0.05, xml=XML, fault_at=0.0, phase0=None):
        self.cfg = load_cfg(cfg) if isinstance(cfg, str) else cfg
        c = self.cfg
        self.policy = policy
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.rs = np.random.RandomState(seed)
        self.m.geom_friction[:, 0] = friction
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.m.body_mass[bid] += mass_offset
        self.m.actuator_gainprm[:, 0] *= kp_scale
        self.m.actuator_biasprm[:, 1] *= kp_scale
        self.decim = int(c["decimation"])
        assert abs(self.m.opt.timestep * self.decim - c["dt"]) < 1e-9
        self.dt = c["dt"]
        self.default_q = c["default_q"]
        self.obs_noise = obs_noise
        self.leg = leg
        self.li = LEGS.index(leg)
        self.fault = fault
        self.fault_at = fault_at          # seconds after reset at which the leg breaks
        self.fault_on = False
        self.cmd = np.zeros(3)
        self.t = 0.0
        self._kp0 = self.m.actuator_gainprm[:, 0].copy()
        self._bias0 = self.m.actuator_biasprm[:, :3].copy()
        self.reset(init_noise, phase0)

    # ---------------------------------------------------------------- broken leg
    def _apply_fault(self, on, set_pose=False):
        m, d = self.m, self.d
        acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.leg}_{j}") for j in ("HAA", "HFE", "KFE")]
        eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{self.leg}_{j}") for j in ("HAA", "HFE", "KFE")]
        tuck = self.cfg["tucked_pose"].copy()
        if self.leg[1] == "R":
            tuck[0] = -tuck[0]
        if self.leg[0] == "H":
            tuck[1:] = -tuck[1:]
        for k, a in enumerate(acts):
            if on:
                m.actuator_gainprm[a, 0] = 0.0
                m.actuator_biasprm[a, :3] = 0.0
            else:
                m.actuator_gainprm[a, 0] = self._kp0[a]
                m.actuator_biasprm[a, :3] = self._bias0[a]
        for k, e in enumerate(eqs):
            m.eq_data[e, :5] = [tuck[k], 0, 0, 0, 0]
            d.eq_active[e] = 1 if (on and self.fault == "locked") else 0
        if on and self.fault == "locked" and set_pose:
            d.qpos[7 + 3 * self.li: 10 + 3 * self.li] = tuck
        self.fault_on = on

    def reset(self, init_noise=0.05, phase0=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0, 0, 0.235]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = self.default_q + self.rs.uniform(-init_noise, init_noise, 12)
        self.t = 0.0
        if self.fault is not None and self.fault_at <= 0.0:
            self._apply_fault(True, set_pose=True)
        else:
            self._apply_fault(False)
        mujoco.mj_forward(m, d)
        # drop so that the lowest foot touches the floor
        zmin = min(d.site(f"{l}_foot").xpos[2] for l in LEGS) - 0.016
        d.qpos[2] -= zmin - 0.001
        mujoco.mj_forward(m, d)
        self.action = np.zeros(12)
        self.phase = self.rs.uniform() if phase0 is None else phase0
        self.cmd = np.zeros(3)

    # ---------------------------------------------------------------- observation (same as training)
    def obs(self):
        c, d = self.cfg, self.d
        q = d.qpos[7:].copy(); dq = d.qvel[6:].copy()
        w, x, y, z = d.qpos[3:7]
        grav = np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])
        ang = d.qvel[3:6].copy()                       # free joint: body-frame angular velocity (= gyro)
        cmd = self.cmd.copy()
        stand = np.linalg.norm(cmd) < c["stand_cmd_norm"]
        clock = np.zeros(2) if stand else np.array([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)])
        o = np.concatenate([ang * c["obs_scale_ang_vel"], grav, cmd * c["obs_scale_cmd"],
                            (q - self.default_q) * c["obs_scale_dof_pos"], dq * c["obs_scale_dof_vel"], self.action, clock])
        if self.obs_noise > 0:
            o = o + self.obs_noise * self.rs.uniform(-1, 1, o.shape) * np.r_[0.05 * np.ones(3), 0.05 * np.ones(3), np.zeros(3),
                                                                                0.01 * np.ones(12), 0.075 * np.ones(12), np.zeros(14)]
        return o.astype(np.float32)

    def policy_step(self, action_override=None):
        if self.fault is not None and not self.fault_on and self.t >= self.fault_at:
            self._apply_fault(True)
        o = self.obs()
        a = np.asarray(self.policy(o[None])[0], np.float64) if action_override is None else action_override
        a = np.clip(a, -self.cfg["action_clip"], self.cfg["action_clip"])
        self.action = a
        self.d.ctrl[:] = self.default_q + self.cfg["action_scale"] * a
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)
        self.phase = (self.phase + self.dt * self.cfg["gait_freq"]) % 1.0
        self.t += self.dt
        return o, a

    # ---------------------------------------------------------------- helpers
    def base(self):
        q = self.d.qpos
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        up_z = 1 - 2 * (x * x + y * y)
        return q[0], q[1], q[2], yaw, up_z

    def body_vel(self):
        w, x, y, z = self.d.qpos[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R.T @ self.d.qvel[:3], self.d.qvel[3:6].copy()

    def fallen(self):
        x, y, z, yaw, up = self.base()
        bt = self.d.sensor("base_touch").data[0]
        return bt > 0.1 or up < 0.5 or z < 0.10

    def leg_ground_contact(self, leg):
        """True if any collision geom of `leg` (thigh, shin, foot) touches the floor."""
        if not hasattr(self, "_leg_geoms"):
            self._floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self._leg_geoms = {l: {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_{p}") for p in ("thigh", "shin", "foot")}
                               for l in LEGS}
        gs = self._leg_geoms[leg]
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if (c.geom1 == self._floor and c.geom2 in gs) or (c.geom2 == self._floor and c.geom1 in gs):
                return True
        return False

    def foot_contacts(self):
        return np.array([self.d.sensor(f"{l}_foot_touch").data[0] > 1.0 for l in LEGS])


def run_episode(policy, cfg, cmd, T=20.0, settle=1.0, fault=None, leg="FL", seed=0, record=False, sim_kw=None,
                on_step=None):
    """Stand still for `settle` s (zero command), then hold `cmd` for T s. Metrics are measured in the
    command phase relative to the pose at its start. Returns a dict (and the trajectory if record)."""
    rs = np.random.RandomState(1000 + seed)
    kw = dict(friction=rs.uniform(0.7, 1.1), mass_offset=rs.uniform(-0.1, 0.2), seed=seed)
    kw.update(sim_kw or {})
    sim = SoloSim(policy, cfg, fault=fault, leg=leg, **kw)
    fell, t_fall = False, None
    for _ in range(int(settle / sim.dt)):
        sim.policy_step()
        if sim.fallen():
            fell, t_fall = True, 0.0
            break
    x0, y0, z0, yaw0, _ = sim.base()
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    sim.cmd = np.array(cmd, float)
    traj, acts, vels, contacts, obs_log, broken_touch = [], [], [], [], [], []
    yaw_prev, yaw_unw = yaw0, 0.0
    n = int(T / sim.dt)
    for i in range(n if not fell else 0):
        o, a = sim.policy_step()
        x, y, z, yaw, up = sim.base()
        dyaw = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_unw += dyaw; yaw_prev = yaw
        fwd = (x - x0) * c0 + (y - y0) * s0
        lat = -(x - x0) * s0 + (y - y0) * c0
        vb, wb = sim.body_vel()
        traj.append((sim.t, x - x0, y - y0, fwd, lat, yaw_unw, z, up))
        vels.append(np.r_[vb, wb]); acts.append(a.copy()); contacts.append(sim.foot_contacts())
        if fault is None or sim.fault_on:          # count only once the leg is broken
            broken_touch.append(sim.leg_ground_contact(leg))
        if record:
            obs_log.append(o)
        if on_step is not None:
            on_step(sim, i)
        if sim.fallen():
            fell, t_fall = True, (i + 1) * sim.dt
            break
    traj = np.array(traj) if traj else np.zeros((1, 8))
    vels = np.array(vels) if vels else np.zeros((1, 6))
    acts = np.array(acts) if acts else np.zeros((2, 12))
    n_ok = len(traj)
    k0 = min(int(2.0 / sim.dt), max(n_ok - 1, 0))      # skip the first 2 s (transient) for the speed averages
    good = [i for i in range(12) if not (fault and i // 3 == LEGS.index(leg))]
    dact = np.abs(np.diff(acts, axis=0))
    path = np.sum(np.linalg.norm(np.diff(traj[:, 1:3], axis=0), axis=1)) if n_ok > 1 else 0.0
    res = dict(cmd=list(map(float, cmd)), seed=seed, fault=fault, fell=bool(fell),
               survival_s=float(T if not fell else (t_fall or 0.0)),
               vx=float(vels[k0:, 0].mean()), vy=float(vels[k0:, 1].mean()), wz=float(vels[k0:, 5].mean()),
               fwd_m=float(traj[-1, 3]), lat_m=float(traj[-1, 4]), yaw_deg=float(math.degrees(traj[-1, 5])),
               max_abs_lat_m=float(np.abs(traj[:, 4]).max()), path_m=float(path),
               net_m=float(np.hypot(traj[-1, 1], traj[-1, 2])),
               height_m=float(traj[k0:, 6].mean()), tilt_deg=float(np.degrees(np.arccos(np.clip(traj[k0:, 7], -1, 1))).mean()),
               dact_all=float(dact.mean()), dact_legs=float(dact[:, good].mean()),
               duration_s=float(n_ok * sim.dt),
               broken_leg_ground_frac=float(np.mean(broken_touch)) if broken_touch else 0.0)
    # with a fault during the episode: drift / heading / speed measured after the fault only
    if fault is not None and kw.get("fault_at", 0.0) > settle and n_ok > 1:
        i0 = min(int((kw["fault_at"] - settle) / sim.dt) + int(2.0 / sim.dt), n_ok - 1)   # 2 s after the fault
        d = max((n_ok - i0) * sim.dt, 1e-6)
        res.update(post_vx=float(vels[i0:, 0].mean()), post_wz=float(vels[i0:, 5].mean()),
                   post_lat_rate_cm_s=float(100 * (traj[-1, 4] - traj[i0, 4]) / d),
                   post_yaw_rate_deg_s=float(math.degrees(traj[-1, 5] - traj[i0, 5]) / d),
                   post_tilt_deg=float(np.degrees(np.arccos(np.clip(traj[i0:, 7], -1, 1))).mean()))
    dur = max(res["duration_s"] - k0 * sim.dt, 1e-6)
    res["lat_rate_cm_s"] = float(100 * (traj[-1, 4] - traj[k0, 4]) / dur) if n_ok > k0 + 1 else 0.0
    res["yaw_rate_deg_s"] = float(math.degrees(traj[-1, 5] - traj[k0, 5]) / dur) if n_ok > k0 + 1 else 0.0
    if record:
        res["traj"] = traj; res["acts"] = acts; res["vels"] = vels; res["contacts"] = np.array(contacts)
        res["obs"] = np.array(obs_log)
    return res
