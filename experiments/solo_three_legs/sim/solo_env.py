"""Vectorised SOLO-12 locomotion environment on CPU MuJoCo (mujoco.rollout thread pool).

Each policy step (20 ms) is one batched call of mujoco.rollout with 10 physics steps of 2 ms; the
joint PD runs inside MuJoCo (position actuators, kp = 4, kd = 0.2, |tau| <= 2.7 Nm).
Observations, rewards and resets are computed in numpy over the whole batch.
"""
import copy
import numpy as np
import mujoco
from mujoco import rollout

from solo_config import *  # noqa: F401,F403

NQ, NV = 19, 18
STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
# sensordata layout (see build_model.py)
SD_FOOT_POS, SD_FOOT_VEL = slice(0, 12), slice(12, 24)
SD_FOOT_TOUCH, SD_KNEE_TOUCH, SD_BASE_TOUCH = slice(24, 28), slice(28, 32), 32
SD_TAU = slice(33, 45)

PRIV_DIM = 3 + 1 + 4 + 4 + 4 + 3      # lin vel, height, contacts, foot heights, DR params, push vel


def quat_to_mat(q):
    """(N,4) wxyz -> (N,3,3) body-to-world rotation."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1)


class RewardCfg:
    track_lin = 1.5
    track_ang = 0.75
    lin_sigma = 0.1
    ang_sigma = 0.2
    lin_vel_z = -2.0
    ang_vel_xy = -0.05
    orientation = -5.0
    base_height = -50.0
    base_height_target = 0.21
    torques = -2e-4
    dof_acc = -2.5e-7
    action_rate = -0.01
    collision = -1.0
    gait_contact = 1.0
    swing_height = -30.0
    swing_peak = 0.05
    foot_slip = -0.2
    stand_still = -0.5
    dof_limits = -5.0
    hip_pos = -0.2
    termination = -5.0


class SoloVecEnv:
    def __init__(self, num_envs=2048, seed=0, nthread=16, dr=True, n_models=256, ep_len_s=20.0,
                 cmd_resample_s=5.0, fault=None, obs_noise=True, push=True):
        """fault: None, or ("locked", leg) / ("passive", leg) to train the optional 3-leg reference."""
        self.N = num_envs
        self.rng = np.random.default_rng(seed)
        self.dr, self.obs_noise, self.push = dr, obs_noise, push
        self.fault = fault
        self.base_model = mujoco.MjModel.from_xml_path(TRAIN_XML)
        self.max_len = int(ep_len_s / DT)
        self.cmd_every = int(cmd_resample_s / DT)
        self.rc = RewardCfg()
        self._build_model_pool(n_models)
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mujoco.MjData(self.base_model) for _ in range(nthread)]
        self.nstate = mujoco.mj_stateSize(self.base_model, STATE_SPEC)

        N = self.N
        self.state = np.zeros((N, self.nstate))
        self.model_idx = np.zeros(N, int)
        self.phase = np.zeros(N)
        self.cmd = np.zeros((N, 3))
        self.action = np.zeros((N, 12))
        self.last_action = np.zeros((N, 12))
        self.last_dq = np.zeros((N, 12))
        self.ep_len = np.zeros(N, int)
        self.push_vel = np.zeros((N, 3))
        self.ep_ret = np.zeros(N)
        self.reset_all()

    # ------------------------------------------------------------------ domain randomisation
    def _build_model_pool(self, n):
        m0 = self.base_model
        self.models, self.dr_params = [], []
        base_id = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        fl_act = [mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.fault[1]}_{j}") for j in ("HAA", "HFE", "KFE")] if self.fault else []
        fl_eq = [mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_EQUALITY, f"lock_{self.fault[1]}_{j}") for j in ("HAA", "HFE", "KFE")] if self.fault else []
        for i in range(n):
            m = copy.copy(m0)
            r = self.rng
            fr = r.uniform(0.5, 1.25) if self.dr else 1.0
            dm = r.uniform(-0.2, 0.5) if self.dr else 0.0
            kps = r.uniform(0.85, 1.15) if self.dr else 1.0
            kds = r.uniform(0.85, 1.15) if self.dr else 1.0
            tms = r.uniform(0.9, 1.1) if self.dr else 1.0
            com = r.uniform(-0.015, 0.015, 3) if self.dr else np.zeros(3)
            m.geom_friction[:, 0] = fr
            m.body_mass[base_id] = m0.body_mass[base_id] + dm
            m.body_ipos[base_id] = m0.body_ipos[base_id] + com
            m.actuator_gainprm[:, 0] = 4.0 * kps
            m.actuator_biasprm[:, 1] = -4.0 * kps
            m.actuator_biasprm[:, 2] = -0.2 * kds
            m.actuator_forcerange[:] = m0.actuator_forcerange * tms
            if self.fault is not None:
                kind, leg = self.fault
                for a in fl_act:
                    m.actuator_gainprm[a, 0] = 0.0; m.actuator_biasprm[a, 1:3] = 0.0
                if kind == "locked":
                    # lock = joint range of +-0.005 rad around the tucked pose (mujoco.rollout does not carry
                    # data.eq_active, so the equality locks of the harness cannot be used here)
                    for jn, q in zip(("HAA", "HFE", "KFE"), TUCKED_FL):
                        jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{jn}")
                        m.jnt_range[jid] = [q - 0.005, q + 0.005]
            self.models.append(m)
            self.dr_params.append([fr, dm, kps, kds])
        self.dr_params = np.array(self.dr_params)

    # ------------------------------------------------------------------ commands / resets
    def _sample_cmd(self, ids):
        n = len(ids)
        r = self.rng
        c = np.stack([r.uniform(-0.5, 0.8, n), r.uniform(-0.4, 0.4, n), r.uniform(-1.0, 1.0, n)], 1)
        mode = r.uniform(size=n)
        c[mode < 0.12] = 0.0                                   # stand
        fwd = (mode >= 0.12) & (mode < 0.35)
        c[fwd, 1:] = 0.0; c[fwd, 0] = r.uniform(0.1, 0.8, fwd.sum())   # straight forward
        turn = (mode >= 0.35) & (mode < 0.45)
        c[turn, :2] = 0.0                                      # turn in place
        c[np.linalg.norm(c, axis=1) < STAND_CMD_NORM] = 0.0
        self.cmd[ids] = c

    def reset_idx(self, ids):
        if len(ids) == 0:
            return
        n = len(ids)
        r = self.rng
        st = np.zeros((n, self.nstate))
        qpos = np.zeros((n, NQ)); qvel = np.zeros((n, NV))
        qpos[:, 2] = 0.235
        qpos[:, 3:7] = yaw_quat(r.uniform(-np.pi, np.pi, n))
        qpos[:, 7:] = DEFAULT_Q + r.uniform(-0.1, 0.1, (n, 12))
        if self.fault is not None and self.fault[0] == "locked":
            qpos[:, 7:10] = TUCKED_FL
        qvel[:, :2] = r.uniform(-0.2, 0.2, (n, 2))
        qvel[:, 3:6] = r.uniform(-0.2, 0.2, (n, 3))
        st[:, 1:1 + NQ] = qpos
        st[:, 1 + NQ:1 + NQ + NV] = qvel
        self.state[ids] = st
        self.model_idx[ids] = r.integers(0, len(self.models), n)
        self.phase[ids] = r.uniform(0, 1, n)
        self.action[ids] = 0.0
        self.last_action[ids] = 0.0
        self.last_dq[ids] = 0.0
        self.ep_len[ids] = 0
        self.ep_ret[ids] = 0.0
        self._sample_cmd(ids)

    def reset_all(self):
        self.reset_idx(np.arange(self.N))
        self.sd = np.zeros((self.N, 51))
        # one zero-action step to fill sensors
        obs, priv, *_ = self.step(np.zeros((self.N, 12)), _skip_rew=True)
        return obs, priv

    # ------------------------------------------------------------------ main step
    def step(self, actions, _skip_rew=False):
        N = self.N
        a = np.clip(np.asarray(actions, np.float64), -ACTION_CLIP, ACTION_CLIP)
        self.last_action[:] = self.action
        self.action[:] = a
        ctrl = (DEFAULT_Q + ACTION_SCALE * a)[:, None, :]
        models = [self.models[i] for i in self.model_idx]
        S, SD = self.pool.rollout(models, self.datas, self.state, ctrl, nstep=DECIMATION)
        self.state = S[:, -1].copy()
        sd = SD[:, -1]
        touch = SD[:, :, SD_FOOT_TOUCH].max(1)
        knee = SD[:, :, SD_KNEE_TOUCH].max(1)
        base_touch = SD[:, :, SD_BASE_TOUCH].max(1)
        tau = SD[:, :, SD_TAU].mean(1)

        qpos = self.state[:, 1:1 + NQ]; qvel = self.state[:, 1 + NQ:1 + NQ + NV]
        R = quat_to_mat(qpos[:, 3:7])
        Rt = np.transpose(R, (0, 2, 1))
        v_body = np.einsum("nij,nj->ni", Rt, qvel[:, :3])
        w_body = qvel[:, 3:6]
        grav = -R[:, 2, :]                      # R^T (0,0,-1)
        q = qpos[:, 7:]; dq = qvel[:, 6:]
        foot_pos = sd[:, SD_FOOT_POS].reshape(N, 4, 3)
        foot_vel = sd[:, SD_FOOT_VEL].reshape(N, 4, 3)
        contact = touch > 1.0

        self.phase = (self.phase + DT * GAIT_FREQ) % 1.0
        standing = np.linalg.norm(self.cmd, axis=1) < STAND_CMD_NORM
        leg_phase = (self.phase[:, None] + LEG_PHASE_OFFSET) % 1.0
        swing = (leg_phase < 0.5) & ~standing[:, None]

        rew = np.zeros(N)
        if not _skip_rew:
            rc = self.rc
            terms = {}
            err_lin = np.sum((self.cmd[:, :2] - v_body[:, :2]) ** 2, 1)
            terms["track_lin"] = rc.track_lin * np.exp(-err_lin / rc.lin_sigma)
            terms["track_ang"] = rc.track_ang * np.exp(-(self.cmd[:, 2] - w_body[:, 2]) ** 2 / rc.ang_sigma)
            terms["lin_vel_z"] = rc.lin_vel_z * v_body[:, 2] ** 2
            terms["ang_vel_xy"] = rc.ang_vel_xy * np.sum(w_body[:, :2] ** 2, 1)
            terms["orientation"] = rc.orientation * np.sum(grav[:, :2] ** 2, 1)
            terms["base_height"] = rc.base_height * (qpos[:, 2] - rc.base_height_target) ** 2
            terms["torques"] = rc.torques * np.sum(tau ** 2, 1)
            terms["dof_acc"] = rc.dof_acc * np.sum(((dq - self.last_dq) / DT) ** 2, 1)
            terms["action_rate"] = rc.action_rate * np.sum((self.action - self.last_action) ** 2, 1)
            terms["collision"] = rc.collision * (np.sum(knee > 0.1, 1) + (base_touch > 0.1))
            match = np.where(swing, ~contact, contact)
            terms["gait_contact"] = rc.gait_contact * match.mean(1)
            prog = np.clip(leg_phase / 0.5, 0, 1)
            z_des = 0.016 + rc.swing_peak * np.sin(np.pi * prog)
            terms["swing_height"] = rc.swing_height * np.sum(swing * (foot_pos[:, :, 2] - z_des) ** 2, 1)
            terms["foot_slip"] = rc.foot_slip * np.sum(contact * np.sum(foot_vel[:, :, :2] ** 2, 2), 1)
            terms["stand_still"] = rc.stand_still * standing * (np.sum(np.abs(q - DEFAULT_Q), 1) + 0.05 * np.sum(np.abs(dq), 1))
            lo = np.array([-1.0, -2.6, -3.0] * 4) * 0.9; hi = -lo
            terms["dof_limits"] = rc.dof_limits * np.sum(np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None), 1)
            terms["hip_pos"] = rc.hip_pos * np.sum((q[:, [0, 3, 6, 9]] - DEFAULT_Q[[0, 3, 6, 9]]) ** 2, 1)
            if self.fault is not None:
                # three-leg fine-tuning (optional TRAINED reference): no imposed trot schedule (the policy
                # may find its own three-leg gait), generic foot clearance instead, and a penalty for
                # letting the broken leg touch the ground.
                fi = LEGS.index(self.fault[1])
                m4 = np.ones(4, bool); m4[fi] = False
                terms["gait_contact"] = np.zeros(N)
                swing_any = ~contact[:, m4]
                terms["swing_height"] = 0.5 * np.sum(swing_any * np.clip(foot_pos[:, m4, 2] - 0.016, 0, 0.05), 1)
                terms["broken_leg_contact"] = -2.0 * ((touch[:, fi] > 1.0) | (knee[:, fi] > 0.1))
            for v in terms.values():
                rew += v
            rew *= DT
            self.last_terms = {k: float(v.mean()) for k, v in terms.items()}

        self.last_dq = dq.copy()
        self.ep_len += 1

        # termination
        fell = (base_touch > 0.1) | (grav[:, 2] > -0.5) | (qpos[:, 2] < 0.10)
        timeout = self.ep_len >= self.max_len
        if not _skip_rew:
            rew += fell * self.rc.termination
        done = fell | timeout
        self.ep_ret += rew

        # privileged critic info (before reset)
        priv = np.concatenate([v_body * 2.0, qpos[:, 2:3] * 5.0 - 1.0, contact.astype(float),
                               foot_pos[:, :, 2] * 10.0, (self.dr_params[self.model_idx] - [0.875, 0.15, 1.0, 1.0]) * 2.0,
                               self.push_vel], 1)

        info = {"timeout": timeout & ~fell}
        if done.any():
            ids = np.where(done)[0]
            info["ep_ret"] = self.ep_ret[ids].copy()
            info["ep_len"] = self.ep_len[ids].copy()
            info["fell"] = fell[ids].copy()
        # command resampling and pushes
        resample = (self.ep_len % self.cmd_every == 0) & ~done
        self._sample_cmd(np.where(resample)[0])
        self.push_vel[:] = 0.0
        if self.push:
            pids = np.where((self.rng.uniform(size=N) < DT / 6.0) & ~done)[0]   # about one push every 6 s
            if len(pids):
                dv = self.rng.uniform(-0.4, 0.4, (len(pids), 2))
                self.state[pids, 1 + NQ:3 + NQ] += dv
                self.push_vel[pids, :2] = dv
        self.reset_idx(np.where(done)[0])
        if done.any():
            # reset envs: observation from the reset state (no step taken yet)
            ids = np.where(done)[0]
            qpos2 = self.state[ids, 1:1 + NQ]; qvel2 = self.state[ids, 1 + NQ:1 + NQ + NV]
            R2 = quat_to_mat(qpos2[:, 3:7])
            w_body[ids] = qvel2[:, 3:6]; grav[ids] = -R2[:, 2, :]
            q = q.copy(); dq = dq.copy()
            q[ids] = qpos2[:, 7:]; dq[ids] = qvel2[:, 6:]
            priv[ids, 0:3] = np.einsum("nji,nj->ni", R2, qvel2[:, :3]) * 2.0
            priv[ids, 3] = qpos2[:, 2] * 5.0 - 1.0
            priv[ids, 4:8] = 0.0
            priv[ids, 8:12] = 0.3
            priv[ids, 12:16] = (self.dr_params[self.model_idx[ids]] - [0.875, 0.15, 1.0, 1.0]) * 2.0
        obs = self.compute_obs(w_body, grav, q, dq)
        return obs, priv.astype(np.float32), rew, done, info

    def compute_obs(self, w_body, grav, q, dq):
        N = self.N
        standing = np.linalg.norm(self.cmd, axis=1) < STAND_CMD_NORM
        clock = np.stack([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], 1)
        clock[standing] = 0.0
        obs = np.concatenate([w_body * SCALE_ANG_VEL, grav, self.cmd * SCALE_CMD, (q - DEFAULT_Q) * SCALE_DOF_POS,
                              dq * SCALE_DOF_VEL, self.action, clock], 1)
        self.obs_clean = obs.copy()
        if self.obs_noise:
            r = self.rng
            noise = np.concatenate([r.uniform(-0.2, 0.2, (N, 3)) * SCALE_ANG_VEL, r.uniform(-0.05, 0.05, (N, 3)),
                                    np.zeros((N, 3)), r.uniform(-0.01, 0.01, (N, 12)),
                                    r.uniform(-1.5, 1.5, (N, 12)) * SCALE_DOF_VEL, np.zeros((N, 14))], 1)
            obs = obs + noise
        return obs.astype(np.float32)

    def close(self):
        self.pool.close()
