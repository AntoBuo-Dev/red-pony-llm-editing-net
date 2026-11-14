"""Foot position of each SOLO-12 leg in the base frame as a function of its 3 joint angles (from the
MJCF via MuJoCo kinematics), numerical Jacobian and inverse kinematics."""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "assets", "solo12", "solo12.xml")
LEGS = ["FL", "FR", "HL", "HR"]
_m = mujoco.MjModel.from_xml_path(XML)
_d = mujoco.MjData(_m)


def foot_pos(leg, q3):
    """foot centre in the base frame for joint angles q3 of `leg` (other legs irrelevant)."""
    li = LEGS.index(leg)
    _d.qpos[:] = 0; _d.qpos[3] = 1.0
    _d.qpos[7 + 3 * li: 10 + 3 * li] = q3
    mujoco.mj_kinematics(_m, _d)
    return _d.site(f"{leg}_foot").xpos.copy()


def jac(leg, q3, eps=1e-5):
    J = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3); dq[k] = eps
        J[:, k] = (foot_pos(leg, q3 + dq) - foot_pos(leg, q3 - dq)) / (2 * eps)
    return J


def ik(leg, p_des, q_init, iters=50):
    q = np.array(q_init, float)
    for _ in range(iters):
        e = p_des - foot_pos(leg, q)
        if np.linalg.norm(e) < 1e-7:
            break
        q += np.linalg.solve(jac(leg, q) + 1e-9 * np.eye(3), e)
    return q
