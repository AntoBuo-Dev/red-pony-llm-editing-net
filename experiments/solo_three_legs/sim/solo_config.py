"""Constants shared by the training environment (the deploy harness keeps its own copy in
policies/*/config.yaml, written by export_policy.py)."""
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_XML = os.path.join(ROOT, "assets", "solo12", "solo12.xml")
TRAIN_XML = os.path.join(ROOT, "assets", "solo12", "solo12_train.xml")

LEGS = ["FL", "FR", "HL", "HR"]
JOINTS = [f"{l}_{j}" for l in LEGS for j in ("HAA", "HFE", "KFE")]
# example-robot-data "standing" pose
DEFAULT_Q = np.array([0.1, 0.8, -1.6, -0.1, 0.8, -1.6, 0.1, -0.8, 1.6, -0.1, -0.8, 1.6])

SIM_DT = 0.002
DECIMATION = 10
DT = SIM_DT * DECIMATION              # policy runs at 50 Hz
ACTION_SCALE = 0.3                    # q_target = DEFAULT_Q + ACTION_SCALE * action
ACTION_CLIP = 6.0

# observation layout (47 numbers, all from on-board sensors / controller state)
OBS_SLICES = dict(ang_vel=slice(0, 3), gravity=slice(3, 6), cmd=slice(6, 9), dof_pos=slice(9, 21),
                  dof_vel=slice(21, 33), last_action=slice(33, 45), clock=slice(45, 47))
OBS_DIM = 47
SCALE_ANG_VEL = 0.25
SCALE_CMD = np.array([2.0, 2.0, 0.25])
SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05

# gait clock: phase advances at GAIT_FREQ; leg i swings while (phase + offset_i) mod 1 < 0.5.
# FL+HR and FR+HL are the two diagonal pairs of a trot.
GAIT_FREQ = 2.5
LEG_PHASE_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])
STAND_CMD_NORM = 0.1                  # |command| below this -> clock (0, 0) and standing still

# broken-leg poses (used by the harness / optional 3-leg training)
TUCKED_FL = np.array([0.0, 1.6, -2.9])  # thigh horizontal pointing backwards, shin folded under it: lowest point ~5 cm below the hip
