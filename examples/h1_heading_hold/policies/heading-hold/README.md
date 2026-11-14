# Heading-hold policy (2026-11-11, weights edited by hand, no training)

Edited from `../2026-09-13_21-27-39`. Same architecture (465 → 512 → 256 → 128 → 27, ELU), same
`deploy.yaml`. With a walking command and the turn stick at zero, the network now holds its heading.

**Tested only in the Python MuJoCo replica of the controller (`evaluation/heading_hold/`), not
with the C++ simulator/controller and not on the robot.** `../2026-11-11_heading-lateral-hold` sorts after this folder and is the one the controller loads
by default; this policy is loaded only if that folder is removed or renamed.

## What was changed

1. **Units freed.** No hidden unit of the original network is unused. The 20 / 9 / 5 most redundant
   units of the three hidden layers (listed in `weight_edit.json`) were folded by least squares into
   the other units of their layer, and their connections set to zero. Action error of this step on
   held-out samples: rms 0.005, max 0.15 (action units; the actions have std 0.06–1.85).
2. **Circuit written into those units:**

   ```text
   rate   = -(w_y g_y + w_z g_z)(1 + g_x^2)       yaw rate from gyro and projected gravity
   act    = 1 if the gait clock runs and |wz command| ~ 0 (fades out up to ~0.03 rad/s), else 0
   theta' = clamp(theta + 0.02 (wz - rate), ±0.35 act)   heading error [rad]
   I'     = clamp(I + 0.02 theta, ±0.30 act)             its integral [rad s]
   wz seen by the original network = wz + 3 theta + 3 I
   ```

3. **Memory.** The network has no state, so `theta` and `I` are stored in two action channels and
   read back through `last_action`: action 25 (`left_wrist_yaw_joint`) = 2·theta, action 26
   (`right_wrist_yaw_joint`) = 2·I. **The two wrist-yaw joints no longer follow the original policy:**
   they sit at 0 and rotate by 0.5·theta and 0.5·I rad (at most 0.175 and 0.15 rad; below 0.09 rad
   in the tests).

Turning the stick releases the hold and clears both registers; the heading at the moment the stick
returns to zero becomes the new reference. Standing (clock frozen) and heights below the 0.70 m
cutoff are unaffected.

## Results (Python replica, 30 s per command, 28 commands at 0.98 / 0.85 / 0.74 / 0.71 m)

| | original | this policy |
|---|---|---|
| mean absolute yaw rate | 2.84 °/s | 0.017 °/s |
| mean absolute yaw after 30 s | 81.8° | 1.05° |
| rms distance of the path from a straight line | 34.8 cm | 1.2 cm |
| mean action change per step (legs, torso) | 0.087 | 0.090 |
| falls | 0 | 0 |

`standard_v1` and `standard_v3` timelines: no fall, velocity gains, turning rates and heights within
run-to-run differences of the original.

## Limits

- The path is straight, but **not exactly along the starting heading**: a lateral velocity of about 2 cm/s on average (up to 9)
  (direction and size depend on speed and height) remains. The observations contain no velocity, so
  the circuit cannot see it. `../2026-11-11_heading-lateral-hold` adds a lateral
  hold based on a velocity readout fitted in simulation.
- The heading is the integral of the gyro: a gyro bias `b` turns the robot at `b` (0.01 rad/s of
  bias gave 0.6 °/s). The simulated gyro has none.
- `model_4999.pt` is not here: the training checkpoint still holds the original network.

## Reproduce

```bash
.venv/bin/pip install mujoco onnx onnxruntime numpy pyyaml
.venv/bin/python evaluation/heading_hold/build_policy.py policies/h1_2_squat/2026-09-13_21-27-39 /tmp/h1_2_heading_hold
python evaluation/heading_hold/export_torchscript.py policies/h1_2_squat/2026-09-13_21-27-39/exported/policy.pt /tmp/h1_2_heading_hold   # needs torch
.venv/bin/python evaluation/heading_hold/evaluate.py policies/h1_2_squat/2026-09-13_21-27-39 policies/h1_2_squat/2026-11-11_heading-hold
```
