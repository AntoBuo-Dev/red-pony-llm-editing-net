# Heading + lateral hold policy (2026-11-11, weights edited by hand)

Edited from `policies/h1_2_squat/2026-09-13_21-27-39`: the heading hold of
`policies/h1_2_squat/2026-11-11_heading-hold` plus a lateral hold. Same architecture and
`deploy.yaml`.

**Tested only in the Python MuJoCo replica of the controller (`evaluation/heading_hold/`), not
with the C++ simulator/controller and not on the robot.** The folder name sorts last in
`policies/h1_2_squat`, so the controller loads this policy by default; delete or rename the folder
to fall back to the heading-hold policy, and that one too for the original. The lateral hold relies
on a velocity readout fitted in that simulation: if the readout is wrong on the robot, the loop can
add up to ±0.3 m/s of lateral command while walking forward.

## What was changed

23 / 13 / 8 hidden units were freed (least-squares fold, listed in `weight_edit.json`) and rewritten:

```text
rate   = -(w_y g_y + w_z g_z)(1 + g_x^2)       yaw rate from gyro and projected gravity
act    = 1 if the gait clock runs and |wz command| ~ 0, else 0
theta' = clamp(theta + 0.02 (wz - rate), ±0.35 act)   heading error [rad]         action 25
I'     = clamp(I + 0.02 theta, ±0.30 act)             its integral [rad s]        action 26
wz seen by the original network = wz + 3 theta + 3 I

v_hat  = w · h1 + b                            lateral velocity, linear readout of hidden layer 1
act_l  = act and |vy command| ~ 0
p'     = clamp(p + 0.02 v_hat, ±0.15 act_l)           lateral displacement [m]    action 21
vy seen by the original network = vy - 2 p
```

The registers are stored in action channels and read back through `last_action`:
`left_wrist_yaw_joint` = 2·theta, `right_wrist_yaw_joint` = 2·I, `left_wrist_roll_joint` = 4·p.
**These three wrist joints no longer follow the original policy** (they move by at most 0.175,
0.15 and 0.15 rad).

`v_hat` estimates the base velocity perpendicular to the reference heading. Its weights are a ridge
regression on the 489 remaining units of the first hidden layer, fitted on closed-loop rollouts in
the Python replica (5 rounds, 3128 steady conditions; held-out error of the per-condition mean:
0.25 cm/s rms, 2.1 cm/s max). This is the only part of the edit fitted to data.

The lateral hold works only for forward/backward walking (vy command at zero). Sideways walking and
turning use the heading hold alone or the original behaviour.

## Results (Python replica)

Drift perpendicular to the heading at the start of the command, 60 commands (vx −0.3 and 0.2…1.0 m/s
at 0.98 / 0.90 / 0.85 / 0.80 / 0.74 / 0.71 m), 30 s each:

| | original | heading hold | this policy |
|---|---|---|---|
| mean absolute lateral drift | 31.2 cm/s | 2.13 cm/s | 0.32 cm/s |
| worst lateral drift | 84.8 cm/s | 8.70 cm/s | 2.74 cm/s |
| mean of the largest lateral deviation in 30 s | 7.38 m | 0.63 m | 0.10 m |
| worst lateral deviation in 30 s | 20.3 m | 2.25 m | 0.62 m |
| falls | 0 | 0 | 0 |

The worst case is 1.0 m/s at 0.98 m, where the heading itself still drifts by 0.08 °/s.
`standard_v1` and `standard_v3` timelines: no fall; velocity gains, turning rates and heights as the
original. Yaw kicks of 1–1.5 rad/s while walking and floor friction 0.5 / 1.5: no fall, heading
back within about 2°.

## Limits

- The readout is fitted to MuJoCo and to this model; on the robot its bias becomes the lateral drift.
- A gyro bias `b` still turns the robot at `b`.

## Reproduce

```bash
.venv/bin/python evaluation/heading_hold/build_lateral_policy.py policies/h1_2_squat/2026-09-13_21-27-39 /tmp/h1_2_lateral
python evaluation/heading_hold/export_torchscript.py policies/h1_2_squat/2026-09-13_21-27-39/exported/policy.pt /tmp/h1_2_lateral   # needs torch
.venv/bin/python evaluation/heading_hold/evaluate.py policies/h1_2_squat/2026-11-11_heading-lateral-hold
```
