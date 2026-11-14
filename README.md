<p align="center"><img src="docs/assets/pony.svg" width="128" alt="Red rocking horse, the mark of Project R.E.D. Pony"></p>

# Project R.E.D. Pony

An LLM edits trained robot policies by hand: no retraining, same network, same file.
**Website: https://antobuo-dev.github.io/red-pony-llm-editing-net/**

| example | robot | edit | result |
|---|---|---|---|
| [`examples/h1_heading_hold`](examples/h1_heading_hold) | Unitree H1_2 | heading (and lateral) hold, memory in action channels | yaw drift 2.84 → 0.017 °/s; runs on the robot |
| [`examples/drone_yaw`](examples/drone_yaw) | quadrotor | yaw tracking on a network that could not yaw | yaw error 0.26–0.50°; flies on the drone |
| [`experiments/solo_three_legs`](experiments/solo_three_legs) | SOLO-12 | walking with a locked leg | goals 0 → 17 of 30 (simulation) |

Numbers are from simulation; a real-robot campaign is under way.

## Rules

- No gradient steps on the edited network; the original is never modified.
- Same layers and ONNX format: the deploy code does not change.
- A builder script per edit (original in, edited out) and a `weight_edit.json` declaring every
  change and every fitted parameter.

Operators, guarantees and procedure: [`METHOD.md`](METHOD.md).

## Run

```bash
uv sync                                         # Python 3.12, torch (CUDA 12.8), mujoco, onnx
python -m http.server 8765 --directory docs     # the website, locally
```

## Credits

Apache-2.0. H1_2 policy and robot model:
[H1_2_Squat_Locomotion_RL](https://github.com/gaetanotorella/H1_2_Squat_Locomotion_RL) (Gaetano
Torella, University of Naples Federico II, PRISMA Lab). SOLO-12: Open Dynamic Robot Initiative,
licence next to its assets. Edits made with Claude.
