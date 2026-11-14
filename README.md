<p align="center"><img src="docs/assets/pony.svg" width="128" alt="Red rocking horse, the mark of Project R.E.D. Pony"></p>

# Project R.E.D. Pony — LLM editing of trained networks

**Website: https://antobuo-dev.github.io/red-pony-llm-editing-net/**

A trained robot policy is a stack of tables of numbers. In this project an LLM agent reads them,
finds the room inside the network, and writes a new behaviour straight into the weights:
**no retraining, same network shape, same file format**. Each example is told in the order it
happened, failed attempts included.

The edited H1_2 and drone networks run on the real robots. We are now running a **systematic
campaign of real-robot experiments** to build a larger pool of trials; the numbers published so far
come from simulation, where every trial can be repeated and measured.

| example | robot | network | what was written into the weights | status |
|---|---|---|---|---|
| [`examples/h1_heading_hold`](examples/h1_heading_hold) | Unitree H1_2 humanoid | MLP 465 → 512 → 256 → 128 → 27, ELU | heading hold (and lateral hold) with memory stored in action channels | runs on the real robot; numbers from the Python/MuJoCo replica |
| [`examples/drone_yaw`](examples/drone_yaw) | quadrotor | MLP 17 → 512 → 512 → 4, ELU | yaw tracking on a hover network that could not yaw | flies on the real drone; numbers from a fast and a full simulator |
| [`experiments/solo_three_legs`](experiments/solo_three_legs) | SOLO-12 quadruped | MLP, ELU | walking on three legs after one leg is disabled | in progress |

## Rules of the game

- No gradient descent or RL on the edited network. The original file is never modified.
- Same layer sizes. The edited policy is a normal ONNX MLP: the deploy code does not change.
- Allowed: reading weights and activations, probing in simulation, folding idle or redundant units
  away (least squares), writing circuits with ELU units, solving linear systems. Anything fitted to
  data is declared as such.
- Every edit is a builder script (original in → edited out) plus a `weight_edit.json` with the
  changed units, the parameters and the sha256 of both files.

## The editing loop

1. **Read** the weights and the normaliser before running anything.
2. **Probe** the network on controlled states: sweep one variable, freeze the rest.
3. **Make room**: measure unit usage, fold idle units into the next bias.
4. **Write** the missing behaviour with ELU units (sensors, integrators, clamps, gates), injected
   where the network already listens and *before* the nonlinearity.
5. **Test** in a fast simulator, then in the full one, against the original.
6. **Look at the failures**, check that they are real, bound every correction, repeat.

## Repository layout

```
docs/                       website (GitHub Pages), with the shared media in docs/assets/
examples/h1_heading_hold/   H1_2 edit: scripts, edited ONNX policies, weight_edit.json
examples/drone_yaw/         quadrotor yaw edit: method and measured results
experiments/solo_three_legs SOLO-12: training, leg failure, LLM edit (in progress)
pyproject.toml, uv.lock     Python environment (uv)
```

## Environment

```bash
uv sync                    # Python 3.12, torch (CUDA 12.8 wheels), mujoco, onnx, onnxruntime
source .venv/bin/activate
```

Preview the website locally:

```bash
python -m http.server 8765 --directory docs
```

## Limits

The edited H1_2 and drone networks work on the real robots, but the measured results published
here come from simulation (one model of each robot, no sensor noise). Real-robot statistics will be
published when the systematic campaign has a large enough pool of trials. How much of the original
networks' robustness survives has not been measured systematically yet. See the *Limits* section of
each example.

## Credits and license

Apache-2.0 (see `LICENSE`). The H1_2 source policy, robot description and controller conventions
come from [H1_2_Squat_Locomotion_RL](https://github.com/gaetanotorella/H1_2_Squat_Locomotion_RL)
(Gaetano Torella, University of Naples Federico II, PRISMA Lab), Apache-2.0. SOLO-12 is an
Open Dynamic Robot Initiative robot; its description is used under its own license (kept next to
the assets). The edits were designed and carried out with an LLM agent (Claude) and tested in
simulation and on the real robots.
