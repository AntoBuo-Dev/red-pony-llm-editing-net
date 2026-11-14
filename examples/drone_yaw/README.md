# Example 2 — Quadrotor hover network: yaw tracking written into the weights

**Source policy.** A hover policy for a quadrotor (AC-MPC project, `ACMPC-TRO`): an MLP
17 → 512 → 512 → 4 with ELU, run every 20 ms, no memory.

- Inputs (17): velocity (3), attitude matrix (9), position error (3), sin and cos of the yaw error (2),
  each normalised by `x̂ = (x − μ) / (σ + 0.01)`.
- Outputs (4): collective thrust `T = m·sat(49.38 r₀ + 9.8066, 0, 59.19)` and body rates, with
  `ω_z = 4·sat(r₃, −1, 1)`; mass 0.988 kg, hover thrust ≈ 9.69 N.
- In training the yaw term of the reward had weight zero.

**Defect.** The network holds position within 2.7 cm, but it cannot turn: whatever the yaw
reference, it brings the nose to about −50° and stays there.

**Edit.** No training. About 21 000 of 272 896 weights were rewritten (3660 + 16 756 + 527),
by hand or by solving a square linear system. Same shape, same file format.

| mission (full simulator) | original | edited |
|---|---|---|
| hold: position error | 2.7 cm | **0.6 cm** (0.6 cm with +30 ms delay) |
| hold: thrust oscillation | 0.01–0.04 N | 0.013 N (0.033 N with delay) |
| yaw: six references 0°, 45°, 90°, −90°, 170°, 0° | stays at ≈ −50° | **error 0.26–0.50°** for jumps up to 100° |
| yaw: steady position error during the mission | — | ≤ 1.6 cm |
| cube (3 m sideways, 1.8 m up): max tilt | 44° / 47° | 39° / 42° |
| missions completed | — | 18 of 20 flights (2 stopped by radio-link loss, cause not isolated) |

The numbers come from a fast Python simulator and a full simulator (motors, ground, flight-controller
firmware), without noise. The edited network also flies on the real drone; a systematic flight
campaign is under way to build a larger pool of flights.

## The path, including the failed attempts

The work is told in the order it happened, because each step came from what the previous one showed.

1. **Read the file before running it.** The weight columns of sin/cos of the yaw error have norm
   0.62 / 0.59, a third of the weakest other input: the network barely reads the yaw error (coherent
   with a zero yaw reward). But the normaliser remembers a mean heading: `μ(cos ψ) = 0.637`,
   `μ(sin ψ) = −0.470`, an arrow of length 0.79 towards −36°. The visited states were not uniform in
   yaw.
2. **Probe at zero error.** Drone still, level, zero error, sweep only the yaw (8 points): `ω_z` changes
   a lot, with an attracting rest point at about −50° and a repelling one at about +140°. Thrust is
   ≈ 10 N (the weight) only near −50°, 0–1.4 N elsewhere. The network does not ignore yaw: it has
   built its own operating point.
3. **Broken, or just shifted?** Freeze yaw, command zero yaw rate, try 24 headings: the drone stops
   everywhere, without oscillating, but up to 1.43 m lower. At every yaw there is a stable controller
   with the wrong operating point. And the network brakes a 0.5 m/s forward motion at every yaw: it has
   learned to rotate world-frame velocity into the body frame — a product that cannot be written by
   hand in the first layer. Without it the edit would probably have been impossible.
4. **Is there room?** Usage of a unit = std of its activation over 200 000 states × norm of its outgoing
   weights. Below 10⁻³: **413 of 512** units are idle in layer 1 and **490 of 512** in layer 2. An idle
   unit is freed by removing its connection and adding `w·c` (weight × its constant value) to the
   next bias: the output does not change.
5. **Easy part: the yaw command.** Disconnect the old yaw-error columns (replace them by their
   zero-error value in the bias) and write a new path sin(error) → one unit → one unit → `ω_z` row:
   `ω_z = 0.8·lim(1.25 sin e + 0.6 (h⁺ − h⁻))`, `h± = elu(20(±sin e − 2(1 + cos e)))`. The two `h`
   units make a steep step near ±180°, where `sin e = 0` would otherwise leave a dead point
   (a 157° error stayed put; with the step, about 3°).
6. **Failed: add the missing thrust at the output.** Adding, at each yaw, what the output lacks fixes the
   value but not the gain: at +90° the thrust reacts to height with 0.8 N/m instead of 61 N/m and the
   flight diverges. A unit pushed onto the flat side of the ELU keeps the right value but loses its
   slope. **The correction has to be applied before the last nonlinearity.**
7. **Neurons that measure yaw.** The attitude matrix already contains cos ψ and sin ψ. A first-layer
   unit with weights on them computes `cos(ψ − ψᵢ)`; subtracting `cos 50°`, scaling by 20 and passing
   through ELU gives a bump: `φᵢ = elu(20 (c̃ cos ψᵢ + s̃ sin ψᵢ − cos 50°))`. **23 freed units become
   yaw sensors**, one every 15.7°.
8. **Exact shift before the fold.** Require `p₂(x₀(ψ)) + Δ(ψ) = p₂(x*)` for every ψ: at zero error the
   second-layer pre-activations must equal those of the reference equilibrium at −50°.
   `Δ(ψ) = Cᵀ φ(ψ)`, with `Φ C = D` — 23 equations, 23 unknowns, no fit, no optimisation (condition
   number 111). The offset disappears.
9. **Failed again: it oscillates with delay.** With 40 ms of delay the thrust oscillates (> 0.5 N at 15
   of 24 headings). The shift fixed where the equilibrium is, not how strongly the network reacts: far
   from −50° the vertical gain triples (up to 46 N/(m/s), above the `Gτ/m = π/2` limit of a delayed
   loop) and the lateral response flips sign.
10. **Two branches, two cures.** *Freeze* the 277 first-layer units that care little about horizontal
    velocity: they read the tilt from the third row of the attitude matrix as if the nose were at −50°.
    *Shift selectively* only the 215 second-layer units that weigh on thrust. Each cure alone fixes one
    branch and spoils the other; together: error below 2.1 cm at every yaw, thrust steady up to 40 ms.
    Three units remove the remaining output residuals (now harmless: units are on the right side).
11. **First full-simulator flight: takeoff fails.** 44 N at start, climbs at 2 m/s, cuts thrust,
    ground at 2.2 s. The removed offset was also braking the climb. A limiter of two units on the
    height error is written in: `ΔT = −(g_c/B_c)[elu(B_c(e_z − 0.35)) − …]`, 20 N/m.
12. **False alarm.** The next flight aborts at 1.76 s for "altitude above 4.3 m". The log shows a jump
    from 0.63 to 1.47 m in 20 ms (42 m/s, impossible): a second simulator on the same network was
    publishing the same topic. Separating the channels fixed it. Physical plausibility checks on
    logs avoided "fixing" a defect that did not exist.
13. **The cube falls 6 times out of 6.** At a 1.8 m height error the limiter removed 29 N, three times
    the hover thrust. One more unit gives the limiter a ceiling above 1.45 m (max 22 N). The cube
    passes with 1 cm steady error; all flights were repeated with the final file.

## The final recipe (8 moves)

1. Disconnect the yaw error from the old network.
2. Retire 28 idle units in layer 1 and 5 in layer 2.
3. Freeze yaw in 277 layer-1 units (they see the nose at −50°).
4. Let the network settle at −50° and read its true equilibrium (nothing written).
5. Write 23 yaw-sensor bumps in layer 1; exact shift on 215 layer-2 units.
6. Three units cancel the output residuals.
7. New yaw command: follows sin(error), limit 0.8 rad/s, steep step near ±180°.
8. Climb limiter: two units on the height error, with a ceiling (max 22 N).

The saved file reproduces the computed network within 3.6·10⁻⁴ on the outputs; the original file is
untouched.

## Lessons

- A network trained with no pressure on yaw can still depend strongly on yaw: zero training error
  says nothing about what it does elsewhere.
- Fixing a value after the nonlinearity is not fixing the controller: gains live in which side of the
  fold each unit sits on.
- Every behaviour you remove (here: the offset that braked the climb) must be replaced or checked.
- A fast simulator only finds the defects it contains. Bound every hand-written correction.

## Limits

The edited network flies on the real drone, but the pool of real flights is still small: the
systematic campaign is collecting more before we publish flight statistics. The limiter is tuned on a 9.7 N hover thrust and the
simulator's motor curve. The shift is exact only level and at zero error; freezing holds for small
tilts. Thresholds, number and shape of the bumps were chosen by hand with few attempts. Flights are
nearly deterministic: 3 repetitions measure repeatability, not robustness.

The code of this example lives in the AC-MPC research repository and is not published here; this
folder documents the method and the measured numbers.
