# The editing method

This document states precisely what an edit is in Project R.E.D. Pony, which operations are
allowed, what each one guarantees, and how an edit is built and validated. The three examples
(H1_2 heading hold, quadrotor yaw, SOLO-12 on three legs) are written in these terms at the end.

## 1. Setting

A policy is a feed-forward MLP `f : ℝⁿ → ℝᵐ` with `L` hidden layers and ELU activation σ:

```
x̂      = (x − μ) ⊘ (s + ε)                       normaliser (absorbed in W₀ when present)
p₁     = W₀ x̂ + b₀,        h₁ = σ(p₁)
p_{ℓ+1} = W_ℓ h_ℓ + b_ℓ,     h_{ℓ+1} = σ(p_{ℓ+1})       ℓ = 1 … L−1
a      = W_L h_L + b_L                             actions
σ(u)   = u for u > 0,   eᵘ − 1 for u ≤ 0          σ'(u) = 1 for u > 0,  eᵘ for u ≤ 0
```

`p_ℓ` are the **pre-activations** ("before the fold"), `h_ℓ` the activations. Unit `j` of hidden
layer `ℓ` has incoming weights `W_{ℓ−1}[j, :]`, bias `b_{ℓ−1}[j]` and outgoing weights `W_ℓ[:, j]`.

`𝒟` denotes a set of recorded observations: closed-loop rollouts of the original policy in the
nominal condition and, when relevant, in the target condition (with a prototype of the edit
running). Every quantity "measured on 𝒟" is computed on held-out samples of it.

**An edit** is a map `E : (W, b) ↦ (W′, b′)` such that

- **I1, same shape.** Every `W′_ℓ, b′_ℓ` has the shape of `W_ℓ, b_ℓ`; the deploy code does not change.
- **I2, no gradient on the result.** `f′ = f_{W′,b′}` is never trained. Parameters may come from
  closed-form solutions, linear systems, least-squares fits on 𝒟, or a search done by hand in
  simulation, and each of them is declared (§4).
- **I3, reproducible.** `E` is a deterministic builder script: original file in, edited file out,
  with the sha256 of both recorded in `weight_edit.json`.
- **I4, the original is untouched** and remains the baseline of every test.

## 2. Operators

Each edit is a composition of the operators below. For each, the change to the weights is exact;
the column "guarantee" says what is preserved and what has to be measured.

### O1 — Fold a redundant unit (least squares)

Fit unit `j` of layer `ℓ` on the other units of its layer over 𝒟:
`h_j ≈ c_j + Σ_{k≠j} α_k h_k` (ridge). Then

```
W_ℓ[:, k] += W_ℓ[:, j] α_k   (k ≠ j)        b_ℓ += W_ℓ[:, j] c_j
W_ℓ[:, j]  = 0,   W_{ℓ−1}[j, :] = 0,   b_{ℓ−1}[j] = 0          → unit j is free
```

*Guarantee:* `p_{ℓ+1}` changes by `W_ℓ[:, j] r_j`, with `r_j` the residual of the fit. Units are
chosen greedily by smallest `std(r_j)·‖W_ℓ[:, j]‖`; the action error of the whole fold is measured
on held-out samples and reported.

### O2 — Retire an idle unit

Usage of unit `j`: `u_j = std_𝒟(h_j) · ‖W_ℓ[:, j]‖₂`. If `u_j < τ` (τ = 10⁻³):
`b_ℓ += W_ℓ[:, j] · E_𝒟[h_j]`, then clear the unit as in O1. This is O1 with `α = 0`.

*Guarantee:* the change of `p_{ℓ+1}` has standard deviation `u_j` on 𝒟.

### O3 — Affine re-coordinatisation of the inputs

The original network is made to see `x̃ = T x + t` instead of `x`:

```
W₀′ = W₀ T,     b₀′ = b₀ + W₀ t                (in normalised coordinates)
```

*Guarantee:* exact. Uses: a constant command trim (`t` on the command entries), a rate loop
(`ω̃_z = ω_cmd + K(ω_cmd − ω_gyro)`: a row mix of `T`), disconnecting an input (its column replaced
by a constant: `T` zeroes it, `t` holds the constant), reading a register as a command shift (O7).
Applied to a subset of units only (rows of `W₀`), it re-coordinatises those units alone.

### O4 — Gated input shift (rank-1)

The original should see `x̃ = x + s·g(x)` only in a condition detected by a **linear** signal
`g(x) = wᵀx + w₀` (≈ 1 in the condition, ≈ 0 nominally). Since a first layer cannot multiply, the
gate must be linear in `x`, and

```
W₀′ = W₀ + (W₀ s) wᵀ,      b₀′ = b₀ + (W₀ s) w₀
```

*Guarantee:* exact in the condition up to `|g − 1|`; in the nominal condition the shift leaks in as
`(W₀ s)·g(x)`, measured by the rms of `g` on nominal 𝒟. `w` is a declared fit (least norm on
nominal 𝒟 under `wᵀx_cond + w₀ = 1`).

### O5 — Circuit primitives in freed units

Freed units (O1, O2) compute new quantities. With sharpness `S`, bias `B`, scale `s`:

| primitive | units | construction | error |
|---|---|---|---|
| pass-through | 1 | `σ(x + B) − B` | 0 for `x > −B` |
| ReLU, `|x|` | 1, 2 | `σ(Sx)/S`, `(σ(Sx) + σ(−Sx))/S` | ≤ 1/S |
| clamp to `±L` | 2 | `(σ(S(x+L)) − σ(S(x−L)))/S − L` | ≤ 1/S |
| square | 2 | `2s²(σ(x/s) + σ(−x/s))` | ≈ `|x|³/(3s)` |
| product | 4 | `((a+b)² − (a−b)²)/4` with two squares | from the squares |
| gate on `z` | 2 | `(σ(Sz) − σ(S(z−1)))/S` ≈ `clamp(z, 0, 1)` | ≤ 1/S |
| bump on an angle ψ | 1 | `φ_i = σ(β(c cos ψ_i + s sin ψ_i − cos α))`, with `c, s` = cos ψ, sin ψ available as inputs | exact shape |

A value computed in layer `ℓ` is available to layer `ℓ+1` only: the depth of the network bounds
how many primitives can be chained (a product takes two layers). A linear combination `Σ C_i φ_i`
of bumps synthesises a periodic function of ψ (yaw, gait phase).

### O6 — Exact pre-activation shift (before the fold)

To restore the operating point of layer `ℓ+1` as a condition variable ψ changes, require

```
p_{ℓ+1}(x₀(ψ)) + Δ(ψ) = p_{ℓ+1}(x*)    for all ψ,       Δ(ψ) = Cᵀ φ(ψ)
```

with `φ` the `N` bumps of O5 in layer `ℓ` and `x*` a reference state. Collocation at the `N` bump
centres gives the square system `Φ C = D` (`Φ_ik = φ_k(ψ_i)`, `D_i = p_{ℓ+1}(x*) − p_{ℓ+1}(x₀(ψ_i))`).

*Guarantee:* exact at the nodes, interpolated between them (report the condition number of `Φ`
and the error between nodes). Because it acts before σ, it restores both the values and the
slopes `σ'(p)` of the downstream units, hence the gains of the controller.

### O7 — Registers in action channels

`f` has no state. If the observation contains the previous action `a_{t−1}`, choose output
channels `R` and write a state update `z_t = F(z_{t−1}, x_t)` (built with O5) into their rows:
`a_t[R] = κ z_t`; at the next step `z_{t−1} = a_{t−1}[R]/κ`.

*Requirements:* the original's own use of `a_{t−1}[R]` must be replaced, because those inputs now
carry the register. Two ways, both an O3 on those columns:

- **constant fold** (exact): `b₀ += W₀[:, R] m_R`, `W₀[:, R] = 0`. Enough when the original barely
  needs `a_{t−1}[R]` (H1_2 wrists);
- **fitted predictor**: the original units read `â_R = M x_{−R}`, a ridge prediction of
  `a_{t−1}[R]` from the other inputs: `W₀[:, −R] += W₀[:, R] M`, `W₀[:, R] = 0`. Needed when the
  original does use those inputs in the nominal condition (SOLO v2: the constant fold turned the
  four-leg heading drift from 2° into 32° in 20 s). `M` is declared as fitted.

The actuators on `R` follow the register (a declared side effect), unless they are idle in the
target condition (a broken leg); then the channels must keep their normal behaviour in the nominal
condition, which needs O9.

### O8 — Gated output injection

`a′ = f(x) + Σ_k G_k(x) d_k(x)` with gates and terms built by O5 in the last hidden layer.

*Guarantee:* exact on the outputs, **but it changes values, not gains**: the units of the original
stay where they were on the fold. Use it only when they are already at the right operating point;
otherwise use O6 or O3/O4 first (the drone's divergent flight, §5.2).

### O9 — Gated output-row replacement

To give output rows `R` a new meaning in a condition `F` (detected by O4/O5) while keeping them
nominal otherwise: fit a readout `L(h_{L−1}) ≈ f_R(x)` of the original rows from the previous
hidden layer on nominal 𝒟, pass it through last-layer units gated off by `F`
(`σ(L_j + B − S F) − σ(B − S F)`), and set `W_L[R, :]` to these units plus the new content gated on
by `F`:

```
a′_R = (1 − F) · L(h_{L−1}) + F · new_R(x)
```

*Guarantee:* with `F = 1` the readout contributes exactly 0 and `a′_R = new_R`; with `F = 0`,
`a′_R` equals the readout, whose residual on nominal 𝒟 is measured and reported.

## 3. Procedure

| step | what | output |
|---|---|---|
| P1 read | column norms `‖W₀[:, i]‖`, normaliser means, what the network barely reads | hypotheses, no simulation |
| P2 probe | `f` on controlled states sweeping one variable; equilibria; Jacobians `∂f/∂x` at operating points | where it is broken: value, operating point or gain |
| P3 room | usage census `u_j`; O2 for idle units, O1 for redundant ones | free units per layer, fold error |
| P4 prototype | wrap the unchanged `f` with the intended input and output changes, test in simulation | a design that works before it is compiled |
| P5 compile | write the design with O3–O9 into the freed units | `policy.onnx`, `weight_edit.json`, checks: same shapes, ONNX vs numpy, compiled vs prototype `max |Δa|` |
| P6 validate | fixed protocol, original as baseline, nominal condition (the edit must not break it), ablations, seeds never used during design | results table, limits |

Rules that came out of the examples:

- Measure before editing (P1, P2): a zero training error says nothing about what the network does
  elsewhere.
- Prefer an input the network already understands (a command) to writing actuators directly.
- Correct before the fold (O3, O4, O6), not after it (O8), whenever a gain matters.
- Every behaviour removed by an edit must be replaced or checked.
- Bound every hand-written correction (clamps, ceilings).
- A fast simulator only finds the defects it contains: validate in the full one.
- Check that a failure is real before editing again (logs, physical plausibility).

## 4. Declaring what is fitted

`weight_edit.json` lists, for every piece: operator, layer, units touched, parameters, and whether
the parameters are exact, solved (linear system), fitted (data, how many samples) or hand-searched
(how many trials, in which simulator). Results of hand-searched designs are reported on seeds that
were not used during the search.

## 5. The three examples in these terms

### 5.1 H1_2 humanoid, heading hold (MLP 465 → 512 → 256 → 128 → 27)

| piece | operator | layer / size | parameters |
|---|---|---|---|
| free 20 / 9 / 5 units (lateral version 23 / 13 / 8) | O1 | hidden 1–3 | fold, action error rms 0.005 |
| yaw rate `−(ω_y g_y + ω_z g_z)(1 + g_x²)` | O5 products, squares | hidden 1–2 | exact construction |
| on/off gate (gait clock running, turn stick ≈ 0) | O5 `|x|`, gate | hidden 1–2 | S = 2000, 200 |
| integrators θ, I with clamps | O5 clamp | hidden 3 → outputs 25, 26 | L = 0.35, 0.30 |
| registers in the wrist-yaw channels | O7 | `last_action[25, 26]` | constant-folded original use |
| `ω_z` seen = `ω_cmd + 3θ + 3I` | O3 | first-layer columns of the registers | exact |
| lateral velocity readout `v̂ = w·h₁ + b` | fitted readout + O5 + O7 | hidden 1 → output 21 | ridge, 3128 conditions (declared) |

### 5.2 Quadrotor hover, yaw tracking (MLP 17 → 512 → 512 → 4)

| piece | operator | layer / size | parameters |
|---|---|---|---|
| disconnect the yaw error | O3 | 2 input columns | zero-error value into the bias |
| retire 28 + 5 idle units | O2 | hidden 1, 2 | usage < 10⁻³ |
| freeze yaw in 277 units | O3 on a subset of units | hidden 1 | ψ₀ = −50°, threshold ρ = 0.5 |
| 23 yaw bumps | O5 bump | hidden 1 | α = 50°, β = 20 |
| exact shift on 215 units | O6 (selective) | hidden 1 → 2 | 23 × 23 system, cond. 111, κ = 3 |
| residuals at the output | O8 | 3 units | — |
| yaw command `0.8·lim(1.25 sin e + 0.6(h⁺ − h⁻))` | O5 | hidden 1–2 → `ω_z` | K = 1, B = 20, λ = 2, H = 12 |
| climb limiter with ceiling | O5 + O8 | 2 + 1 units → thrust | g_c = 20 N/m, max 22 N |

The failed attempt "add the missing thrust at the output" was an O8 where O6 was needed.

### 5.3 SOLO-12, three legs (MLP 47 → 512 → 256 → 128 → 12)

| piece | operator | layer / size | parameters |
|---|---|---|---|
| free 46 / 6 / 5 units | O1 | hidden 1–3 | action error rms 0.001 |
| yaw-rate loop `ω_z + 0.5(ω_z − gyro_z)` | O3 | first layer | exact |
| fault signal `g = w·[q_FL, dq_FL]` | linear signal for O4 | — | least norm, 24 000 four-leg samples (declared) |
| hide the posture offset, command trim | O4 | rank-1 on the six FL columns | `A` from inverse kinematics; trim (0.7, 0.2, 0.2) hand-searched |
| fault flag `F`, gait gate `G` | O5 gate, `|x|` | hidden 1–2 | S = 50, 2000 |
| front-right re-timing `D_FR(φ)` | O5 40 bumps on the gait phase | hidden 1 → 3 | least squares to the hand-designed `D` |
| posture offset and re-timing at the output | O8 | outputs of the nine working joints and FR | `F·A`, `G·D_FR` |

O8 is safe here because O4 has already moved the original units to the operating point they have
on four legs ("hiding the posture"); without O4 the robot leans on the broken leg 74 % of the time.

**v2, heading hold** (adds to v1; 62 / 15 / 14 units freed in total, fold error rms 0.0019):

| piece | operator | layer / size | parameters |
|---|---|---|---|
| H1 integrator ported: rate, θ, I, clamps, gate on `F` | O5 | 16 / 9 / 5 units | θ ±0.20, I ±0.34, y = 3θ + 1.5I capped at 1.11 rad/s |
| registers θ, I, y in the dead leg's three action channels | O7 | FL outputs 0–2 | exact; error ≤ 0.005 rad (θ, I), 0.017 rad/s (y) |
| original units stop reading `last_action[FL]` | O7, fitted predictor | first layer, 44 columns | ridge on v1 rollouts, four and three legs (declared) |
| `ω_z` seen = `ω_z + y` | O3 on the register channel | first layer | exact with the leg broken; on four legs yaw-command noise std 0.055 rad/s |
| FL rows keep walking on four legs, carry registers on three | O9 | 3 + 1 last-layer units, FL rows | readout residual on four legs rms 0.010–0.018 |

O2 and O6 were not needed: idle units are freed by the O1 fold.
