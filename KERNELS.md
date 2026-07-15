# Fitted kernels: complete reference across all models

This document records every covariance structure fitted in this project, the
hyperparameters obtained, and — flagged explicitly throughout — every
approximation made for numerical or computational reasons. Fitted values refer
to the artifacts currently persisted under `results/` (seed 0 throughout).

**Summary of the approximation situation up front:** GP *inference* is exact
everywhere (dense Cholesky, no sparse/variational/Vecchia/inducing-point
methods anywhere in the project). All computational approximations are **data
subsampling**: the exact posterior is computed on a subset of the available
training data because the full designs exceed memory. The remaining numerical
devices (jitter, noise floors, bound constraints, standardizations) are listed
in §5.

---

## 1. Common building blocks

### 1.1 Base kernels (all GP models)

Every GP in the project is built from the same two scikit-learn components
(`exoplanets_mf.mf_gp.make_joint_mf_kernel`):

| Component | Form | Role |
|---|---|---|
| Amplitude | `ConstantKernel(1.0, bounds=(1e-3, 1e3))` | signal variance σ² |
| SE-ARD | `RBF(length_scale=1.0 per dim, bounds=(1e-2, 1e2))` | anisotropic squared exponential |

so each kernel factor is k(x, x′) = σ² · exp(−½ Σ_d (x_d − x′_d)²/ℓ_d²), with
one length scale per input dimension, in **standardized input units**
(`StandardScaler` fitted on the LF inputs).

### 1.2 The AR(1) multi-fidelity kernel

`AR1MultiFidelityKernel` (`src/exoplanets_mf/mf_gp.py`) implements the joint
two-fidelity covariance implied by f_H(x) = ρ·f_L(x) + δ(x) with f_L ⟂ δ:

```
K_LL = k_L            K_LH = ρ·k_L           K_HH = ρ²·k_L + k_δ
```

The fidelity indicator (0 = LF, 1 = HF) is the **last input column**; k_L and
k_δ act on the preceding columns only. Two kernel *components* (k_L, k_δ) form
ONE covariance function fitted in ONE marginal likelihood — this is not a
two-step / recursive (Le Gratiet) scheme, which was explicitly rejected as a
design decision. Additional hyperparameters of the composite kernel:

| Parameter | Init | Bounds | Meaning |
|---|---|---|---|
| ρ | 1.0 | (1e-3, 1e3) | scalar LF→HF scaling |
| low_noise | 1e-4 | (1e-8, 1e1) | learned LF diagonal noise |
| high_noise | 1e-4 | (1e-8, 1e1) | learned HF diagonal noise |

All parameters are optimized in log space by L-BFGS-B with analytic kernel
gradients, `n_restarts_optimizer = 1` (one extra random restart), inside
sklearn's `GaussianProcessRegressor` with `alpha = 1e-10` and
`normalize_y = True`.

---

## 2. Model 1A / 1B: closed-form rho layers (kernel-free)

**No GP kernel is fitted.** These are closed-form through-origin AR(1) MLEs on
per-wavelength-standardized spectra (`mf_gp.per_wavelength_rho`,
`mf_gp.global_rho`; CV variants in `cv.fit_rho_model`):

- z_LF[i,j] = (YLF[i,j] − μ_LF[j]) / sd_LF[j],  r_HF[i,j] = (YHF[i,j] − μ_HF[j]) / sd_LF[j]
- **Model 1B**: ρ_j = Σ_i r_HF·z_LF / Σ_i z_LF² per wavelength (195 values).
- **Model 1A**: one pooled ρ over all (sample, wavelength) pairs
  (variance-weighted; not the mean of the ρ_j).

**Approximations: none.** Exact closed forms on the full 97 paired samples
(fold-restricted moments during CV). Note these models consume the held-out
sample's paired LF spectrum at prediction time — a structural information
advantage over the GP models, not an approximation.

Fitted values (full data): pooled ρ̂ = 0.350 (linear), 0.460 (log10);
ρ̂_j ranges ≈ 0.10–0.67 (linear).

---

## 3. Model 1: per-wavelength joint AR(1) MF-GP

**Structure**: 195 *independent* GPs, one per wavelength bin j. Each uses the
AR(1) kernel of §1.2 over the **9 atmospheric inputs** (Kzz, Rp, Tint, C, N,
O, S, logg, f) + fidelity flag → **22 hyperparameters per wavelength**
(2 signal variances + 2×9 ARD length scales + ρ_j + 2 noises), i.e. 195
separate marginal-likelihood optimizations and 195 independent ρ_j.
No correlation across wavelengths is modelled.

Fitted per scale (`fit_joint_mf_gp`), artifacts in
`results/modelling/01_per_wavelength_ar1/joint_mf_gp/` and its
`log_modelling` mirror:

| Quantity (median over 195 λ, min–max) | Linear | log10 |
|---|---|---|
| ρ_j | 0.259 (0.001–0.635) | 0.329 (0.001–0.781) |
| LF signal variance | 2.01 (1.31–6.48) | 2.03 (1.31–4.91) |
| δ signal variance | 0.656 (0.204–3.27) | 0.707 (0.133–2.97) |
| LF noise | 0.0276 | 0.0275 |
| HF noise | 9.1e-07 (floor 1e-8 hit at some λ) | 9.7e-07 (floor hit at some λ) |

Median ARD length scales (standardized units) reproduce the input-sensitivity
findings: short for C (≈0.7), O (≈0.8), N (≈2.3), f (≈3.8); long/inert for
logg (≈40, at the 100 bound for 19% of wavelengths on linear scale), S, Kzz,
Rp, Tint. The δ kernel pins Kzz and logg at the 100 upper bound for most λ.

**⚠ Approximation (computational, data subsampling):** the exact joint fit on
all 10,000 LF + 97 HF rows per wavelength needs ~40.8 GB per optimizer
iteration (kernel-gradient stack, `exact_fit_memory_bytes`) vs 17.2 GB RAM —
infeasible. **The persisted layers use a fixed random LF subsample of 400
rows** (+ all 97 HF rows; `SUBSAMPLE_SIZE = 400`, the documented TESTING size;
~8.2–8.9 s/wavelength, ~27–29 min per scale). The benchmark
(`00_benchmark_exact_fit.py`) recommends ~1500–1700 LF for an ~8 h production
run; a 1700-LF run (~6.7 h) was executed historically but is not the persisted
artifact. Inference given the subsample is exact. The 5-fold CV of this model
(`validation/02_full_cv`, and the Model 1 joint baseline inside the Model 2
comparison) uses an even smaller **200-LF testing subsample** because every
fold refits all 195 GPs.

**⚠ Numerical devices:** per-wavelength `normalize_y=True` (output
standardization inside each GP — this is what makes a unit-scale signal-variance
prior reasonable and absorbs the orders-of-magnitude differences between
wavelength bins), `alpha = 1e-10` diagonal jitter, learned noise floors at
1e-8 (the HF noise rides this floor at many wavelengths — the HF simulator is
effectively noiseless and the floor acts as jitter).

---

## 4. Model 2: wavelength-augmented joint MF-GP

**Structure**: ONE scalar-valued AR(1) MF-GP over the augmented input
z = (θ, λ) ∈ ℝ¹⁰ per output scale. Same composite kernel as §1.2 but with
**10-dimensional** SE-ARD factors (9 atmospheric + wavelength, then the
fidelity flag) → **25 hyperparameters total** for the entire spectrum, single
joint marginal likelihood, single scalar ρ. Wavelength enters k_L and k_δ as
an ordinary ARD dimension; cross-wavelength correlation is therefore modelled
(stationary in standardized λ), and the wavelength dependence of the LF–HF
link is absorbed by δ(θ, λ).

Fitted values (`results/modelling/02_augmented_wavelength/{linear,log10}/
model2_hyperparameters.csv`; n = 3385 points, ~40–43 min per fit):

| Parameter | Linear | log10 |
|---|---|---|
| ρ (scalar) | **0.708** | **0.728** |
| LF signal variance | 1.60 | 1.86 |
| δ signal variance | 0.262 | 0.0313 |
| LF noise | 0.0341 | 0.0260 |
| HF noise | 0.00953 | **1e-8 (pinned at floor)** |
| LF λ length scale | 0.112 std ≈ **0.20 µm** | 0.177 std ≈ **0.31 µm** |
| δ λ length scale | **100 (upper bound)** ≈ 174 µm (δ ~constant in λ) | **0.010 (lower bound)** ≈ 0.018 µm (δ ~white in λ) |
| log marginal likelihood | −739.1 | −286.3 |

Atmospheric length scales mirror Model 1 (short C/O/N/f, logg at or near the
100 bound). Note the qualitative flip of the δ kernel's λ behaviour between
scales, and that the scalar ρ̂ ≈ 0.71 sits *above* the entire Model 1B ρ_j
curve — in the joint standardized model ρ is a covariance-weighted quantity,
not a pointwise regression slope; the two are not directly comparable.

**⚠ Approximation (computational, data subsampling — the dominant one in the
project):** the full augmented design has (10,000 + 97) × 195 ≈ **1.97 M**
scalar points; the memory gate caps one exact optimizer iteration at
**n ≈ 5,640** points (25-hyperparameter gradient stack, 80% of 17.2 GB RAM).
Both fidelities are therefore subsampled, structurally:

- all **97 HF samples × a stride-8 wavelength subgrid** (25 of 195 λ);
- **40 LF samples** (fixed random subset) **× the half-stride-offset subgrid**
  (24 λ, offset 4 — offsetting doubles the distinct wavelengths seen);
- total n = 97·25 + 40·24 = **3,385** of 1,968,915 points (~0.17%).

The remaining ~7/8 of wavelengths are *never trained on* and are predicted
purely through the fitted λ length scale — this interpolation is a modelling
consequence of the subsampling and is monitored explicitly (in-sample
residual RMS at trained vs interpolated λ, `01_model2_diagnostics.png`:
interpolated wavelengths run ~3–10× higher residuals, worst toward the MIRI
long-λ end). The benchmark-recommended production size is 120–130 LF samples
(n ≈ 5,300–5,500, ~1–2 h per scale); measured fit-time scaling between ladder
anchors was **worse than cubic**, so extrapolations are optimistic.

**⚠ Approximation (CV configuration):** the 5-fold CV refits the joint GP per
fold, so it runs coarser still: **stride 13** (15 HF + 15 LF wavelengths),
40 LF samples, n = 2,055 per fold. LOO (97 folds) was ruled out on cost.
All CV comparisons pair identical KFold assignments (asserted at runtime).

**⚠ Numerical devices:**
- **Per-wavelength output standardization by LF column moments** (μ_λ, s_λ
  from all 10k LF spectra) before flattening, inverted at prediction. This is
  the single-GP analog of Model 1's per-wavelength `normalize_y=True` and is
  *required* for a stationary λ kernel on the linear scale (bins differ by
  orders of magnitude). It is leakage-free (LF-only moments) and leaves ρ
  invariant. Global `normalize_y=True` is applied on top by sklearn.
- Same `alpha = 1e-10` jitter, log-space L-BFGS-B, 1 restart, StandardScaler
  now over all 10 augmented columns (λ standardized like any input).
- **Bound-pinned parameters in the fitted optima** (constraints acting as
  regularizers): log10 HF noise at the 1e-8 floor; δ λ length scale at the
  1e-2 lower bound (log10) / 1e2 upper bound (linear); logg length scale at
  1e2. Interpret these parameters as "at their constraint", not as interior
  maximum-likelihood estimates; sklearn emits `ConvergenceWarning`s for them.

---

## 5. Consolidated list of approximations and numerical devices

| # | Device | Where | Reason | Nature |
|---|---|---|---|---|
| 1 | LF subsample 400 (persisted) / 1500–1700 (production rec.) of 10,000 | Model 1 joint GP, per λ | 40.8 GB/iter exact fit vs 17.2 GB RAM | data subsampling; inference exact |
| 2 | LF subsample 200 | Model 1 joint GP inside all 5-fold CVs | 5× refit of 195 GPs per scale | data subsampling (testing config) |
| 3 | Augmented design 3,385 of 1.97 M points (stride-8 λ subgrids + 40 LF samples) | Model 2 production fits | memory cap n ≈ 5,640; super-cubic time | data subsampling; inference exact |
| 4 | CV design 2,055 points (stride 13) | Model 2 5-fold CV | fold refit cost; LOO ruled out | data subsampling |
| 5 | `alpha = 1e-10` diagonal jitter | every GP | Cholesky PSD safety | numerical |
| 6 | Learned noise floors 1e-8 (hit by HF noise, Model 1 many λ; Model 2 log10) | every GP | keeps K + noise invertible; simulators near-noiseless | numerical / constraint |
| 7 | Hyperparameter bounds (1e-3–1e3 variance & ρ; 1e-2–1e2 length scales) with observed pinning (logg, δ-λ) | every GP | optimizer stability; implicit regularization | constraint |
| 8 | Input StandardScaler (9 or 10 cols, fitted on LF) | every GP | unit-scale ARD priors/inits | preprocessing (exact, invertible) |
| 9 | `normalize_y=True` per GP; Model 2 additionally per-λ standardization by LF moments | all GPs / Model 2 | λ-heteroscedasticity; unit-scale signal variance | preprocessing (exact, invertible, leakage-free) |
| 10 | L-BFGS-B, log-parametrization, 1 restart | every GP | non-convex likelihood | local optimization (optima may be local; occasional `ConvergenceWarning: ABNORMAL`) |
| 11 | Log10 predictions back-transformed as 10^μ (log-normal median) | log-scale CV comparisons | point summary choice for original-unit scoring | modelling convention, not approximation of the fit |

**Explicitly NOT used anywhere:** Vecchia approximations, inducing points /
sparse variational GPs, Nyström, Kronecker/structured solvers, and the
recursive (Le Gratiet) two-step decomposition — the last was a deliberate
methodological decision (single joint likelihood defines the model), the
others were ruled out in favour of exact inference on subsampled designs.

---

## 6. Pointers

- Kernel implementation: `src/exoplanets_mf/mf_gp.py` (`AR1MultiFidelityKernel`,
  `make_joint_mf_kernel`, bounds/inits, memory gate), `src/exoplanets_mf/model2.py`
  (augmented design, per-λ standardization, Model 2 fit/predict/CV).
- Fitted hyperparameters: `results/modelling/01_per_wavelength_ar1/joint_mf_gp/
  joint_mf_gp_hyperparameters.csv` (+ `log_modelling` mirror);
  `results/modelling/02_augmented_wavelength/{linear,log10}/model2_hyperparameters.csv`.
- Sizing derivations: `results/modelling/01_per_wavelength_ar1/benchmark/
  exact_fit_benchmark.json`; `results/modelling/02_augmented_wavelength/
  benchmark/model2_fit_benchmark.json`.
- Methodology: `Model1_methodology.tex`, `Notes_v1.tex` (§ Model 2).
