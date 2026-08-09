# 01: Joint per-wavelength AR(1) multi-fidelity GPs (Models 1A and 1B)

## Model

Each wavelength has one jointly fitted model

\[
f_H(\theta)=\rho f_L(\theta)+\delta(\theta).
\]

LF and HF observations are stacked into one training vector. The LF kernel,
discrepancy kernel, scaling factor \(\rho\), and both observation-noise levels
are optimized through the marginal likelihood. HF prediction needs only the
atmospheric inputs and does not require an LF output at the new point.

The block covariance is

\[
K =
\begin{bmatrix}
k_L & \rho k_L\\
\rho k_L & \rho^2 k_L + k_\delta
\end{bmatrix}.
\]

**Both variants are the same 195-GP multi-fidelity layer; they differ ONLY
in the rho criterion:**

- **Model 1A (global rho)**: ONE scalar \(\rho\) shared by all 195
  wavelengths, optimized against the SUM of the per-wavelength log marginal
  likelihoods — the joint likelihood, since wavelengths are conditionally
  independent given the hyperparameters. Fitted by block-coordinate ascent
  (`exoplanets_mf.mf_gp.fit_joint_mf_gp_global_rho`): with rho fixed, every
  wavelength's remaining hyperparameters are refit; with all kernels fixed,
  the shared rho is updated by a staged global grid search of the summed
  likelihood; neither step can decrease the joint likelihood, and sweeps
  repeat until the rho update moves less than `GLOBAL_RHO_LOG_TOL`.
- **Model 1B (per-wavelength rho)**: one free \(\rho_j\) per wavelength,
  optimized inside that wavelength's own marginal likelihood
  (`exoplanets_mf.mf_gp.fit_joint_mf_gp`).

There is deliberately NO two-stage discrepancy-vs-LF model anywhere: the
former closed-form rho layers (rho estimated from paired spectra, then a
mean-discrepancy correction) were removed, and every model in the project
is a jointly fitted multi-fidelity GP.

## Input data

- `XHF`, `YHF` (97x9, 97x195): high-fidelity design and spectra.
- `XLF_10k`, `YLF_10k` (10000x9, 10000x195): low-fidelity design and spectra.

One fixed LF subsample is stacked with all 97 HF observations at every
wavelength. Its size is the project-wide `LF_SUBSAMPLE_SIZE` (see
`exoplanets_mf.reproducibility`) — the **one canonical LF subsample shared by
every model and CV**: because all models draw from the same LF pool
(`XLF_10k`) with the same size and seed, they select the identical rows
point-for-point. The exact fit on all 10,097 points is infeasible: one
optimizer iteration materializes a ~40.8 GB kernel-gradient stack vs 16 GB
RAM, and would take >1000 h even with the memory — `00_benchmark_exact_fit.py`
measures this and records the largest LF subsample that fits an 8 h budget in
`benchmark/exact_fit_benchmark.json` (`recommended_lf_subsample`, ~1500 on the
benchmark machine; memory alone would allow n ≈ 5800 but time binds).
`fit_joint_mf_gp(subsample_size=None)` remains the exact fit for larger
machines, guarded by a fail-fast memory check.

## Workflow

Run `python -m research.modelling.01_per_wavelength_ar1`. It executes the
numbered scripts in order and writes results plus provenance under
`results/modelling/01_per_wavelength_ar1/`:

- `01_fit_model1b.py` — Model 1B layer, hyperparameter table, timing, and
  diagnostic plot in `model_1b/`.
- `02_fit_model1a.py` — Model 1A layer (shared rho, warm-started from the
  saved 1B layer), sweep history, and diagnostics in `model_1a/`.
- `03_compare_model1a_vs_1b.py` — CV comparison in `comparison/`.

Random sampling uses `exoplanets_mf.reproducibility.RANDOM_SEED`.

## Both fidelity scales in one run

Each script fits **both** the linear spectra and the log10-transformed spectra
(`exoplanets_mf.transforms.log10_spectra` applied to `YHF`/`YLF_10k` after
loading; the atmospheric inputs `X` are untouched). Linear-scale artifacts go
to `results/modelling/01_per_wavelength_ar1/`; the log10 mirror goes to
`results/log_modelling/01_per_wavelength_ar1/`. This replaces the former
separate `research/log_modelling/` tree — the same consolidation Model 2
already uses (`research/modelling/02_augmented_wavelength/01_fit_model2.py`).

Quantities fitted on log10 depths (rho, GP hyperparameters) are *different
quantities* from their linear-scale counterparts — a scaling rho on log depths
corresponds to a power law in linear units — so compare across scales only
through back-transformed predictions. Motivation for the log scale:
`research/exploratory/05_log_spectra_exploration.py` (eclipse depths span ~4
orders of magnitude with strong right skew that log10 removes).

**Back-transforming log10 predictions.** For a GP predictive mean `m` and std
`s` (both log10): `10**m` is the predictive **median** in linear units (the
log-normal mean is `10**m * exp((ln(10) * s)**2 / 2)`); a 95% interval
back-transforms to `[10**(m - 1.96 s), 10**(m + 1.96 s)]`. Use
`exoplanets_mf.transforms.inverse_log10_spectra` for the point back-transform.

## Model comparison (cross-validation)

Question: is one global scaling \(\rho\) (Model 1A) enough, or does a
per-wavelength \(\rho_j\) (Model 1B) predict better?

Method: 5-fold cross-validation over all 97 paired HF samples
(`exoplanets_mf.cv.cv_predict_joint_mf_gp`, identical fold assignment for
both models). Every fold refits both full 195-GP layers from scratch on the
shared canonical LF subsample (`CV_SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE`) and
predicts the held-out HF spectra from atmospheric inputs alone — no model
sees the
held-out sample's LF spectrum at prediction time. Scores: per-sample and
per-wavelength held-out RMSE, paired per-sample deltas, and 95% CI coverage
from the GP predictive stds. LOO is not affordable here because each of the
97 folds would refit 2 x 195 GPs.

The former AIC and nested-F-test comparison was removed deliberately:
held-out predictive error needs neither a parameter count nor an i.i.d.
residual assumption.

Outputs in `results/modelling/01_per_wavelength_ar1/comparison/`:
`cv_per_wavelength.csv`, `cv_per_sample.csv`, `cv_summary.json`,
`03_model_comparison.png`. Seed: `RANDOM_SEED` (= 0),
`N_SPLITS = CV_FULL_MODEL_SPLITS` (= 5).
