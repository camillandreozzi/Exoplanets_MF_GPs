# 01: Joint per-wavelength AR(1) multi-fidelity GP — log10 spectra

> **Log-modelling variant** of
> [`research/modelling/01_per_wavelength_ar1/`](../../modelling/01_per_wavelength_ar1/README.md).
> All spectra (YHF, YLF, YLF_10k) are log10-transformed after loading
> (`exoplanets_mf.transforms.log10_spectra`); inputs X are untouched. Every
> fitted quantity and RMSE below therefore lives in log10 space; predictions
> back-transform per [`../README.md`](../README.md). Outputs go to
> `results/log_modelling/01_per_wavelength_ar1/`. Run with `make log-modelling`.

## Model

Each wavelength has one jointly fitted model

\[
f_H(\theta)=\rho f_L(\theta)+\delta(\theta).
\]

LF and HF observations are stacked into one training vector. The LF kernel,
discrepancy kernel, scaling factor \(\rho\), and both observation-noise levels
are optimized together through one marginal likelihood. HF prediction needs
only the atmospheric inputs and does not require an LF output at the new point.

The block covariance is

\[
K =
\begin{bmatrix}
k_L & \rho k_L\\
\rho k_L & \rho^2 k_L + k_\delta
\end{bmatrix}.
\]

## Input data

- `XHF`, `YHF` (97x9, 97x195): high-fidelity design and spectra.
- `XLF_10k`, `YLF_10k` (10000x9, 10000x195): low-fidelity design and spectra.

One fixed LF subsample (size = `SUBSAMPLE_SIZE` in `01_fit_joint_mf_gp.py`,
kept equal to the linear-scale fit's) is stacked with all 97 HF observations
at every wavelength, mirroring `research/modelling/01_per_wavelength_ar1/`
where the sizing notes and the cost benchmark live (exact fit infeasible:
~40.8 GB per optimizer iteration).

## Workflow

Run `python -m research.log_modelling.01_per_wavelength_ar1` (or
`make log-modelling`). It executes the numbered scripts in order and writes
results plus provenance under `results/log_modelling/01_per_wavelength_ar1/`:

- `01_fit_joint_mf_gp.py` — fitted layer, hyperparameter table, timing, and
  diagnostic plot in `joint_mf_gp/`.
- `02_compare_rho_models.py` — Model 1A vs 1B comparison in `comparison/`.

Random sampling uses `exoplanets_mf.reproducibility.RANDOM_SEED`.

## Model comparison (cross-validation)

Question: is one global scaling \(\rho\) (Model 1A) enough, or does a
per-wavelength \(\rho_j\) (Model 1B) predict better?

Method: leave-one-out cross-validation over **all 97** paired HF samples on
the closed-form rho layer (`exoplanets_mf.cv`). Every spectrum is predicted
from a fit that never saw it (fold-restricted means and rho — no leakage),
and the two models are scored on identical held-out predictions: per-sample
and per-wavelength RMSE plus paired per-sample deltas.

The former AIC and nested-F-test comparison was removed deliberately:
likelihoods and effective parameter counts are hard to define credibly in
this framework, and the F reference distribution assumed i.i.d. residuals
that spectrally correlated data violates. Held-out predictive error needs
neither.

Outputs in `results/log_modelling/01_per_wavelength_ar1/comparison/`
(all quantities in log10 space):
`cv_per_wavelength.csv`, `cv_per_sample.csv`, `cv_summary.json`,
`02_model_comparison.png`. Seed: `RANDOM_SEED` (= 0), `N_SPLITS = 97` (LOO).
