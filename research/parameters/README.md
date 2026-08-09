# Parameter visualisations

## Question

Did every fitted model converge to sensible covariance hyperparameters? This
stage reviews the fitted parameters of **all** models side by side so we can
check the models are doing the right thing — that `rho` is reasonable, the
signal and noise variances are not degenerate, and the ARD length scales are not
pinned at their optimiser bounds (a length scale resting at the upper bound flags
an inert input).

## Input data

Read-only. Consumes the `*_hyperparameters.csv` tables that the fit stages
already wrote — nothing is refitted and `data/` is never touched. Any model whose
CSV is absent is skipped, so the stage runs on whatever has been fitted. The
registry, the sklearn-vs-GPBoost column normalisation, and the shared plotting
helpers live in `src/exoplanets_mf/parameter_tables.py`. Covered models:

- **Per-wavelength** (`results/modelling/01_per_wavelength_ar1/`,
  `results/log_modelling/01_per_wavelength_ar1/`,
  `results/validation/01_loo_holdout_validation/fit/`):
  Model 1A, Model 1B, Model 1 (GPBoost), log-space 1A/1B, LOO-holdout 1A/1B.
- **Scalar** (`results/modelling/02_augmented_wavelength/`):
  Model 2 linear, Model 2 log10, Model 2 (GPBoost).

Instrument-band shading uses the shared `exoplanets_mf.instruments` definitions
(NIRCam F322W2 `[2.4, 4)`, NIRCam F444W `[4, 5)`, MIRI LRS `[5, 12]` um).

## Configuration

No random seed or refit — purely a reporting stage. Optimiser bounds used for
the reference lines (`RHO_BOUNDS`, `SIGNAL_VARIANCE_BOUNDS`, `NOISE_LEVEL_BOUNDS`,
`LENGTH_SCALE_BOUNDS`) come from `exoplanets_mf.mf_gp`. GPBoost "range" parameters
are renamed to the sklearn length-scale vocabulary for a uniform API but are not
in the same units, so they are excluded from the cross-model length-scale
comparison.

## Workflow

1. `01_per_wavelength_parameters.py` — one diagnostic grid per per-wavelength
   model: rho, signal variances, noise, LF/discrepancy ARD length scales, and the
   joint log marginal likelihood vs wavelength, with instrument bands and bound
   reference lines.
2. `02_scalar_parameters.py` — one figure per Model 2 variant: the 9 atmospheric
   ARD length scales (LF vs discrepancy) plus a scalar summary of the remaining
   covariance parameters (signal variances, noise, wavelength length scale in um,
   rho annotated).
3. `03_cross_model_overlays.py` — overlays across all models (rho, signal
   variances, noise vs wavelength; per-input ARD length scales for the sklearn
   models) and the `parameter_summary.csv` at-a-glance table.

## Expected outputs

Written under `results/parameters/`:

- `per_wavelength/<model_key>.png` — per-model diagnostic grids.
- `scalar/<model_key>.png` — per Model 2 variant scalar summaries.
- `overlays/rho_vs_wavelength.png`, `overlays/low_signal_variance.png`,
  `overlays/delta_signal_variance.png`, `overlays/noise.png`,
  `overlays/length_scale_by_input.png` — cross-model overlays.
- `overlays/parameter_summary.csv` — one row per model (rho summary,
  variance/noise ranges, count of length scales pinned at a bound).
- `run_metadata.json` — provenance for the run.

## Run

```bash
make parameters
```
