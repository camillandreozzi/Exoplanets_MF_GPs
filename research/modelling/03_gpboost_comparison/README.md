# GPBoost Model Comparison

## Purpose

This experiment produces clean paired-CV GPBoost model-comparison outputs that
mirror the sklearn/mf_gp comparison in `results/modelling/02_augmented_wavelength/cv/`.

It has two modes:

- `matched`: GPBoost Model 1B vs GPBoost Model 2 on the same canonical LF
  subsample (`LF_SUBSAMPLE_SIZE`) and Model 2 CV point budget used by the
  sklearn/mf_gp custom-kernel comparison. This is the apples-to-apples
  implementation comparison.
- `max-data`: GPBoost Model 1B vs GPBoost Model 2 using larger data settings.
  By default it uses 1,000 LF rows, `gp_approx=vecchia`, and a configurable
  Model 2 augmented-design point budget.

Both modes use 5-fold CV over whole HF spectra with the shared random seed.
The HF-only sklearn GP is included as the same single-fidelity reference floor
used in the stage-1 comparison.

## Commands

```bash
make modelling_gpboost_comparison
make modelling_gpboost
```

`modelling_gpboost` can be scaled through environment variables:

```bash
GPBOOST_MAX_MODEL1_LF_SAMPLE_SIZE=2000 \
GPBOOST_MAX_MODEL2_LF_SAMPLE_SIZE=2000 \
GPBOOST_MAX_MODEL2_MAX_POINTS=20000 \
GPBOOST_MAX_NUM_NEIGHBORS=40 \
make modelling_gpboost
```

Available controls:

- `GPBOOST_MAX_MODEL1_LF_SAMPLE_SIZE`: integer or `all` (default: `1000`).
- `GPBOOST_MAX_MODEL2_LF_SAMPLE_SIZE`: integer (default: `1000`).
- `GPBOOST_MAX_MODEL2_MAX_POINTS`: augmented scalar-row budget for Model 2
  (default: `10000`).
- `GPBOOST_MAX_GP_APPROX`: GPBoost approximation (default: `vecchia`).
- `GPBOOST_MAX_NUM_NEIGHBORS`: Vecchia neighbors, or `none` for GPBoost's
  internal default (default: `30`).
- `GPBOOST_MAX_N_SPLITS`: CV folds (default: `5`).

## Outputs

- `results/modelling/03_gpboost_comparison/matched_subsample/`
- `results/modelling/03_gpboost_comparison/max_data/`

Each output directory contains:

- `CV_REPORT.md`
- `cv_summary.json`
- `cv_per_sample.csv`
- `cv_per_wavelength.csv`
- `cv_predictions.npz`
- `01_gpboost_model_comparison.png`
- `run_metadata.json`
