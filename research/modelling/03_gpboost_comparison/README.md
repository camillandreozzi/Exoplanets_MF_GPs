# GPBoost Model Comparison

## Purpose

This experiment produces clean paired-CV GPBoost model-comparison outputs that
mirror the sklearn/mf_gp comparison in `results/modelling/02_augmented_wavelength/cv/`.

It has two modes:

- `matched`: HF-only vs GPBoost Model 1A vs Model 1B vs Model 2 on the same
  canonical LF subsample (`LF_SUBSAMPLE_SIZE` = 200 rows) and Model 2 CV point
  budget used by the sklearn/mf_gp custom-kernel comparison, with exact
  inference (`gp_approx="none"`, no Vecchia). This is the apples-to-apples
  implementation comparison. Every multi-fidelity model draws its LF rows from
  `mf_gp.select_lf_subsample(n_lf, 200, seed=0)`, i.e. literally the same rows;
  the run records their sha256 in `cv_summary.json`.
- `max-data`: the same comparison using larger data settings. By default it
  uses 1,000 LF rows, `gp_approx=vecchia`, and a configurable Model 2
  augmented-design point budget. Model 1A is skipped there by default (its
  block-coordinate sweep refits the whole per-wavelength stack several times);
  set `GPBOOST_MAX_INCLUDE_MODEL1A=1` to include it.

Both modes use 5-fold CV over whole HF spectra with the shared random seed.
HF-only is fitted with GPBoost as the single-fidelity reference floor. Model
1A is the GPBoost counterpart of the sklearn shared-rho fit: the same
block-coordinate ascent on one joint criterion (rho fixed -> GPBoost estimates
every other parameter per
wavelength; everything else fixed -> 1-D search of the shared rho against the
summed marginal likelihood), with rho pinned through GPBoost's
`estimate_cov_par_index`. Its search grid is linear, not logarithmic, because
GPBoost's rho is unrestricted and Model 1B fits rho_j < 0 at most wavelengths.

Per-model CV wall time is printed as the run proceeds and written to
`cv_summary.json` (`runtime_seconds`) and to the `## Runtime` section of
`CV_REPORT.md`.

## Commands

For full-data parameter fits of all four GPBoost variants:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u research/modelling/03_gpboost_comparison/02_fit_all_gpboost.py \
  --gp-approx none
```

For an exact LF=1000 parameter run:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u research/modelling/03_gpboost_comparison/02_fit_all_gpboost.py \
  --gp-approx none \
  --model1-lf-sample-size 1000 \
  --model2-lf-sample-size 1000 \
  --model2-max-points 10000
```

The matched comparison runs as part of `make models` (stage
`.stage-gpb-matched`). To run just this experiment, with a timestamped log:

```bash
mkdir -p logs
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u -m research.modelling.03_gpboost_comparison \
  --mode matched --gp-approx none \
  2>&1 | tee "logs/gpboost_matched_$(date +%Y%m%d_%H%M%S).log"
```

### Choosing the GPBoost approximation

`--gp-approx` overrides the mode's default (`matched` is exact, `max-data` is
Vecchia), and `--num-neighbors` sets the Vecchia neighbour count:

```bash
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u -m research.modelling.03_gpboost_comparison \
  --mode matched --gp-approx vecchia --num-neighbors 30
```

An overridden run writes to its own directory (`matched_subsample_vecchia_k30`
here), never into the mode's default directory: exact and approximate fits of
the same model are different estimators and must not overwrite each other.
The same two flags exist on `04_fit_model1_gpboost.py` and
`03_fit_model2_gpboost.py`, with the same directory-suffix rule.

Vecchia is a large-data approximation and is **slower than exact inference at
the matched subsample's size**. Model 1A/1B fit n = 278 points per wavelength,
far below the dense-Cholesky crossover, and for an `ar1_mf_*` covariance
GPBoost re-selects each point's neighbours by absolute correlation, so the
neighbour search reruns as the covariance parameters move. Measured on 13
wavelengths: 1.5 s/wavelength exact against 5.5 (k=20), 11.1 (k=30) and 23.7
(k=50) s/wavelength with Vecchia -- a full Model 1B CV goes from 25 minutes to
1.5-6.4 hours, for held-out RMSE that is not better (1.39e-04 exact vs
1.59e-04, 1.30e-04, 1.36e-04). Use `--gp-approx vecchia` for the larger
`max-data` designs, not for the matched comparison.

The max-data mode is the same command with `--mode max-data`, and can be
scaled through environment variables:

```bash
GPBOOST_MAX_MODEL1_LF_SAMPLE_SIZE=2000 \
GPBOOST_MAX_MODEL2_LF_SAMPLE_SIZE=2000 \
GPBOOST_MAX_MODEL2_MAX_POINTS=20000 \
GPBOOST_MAX_NUM_NEIGHBORS=40 \
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u -m research.modelling.03_gpboost_comparison --mode max-data
```

To run the LF=1000 CV configuration with exact GPBoost and Model 1A included:

```bash
GPBOOST_MAX_MODEL1_LF_SAMPLE_SIZE=1000 \
GPBOOST_MAX_MODEL2_LF_SAMPLE_SIZE=1000 \
GPBOOST_MAX_MODEL2_MAX_POINTS=10000 \
GPBOOST_MAX_INCLUDE_MODEL1A=1 \
PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib \
  .venv/bin/python -u -m research.modelling.03_gpboost_comparison \
  --mode max-data --gp-approx none
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
- `GPBOOST_MAX_INCLUDE_MODEL1A`: `1` to also run Model 1A (default: off).

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
