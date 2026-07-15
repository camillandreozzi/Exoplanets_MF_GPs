# 02 — Full cross-validation of the joint MF-GP

## Question

How well does the joint AR(1) multi-fidelity GP (Model 1) predict HF spectra
it has never seen — measured on **all 97 HF samples**, not a single fixed
holdout? This experiment supersedes `01_loo_holdout_validation`, which
validated only sample 81.

## Protocol

K-fold cross-validation (K = 5, `exoplanets_mf.cv.CV_FULL_MODEL_SPLITS`) over
the 97 HF rows, shuffled with the project seed:

- every HF sample is held out exactly once;
- each fold refits the entire per-wavelength joint layer
  (`exoplanets_mf.mf_gp.fit_joint_mf_gp`) using only the fold's training HF
  rows — the LF data (10k design) is always fully included, since only HF
  availability is being validated;
- held-out spectra are predicted with `predict_hf` from atmospheric inputs
  alone (no paired LF output needed at prediction time).

The engine (`exoplanets_mf.cv.cv_predict`) is model-agnostic: a future
Model 2 plugs in with its own fit/predict callbacks and is compared to this
run's out-of-fold predictions via `exoplanets_mf.cv.compare_cv`.

## Input data

`exoplanets_mf.data.load_all()`: `XLF_10k (10000, 9)`, `YLF_10k (10000, 195)`,
`XHF (97, 9)`, `YHF (97, 195)`.

## Workflow

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m research.validation.02_full_cv
```

Cost note: this refits the full 195-GP layer K times, so it uses a smaller
LF subsample than the single fit (`SUBSAMPLE_SIZE` in the script; testing
value 200 ≈ 30–40 min for all 5 folds; a production value like 900 LF
measured ~27 s/wavelength ≈ 7.3 h for 5 folds).

## Configuration

- `N_SPLITS = 5`
- Random seed: `exoplanets_mf.reproducibility.RANDOM_SEED` (= 0)

## Evaluation

Per-sample and per-wavelength held-out RMSE, pooled RMSE/NRMSE, mean |z|, and
95% CI coverage (`exoplanets_mf.cv.cv_metrics`).

## Expected outputs

`results/validation/02_full_cv/evaluation/`:

- `cv_per_sample.csv` — sample, fold, RMSE, mean |z|, within-95%-CI fraction
- `cv_per_wavelength.csv` — wavelength, RMSE, 95% coverage
- `cv_summary.json` — pooled RMSE/NRMSE, mean |z|, coverage
- `01_cv_diagnostics.png` — predicted vs actual, coverage vs wavelength,
  per-sample RMSE

plus `results/validation/02_full_cv/run_metadata.json` (provenance).
