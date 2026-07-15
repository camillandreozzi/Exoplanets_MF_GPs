# 04: Statistics-101 summary for all models

## Question

How do Models 1A, 1B, 1 joint, and 2 compare in plain introductory statistics
(R^2, Pearson r, MAE, RMSE, bias) on linear and log10 scales, with everything
in one table and two simple figures?

## Protocol

`01_stats101_summary.py` refits nothing. It reuses the out-of-fold predictions
saved by `research/modelling/02_augmented_wavelength/02_cv_model2_vs_model1.py`
(`results/modelling/02_augmented_wavelength/cv/{linear,log10}/cv_predictions.npz`,
one shared 5-fold CV over the 97 paired HF samples) and computes pooled
statistics for three scale views:

- linear: linear-scale fits in original eclipse-depth units;
- log10: log10-scale fits in log10 units;
- log10 back-transformed: log10 predictions mapped through `10**prediction`
  (implied log-normal median) and scored in original units.

Recomputed RMSEs are asserted to match the source run's `cv_summary.json`
before any output is written.

## Workflow

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python \
  -m research.validation.04_stats101_summary
```

or:

```bash
make stats101
```

Requires `make model2` to have been run first (it produces the CV predictions).

## Expected outputs

`results/validation/04_stats101_summary/`:

- `stats101_summary.csv` - one row per model x scale view with all statistics
- `cv_predictions_long.csv` - one row per held-out point with actual values
  and every model's predictions in all three scale views
- `STATS101_REPORT.md` - readable tables plus a plain-language legend
- `01_pred_vs_actual.png` - predicted-vs-actual scatter grid with identity line
- `02_residuals.png` - residual histograms and per-model boxplots
- `03_spectrum81_holdout.png` - spectrum 81 actual vs every model's
  out-of-fold prediction (from `02_holdout_spectrum.py`; edit
  `SPECTRUM_LABEL` there to plot a different spectrum)
- `run_metadata.json` - provenance
