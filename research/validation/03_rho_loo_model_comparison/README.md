# 03: All-HF LOO validation for Model 1A/1B rho models

## Question

How do the closed-form rho variants behave under leave-one-out validation over
all 97 paired HF samples, and does fitting in log10 output space improve
linear-unit predictions after back-transformation?

This is separate from `../01_loo_holdout_validation/`, which intentionally
keeps only sample 81 (`spectrum 81`, 0-indexed row 80) as a fixed holdout for
the joint MF-GP reference check.

## Protocol

`01_compare_rho_loo.py` runs three comparisons:

- linear spectra: Model 1A (one global rho) vs Model 1B (rho per wavelength);
- log10 spectra: the same Model 1A vs Model 1B comparison in log10 units;
- log10 vs linear: log10 held-out predictions are back-transformed with
  `10**prediction` and compared to the linear-scale models in original
  eclipse-depth units.

Every comparison uses LOO over the 97 paired `YHF`/`YLF` rows. Each held-out
spectrum is predicted from fold-restricted means and rho estimates that never
use that row.

## Workflow

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python \
  -m research.validation.03_rho_loo_model_comparison
```

or:

```bash
make rho-loo-validation
```

## Expected outputs

`results/validation/03_rho_loo_model_comparison/`:

- `linear/` - Model 1A vs 1B LOO tables/plot in original units
- `log10/` - Model 1A vs 1B LOO tables/plot in log10 units
- `log_vs_linear/` - per-sample, per-wavelength, and summary comparison of
  linear models vs back-transformed log10 models
- `CV_COMPARISON_REPORT.md` - combined Markdown report with actual RMSE values
- `linear/CV_REPORT.md` - Markdown report for non-log Model 1A vs 1B
- `log10/CV_REPORT.md` - Markdown report for log10 Model 1A vs 1B
- `log_vs_linear/CV_REPORT.md` - Markdown report for log10 vs non-log
- `cv_summary.json` - combined summary
- `run_metadata.json` - provenance
