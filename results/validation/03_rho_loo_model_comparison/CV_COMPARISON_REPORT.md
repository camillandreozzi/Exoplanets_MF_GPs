# CV Comparison Report

## Scope

This report summarizes leave-one-out cross-validation over all 97 paired HF samples for the closed-form rho models.

## Main Results

| Comparison | Units | Model/contrast | RMSE 1 | RMSE 2 | Delta 2 - 1 | Winner |
|---|---|---|---:|---:|---:|---|
| Model 1A vs Model 1B | Original eclipse-depth | A vs B | 0.000288375 | 0.000280369 | -8.00659e-06 | Model 1B |
| Model 1A vs Model 1B | log10 eclipse-depth | A vs B | 0.153393 | 0.148269 | -0.00512471 | Model 1B |
| Linear/non-log vs log10 | Original eclipse-depth | Model 1A linear/non-log vs log10 back-transformed | 0.000288375 | 0.000280875 | -7.50092e-06 | log10 |
| Linear/non-log vs log10 | Original eclipse-depth | Model 1B linear/non-log vs log10 back-transformed | 0.000280369 | 0.000272299 | -8.06973e-06 | log10 |

## Technical Notes

- All comparisons use the same 97 LOO splits with seed `0`.
- Model 1A uses one global train-fold rho; Model 1B uses one train-fold rho per wavelength.
- For the log-vs-linear comparison, log predictions are scored only after back-transforming to original eclipse-depth units with `10**mu_log10`.
- Detailed per-comparison reports are in:
  - `linear/CV_REPORT.md`
  - `log10/CV_REPORT.md`
  - `log_vs_linear/CV_REPORT.md`
- CSV files provide per-sample and per-wavelength diagnostics for paper tables or plotting.
