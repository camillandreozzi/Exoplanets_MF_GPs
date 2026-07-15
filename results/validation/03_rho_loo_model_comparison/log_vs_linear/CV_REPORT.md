# LOO CV Report: Linear (Non-Log) vs Log10 Models

## Result

The log10-fitted rho models are compared against the linear/non-log models after back-transforming their held-out predictions with `10**prediction`, so all metrics in this report are in original eclipse-depth units.

| Model | Lower pooled RMSE | Linear RMSE | Log10 back-transformed RMSE | Delta log - linear | Linear NRMSE | Log10 back-transformed NRMSE | Delta NRMSE | Log wins by sample | Mean sample delta | Sample delta std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model 1A (global rho) | log10 back-transformed | 0.000288375 | 0.000280875 | -7.50092e-06 | 0.0431439 | 0.0420217 | -0.00112221 | 68.0% | -9.29214e-06 | 3.59466e-05 |
| Model 1B (per-wavelength rho) | log10 back-transformed | 0.000280369 | 0.000272299 | -8.06973e-06 | 0.041946 | 0.0407387 | -0.00120731 | 59.8% | -6.5157e-06 | 3.20963e-05 |

Negative deltas mean the log10-fitted model is better after back-transformation.

## Computation

- Samples: 97 paired HF/LF spectra.
- Wavelength bins: 195.
- CV protocol: leave-one-out, `n_splits = 97`, seed `0`.
- Linear/non-log models are fitted and scored directly in original eclipse-depth units.
- Log models are fitted in `log10(YHF)` and `log10(YLF)` space.
- Log point predictions are converted back to original units as `10**mu_log10`; this is the predictive median implied by a log-normal interpretation.
- The fold assignment is identical between linear and log10 fits for each model before comparing paired errors.
- RMSE and NRMSE are computed against original-unit held-out `YHF`.

## Files

- `cv_per_sample.csv`: per-held-out-spectrum linear/log RMSE and paired delta for Models 1A and 1B.
- `cv_per_wavelength.csv`: per-wavelength linear/log RMSE and paired delta for Models 1A and 1B.
- `cv_summary.json`: machine-readable version of this summary.
- `01_log_vs_linear.png`: diagnostic figure.
