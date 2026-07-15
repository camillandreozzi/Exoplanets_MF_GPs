# LOO CV Report: Model 1A vs Model 1B

## Result

Model 1B (per-wavelength rho) has the lower pooled held-out RMSE in this comparison.

| Metric | Model 1A | Model 1B | Delta B - A |
|---|---:|---:|---:|
| Pooled RMSE | 0.153393 | 0.148269 | -0.00512471 |
| Pooled NRMSE | 0.056383 | 0.0544993 | -0.0018837 |
| Mean per-sample RMSE | 0.139843 | 0.132116 | -0.00772708 |
| Median per-sample RMSE | 0.123332 | 0.111625 | -0.011707 |
| Mean per-wavelength RMSE | 0.144182 | 0.139 | -0.00518181 |
| Max per-wavelength RMSE | 0.267666 | 0.260606 | -0.00706009 |

Model 1B has lower per-sample RMSE on 72.2% of held-out spectra. The mean paired per-sample delta RMSE (B - A) is -0.00772708 with standard deviation 0.0145376.

## Computation

- Output space: log10 eclipse-depth units.
- Samples: 97 paired HF/LF spectra.
- Wavelength bins: 195.
- CV protocol: leave-one-out, `n_splits = 97`, seed `0`.
- Each fold fits on 96 paired spectra and predicts the one held-out spectrum.
- Model 1A estimates one train-fold global rho pooled over all wavelengths.
- Model 1B estimates one train-fold rho per wavelength.
- Fold prediction formula: `mu_HF[j] + rho[j] * (YLF_holdout[j] - mu_LF[j])`.
- All train-fold moments (`mu_LF`, `sd_LF`, `mu_HF`) and rho values are computed without the held-out row.
- Full-data global rho, reported only as a diagnostic: `0.46029`.
- RMSE is computed against held-out `YHF`; NRMSE divides pooled RMSE by the full true-spectrum range in this output space.

## Files

- `cv_per_sample.csv`: per-held-out-spectrum RMSE and paired delta.
- `cv_per_wavelength.csv`: per-wavelength RMSE, rho, and paired delta.
- `cv_summary.json`: machine-readable version of this summary.
- `01_model_1a_vs_1b.png`: diagnostic figure.
