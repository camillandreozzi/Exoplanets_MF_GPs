# LOO CV Report: Model 1A vs Model 1B

## Result

Model 1B (per-wavelength rho) has the lower pooled held-out RMSE in this comparison.

| Metric | Model 1A | Model 1B | Delta B - A |
|---|---:|---:|---:|
| Pooled RMSE | 0.000288375 | 0.000280369 | -8.00659e-06 |
| Pooled NRMSE | 0.0431439 | 0.041946 | -0.00119787 |
| Mean per-sample RMSE | 0.000257179 | 0.000246433 | -1.07466e-05 |
| Median per-sample RMSE | 0.00021813 | 0.000209554 | -8.5764e-06 |
| Mean per-wavelength RMSE | 0.000249342 | 0.000241776 | -7.56637e-06 |
| Max per-wavelength RMSE | 0.00087246 | 0.000795978 | -7.64816e-05 |

Model 1B has lower per-sample RMSE on 73.2% of held-out spectra. The mean paired per-sample delta RMSE (B - A) is -1.07466e-05 with standard deviation 2.19241e-05.

## Computation

- Output space: original eclipse-depth units.
- Samples: 97 paired HF/LF spectra.
- Wavelength bins: 195.
- CV protocol: leave-one-out, `n_splits = 97`, seed `0`.
- Each fold fits on 96 paired spectra and predicts the one held-out spectrum.
- Model 1A estimates one train-fold global rho pooled over all wavelengths.
- Model 1B estimates one train-fold rho per wavelength.
- Fold prediction formula: `mu_HF[j] + rho[j] * (YLF_holdout[j] - mu_LF[j])`.
- All train-fold moments (`mu_LF`, `sd_LF`, `mu_HF`) and rho values are computed without the held-out row.
- Full-data global rho, reported only as a diagnostic: `0.350027`.
- RMSE is computed against held-out `YHF`; NRMSE divides pooled RMSE by the full true-spectrum range in this output space.

## Files

- `cv_per_sample.csv`: per-held-out-spectrum RMSE and paired delta.
- `cv_per_wavelength.csv`: per-wavelength RMSE, rho, and paired delta.
- `cv_summary.json`: machine-readable version of this summary.
- `01_model_1a_vs_1b.png`: diagnostic figure.
