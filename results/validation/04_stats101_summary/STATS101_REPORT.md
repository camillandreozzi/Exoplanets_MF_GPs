# Statistics-101 Summary: Models 1A, 1B, 1 joint, and 2

## What each statistic means

- **R^2** (coefficient of determination): fraction of the variance in the true
  spectra explained by the predictions; 1 is perfect, 0 means no better than
  always predicting the overall mean.
- **Pearson r**: linear correlation between predicted and actual values;
  1 is a perfect positive linear relationship.
- **MAE** (mean absolute error): average size of the errors, in the units of
  the view; robust to a few large misses.
- **RMSE** (root mean squared error): like MAE but penalizes large errors more;
  identical to the pooled RMSE reported by the CV workflows.
- **Mean error (bias)**: average of prediction minus actual; near zero means
  the model neither systematically over- nor under-predicts.

All statistics are pooled over every held-out point
(97 spectra x 195 wavelength bins).

## Results

### Linear (original eclipse-depth units)

| Model | R^2 | Pearson r | MAE | RMSE | Mean error (bias) |
|---|---:|---:|---:|---:|---:|
| Model 1A (global rho) | 0.8872 | 0.9419 | 0.000192406 | 0.000288621 | 5.41178e-07 |
| Model 1B (per-wavelength rho) | 0.8932 | 0.9451 | 0.000183784 | 0.000280735 | 1.78387e-06 |
| Model 1 joint (per-wavelength MF-GP) | 0.9705 | 0.9852 | 8.68973e-05 | 0.000147521 | 1.56817e-06 |
| Model 2 (wavelength-augmented joint MF-GP) | 0.8904 | 0.9449 | 0.000164283 | 0.000284395 | 3.62375e-05 |

### log10 (log10 eclipse-depth units)

| Model | R^2 | Pearson r | MAE | RMSE | Mean error (bias) |
|---|---:|---:|---:|---:|---:|
| Model 1A (global rho) | 0.8974 | 0.9473 | 0.113334 | 0.153251 | -4.4001e-05 |
| Model 1B (per-wavelength rho) | 0.9043 | 0.9509 | 0.107245 | 0.148054 | 0.000191185 |
| Model 1 joint (per-wavelength MF-GP) | 0.9665 | 0.9831 | 0.0557127 | 0.0875635 | 0.00356829 |
| Model 2 (wavelength-augmented joint MF-GP) | 0.9409 | 0.9702 | 0.0818309 | 0.116281 | 0.00795624 |

### log10 back-transformed (original eclipse-depth units)

| Model | R^2 | Pearson r | MAE | RMSE | Mean error (bias) |
|---|---:|---:|---:|---:|---:|
| Model 1A (global rho) | 0.8931 | 0.9458 | 0.000184631 | 0.00028088 | -3.10178e-05 |
| Model 1B (per-wavelength rho) | 0.8999 | 0.9492 | 0.000176711 | 0.000271844 | -2.74388e-05 |
| Model 1 joint (per-wavelength MF-GP) | 0.9670 | 0.9834 | 9.15728e-05 | 0.000156055 | -1.23003e-06 |
| Model 2 (wavelength-augmented joint MF-GP) | 0.9093 | 0.9536 | 0.000150179 | 0.000258692 | 1.53958e-06 |

## Computation

- Predictions are the out-of-fold CV predictions saved by
  `research/modelling/02_augmented_wavelength/02_cv_model2_vs_model1.py`
  (`cv_predictions.npz`); nothing is refitted here.
- CV protocol: 5-fold over 97 paired HF samples, seed `0`;
  the same fold assignment is shared by all models and both scales.
- The log10 view scores log10-scale fits against `log10(YHF)` in log10 units,
  so its error magnitudes are not comparable to the other two views.
- The back-transformed view maps log10 predictions through `10**prediction`
  (the median of the implied log-normal distribution, not its mean) and scores
  them in original eclipse-depth units, so all models are comparable in one
  table.
- Recomputed RMSEs are asserted to match `cv_summary.json` from the source CV
  run before anything is written.

## Files

- `stats101_summary.csv`: machine-readable version of the tables above.
- `cv_predictions_long.csv`: one row per held-out point (sample, spectrum,
  fold, wavelength) with the actual value and every model's prediction in all
  three scale views.
- `01_pred_vs_actual.png`: predicted-vs-actual scatter grid with the identity
  line (perfect predictions fall on the dashed diagonal).
- `02_residuals.png`: residual histograms (linear fits) and residual boxplots
  per model and fitting scale, in original units.
