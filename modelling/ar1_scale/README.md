Edit the settings at the top of `cv_5fold.py`, then run from the repository root:

```sh
python3 modelling/ar1_scale/cv_5fold.py
```

There are no command-line arguments. The script contains data loading, HF
splitting, the custom scikit-learn AR1 kernel, both model fitting utilities,
prediction, metrics, and plots. It has no dependency on `src.model_1`, GPBoost,
or the kernel comparison module. Its kernel is copied from
`kernel_comparison/sklearn_ar1.py` and extended to support fixed rho.

Defaults: five shuffled HF folds, seed 2026, all 97 HF samples, all 195
wavelengths, all 10,000 LF rows, outputs in `results/ar1_scale`.
`LF_SAMPLE_SIZE` and `RESPONSE_INDEXES` optionally select subsets; `None`
means all available rows/wavelengths. Both models use identical folds and LF
samples, with all selected LF rows retained in every training fold.

`fit_model1a` fixes rho to `FIXED_RHO` (default 1). `fit_model1b` estimates
rho independently for each wavelength within each training fold. Both fit
`GaussianProcessRegressor` with the custom AR1 Matérn kernel plus WhiteKernel.
Fixed rho is excluded from the optimization vector, bounds, and gradients.
Free rho is optimized on its natural signed scale within `RHO_BOUNDS`.
Other kernel initial values, bounds, jitter, optimizer iterations, restarts,
and plot resolution are adjustable at the top of the script.

Both models use a zero-mean GP on standardized responses, with no linear
baseline. Every atmospheric input is centered and divided by its standard
deviation using training HF statistics, applied to both LF and HF rows.
Each wavelength's responses are centered and scaled using one shared mean
and standard deviation from its training LF+HF rows. Constant variables use
a scale of 1. Validation uses the stored training statistics only.

The fidelity flag stays 0/1 because it selects the LF/HF kernel component;
it is categorical, not a continuous predictor. Wavelength identifies each
separate GP and is not an input variable. Predictions are transformed back
using the training response mean and scale; variances use the scale squared.
Metrics and spectra remain in original response units.

These are exact dense scikit-learn GPs. The default experiment requires 1,950
fits on roughly 10,078 training rows each; memory and runtime can be large
(cubic factorization cost, quadratic covariance/gradient storage).
`LF_SAMPLE_SIZE` controls the training size explicitly.

CSV outputs include fold checkpoints, pooled out-of-fold predictions,
covariance parameters (including rho), metrics per wavelength, fold, and
spectrum, and an overall summary. The wavelength comparison reports NRMSE,
MAE, RMSE, and B-minus-A differences (negative favors B). NRMSE divides RMSE
by the observed target population standard deviation (`ddof=0`) within each
reported group; zero standard deviation gives NaN.
The summary distinguishes pooled metrics from mean per-wavelength metrics.
Noise and signal variances are saved in original response units squared;
length scales refer to standardized input coordinates.

`predicted_spectra_and_residuals.pdf` contains one page per held-out HF sample
with observations, both predictions, and predicted-minus-observed residuals.
The matching PNG previews the first sample by source index.
`metrics_per_wavelength.png` compares all three metrics;
`rho_per_wavelength.png` shows estimated coefficients for every fold.

Run the synthetic integration and kernel gradient checks with:

```sh
python3 -m unittest discover -s tests -p test_ar1_scale.py
```
