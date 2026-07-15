# Log-modelling

Mirror of `research/modelling/` with the spectra (the y values) log10-
transformed via `exoplanets_mf.transforms.log10_spectra` immediately after
loading. The atmospheric inputs X are untouched. Motivation and diagnostics:
`research/exploratory/05_log_spectra_exploration.py` (eclipse depths span ~4
orders of magnitude with strong right skew that log10 removes; HF/LF
correlation is also slightly higher on log scale).

Conventions:

- log10. The simulated spectra are strictly positive, so the transform is exact.
- Every experiment here is a near-copy of its `research/modelling/`
  counterpart; the only diffs are the output directory
  (`results/log_modelling/`), the y transform after loading, and axis labels.
- Model quantities fitted on log10 depths (rho, GP
  hyperparameters) are *different quantities* from their linear-scale
  counterparts. Compare models across scales only through back-transformed
  predictions.

## Back-transforming predictions

Predictions are in log10 space. For a GP predictive mean `m` and standard
deviation `s` (both log10):

- `10**m` is the predictive **median** in linear units (the mean of the
  implied log-normal is `10**m * exp((ln(10) * s)**2 / 2)`);
- a 95% interval back-transforms to `[10**(m - 1.96 s), 10**(m + 1.96 s)]`.

Use `exoplanets_mf.transforms.inverse_log10_spectra` for the point
back-transform.

Run the joint MF-GP experiment with `make log-modelling`.
