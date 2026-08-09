# 01: Sample-81-only holdout validation of the joint MF-GP

> **Superseded by [`../02_full_cv/`](../02_full_cv/README.md)**, which
> cross-validates the same model over **all 97 HF samples** instead of this
> single fixed holdout. Kept for reference; its sample-81 result is a
> consistency anchor for the corresponding row of the full CV.

This workflow excludes 0-indexed HF row 80 (`spectrum 81`,
`Kzz = 8.47701413791753`) and refits BOTH joint multi-fidelity variants —
Model 1B (one free rho per wavelength) and Model 1A (one shared rho,
warm-started from the Model 1B holdout fit) — using:

- the same LF subsample size as the modelling fit scripts;
- the remaining 96 `XHF`/`YHF` rows.

The held-out HF spectrum is predicted directly from its atmospheric inputs.
Its paired LF spectrum is not supplied to the predictor.

`01_fit_excluding_holdout.py` performs the two joint refits.
`02_predict_and_evaluate.py` reports, per model, residuals,
posterior-standardized residuals, 95% interval coverage, and
range-normalized error, plus a 1A-vs-1B comparison summary and overlay
plot.

Outputs are written under
`results/validation/01_loo_holdout_validation/{fit,evaluation}/`.
