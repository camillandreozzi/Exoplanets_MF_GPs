# 02: Wavelength-augmented joint MF-GP (Model 2)

## Model

Wavelength is a 10th input: the augmented input is \(z = (\theta, \lambda)\)
and ONE scalar-valued AR(1) multi-fidelity GP is fitted over the joint
atmosphere–wavelength domain per output scale,

\[
f_H(\theta, \lambda) = \rho\, f_L(\theta, \lambda) + \delta(\theta, \lambda),
\]

with a single scalar \(\rho\) and one joint marginal likelihood over the
stacked LF + HF augmented design (`exoplanets_mf.model2`, reusing Model 1's
`AR1MultiFidelityKernel` at `n_dims = 10`). Wavelength dependence of the
LF–HF link is absorbed by \(\delta(\theta, \lambda)\). This contrasts with
Model 1 (`01_per_wavelength_ar1`), which fits 195 independent per-wavelength
models. Prediction expands each new \(\theta^*\) to all 195 wavelengths and
reads the per-wavelength marginals (mean, std) of the joint posterior.

## Input data and subsampling

- `XHF`, `YHF` (97×9, 97×195) and `XLF_10k`, `YLF_10k` (10000×9, 10000×195).
- `YLF` (97×195, paired) is no longer consumed by any model or baseline; the
  per-wavelength rho diagnostic overlay now reads the fitted Model 1B MF-GP
  layer's rho_j from its results CSV.

The full augmented grid has \((10000 + 97) \times 195 \approx 1.97\)M scalar
points — far beyond the exact-fit memory gate (the 25-hyperparameter kernel
allows only n ≈ 5600 points on a 17 GB machine). BOTH fidelities are
therefore subsampled: all 97 HF samples on a stride-`LAMBDA_STRIDE`
wavelength subgrid, plus `LF_SAMPLE_SIZE` LF samples on the
half-stride-offset subgrid (offsetting doubles the distinct wavelengths seen
in training). `LF_SAMPLE_SIZE` is the project-wide `LF_SUBSAMPLE_SIZE` — the
**one canonical LF subsample shared by every model and CV** (identical LF rows
to Model 1). Because the augmented design costs `LF_SAMPLE_SIZE ×
n_wavelengths`, the wavelength stride is **derived** from a runtime point
budget (`model2.derive_lambda_stride`, finest grid that fits) rather than
pinned — production uses a larger budget than CV, which refits per fold. Many
wavelengths are never trained on and are predicted purely by the fitted
\(\lambda\) kernel — that interpolation is exactly what Model 2 is meant to
test. Outputs are standardized per
wavelength by LF column moments (the single-GP analog of Model 1's
per-wavelength `normalize_y=True`; leakage-free and rho-invariant).

## Workflow

Run `make modelling_sklearn` for the full sklearn/custom-kernel comparison. The
Model 2 portion is `python -m research.modelling.02_augmented_wavelength`; it
executes the numbered non-GPBoost scripts in order and writes linear results plus
provenance under `results/modelling/02_augmented_wavelength/`, with log10-space
artifacts under `results/log_modelling/02_augmented_wavelength/`:

- `00_benchmark_model2_fit.py` — memory cap and timing ladder in
  `benchmark/model2_fit_benchmark.json`. The fit/CV scripts now fix the LF
  count at the shared `LF_SUBSAMPLE_SIZE` and derive the wavelength stride from
  the runtime point budgets (`MODEL2_MAX_AUGMENTED_POINTS` /
  `MODEL2_CV_MAX_AUGMENTED_POINTS` in `model2.py`).
- `01_fit_model2.py` — production fits on linear and log10 spectra:
  fitted layer, hyperparameter table (including the \(\lambda\) length
  scale in µm), timing, and diagnostics in `results/modelling/.../linear/`
  and `results/log_modelling/.../log10/`.
- `02_cv_model2_vs_model1.py` — 5-fold CV (`CV_FULL_MODEL_SPLITS`, matching
  `validation/02_full_cv`; LOO is infeasible because every fold refits the
  joint GP) against the Model 1A (global-rho) and Model 1B (per-wavelength
  rho) MF-GP layers — all sharing the one canonical LF subsample — on both
  scales plus the back-transformed log-vs-linear contrast, split between
  `results/modelling/.../cv/linear/` and `results/log_modelling/.../cv/`.

Random sampling uses `exoplanets_mf.reproducibility.RANDOM_SEED`. All paired
comparisons share the identical KFold assignment (asserted at runtime).

## Caveats for reading the comparison

- Every model predicts from atmospheric inputs alone; no model consumes the
  held-out sample's paired LF spectrum.
- Model 2's CV runs at a coarser wavelength stride than its production fit
  (tighter per-fold budget); all models share the one canonical LF subsample.
- A single stationary RBF over \(\lambda\), shared across all \(\theta\), may
  be too rigid for the band/block correlation structure seen in
  `exploratory/04_wavelength_in_design.py`; Model 2 losing to Model 1B would
  itself be a valid finding.
