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
- `YLF` (97×195, paired) is used only for the Model 1A/1B baselines and the
  per-wavelength rho diagnostic.

The full augmented grid has \((10000 + 97) \times 195 \approx 1.97\)M scalar
points — far beyond the exact-fit memory gate (the 25-hyperparameter kernel
allows only n ≈ 5600 points on a 17 GB machine). BOTH fidelities are
therefore subsampled: all 97 HF samples on a stride-`LAMBDA_STRIDE`
wavelength subgrid, plus `LF_SAMPLE_SIZE` LF samples on the
half-stride-offset subgrid (offsetting doubles the distinct wavelengths seen
in training). The remaining ~7/8 of wavelengths are never trained on and are
predicted purely by the fitted \(\lambda\) kernel — that interpolation is
exactly what Model 2 is meant to test. Outputs are standardized per
wavelength by LF column moments (the single-GP analog of Model 1's
per-wavelength `normalize_y=True`; leakage-free and rho-invariant).

## Workflow

Run `make model2` (= `python -m research.modelling.02_augmented_wavelength`).
It executes the numbered scripts in order and writes results plus provenance
under `results/modelling/02_augmented_wavelength/`:

- `00_benchmark_model2_fit.py` — memory cap and timing ladder; records the
  recommended production and CV `lf_sample_size` in
  `benchmark/model2_fit_benchmark.json`. The fit/CV scripts pin their
  constants from this file with the derivation in a comment.
- `01_fit_model2.py` — production fits on linear and log10 spectra:
  fitted layer, hyperparameter table (including the \(\lambda\) length
  scale in µm), timing, and diagnostics in `linear/` and `log10/`.
- `02_cv_model2_vs_model1.py` — 5-fold CV (`CV_FULL_MODEL_SPLITS`, matching
  `validation/02_full_cv`; LOO is infeasible because every fold refits the
  joint GP) against Model 1A/1B closed form and optionally the Model 1
  joint GP at its 200-LF testing config, on both scales plus the
  back-transformed log-vs-linear contrast, in `cv/`.

Random sampling uses `exoplanets_mf.reproducibility.RANDOM_SEED`. All paired
comparisons share the identical KFold assignment (asserted at runtime).

## Caveats for reading the comparison

- Model 1A/1B consume the held-out sample's paired LF spectrum at prediction
  time; Model 2 predicts from atmospheric inputs alone (strictly less
  information per test sample).
- Model 2's CV runs at a reduced configuration relative to its production
  fit; the Model 1 joint GP baseline runs at its testing configuration.
- A single stationary RBF over \(\lambda\), shared across all \(\theta\), may
  be too rigid for the band/block correlation structure seen in
  `exploratory/04_wavelength_in_design.py`; Model 2 losing to Model 1B would
  itself be a valid finding.
