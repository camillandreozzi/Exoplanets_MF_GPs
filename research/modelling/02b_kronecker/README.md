# 02b: Exact Kronecker two-stage MF-GP (Model 2K)

## Model

Model 2K removes Model 2's data subsampling. The design is a COMPLETE
Cartesian grid (every \(\theta\) evaluated at every \(\lambda\)) and the
SE-ARD kernel over \(z = (\theta, \lambda)\) factorizes as
\(k_\theta \cdot k_\lambda\), so per stage the covariance of the flattened
\(n \times m\) output grid is a Kronecker product and EXACT inference on all
\(nm\) points costs \(O(n^3 + m^3)\) (`exoplanets_mf.kron_gp`), not
\(O(n^3 m^3)\).

The joint two-fidelity covariance is a SUM of Kronecker products and does
not jointly diagonalize, so Model 2K uses the Le Gratiet recursion, which
under the nested co-located design (all 97 HF inputs carry observed paired
LF spectra, stacked into the LF design) exactly factorizes the joint AR(1)
likelihood (Le Gratiet & Garnier 2014):

- **stage L**: zero-mean GP on the stacked LF grid, \(10{,}097 \times 195\)
  points, exactly;
- **stage \(\delta\)**: zero-mean GP on the HF residual grid
  \(D = Y_{HF} - \rho\, Y_{LF,\text{paired}}\) (\(97 \times 195\)), with the
  scalar \(\rho\) profiled in closed form (GLS) inside the marginal
  likelihood. \(\rho\) may be negative and is never log-parametrized.

Prediction: \(\mu_H = \rho\,\mu_L + \mu_\delta\),
\(\sigma_H^2 = \rho^2 \sigma_L^2 + \sigma_\delta^2\) (independent stages,
fitted noise included). Stage \(\delta\) is STRICTLY zero-mean, unlike
Model 2's `normalize_y=True`. Preprocessing otherwise matches Model 2:
per-wavelength LF-moment output standardization, standardized \(\theta\)
and \(\lambda\), same hyperparameter bounds, both output scales.

## Optimization

L-BFGS-B in log space with ANALYTIC gradients derived in the eigenbasis
(verified against finite differences in `tests/test_kron_gp.py`), 10
restarts. Stage L uses screen-then-polish: the 10 restarts run on a
2,000-row \(\theta\) subset (~1 s per objective evaluation), then the best
optimum warm-starts one polish on the full 10,097-row grid (~2.5 min per
evaluation, dominated by `eigh` of the \(\theta\) factor). Only the restart
initialization is screened; the reported fit is exact on all rows.

## Workflow

Run `make model2k` (= `python -m research.modelling.02b_kronecker`). It
executes the numbered scripts in order and writes results plus provenance
under `results/modelling/02b_kronecker/`:

- `01_fit_model2k.py` — exact production fits on linear and log10 spectra:
  hyperparameter table (including \(\rho\), the \(\lambda\) length scales in
  µm, and bound-pinning flags), restart-robustness spread across seeds 0-4
  (full stage-\(\delta\) refits + stage-L screening refits), fitted layer,
  timing (including the monitored `eigh(10097)` cost), and diagnostics with
  the 5 µm NIRCam/MIRI break marked, in `linear/` and `log10/`.
- `02_cv_model2k.py` — extends the STORED 5-fold CV comparison of
  `02_augmented_wavelength/cv` with Model 2K as an additional entry: the
  baselines (Model 1A/1B, Model 1 joint GP, subsampled Model 2) are loaded
  read-only from their `cv_predictions.npz` and their fold assignment is
  asserted equal before any metric. Nothing under
  `02_augmented_wavelength/` is modified — the old-vs-new Model 2 contrast
  is itself a result. Stage L (HF-free) is reused from the production fit
  across all folds; each fold refits only stage \(\delta\) (seconds).

Random sampling uses `exoplanets_mf.reproducibility.RANDOM_SEED` (0); seeds
1-4 appear only in `restart_robustness.csv`.

## Caveats

- The exact recursion-equals-joint-likelihood argument assumes deterministic
  LF outputs; with learned LF noise it is an approximation whose error is
  second-order for a near-noiseless simulator.
- The stored Model 2 CV rows come from its reduced stride-13 configuration
  and the Model 1 joint GP from its 200-LF testing configuration (their
  caveats still apply); Model 1A/1B consume the held-out sample's paired LF
  spectrum at prediction time.
- The \(\lambda\) factor is still one stationary SE kernel; `kron_gp` keeps
  the \(\lambda\) kernel pluggable (Kronecker needs no stationarity), so a
  nonstationary two-block NIRCam/MIRI kernel can be swapped in later.
