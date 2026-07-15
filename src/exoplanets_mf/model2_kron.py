"""Model 2K: exact two-stage Kronecker reformulation of Model 2.

Model 2 subsamples 3,385 of ~1.97M available (theta, lambda) points because
sklearn's exact GP cannot exploit the design structure. But every theta is
evaluated at every lambda (a complete Cartesian grid) and the SE-ARD kernel
over the augmented input factorizes, so per stage the covariance is a
Kronecker product and exact inference on ALL points costs O(n^3 + m^3)
(kron_gp). The joint two-fidelity covariance is a SUM of Kronecker products
and does not jointly diagonalize, so Model 2K instead uses the Le Gratiet
recursion, which under the nested co-located design (all 97 HF inputs carry
observed paired LF spectra) exactly factorizes the joint AR(1) likelihood
(Le Gratiet & Garnier 2014):

    stage L:      f_L ~ GP(0, sigma_L^2 k_thetaL k_lambdaL) + noise,
                  trained on the stacked LF grid [XLF_10k; XHF] x lambda
                  (stacking the 97 paired rows is what makes the design
                  strictly nested);
    stage delta:  D = Y_HF - rho * Y_LF_paired,
                  delta ~ GP(0, sigma_d^2 k_thetad k_lambdad) + noise,
                  with scalar rho profiled in closed form (kron_gp).

Stage delta is STRICTLY zero-mean (unlike Model 2's ``normalize_y=True``).
Stage L sees no HF data, so it is fitted ONCE and reused across CV folds --
the same "LF data is always fully available" convention as Model 2's LF
subsample (and Models 1A/1B, which consume observed LF at held-out inputs).

Prediction at new theta, all lambdas: mean_H = rho mean_L + mean_delta and
var_H = rho^2 var_L + var_delta (independent stages; fitted noise included,
parity with Model 2's predictive stds).

Preprocessing parity with Model 2: per-wavelength output standardization by
LF column moments (from the 10k LF spectra only -- leakage-free), theta
standardized by a StandardScaler fitted on the stacked LF design, lambda
standardized by the moments of the full 195-point grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.kron_gp import (
    KRON_N_RESTARTS,
    KronGPFit,
    SEKernel1D,
    fit_kron_gp,
    posterior_mean_var,
)
from exoplanets_mf.model2 import lf_column_moments

# Screening restarts on a 2,000-row theta subset keep the full stage-L fit at
# ~2 h instead of ~2 days; the final polish is exact on all rows, only the
# restart INITIALIZATION uses the subset. None = literal full-grid restarts.
STAGE_L_SCREEN_SIZE = 2000


# ---------------------------------------------------------------------------
# 1. Stage L: the LF grid GP (fitted once, HF-free)
# ---------------------------------------------------------------------------

@dataclass
class StageLFit:
    """Everything reusable across CV folds: scalers, moments, and the LF GP."""

    theta_scaler: StandardScaler
    lambda_mu: float
    lambda_sd: float
    column_mu: np.ndarray
    column_sd: np.ndarray
    gp: KronGPFit
    screen_size: int | None
    screen_records: list[dict]     # per-restart records of the screening fit
    screen_seconds: float
    fit_seconds: float             # screening + polish wall time

    def standardize_theta(self, X: np.ndarray) -> np.ndarray:
        return self.theta_scaler.transform(np.asarray(X, dtype=float))

    def standardize_columns(self, Y: np.ndarray) -> np.ndarray:
        return (np.asarray(Y, dtype=float) - self.column_mu) / self.column_sd


def build_stage_l_design(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_lf_paired: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the LF-only design with the paired LF rows at the HF inputs.

    The stack makes the design strictly nested (every HF theta is in the LF
    design), the condition under which the two-stage recursion exactly
    factorizes the joint likelihood. Duplicated theta rows would make
    K_theta exactly singular, so they are rejected.
    """
    X_lf = np.asarray(X_lf, dtype=float)
    X_hf = np.asarray(X_hf, dtype=float)
    Y_lf = np.asarray(Y_lf, dtype=float)
    Y_lf_paired = np.asarray(Y_lf_paired, dtype=float)
    if X_lf.shape[0] != Y_lf.shape[0] or X_hf.shape[0] != Y_lf_paired.shape[0]:
        raise ValueError("inputs and LF spectra must have matching row counts")
    if Y_lf.shape[1] != Y_lf_paired.shape[1]:
        raise ValueError("LF spectra must share one wavelength grid")
    X = np.vstack((X_lf, X_hf))
    Y = np.vstack((Y_lf, Y_lf_paired))
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("stage-L design contains non-finite values")
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("stage-L design contains duplicated theta rows")
    return X, Y


def fit_stage_l(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_lf_paired: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    n_restarts: int = KRON_N_RESTARTS,
    screen_size: int | None = STAGE_L_SCREEN_SIZE,
    lambda_kernel: SEKernel1D | None = None,
    maxiter: int = 200,
) -> StageLFit:
    """Fit the LF grid GP with screen-then-polish restarts.

    Column moments come from ``Y_lf`` (the LF-only spectra) ONLY -- exact
    parity with Model 2's lf_column_moments(YLF_10k), leakage-free either
    way because no HF output enters.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    column_mu, column_sd = lf_column_moments(np.asarray(Y_lf, dtype=float))
    X, Y = build_stage_l_design(X_lf, Y_lf, X_hf, Y_lf_paired)

    theta_scaler = StandardScaler().fit(X)
    X_std = theta_scaler.transform(X)
    lambda_mu = float(wavelengths.mean())
    lambda_sd = float(wavelengths.std())
    t = (wavelengths - lambda_mu) / lambda_sd
    Y_std = (Y - column_mu) / column_sd

    t_start = time.perf_counter()
    screen_records: list[dict] = []
    screen_seconds = 0.0
    initial_log_eta = None
    polish_restarts = n_restarts
    if screen_size is not None and screen_size < X_std.shape[0]:
        subset = np.random.default_rng(seed).choice(
            X_std.shape[0], size=screen_size, replace=False
        )
        screen_fit = fit_kron_gp(
            X_std[subset],
            t,
            Y_std[subset],
            seed=seed,
            n_restarts=n_restarts,
            lambda_kernel=lambda_kernel,
            maxiter=maxiter,
        )
        screen_records = screen_fit.restart_records
        screen_seconds = screen_fit.fit_seconds
        initial_log_eta = screen_fit.log_eta
        polish_restarts = 1

    gp = fit_kron_gp(
        X_std,
        t,
        Y_std,
        seed=seed,
        n_restarts=polish_restarts,
        lambda_kernel=lambda_kernel,
        initial_log_eta=initial_log_eta,
        maxiter=maxiter,
    )
    fit_seconds = time.perf_counter() - t_start

    return StageLFit(
        theta_scaler=theta_scaler,
        lambda_mu=lambda_mu,
        lambda_sd=lambda_sd,
        column_mu=column_mu,
        column_sd=column_sd,
        gp=gp,
        screen_size=screen_size,
        screen_records=screen_records,
        screen_seconds=screen_seconds,
        fit_seconds=fit_seconds,
    )


# ---------------------------------------------------------------------------
# 2. The full two-stage layer
# ---------------------------------------------------------------------------

@dataclass
class Model2KronLayer:
    """Fitted Model 2K: shared stage L plus one stage-delta residual GP."""

    wavelengths: np.ndarray
    stage_l: StageLFit
    stage_delta: KronGPFit
    fit_seconds: float             # stage-delta fit only (stage L may be shared)

    @property
    def rho(self) -> float:
        return float(self.stage_delta.rho)


def fit_model2_kron(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    Y_lf_paired: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    n_restarts: int = KRON_N_RESTARTS,
    stage_l: StageLFit | None = None,
    lambda_kernel: SEKernel1D | None = None,
    screen_size: int | None = STAGE_L_SCREEN_SIZE,
) -> Model2KronLayer:
    """Fit Model 2K; pass a prefit ``stage_l`` to reuse it (CV folds do)."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if stage_l is None:
        stage_l = fit_stage_l(
            X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
            seed=seed, n_restarts=n_restarts,
            screen_size=screen_size, lambda_kernel=lambda_kernel,
        )

    t0 = time.perf_counter()
    stage_delta = fit_kron_gp(
        stage_l.standardize_theta(X_hf),
        stage_l.gp.t,
        stage_l.standardize_columns(Y_hf),
        Y_pair=stage_l.standardize_columns(Y_lf_paired),
        seed=seed,
        n_restarts=n_restarts,
        lambda_kernel=lambda_kernel,
    )
    fit_seconds = time.perf_counter() - t0

    return Model2KronLayer(
        wavelengths=wavelengths,
        stage_l=stage_l,
        stage_delta=stage_delta,
        fit_seconds=fit_seconds,
    )


def predict_hf_model2_kron(
    layer: Model2KronLayer, X_new: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """HF spectra at all wavelengths: rho * stage L + stage delta.

    Returns (means, stds) of shape (n_new, n_wavelengths) in original data
    units; the stage variances add because the stages are independent GPs.
    """
    X_std = layer.stage_l.standardize_theta(X_new)
    mean_l, var_l = posterior_mean_var(layer.stage_l.gp, X_std, include_noise=True)
    mean_d, var_d = posterior_mean_var(layer.stage_delta, X_std, include_noise=True)
    rho = layer.rho
    mean_std = rho * mean_l + mean_d
    std_std = np.sqrt(rho**2 * var_l + var_d)
    means = mean_std * layer.stage_l.column_sd + layer.stage_l.column_mu
    stds = std_std * layer.stage_l.column_sd
    return means, stds


# ---------------------------------------------------------------------------
# 3. Cross-validation adapter
# ---------------------------------------------------------------------------

def cv_predict_model2_kron(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    Y_lf_paired: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    n_restarts: int = KRON_N_RESTARTS,
    stage_l: StageLFit | None = None,
    progress: bool = False,
    **stage_l_kwargs,
) -> CVPredictions:
    """Out-of-fold Model 2K predictions on the paired HF rows.

    Stage L is HF-free, so it is fitted once up front (or passed in) and
    shared by all folds; each fold refits only stage delta on its training
    HF rows and their paired LF spectra. Fold assignment is the shared
    KFold(shuffle, random_state=seed) of cv_predict, identical to every
    other model at the same (n_samples, n_splits, seed).
    """
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
    Y_lf_paired = np.asarray(Y_lf_paired, dtype=float)
    if stage_l is None:
        stage_l = fit_stage_l(
            X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
            seed=seed, n_restarts=n_restarts, **stage_l_kwargs,
        )

    def fit_fold(train_idx: np.ndarray) -> Model2KronLayer:
        if progress:
            print(f"  model 2K fold fit on {len(train_idx)} HF samples", flush=True)
        return fit_model2_kron(
            X_lf, Y_lf,
            X_hf[train_idx], Y_hf[train_idx], Y_lf_paired[train_idx],
            wavelengths,
            seed=seed, n_restarts=n_restarts, stage_l=stage_l,
        )

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model2_kron(
            layer, X_hf[test_idx]
        ),
        n_samples=X_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )
