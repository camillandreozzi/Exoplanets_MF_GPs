"""Model 2: one joint AR(1) multi-fidelity GP with wavelength as an input.

Model 1 fits 195 independent per-wavelength MF-GPs. Model 2 instead treats
wavelength as a 10th input dimension: the augmented input is z = (theta,
lambda) and a single scalar-valued MF-GP is fitted over the joint
atmosphere-wavelength domain,

    f_H(theta, lambda) = rho * f_L(theta, lambda) + delta(theta, lambda),

with one scalar rho and one joint marginal likelihood over the stacked
LF + HF augmented design (same AR1MultiFidelityKernel as Model 1, just with
n_dims=10). Wavelength dependence of the LF-HF link is absorbed by
delta(theta, lambda).

The full augmented grid (10,000 x 195 LF + 97 x 195 HF ~ 1.97M scalar
points) is far beyond the exact-fit memory gate, so BOTH fidelities are
subsampled: all HF samples on a wavelength-stride subgrid, plus an LF sample
subset on a half-stride-offset wavelength subgrid (offsetting doubles the
distinct wavelengths seen in training). Prediction is at all 195
wavelengths -- the lambda dimension of the SE-ARD kernel interpolates
between training wavelengths, which is exactly what Model 2 is meant to
test. Predictions are per-wavelength marginals (mean, std), so they feed the
exoplanets_mf.cv engine unchanged.

Outputs are standardized per wavelength by LF column moments before
flattening (the single-GP analog of Model 1's per-wavelength
``normalize_y=True``): it removes
the orders-of-magnitude heteroscedasticity across wavelengths that would
otherwise break the stationary lambda kernel, uses no HF information (no CV
fold leakage), and leaves rho invariant (a common per-column scale on both
fidelities cancels in the AR(1) relation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.mf_gp import (
    MF_GP_ALPHA,
    MF_GP_N_RESTARTS_OPTIMIZER,
    MF_GP_NORMALIZE_Y,
    N_INPUT_DIMS,
    _append_fidelity,
    assert_exact_fit_fits_in_memory,
    make_joint_mf_kernel,
    select_lf_subsample,
)

N_AUGMENTED_DIMS = N_INPUT_DIMS + 1  # 9 atmospheric inputs + wavelength


def model2_n_theta(n_input_dims: int = N_INPUT_DIMS) -> int:
    """Hyperparameter count of the augmented joint kernel (25 for 9 inputs).

    Feeds the same memory gate as Model 1: the exact-fit footprint is
    exact_fit_memory_bytes(n_points, model2_n_theta()).
    """
    return len(make_joint_mf_kernel(n_input_dims + 1).theta)


# ---------------------------------------------------------------------------
# 1. Augmented design construction
# ---------------------------------------------------------------------------

def select_wavelength_subgrid(
    n_wavelengths: int, *, stride: int, offset: int = 0
) -> np.ndarray:
    """Every ``stride``-th wavelength index, starting at ``offset``."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, found {stride}")
    if not 0 <= offset < stride:
        raise ValueError(
            f"offset must lie in [0, stride); found offset={offset}, "
            f"stride={stride}"
        )
    return np.arange(offset, n_wavelengths, stride)


# Augmented-design point budgets. Since the shared LF subsample is fixed at
# LF_SUBSAMPLE_SIZE (200) rows -- much larger than Model 2's old bespoke count --
# the wavelength stride is derived from these budgets rather than pinned, so the
# design stays within the measured runtime envelope (benchmark anchor n=2955
# fitted in ~19 min; see 00_benchmark_model2_fit.py). Production tolerates a
# single fit; CV refits every fold (5 folds x 2 scales) and so uses a tighter
# budget.
MODEL2_MAX_AUGMENTED_POINTS = 3400
MODEL2_CV_MAX_AUGMENTED_POINTS = 2100


def derive_lambda_stride(
    n_lf_samples: int,
    n_hf_samples: int,
    n_wavelengths: int,
    *,
    max_points: int,
) -> int:
    """Smallest wavelength stride keeping the augmented design within budget.

    The augmented design has ``n_lf_samples * n_lf_lambda + n_hf_samples *
    n_hf_lambda`` scalar rows, where the LF wavelengths sit on a half-stride
    offset subgrid (matching fit_model2). Returns the smallest stride whose
    total point count is <= ``max_points`` -- i.e. the finest wavelength grid
    affordable for the shared LF subsample size.
    """
    for stride in range(1, n_wavelengths + 1):
        offset = (stride // 2) % stride
        n_lf_lambda = len(
            select_wavelength_subgrid(n_wavelengths, stride=stride, offset=offset)
        )
        n_hf_lambda = len(select_wavelength_subgrid(n_wavelengths, stride=stride))
        n_total = n_lf_samples * n_lf_lambda + n_hf_samples * n_hf_lambda
        if n_total <= max_points:
            return stride
    raise ValueError(
        f"no stride in [1, {n_wavelengths}] keeps the augmented design "
        f"(n_lf={n_lf_samples}, n_hf={n_hf_samples}) within "
        f"max_points={max_points}"
    )


def lf_column_moments(Y_lf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-wavelength mean and std of the LF spectra (all rows, all columns).

    LF data is always fully available (only HF availability is validated in
    CV), so these moments are leakage-free for held-out HF samples.
    """
    mu = Y_lf.mean(axis=0)
    sd = Y_lf.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("LF spectra have zero variance in some wavelength bin")
    return mu, sd


def build_augmented_design(
    X: np.ndarray,
    Y: np.ndarray,
    wavelengths: np.ndarray,
    *,
    sample_indices: np.ndarray,
    lambda_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (samples x wavelengths) into scalar rows z = (theta, lambda).

    Sample-major order: all selected wavelengths of the first selected sample,
    then the next sample, ... The fidelity indicator is NOT appended here;
    fit_model2 appends it last via mf_gp._append_fidelity, as the kernel
    requires.
    """
    X_selected = np.asarray(X, dtype=float)[sample_indices]
    lambda_values = np.asarray(wavelengths, dtype=float)[lambda_indices]
    n_samples, n_lambdas = len(sample_indices), len(lambda_indices)

    Z = np.empty((n_samples * n_lambdas, X.shape[1] + 1))
    Z[:, :-1] = np.repeat(X_selected, n_lambdas, axis=0)
    Z[:, -1] = np.tile(lambda_values, n_samples)
    y = np.asarray(Y, dtype=float)[np.ix_(sample_indices, lambda_indices)].ravel()
    return Z, y


# ---------------------------------------------------------------------------
# 2. Joint fit on the subsampled augmented design
# ---------------------------------------------------------------------------

@dataclass
class Model2Layer:
    """One jointly fitted wavelength-augmented AR(1) multi-fidelity GP."""

    wavelengths: np.ndarray
    scaler: StandardScaler                 # over the augmented columns, fit on LF
    lf_sample_indices: np.ndarray
    lf_lambda_indices: np.ndarray
    hf_lambda_indices: np.ndarray
    column_mu: np.ndarray | None           # per-wavelength LF moments, or None
    column_sd: np.ndarray | None           # when standardization is disabled
    model: GaussianProcessRegressor
    fit_seconds: float

    @property
    def rho(self) -> float:
        return float(self.model.kernel_.rho)


def _standardize_columns(
    Y: np.ndarray, mu: np.ndarray | None, sd: np.ndarray | None
) -> np.ndarray:
    if mu is None:
        return np.asarray(Y, dtype=float)
    return (Y - mu) / sd


def fit_model2(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    lf_sample_size: int,
    lambda_stride: int,
    lf_lambda_offset: int | None = None,
    n_restarts_optimizer: int = MF_GP_N_RESTARTS_OPTIMIZER,
    alpha: float = MF_GP_ALPHA,
    normalize_y: bool = MF_GP_NORMALIZE_Y,
    standardize_per_wavelength: bool = True,
) -> Model2Layer:
    """Fit one joint MF-GP on the wavelength-augmented subsampled design.

    All HF samples enter on the stride-``lambda_stride`` wavelength subgrid;
    ``lf_sample_size`` LF samples (drawn once with mf_gp.select_lf_subsample)
    enter on a subgrid offset by ``lf_lambda_offset`` (default: half a
    stride). The memory gate is checked on the total scalar point count
    before anything expensive happens.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    n_lambdas = len(wavelengths)
    if lf_lambda_offset is None:
        lf_lambda_offset = (lambda_stride // 2) % lambda_stride
    hf_lambda_indices = select_wavelength_subgrid(n_lambdas, stride=lambda_stride)
    lf_lambda_indices = select_wavelength_subgrid(
        n_lambdas, stride=lambda_stride, offset=lf_lambda_offset
    )
    lf_sample_indices = select_lf_subsample(
        X_lf.shape[0], lf_sample_size, seed=seed
    )

    n_total = (
        len(lf_sample_indices) * len(lf_lambda_indices)
        + X_hf.shape[0] * len(hf_lambda_indices)
    )
    assert_exact_fit_fits_in_memory(n_total, model2_n_theta(X_lf.shape[1]))

    if standardize_per_wavelength:
        column_mu, column_sd = lf_column_moments(np.asarray(Y_lf, dtype=float))
    else:
        column_mu = column_sd = None
    Y_lf_std = _standardize_columns(Y_lf, column_mu, column_sd)
    Y_hf_std = _standardize_columns(Y_hf, column_mu, column_sd)

    Z_lf, y_lf = build_augmented_design(
        X_lf,
        Y_lf_std,
        wavelengths,
        sample_indices=lf_sample_indices,
        lambda_indices=lf_lambda_indices,
    )
    Z_hf, y_hf = build_augmented_design(
        X_hf,
        Y_hf_std,
        wavelengths,
        sample_indices=np.arange(X_hf.shape[0]),
        lambda_indices=hf_lambda_indices,
    )

    scaler = StandardScaler().fit(Z_lf)
    Z_joint = np.vstack(
        (
            _append_fidelity(scaler.transform(Z_lf), 0),
            _append_fidelity(scaler.transform(Z_hf), 1),
        )
    )
    y_joint = np.concatenate((y_lf, y_hf))

    model = GaussianProcessRegressor(
        kernel=make_joint_mf_kernel(X_lf.shape[1] + 1),
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=normalize_y,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(Z_joint, y_joint)
    fit_seconds = time.perf_counter() - t0

    return Model2Layer(
        wavelengths=wavelengths,
        scaler=scaler,
        lf_sample_indices=lf_sample_indices,
        lf_lambda_indices=lf_lambda_indices,
        hf_lambda_indices=hf_lambda_indices,
        column_mu=column_mu,
        column_sd=column_sd,
        model=model,
        fit_seconds=fit_seconds,
    )


# ---------------------------------------------------------------------------
# 3. Prediction: per-wavelength marginals of the joint posterior
# ---------------------------------------------------------------------------

def predict_hf_model2(
    layer: Model2Layer, X_new: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """HF spectra at all wavelengths: the marginal over wavelength.

    Every new atmospheric input is expanded to (theta, lambda_j, fidelity=1)
    for the full 195-wavelength grid -- including wavelengths never seen in
    training, where the lambda kernel interpolates. Returns (means, stds)
    of shape (n_new, n_wavelengths) in original data units.
    """
    X_new = np.asarray(X_new, dtype=float)
    n_new = X_new.shape[0]
    n_lambdas = len(layer.wavelengths)

    Z_new = np.empty((n_new * n_lambdas, X_new.shape[1] + 1))
    Z_new[:, :-1] = np.repeat(X_new, n_lambdas, axis=0)
    Z_new[:, -1] = np.tile(layer.wavelengths, n_new)
    Z_augmented = _append_fidelity(layer.scaler.transform(Z_new), 1)

    means_flat, stds_flat = layer.model.predict(Z_augmented, return_std=True)
    means = means_flat.reshape(n_new, n_lambdas)
    stds = stds_flat.reshape(n_new, n_lambdas)
    if layer.column_mu is not None:
        means = means * layer.column_sd + layer.column_mu
        stds = stds * layer.column_sd
    return means, stds


# ---------------------------------------------------------------------------
# 4. CV adapter (same engine and fold assignment as every other model)
# ---------------------------------------------------------------------------

def cv_predict_model2(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    lf_sample_size: int,
    lambda_stride: int,
    progress: bool = False,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold Model 2 predictions over the paired HF samples.

    Whole HF spectra are held out per fold (the fold's rows never enter the
    augmented design at any wavelength), so there is no wavelength-level
    leakage. LF data is always fully eligible -- only HF availability is
    validated, matching validation/02_full_cv.
    """

    def fit_fold(train_idx):
        t0 = time.perf_counter()
        layer = fit_model2(
            X_lf,
            Y_lf,
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            lf_sample_size=lf_sample_size,
            lambda_stride=lambda_stride,
            **fit_kwargs,
        )
        if progress:
            print(
                f"  fold fitted in {time.perf_counter() - t0:.1f}s "
                f"(rho={layer.rho:.3f})",
                flush=True,
            )
        return layer

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model2(layer, X_hf[test_idx]),
        n_samples=X_hf.shape[0],
        n_wavelengths=len(wavelengths),
        n_splits=n_splits,
        seed=seed,
    )
