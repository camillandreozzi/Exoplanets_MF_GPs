"""GPBoost adapter for the wavelength-augmented AR(1) MF-GP (Model 2)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.gpboost_mf import (
    GPBOOST_COV_FCT_SHAPE,
    GPBOOST_COV_FUNCTION,
    GPBOOST_GP_APPROX,
    GPBOOST_HEURISTIC_INIT,
    GPBOOST_NUM_NEIGHBORS,
    _fit_gpboost_ar1,
    _model_to_payload,
    _payload_to_model,
    _predict_gpboost_ar1,
    _cov_pars,
)
from exoplanets_mf.mf_gp import _append_fidelity, select_lf_subsample
from exoplanets_mf.model2 import (
    build_augmented_design,
    lf_column_moments,
    select_wavelength_subgrid,
)


@dataclass
class GPBoostModel2Layer:
    """One GPBoost AR(1) model over the joint (theta, lambda) domain."""

    wavelengths: np.ndarray
    scaler: StandardScaler
    lf_sample_indices: np.ndarray
    lf_lambda_indices: np.ndarray
    hf_lambda_indices: np.ndarray
    column_mu: np.ndarray | None
    column_sd: np.ndarray | None
    model: Any
    fit_seconds: float
    cov_function: str = GPBOOST_COV_FUNCTION
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE
    gp_approx: str = GPBOOST_GP_APPROX
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS

    @property
    def rho(self) -> float:
        return _cov_pars(self.model)["rho"]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model"] = _model_to_payload(self.model)
        state["_model_is_payload"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_model_is_payload", False):
            state["model"] = _payload_to_model(state["model"])
        self.__dict__.update(state)


def _standardize_columns(
    Y: np.ndarray, mu: np.ndarray | None, sd: np.ndarray | None
) -> np.ndarray:
    if mu is None:
        return np.asarray(Y, dtype=float)
    return (np.asarray(Y, dtype=float) - mu) / sd


def fit_model2_gpboost(
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
    cov_function: str = GPBOOST_COV_FUNCTION,
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE,
    gp_approx: str = GPBOOST_GP_APPROX,
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS,
    standardize_per_wavelength: bool = True,
    maxit: int = 1000,
    trace: bool = False,
    num_parallel_threads: int | None = None,
    fidelity_specific_mean: bool = True,
    heuristic_init: bool = GPBOOST_HEURISTIC_INIT,
) -> GPBoostModel2Layer:
    """Fit GPBoost's native AR(1) covariance on Model 2's augmented design."""
    X_lf = np.asarray(X_lf, dtype=float)
    Y_lf = np.asarray(Y_lf, dtype=float)
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
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

    if standardize_per_wavelength:
        column_mu, column_sd = lf_column_moments(Y_lf)
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
    coords = np.vstack(
        (
            _append_fidelity(scaler.transform(Z_lf), 0),
            _append_fidelity(scaler.transform(Z_hf), 1),
        )
    )
    y = np.concatenate((y_lf, y_hf))

    t0 = time.perf_counter()
    model = _fit_gpboost_ar1(
        coords,
        y,
        seed=seed,
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
        maxit=maxit,
        trace=trace,
        num_parallel_threads=num_parallel_threads,
        fidelity_specific_mean=fidelity_specific_mean,
        heuristic_init=heuristic_init,
    )
    fit_seconds = time.perf_counter() - t0

    return GPBoostModel2Layer(
        wavelengths=wavelengths,
        scaler=scaler,
        lf_sample_indices=lf_sample_indices,
        lf_lambda_indices=lf_lambda_indices,
        hf_lambda_indices=hf_lambda_indices,
        column_mu=column_mu,
        column_sd=column_sd,
        model=model,
        fit_seconds=fit_seconds,
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
    )


def predict_hf_model2_gpboost(
    layer: GPBoostModel2Layer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra at all wavelengths from a fitted GPBoost Model 2."""
    X_new = np.asarray(X_new, dtype=float)
    n_new = X_new.shape[0]
    n_lambdas = len(layer.wavelengths)

    Z_new = np.empty((n_new * n_lambdas, X_new.shape[1] + 1), dtype=float)
    Z_new[:, :-1] = np.repeat(X_new, n_lambdas, axis=0)
    Z_new[:, -1] = np.tile(layer.wavelengths, n_new)
    coords = _append_fidelity(layer.scaler.transform(Z_new), 1)
    mean_flat, std_flat = _predict_gpboost_ar1(layer.model, coords)
    means = mean_flat.reshape(n_new, n_lambdas)
    stds = std_flat.reshape(n_new, n_lambdas)
    if layer.column_mu is not None:
        means = means * layer.column_sd + layer.column_mu
        stds = stds * layer.column_sd
    return means, stds


def hyperparameter_table_model2_gpboost(
    layer: GPBoostModel2Layer,
    *,
    n_points: int | None = None,
    lambda_stride: int | None = None,
) -> pd.DataFrame:
    """One-row covariance summary for the GPBoost Model 2 fit."""
    cov = _cov_pars(layer.model)
    row = {
        "cov_function": layer.cov_function,
        "rho": cov["rho"],
        "error_var": cov.get("Error_var", np.nan),
        "low_signal_variance": cov.get("low_GP_var", np.nan),
        "delta_signal_variance": cov.get("discrepancy_GP_var", np.nan),
        "fit_seconds": float(layer.fit_seconds),
        "n_points": (
            int(n_points)
            if n_points is not None
            else int(getattr(layer.model, "n_points", 0))
        ),
    }
    if lambda_stride is not None:
        row["lambda_stride"] = int(lambda_stride)
    row.update({f"cov_{name}": value for name, value in cov.items()})
    n_dims = int(layer.model.dim_coords) - 1
    for i in range(1, n_dims + 1):
        row[f"low_range_{i - 1}"] = cov.get(f"low_GP_range_{i}", np.nan)
        row[f"delta_range_{i - 1}"] = cov.get(f"discrepancy_GP_range_{i}", np.nan)
    if "low_GP_range" in cov:
        row["low_range"] = cov["low_GP_range"]
    if "discrepancy_GP_range" in cov:
        row["delta_range"] = cov["discrepancy_GP_range"]
    return pd.DataFrame([row])


def cv_predict_model2_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    lf_sample_size: int,
    lambda_stride: int,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    progress: bool = False,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold GPBoost Model 2 predictions over paired HF samples."""

    def fit_fold(train_idx: np.ndarray) -> GPBoostModel2Layer:
        t0 = time.perf_counter()
        layer = fit_model2_gpboost(
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
                f"  GPBoost Model 2 fold fitted in "
                f"{time.perf_counter() - t0:.1f}s (rho={layer.rho:.3f})",
                flush=True,
            )
        return layer

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model2_gpboost(
            layer, X_hf[test_idx]
        ),
        n_samples=Y_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )
