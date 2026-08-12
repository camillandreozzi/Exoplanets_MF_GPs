"""GPBoost adapter for the high-fidelity-only single-fidelity GP baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.gpboost_mf import (
    GPBOOST_COV_FCT_SHAPE,
    GPBOOST_GP_APPROX,
    GPBOOST_INIT_ERROR_VAR_FRACTION,
    GPBOOST_INIT_LOW_VAR_FRACTION,
    GPBOOST_NUM_NEIGHBORS,
    _attach_compat_methods,
    _cov_pars,
    _median_length_scales,
    _model_to_payload,
    _payload_to_model,
    _require_gpboost,
)

GPBOOST_HF_COV_FUNCTION = "gaussian_ard"


@dataclass
class GPBoostHFOnlyLayer:
    """One GPBoost single-fidelity GP per wavelength, fitted on HF rows only."""

    wavelengths: np.ndarray
    scaler: StandardScaler
    column_mu: np.ndarray | None
    column_sd: np.ndarray | None
    models: list[Any] = field(default_factory=list)
    fit_seconds: np.ndarray | None = None
    cov_function: str = GPBOOST_HF_COV_FUNCTION
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE
    gp_approx: str = GPBOOST_GP_APPROX
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["models"] = [_model_to_payload(model) for model in self.models]
        state["_models_are_payloads"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_models_are_payloads", False):
            state["models"] = [
                _payload_to_model(payload) for payload in state["models"]
            ]
        self.__dict__.update(state)


def _column_moments(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(Y, dtype=float).mean(axis=0)
    sd = np.asarray(Y, dtype=float).std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("HF spectra have zero variance in some wavelength bin")
    return mu, sd


def _standardize_columns(
    Y: np.ndarray, mu: np.ndarray | None, sd: np.ndarray | None
) -> np.ndarray:
    if mu is None:
        return np.asarray(Y, dtype=float)
    return (np.asarray(Y, dtype=float) - mu) / sd


def heuristic_init_cov_pars_hf(
    coords: np.ndarray,
    y: np.ndarray,
    num_cov_pars: int,
    *,
    seed: int,
) -> np.ndarray:
    """Starting covariance parameters for a single-fidelity GPBoost GP.

    Parameter order for a single-fidelity ``*_ard`` covariance is
    ``[Error_var, GP_var, GP_range_1, ...]``.
    """
    n_ranges = num_cov_pars - 2
    if n_ranges < 1:
        raise ValueError(
            f"cannot infer the single-fidelity parameter layout from "
            f"{num_cov_pars} covariance parameters"
        )
    y = np.asarray(y, dtype=float)
    total_var = max(float(np.var(y)), 1e-8)
    init = np.empty(num_cov_pars)
    init[0] = GPBOOST_INIT_ERROR_VAR_FRACTION * total_var
    init[1] = GPBOOST_INIT_LOW_VAR_FRACTION * total_var
    init[2:] = _median_length_scales(
        np.asarray(coords, dtype=float), n_ranges, seed=seed
    )
    return init


def _fit_gpboost_single_fidelity(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cov_function: str,
    cov_fct_shape: float,
    gp_approx: str,
    num_neighbors: int | None,
    maxit: int,
    trace: bool,
    num_parallel_threads: int | None,
    heuristic_init: bool,
) -> Any:
    gpb = _require_gpboost()
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float)
    model = gpb.GPModel(
        gp_coords=coords,
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
        likelihood="gaussian",
        seed=seed,
        num_parallel_threads=num_parallel_threads,
        fidelity_specific_mean=False,
    )
    params: dict[str, Any] = {"maxit": maxit, "trace": trace}
    if heuristic_init:
        params["init_cov_pars"] = heuristic_init_cov_pars_hf(
            coords, y, int(model.num_cov_pars), seed=seed
        )
    model.fit(
        y=y,
        X=np.ones((coords.shape[0], 1), dtype=float),
        params=params,
    )
    return _attach_compat_methods(model, n_points=int(coords.shape[0]))


def _predict_gpboost_single_fidelity(
    model: Any, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    prediction = model.predict(
        gp_coords_pred=coords,
        X_pred=np.ones((coords.shape[0], 1), dtype=float),
        predict_var=True,
    )
    mean = np.asarray(prediction["mu"], dtype=float)
    variance = np.asarray(prediction["var"], dtype=float)
    return mean, np.sqrt(np.maximum(variance, 1e-12))


def fit_hf_only_gpboost(
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    cov_function: str = GPBOOST_HF_COV_FUNCTION,
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE,
    gp_approx: str = GPBOOST_GP_APPROX,
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS,
    maxit: int = 1000,
    trace: bool = False,
    num_parallel_threads: int | None = None,
    standardize_per_wavelength: bool = True,
    heuristic_init: bool = True,
    progress_every: int | None = None,
) -> GPBoostHFOnlyLayer:
    """Fit one GPBoost single-fidelity GP per wavelength on HF data only."""
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if Y_hf.shape[1] != len(wavelengths):
        raise ValueError("Y_hf and wavelengths must share columns")

    if standardize_per_wavelength:
        column_mu, column_sd = _column_moments(Y_hf)
    else:
        column_mu = column_sd = None
    Y_fit = _standardize_columns(Y_hf, column_mu, column_sd)

    scaler = StandardScaler().fit(X_hf)
    coords = scaler.transform(X_hf)
    models = []
    fit_seconds = []
    for j, _ in enumerate(wavelengths):
        t0 = time.perf_counter()
        model = _fit_gpboost_single_fidelity(
            coords,
            Y_fit[:, j],
            seed=seed,
            cov_function=cov_function,
            cov_fct_shape=cov_fct_shape,
            gp_approx=gp_approx,
            num_neighbors=num_neighbors,
            maxit=maxit,
            trace=trace,
            num_parallel_threads=num_parallel_threads,
            heuristic_init=heuristic_init,
        )
        fit_seconds.append(time.perf_counter() - t0)
        models.append(model)
        if progress_every and (j + 1) % progress_every == 0:
            print(
                f"  fit {j + 1}/{len(wavelengths)} GPBoost HF-only GPs "
                f"({fit_seconds[-1]:.1f}s last)",
                flush=True,
            )

    return GPBoostHFOnlyLayer(
        wavelengths=wavelengths,
        scaler=scaler,
        column_mu=column_mu,
        column_sd=column_sd,
        models=models,
        fit_seconds=np.asarray(fit_seconds),
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
    )


def predict_hf_only_gpboost(
    layer: GPBoostHFOnlyLayer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra from atmospheric inputs with GPBoost HF-only GPs."""
    coords = layer.scaler.transform(np.asarray(X_new, dtype=float))
    means, stds = zip(
        *(_predict_gpboost_single_fidelity(model, coords) for model in layer.models)
    )
    mean_matrix = np.column_stack(means)
    std_matrix = np.column_stack(stds)
    if layer.column_mu is not None:
        mean_matrix = mean_matrix * layer.column_sd + layer.column_mu
        std_matrix = std_matrix * layer.column_sd
    return mean_matrix, std_matrix


def hyperparameter_table_hf_only_gpboost(
    layer: GPBoostHFOnlyLayer,
) -> pd.DataFrame:
    """One row per wavelength of GPBoost HF-only covariance parameters."""
    rows = []
    for j, (wavelength, model) in enumerate(zip(layer.wavelengths, layer.models)):
        cov = _cov_pars(model)
        row = {
            "wavelength": float(wavelength),
            "error_var": cov.get("Error_var", np.nan),
            "signal_variance": cov.get("GP_var", np.nan),
        }
        row.update({f"cov_{name}": value for name, value in cov.items()})
        if layer.fit_seconds is not None:
            row["fit_seconds"] = float(layer.fit_seconds[j])
        n_dims = int(model.dim_coords)
        for i in range(1, n_dims + 1):
            row[f"range_{i - 1}"] = cov.get(f"GP_range_{i}", np.nan)
        if "GP_range" in cov:
            row["range"] = cov["GP_range"]
        rows.append(row)
    return pd.DataFrame(rows)


def cv_predict_hf_only_gpboost(
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    seed: int,
    progress_every: int | None = None,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold GPBoost HF-only predictions over paired HF samples."""

    def fit_fold(train_idx: np.ndarray) -> GPBoostHFOnlyLayer:
        return fit_hf_only_gpboost(
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            progress_every=progress_every,
            **fit_kwargs,
        )

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_only_gpboost(
            layer, X_hf[test_idx]
        ),
        n_samples=Y_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )
