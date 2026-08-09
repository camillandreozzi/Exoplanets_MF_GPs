"""GPBoost adapters for per-wavelength AR(1) multi-fidelity GPs.

The rest of the project uses sklearn kernels for the main MF-GP
implementations. This module keeps GPBoost optional and contained: importing
the module succeeds even when GPBoost is absent, while calling the fit helpers
requires the package to be importable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.mf_gp import _append_fidelity, select_lf_subsample
from exoplanets_mf.model2 import lf_column_moments

GPBOOST_COV_FUNCTION = "ar1_mf_gaussian_ard"
GPBOOST_COV_FCT_SHAPE = 1.5
GPBOOST_GP_APPROX = "none"
GPBOOST_NUM_NEIGHBORS = None
_VAR_FLOOR = 1e-12


@lru_cache(maxsize=1)
def _gpboost_module():
    import gpboost as gpb

    return gpb


def gpboost_available() -> bool:
    """Return True when the optional GPBoost dependency can be imported."""
    try:
        _gpboost_module()
    except Exception:
        return False
    return True


def _require_gpboost():
    try:
        return _gpboost_module()
    except Exception as exc:  # pragma: no cover - exercised only without GPBoost
        raise ImportError(
            "GPBoost is required for the GPBoost MF-GP adapters. "
            "Install gpboost or let tests skip these optional models."
        ) from exc


def _fit_gpboost_ar1(
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
    fidelity_specific_mean: bool,
) -> Any:
    """Fit one zero-boosting GPBoost AR(1) model on augmented coordinates."""
    if not cov_function.startswith("ar1_mf_"):
        raise ValueError(
            "GPBoost multi-fidelity models require an ar1_mf_* covariance; "
            f"found {cov_function!r}"
        )
    gpb = _require_gpboost()
    model = gpb.GPModel(
        gp_coords=np.asarray(coords, dtype=float),
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
        likelihood="gaussian",
        seed=seed,
        num_parallel_threads=num_parallel_threads,
        fidelity_specific_mean=fidelity_specific_mean,
    )
    # The intercept gives GPBoost an explicit mean term. With
    # fidelity_specific_mean=True, GPBoost expands it into separate LF/HF
    # means, which is important after column standardization because HF can
    # have a shifted standardized mean even when the AR(1) rho is correct.
    intercept = np.ones((coords.shape[0], 1), dtype=float)
    model.fit(
        y=np.asarray(y, dtype=float),
        X=intercept,
        params={"maxit": maxit, "trace": trace},
    )
    _attach_compat_methods(model, n_points=int(coords.shape[0]))
    return model


def _predict_gpboost_ar1(
    model: Any,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and sanitized standard deviation from one GPBoost model."""
    coords = np.asarray(coords, dtype=float)
    prediction = model.predict(
        gp_coords_pred=coords,
        X_pred=np.ones((coords.shape[0], 1), dtype=float),
        predict_var=True,
    )
    mean = np.asarray(prediction["mu"], dtype=float)
    variance = np.asarray(prediction["var"], dtype=float)
    return mean, np.sqrt(np.maximum(variance, _VAR_FLOOR))


def _cov_pars(model: Any) -> dict[str, float]:
    table = model.get_cov_pars(format_pandas=True)
    return {name: float(value) for name, value in table.iloc[0].items()}


def _attach_compat_methods(model: Any, *, n_points: int | None = None) -> Any:
    """Expose the small API older research scripts expect from GPBoost models."""
    if n_points is not None:
        model.n_points = int(n_points)
    if not hasattr(model, "cov_pars"):
        model.cov_pars = lambda: _cov_pars(model)
    return model


def _model_to_payload(model: Any) -> dict[str, Any]:
    return {
        "model_dict": model.model_to_dict(include_response_data=True),
        "n_points": getattr(model, "n_points", None),
    }


def _payload_to_model(payload: dict[str, Any]) -> Any:
    gpb = _require_gpboost()
    model = gpb.GPModel(model_dict=payload["model_dict"])
    return _attach_compat_methods(model, n_points=payload.get("n_points"))


@dataclass
class GPBoostModel1Layer:
    """One GPBoost AR(1) MF-GP per wavelength."""

    wavelengths: np.ndarray
    scaler: StandardScaler
    subsample_indices: np.ndarray | None
    column_mu: np.ndarray | None
    column_sd: np.ndarray | None
    models: list[Any] = field(default_factory=list)
    fit_seconds: np.ndarray | None = None
    cov_function: str = GPBOOST_COV_FUNCTION
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE
    gp_approx: str = GPBOOST_GP_APPROX
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS

    @property
    def rho(self) -> np.ndarray:
        return np.asarray([_cov_pars(model)["rho"] for model in self.models])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["models"] = [_model_to_payload(model) for model in self.models]
        state["_models_are_payloads"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_models_are_payloads", False):
            state["models"] = [_payload_to_model(payload) for payload in state["models"]]
        self.__dict__.update(state)


def fit_model1_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    cov_function: str = GPBOOST_COV_FUNCTION,
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE,
    gp_approx: str = GPBOOST_GP_APPROX,
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS,
    maxit: int = 1000,
    trace: bool = False,
    num_parallel_threads: int | None = None,
    standardize_per_wavelength: bool = True,
    fidelity_specific_mean: bool = True,
    progress_every: int | None = None,
) -> GPBoostModel1Layer:
    """Fit GPBoost's native AR(1) MF covariance independently per wavelength."""
    X_lf = np.asarray(X_lf, dtype=float)
    Y_lf = np.asarray(Y_lf, dtype=float)
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if Y_lf.shape[1] != len(wavelengths) or Y_hf.shape[1] != len(wavelengths):
        raise ValueError("Y_lf, Y_hf, and wavelengths must share columns")
    if standardize_per_wavelength:
        column_mu, column_sd = lf_column_moments(Y_lf)
        Y_lf_fit = (Y_lf - column_mu) / column_sd
        Y_hf_fit = (Y_hf - column_mu) / column_sd
    else:
        column_mu = column_sd = None
        Y_lf_fit = Y_lf
        Y_hf_fit = Y_hf

    if subsample_size is None:
        subsample_indices = None
        lf_rows = np.arange(X_lf.shape[0])
    else:
        subsample_indices = select_lf_subsample(
            X_lf.shape[0], subsample_size, seed=seed
        )
        lf_rows = subsample_indices

    scaler = StandardScaler().fit(X_lf)
    coords = np.vstack(
        (
            _append_fidelity(scaler.transform(X_lf[lf_rows]), 0),
            _append_fidelity(scaler.transform(X_hf), 1),
        )
    )

    models = []
    fit_seconds = []
    for j, _ in enumerate(wavelengths):
        y = np.concatenate((Y_lf_fit[lf_rows, j], Y_hf_fit[:, j]))
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
        )
        fit_seconds.append(time.perf_counter() - t0)
        models.append(model)
        if progress_every and (j + 1) % progress_every == 0:
            print(
                f"  fit {j + 1}/{len(wavelengths)} GPBoost MF-GPs "
                f"({fit_seconds[-1]:.1f}s last)",
                flush=True,
            )

    return GPBoostModel1Layer(
        wavelengths=wavelengths,
        scaler=scaler,
        subsample_indices=subsample_indices,
        column_mu=column_mu,
        column_sd=column_sd,
        models=models,
        fit_seconds=np.asarray(fit_seconds),
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
    )


def predict_hf_model1_gpboost(
    layer: GPBoostModel1Layer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra from atmospheric inputs with all fitted wavelengths."""
    X_scaled = layer.scaler.transform(np.asarray(X_new, dtype=float))
    coords = _append_fidelity(X_scaled, 1)
    means, stds = zip(*(_predict_gpboost_ar1(model, coords) for model in layer.models))
    mean_matrix = np.column_stack(means)
    std_matrix = np.column_stack(stds)
    if layer.column_mu is not None:
        mean_matrix = mean_matrix * layer.column_sd + layer.column_mu
        std_matrix = std_matrix * layer.column_sd
    return mean_matrix, std_matrix


def hyperparameter_table_model1_gpboost(layer: GPBoostModel1Layer) -> pd.DataFrame:
    """One row per wavelength of GPBoost covariance parameters."""
    rows = []
    for j, (wavelength, model) in enumerate(zip(layer.wavelengths, layer.models)):
        cov = _cov_pars(model)
        row = {
            "wavelength": float(wavelength),
            "rho": cov["rho"],
            "error_var": cov.get("Error_var", np.nan),
            "low_signal_variance": cov.get("low_GP_var", np.nan),
            "delta_signal_variance": cov.get("discrepancy_GP_var", np.nan),
        }
        row.update({f"cov_{name}": value for name, value in cov.items()})
        if layer.fit_seconds is not None:
            row["fit_seconds"] = float(layer.fit_seconds[j])
        n_dims = int(model.dim_coords) - 1
        for i in range(1, n_dims + 1):
            row[f"low_range_{i - 1}"] = cov.get(f"low_GP_range_{i}", np.nan)
            row[f"delta_range_{i - 1}"] = cov.get(
                f"discrepancy_GP_range_{i}", np.nan
            )
        if "low_GP_range" in cov:
            row["low_range"] = cov["low_GP_range"]
        if "discrepancy_GP_range" in cov:
            row["delta_range"] = cov["discrepancy_GP_range"]
        rows.append(row)
    return pd.DataFrame(rows)


def cv_predict_model1_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    progress_every: int | None = None,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold GPBoost Model 1 predictions over paired HF samples."""

    def fit_fold(train_idx: np.ndarray) -> GPBoostModel1Layer:
        return fit_model1_gpboost(
            X_lf,
            Y_lf,
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            subsample_size=subsample_size,
            progress_every=progress_every,
            **fit_kwargs,
        )

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model1_gpboost(
            layer, X_hf[test_idx]
        ),
        n_samples=Y_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )
