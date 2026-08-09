"""High-fidelity-only Gaussian process baseline (single-fidelity reference).

For each wavelength bin j this fits a plain SE-ARD Gaussian process on the HF
spectra alone, ignoring the low-fidelity design entirely:

    f_H,j(theta) = GP(0, sigma^2 * k_SE-ARD(theta, theta') + noise).

It is deliberately NOT a multi-fidelity model: it is the single-fidelity floor
that shows what the low-fidelity data actually buys the joint MF-GPs (Models
1A/1B/2). It reuses the same ARD length-scale / signal-variance / noise
bounds as the joint kernel (imported from ``mf_gp``), so widening those bounds
applies here too, and exposes the same ``predict``-style ``(mean, std)`` return
as ``mf_gp.predict_hf`` so the model-agnostic CV engine can consume it unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.mf_gp import (
    LENGTH_SCALE_BOUNDS,
    LENGTH_SCALE_INIT,
    MF_GP_ALPHA,
    MF_GP_N_RESTARTS_OPTIMIZER,
    MF_GP_NORMALIZE_Y,
    N_INPUT_DIMS,
    NOISE_LEVEL_BOUNDS,
    NOISE_LEVEL_INIT,
    SIGNAL_VARIANCE_BOUNDS,
    warn_on_pinned_bounds,
)


def make_hf_only_kernel(n_dims: int = N_INPUT_DIMS):
    """SE-ARD kernel with a learned homoscedastic noise term (single fidelity)."""
    length_scales = np.full(n_dims, LENGTH_SCALE_INIT)
    return (
        ConstantKernel(1.0, SIGNAL_VARIANCE_BOUNDS)
        * RBF(length_scales, LENGTH_SCALE_BOUNDS)
        + WhiteKernel(NOISE_LEVEL_INIT, NOISE_LEVEL_BOUNDS)
    )


@dataclass
class HFOnlyGPLayer:
    """One single-fidelity SE-ARD GP per wavelength, fitted on HF data alone."""

    wavelengths: np.ndarray
    scaler: StandardScaler
    models: list[GaussianProcessRegressor] = field(default_factory=list)
    fit_seconds: np.ndarray | None = None  # per-wavelength fit wall time


def fit_hf_only_gp(
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    n_restarts_optimizer: int = MF_GP_N_RESTARTS_OPTIMIZER,
    alpha: float = MF_GP_ALPHA,
    normalize_y: bool = MF_GP_NORMALIZE_Y,
    progress_every: int | None = None,
) -> HFOnlyGPLayer:
    """Fit one plain SE-ARD GP per wavelength on the HF spectra only.

    The low-fidelity design is never touched: this is the single-fidelity
    baseline for the joint MF-GPs. Inputs are standardized with a scaler fitted
    on ``X_hf`` (the only data the model sees), matching ``predict_hf_only``.
    """
    scaler = StandardScaler().fit(X_hf)
    X_scaled = scaler.transform(X_hf)

    models = []
    fit_seconds = []
    for j in range(len(wavelengths)):
        model = GaussianProcessRegressor(
            kernel=make_hf_only_kernel(X_hf.shape[1]),
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=seed,
        )
        t0 = time.perf_counter()
        model.fit(X_scaled, Y_hf[:, j])
        fit_seconds.append(time.perf_counter() - t0)
        models.append(model)
        if progress_every and (j + 1) % progress_every == 0:
            print(
                f"  fit {j + 1}/{len(wavelengths)} HF-only GPs "
                f"({fit_seconds[-1]:.1f}s last)",
                flush=True,
            )

    warn_on_pinned_bounds(models, label="HF-only GP (single-fidelity baseline)")
    return HFOnlyGPLayer(
        wavelengths=np.asarray(wavelengths),
        scaler=scaler,
        models=models,
        fit_seconds=np.asarray(fit_seconds),
    )


def predict_hf_only(
    layer: HFOnlyGPLayer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra (mean, std) at every wavelength from inputs alone."""
    X_scaled = layer.scaler.transform(X_new)
    means, stds = zip(
        *(model.predict(X_scaled, return_std=True) for model in layer.models)
    )
    return np.column_stack(means), np.column_stack(stds)


def hyperparameter_table(layer: HFOnlyGPLayer) -> pd.DataFrame:
    """One row per wavelength of the fitted single-fidelity covariance params."""
    rows = []
    for j, (wl, gp) in enumerate(zip(layer.wavelengths, layer.models)):
        kernel = gp.kernel_
        amplitude, rbf = kernel.k1.k1, kernel.k1.k2
        white = kernel.k2
        row = {
            "wavelength": wl,
            "signal_variance": amplitude.constant_value,
            "noise_level": white.noise_level,
            "log_marginal_likelihood": gp.log_marginal_likelihood(),
        }
        if layer.fit_seconds is not None:
            row["fit_seconds"] = layer.fit_seconds[j]
        row.update(
            {f"length_scale_{i}": ell for i, ell in enumerate(rbf.length_scale)}
        )
        rows.append(row)
    return pd.DataFrame(rows)
