"""Cross-validation over the paired HF samples for model comparison.

Replaces the former AIC / ANOVA F-test comparison: likelihoods and effective
parameter counts are hard to define credibly across the models compared here
(joint MF-GP variants, Model 2 variants), whereas held-out predictive error
is defined identically for all of them.

The engine is model-agnostic: `cv_predict` only needs a fit callback and a
predict callback, splits the paired HF rows so that *every* sample is held
out exactly once (all 97 in the real data, not a single fixed holdout), and
returns one out-of-fold prediction per sample. `cv_metrics` scores one model;
`compare_cv` reports the paired per-sample comparison between two models --
use it for Model 1A vs 1B and Model 1 vs Model 2 alike.

Every model in this project is a joint multi-fidelity GP; the former
closed-form rho layers (two-stage discrepancy-vs-LF baselines) were removed
deliberately and must not come back. `cv_predict_joint_mf_gp` is the shared
adapter for the per-wavelength AR(1) layers (Model 1A: shared rho, Model 1B:
per-wavelength rho).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from exoplanets_mf.mf_gp import (
    fit_joint_mf_gp,
    fit_joint_mf_gp_global_rho,
    predict_hf,
)

# K-fold: every fold refits the full per-wavelength GP stack, so LOO (97
# refits) is not affordable for any model in the comparison.
CV_FULL_MODEL_SPLITS = 5


# ---------------------------------------------------------------------------
# 1. Model-agnostic CV engine
# ---------------------------------------------------------------------------

@dataclass
class CVPredictions:
    """Out-of-fold predictions: exactly one held-out prediction per sample."""

    y_pred: np.ndarray            # (n_samples, n_wavelengths)
    y_std: np.ndarray | None      # (n_samples, n_wavelengths) or None
    fold_of_sample: np.ndarray    # (n_samples,) fold index that held sample out
    n_splits: int

    def with_values(
        self,
        y_pred: np.ndarray,
        y_std: np.ndarray | None = None,
    ) -> "CVPredictions":
        """Return the same CV assignment with replaced prediction values.

        Useful when a model is fitted/scored in a transformed output space and
        its held-out predictions need to be compared after back-transformation.
        """
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != self.y_pred.shape:
            raise ValueError(
                "replacement predictions must have shape "
                f"{self.y_pred.shape}; found {y_pred.shape}"
            )
        if y_std is not None:
            y_std = np.asarray(y_std, dtype=float)
            if y_std.shape != self.y_pred.shape:
                raise ValueError(
                    "replacement standard deviations must have shape "
                    f"{self.y_pred.shape}; found {y_std.shape}"
                )
        return CVPredictions(
            y_pred=y_pred,
            y_std=y_std,
            fold_of_sample=self.fold_of_sample.copy(),
            n_splits=self.n_splits,
        )


def cv_predict(
    fit_fn: Callable[[np.ndarray], object],
    predict_fn: Callable[[object, np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]],
    n_samples: int,
    n_wavelengths: int,
    *,
    n_splits: int,
    seed: int,
) -> CVPredictions:
    """Fill an (n_samples, n_wavelengths) prediction matrix fold by fold.

    fit_fn(train_indices) -> model; predict_fn(model, test_indices) -> either
    predicted spectra (n_test, J) or a (mean, std) tuple. Callbacks close over
    their own data, so the engine never assumes what a model consumes (paired
    LF spectra, atmospheric inputs, ...). n_splits == n_samples is LOO.
    """
    y_pred = np.full((n_samples, n_wavelengths), np.nan)
    y_std = np.full((n_samples, n_wavelengths), np.nan)
    fold_of_sample = np.full(n_samples, -1, dtype=int)
    saw_std = False

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_samples))):
        model = fit_fn(train_idx)
        prediction = predict_fn(model, test_idx)
        if isinstance(prediction, tuple):
            y_pred[test_idx], y_std[test_idx] = prediction
            saw_std = True
        else:
            y_pred[test_idx] = prediction
        fold_of_sample[test_idx] = fold

    if np.isnan(y_pred).any() or (fold_of_sample < 0).any():
        raise RuntimeError("cross-validation left samples without a held-out prediction")

    return CVPredictions(
        y_pred=y_pred,
        y_std=y_std if saw_std else None,
        fold_of_sample=fold_of_sample,
        n_splits=n_splits,
    )


# ---------------------------------------------------------------------------
# 2. Scoring and paired model comparison
# ---------------------------------------------------------------------------

@dataclass
class CVMetrics:
    """Held-out predictive error of one model (original data units)."""

    rmse_per_wavelength: np.ndarray   # (n_wavelengths,)
    rmse_per_sample: np.ndarray       # (n_samples,)
    rmse_pooled: float
    nrmse_pooled: float               # rmse / (max - min) of the true spectra
    coverage_95: float | None         # fraction of |z| <= 1.96, if stds exist


def cv_metrics(Y_true: np.ndarray, predictions: CVPredictions) -> CVMetrics:
    errors = predictions.y_pred - Y_true
    coverage = None
    if predictions.y_std is not None:
        z = errors / predictions.y_std
        coverage = float(np.mean(np.abs(z) <= 1.96))
    return CVMetrics(
        rmse_per_wavelength=np.sqrt((errors**2).mean(axis=0)),
        rmse_per_sample=np.sqrt((errors**2).mean(axis=1)),
        rmse_pooled=float(np.sqrt((errors**2).mean())),
        nrmse_pooled=float(np.sqrt((errors**2).mean()) / (Y_true.max() - Y_true.min())),
        coverage_95=coverage,
    )


@dataclass
class CVComparison:
    """Paired per-sample comparison of two models' out-of-fold errors.

    delta_* = model_2 - model_1, so negative deltas mean model 2 wins.
    """

    labels: tuple[str, str]
    metrics_1: CVMetrics
    metrics_2: CVMetrics
    delta_rmse_per_sample: np.ndarray      # (n_samples,)
    delta_rmse_per_wavelength: np.ndarray  # (n_wavelengths,)
    delta_rmse_pooled: float
    delta_mean: float                      # mean of per-sample deltas
    delta_std: float
    fraction_samples_model2_wins: float


def compare_cv(
    Y_true: np.ndarray,
    predictions_1: CVPredictions,
    predictions_2: CVPredictions,
    labels: tuple[str, str],
) -> CVComparison:
    metrics_1 = cv_metrics(Y_true, predictions_1)
    metrics_2 = cv_metrics(Y_true, predictions_2)
    delta_per_sample = metrics_2.rmse_per_sample - metrics_1.rmse_per_sample
    return CVComparison(
        labels=labels,
        metrics_1=metrics_1,
        metrics_2=metrics_2,
        delta_rmse_per_sample=delta_per_sample,
        delta_rmse_per_wavelength=(
            metrics_2.rmse_per_wavelength - metrics_1.rmse_per_wavelength
        ),
        delta_rmse_pooled=metrics_2.rmse_pooled - metrics_1.rmse_pooled,
        delta_mean=float(delta_per_sample.mean()),
        delta_std=float(delta_per_sample.std()),
        fraction_samples_model2_wins=float((delta_per_sample < 0).mean()),
    )


# ---------------------------------------------------------------------------
# 3. Joint MF-GP adapters (Model 1A: shared rho, Model 1B: per-wavelength rho)
# ---------------------------------------------------------------------------

def cv_predict_joint_mf_gp(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    per_wavelength_rho: bool,
    subsample_size: int | None,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    seed: int,
    progress_every: int | None = None,
) -> CVPredictions:
    """Out-of-fold HF predictions of the per-wavelength joint MF-GP layer.

    Every fold refits the full layer on the training HF rows (the LF design
    is always fully eligible) and predicts the held-out HF spectra from
    atmospheric inputs alone. ``per_wavelength_rho=True`` is Model 1B (one
    free rho_j per wavelength); ``False`` is Model 1A (one shared rho against
    the summed marginal likelihood).
    """
    def fit_fold(train_idx: np.ndarray):
        if per_wavelength_rho:
            return fit_joint_mf_gp(
                X_lf,
                Y_lf,
                X_hf[train_idx],
                Y_hf[train_idx],
                wavelengths,
                seed=seed,
                subsample_size=subsample_size,
                progress_every=progress_every,
            )
        return fit_joint_mf_gp_global_rho(
            X_lf,
            Y_lf,
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            subsample_size=subsample_size,
            progress_every=progress_every,
        )

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf(layer, X_hf[test_idx]),
        n_samples=Y_hf.shape[0],
        n_wavelengths=Y_hf.shape[1],
        n_splits=n_splits,
        seed=seed,
    )
