"""Registry and loaders for the fitted-hyperparameter tables of every model.

The ``research/parameters`` workflow reviews the covariance hyperparameters that
every fit already persists to a ``*_hyperparameters.csv``. Those CSVs use two
different schemas -- the sklearn AR(1) models (Models 1A/1B, log-space variants,
holdout refits, Model 2) versus the GPBoost backends -- and split into
per-wavelength tables (one row per wavelength) and scalar tables (a single row,
because Model 2 folds wavelength into the kernel).

This module hides that divergence behind a single registry plus a loader that
normalises column names to a common vocabulary:

    rho, signal_variance, low_signal_variance, delta_signal_variance,
    low_length_scale_<i>, delta_length_scale_<i>,     # 0-indexed ARD dims
    low_noise / high_noise (sklearn) or error_var (gpboost),
    joint_log_marginal_likelihood (sklearn only),
    wavelength (per-wavelength tables only).

GPBoost "range" parameters are length-scale analogues but are NOT in the same
standardized units as the sklearn ARD length scales; they are renamed for a
uniform API, not to imply the values are directly comparable across backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_XHF
from .instruments import instrument_mode_masks
from .mf_gp import (
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    RHO_BOUNDS,
    SIGNAL_VARIANCE_BOUNDS,
)
from .paths import (
    LOG_MODELLING_RESULTS_DIR,
    MODELLING_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
)

__all__ = [
    "ModelEntry",
    "REGISTRY",
    "MODEL_COLORS",
    "SIGNAL_VARIANCE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "RHO_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "input_names",
    "load_parameter_table",
    "available_entries",
    "shade_instrument_modes",
    "signal_variance_columns",
    "length_scale_columns",
    "noise_columns",
    "count_pinned",
]


@lru_cache(maxsize=1)
def input_names() -> tuple[str, ...]:
    """Atmospheric-input labels for the ARD length-scale columns.

    Derived from the canonical ``XHF`` header (Kzz, Rp, Tint, C, N, O, S, logg,
    f) so the labels never drift from the data.
    """
    return tuple(load_XHF().columns)


# CVD-safe base palette shared with the stats101 figures, extended with distinct
# hues for the GPBoost / log-space / holdout series. Panels are also direct-
# labelled, so identity never rides on colour alone.
MODEL_COLORS = {
    "model_1a": "#2a78d6",
    "model_1b": "#1baf7a",
    "model1_gpboost": "#d68a2a",
    "log_model_1a": "#7a5cd6",
    "log_model_1b": "#1b9fbf",
    "holdout_model_1a": "#c0504d",
    "holdout_model_1b": "#4d9c5a",
    "model2_linear": "#008300",
    "model2_log10": "#5aa02c",
    "model2_gpboost": "#b26a00",
    "full_hf_only_gpboost": "#595959",
    "full_model_1a_gpboost": "#a85c2f",
    "full_model_1b_gpboost": "#c43b71",
    "full_model_2_gpboost": "#3f7aa8",
}


@dataclass(frozen=True)
class ModelEntry:
    """One fitted model's parameter table and how to read it."""

    key: str
    label: str
    kind: str  # "per_wavelength" | "scalar"
    backend: str  # "sklearn" | "gpboost"
    space: str  # "linear" | "log10"
    csv_path: Path

    @property
    def color(self) -> str:
        return MODEL_COLORS.get(self.key, "tab:gray")


_MODELLING_1 = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1"
_MODELLING_2 = MODELLING_RESULTS_DIR / "02_augmented_wavelength"
_GPBOOST_FULL = MODELLING_RESULTS_DIR / "03_gpboost_comparison" / "full_fit"
_LOG_1 = LOG_MODELLING_RESULTS_DIR / "01_per_wavelength_ar1"
_LOG_2 = LOG_MODELLING_RESULTS_DIR / "02_augmented_wavelength"
_HOLDOUT = VALIDATION_RESULTS_DIR / "01_loo_holdout_validation" / "fit"

REGISTRY: tuple[ModelEntry, ...] = (
    # --- per-wavelength AR(1) models -------------------------------------
    ModelEntry(
        "model_1a", "Model 1A (shared rho)", "per_wavelength", "sklearn", "linear",
        _MODELLING_1 / "model_1a" / "model_1a_hyperparameters.csv",
    ),
    ModelEntry(
        "model_1b", "Model 1B (per-wavelength rho)", "per_wavelength", "sklearn", "linear",
        _MODELLING_1 / "model_1b" / "model_1b_hyperparameters.csv",
    ),
    ModelEntry(
        "model1_gpboost", "Model 1 (GPBoost)", "per_wavelength", "gpboost", "linear",
        _MODELLING_1 / "model1_gpboost" / "model1_gpboost_hyperparameters.csv",
    ),
    ModelEntry(
        "log_model_1a", "Model 1A (log10 space)", "per_wavelength", "sklearn", "log10",
        _LOG_1 / "model_1a" / "model_1a_hyperparameters.csv",
    ),
    ModelEntry(
        "log_model_1b", "Model 1B (log10 space)", "per_wavelength", "sklearn", "log10",
        _LOG_1 / "model_1b" / "model_1b_hyperparameters.csv",
    ),
    ModelEntry(
        "holdout_model_1a", "Model 1A (LOO holdout refit)", "per_wavelength", "sklearn", "linear",
        _HOLDOUT / "model_1a_hyperparameters.csv",
    ),
    ModelEntry(
        "holdout_model_1b", "Model 1B (LOO holdout refit)", "per_wavelength", "sklearn", "linear",
        _HOLDOUT / "model_1b_hyperparameters.csv",
    ),
    ModelEntry(
        "full_hf_only_gpboost", "HF-only GPBoost (full fit)", "per_wavelength", "gpboost", "linear",
        _GPBOOST_FULL / "hf_only_gpboost_hyperparameters.csv",
    ),
    ModelEntry(
        "full_model_1a_gpboost", "Model 1A GPBoost (full fit)", "per_wavelength", "gpboost", "linear",
        _GPBOOST_FULL / "model_1a_gpboost_hyperparameters.csv",
    ),
    ModelEntry(
        "full_model_1b_gpboost", "Model 1B GPBoost (full fit)", "per_wavelength", "gpboost", "linear",
        _GPBOOST_FULL / "model_1b_gpboost_hyperparameters.csv",
    ),
    # --- scalar (wavelength-augmented) Model 2 ---------------------------
    ModelEntry(
        "model2_linear", "Model 2 (linear)", "scalar", "sklearn", "linear",
        _MODELLING_2 / "linear" / "model2_hyperparameters.csv",
    ),
    ModelEntry(
        "model2_log10", "Model 2 (log10 space)", "scalar", "sklearn", "log10",
        _LOG_2 / "log10" / "model2_hyperparameters.csv",
    ),
    ModelEntry(
        "model2_gpboost", "Model 2 (GPBoost)", "scalar", "gpboost", "linear",
        _MODELLING_2 / "model2_gpboost" / "model2_gpboost_hyperparameters.csv",
    ),
    ModelEntry(
        "full_model_2_gpboost", "Model 2 GPBoost (full fit)", "scalar", "gpboost", "linear",
        _GPBOOST_FULL / "model_2_gpboost_hyperparameters.csv",
    ),
)


_GPBOOST_LOW_RANGE = re.compile(r"^(?:cov_)?low_GP_range_(\d+)$")
_GPBOOST_DELTA_RANGE = re.compile(r"^(?:cov_)?discrepancy_GP_range_(\d+)$")
_GPBOOST_HF_RANGE = re.compile(r"^(?:cov_)?GP_range_(\d+)$")
_GPBOOST_LOW_COMPACT_RANGE = re.compile(r"^low_range_(\d+)$")
_GPBOOST_DELTA_COMPACT_RANGE = re.compile(r"^delta_range_(\d+)$")
_GPBOOST_HF_COMPACT_RANGE = re.compile(r"^range_(\d+)$")


def _copy_column_if_absent(df: pd.DataFrame, source: str, target: str) -> None:
    if source in df.columns and target not in df.columns:
        df[target] = df[source]


def _copy_gpboost_range_columns(df: pd.DataFrame, *, entry: ModelEntry) -> None:
    """Expose GPBoost range parameters with the common length-scale names."""
    range_sources: dict[str, str] = {}
    for col in df.columns:
        if match := _GPBOOST_LOW_COMPACT_RANGE.match(col):
            range_sources[f"low_length_scale_{int(match.group(1))}"] = col
        elif match := _GPBOOST_DELTA_COMPACT_RANGE.match(col):
            range_sources[f"delta_length_scale_{int(match.group(1))}"] = col
        elif match := _GPBOOST_HF_COMPACT_RANGE.match(col):
            range_sources[f"low_length_scale_{int(match.group(1))}"] = col

    for col in df.columns:
        if match := _GPBOOST_LOW_RANGE.match(col):
            range_sources.setdefault(f"low_length_scale_{int(match.group(1)) - 1}", col)
        elif match := _GPBOOST_DELTA_RANGE.match(col):
            range_sources.setdefault(f"delta_length_scale_{int(match.group(1)) - 1}", col)
        elif match := _GPBOOST_HF_RANGE.match(col):
            range_sources.setdefault(f"low_length_scale_{int(match.group(1)) - 1}", col)

    for target, source in range_sources.items():
        _copy_column_if_absent(df, source, target)

    _copy_column_if_absent(df, "low_lambda_range_um", "low_lambda_length_scale_um")
    _copy_column_if_absent(df, "delta_lambda_range_um", "delta_lambda_length_scale_um")

    if entry.kind == "scalar":
        lambda_index = len(input_names())
        _copy_column_if_absent(
            df, f"low_length_scale_{lambda_index}", "low_lambda_length_scale_um"
        )
        _copy_column_if_absent(
            df, f"delta_length_scale_{lambda_index}", "delta_lambda_length_scale_um"
        )


def load_parameter_table(entry: ModelEntry) -> pd.DataFrame | None:
    """Read one model's CSV, normalised to the common vocabulary.

    Returns ``None`` (after logging a skip) when the CSV is absent, so the
    workflow runs over whatever has actually been fitted.
    """
    if not entry.csv_path.exists():
        print(f"  skip {entry.key}: no table at {entry.csv_path}")
        return None

    df = pd.read_csv(entry.csv_path)

    if entry.backend == "gpboost":
        _copy_gpboost_range_columns(df, entry=entry)

    return df


def available_entries(
    *, kind: str | None = None, backend: str | None = None
) -> list[ModelEntry]:
    """Registry entries whose CSV exists, optionally filtered by kind/backend."""
    entries = []
    for entry in REGISTRY:
        if kind is not None and entry.kind != kind:
            continue
        if backend is not None and entry.backend != backend:
            continue
        if entry.csv_path.exists():
            entries.append(entry)
    return entries


def shade_instrument_modes(ax, wavelengths) -> None:
    """Shade JWST instrument bands behind a wavelength-axis plot."""
    for mode, mask in instrument_mode_masks(wavelengths):
        wavelength_range = np.asarray(wavelengths)[mask]
        if wavelength_range.size == 0:
            continue
        ax.axvspan(
            wavelength_range.min(),
            wavelength_range.max(),
            color=mode.color,
            alpha=0.4,
            zorder=0,
        )


def length_scale_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Ordered ``<prefix>_length_scale_<i>`` columns present in ``df``.

    ``prefix`` is ``"low"`` or ``"delta"``. Sorted by the numeric dim index.
    """
    pattern = re.compile(rf"^{prefix}_length_scale_(\d+)$")
    matches = [(int(m.group(1)), c) for c in df.columns if (m := pattern.match(c))]
    return [col for _, col in sorted(matches)]


def signal_variance_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map display labels to whichever signal-variance columns are present."""
    labels = {
        "signal_variance": "signal variance",
        "low_signal_variance": "LF signal variance",
        "delta_signal_variance": "discrepancy signal variance",
    }
    return {label: col for col, label in labels.items() if col in df.columns}


def noise_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map a display label to whichever noise column(s) the table carries."""
    labels = {
        "low_noise": "LF noise",
        "high_noise": "HF noise",
        "error_var": "error variance",
    }
    return {label: col for col, label in labels.items() if col in df.columns}


def count_pinned(values, bounds, *, rtol: float = 1e-2) -> int:
    """Count entries resting within ``rtol`` (in log space) of either bound.

    Length scales pinned at the upper bound flag inert inputs; variances/noise
    pinned at a bound flag a degenerate fit. NaNs are ignored.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0
    lo, hi = bounds
    log_arr = np.log10(arr)
    near_lo = np.abs(log_arr - np.log10(lo)) <= rtol * abs(np.log10(lo))
    near_hi = np.abs(log_arr - np.log10(hi)) <= rtol * abs(np.log10(hi))
    return int(np.count_nonzero(near_lo | near_hi))
