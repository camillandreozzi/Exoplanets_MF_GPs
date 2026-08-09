"""Paired GPBoost Model 2 vs Model 1B comparisons.

This experiment gives the GPBoost results their own clean comparison surface:

- ``matched`` uses the same canonical LF subsample and CV point budget as the
  sklearn/mf_gp custom-kernel comparison, so its errors can be compared against
  ``results/modelling/02_augmented_wavelength/cv/``.
- ``max-data`` uses GPBoost's large-data approximation and a larger capped data
  slice controlled by the configurable budgets below.

Both modes hold out whole HF spectra on the same K-fold assignment and compare
Model 2 (one wavelength-augmented AR(1) MF-GP) against the GPBoost Model 1B
baseline (one AR(1) MF-GP per wavelength). The HF-only sklearn GP is included
as a single-fidelity reference floor, matching the stage-1 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import (
    CV_FULL_MODEL_SPLITS,
    CVComparison,
    CVPredictions,
    compare_cv,
    cv_metrics,
    cv_predict_hf_only_gp,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.gpboost_mf import (
    GPBOOST_COV_FCT_SHAPE,
    GPBOOST_COV_FUNCTION,
    GPBOOST_GP_APPROX,
    GPBOOST_NUM_NEIGHBORS,
    cv_predict_model1_gpboost,
    gpboost_available,
)
from exoplanets_mf.gpboost_model2 import cv_predict_model2_gpboost
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.model2 import (
    MODEL2_CV_MAX_AUGMENTED_POINTS,
    derive_lambda_stride,
    select_wavelength_subgrid,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.provenance import write_run_metadata
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

RESULTS_DIR = MODELLING_RESULTS_DIR / "03_gpboost_comparison"

MATCHED_MODE = "matched"
MAX_DATA_MODE = "max-data"

# Conservative defaults for the max-data mode. These are deliberately larger
# than the matched 200-LF comparison but small enough to finish interactively.
# Override them with environment variables when the machine can tolerate more.
DEFAULT_MAX_MODEL1_LF_SAMPLE_SIZE = 1_000
DEFAULT_MAX_MODEL2_LF_SAMPLE_SIZE = 1_000
DEFAULT_MAX_MODEL2_MAX_POINTS = 10_000
DEFAULT_MAX_GP_APPROX = "vecchia"
DEFAULT_MAX_NUM_NEIGHBORS = 30

MODEL_LABELS = {
    "hf_only": "HF-only GP (single-fidelity sklearn baseline)",
    "model_1b_gpboost": "Model 1B GPBoost (per-wavelength rho)",
    "model_2_gpboost": "Model 2 GPBoost (wavelength-augmented scalar rho)",
}


@dataclass(frozen=True)
class ComparisonConfig:
    mode: str
    output_name: str
    description: str
    model1_subsample_size: int | None
    model2_lf_sample_size: int
    model2_max_augmented_points: int
    gp_approx: str
    num_neighbors: int | None
    n_splits: int
    seed: int
    progress_every: int | None = 40

    @property
    def output_dir(self) -> Path:
        return RESULTS_DIR / self.output_name


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive; found {value}")
    return value


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"none", "null", "default"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive or 'none'; found {value}")
    return value


def _env_subsample_size(name: str, default: str, n_available: int) -> int | None:
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"all", "full", "none"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive or 'all'; found {value}")
    return min(value, n_available)


def build_config(mode: str, *, n_lf: int) -> ComparisonConfig:
    if mode == MATCHED_MODE:
        return ComparisonConfig(
            mode=mode,
            output_name="matched_subsample",
            description=(
                "GPBoost comparison on the canonical LF subsample used by the "
                "sklearn/mf_gp custom-kernel comparison."
            ),
            model1_subsample_size=LF_SUBSAMPLE_SIZE,
            model2_lf_sample_size=LF_SUBSAMPLE_SIZE,
            model2_max_augmented_points=MODEL2_CV_MAX_AUGMENTED_POINTS,
            gp_approx=GPBOOST_GP_APPROX,
            num_neighbors=GPBOOST_NUM_NEIGHBORS,
            n_splits=CV_FULL_MODEL_SPLITS,
            seed=RANDOM_SEED,
        )
    if mode == MAX_DATA_MODE:
        model1_subsample_size = _env_subsample_size(
            "GPBOOST_MAX_MODEL1_LF_SAMPLE_SIZE",
            str(DEFAULT_MAX_MODEL1_LF_SAMPLE_SIZE),
            n_lf,
        )
        return ComparisonConfig(
            mode=mode,
            output_name="max_data",
            description=(
                "GPBoost comparison using the largest configured LF/data slice. "
                "Defaults use a capped LF sample with a Vecchia approximation."
            ),
            model1_subsample_size=model1_subsample_size,
            model2_lf_sample_size=min(
                _env_int(
                    "GPBOOST_MAX_MODEL2_LF_SAMPLE_SIZE",
                    DEFAULT_MAX_MODEL2_LF_SAMPLE_SIZE,
                ),
                n_lf,
            ),
            model2_max_augmented_points=_env_int(
                "GPBOOST_MAX_MODEL2_MAX_POINTS", DEFAULT_MAX_MODEL2_MAX_POINTS
            ),
            gp_approx=os.environ.get("GPBOOST_MAX_GP_APPROX", DEFAULT_MAX_GP_APPROX),
            num_neighbors=_env_optional_int(
                "GPBOOST_MAX_NUM_NEIGHBORS", DEFAULT_MAX_NUM_NEIGHBORS
            ),
            n_splits=_env_int("GPBOOST_MAX_N_SPLITS", CV_FULL_MODEL_SPLITS),
            seed=RANDOM_SEED,
        )
    raise ValueError(f"unknown mode {mode!r}")


def n_augmented_points(
    n_lf_samples: int,
    n_hf_samples: int,
    n_wavelengths: int,
    lambda_stride: int,
) -> int:
    offset = (lambda_stride // 2) % lambda_stride
    n_lf_lambda = len(
        select_wavelength_subgrid(n_wavelengths, stride=lambda_stride, offset=offset)
    )
    n_hf_lambda = len(select_wavelength_subgrid(n_wavelengths, stride=lambda_stride))
    return n_lf_samples * n_lf_lambda + n_hf_samples * n_hf_lambda


def actual_lf_count(subsample_size: int | None, n_lf: int) -> int:
    return n_lf if subsample_size is None else int(subsample_size)


def max_train_hf_count(n_samples: int, n_splits: int) -> int:
    """Largest train-fold size for sklearn KFold over n_samples."""
    return n_samples - (n_samples // n_splits)


def assert_same_cv_assignment(
    reference: CVPredictions, candidate: CVPredictions, *, label: str
) -> None:
    if reference.n_splits != candidate.n_splits:
        raise RuntimeError(
            f"{label}: expected {reference.n_splits} splits, "
            f"found {candidate.n_splits}"
        )
    if not np.array_equal(reference.fold_of_sample, candidate.fold_of_sample):
        raise RuntimeError(f"{label}: fold assignments do not match")


def comparison_summary(comparison: CVComparison) -> dict:
    return {
        "model_1": comparison.labels[0],
        "model_2": comparison.labels[1],
        "rmse_pooled_model_1": comparison.metrics_1.rmse_pooled,
        "rmse_pooled_model_2": comparison.metrics_2.rmse_pooled,
        "nrmse_pooled_model_1": comparison.metrics_1.nrmse_pooled,
        "nrmse_pooled_model_2": comparison.metrics_2.nrmse_pooled,
        "delta_rmse_pooled_model_2_minus_model_1": comparison.delta_rmse_pooled,
        "delta_rmse_per_sample_mean": comparison.delta_mean,
        "delta_rmse_per_sample_std": comparison.delta_std,
        "fraction_samples_model_2_wins": comparison.fraction_samples_model2_wins,
    }


def shade_instrument_modes(ax, wavelengths: np.ndarray) -> None:
    for mode, mask in instrument_mode_masks(wavelengths):
        wavelength_range = wavelengths[mask]
        ax.axvspan(
            wavelength_range.min(),
            wavelength_range.max(),
            color=mode.color,
            alpha=0.4,
            zorder=0,
        )


def plot_diagnostics(
    wavelengths: np.ndarray,
    Y_hf: np.ndarray,
    predictions: dict[str, CVPredictions],
    comparison_vs_model1: CVComparison,
    savepath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    colors = {
        "hf_only": "tab:gray",
        "model_1b_gpboost": "tab:blue",
        "model_2_gpboost": "tab:red",
    }

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    for key, prediction in predictions.items():
        metrics = cv_metrics(Y_hf, prediction)
        ax.plot(
            wavelengths,
            metrics.rmse_per_wavelength,
            lw=1.2,
            color=colors[key],
            label=MODEL_LABELS[key],
        )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="held-out RMSE",
        title="Per-wavelength CV RMSE",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths)
    for key, prediction in predictions.items():
        if prediction.y_std is None:
            continue
        z = (prediction.y_pred - Y_hf) / prediction.y_std
        ax.plot(
            wavelengths,
            np.mean(np.abs(z) <= 1.96, axis=0),
            lw=1.2,
            color=colors[key],
            label=MODEL_LABELS[key],
        )
    ax.axhline(0.95, color="black", ls="--", lw=0.8, label="nominal 95%")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="95% CI coverage",
        title="Coverage vs wavelength",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    order = np.argsort(comparison_vs_model1.delta_rmse_per_sample)
    colors_delta = np.where(
        comparison_vs_model1.delta_rmse_per_sample[order] < 0,
        "tab:red",
        "tab:blue",
    )
    ax.bar(
        np.arange(len(order)),
        comparison_vs_model1.delta_rmse_per_sample[order],
        color=colors_delta,
        width=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(
        xlabel="sample (sorted)",
        ylabel="RMSE(Model 2) - RMSE(Model 1B)",
        title=(
            "Per-sample paired delta (Model 2 wins on "
            f"{comparison_vs_model1.fraction_samples_model2_wins:.0%})"
        ),
    )

    ax = axes[1, 1]
    ax.scatter(
        Y_hf.ravel(),
        predictions["model_1b_gpboost"].y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:blue",
        label=MODEL_LABELS["model_1b_gpboost"],
    )
    ax.scatter(
        Y_hf.ravel(),
        predictions["model_2_gpboost"].y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:red",
        label=MODEL_LABELS["model_2_gpboost"],
    )
    limits = [Y_hf.min(), Y_hf.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(
        xlabel="actual HF eclipse depth",
        ylabel="held-out prediction",
        title="Out-of-fold predictions vs actual",
    )
    ax.legend(markerscale=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    config: ComparisonConfig,
    *,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    Y_hf: np.ndarray,
    predictions: dict[str, CVPredictions],
    lambda_stride: int,
    model2_stride_sizing_points: int,
    model2_max_fold_points: int,
    n_lf_total: int,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = {
        "hf_only": compare_cv(
            Y_hf,
            predictions["hf_only"],
            predictions["model_2_gpboost"],
            labels=(MODEL_LABELS["hf_only"], MODEL_LABELS["model_2_gpboost"]),
        ),
        "model_1b_gpboost": compare_cv(
            Y_hf,
            predictions["model_1b_gpboost"],
            predictions["model_2_gpboost"],
            labels=(
                MODEL_LABELS["model_1b_gpboost"],
                MODEL_LABELS["model_2_gpboost"],
            ),
        ),
    }
    all_metrics = {key: cv_metrics(Y_hf, value) for key, value in predictions.items()}

    per_wavelength = {"wavelength": wavelengths}
    per_sample = {
        "sample": np.arange(Y_hf.shape[0]),
        "spectrum": sample_labels,
        "fold": predictions["model_2_gpboost"].fold_of_sample,
    }
    for key, metrics in all_metrics.items():
        per_wavelength[f"rmse_{key}"] = metrics.rmse_per_wavelength
        per_sample[f"rmse_{key}"] = metrics.rmse_per_sample
    for key, comparison in comparisons.items():
        per_wavelength[f"delta_rmse_model2_minus_{key}"] = (
            comparison.delta_rmse_per_wavelength
        )
        per_sample[f"delta_rmse_model2_minus_{key}"] = (
            comparison.delta_rmse_per_sample
        )

    pd.DataFrame(per_wavelength).to_csv(
        output_dir / "cv_per_wavelength.csv", index=False
    )
    pd.DataFrame(per_sample).to_csv(output_dir / "cv_per_sample.csv", index=False)

    summary = {
        "result_set": config.output_name,
        "description": config.description,
        "output_units": "original eclipse-depth units",
        "n_samples": int(Y_hf.shape[0]),
        "n_wavelengths": int(Y_hf.shape[1]),
        "n_splits": config.n_splits,
        "seed": config.seed,
        "cov_function": GPBOOST_COV_FUNCTION,
        "cov_fct_shape": GPBOOST_COV_FCT_SHAPE,
        "gp_approx": config.gp_approx,
        "num_neighbors": config.num_neighbors,
        "total_lf_rows_available": int(n_lf_total),
        "model_1b_lf_sample_size": actual_lf_count(
            config.model1_subsample_size, n_lf_total
        ),
        "model_1b_subsample_is_all_lf": config.model1_subsample_size is None,
        "model_2_lf_sample_size": config.model2_lf_sample_size,
        "model_2_max_augmented_points": config.model2_max_augmented_points,
        "model_2_lambda_stride": lambda_stride,
        "model_2_stride_sizing_augmented_points": model2_stride_sizing_points,
        "model_2_max_cv_fold_augmented_points": model2_max_fold_points,
        "pooled_rmse": {
            key: all_metrics[key].rmse_pooled for key in all_metrics
        },
        "coverage_95": {
            key: all_metrics[key].coverage_95
            for key in all_metrics
            if all_metrics[key].coverage_95 is not None
        },
        "comparisons_vs_model_2": {
            key: comparison_summary(value) for key, value in comparisons.items()
        },
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    np.savez(
        output_dir / "cv_predictions.npz",
        fold_of_sample=predictions["model_2_gpboost"].fold_of_sample,
        **{f"y_pred_{key}": value.y_pred for key, value in predictions.items()},
        **{
            f"y_std_{key}": value.y_std
            for key, value in predictions.items()
            if value.y_std is not None
        },
    )

    plot_diagnostics(
        wavelengths,
        Y_hf,
        predictions,
        comparisons["model_1b_gpboost"],
        output_dir / "01_gpboost_model_comparison.png",
    )
    write_report(
        output_dir / "CV_REPORT.md",
        summary=summary,
        comparisons=comparisons,
    )
    write_run_metadata(
        output_dir / "run_metadata.json",
        workflow="modelling/03_gpboost_comparison",
        scripts=[Path(__file__).name, f"mode={config.mode}"],
    )


def fmt(value: float) -> str:
    return f"{value:.6g}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(
    savepath: Path,
    *,
    summary: dict,
    comparisons: dict[str, CVComparison],
) -> None:
    rows = []
    for key, comparison in comparisons.items():
        winner = (
            MODEL_LABELS["model_2_gpboost"]
            if comparison.delta_rmse_pooled < 0
            else MODEL_LABELS[key]
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[key],
                    fmt(comparison.metrics_1.rmse_pooled),
                    fmt(comparison.metrics_2.rmse_pooled),
                    fmt(comparison.delta_rmse_pooled),
                    pct(comparison.fraction_samples_model2_wins),
                    winner,
                ]
            )
            + " |"
        )
    coverage_lines = "\n".join(
        f"- {MODEL_LABELS[key]}: {pct(value)} (nominal 95%)"
        for key, value in summary["coverage_95"].items()
    )
    report = f"""# GPBoost CV Report: {summary["result_set"]}

## Result

Each row is a paired comparison of held-out errors on identical folds. Delta is
Model 2 minus the baseline, so negative deltas mean Model 2 wins.

| Baseline | Baseline RMSE | Model 2 RMSE | Delta | Model 2 wins by sample | Winner |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

95% CI coverage of models with predictive uncertainties:

{coverage_lines}

## Configuration

- {summary["description"]}
- Output scale: original eclipse-depth units.
- CV protocol: {summary["n_splits"]}-fold over {summary["n_samples"]} paired HF spectra, seed `{summary["seed"]}`.
- GPBoost covariance: `{summary["cov_function"]}` with `cov_fct_shape={summary["cov_fct_shape"]}`.
- GPBoost approximation: `gp_approx={summary["gp_approx"]}`, `num_neighbors={summary["num_neighbors"]}`.
- Model 1B LF rows: {summary["model_1b_lf_sample_size"]} of {summary["total_lf_rows_available"]}.
- Model 2 LF rows: {summary["model_2_lf_sample_size"]} of {summary["total_lf_rows_available"]}; wavelength stride {summary["model_2_lambda_stride"]}; {summary["model_2_max_cv_fold_augmented_points"]} augmented scalar rows in the largest CV fold.
- Model 2 stride was sized against {summary["model_2_stride_sizing_augmented_points"]} augmented scalar rows, using all HF rows as the conservative budget check.

## Files

- `cv_per_sample.csv`, `cv_per_wavelength.csv`: per-model RMSE and paired deltas.
- `cv_summary.json`: machine-readable summary and configuration.
- `cv_predictions.npz`: out-of-fold predictions and standard deviations.
- `01_gpboost_model_comparison.png`: diagnostic figure.
- `run_metadata.json`: environment, git status, seed, package versions, and data checksums.
"""
    savepath.write_text(report, encoding="utf-8")


def run(mode: str) -> None:
    data = load_all()
    wavelengths = data["wavelengths"]
    sample_labels = data["YHF"].index.to_numpy()
    X_lf = data["XLF_10k"].to_numpy()
    Y_lf = data["YLF_10k"].to_numpy()
    X_hf = data["XHF"].to_numpy()
    Y_hf = data["YHF"].to_numpy()

    config = build_config(mode, n_lf=len(X_lf))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not gpboost_available():
        message = (
            "gpboost not available -- skipping GPBoost comparison.\n"
            "Install gpboost and run tools/fix_gpboost_libomp.py; see README.\n"
        )
        (config.output_dir / "SKIPPED.txt").write_text(message, encoding="utf-8")
        print(message.rstrip())
        return

    lambda_stride = derive_lambda_stride(
        config.model2_lf_sample_size,
        len(X_hf),
        len(wavelengths),
        max_points=config.model2_max_augmented_points,
    )
    model2_stride_sizing_points = n_augmented_points(
        config.model2_lf_sample_size,
        len(X_hf),
        len(wavelengths),
        lambda_stride,
    )
    model2_max_fold_points = n_augmented_points(
        config.model2_lf_sample_size,
        max_train_hf_count(len(X_hf), config.n_splits),
        len(wavelengths),
        lambda_stride,
    )

    print(
        f"GPBoost {config.mode} CV: model1_lf="
        f"{actual_lf_count(config.model1_subsample_size, len(X_lf))}, "
        f"model2_lf={config.model2_lf_sample_size}, "
        f"model2_stride={lambda_stride}, gp_approx={config.gp_approx}, "
        f"n_splits={config.n_splits}"
    )

    predictions = {}
    print("  HF-only baseline CV ...", flush=True)
    predictions["hf_only"] = cv_predict_hf_only_gp(
        X_hf,
        Y_hf,
        wavelengths,
        n_splits=config.n_splits,
        seed=config.seed,
    )

    print("  Model 1B GPBoost CV ...", flush=True)
    predictions["model_1b_gpboost"] = cv_predict_model1_gpboost(
        X_lf,
        Y_lf,
        X_hf,
        Y_hf,
        wavelengths,
        seed=config.seed,
        subsample_size=config.model1_subsample_size,
        n_splits=config.n_splits,
        progress_every=config.progress_every,
        gp_approx=config.gp_approx,
        num_neighbors=config.num_neighbors,
    )

    print("  Model 2 GPBoost CV ...", flush=True)
    predictions["model_2_gpboost"] = cv_predict_model2_gpboost(
        X_lf,
        Y_lf,
        X_hf,
        Y_hf,
        wavelengths,
        seed=config.seed,
        lf_sample_size=config.model2_lf_sample_size,
        lambda_stride=lambda_stride,
        n_splits=config.n_splits,
        progress=True,
        gp_approx=config.gp_approx,
        num_neighbors=config.num_neighbors,
    )

    for key, prediction in predictions.items():
        assert_same_cv_assignment(
            predictions["model_2_gpboost"], prediction, label=key
        )

    write_outputs(
        config,
        wavelengths=wavelengths,
        sample_labels=sample_labels,
        Y_hf=Y_hf,
        predictions=predictions,
        lambda_stride=lambda_stride,
        model2_stride_sizing_points=model2_stride_sizing_points,
        model2_max_fold_points=model2_max_fold_points,
        n_lf_total=len(X_lf),
    )
    print(f"outputs written to {config.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(MATCHED_MODE, MAX_DATA_MODE),
        default=MATCHED_MODE,
        help=(
            "matched: canonical LF subsample comparable to sklearn/mf_gp; "
            "max-data: configurable larger GPBoost/Vecchia run"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
