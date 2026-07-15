"""LOO validation of Model 1A/1B on linear and log10 spectra.

This workflow keeps three comparisons side by side:

1. Model 1A vs 1B on the original eclipse-depth scale.
2. Model 1A vs 1B on log10-transformed spectra.
3. Linear-scale models vs log10 models after back-transforming the log10
   held-out predictions to original eclipse-depth units.

All fits are leave-one-out over the paired HF samples: each of the 97 spectra
is held out exactly once, and every fold-restricted rho fit uses only the
remaining paired HF/LF rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import (
    CVComparison,
    CVPredictions,
    compare_cv,
    cv_predict_rho_model,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import global_rho, per_wavelength_rho
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import inverse_log10_spectra, log10_spectra

OUTPUT_DIR = VALIDATION_RESULTS_DIR / "03_rho_loo_model_comparison"

SEED = RANDOM_SEED
MODEL_A_KEY = "model_1a"
MODEL_B_KEY = "model_1b"
MODEL_A_LABEL = "Model 1A (global rho)"
MODEL_B_LABEL = "Model 1B (per-wavelength rho)"
LINEAR_LABEL = "linear"
LOG_BACKTRANSFORMED_LABEL = "log10 back-transformed"


def fmt(value: float) -> str:
    """Compact numeric formatting for Markdown tables."""
    return f"{value:.6g}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def shade_instrument_modes(ax, wavelengths) -> None:
    for mode, mask in instrument_mode_masks(wavelengths):
        wavelength_range = wavelengths[mask]
        ax.axvspan(
            wavelength_range.min(),
            wavelength_range.max(),
            color=mode.color,
            alpha=0.4,
            zorder=0,
        )


def loo_rho_predictions(
    YHF: np.ndarray,
    YLF: np.ndarray,
    *,
    n_splits: int,
) -> dict[str, CVPredictions]:
    return {
        MODEL_A_KEY: cv_predict_rho_model(
            YHF,
            YLF,
            per_wavelength=False,
            n_splits=n_splits,
            seed=SEED,
        ),
        MODEL_B_KEY: cv_predict_rho_model(
            YHF,
            YLF,
            per_wavelength=True,
            n_splits=n_splits,
            seed=SEED,
        ),
    }


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


def additional_metric_summary(comparison: CVComparison) -> dict:
    """Extra descriptive metrics for readable reports."""
    return {
        "mean_per_sample_rmse_model_1": float(comparison.metrics_1.rmse_per_sample.mean()),
        "mean_per_sample_rmse_model_2": float(comparison.metrics_2.rmse_per_sample.mean()),
        "median_per_sample_rmse_model_1": float(
            np.median(comparison.metrics_1.rmse_per_sample)
        ),
        "median_per_sample_rmse_model_2": float(
            np.median(comparison.metrics_2.rmse_per_sample)
        ),
        "mean_per_wavelength_rmse_model_1": float(
            comparison.metrics_1.rmse_per_wavelength.mean()
        ),
        "mean_per_wavelength_rmse_model_2": float(
            comparison.metrics_2.rmse_per_wavelength.mean()
        ),
        "max_per_wavelength_rmse_model_1": float(
            comparison.metrics_1.rmse_per_wavelength.max()
        ),
        "max_per_wavelength_rmse_model_2": float(
            comparison.metrics_2.rmse_per_wavelength.max()
        ),
    }


def plot_model_1a_vs_1b(
    wavelengths: np.ndarray,
    YHF: np.ndarray,
    comparison: CVComparison,
    rho_b: np.ndarray,
    rho_a: float,
    predictions_a: CVPredictions,
    predictions_b: CVPredictions,
    *,
    output_units: str,
    actual_label: str,
    savepath: Path,
) -> None:
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(
        wavelengths,
        rho_b,
        ".-",
        color="tab:blue",
        lw=1,
        label="rho_j (Model 1B)",
    )
    ax.axhline(
        rho_a,
        color="tab:red",
        ls="--",
        label=f"global rho = {rho_a:.3f} (Model 1A)",
    )
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="rho",
        title=f"Fitted scaling coefficients ({output_units})",
    )
    ax.legend()

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(
        wavelengths,
        metrics_a.rmse_per_wavelength,
        lw=1.2,
        color="tab:red",
        label=MODEL_A_LABEL,
    )
    ax.plot(
        wavelengths,
        metrics_b.rmse_per_wavelength,
        lw=1.2,
        color="tab:blue",
        label=MODEL_B_LABEL,
    )
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="held-out RMSE",
        title="Per-wavelength LOO RMSE",
    )
    ax.legend()

    ax = axes[1, 0]
    order = np.argsort(comparison.delta_rmse_per_sample)
    colors = np.where(
        comparison.delta_rmse_per_sample[order] < 0, "tab:blue", "tab:red"
    )
    ax.bar(
        np.arange(len(order)),
        comparison.delta_rmse_per_sample[order],
        color=colors,
        width=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(
        xlabel="sample (sorted)",
        ylabel="RMSE(B) - RMSE(A)",
        title=(
            "Per-sample paired delta "
            f"(B wins on {comparison.fraction_samples_model2_wins:.0%})"
        ),
    )

    ax = axes[1, 1]
    ax.scatter(
        YHF.ravel(),
        predictions_a.y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:red",
        label=MODEL_A_LABEL,
    )
    ax.scatter(
        YHF.ravel(),
        predictions_b.y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:blue",
        label=MODEL_B_LABEL,
    )
    limits = [YHF.min(), YHF.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(
        xlabel=actual_label,
        ylabel="held-out prediction",
        title="Out-of-fold predictions vs actual",
    )
    ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_model_1a_vs_1b_outputs(
    output_dir: Path,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    YHF: np.ndarray,
    YLF: np.ndarray,
    predictions: dict[str, CVPredictions],
    *,
    output_units: str,
    actual_label: str,
) -> tuple[CVComparison, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_a = predictions[MODEL_A_KEY]
    predictions_b = predictions[MODEL_B_KEY]
    comparison = compare_cv(
        YHF,
        predictions_a,
        predictions_b,
        labels=(MODEL_A_LABEL, MODEL_B_LABEL),
    )
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    rho_a = global_rho(YHF, YLF)
    rho_b = per_wavelength_rho(YHF, YLF)

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rho_b": rho_b,
            "rmse_a": metrics_a.rmse_per_wavelength,
            "rmse_b": metrics_b.rmse_per_wavelength,
            "delta_rmse_b_minus_a": comparison.delta_rmse_per_wavelength,
        }
    ).to_csv(output_dir / "cv_per_wavelength.csv", index=False)

    pd.DataFrame(
        {
            "sample": np.arange(YHF.shape[0]),
            "spectrum": sample_labels,
            "fold": predictions_a.fold_of_sample,
            "rmse_a": metrics_a.rmse_per_sample,
            "rmse_b": metrics_b.rmse_per_sample,
            "delta_rmse_b_minus_a": comparison.delta_rmse_per_sample,
        }
    ).to_csv(output_dir / "cv_per_sample.csv", index=False)

    summary = {
        "comparison": "Model 1A vs Model 1B",
        "output_units": output_units,
        "model_a": MODEL_A_LABEL,
        "model_b": MODEL_B_LABEL,
        "n_samples": int(YHF.shape[0]),
        "n_splits": predictions_a.n_splits,
        "seed": SEED,
        "rho_global": rho_a,
        **comparison_summary(comparison),
        **additional_metric_summary(comparison),
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    plot_model_1a_vs_1b(
        wavelengths,
        YHF,
        comparison,
        rho_b,
        rho_a,
        predictions_a,
        predictions_b,
        output_units=output_units,
        actual_label=actual_label,
        savepath=output_dir / "01_model_1a_vs_1b.png",
    )
    write_model_1a_vs_1b_report(
        output_dir / "CV_REPORT.md",
        comparison=comparison,
        summary=summary,
        output_units=output_units,
        rho_a=rho_a,
        n_wavelengths=len(wavelengths),
    )
    return comparison, summary


def write_model_1a_vs_1b_report(
    savepath: Path,
    *,
    comparison: CVComparison,
    summary: dict,
    output_units: str,
    rho_a: float,
    n_wavelengths: int,
) -> None:
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    winner = MODEL_B_LABEL if comparison.delta_rmse_pooled < 0 else MODEL_A_LABEL
    report = f"""# LOO CV Report: Model 1A vs Model 1B

## Result

{winner} has the lower pooled held-out RMSE in this comparison.

| Metric | Model 1A | Model 1B | Delta B - A |
|---|---:|---:|---:|
| Pooled RMSE | {fmt(metrics_a.rmse_pooled)} | {fmt(metrics_b.rmse_pooled)} | {fmt(comparison.delta_rmse_pooled)} |
| Pooled NRMSE | {fmt(metrics_a.nrmse_pooled)} | {fmt(metrics_b.nrmse_pooled)} | {fmt(metrics_b.nrmse_pooled - metrics_a.nrmse_pooled)} |
| Mean per-sample RMSE | {fmt(summary["mean_per_sample_rmse_model_1"])} | {fmt(summary["mean_per_sample_rmse_model_2"])} | {fmt(summary["mean_per_sample_rmse_model_2"] - summary["mean_per_sample_rmse_model_1"])} |
| Median per-sample RMSE | {fmt(summary["median_per_sample_rmse_model_1"])} | {fmt(summary["median_per_sample_rmse_model_2"])} | {fmt(summary["median_per_sample_rmse_model_2"] - summary["median_per_sample_rmse_model_1"])} |
| Mean per-wavelength RMSE | {fmt(summary["mean_per_wavelength_rmse_model_1"])} | {fmt(summary["mean_per_wavelength_rmse_model_2"])} | {fmt(summary["mean_per_wavelength_rmse_model_2"] - summary["mean_per_wavelength_rmse_model_1"])} |
| Max per-wavelength RMSE | {fmt(summary["max_per_wavelength_rmse_model_1"])} | {fmt(summary["max_per_wavelength_rmse_model_2"])} | {fmt(summary["max_per_wavelength_rmse_model_2"] - summary["max_per_wavelength_rmse_model_1"])} |

Model 1B has lower per-sample RMSE on {pct(comparison.fraction_samples_model2_wins)} of held-out spectra. The mean paired per-sample delta RMSE (B - A) is {fmt(comparison.delta_mean)} with standard deviation {fmt(comparison.delta_std)}.

## Computation

- Output space: {output_units}.
- Samples: {summary["n_samples"]} paired HF/LF spectra.
- Wavelength bins: {n_wavelengths}.
- CV protocol: leave-one-out, `n_splits = {summary["n_splits"]}`, seed `{summary["seed"]}`.
- Each fold fits on 96 paired spectra and predicts the one held-out spectrum.
- Model 1A estimates one train-fold global rho pooled over all wavelengths.
- Model 1B estimates one train-fold rho per wavelength.
- Fold prediction formula: `mu_HF[j] + rho[j] * (YLF_holdout[j] - mu_LF[j])`.
- All train-fold moments (`mu_LF`, `sd_LF`, `mu_HF`) and rho values are computed without the held-out row.
- Full-data global rho, reported only as a diagnostic: `{fmt(rho_a)}`.
- RMSE is computed against held-out `YHF`; NRMSE divides pooled RMSE by the full true-spectrum range in this output space.

## Files

- `cv_per_sample.csv`: per-held-out-spectrum RMSE and paired delta.
- `cv_per_wavelength.csv`: per-wavelength RMSE, rho, and paired delta.
- `cv_summary.json`: machine-readable version of this summary.
- `01_model_1a_vs_1b.png`: diagnostic figure.
"""
    savepath.write_text(report)


def assert_same_cv_assignment(
    reference: CVPredictions,
    candidate: CVPredictions,
    *,
    label: str,
) -> None:
    if reference.n_splits != candidate.n_splits:
        raise RuntimeError(
            f"{label}: expected {reference.n_splits} splits, "
            f"found {candidate.n_splits}"
        )
    if not np.array_equal(reference.fold_of_sample, candidate.fold_of_sample):
        raise RuntimeError(f"{label}: fold assignments do not match")


def plot_log_vs_linear(
    wavelengths: np.ndarray,
    comparison_a: CVComparison,
    comparison_b: CVComparison,
    savepath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")

    for ax, comparison, label in (
        (axes[0, 0], comparison_a, MODEL_A_LABEL),
        (axes[1, 0], comparison_b, MODEL_B_LABEL),
    ):
        shade_instrument_modes(ax, wavelengths)
        ax.plot(
            wavelengths,
            comparison.metrics_1.rmse_per_wavelength,
            color="tab:green",
            lw=1.2,
            label="linear",
        )
        ax.plot(
            wavelengths,
            comparison.metrics_2.rmse_per_wavelength,
            color="tab:purple",
            lw=1.2,
            label="log10 back-transformed",
        )
        ax.set(
            ylabel="held-out RMSE",
            title=f"{label}: per-wavelength RMSE",
        )
        ax.legend()

    for ax, comparison, label in (
        (axes[0, 1], comparison_a, MODEL_A_LABEL),
        (axes[1, 1], comparison_b, MODEL_B_LABEL),
    ):
        order = np.argsort(comparison.delta_rmse_per_sample)
        colors = np.where(
            comparison.delta_rmse_per_sample[order] < 0,
            "tab:purple",
            "tab:green",
        )
        ax.bar(
            np.arange(len(order)),
            comparison.delta_rmse_per_sample[order],
            color=colors,
            width=1.0,
        )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set(
            ylabel="RMSE(log) - RMSE(linear)",
            title=(
                f"{label}: log wins on "
                f"{comparison.fraction_samples_model2_wins:.0%}"
            ),
        )

    axes[1, 0].set_xlabel(r"wavelength $\lambda$ [$\mu$m]")
    axes[1, 1].set_xlabel("sample (sorted)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_log_vs_linear_outputs(
    output_dir: Path,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    YHF_linear: np.ndarray,
    linear_predictions: dict[str, CVPredictions],
    log_predictions_linear_units: dict[str, CVPredictions],
) -> tuple[dict[str, CVComparison], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_a = compare_cv(
        YHF_linear,
        linear_predictions[MODEL_A_KEY],
        log_predictions_linear_units[MODEL_A_KEY],
        labels=(
            f"{MODEL_A_LABEL} ({LINEAR_LABEL})",
            f"{MODEL_A_LABEL} ({LOG_BACKTRANSFORMED_LABEL})",
        ),
    )
    comparison_b = compare_cv(
        YHF_linear,
        linear_predictions[MODEL_B_KEY],
        log_predictions_linear_units[MODEL_B_KEY],
        labels=(
            f"{MODEL_B_LABEL} ({LINEAR_LABEL})",
            f"{MODEL_B_LABEL} ({LOG_BACKTRANSFORMED_LABEL})",
        ),
    )

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rmse_linear_a": comparison_a.metrics_1.rmse_per_wavelength,
            "rmse_log_backtransformed_a": comparison_a.metrics_2.rmse_per_wavelength,
            "delta_rmse_log_minus_linear_a": comparison_a.delta_rmse_per_wavelength,
            "rmse_linear_b": comparison_b.metrics_1.rmse_per_wavelength,
            "rmse_log_backtransformed_b": comparison_b.metrics_2.rmse_per_wavelength,
            "delta_rmse_log_minus_linear_b": comparison_b.delta_rmse_per_wavelength,
        }
    ).to_csv(output_dir / "cv_per_wavelength.csv", index=False)

    pd.DataFrame(
        {
            "sample": np.arange(YHF_linear.shape[0]),
            "spectrum": sample_labels,
            "fold": linear_predictions[MODEL_A_KEY].fold_of_sample,
            "rmse_linear_a": comparison_a.metrics_1.rmse_per_sample,
            "rmse_log_backtransformed_a": comparison_a.metrics_2.rmse_per_sample,
            "delta_rmse_log_minus_linear_a": comparison_a.delta_rmse_per_sample,
            "rmse_linear_b": comparison_b.metrics_1.rmse_per_sample,
            "rmse_log_backtransformed_b": comparison_b.metrics_2.rmse_per_sample,
            "delta_rmse_log_minus_linear_b": comparison_b.delta_rmse_per_sample,
        }
    ).to_csv(output_dir / "cv_per_sample.csv", index=False)

    summary = {
        "comparison": "linear vs log10 rho models",
        "comparison_units": "original eclipse-depth units",
        "log_prediction_point_summary": (
            "10**mu_log10, the median of the implied linear-scale distribution"
        ),
        "n_samples": int(YHF_linear.shape[0]),
        "n_splits": linear_predictions[MODEL_A_KEY].n_splits,
        "seed": SEED,
        MODEL_A_KEY: {
            **comparison_summary(comparison_a),
            **additional_metric_summary(comparison_a),
        },
        MODEL_B_KEY: {
            **comparison_summary(comparison_b),
            **additional_metric_summary(comparison_b),
        },
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    plot_log_vs_linear(
        wavelengths,
        comparison_a,
        comparison_b,
        output_dir / "01_log_vs_linear.png",
    )
    write_log_vs_linear_report(
        output_dir / "CV_REPORT.md",
        comparison_a=comparison_a,
        comparison_b=comparison_b,
        summary=summary,
        n_wavelengths=len(wavelengths),
    )
    return {MODEL_A_KEY: comparison_a, MODEL_B_KEY: comparison_b}, summary


def write_log_vs_linear_report(
    savepath: Path,
    *,
    comparison_a: CVComparison,
    comparison_b: CVComparison,
    summary: dict,
    n_wavelengths: int,
) -> None:
    rows = []
    for key, label, comparison in (
        (MODEL_A_KEY, MODEL_A_LABEL, comparison_a),
        (MODEL_B_KEY, MODEL_B_LABEL, comparison_b),
    ):
        model_summary = summary[key]
        winner = "log10 back-transformed" if comparison.delta_rmse_pooled < 0 else "linear"
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    winner,
                    fmt(comparison.metrics_1.rmse_pooled),
                    fmt(comparison.metrics_2.rmse_pooled),
                    fmt(comparison.delta_rmse_pooled),
                    fmt(comparison.metrics_1.nrmse_pooled),
                    fmt(comparison.metrics_2.nrmse_pooled),
                    fmt(
                        comparison.metrics_2.nrmse_pooled
                        - comparison.metrics_1.nrmse_pooled
                    ),
                    pct(comparison.fraction_samples_model2_wins),
                    fmt(model_summary["delta_rmse_per_sample_mean"]),
                    fmt(model_summary["delta_rmse_per_sample_std"]),
                ]
            )
            + " |"
        )

    report = f"""# LOO CV Report: Linear (Non-Log) vs Log10 Models

## Result

The log10-fitted rho models are compared against the linear/non-log models after back-transforming their held-out predictions with `10**prediction`, so all metrics in this report are in original eclipse-depth units.

| Model | Lower pooled RMSE | Linear RMSE | Log10 back-transformed RMSE | Delta log - linear | Linear NRMSE | Log10 back-transformed NRMSE | Delta NRMSE | Log wins by sample | Mean sample delta | Sample delta std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Negative deltas mean the log10-fitted model is better after back-transformation.

## Computation

- Samples: {summary["n_samples"]} paired HF/LF spectra.
- Wavelength bins: {n_wavelengths}.
- CV protocol: leave-one-out, `n_splits = {summary["n_splits"]}`, seed `{summary["seed"]}`.
- Linear/non-log models are fitted and scored directly in original eclipse-depth units.
- Log models are fitted in `log10(YHF)` and `log10(YLF)` space.
- Log point predictions are converted back to original units as `10**mu_log10`; this is the predictive median implied by a log-normal interpretation.
- The fold assignment is identical between linear and log10 fits for each model before comparing paired errors.
- RMSE and NRMSE are computed against original-unit held-out `YHF`.

## Files

- `cv_per_sample.csv`: per-held-out-spectrum linear/log RMSE and paired delta for Models 1A and 1B.
- `cv_per_wavelength.csv`: per-wavelength linear/log RMSE and paired delta for Models 1A and 1B.
- `cv_summary.json`: machine-readable version of this summary.
- `01_log_vs_linear.png`: diagnostic figure.
"""
    savepath.write_text(report)


def write_combined_report(
    savepath: Path,
    *,
    linear_comparison: CVComparison,
    log_comparison: CVComparison,
    scale_comparisons: dict[str, CVComparison],
    combined_summary: dict,
) -> None:
    linear_delta = linear_comparison.delta_rmse_pooled
    log_delta = log_comparison.delta_rmse_pooled
    scale_a = scale_comparisons[MODEL_A_KEY]
    scale_b = scale_comparisons[MODEL_B_KEY]

    report = f"""# CV Comparison Report

## Scope

This report summarizes leave-one-out cross-validation over all {combined_summary["n_samples"]} paired HF samples for the closed-form rho models.

## Main Results

| Comparison | Units | Model/contrast | RMSE 1 | RMSE 2 | Delta 2 - 1 | Winner |
|---|---|---|---:|---:|---:|---|
| Model 1A vs Model 1B | Original eclipse-depth | A vs B | {fmt(linear_comparison.metrics_1.rmse_pooled)} | {fmt(linear_comparison.metrics_2.rmse_pooled)} | {fmt(linear_delta)} | {"Model 1B" if linear_delta < 0 else "Model 1A"} |
| Model 1A vs Model 1B | log10 eclipse-depth | A vs B | {fmt(log_comparison.metrics_1.rmse_pooled)} | {fmt(log_comparison.metrics_2.rmse_pooled)} | {fmt(log_delta)} | {"Model 1B" if log_delta < 0 else "Model 1A"} |
| Linear/non-log vs log10 | Original eclipse-depth | Model 1A linear/non-log vs log10 back-transformed | {fmt(scale_a.metrics_1.rmse_pooled)} | {fmt(scale_a.metrics_2.rmse_pooled)} | {fmt(scale_a.delta_rmse_pooled)} | {"log10" if scale_a.delta_rmse_pooled < 0 else "linear/non-log"} |
| Linear/non-log vs log10 | Original eclipse-depth | Model 1B linear/non-log vs log10 back-transformed | {fmt(scale_b.metrics_1.rmse_pooled)} | {fmt(scale_b.metrics_2.rmse_pooled)} | {fmt(scale_b.delta_rmse_pooled)} | {"log10" if scale_b.delta_rmse_pooled < 0 else "linear/non-log"} |

## Technical Notes

- All comparisons use the same 97 LOO splits with seed `{combined_summary["seed"]}`.
- Model 1A uses one global train-fold rho; Model 1B uses one train-fold rho per wavelength.
- For the log-vs-linear comparison, log predictions are scored only after back-transforming to original eclipse-depth units with `10**mu_log10`.
- Detailed per-comparison reports are in:
  - `linear/CV_REPORT.md`
  - `log10/CV_REPORT.md`
  - `log_vs_linear/CV_REPORT.md`
- CSV files provide per-sample and per-wavelength diagnostics for paper tables or plotting.
"""
    savepath.write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    sample_labels = data["YHF"].index.to_numpy()
    YHF_linear = data["YHF"].to_numpy()
    YLF_linear = data["YLF"].to_numpy()
    YHF_log = log10_spectra(YHF_linear)
    YLF_log = log10_spectra(YLF_linear)
    n_splits = YHF_linear.shape[0]

    print(
        f"LOO rho validation over all {n_splits} HF samples "
        f"(seed={SEED})"
    )
    linear_predictions = loo_rho_predictions(
        YHF_linear,
        YLF_linear,
        n_splits=n_splits,
    )
    log_predictions = loo_rho_predictions(
        YHF_log,
        YLF_log,
        n_splits=n_splits,
    )

    for key in (MODEL_A_KEY, MODEL_B_KEY):
        assert_same_cv_assignment(
            linear_predictions[key],
            log_predictions[key],
            label=key,
        )

    linear_comparison, linear_summary = write_model_1a_vs_1b_outputs(
        OUTPUT_DIR / "linear",
        wavelengths,
        sample_labels,
        YHF_linear,
        YLF_linear,
        linear_predictions,
        output_units="original eclipse-depth units",
        actual_label="actual HF eclipse depth",
    )
    log_comparison, log_summary = write_model_1a_vs_1b_outputs(
        OUTPUT_DIR / "log10",
        wavelengths,
        sample_labels,
        YHF_log,
        YLF_log,
        log_predictions,
        output_units="log10 eclipse-depth units",
        actual_label="actual HF log10 eclipse depth",
    )

    log_predictions_linear_units = {
        key: predictions.with_values(
            inverse_log10_spectra(predictions.y_pred),
            y_std=None,
        )
        for key, predictions in log_predictions.items()
    }
    scale_comparisons, scale_summary = write_log_vs_linear_outputs(
        OUTPUT_DIR / "log_vs_linear",
        wavelengths,
        sample_labels,
        YHF_linear,
        linear_predictions,
        log_predictions_linear_units,
    )

    combined_summary = {
        "n_samples": int(YHF_linear.shape[0]),
        "n_splits": n_splits,
        "seed": SEED,
        "linear_model_1a_vs_1b": linear_summary,
        "log10_model_1a_vs_1b": log_summary,
        "log_vs_linear": scale_summary,
    }
    (OUTPUT_DIR / "cv_summary.json").write_text(
        json.dumps(combined_summary, indent=2)
    )
    write_combined_report(
        OUTPUT_DIR / "CV_COMPARISON_REPORT.md",
        linear_comparison=linear_comparison,
        log_comparison=log_comparison,
        scale_comparisons=scale_comparisons,
        combined_summary=combined_summary,
    )

    print(
        "linear RMSE: "
        f"A={linear_comparison.metrics_1.rmse_pooled:.4e}, "
        f"B={linear_comparison.metrics_2.rmse_pooled:.4e}"
    )
    print(
        "log10 RMSE: "
        f"A={log_comparison.metrics_1.rmse_pooled:.4e}, "
        f"B={log_comparison.metrics_2.rmse_pooled:.4e}"
    )
    print(
        "linear-units log-vs-linear delta RMSE (log - linear): "
        f"A={scale_comparisons[MODEL_A_KEY].delta_rmse_pooled:+.4e}, "
        f"B={scale_comparisons[MODEL_B_KEY].delta_rmse_pooled:+.4e}"
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
