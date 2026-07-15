"""5-fold CV of Model 2K, extending the stored Model 2 comparison.

The baselines (Model 1A/1B, Model 1 joint GP, subsampled Model 2) are NOT
re-run: their out-of-fold predictions are loaded read-only from
results/modelling/02_augmented_wavelength/cv/{scale}/cv_predictions.npz and
extended with Model 2K as an additional entry -- the old-vs-new contrast is
itself a result, so nothing under 02_augmented_wavelength is modified. The
fold assignment is asserted equal BEFORE any metric is computed: every model
uses the same KFold(n_splits, shuffle, random_state=seed) on the same 97
samples, so a mismatch would mean the pairing is invalid.

Model 2K's stage L is HF-free, so the production stage-L fit from
01_fit_model2k.py is reused across all folds (loaded from
model2k_layer.joblib); each fold refits only stage delta (97 x 195 grid,
seconds) on its training HF rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from exoplanets_mf.cv import (
    CV_FULL_MODEL_SPLITS,
    CVComparison,
    CVPredictions,
    compare_cv,
    cv_metrics,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.model2_kron import cv_predict_model2_kron
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import inverse_log10_spectra, log10_spectra

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02b_kronecker" / "cv"
PRODUCTION_DIR = MODELLING_RESULTS_DIR / "02b_kronecker"
BASELINE_CV_DIR = MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv"

SEED = RANDOM_SEED
N_SPLITS = CV_FULL_MODEL_SPLITS
N_RESTARTS = 10

CI_Z = stats.norm.ppf(0.975)
BASELINE_LABELS = {
    "model_1a": "Model 1A (global rho)",
    "model_1b": "Model 1B (per-wavelength rho)",
    "model_1_joint": "Model 1 joint GP (200-LF testing config)",
    "model_2": "Model 2 (subsampled joint MF-GP)",
}
MODEL_2K_LABEL = "Model 2K (exact Kronecker two-stage)"
MODEL_LABELS = {**BASELINE_LABELS, "model_2k": MODEL_2K_LABEL}


def fmt(value: float) -> str:
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


def load_baseline_predictions(scale: str) -> dict[str, CVPredictions]:
    """Stored out-of-fold predictions of every baseline (read-only)."""
    path = BASELINE_CV_DIR / scale / "cv_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run research.modelling.02_augmented_wavelength "
            "first; the Model 2K comparison extends its stored CV predictions"
        )
    predictions: dict[str, CVPredictions] = {}
    with np.load(path) as npz:
        fold_of_sample = npz["fold_of_sample"]
        for key in BASELINE_LABELS:
            pred_key, std_key = f"y_pred_{key}", f"y_std_{key}"
            if pred_key not in npz.files:
                continue
            predictions[key] = CVPredictions(
                y_pred=npz[pred_key],
                y_std=npz[std_key] if std_key in npz.files else None,
                fold_of_sample=fold_of_sample,
                n_splits=N_SPLITS,
            )
    if "model_2" not in predictions:
        raise RuntimeError(f"stored baseline file {path} lacks Model 2 predictions")
    return predictions


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


def load_production_stage_l(scale: str):
    path = PRODUCTION_DIR / scale / "model2k_layer.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run 01_fit_model2k.py first; CV reuses its "
            "HF-free stage-L fit across all folds"
        )
    return joblib.load(path).stage_l


def comparison_summary(comparison: CVComparison) -> dict:
    return {
        "baseline": comparison.labels[0],
        "model_2k": comparison.labels[1],
        "rmse_pooled_baseline": comparison.metrics_1.rmse_pooled,
        "rmse_pooled_model_2k": comparison.metrics_2.rmse_pooled,
        "delta_rmse_pooled_model_2k_minus_baseline": comparison.delta_rmse_pooled,
        "delta_rmse_per_sample_mean": comparison.delta_mean,
        "delta_rmse_per_sample_std": comparison.delta_std,
        "fraction_samples_model_2k_wins": comparison.fraction_samples_model2_wins,
    }


def plot_scale_diagnostics(
    wavelengths: np.ndarray,
    YHF: np.ndarray,
    predictions: dict[str, CVPredictions],
    comparison_vs_model2: CVComparison,
    *,
    actual_label: str,
    savepath: Path,
) -> None:
    metric_lines = [
        ("model_1b", "tab:blue"),
        ("model_2", "tab:green"),
        ("model_2k", "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    for key, color in metric_lines:
        metrics = cv_metrics(YHF, predictions[key])
        ax.plot(
            wavelengths, metrics.rmse_per_wavelength, lw=1.2, color=color,
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
    for key, color in metric_lines:
        if predictions[key].y_std is None:
            continue
        z = (predictions[key].y_pred - YHF) / predictions[key].y_std
        ax.plot(
            wavelengths, np.mean(np.abs(z) <= CI_Z, axis=0), lw=1.2, color=color,
            label=MODEL_LABELS[key],
        )
    ax.axhline(0.95, color="black", ls="--", lw=0.8, label="nominal 95%")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="95% CI coverage",
        title="Coverage vs wavelength",
    )
    ax.legend(fontsize=8)

    # The old-vs-new contrast is the headline result of this workflow, so the
    # paired-delta panel compares Model 2K against the subsampled Model 2.
    ax = axes[1, 0]
    order = np.argsort(comparison_vs_model2.delta_rmse_per_sample)
    colors = np.where(
        comparison_vs_model2.delta_rmse_per_sample[order] < 0, "tab:red", "tab:blue"
    )
    ax.bar(
        np.arange(len(order)),
        comparison_vs_model2.delta_rmse_per_sample[order],
        color=colors,
        width=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(
        xlabel="sample (sorted)",
        ylabel="RMSE(Model 2K) - RMSE(Model 2)",
        title=(
            "Per-sample paired delta (Model 2K wins on "
            f"{comparison_vs_model2.fraction_samples_model2_wins:.0%})"
        ),
    )

    ax = axes[1, 1]
    ax.scatter(
        YHF.ravel(), predictions["model_2"].y_pred.ravel(), s=2, alpha=0.1,
        color="tab:green", label=MODEL_LABELS["model_2"],
    )
    ax.scatter(
        YHF.ravel(), predictions["model_2k"].y_pred.ravel(), s=2, alpha=0.1,
        color="tab:red", label=MODEL_LABELS["model_2k"],
    )
    limits = [YHF.min(), YHF.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(
        xlabel=actual_label,
        ylabel="held-out prediction",
        title="Out-of-fold predictions vs actual",
    )
    ax.legend(markerscale=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_scale_outputs(
    output_dir: Path,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    YHF: np.ndarray,
    predictions: dict[str, CVPredictions],
    *,
    output_units: str,
    actual_label: str,
) -> dict[str, CVComparison]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = [key for key in BASELINE_LABELS if key in predictions]
    comparisons = {
        key: compare_cv(
            YHF,
            predictions[key],
            predictions["model_2k"],
            labels=(MODEL_LABELS[key], MODEL_2K_LABEL),
        )
        for key in baselines
    }
    all_metrics = {key: cv_metrics(YHF, value) for key, value in predictions.items()}

    per_wavelength = {"wavelength": wavelengths}
    per_sample = {
        "sample": np.arange(YHF.shape[0]),
        "spectrum": sample_labels,
        "fold": predictions["model_2k"].fold_of_sample,
    }
    for key, metrics in all_metrics.items():
        per_wavelength[f"rmse_{key}"] = metrics.rmse_per_wavelength
        per_sample[f"rmse_{key}"] = metrics.rmse_per_sample
    for key in baselines:
        per_wavelength[f"delta_rmse_model2k_minus_{key}"] = comparisons[
            key
        ].delta_rmse_per_wavelength
        per_sample[f"delta_rmse_model2k_minus_{key}"] = comparisons[
            key
        ].delta_rmse_per_sample
    pd.DataFrame(per_wavelength).to_csv(
        output_dir / "cv_per_wavelength.csv", index=False
    )
    pd.DataFrame(per_sample).to_csv(output_dir / "cv_per_sample.csv", index=False)

    summary = {
        "output_units": output_units,
        "n_samples": int(YHF.shape[0]),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "model_2k_n_restarts": N_RESTARTS,
        "baseline_predictions_source": str(
            BASELINE_CV_DIR / output_dir.name / "cv_predictions.npz"
        ),
        "pooled_rmse": {key: all_metrics[key].rmse_pooled for key in all_metrics},
        "coverage_95": {
            key: all_metrics[key].coverage_95
            for key in all_metrics
            if all_metrics[key].coverage_95 is not None
        },
        "comparisons_vs_model_2k": {
            key: comparison_summary(comparisons[key]) for key in comparisons
        },
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    # Only Model 2K's arrays are stored here; the baselines stay in their
    # original npz (referenced in the summary), which is never rewritten.
    np.savez(
        output_dir / "cv_predictions.npz",
        fold_of_sample=predictions["model_2k"].fold_of_sample,
        y_pred_model_2k=predictions["model_2k"].y_pred,
        y_std_model_2k=predictions["model_2k"].y_std,
    )

    plot_scale_diagnostics(
        wavelengths,
        YHF,
        predictions,
        comparisons["model_2"],
        actual_label=actual_label,
        savepath=output_dir / "02_model2k_vs_baselines.png",
    )
    write_scale_report(
        output_dir / "CV_REPORT.md",
        comparisons=comparisons,
        summary=summary,
        output_units=output_units,
    )
    return comparisons


def write_scale_report(
    savepath: Path,
    *,
    comparisons: dict[str, CVComparison],
    summary: dict,
    output_units: str,
) -> None:
    rows = []
    for key, comparison in comparisons.items():
        winner = (
            MODEL_2K_LABEL if comparison.delta_rmse_pooled < 0 else MODEL_LABELS[key]
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
                    fmt(comparison.delta_mean),
                    winner,
                ]
            )
            + " |"
        )
    coverage_lines = "\n".join(
        f"- {MODEL_LABELS[key]}: {pct(value)} (nominal 95%)"
        for key, value in summary["coverage_95"].items()
    )
    report = f"""# CV Report: Model 2K vs stored baselines ({output_units})

## Result

Each row is a paired comparison of held-out errors on identical folds; the
delta is Model 2K minus the baseline, so negative deltas mean Model 2K wins.

| Baseline | Baseline RMSE | Model 2K RMSE | Delta | Model 2K wins by sample | Mean sample delta | Winner |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

95% CI coverage of the models with predictive uncertainties:

{coverage_lines}

## Computation

- Output space: {output_units}.
- Samples: {summary["n_samples"]} paired HF spectra; whole spectra held out per fold.
- CV protocol: {summary["n_splits"]}-fold, seed `{summary["seed"]}`; the baselines' stored fold assignment was asserted equal to Model 2K's before any metric.
- Model 2K trains EXACTLY on the complete grids per fold: the shared HF-free stage L (10,097 x 195 LF points, reused from 01_fit_model2k.py) plus a stage-delta refit on the training HF rows x all 195 wavelengths ({summary["model_2k_n_restarts"]} restarts). No wavelength subsampling -- unlike the stored Model 2 rows (stride-13 CV configuration).
- Baseline predictions loaded read-only from `{summary["baseline_predictions_source"]}`.
- Model 1A/1B consume the paired LF spectrum of the held-out sample at prediction time; Model 2, Model 2K and the Model 1 joint GP predict from atmospheric inputs alone (Model 2K's stage L, like every model's LF design, contains LF information at the held-out inputs -- the standing "LF is always fully available" protocol).

## Files

- `cv_per_sample.csv`, `cv_per_wavelength.csv`: per-model RMSE and paired deltas.
- `cv_summary.json`: machine-readable version of this summary.
- `cv_predictions.npz`: Model 2K's out-of-fold predictions (baselines stay in their original npz).
- `02_model2k_vs_baselines.png`: diagnostic figure.
"""
    savepath.write_text(report)


def write_log_vs_linear_outputs(
    output_dir: Path,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    YHF_linear: np.ndarray,
    model2k_linear: CVPredictions,
    model2k_log_backtransformed: CVPredictions,
) -> CVComparison:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_cv(
        YHF_linear,
        model2k_linear,
        model2k_log_backtransformed,
        labels=("Model 2K (linear)", "Model 2K (log10 back-transformed)"),
    )
    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rmse_linear": comparison.metrics_1.rmse_per_wavelength,
            "rmse_log_backtransformed": comparison.metrics_2.rmse_per_wavelength,
            "delta_rmse_log_minus_linear": comparison.delta_rmse_per_wavelength,
        }
    ).to_csv(output_dir / "cv_per_wavelength.csv", index=False)
    pd.DataFrame(
        {
            "sample": np.arange(YHF_linear.shape[0]),
            "spectrum": sample_labels,
            "fold": model2k_linear.fold_of_sample,
            "rmse_linear": comparison.metrics_1.rmse_per_sample,
            "rmse_log_backtransformed": comparison.metrics_2.rmse_per_sample,
            "delta_rmse_log_minus_linear": comparison.delta_rmse_per_sample,
        }
    ).to_csv(output_dir / "cv_per_sample.csv", index=False)
    (output_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "comparison": "Model 2K linear vs Model 2K log10 back-transformed",
                "comparison_units": "original eclipse-depth units",
                "log_prediction_point_summary": (
                    "10**mu_log10, the median of the implied linear-scale "
                    "distribution"
                ),
                "n_samples": int(YHF_linear.shape[0]),
                "n_splits": N_SPLITS,
                "seed": SEED,
                "rmse_pooled_linear": comparison.metrics_1.rmse_pooled,
                "rmse_pooled_log_backtransformed": comparison.metrics_2.rmse_pooled,
                "delta_rmse_pooled_log_minus_linear": comparison.delta_rmse_pooled,
                "fraction_samples_log_wins": comparison.fraction_samples_model2_wins,
            },
            indent=2,
        )
    )
    return comparison


def write_combined_report(
    savepath: Path,
    *,
    linear_comparisons: dict[str, CVComparison],
    log_comparisons: dict[str, CVComparison],
    scale_comparison: CVComparison,
) -> None:
    rows = []
    for units, comparisons in (
        ("original eclipse-depth", linear_comparisons),
        ("log10 eclipse-depth", log_comparisons),
    ):
        for key, comparison in comparisons.items():
            winner = (
                "Model 2K" if comparison.delta_rmse_pooled < 0 else MODEL_LABELS[key]
            )
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"Model 2K vs {MODEL_LABELS[key]}",
                        units,
                        fmt(comparison.metrics_1.rmse_pooled),
                        fmt(comparison.metrics_2.rmse_pooled),
                        fmt(comparison.delta_rmse_pooled),
                        winner,
                    ]
                )
                + " |"
            )
    rows.append(
        "| "
        + " | ".join(
            [
                "Model 2K log10 (back-transformed) vs Model 2K linear",
                "original eclipse-depth",
                fmt(scale_comparison.metrics_1.rmse_pooled),
                fmt(scale_comparison.metrics_2.rmse_pooled),
                fmt(scale_comparison.delta_rmse_pooled),
                "log10" if scale_comparison.delta_rmse_pooled < 0 else "linear",
            ]
        )
        + " |"
    )

    report = f"""# CV Comparison Report: Model 2K (exact Kronecker two-stage MF-GP)

## Scope

{N_SPLITS}-fold cross-validation over all 97 paired HF samples, seed `{SEED}`,
identical fold assignment for every model (asserted against the stored
baseline predictions of research.modelling.02_augmented_wavelength). Model 2K
replaces Model 2's 0.17% data subsample with EXACT inference on the complete
Cartesian (theta, lambda) grids via the two-stage Le Gratiet recursion and
per-stage Kronecker structure; the baselines' numbers are the stored ones --
the old-vs-new Model 2 contrast is itself a result.

## Main Results

| Comparison | Units | Baseline RMSE | Model 2K RMSE | Delta (2K - baseline) | Winner |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

Negative deltas mean Model 2K (or the log10 variant in the last row) wins.

## Technical Notes

- Model 2K per fold: shared HF-free stage L (exact fit on the 10,097 x 195 LF grid, fitted once in 01_fit_model2k.py) + stage-delta refit on the training HF rows at ALL 195 wavelengths with profiled scalar rho ({N_RESTARTS} restarts, seconds per fold).
- Stored Model 2 rows come from its stride-13 / 40-LF-sample CV configuration; Model 1 joint GP from its 200-LF testing configuration -- their caveats in `02_augmented_wavelength/cv/CV_COMPARISON_REPORT.md` still apply.
- Model 1A/1B closed-form baselines consume the paired LF spectrum of the held-out sample at prediction time; the GP models predict from atmospheric inputs alone.
- Log10 predictions are back-transformed with `10**mu` (implied log-normal median) before comparison in original units.
- Detailed per-scale reports: `linear/CV_REPORT.md`, `log10/CV_REPORT.md`; scale contrast in `log_vs_linear/`.
"""
    savepath.write_text(report)


def scale_predictions(
    scale: str,
    XLF_10k: np.ndarray,
    YLF_10k: np.ndarray,
    XHF: np.ndarray,
    YHF: np.ndarray,
    YLF_paired: np.ndarray,
    wavelengths: np.ndarray,
) -> dict[str, CVPredictions]:
    """Stored baselines + freshly cross-validated Model 2K on one scale."""
    predictions = load_baseline_predictions(scale)
    print(f"  [{scale}] Model 2K CV (stage L reused from production fit) ...", flush=True)
    predictions["model_2k"] = cv_predict_model2_kron(
        XLF_10k, YLF_10k, XHF, YHF, YLF_paired, wavelengths,
        seed=SEED, n_splits=N_SPLITS, n_restarts=N_RESTARTS,
        stage_l=load_production_stage_l(scale), progress=True,
    )
    for key in predictions:
        assert_same_cv_assignment(
            predictions["model_2k"], predictions[key], label=key
        )
    return predictions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    sample_labels = data["YHF"].index.to_numpy()
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF_linear = data["YHF"].to_numpy()
    YLF_paired_linear = data["YLF"].to_numpy()

    print(
        f"{N_SPLITS}-fold CV of Model 2K (exact, {N_RESTARTS} restarts) vs the "
        f"stored baselines of 02_augmented_wavelength, seed={SEED}"
    )
    linear_predictions = scale_predictions(
        "linear", XLF_10k, YLF_10k, XHF, YHF_linear, YLF_paired_linear, wavelengths
    )
    log_predictions = scale_predictions(
        "log10",
        XLF_10k,
        log10_spectra(YLF_10k),
        XHF,
        log10_spectra(YHF_linear),
        log10_spectra(YLF_paired_linear),
        wavelengths,
    )

    linear_comparisons = write_scale_outputs(
        OUTPUT_DIR / "linear",
        wavelengths,
        sample_labels,
        YHF_linear,
        linear_predictions,
        output_units="original eclipse-depth units",
        actual_label="actual HF eclipse depth",
    )
    log_comparisons = write_scale_outputs(
        OUTPUT_DIR / "log10",
        wavelengths,
        sample_labels,
        log10_spectra(YHF_linear),
        log_predictions,
        output_units="log10 eclipse-depth units",
        actual_label="actual HF log10 eclipse depth",
    )

    model2k_log_backtransformed = log_predictions["model_2k"].with_values(
        inverse_log10_spectra(log_predictions["model_2k"].y_pred), y_std=None
    )
    scale_comparison = write_log_vs_linear_outputs(
        OUTPUT_DIR / "log_vs_linear",
        wavelengths,
        sample_labels,
        YHF_linear,
        linear_predictions["model_2k"],
        model2k_log_backtransformed,
    )

    write_combined_report(
        OUTPUT_DIR / "CV_COMPARISON_REPORT.md",
        linear_comparisons=linear_comparisons,
        log_comparisons=log_comparisons,
        scale_comparison=scale_comparison,
    )

    for units, comparisons in (
        ("linear", linear_comparisons),
        ("log10", log_comparisons),
    ):
        deltas = ", ".join(
            f"vs {key}: {comparison.delta_rmse_pooled:+.4e}"
            for key, comparison in comparisons.items()
        )
        print(f"{units} pooled delta RMSE (Model 2K - baseline): {deltas}")
    print(
        "linear-units log-vs-linear delta RMSE (log - linear): "
        f"{scale_comparison.delta_rmse_pooled:+.4e}"
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
