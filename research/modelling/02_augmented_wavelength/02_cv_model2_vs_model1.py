"""5-fold CV of Model 2 against the Model 1 baselines, linear and log10.

Every comparison is paired: all models share the identical KFold assignment
(same n_samples, n_splits, seed), verified before any metric is computed.
Whole HF spectra are held out per fold; LF data is always fully eligible --
only HF availability is validated (same protocol as validation/02_full_cv).

Baselines per output scale:

- Model 1A / 1B: closed-form rho layers (instant refits).
- Model 1 joint GP (optional): the per-wavelength joint AR(1) MF-GP at its
  200-LF TESTING configuration -- the same caveat as validation/02_full_cv:
  this is NOT Model 1's production size, so read its rows as indicative.

Model 2 uses a reduced CV configuration relative to 01_fit_model2.py because
every fold refits the joint GP from scratch (5 folds x 2 scales); see
benchmark/model2_fit_benchmark.json (recommended_cv_lf_sample_size).

A final section compares Model 2 fitted on log10 spectra against Model 2
fitted on linear spectra, after back-transforming the log10 held-out
predictions with 10**mu (the implied log-normal median), so both are scored
in original eclipse-depth units.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    cv_predict,
    cv_predict_rho_model,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import fit_joint_mf_gp, predict_hf
from exoplanets_mf.model2 import cv_predict_model2
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import inverse_log10_spectra, log10_spectra

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv"

SEED = RANDOM_SEED
N_SPLITS = CV_FULL_MODEL_SPLITS  # each fold refits the full joint GP
# CV runs at a coarser wavelength stride than the production fit because
# every fold refits the joint GP from scratch (5 folds x 2 scales): stride
# 13 keeps 15 HF + 15 offset LF wavelengths -> n = 97*15 + 40*15 = 2055
# points, ~6-7 min per fold (~65 min for both scales) by cubic
# extrapolation from the benchmark anchor (n=2955 ~ 19 min). At the
# production stride 8 the benchmark instead recommends lf_sample_size 40
# (n=3385, ~28 min per fold, ~4.7 h total) -- see
# benchmark/model2_fit_benchmark.json (recommended_cv_lf_sample_size).
LAMBDA_STRIDE = 13
CV_LF_SAMPLE_SIZE = 40
# The joint per-wavelength Model 1 baseline refits 195 GPs per fold; 200 LF
# rows is its TESTING config (~2 s/wavelength, ~30-40 min for 5 folds per
# scale) -- identical to validation/02_full_cv's caveat.
INCLUDE_MODEL1_JOINT = True
MODEL1_JOINT_SUBSAMPLE_SIZE = 200

CI_Z = stats.norm.ppf(0.975)
MODEL_LABELS = {
    "model_1a": "Model 1A (global rho)",
    "model_1b": "Model 1B (per-wavelength rho)",
    "model_1_joint": f"Model 1 joint GP ({MODEL1_JOINT_SUBSAMPLE_SIZE}-LF testing config)",
    "model_2": "Model 2 (wavelength-augmented joint MF-GP)",
}


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


def scale_predictions(
    XLF_10k: np.ndarray,
    YLF_10k: np.ndarray,
    XHF: np.ndarray,
    YHF: np.ndarray,
    YLF_paired: np.ndarray,
    wavelengths: np.ndarray,
    *,
    scale_label: str,
) -> dict[str, CVPredictions]:
    """All models' out-of-fold predictions on one output scale."""
    predictions = {
        "model_1a": cv_predict_rho_model(
            YHF, YLF_paired, per_wavelength=False, n_splits=N_SPLITS, seed=SEED
        ),
        "model_1b": cv_predict_rho_model(
            YHF, YLF_paired, per_wavelength=True, n_splits=N_SPLITS, seed=SEED
        ),
    }

    if INCLUDE_MODEL1_JOINT:
        print(f"  [{scale_label}] Model 1 joint GP CV ...", flush=True)
        predictions["model_1_joint"] = cv_predict(
            fit_fn=lambda train_idx: fit_joint_mf_gp(
                XLF_10k,
                YLF_10k,
                XHF[train_idx],
                YHF[train_idx],
                wavelengths,
                seed=SEED,
                subsample_size=MODEL1_JOINT_SUBSAMPLE_SIZE,
                progress_every=100,
            ),
            predict_fn=lambda layer, test_idx: predict_hf(layer, XHF[test_idx]),
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=N_SPLITS,
            seed=SEED,
        )

    print(f"  [{scale_label}] Model 2 CV ...", flush=True)
    predictions["model_2"] = cv_predict_model2(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        n_splits=N_SPLITS,
        lf_sample_size=CV_LF_SAMPLE_SIZE,
        lambda_stride=LAMBDA_STRIDE,
        progress=True,
    )

    for key in predictions:
        assert_same_cv_assignment(
            predictions["model_1a"], predictions[key], label=key
        )
    return predictions


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


def plot_scale_diagnostics(
    wavelengths: np.ndarray,
    YHF: np.ndarray,
    predictions: dict[str, CVPredictions],
    comparison_vs_1b: CVComparison,
    *,
    actual_label: str,
    savepath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_lines = [
        ("model_1b", "tab:blue"),
        ("model_2", "tab:red"),
    ]
    if "model_1_joint" in predictions:
        metric_lines.insert(1, ("model_1_joint", "tab:green"))

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    for key, color in metric_lines:
        metrics = cv_metrics(YHF, predictions[key])
        ax.plot(
            wavelengths,
            metrics.rmse_per_wavelength,
            lw=1.2,
            color=color,
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
            wavelengths,
            np.mean(np.abs(z) <= CI_Z, axis=0),
            lw=1.2,
            color=color,
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
    order = np.argsort(comparison_vs_1b.delta_rmse_per_sample)
    colors = np.where(
        comparison_vs_1b.delta_rmse_per_sample[order] < 0, "tab:red", "tab:blue"
    )
    ax.bar(
        np.arange(len(order)),
        comparison_vs_1b.delta_rmse_per_sample[order],
        color=colors,
        width=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(
        xlabel="sample (sorted)",
        ylabel="RMSE(Model 2) - RMSE(Model 1B)",
        title=(
            "Per-sample paired delta (Model 2 wins on "
            f"{comparison_vs_1b.fraction_samples_model2_wins:.0%})"
        ),
    )

    ax = axes[1, 1]
    ax.scatter(
        YHF.ravel(),
        predictions["model_1b"].y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:blue",
        label=MODEL_LABELS["model_1b"],
    )
    ax.scatter(
        YHF.ravel(),
        predictions["model_2"].y_pred.ravel(),
        s=2,
        alpha=0.1,
        color="tab:red",
        label=MODEL_LABELS["model_2"],
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
    baselines = [key for key in MODEL_LABELS if key in predictions and key != "model_2"]
    comparisons = {
        key: compare_cv(
            YHF,
            predictions[key],
            predictions["model_2"],
            labels=(MODEL_LABELS[key], MODEL_LABELS["model_2"]),
        )
        for key in baselines
    }
    all_metrics = {key: cv_metrics(YHF, value) for key, value in predictions.items()}

    per_wavelength = {"wavelength": wavelengths}
    per_sample = {
        "sample": np.arange(YHF.shape[0]),
        "spectrum": sample_labels,
        "fold": predictions["model_2"].fold_of_sample,
    }
    for key, metrics in all_metrics.items():
        per_wavelength[f"rmse_{key}"] = metrics.rmse_per_wavelength
        per_sample[f"rmse_{key}"] = metrics.rmse_per_sample
    for key in baselines:
        per_wavelength[f"delta_rmse_model2_minus_{key}"] = comparisons[
            key
        ].delta_rmse_per_wavelength
        per_sample[f"delta_rmse_model2_minus_{key}"] = comparisons[
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
        "model_2_cv_lf_sample_size": CV_LF_SAMPLE_SIZE,
        "model_2_lambda_stride": LAMBDA_STRIDE,
        "pooled_rmse": {
            key: all_metrics[key].rmse_pooled for key in all_metrics
        },
        "coverage_95": {
            key: all_metrics[key].coverage_95
            for key in all_metrics
            if all_metrics[key].coverage_95 is not None
        },
        "comparisons_vs_model_2": {
            key: comparison_summary(comparisons[key]) for key in comparisons
        },
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    np.savez(
        output_dir / "cv_predictions.npz",
        fold_of_sample=predictions["model_2"].fold_of_sample,
        **{f"y_pred_{key}": value.y_pred for key, value in predictions.items()},
        **{
            f"y_std_{key}": value.y_std
            for key, value in predictions.items()
            if value.y_std is not None
        },
    )

    plot_scale_diagnostics(
        wavelengths,
        YHF,
        predictions,
        comparisons["model_1b"],
        actual_label=actual_label,
        savepath=output_dir / "02_model2_vs_model1.png",
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
            MODEL_LABELS["model_2"]
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
                    fmt(comparison.delta_mean),
                    winner,
                ]
            )
            + " |"
        )
    coverage = summary["coverage_95"]
    coverage_lines = "\n".join(
        f"- {MODEL_LABELS[key]}: {pct(value)} (nominal 95%)"
        for key, value in coverage.items()
    )
    caveat = (
        f"- The Model 1 joint GP baseline runs at its {MODEL1_JOINT_SUBSAMPLE_SIZE}-LF "
        "TESTING configuration (validation/02_full_cv's caveat applies): its row is "
        "indicative, not Model 1's production accuracy.\n"
        if "model_1_joint" in comparisons
        else ""
    )
    report = f"""# CV Report: Model 2 vs Model 1 baselines ({output_units})

## Result

Each row is a paired comparison of held-out errors on identical folds; the
delta is Model 2 minus the baseline, so negative deltas mean Model 2 wins.

| Baseline | Baseline RMSE | Model 2 RMSE | Delta | Model 2 wins by sample | Mean sample delta | Winner |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

95% CI coverage of the models with predictive uncertainties:

{coverage_lines}

## Computation

- Output space: {output_units}.
- Samples: {summary["n_samples"]} paired HF spectra; whole spectra held out per fold.
- CV protocol: {summary["n_splits"]}-fold, seed `{summary["seed"]}`; identical KFold assignment across every model (verified).
- Model 2 CV configuration: {summary["model_2_cv_lf_sample_size"]} LF samples, wavelength stride {summary["model_2_lambda_stride"]} -- reduced relative to the production fit (01_fit_model2.py) because every fold refits the joint GP; see benchmark/model2_fit_benchmark.json.
- Model 2 predicts all 195 wavelengths, including the ~7/8 never seen in training, via the fitted lambda kernel.
{caveat}
## Files

- `cv_per_sample.csv`, `cv_per_wavelength.csv`: per-model RMSE and paired deltas.
- `cv_summary.json`: machine-readable version of this summary.
- `cv_predictions.npz`: out-of-fold predictions (and stds where available) of every model.
- `02_model2_vs_model1.png`: diagnostic figure.
"""
    savepath.write_text(report)


def write_log_vs_linear_outputs(
    output_dir: Path,
    wavelengths: np.ndarray,
    sample_labels: np.ndarray,
    YHF_linear: np.ndarray,
    model2_linear: CVPredictions,
    model2_log_backtransformed: CVPredictions,
) -> CVComparison:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_cv(
        YHF_linear,
        model2_linear,
        model2_log_backtransformed,
        labels=("Model 2 (linear)", "Model 2 (log10 back-transformed)"),
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
            "fold": model2_linear.fold_of_sample,
            "rmse_linear": comparison.metrics_1.rmse_per_sample,
            "rmse_log_backtransformed": comparison.metrics_2.rmse_per_sample,
            "delta_rmse_log_minus_linear": comparison.delta_rmse_per_sample,
        }
    ).to_csv(output_dir / "cv_per_sample.csv", index=False)
    (output_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "comparison": "Model 2 linear vs Model 2 log10 back-transformed",
                "comparison_units": "original eclipse-depth units",
                "log_prediction_point_summary": (
                    "10**mu_log10, the median of the implied linear-scale "
                    "distribution"
                ),
                "n_samples": int(YHF_linear.shape[0]),
                "n_splits": N_SPLITS,
                "seed": SEED,
                **comparison_summary(comparison),
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
                "Model 2" if comparison.delta_rmse_pooled < 0 else MODEL_LABELS[key]
            )
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"Model 2 vs {MODEL_LABELS[key]}",
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
                "Model 2 log10 (back-transformed) vs Model 2 linear",
                "original eclipse-depth",
                fmt(scale_comparison.metrics_1.rmse_pooled),
                fmt(scale_comparison.metrics_2.rmse_pooled),
                fmt(scale_comparison.delta_rmse_pooled),
                "log10" if scale_comparison.delta_rmse_pooled < 0 else "linear",
            ]
        )
        + " |"
    )

    report = f"""# CV Comparison Report: Model 2 (wavelength-augmented joint MF-GP)

## Scope

{N_SPLITS}-fold cross-validation over all 97 paired HF samples, seed `{SEED}`,
identical fold assignment for every model. Model 2 fits ONE joint MF-GP over
z = (theta, lambda) with a single scalar rho; Model 1 fits 195 independent
per-wavelength models.

## Main Results

| Comparison | Units | Baseline RMSE | Model 2 RMSE | Delta (2 - baseline) | Winner |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

Negative deltas mean Model 2 (or the log10 variant in the last row) wins.

## Technical Notes

- Model 2 CV configuration: {CV_LF_SAMPLE_SIZE} LF samples, wavelength stride {LAMBDA_STRIDE} (reduced vs the production fit; every fold refits the joint GP from scratch).
- Model 1 joint GP baseline (if present) runs at its {MODEL1_JOINT_SUBSAMPLE_SIZE}-LF TESTING configuration.
- Model 1A/1B closed-form baselines consume the paired LF spectrum of the held-out sample at prediction time; Model 2 and the Model 1 joint GP predict from atmospheric inputs alone. Model 1A/1B therefore have strictly more information per test sample -- keep that in mind when reading the deltas.
- Log10 predictions are back-transformed with `10**mu` (implied log-normal median) before comparison in original units.
- Detailed per-scale reports: `linear/CV_REPORT.md`, `log10/CV_REPORT.md`; scale contrast in `log_vs_linear/`.
"""
    savepath.write_text(report)


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
        f"{N_SPLITS}-fold CV of Model 2 (lf={CV_LF_SAMPLE_SIZE}, "
        f"stride={LAMBDA_STRIDE}) vs Model 1 baselines "
        f"(joint baseline: {INCLUDE_MODEL1_JOINT}), seed={SEED}"
    )
    linear_predictions = scale_predictions(
        XLF_10k, YLF_10k, XHF, YHF_linear, YLF_paired_linear, wavelengths,
        scale_label="linear",
    )
    log_predictions = scale_predictions(
        XLF_10k,
        log10_spectra(YLF_10k),
        XHF,
        log10_spectra(YHF_linear),
        log10_spectra(YLF_paired_linear),
        wavelengths,
        scale_label="log10",
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

    model2_log_backtransformed = log_predictions["model_2"].with_values(
        inverse_log10_spectra(log_predictions["model_2"].y_pred), y_std=None
    )
    scale_comparison = write_log_vs_linear_outputs(
        OUTPUT_DIR / "log_vs_linear",
        wavelengths,
        sample_labels,
        YHF_linear,
        linear_predictions["model_2"],
        model2_log_backtransformed,
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
        print(f"{units} pooled delta RMSE (Model 2 - baseline): {deltas}")
    print(
        "linear-units log-vs-linear delta RMSE (log - linear): "
        f"{scale_comparison.delta_rmse_pooled:+.4e}"
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
