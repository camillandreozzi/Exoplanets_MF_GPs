"""Statistics-101 summary of held-out CV predictions for all models.

This workflow does not refit anything. It reuses the out-of-fold predictions
saved by ``research/modelling/02_augmented_wavelength/02_cv_model2_vs_model1.py``
(one consistent 5-fold CV protocol for Models 1A, 1B, and 2 -- all joint
MF-GPs -- on both linear and log10 spectra) and reports familiar
introductory statistics:

- R^2 (coefficient of determination), Pearson correlation r
- MAE, RMSE, mean error (bias)
- predicted-vs-actual scatter panels and residual histograms/boxplots

Three scale views are reported:

1. linear: linear-scale fits scored in original eclipse-depth units;
2. log10: log10-scale fits scored in log10 units;
3. log10 back-transformed: log10-scale predictions mapped through
   ``10**prediction`` (the implied log-normal median) and scored in original
   eclipse-depth units, matching the existing log-vs-linear convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_YHF
from exoplanets_mf.paths import (
    LOG_MODELLING_RESULTS_DIR,
    MODELLING_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
)
from exoplanets_mf.transforms import inverse_log10_spectra, log10_spectra

CV_DIRS = {
    "linear": MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv" / "linear",
    "log10": (
        LOG_MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv" / "log10"
    ),
}
OUTPUT_DIR = VALIDATION_RESULTS_DIR / "04_stats101_summary"

MODEL_LABELS = {
    "model_1a": "Model 1A (global-rho MF-GP)",
    "model_1b": "Model 1B (per-wavelength-rho MF-GP)",
    "model_2": "Model 2 (wavelength-augmented joint MF-GP)",
}
MODEL_SHORT_LABELS = {
    "model_1a": "Model 1A",
    "model_1b": "Model 1B",
    "model_2": "Model 2",
}
# Categorical palette validated for CVD separation on a white surface; every
# panel and box is also direct-labeled, so identity never rides on color alone.
MODEL_COLORS = {
    "model_1a": "#2a78d6",
    "model_1b": "#1baf7a",
    "model_2": "#008300",
}

LINEAR_VIEW = "linear"
LOG10_VIEW = "log10"
BACKTRANSFORMED_VIEW = "log10 back-transformed"
VIEW_UNITS = {
    LINEAR_VIEW: "original eclipse-depth units",
    LOG10_VIEW: "log10 eclipse-depth units",
    BACKTRANSFORMED_VIEW: "original eclipse-depth units",
}


def fmt(value: float) -> str:
    """Compact numeric formatting for Markdown tables."""
    return f"{value:.6g}"


def load_cv_predictions(scale: str) -> dict[str, np.ndarray]:
    path = CV_DIRS[scale] / "cv_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make modelling_sklearn` first to generate the "
            "out-of-fold CV predictions this summary reuses."
        )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def load_cv_protocol(scale: str) -> dict:
    return json.loads((CV_DIRS[scale] / "cv_summary.json").read_text())


def stats101(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pooled introductory statistics over all held-out points."""
    true = y_true.ravel()
    pred = y_pred.ravel()
    residuals = pred - true
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "n_points": int(true.size),
        "r2": 1.0 - ss_res / ss_tot,
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "bias": float(residuals.mean()),
    }


def assert_matches_saved_rmse(
    computed: dict[str, dict[str, float]],
    saved_pooled_rmse: dict[str, float],
    *,
    view: str,
) -> None:
    """Guard against truth/prediction misalignment by reproducing saved RMSEs."""
    for model_key, saved in saved_pooled_rmse.items():
        recomputed = computed[model_key]["rmse"]
        if not np.isclose(recomputed, saved, rtol=1e-10, atol=0.0):
            raise RuntimeError(
                f"{view} RMSE mismatch for {model_key}: recomputed "
                f"{recomputed!r} vs saved {saved!r} in cv_summary.json"
            )


def write_predictions_csv(
    savepath: Path,
    YHF: pd.DataFrame,
    fold_of_sample: np.ndarray,
    predictions_by_view: dict[str, dict[str, np.ndarray]],
) -> None:
    """Long-format CSV: one row per held-out point, actuals plus every model."""
    n_samples, n_wavelengths = YHF.shape
    frame = pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_samples), n_wavelengths),
            "spectrum": np.repeat(YHF.index.to_numpy(), n_wavelengths),
            "fold": np.repeat(fold_of_sample, n_wavelengths),
            "wavelength": np.tile(
                YHF.columns.to_numpy(dtype=float), n_samples
            ),
            "actual_linear": YHF.to_numpy().ravel(),
            "actual_log10": log10_spectra(YHF.to_numpy()).ravel(),
        }
    )
    view_suffixes = {
        LINEAR_VIEW: "linear",
        LOG10_VIEW: "log10",
        BACKTRANSFORMED_VIEW: "log10_backtransformed",
    }
    for view, suffix in view_suffixes.items():
        for model_key in MODEL_LABELS:
            frame[f"pred_{model_key}_{suffix}"] = predictions_by_view[view][
                model_key
            ].ravel()
    frame.to_csv(savepath, index=False)


def plot_pred_vs_actual(
    Y_linear: np.ndarray,
    predictions_by_view: dict[str, dict[str, np.ndarray]],
    stats_by_view: dict[str, dict[str, dict[str, float]]],
    savepath: Path,
) -> None:
    views = (LINEAR_VIEW, BACKTRANSFORMED_VIEW)
    fig, axes = plt.subplots(
        len(views),
        len(MODEL_LABELS),
        figsize=(16, 8.5),
        sharex=True,
        sharey=True,
    )
    limits = [Y_linear.min(), Y_linear.max()]

    for row, view in enumerate(views):
        for col, model_key in enumerate(MODEL_LABELS):
            ax = axes[row, col]
            ax.scatter(
                Y_linear.ravel(),
                predictions_by_view[view][model_key].ravel(),
                s=2,
                alpha=0.08,
                color=MODEL_COLORS[model_key],
                rasterized=True,
            )
            ax.plot(limits, limits, color="black", lw=0.8, ls="--")
            stats = stats_by_view[view][model_key]
            ax.text(
                0.03,
                0.97,
                f"$R^2$ = {stats['r2']:.3f}\nRMSE = {stats['rmse']:.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
            if row == 0:
                ax.set_title(MODEL_SHORT_LABELS[model_key])
            if row == len(views) - 1:
                ax.set_xlabel("actual HF eclipse depth")

    axes[0, 0].set_ylabel("held-out prediction\n(linear fit)")
    axes[1, 0].set_ylabel("held-out prediction\n(log10 fit, back-transformed)")
    fig.suptitle(
        "Out-of-fold predictions vs actual (original eclipse-depth units)"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    Y_linear: np.ndarray,
    predictions_by_view: dict[str, dict[str, np.ndarray]],
    savepath: Path,
) -> None:
    residuals = {
        view: {
            model_key: (predictions_by_view[view][model_key] - Y_linear).ravel()
            for model_key in MODEL_LABELS
        }
        for view in (LINEAR_VIEW, BACKTRANSFORMED_VIEW)
    }
    all_linear = np.concatenate(list(residuals[LINEAR_VIEW].values()))
    span = float(np.max(np.abs(all_linear)))
    bins = np.linspace(-span, span, 61)

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, len(MODEL_LABELS), height_ratios=[1.0, 1.1])

    hist_axes = []
    for col, model_key in enumerate(MODEL_LABELS):
        ax = fig.add_subplot(grid[0, col], sharey=hist_axes[0] if hist_axes else None)
        hist_axes.append(ax)
        ax.hist(
            residuals[LINEAR_VIEW][model_key],
            bins=bins,
            color=MODEL_COLORS[model_key],
        )
        ax.axvline(0.0, color="black", lw=0.8, ls="--")
        ax.set_title(MODEL_SHORT_LABELS[model_key])
        ax.set_xlabel("residual (prediction - actual)")
        if col == 0:
            ax.set_ylabel("held-out points (linear fit)")

    ax = fig.add_subplot(grid[1, :])
    positions, box_data, box_colors, filled = [], [], [], []
    for index, model_key in enumerate(MODEL_LABELS):
        for offset, view in ((0.0, LINEAR_VIEW), (0.8, BACKTRANSFORMED_VIEW)):
            positions.append(index * 2.4 + offset)
            box_data.append(residuals[view][model_key])
            box_colors.append(MODEL_COLORS[model_key])
            filled.append(view == LINEAR_VIEW)

    boxes = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    for patch, median, color, is_filled in zip(
        boxes["boxes"], boxes["medians"], box_colors, filled
    ):
        patch.set_edgecolor(color)
        patch.set_facecolor(color if is_filled else "white")
        median.set_color("black")

    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xticks([index * 2.4 + 0.4 for index in range(len(MODEL_LABELS))])
    ax.set_xticklabels([MODEL_SHORT_LABELS[key] for key in MODEL_LABELS])
    ax.set_ylabel("residual (original units)")
    ax.set_title("Residual spread per model and fitting scale")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="0.5"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.5"),
    ]
    ax.legend(
        legend_handles,
        ["linear fit", "log10 fit, back-transformed"],
        loc="upper right",
        fontsize=9,
    )

    fig.suptitle("Held-out residuals in original eclipse-depth units")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    savepath: Path,
    summary_table: pd.DataFrame,
    *,
    n_samples: int,
    n_wavelengths: int,
    n_splits: int,
    seed: int,
) -> None:
    view_tables = []
    for view in (LINEAR_VIEW, LOG10_VIEW, BACKTRANSFORMED_VIEW):
        rows = summary_table[summary_table["scale_view"] == view]
        lines = [
            f"### {view.capitalize() if view == LINEAR_VIEW else view} "
            f"({VIEW_UNITS[view]})",
            "",
            "| Model | R^2 | Pearson r | MAE | RMSE | Mean error (bias) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in rows.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["model_label"],
                        f"{row['r2']:.4f}",
                        f"{row['pearson_r']:.4f}",
                        fmt(row["mae"]),
                        fmt(row["rmse"]),
                        fmt(row["bias"]),
                    ]
                )
                + " |"
            )
        view_tables.append("\n".join(lines))

    report = f"""# Statistics-101 Summary: Models 1A, 1B, and 2 (all joint MF-GPs)

## What each statistic means

- **R^2** (coefficient of determination): fraction of the variance in the true
  spectra explained by the predictions; 1 is perfect, 0 means no better than
  always predicting the overall mean.
- **Pearson r**: linear correlation between predicted and actual values;
  1 is a perfect positive linear relationship.
- **MAE** (mean absolute error): average size of the errors, in the units of
  the view; robust to a few large misses.
- **RMSE** (root mean squared error): like MAE but penalizes large errors more;
  identical to the pooled RMSE reported by the CV workflows.
- **Mean error (bias)**: average of prediction minus actual; near zero means
  the model neither systematically over- nor under-predicts.

All statistics are pooled over every held-out point
({n_samples} spectra x {n_wavelengths} wavelength bins).

## Results

{view_tables[0]}

{view_tables[1]}

{view_tables[2]}

## Computation

- Predictions are the out-of-fold CV predictions saved by
  `research/modelling/02_augmented_wavelength/02_cv_model2_vs_model1.py`
  (`cv_predictions.npz`); nothing is refitted here.
- CV protocol: {n_splits}-fold over {n_samples} paired HF samples, seed `{seed}`;
  the same fold assignment is shared by all models and both scales.
- The log10 view scores log10-scale fits against `log10(YHF)` in log10 units,
  so its error magnitudes are not comparable to the other two views.
- The back-transformed view maps log10 predictions through `10**prediction`
  (the median of the implied log-normal distribution, not its mean) and scores
  them in original eclipse-depth units, so all models are comparable in one
  table.
- Recomputed RMSEs are asserted to match `cv_summary.json` from the source CV
  run before anything is written.

## Files

- `stats101_summary.csv`: machine-readable version of the tables above.
- `cv_predictions_long.csv`: one row per held-out point (sample, spectrum,
  fold, wavelength) with the actual value and every model's prediction in all
  three scale views.
- `01_pred_vs_actual.png`: predicted-vs-actual scatter grid with the identity
  line (perfect predictions fall on the dashed diagonal).
- `02_residuals.png`: residual histograms (linear fits) and residual boxplots
  per model and fitting scale, in original units.
"""
    savepath.write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    YHF = load_YHF()
    Y_linear = YHF.to_numpy()
    Y_log = log10_spectra(Y_linear)

    linear_archive = load_cv_predictions("linear")
    log_archive = load_cv_predictions("log10")
    protocol = load_cv_protocol("linear")
    if not np.array_equal(
        linear_archive["fold_of_sample"], log_archive["fold_of_sample"]
    ):
        raise RuntimeError("linear and log10 CV fold assignments do not match")

    predictions_by_view = {
        LINEAR_VIEW: {
            key: linear_archive[f"y_pred_{key}"] for key in MODEL_LABELS
        },
        LOG10_VIEW: {
            key: log_archive[f"y_pred_{key}"] for key in MODEL_LABELS
        },
        BACKTRANSFORMED_VIEW: {
            key: inverse_log10_spectra(log_archive[f"y_pred_{key}"])
            for key in MODEL_LABELS
        },
    }
    truth_by_view = {
        LINEAR_VIEW: Y_linear,
        LOG10_VIEW: Y_log,
        BACKTRANSFORMED_VIEW: Y_linear,
    }

    stats_by_view = {
        view: {
            model_key: stats101(truth_by_view[view], predictions)
            for model_key, predictions in view_predictions.items()
        }
        for view, view_predictions in predictions_by_view.items()
    }
    assert_matches_saved_rmse(
        stats_by_view[LINEAR_VIEW],
        protocol["pooled_rmse"],
        view=LINEAR_VIEW,
    )
    assert_matches_saved_rmse(
        stats_by_view[LOG10_VIEW],
        load_cv_protocol("log10")["pooled_rmse"],
        view=LOG10_VIEW,
    )

    rows = []
    for view in (LINEAR_VIEW, LOG10_VIEW, BACKTRANSFORMED_VIEW):
        for model_key in MODEL_LABELS:
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "scale_view": view,
                    "units": VIEW_UNITS[view],
                    **stats_by_view[view][model_key],
                }
            )
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(OUTPUT_DIR / "stats101_summary.csv", index=False)

    write_predictions_csv(
        OUTPUT_DIR / "cv_predictions_long.csv",
        YHF,
        linear_archive["fold_of_sample"],
        predictions_by_view,
    )

    plot_pred_vs_actual(
        Y_linear,
        predictions_by_view,
        stats_by_view,
        OUTPUT_DIR / "01_pred_vs_actual.png",
    )
    plot_residuals(
        Y_linear,
        predictions_by_view,
        OUTPUT_DIR / "02_residuals.png",
    )
    write_report(
        OUTPUT_DIR / "STATS101_REPORT.md",
        summary_table,
        n_samples=Y_linear.shape[0],
        n_wavelengths=Y_linear.shape[1],
        n_splits=protocol["n_splits"],
        seed=protocol["seed"],
    )

    printable = summary_table[
        ["model_label", "scale_view", "r2", "pearson_r", "mae", "rmse", "bias"]
    ]
    with pd.option_context("display.width", 160, "display.float_format", fmt):
        print(printable.to_string(index=False))
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
