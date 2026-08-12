"""Plot GPBoost full-fit diagnostics and CV validation figures from saved outputs.

This script is intentionally read-only with respect to fitted models: it loads
the CSV/NPZ/JSON artifacts produced by ``02_fit_all_gpboost.py`` and
``01_compare_gpboost.py`` and regenerates figure files without refitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import CVPredictions, cv_metrics
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.paths import MODELLING_RESULTS_DIR

RESULTS_DIR = MODELLING_RESULTS_DIR / "03_gpboost_comparison"

MODEL_LABELS = {
    "hf_only": "HF-only GPBoost",
    "model_1a_gpboost": "Model 1A GPBoost",
    "model_1b_gpboost": "Model 1B GPBoost",
    "model_2_gpboost": "Model 2 GPBoost",
}
MODEL_COLORS = {
    "hf_only": "tab:gray",
    "model_1a_gpboost": "tab:green",
    "model_1b_gpboost": "tab:blue",
    "model_2_gpboost": "tab:red",
}
RUNTIME_LABELS = {
    "hf_only_gpboost": "HF-only GPBoost",
    "hf_only": "HF-only GPBoost",
    "model_1a_gpboost": "Model 1A GPBoost",
    "model_1b_gpboost": "Model 1B GPBoost",
    "model_2_gpboost": "Model 2 GPBoost",
}
RUNTIME_COLORS = {
    "hf_only_gpboost": MODEL_COLORS["hf_only"],
    "hf_only": MODEL_COLORS["hf_only"],
    "model_1a_gpboost": MODEL_COLORS["model_1a_gpboost"],
    "model_1b_gpboost": MODEL_COLORS["model_1b_gpboost"],
    "model_2_gpboost": MODEL_COLORS["model_2_gpboost"],
}


def shade_instrument_modes(ax, wavelengths: np.ndarray) -> None:
    for mode, mask in instrument_mode_masks(wavelengths):
        span = wavelengths[mask]
        ax.axvspan(span.min(), span.max(), color=mode.color, alpha=0.25, zorder=0)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_full_fit_dirs() -> list[Path]:
    return sorted(
        path for path in RESULTS_DIR.glob("full_fit*") if (path / "fit_summary.json").exists()
    )


def find_cv_dirs() -> list[Path]:
    return sorted(
        path for path in RESULTS_DIR.iterdir() if (path / "cv_predictions.npz").exists()
    )


def load_full_fit_tables(full_fit_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "hf_only": full_fit_dir / "hf_only_gpboost_hyperparameters.csv",
        "model_1a_gpboost": full_fit_dir / "model_1a_gpboost_hyperparameters.csv",
        "model_1b_gpboost": full_fit_dir / "model_1b_gpboost_hyperparameters.csv",
        "model_2_gpboost": full_fit_dir / "model_2_gpboost_hyperparameters.csv",
    }
    return {key: pd.read_csv(path) for key, path in paths.items() if path.exists()}


def plot_full_fit_diagnostics(full_fit_dir: Path) -> None:
    tables = load_full_fit_tables(full_fit_dir)
    if not tables:
        print(f"skip {full_fit_dir}: no hyperparameter tables")
        return
    summary = load_json(full_fit_dir / "fit_summary.json")
    wavelengths = tables.get("model_1b_gpboost", tables["hf_only"])["wavelength"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths.to_numpy())
    if "model_1a_gpboost" in tables and "rho" in tables["model_1a_gpboost"]:
        ax.plot(
            wavelengths,
            tables["model_1a_gpboost"]["rho"],
            color=MODEL_COLORS["model_1a_gpboost"],
            lw=1.2,
            label=MODEL_LABELS["model_1a_gpboost"],
        )
    if "model_1b_gpboost" in tables and "rho" in tables["model_1b_gpboost"]:
        ax.plot(
            wavelengths,
            tables["model_1b_gpboost"]["rho"],
            color=MODEL_COLORS["model_1b_gpboost"],
            lw=1.0,
            label=MODEL_LABELS["model_1b_gpboost"],
        )
    if "model_2_rho" in summary:
        ax.axhline(
            summary["model_2_rho"],
            color=MODEL_COLORS["model_2_gpboost"],
            ls="--",
            lw=1.2,
            label=MODEL_LABELS["model_2_gpboost"],
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel="wavelength [um]", ylabel="rho", title="Fitted AR(1) scaling")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths.to_numpy())
    for key, column in (
        ("hf_only", "signal_variance"),
        ("model_1a_gpboost", "low_signal_variance"),
        ("model_1b_gpboost", "low_signal_variance"),
    ):
        if key in tables and column in tables[key]:
            ax.plot(
                tables[key]["wavelength"],
                tables[key][column],
                color=MODEL_COLORS[key],
                lw=1.0,
                label=MODEL_LABELS[key],
            )
    if "model_2_gpboost" in tables and "low_signal_variance" in tables["model_2_gpboost"]:
        ax.axhline(
            float(tables["model_2_gpboost"]["low_signal_variance"].iloc[0]),
            color=MODEL_COLORS["model_2_gpboost"],
            ls="--",
            lw=1.2,
            label=MODEL_LABELS["model_2_gpboost"],
        )
    ax.set_yscale("log")
    ax.set(
        xlabel="wavelength [um]",
        ylabel="variance",
        title="LF / single-fidelity signal variance",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    shade_instrument_modes(ax, wavelengths.to_numpy())
    for key in ("model_1a_gpboost", "model_1b_gpboost"):
        if key in tables and "delta_signal_variance" in tables[key]:
            ax.plot(
                tables[key]["wavelength"],
                tables[key]["delta_signal_variance"],
                color=MODEL_COLORS[key],
                lw=1.0,
                label=MODEL_LABELS[key],
            )
    if (
        "model_2_gpboost" in tables
        and "delta_signal_variance" in tables["model_2_gpboost"]
    ):
        ax.axhline(
            float(tables["model_2_gpboost"]["delta_signal_variance"].iloc[0]),
            color=MODEL_COLORS["model_2_gpboost"],
            ls="--",
            lw=1.2,
            label=MODEL_LABELS["model_2_gpboost"],
        )
    ax.set_yscale("log")
    ax.set(
        xlabel="wavelength [um]",
        ylabel="variance",
        title="Discrepancy signal variance",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    runtime = {
        key: value
        for key, value in summary.get("runtime_seconds", {}).items()
        if key != "total"
    }
    labels = [RUNTIME_LABELS.get(key, key) for key in runtime]
    values = np.array(list(runtime.values()), dtype=float) / 60.0
    ax.bar(
        np.arange(len(values)),
        values,
        color=[RUNTIME_COLORS.get(key, "0.5") for key in runtime],
    )
    ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    ax.set(ylabel="minutes", title="Full-fit wall time")

    fig.suptitle(f"GPBoost Full-Fit Diagnostics: {full_fit_dir.name}")
    fig.tight_layout()
    savepath = full_fit_dir / "01_full_fit_diagnostics.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_range_summary(full_fit_dir, tables)
    print(f"wrote {savepath}")


def range_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )


def summarize_ranges(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if len(frame) == 1:
        return frame[columns].iloc[0].to_numpy(dtype=float)
    return frame[columns].median(axis=0).to_numpy(dtype=float)


def plot_range_summary(full_fit_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    series = []
    for key, label, prefix in (
        ("hf_only", "HF-only", "range_"),
        ("model_1a_gpboost", "1A LF", "low_range_"),
        ("model_1a_gpboost", "1A discrepancy", "delta_range_"),
        ("model_1b_gpboost", "1B LF", "low_range_"),
        ("model_1b_gpboost", "1B discrepancy", "delta_range_"),
        ("model_2_gpboost", "2 LF", "low_range_"),
        ("model_2_gpboost", "2 discrepancy", "delta_range_"),
    ):
        if key not in tables:
            continue
        columns = range_columns(tables[key], prefix)
        if not columns:
            continue
        values = summarize_ranges(tables[key], columns)
        series.append((label, values))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    max_len = max(len(values) for _, values in series)
    width = min(0.12, 0.8 / max(len(series), 1))
    offset = -0.5 * width * (len(series) - 1)
    for i, (label, values) in enumerate(series):
        positions = np.arange(len(values))
        ax.bar(positions + offset + i * width, values, width=width, label=label)
    positions = np.arange(max_len)
    ax.set_xticks(positions, [f"x{i}" for i in positions])
    ax.set_yscale("log")
    ax.set(
        xlabel="input dimension",
        ylabel="GPBoost range",
        title="Median fitted ranges by input dimension",
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    savepath = full_fit_dir / "02_full_fit_range_summary.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def cv_predictions_from_npz(path: Path) -> dict[str, CVPredictions]:
    archive = np.load(path)
    fold = archive["fold_of_sample"]
    predictions = {}
    for name in archive.files:
        if not name.startswith("y_pred_"):
            continue
        key = name.removeprefix("y_pred_")
        std_name = f"y_std_{key}"
        predictions[key] = CVPredictions(
            y_pred=archive[name],
            y_std=archive[std_name] if std_name in archive.files else None,
            fold_of_sample=fold,
            n_splits=len(np.unique(fold)),
        )
    return predictions


def plot_cv_validation(cv_dir: Path, Y_hf: np.ndarray, wavelengths: np.ndarray) -> None:
    predictions = cv_predictions_from_npz(cv_dir / "cv_predictions.npz")
    if not predictions:
        print(f"skip {cv_dir}: no CV predictions")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    for key, prediction in predictions.items():
        metrics = cv_metrics(Y_hf, prediction)
        ax.plot(
            wavelengths,
            metrics.rmse_per_wavelength,
            color=MODEL_COLORS.get(key, "0.5"),
            lw=1.2,
            label=MODEL_LABELS.get(key, key),
        )
    ax.set_yscale("log")
    ax.set(xlabel="wavelength [um]", ylabel="held-out RMSE", title="CV RMSE")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for key, prediction in predictions.items():
        metrics = cv_metrics(Y_hf, prediction)
        ax.bar(
            MODEL_LABELS.get(key, key),
            metrics.rmse_pooled,
            color=MODEL_COLORS.get(key, "0.5"),
        )
    ax.tick_params(axis="x", rotation=30)
    ax.set(ylabel="pooled RMSE", title="Pooled held-out error")

    ax = axes[1, 0]
    for key, prediction in predictions.items():
        residual = (prediction.y_pred - Y_hf).ravel()
        ax.hist(
            residual,
            bins=60,
            histtype="step",
            color=MODEL_COLORS.get(key, "0.5"),
            label=MODEL_LABELS.get(key, key),
        )
    ax.set(xlabel="prediction - actual", ylabel="count", title="Residuals")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sample = min(80, Y_hf.shape[0] - 1)
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, Y_hf[sample], color="black", lw=1.2, label="actual")
    for key, prediction in predictions.items():
        ax.plot(
            wavelengths,
            prediction.y_pred[sample],
            color=MODEL_COLORS.get(key, "0.5"),
            lw=1.0,
            label=MODEL_LABELS.get(key, key),
        )
    ax.set(
        xlabel="wavelength [um]",
        ylabel="eclipse depth",
        title=f"Held-out spectrum {sample + 1}",
    )
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"GPBoost CV Validation: {cv_dir.name}")
    fig.tight_layout()
    savepath = cv_dir / "02_cv_validation_diagnostics.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {savepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-fit-dir",
        action="append",
        type=Path,
        help="Full-fit output directory to plot. Defaults to every full_fit* directory.",
    )
    parser.add_argument(
        "--cv-dir",
        action="append",
        type=Path,
        help="CV output directory to plot. Defaults to every directory with cv_predictions.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_fit_dirs = args.full_fit_dir or find_full_fit_dirs()
    cv_dirs = args.cv_dir or find_cv_dirs()
    data = load_all()
    Y_hf = data["YHF"].to_numpy()
    wavelengths = np.asarray(data["wavelengths"], dtype=float)

    for full_fit_dir in full_fit_dirs:
        plot_full_fit_diagnostics(full_fit_dir)
    for cv_dir in cv_dirs:
        plot_cv_validation(cv_dir, Y_hf, wavelengths)


if __name__ == "__main__":
    main()
