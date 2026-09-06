"""Plot 5-fold CV results saved by modelling/cv/cv_5fold.py."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "cv" / "5fold"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
CV_LABEL = "5fold"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nrmse = read_csv(f"{CV_LABEL}_nrmse_per_wavelength.csv")
    model1 = read_csv(f"model1_{CV_LABEL}_sf_vs_mf_nrmse_per_wavelength.csv")
    model2 = read_csv(f"model2_{CV_LABEL}_sf_vs_mf_nrmse_per_wavelength.csv")
    summary = read_csv(f"{CV_LABEL}_summary.csv")
    predictions = read_csv(f"{CV_LABEL}_predictions.csv")

    plot_nrmse_curves(nrmse)
    plot_sf_mf_delta(model1, model2)
    plot_summary(summary)
    plot_spectrum_examples(predictions)

    print(f"Saved CV plots to {RESULTS_DIR}")


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing CV result file: {path}")
    return pd.read_csv(path)


def plot_nrmse_curves(nrmse):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

    plot_model_nrmse(
        axes[0],
        nrmse,
        model="model1",
        title="Model 1 wavelength-wise 5-fold CV",
        colors=("tab:blue", "tab:cyan"),
    )
    plot_model_nrmse(
        axes[1],
        nrmse,
        model="model2",
        title="Model 2 augmented-wavelength 5-fold CV",
        colors=("tab:orange", "tab:red"),
    )

    axes[1].set_xlabel("Wavelength")
    fig.savefig(RESULTS_DIR / "cv_nrmse_per_wavelength.png", dpi=200)
    plt.close(fig)


def plot_model_nrmse(ax, nrmse, model, title, colors):
    ax.plot(
        nrmse["wavelength"],
        nrmse[f"{model}_sf_nrmse"],
        label="SF",
        color=colors[0],
        linewidth=1.6,
    )
    ax.plot(
        nrmse["wavelength"],
        nrmse[f"{model}_mf_nrmse"],
        label="MF",
        color=colors[1],
        linewidth=1.6,
    )
    ax.set_title(title)
    ax.set_ylabel("NRMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def plot_sf_mf_delta(model1, model2):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    plot_delta_panel(axes[0], model1, "Model 1 MF - SF NRMSE", "tab:blue")
    plot_delta_panel(axes[1], model2, "Model 2 MF - SF NRMSE", "tab:orange")

    axes[1].set_xlabel("Wavelength")
    fig.savefig(RESULTS_DIR / "cv_mf_minus_sf_delta_per_wavelength.png", dpi=200)
    plt.close(fig)


def plot_delta_panel(ax, data, title, color):
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.plot(
        data["wavelength"],
        data["delta_mf_minus_sf"],
        color=color,
        linewidth=1.5,
    )
    ax.fill_between(
        data["wavelength"],
        data["delta_mf_minus_sf"],
        0,
        where=data["delta_mf_minus_sf"] < 0,
        color="tab:green",
        alpha=0.18,
        interpolate=True,
    )
    ax.fill_between(
        data["wavelength"],
        data["delta_mf_minus_sf"],
        0,
        where=data["delta_mf_minus_sf"] > 0,
        color="tab:red",
        alpha=0.14,
        interpolate=True,
    )
    ax.set_title(title)
    ax.set_ylabel("Delta NRMSE")
    ax.grid(alpha=0.25)


def plot_summary(summary):
    ordered = (
        summary
        .assign(label=summary["model"] + " " + summary["variant"].str.upper())
        .sort_values(["model", "variant"])
    )
    x = np.arange(len(ordered))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(
        x - width / 2,
        ordered["mean_nrmse"],
        width,
        label="Mean NRMSE",
        color="tab:purple",
        alpha=0.78,
    )
    ax.bar(
        x + width / 2,
        ordered["median_nrmse"],
        width,
        label="Median NRMSE",
        color="tab:gray",
        alpha=0.78,
    )
    ax.set_xticks(x, ordered["label"])
    ax.set_ylabel("NRMSE")
    ax.set_title("5-fold CV summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(RESULTS_DIR / "cv_summary_nrmse.png", dpi=200)
    plt.close(fig)


def plot_spectrum_examples(predictions):
    examples = select_example_cases(predictions)
    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(11, 3.4 * len(examples)),
        sharex=True,
        constrained_layout=True,
    )
    if len(examples) == 1:
        axes = [axes]

    for ax, (fold, source_index, label) in zip(axes, examples):
        fold_data = predictions[
            predictions["fold"].eq(fold)
            & predictions["held_out_source_index"].eq(source_index)
        ]
        actual = (
            fold_data[fold_data["model"].eq("model1")]
            .drop_duplicates("response_index")
            .sort_values("response_index")
        )
        ax.plot(
            actual["wavelength"],
            actual["y_true"],
            color="black",
            linewidth=1.8,
            label="Actual",
        )

        for model, variant, color, linestyle in [
            ("model1", "sf", "tab:blue", "-"),
            ("model1", "mf", "tab:cyan", "--"),
            ("model2", "sf", "tab:orange", "-"),
            ("model2", "mf", "tab:red", "--"),
        ]:
            series = fold_data[
                fold_data["model"].eq(model) & fold_data["variant"].eq(variant)
            ].sort_values("response_index")
            ax.plot(
                series["wavelength"],
                series["y_pred"],
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                label=f"{model} {variant.upper()}",
            )

        ax.set_title(f"{label} held-out spectrum: fold {fold}, source {source_index}")
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3)

    axes[-1].set_xlabel("Wavelength")
    fig.savefig(RESULTS_DIR / "cv_example_spectra.png", dpi=200)
    plt.close(fig)


def select_example_cases(predictions):
    errors = (
        predictions
        .assign(abs_residual=lambda df: df["residual"].abs())
        .groupby(["fold", "held_out_source_index"])["abs_residual"]
        .mean()
        .sort_values()
    )

    best_fold, best_source = errors.index[0]
    median_fold, median_source = errors.index[len(errors) // 2]
    worst_fold, worst_source = errors.index[-1]

    examples = [
        (int(best_fold), int(best_source), "Best average"),
        (int(median_fold), int(median_source), "Median average"),
        (int(worst_fold), int(worst_source), "Worst average"),
    ]
    deduped = []
    seen = set()
    for fold, source_index, label in examples:
        key = (fold, source_index)
        if key not in seen:
            deduped.append((fold, source_index, label))
            seen.add(key)
    return deduped


if __name__ == "__main__":
    main()
