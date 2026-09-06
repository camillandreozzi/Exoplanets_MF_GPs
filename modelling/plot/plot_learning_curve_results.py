"""Plot checkpointed learning-curve CV results."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "cv_learning_curve" / "5-fold_1_SF_vs_MF"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_load import RESPONSE_DIM


PREDICTIONS_FILE = RESULTS_DIR / "learning_curve_predictions.csv"
SUMMARY_FILE = RESULTS_DIR / "learning_curve_summary.csv"
OUTPUT_FILE = RESULTS_DIR / "learning_curve_mean_nrmse_vs_hf.png"


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"No predictions found at {PREDICTIONS_FILE}")

    complete_summary = load_complete_summary()
    checkpoint_summary = summarize_checkpointed_predictions()
    plot_learning_curve(complete_summary, checkpoint_summary)
    print(f"Saved plot to {OUTPUT_FILE}")


def load_complete_summary():
    if not SUMMARY_FILE.exists():
        return pd.DataFrame()

    summary = pd.read_csv(SUMMARY_FILE)
    if summary.empty:
        return summary

    return summary[summary["n_wavelengths"].eq(RESPONSE_DIM)].copy()


def summarize_checkpointed_predictions():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions = predictions.drop_duplicates(
        subset=[
            "n_hf_train",
            "fold",
            "held_out_source_index",
            "model",
            "variant",
            "response_index",
        ],
        keep="last",
    )

    metrics = (
        predictions
        .groupby(["n_hf_train", "model", "variant", "response_index", "wavelength"])
        .agg(
            n_cv=("y_true", "size"),
            y_true_min=("y_true", "min"),
            y_true_max=("y_true", "max"),
            rmse=("residual", rmse),
        )
        .reset_index()
    )
    y_range = metrics["y_true_max"] - metrics["y_true_min"]
    metrics["nrmse"] = metrics["rmse"] / y_range.replace(0, np.nan)

    summary = (
        metrics
        .groupby(["n_hf_train", "model", "variant"])
        .agg(
            n_wavelengths=("response_index", "nunique"),
            completed_validation_rows=("n_cv", "max"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
        )
        .reset_index()
    )
    summary["is_full_wavelength_set"] = summary["n_wavelengths"].eq(RESPONSE_DIM)
    return summary


def plot_learning_curve(complete_summary, checkpoint_summary):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    colors = {
        "sf": "tab:blue",
        "mf": "tab:orange",
    }
    labels = {
        "sf": "SF",
        "mf": "MF",
    }

    plotted_anything = False
    for variant in ["sf", "mf"]:
        complete = complete_summary[
            complete_summary["variant"].eq(variant)
        ].sort_values("n_hf_train")

        if not complete.empty:
            ax.plot(
                complete["n_hf_train"],
                complete["mean_nrmse"],
                marker="o",
                linewidth=2,
                color=colors.get(variant),
                label=f"{labels.get(variant, variant.upper())} complete",
            )
            plotted_anything = True

        partial = checkpoint_summary[
            checkpoint_summary["variant"].eq(variant)
            & ~checkpoint_summary["is_full_wavelength_set"]
        ].sort_values("n_hf_train")

        if not partial.empty:
            ax.scatter(
                partial["n_hf_train"],
                partial["mean_nrmse"],
                marker="o",
                s=70,
                facecolors="none",
                edgecolors=colors.get(variant),
                linewidths=1.8,
                label=f"{labels.get(variant, variant.upper())} partial",
            )
            plotted_anything = True

            for row in partial.itertuples(index=False):
                ax.annotate(
                    f"{int(row.n_wavelengths)}/{RESPONSE_DIM} wl",
                    xy=(row.n_hf_train, row.mean_nrmse),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=colors.get(variant),
                )

    if not plotted_anything:
        raise ValueError("No complete or partial learning-curve points to plot.")

    ax.set_xlabel("Number of HF training samples")
    ax.set_ylabel("Mean NRMSE across wavelengths")
    ax.set_title("Model 1 learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    available_x = sorted(checkpoint_summary["n_hf_train"].unique())
    ax.set_xticks(available_x)

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=220)
    plt.close(fig)


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


if __name__ == "__main__":
    main()
