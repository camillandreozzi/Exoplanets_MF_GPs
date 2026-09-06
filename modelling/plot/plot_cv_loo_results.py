"""Plot LOO CV results saved by modelling/cv/cv_loo.py."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "cv" / "loo"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
CV_LABEL = "loo"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["model1", "model2"]
VARIANT_ORDER = ["sf", "mf"]
MODEL_LABELS = {
    "model1": "Model 1 wavelength-wise",
    "model2": "Model 2 augmented-wavelength",
}
VARIANT_LABELS = {
    "sf": "SF",
    "mf": "MF",
}
COLORS = {
    ("model1", "sf"): "tab:blue",
    ("model1", "mf"): "tab:cyan",
    ("model2", "sf"): "tab:orange",
    ("model2", "mf"): "tab:red",
}
LINESTYLES = {
    "sf": "-",
    "mf": "--",
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = read_csv(f"{CV_LABEL}_metrics_per_wavelength.csv")
    validate_metrics(metrics)

    predictions = read_optional_csv(f"{CV_LABEL}_predictions.csv")
    if predictions is not None:
        validate_predictions(predictions)

    generated = []
    generated.append(
        plot_metric_curves(
            metrics,
            metric="rmse",
            ylabel="RMSE",
            filename=f"{CV_LABEL}_rmse_per_wavelength.png",
        )
    )
    generated.append(
        plot_metric_curves(
            metrics,
            metric="nrmse",
            ylabel="NRMSE",
            filename=f"{CV_LABEL}_nrmse_per_wavelength.png",
        )
    )

    delta_file = plot_sf_mf_delta(metrics)
    if delta_file is not None:
        generated.append(delta_file)

    generated.append(plot_summary(metrics))

    if predictions is None:
        print(
            f"No {CV_LABEL}_predictions.csv found in {RESULTS_DIR}; "
            "skipped prediction-based plots.",
            flush=True,
        )
    else:
        generated.append(plot_actual_vs_predicted(predictions))
        generated.append(plot_residual_distributions(predictions))
        generated.append(plot_spectrum_examples(predictions))

    print("Saved LOO CV plots:")
    for path in generated:
        print(f"  {path}")


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing LOO CV result file: {path}. "
            "Run modelling/cv/cv_loo.py before plotting."
        )
    return pd.read_csv(path)


def read_optional_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def validate_metrics(metrics):
    required_columns = {
        "model",
        "variant",
        "response_index",
        "wavelength",
        "rmse",
        "nrmse",
        "mae",
        "bias",
    }
    missing = sorted(required_columns - set(metrics.columns))
    if missing:
        raise ValueError(f"LOO metrics are missing required columns: {missing}")
    if metrics.empty:
        raise ValueError("LOO metrics are empty.")


def validate_predictions(predictions):
    required_columns = {
        "fold",
        "held_out_source_index",
        "model",
        "variant",
        "response_index",
        "wavelength",
        "y_true",
        "y_pred",
        "residual",
    }
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"LOO predictions are missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("LOO predictions are empty.")


def plot_metric_curves(metrics, metric, ylabel, filename):
    models = active_models(metrics)
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(11, 3.8 * len(models)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        model_data = metrics[metrics["model"].eq(model)]
        for variant in active_variants(model_data):
            series = (
                model_data[model_data["variant"].eq(variant)]
                .sort_values("wavelength")
            )
            ax.plot(
                series["wavelength"],
                series[metric],
                label=variant_label(variant),
                color=color_for(model, variant),
                linestyle=LINESTYLES.get(variant, "-"),
                linewidth=1.6,
            )

        ax.set_title(f"{model_label(model)} LOO CV")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Wavelength")
    output_path = RESULTS_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_sf_mf_delta(metrics):
    models = [
        model
        for model in active_models(metrics)
        if {"sf", "mf"}.issubset(set(metrics.loc[metrics["model"].eq(model), "variant"]))
    ]
    if not models:
        print("Skipped SF-vs-MF delta plot; both variants are not present.", flush=True)
        return None

    fig, axes = plt.subplots(
        len(models),
        2,
        figsize=(13, 3.8 * len(models)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(models), 2)

    for row_number, model in enumerate(models):
        model_data = metrics[metrics["model"].eq(model)]
        rmse_delta = metric_delta(model_data, "rmse")
        nrmse_delta = metric_delta(model_data, "nrmse")

        plot_delta_panel(
            axes[row_number, 0],
            rmse_delta,
            title=f"{model_label(model)} MF - SF RMSE",
            ylabel="Delta RMSE",
            color=color_for(model, "sf"),
        )
        plot_delta_panel(
            axes[row_number, 1],
            nrmse_delta,
            title=f"{model_label(model)} MF - SF NRMSE",
            ylabel="Delta NRMSE",
            color=color_for(model, "mf"),
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength")

    output_path = RESULTS_DIR / f"{CV_LABEL}_mf_minus_sf_delta_per_wavelength.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def metric_delta(model_data, metric):
    wide = (
        model_data
        .pivot_table(
            index=["response_index", "wavelength"],
            columns="variant",
            values=metric,
            aggfunc="first",
        )
        .reset_index()
        .sort_values("wavelength")
    )
    wide["delta_mf_minus_sf"] = wide["mf"] - wide["sf"]
    return wide


def plot_delta_panel(ax, data, title, ylabel, color):
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
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_summary(metrics):
    summary = (
        metrics
        .groupby(["model", "variant"])
        .agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .reset_index()
        .assign(label=lambda df: df["model"] + " " + df["variant"].str.upper())
    )
    summary["model_order"] = summary["model"].map(model_sort_key)
    summary["variant_order"] = summary["variant"].map(variant_sort_key)
    summary = summary.sort_values(["model_order", "variant_order", "label"])

    x = np.arange(len(summary))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_summary_panel(
        axes[0],
        x,
        summary,
        mean_col="mean_rmse",
        median_col="median_rmse",
        ylabel="RMSE",
        title="LOO CV RMSE summary",
        width=width,
    )
    plot_summary_panel(
        axes[1],
        x,
        summary,
        mean_col="mean_nrmse",
        median_col="median_nrmse",
        ylabel="NRMSE",
        title="LOO CV NRMSE summary",
        width=width,
    )

    for ax in axes:
        ax.set_xticks(x, summary["label"], rotation=20, ha="right")

    output_path = RESULTS_DIR / f"{CV_LABEL}_summary_rmse_nrmse.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_summary_panel(ax, x, summary, mean_col, median_col, ylabel, title, width):
    ax.bar(
        x - width / 2,
        summary[mean_col],
        width,
        label="Mean",
        color="tab:purple",
        alpha=0.78,
    )
    ax.bar(
        x + width / 2,
        summary[median_col],
        width,
        label="Median",
        color="tab:gray",
        alpha=0.78,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)


def plot_actual_vs_predicted(predictions):
    models = active_models(predictions)
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7.5, 5 * len(models)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        model_data = predictions[predictions["model"].eq(model)]
        for variant in active_variants(model_data):
            series = model_data[model_data["variant"].eq(variant)]
            ax.scatter(
                series["y_true"],
                series["y_pred"],
                s=14,
                alpha=0.45,
                color=color_for(model, variant),
                label=variant_label(variant),
                edgecolors="none",
            )

        add_identity_line(ax, model_data)
        ax.set_title(f"{model_label(model)} actual vs predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    output_path = RESULTS_DIR / f"{CV_LABEL}_actual_vs_predicted.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def add_identity_line(ax, data):
    values = pd.concat([data["y_true"], data["y_pred"]]).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    low = float(np.min(values))
    high = float(np.max(values))
    ax.plot([low, high], [low, high], color="black", linewidth=1, alpha=0.65)


def plot_residual_distributions(predictions):
    groups = []
    labels = []
    colors = []

    for model in active_models(predictions):
        model_data = predictions[predictions["model"].eq(model)]
        for variant in active_variants(model_data):
            series = model_data.loc[model_data["variant"].eq(variant), "residual"]
            groups.append(series.to_numpy(dtype=float))
            labels.append(f"{model}\n{variant.upper()}")
            colors.append(color_for(model, variant))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 5), constrained_layout=True)
    box = ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("LOO CV residual distributions")
    ax.set_ylabel("Prediction residual")
    ax.grid(axis="y", alpha=0.25)

    output_path = RESULTS_DIR / f"{CV_LABEL}_residual_distributions.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_spectrum_examples(predictions):
    examples = select_example_cases(predictions)
    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(11, 3.4 * len(examples)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (fold, source_index, label) in zip(axes, examples):
        fold_data = predictions[
            predictions["fold"].eq(fold)
            & predictions["held_out_source_index"].eq(source_index)
        ]
        actual = (
            fold_data
            .drop_duplicates(["response_index", "wavelength"])
            .sort_values("wavelength")
        )
        ax.plot(
            actual["wavelength"],
            actual["y_true"],
            color="black",
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label="Actual",
        )

        for model in active_models(fold_data):
            model_data = fold_data[fold_data["model"].eq(model)]
            for variant in active_variants(model_data):
                series = (
                    model_data[model_data["variant"].eq(variant)]
                    .sort_values("wavelength")
                )
                ax.plot(
                    series["wavelength"],
                    series["y_pred"],
                    color=color_for(model, variant),
                    linestyle=LINESTYLES.get(variant, "-"),
                    linewidth=1.1,
                    marker=".",
                    markersize=2.5,
                    label=f"{model} {variant.upper()}",
                )

        ax.set_title(f"{label}: fold {fold}, source {source_index}")
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=min(3, len(ax.get_legend_handles_labels()[0])))

    axes[-1].set_xlabel("Wavelength")
    output_path = RESULTS_DIR / f"{CV_LABEL}_example_spectra.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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
        (int(best_fold), int(best_source), "Best average residual"),
        (int(median_fold), int(median_source), "Median average residual"),
        (int(worst_fold), int(worst_source), "Worst average residual"),
    ]
    deduped = []
    seen = set()
    for fold, source_index, label in examples:
        key = (fold, source_index)
        if key not in seen:
            deduped.append((fold, source_index, label))
            seen.add(key)
    return deduped


def active_models(data):
    models = [model for model in MODEL_ORDER if model in set(data["model"])]
    extras = sorted(set(data["model"]) - set(MODEL_ORDER))
    return [*models, *extras]


def active_variants(data):
    variants = [variant for variant in VARIANT_ORDER if variant in set(data["variant"])]
    extras = sorted(set(data["variant"]) - set(VARIANT_ORDER))
    return [*variants, *extras]


def model_label(model):
    return MODEL_LABELS.get(model, model)


def variant_label(variant):
    return VARIANT_LABELS.get(variant, variant.upper())


def color_for(model, variant):
    return COLORS.get((model, variant), "tab:gray")


def model_sort_key(model):
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model)
    return len(MODEL_ORDER)


def variant_sort_key(variant):
    if variant in VARIANT_ORDER:
        return VARIANT_ORDER.index(variant)
    return len(VARIANT_ORDER)


if __name__ == "__main__":
    main()
