"""Compare LOO residuals from Model 1 MF with Celine's CN model.

YSF_LOO.csv has no sample identifier, so its 97 rows are assumed to follow the
same high-fidelity sample order as YHF.csv and the LOO folds (folds 1--97).
Residuals use the repository convention: prediction minus truth.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "cv" / "loo"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL1_COLOR = "#eb6834"
CN_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
ZERO = "#898781"


def main():
    residuals = read_residuals()
    summary = summarise(residuals)
    summary_path = RESULTS_DIR / "model1_mf_vs_cn_loo_residuals_per_wavelength.csv"
    summary.to_csv(summary_path, index=False)
    overall = overall_summary(summary)
    overall_path = RESULTS_DIR / "model1_mf_vs_cn_loo_metrics.csv"
    overall.to_csv(overall_path, index=False)

    figure = plot_residuals(residuals, summary)
    output_path = RESULTS_DIR / "model1_mf_vs_cn_loo_residuals.png"
    figure.savefig(output_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {output_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {overall_path}")
    for row in overall.itertuples(index=False):
        print(
            f"{row.model}: mean NRMSE={row.mean_nrmse:.6g}, "
            f"MAE={row.mae:.6g}, RMSE={row.rmse:.6g}, bias={row.bias:.6g}"
        )


def read_residuals():
    prediction_path = RESULTS_DIR / "loo_predictions.csv"
    cn_path = PROJECT_ROOT / "data" / "lit" / "YSF_LOO.csv"
    for path in (prediction_path, cn_path):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(prediction_path)
    model1 = predictions[
        (predictions["model"] == "model1") & (predictions["variant"] == "mf")
    ].copy()
    model1 = model1.sort_values(["fold", "response_index"])

    cn = pd.read_csv(cn_path)
    wavelengths = np.asarray(cn.columns, dtype=float)
    n_folds, n_wavelengths = cn.shape
    if n_folds != model1["fold"].nunique():
        raise ValueError(
            f"CN has {n_folds} rows but Model 1 MF has "
            f"{model1['fold'].nunique()} folds."
        )
    if n_wavelengths != model1["response_index"].nunique():
        raise ValueError("CN and Model 1 MF wavelength counts differ.")

    model1_wavelengths = (
        model1.drop_duplicates("response_index")
        .sort_values("response_index")["wavelength"]
        .to_numpy()
    )
    if not np.allclose(wavelengths, model1_wavelengths):
        raise ValueError("CN and Model 1 MF wavelength grids differ.")

    truth = (
        model1.pivot(index="fold", columns="response_index", values="y_true")
        .sort_index()
        .to_numpy()
    )
    model1_residual = (
        model1.pivot(index="fold", columns="response_index", values="residual")
        .sort_index()
        .to_numpy()
    )
    cn_residual = cn.to_numpy(dtype=float) - truth

    common = {
        "fold": np.repeat(np.arange(1, n_folds + 1), n_wavelengths),
        "wavelength": np.tile(wavelengths, n_folds),
    }
    return pd.concat(
        [
            pd.DataFrame(
                {**common, "model": "Model 1 MF", "y_true": truth.ravel(),
                 "residual": model1_residual.ravel()}
            ),
            pd.DataFrame(
                {**common, "model": "CN model", "y_true": truth.ravel(),
                 "residual": cn_residual.ravel()}
            ),
        ],
        ignore_index=True,
    )


def summarise(residuals):
    summary = (
        residuals.groupby(["model", "wavelength"], as_index=False)["residual"]
        .agg(
            mean="mean",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            rmse=lambda x: float(np.sqrt(np.mean(x**2))),
            mae=lambda x: float(np.mean(np.abs(x))),
        )
        .sort_values(["model", "wavelength"])
    )
    y_range = (
        residuals.groupby("wavelength")["y_true"].agg(lambda x: x.max() - x.min())
    )
    summary["y_true_range"] = summary["wavelength"].map(y_range)
    summary["nrmse"] = summary["rmse"] / summary["y_true_range"].replace(0, np.nan)
    return summary


def overall_summary(summary):
    """Match loo_summary.csv: NRMSE is averaged across wavelengths."""
    rows = []
    for model in ("Model 1 MF", "CN model"):
        panel = summary[summary["model"] == model]
        # Overall MAE/RMSE/bias are reconstructed from equal-sized wavelength groups.
        rows.append(
            {
                "model": model,
                "n_wavelengths": len(panel),
                "mean_nrmse": panel["nrmse"].mean(),
                "median_nrmse": panel["nrmse"].median(),
                "mae": panel["mae"].mean(),
                "rmse": float(np.sqrt(np.mean(panel["rmse"] ** 2))),
                "bias": panel["mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_residuals(residuals, summary):
    figure, (ax_curve, ax_dist) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.3, 1]}
    )
    figure.patch.set_facecolor(SURFACE)

    for label, color in (("Model 1 MF", MODEL1_COLOR), ("CN model", CN_COLOR)):
        panel = summary[summary["model"] == label]
        x = panel["wavelength"].to_numpy()
        ax_curve.fill_between(
            x, panel["q25"].to_numpy(), panel["q75"].to_numpy(),
            color=color, alpha=0.16, linewidth=0,
        )
        ax_curve.plot(x, panel["mean"], color=color, linewidth=1.8, label=label)

    ax_curve.axhline(0, color=ZERO, linewidth=1)
    ax_curve.set_ylabel("Residual (prediction − truth)")
    ax_curve.set_xlabel("Wavelength (μm)")
    ax_curve.legend(frameon=False, loc="upper left", ncol=2)
    ax_curve.text(
        0.995, 0.97, "line: mean · band: interquartile range",
        transform=ax_curve.transAxes, ha="right", va="top",
        fontsize=9, color=INK_SECONDARY,
    )

    data = [
        residuals.loc[residuals["model"] == label, "residual"].to_numpy()
        for label in ("Model 1 MF", "CN model")
    ]
    parts = ax_dist.violinplot(data, positions=[1, 2], showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], (MODEL1_COLOR, CN_COLOR)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    parts["cmeans"].set_color(INK)
    ax_dist.axhline(0, color=ZERO, linewidth=1)
    ax_dist.set_xticks([1, 2], ["Model 1 MF", "CN model"])
    ax_dist.set_ylabel("Residual")

    for ax in (ax_curve, ax_dist):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY)

    figure.suptitle(
        "LOO residuals: Model 1 multi-fidelity vs CN model",
        x=0.08, y=0.99, ha="left", fontsize=15, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


if __name__ == "__main__":
    main()
