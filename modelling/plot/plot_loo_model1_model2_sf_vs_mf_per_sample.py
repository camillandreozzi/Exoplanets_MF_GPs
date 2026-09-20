"""Per-sample LOO CV error for Model 1 and Model 2, single- vs multi-fidelity.

The companion figure (plot_loo_model1_model2_sf_vs_mf.py) collapses folds and
keeps wavelength. This one is the transpose: it collapses wavelength and keeps
the held-out sample, so each point is one LOO fold scored over its whole
spectrum.

Metric definitions (population standard deviation, ddof=0):

    NRMSE_s = sqrt( mean_w residual_sw^2 ) / std_s
    MAE_s   = mean_w | residual_sw |

where std_s is the standard deviation of the held-out sample's y_true spectrum.
Note this is a different normaliser from the per-wavelength figure, which
divides by each wavelength's standard deviation across folds, so the NRMSE
levels here are not directly comparable with that figure's.

Each panel is a paired scatter: one point per held-out sample, SF on x and MF
on y, against the y = x line. Below the line means multi-fidelity won for that
sample. Colour follows the winner, matching the delta convention used elsewhere
in the repo.
"""

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

# Validated categorical slots 1 and 2: worst-pair CVD dE 24.7, normal-vision 33.6.
SF_COLOR = "#2a78d6"
MF_COLOR = "#eb6834"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

MODELS = ("model1", "model2")
MODEL_TITLES = {
    "model1": "Model 1 — wavelength-wise GP",
    "model2": "Model 2 — augmented GP",
}

MAE_SCALE = 1e4
METRICS = (
    ("nrmse", "NRMSE", 1.0, "{:.4f}"),
    ("mae", "MAE (×10⁻⁴)", MAE_SCALE, "{:.3f}"),
)


def main():
    predictions = read_predictions()
    per_sample = compute_per_sample(predictions)

    table_path = RESULTS_DIR / f"model1_model2_{CV_LABEL}_per_sample_metrics.csv"
    per_sample.to_csv(table_path, index=False)
    print(f"Saved {table_path} ({len(per_sample)} rows)")

    figure = plot_grid(per_sample)
    output_path = RESULTS_DIR / f"model1_model2_{CV_LABEL}_sf_vs_mf_per_sample.png"
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {output_path}")
    for line in summary_lines(per_sample):
        print(line)


def read_predictions():
    predictions_file = RESULTS_DIR / f"{CV_LABEL}_predictions.csv"
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"{predictions_file} not found; run modelling/cv/cv_loo.py first."
        )

    predictions = pd.read_csv(predictions_file)
    required = {
        "held_out_source_index",
        "fold",
        "model",
        "variant",
        "wavelength",
        "y_true",
        "y_pred",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")

    predictions = predictions[predictions["model"].isin(MODELS)].copy()
    if predictions.empty:
        raise ValueError(f"Predictions file holds no rows for {MODELS}.")

    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    return predictions


def compute_per_sample(predictions):
    """Collapse wavelength within each (model, variant, held-out sample)."""
    grouped = predictions.groupby(
        ["model", "variant", "held_out_source_index", "fold"], as_index=False
    ).agg(
        n_wavelengths=("wavelength", "size"),
        rmse=("error", lambda column: float(np.sqrt(np.mean(column**2)))),
        mae=("error", lambda column: float(np.mean(np.abs(column)))),
        bias=("error", "mean"),
        max_abs_error=("error", lambda column: float(np.max(np.abs(column)))),
        y_true_std=("y_true", lambda column: float(column.std(ddof=0))),
    )

    # Normalize each held-out sample by its spectrum's population std.
    grouped["nrmse"] = grouped["rmse"] / grouped["y_true_std"].replace(0, np.nan)

    wide = grouped.pivot(
        index=["model", "held_out_source_index", "fold"],
        columns="variant",
        values=[
            "n_wavelengths",
            "nrmse",
            "rmse",
            "mae",
            "bias",
            "max_abs_error",
            "y_true_std",
        ],
    )
    wide.columns = [f"{metric}_{variant}" for metric, variant in wide.columns]
    wide = wide.reset_index()

    for metric in ("nrmse", "mae"):
        wide[f"{metric}_delta_mf_minus_sf"] = wide[f"{metric}_mf"] - wide[f"{metric}_sf"]

    return wide.sort_values(["model", "fold"])


def plot_grid(per_sample):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    figure.patch.set_facecolor(SURFACE)

    for row, (metric, metric_label, scale, fmt) in enumerate(METRICS):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            panel = per_sample[per_sample["model"] == model]
            plot_panel(ax, panel, metric, metric_label, scale, fmt)

            if row == 0:
                ax.set_title(
                    MODEL_TITLES[model],
                    fontsize=12,
                    color=INK_PRIMARY,
                    loc="left",
                    pad=42,
                )

    legend = figure.legend(
        handles=winner_legend_handles(),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=10,
        ncol=2,
        handletextpad=0.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.suptitle(
        "Leave-one-out CV error per held-out sample, averaged over wavelength",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK_PRIMARY,
    )
    figure.text(
        0.06,
        0.947,
        "97 held-out samples · 195 wavelengths each · NRMSE normalised by each "
        "sample's spectrum standard deviation · below the diagonal = MF better",
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.subplots_adjust(hspace=0.3, wspace=0.22)
    return figure


def plot_panel(ax, panel, metric, metric_label, scale, fmt):
    sf = panel[f"{metric}_sf"] * scale
    mf = panel[f"{metric}_mf"] * scale
    mf_wins = mf < sf

    low = min(sf.min(), mf.min())
    high = max(sf.max(), mf.max())
    pad = (high - low) * 0.08
    limits = (low - pad, high + pad)

    ax.plot(limits, limits, color=BASELINE, linewidth=1, zorder=1)

    # Colour follows the winner, matching the MF-minus-SF delta panels.
    for mask, color in ((mf_wins, MF_COLOR), (~mf_wins, SF_COLOR)):
        ax.scatter(
            sf[mask],
            mf[mask],
            s=42,
            color=color,
            edgecolor=SURFACE,
            linewidth=1,
            alpha=0.9,
            zorder=3,
        )

    style_axes(ax)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Single-fidelity {metric_label}", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel(f"Multi-fidelity {metric_label}", fontsize=11, color=INK_SECONDARY)

    annotate_panel(ax, sf, mf, int(mf_wins.sum()), len(panel), fmt)


def annotate_panel(ax, sf, mf, mf_wins, total, fmt):
    improvement = (1 - mf.mean() / sf.mean()) * 100
    ax.annotate(
        f"mean  SF {fmt.format(sf.mean())}   MF {fmt.format(mf.mean())}"
        f"   ({improvement:+.1f}% MF)\n"
        f"MF better for {mf_wins} of {total} samples",
        xy=(0.0, 1.03),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=9,
        color=INK_MUTED,
    )

    ax.annotate(
        "MF better",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.annotate(
        "SF better",
        xy=(0.03, 0.94),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )


def winner_legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            label=label,
        )
        for label, color in (("MF better", MF_COLOR), ("SF better", SF_COLOR))
    ]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=10)


def summary_lines(per_sample):
    lines = []
    for metric, label, _scale, _fmt in METRICS:
        for model in MODELS:
            panel = per_sample[per_sample["model"] == model]
            sf = panel[f"{metric}_sf"].mean()
            mf = panel[f"{metric}_mf"].mean()
            wins = int((panel[f"{metric}_mf"] < panel[f"{metric}_sf"]).sum())
            lines.append(
                f"{model} {label.split(' ')[0]}: mean {sf:.6g} SF vs {mf:.6g} MF "
                f"({(1 - mf / sf) * 100:+.1f}%) · MF better for {wins} of {len(panel)} samples"
            )
    return lines


if __name__ == "__main__":
    main()
