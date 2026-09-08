"""Model 1 vs Model 2 LOO CV error, single- vs multi-fidelity, in NRMSE and MAE.

Reads the per-wavelength metrics written by modelling/cv/cv_loo.py and draws a
2x2 grid of small multiples: one column per model, one row per metric. Each
panel carries the same two curves (SF, MF) against wavelength, so a column is
read down for "does this model's fidelity story change with the metric" and a
row is read across for "which model is better on this metric".

Panels keep independent y-limits: Model 2's NRMSE runs an order of magnitude
above Model 1's, and a shared scale would flatten Model 1 into a line. Each
panel is annotated with its own means so the levels stay comparable by number.
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
MODEL_TITLES = {"model1": "Model 1 — wavelength-wise GP", "model2": "Model 2 — augmented GP"}

# MAE lives around 1e-4; rescaling keeps the tick labels readable.
MAE_SCALE = 1e4
METRICS = (
    ("nrmse", "NRMSE", 1.0, "{:.3f}"),
    ("mae", "MAE (×10⁻⁴)", MAE_SCALE, "{:.3f}"),
)


def main():
    metrics = read_metrics()
    figure = plot_grid(metrics)

    output_path = RESULTS_DIR / f"model1_model2_{CV_LABEL}_sf_vs_mf_nrmse_mae.png"
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {output_path}")
    for line in summary_lines(metrics):
        print(line)


def read_metrics():
    metrics_file = RESULTS_DIR / f"{CV_LABEL}_metrics_per_wavelength.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"{metrics_file} not found; run modelling/cv/cv_loo.py first."
        )

    metrics = pd.read_csv(metrics_file)
    required = {"model", "variant", "wavelength", "nrmse", "mae"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")

    metrics = metrics[metrics["model"].isin(MODELS)]
    if metrics.empty:
        raise ValueError(f"Metrics file holds no rows for {MODELS}.")

    return metrics.sort_values("wavelength")


def plot_grid(metrics):
    figure, axes = plt.subplots(
        2,
        2,
        sharex=True,
        figsize=(12.5, 8),
    )
    figure.patch.set_facecolor(SURFACE)

    for row, (column, metric_label, scale, fmt) in enumerate(METRICS):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            panel = metrics[metrics["model"] == model]
            plot_panel(ax, panel, column, scale, fmt)

            style_axes(ax)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11, color=INK_SECONDARY)
            if row == 0:
                ax.set_title(
                    MODEL_TITLES[model],
                    fontsize=12,
                    color=INK_PRIMARY,
                    loc="left",
                    pad=26,
                )
            if row == len(METRICS) - 1:
                ax.set_xlabel("Wavelength (μm)", fontsize=11, color=INK_SECONDARY)

    legend_handles, legend_labels = axes[0][0].get_legend_handles_labels()
    legend = figure.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        frameon=False,
        fontsize=10,
        ncol=2,
        handlelength=1.8,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.suptitle(
        "Leave-one-out CV error: single- vs multi-fidelity, per wavelength",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK_PRIMARY,
    )
    figure.text(
        0.07,
        0.942,
        "97 LOO folds per wavelength · 195 wavelengths · panels keep independent y-scales",
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.subplots_adjust(hspace=0.3, wspace=0.16)
    return figure


def plot_panel(ax, panel, column, scale, fmt):
    means = {}
    for label, variant, color in (("SF", "sf", SF_COLOR), ("MF", "mf", MF_COLOR)):
        series = panel[panel["variant"] == variant].sort_values("wavelength")
        values = series[column] * scale
        means[label] = values.mean()
        ax.plot(
            series["wavelength"],
            values,
            color=color,
            linewidth=2,
            label=label,
            zorder=3 if variant == "mf" else 2,
        )

    label_line_ends(ax, panel, column, scale)
    annotate_means(ax, means, fmt)


def label_line_ends(ax, panel, column, scale):
    """Direct labels so identity is never carried by colour alone."""
    ends = {}
    for label, variant in (("SF", "sf"), ("MF", "mf")):
        series = panel[panel["variant"] == variant].sort_values("wavelength")
        last = series.iloc[-1]
        ends[label] = (last["wavelength"], last[column] * scale)

    span = (panel[column].max() - panel[column].min()) * scale
    nudge = 0.03 * span
    if ends["MF"][1] > ends["SF"][1]:
        offsets = {"MF": nudge, "SF": -nudge}
    else:
        offsets = {"MF": -nudge, "SF": nudge}

    x_span = panel["wavelength"].max() - panel["wavelength"].min()
    for label, color in (("SF", SF_COLOR), ("MF", MF_COLOR)):
        x, y = ends[label]
        y += offsets[label]
        ax.plot(
            [x + 0.008 * x_span],
            [y],
            marker="o",
            markersize=5,
            color=color,
            clip_on=False,
            zorder=4,
        )
        ax.annotate(
            label,
            xy=(x + 0.022 * x_span, y),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def annotate_means(ax, means, fmt):
    improvement = (1 - means["MF"] / means["SF"]) * 100
    text = (
        f"mean  SF {fmt.format(means['SF'])}   MF {fmt.format(means['MF'])}"
        f"   ({improvement:+.1f}% MF)"
    )
    # Above the axes, not inside: the NRMSE panels both spike into the corner
    # a caption would otherwise sit in.
    ax.annotate(
        text,
        xy=(0.0, 1.03),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=9,
        color=INK_MUTED,
    )


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.margins(x=0.03)


def summary_lines(metrics):
    lines = []
    for column, label, _scale, _fmt in METRICS:
        for model in MODELS:
            panel = metrics[metrics["model"] == model]
            sf = panel[panel["variant"] == "sf"][column].mean()
            mf = panel[panel["variant"] == "mf"][column].mean()
            wins = wavelength_wins(panel, column)
            lines.append(
                f"{model} {label.split(' ')[0]}: mean {sf:.6g} SF vs {mf:.6g} MF "
                f"({(1 - mf / sf) * 100:+.1f}%) · MF lower at {wins[0]} of {wins[1]} wavelengths"
            )
    return lines


def wavelength_wins(panel, column):
    wide = panel.pivot_table(index="wavelength", columns="variant", values=column)
    wide = wide.dropna()
    return int((wide["mf"] < wide["sf"]).sum()), len(wide)


if __name__ == "__main__":
    main()
