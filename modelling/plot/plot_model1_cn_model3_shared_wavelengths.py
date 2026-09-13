"""Model 1 vs CN vs Model 3 on the 28 wavelengths all three cover.

Reads the tables written by
modelling/cv/compare_model1_cn_model3_shared_wavelengths.py and draws the two
metric views the earlier LOO figures use, restricted to the shared band.

Encoding: colour is the model, dash is the fidelity variant (solid MF, dashed
SF, CN solid since it has no split). That keeps the categorical palette at
three validated hues rather than five, and means fidelity is never carried by
hue. Note this differs from the SF/MF figures, where colour *is* the variant.

The NRMSE row is log-scaled: Model 3 MF spans an order of magnitude more than
the rest, and a linear axis would flatten every other series into one line.
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
SUFFIX = "shared28wl"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Categorical slots 1-3, validated all-pairs: worst CVD dE 9.2, normal 24.0.
# Aqua warns on contrast (2.74:1), so every series is also direct-labelled and
# the companion CSVs stand as the table view.
MODEL_COLORS = {"model1": "#2a78d6", "cn": "#eb6834", "model3": "#1baf7a"}
VARIANT_STYLE = {"sf": (2, 2), "mf": None, "cn": None}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

PREDICTORS = (
    ("model1-sf", "model1", "sf"),
    ("model1-mf", "model1", "mf"),
    ("model3-sf", "model3", "sf"),
    ("model3-mf", "model3", "mf"),
    ("cn", "cn", "cn"),
)

MAE_SCALE = 1e4
METRICS = (
    ("nrmse", "NRMSE", 1.0, True),
    ("mae", "MAE (×10⁻⁴)", MAE_SCALE, False),
)


def main():
    per_wavelength = pd.read_csv(
        RESULTS_DIR / f"model1_cn_model3_{CV_LABEL}_{SUFFIX}_per_wavelength.csv"
    )
    per_sample = pd.read_csv(
        RESULTS_DIR / f"model1_cn_model3_{CV_LABEL}_{SUFFIX}_per_sample.csv"
    )

    figure = plot_grid(per_wavelength, per_sample)
    output_path = RESULTS_DIR / f"model1_cn_model3_{CV_LABEL}_{SUFFIX}.png"
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {output_path}")


def plot_grid(per_wavelength, per_sample):
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), width_ratios=[1.6, 1])
    figure.patch.set_facecolor(SURFACE)

    for row, (metric, label, scale, log) in enumerate(METRICS):
        plot_wavelength_panel(axes[row][0], per_wavelength, metric, label, scale, log)
        plot_sample_panel(axes[row][1], per_sample, metric, label, scale, log)

    legend = figure.legend(
        handles=legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.053, 0.925),
        frameon=False,
        fontsize=10,
        ncol=5,
        columnspacing=1.2,
        handlelength=2.2,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.suptitle(
        "Model 1 vs CN vs Model 3 — LOO error on the 28 shared wavelengths",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK_PRIMARY,
    )
    figure.text(
        0.055,
        0.945,
        "2.458–2.862 μm — the only band Model 3 has completed · 97 folds · "
        "colour = model, dashed = single-fidelity",
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.885))
    figure.subplots_adjust(hspace=0.28, wspace=0.2)
    return figure


def plot_wavelength_panel(ax, per_wavelength, metric, label, scale, log):
    for name, model, variant in PREDICTORS:
        series = per_wavelength[per_wavelength["predictor"] == name].sort_values(
            "wavelength"
        )
        ax.plot(
            series["wavelength"],
            series[metric] * scale,
            color=MODEL_COLORS[model],
            linewidth=2,
            dashes=VARIANT_STYLE[variant] or (None, None),
            marker="o",
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            label=name,
        )

    style_axes(ax, log)
    # Reserve a right margin inside the axes so the direct labels sit on the
    # panel rather than in the gap between panels.
    low = per_wavelength["wavelength"].min()
    high = per_wavelength["wavelength"].max()
    span = high - low
    ax.set_xlim(low - 0.03 * span, high + 0.22 * span)
    ax.set_ylabel(label, fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Wavelength (μm)", fontsize=11, color=INK_SECONDARY)
    ax.set_title(
        f"Per wavelength (n=28)", fontsize=11, color=INK_PRIMARY, loc="left", pad=8
    )
    label_line_ends(ax, per_wavelength, metric, scale, log)


def label_line_ends(ax, per_wavelength, metric, scale, log):
    """Direct labels - the aqua slot warns on contrast, so text carries identity.

    Four of the five series converge at the right edge, so label positions are
    spread apart before drawing. Axes fractions are computed from the limits
    rather than through ax.transData, which is not yet settled at this point.
    """
    low, high = ax.get_ylim()
    x_low, x_high = ax.get_xlim()
    label_x = (
        per_wavelength["wavelength"].max() - x_low
    ) / (x_high - x_low) + 0.015

    def to_fraction(value):
        if log:
            return (np.log10(value) - np.log10(low)) / (np.log10(high) - np.log10(low))
        return (value - low) / (high - low)

    placed = sorted(
        (
            (
                name,
                to_fraction(
                    per_wavelength[per_wavelength["predictor"] == name]
                    .sort_values("wavelength")[metric]
                    .iloc[-1]
                    * scale
                ),
            )
            for name, _model, _variant in PREDICTORS
        ),
        key=lambda item: item[1],
    )

    # Spread around the block's centre and clamp inside the axes, so labels
    # are never pushed off an edge.
    gap = min(0.058, 0.94 / max(len(placed) - 1, 1))
    positions = [fraction for _name, fraction in placed]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)

    shift = np.mean([fraction for _name, fraction in placed]) - np.mean(positions)
    positions = [y + shift for y in positions]
    overflow = max(0.0, positions[-1] - 0.97) or min(0.0, positions[0] - 0.03)
    positions = [y - overflow for y in positions]

    for (name, _original), y in zip(placed, positions):
        ax.annotate(
            name,
            xy=(label_x, y),
            xycoords="axes fraction",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def plot_sample_panel(ax, per_sample, metric, label, scale, log):
    names = [name for name, _model, _variant in PREDICTORS]
    data = [
        per_sample[per_sample["predictor"] == name][metric].values * scale
        for name in names
    ]

    boxes = ax.boxplot(
        data,
        positions=np.arange(len(names)),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": INK_MUTED,
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": SURFACE, "linewidth": 1.6},
        whiskerprops={"color": BASELINE, "linewidth": 1},
        capprops={"color": BASELINE, "linewidth": 1},
    )
    for patch, (name, model, variant) in zip(boxes["boxes"], PREDICTORS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(1)
        patch.set_alpha(0.55 if variant == "sf" else 0.95)

    style_axes(ax, log)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9, color=INK_SECONDARY, rotation=20, ha="right")
    ax.set_ylabel(label, fontsize=11, color=INK_SECONDARY)
    ax.set_title(
        "Per held-out sample (n=97)",
        fontsize=11,
        color=INK_PRIMARY,
        loc="left",
        pad=8,
    )


def legend_handles():
    return [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            linewidth=2,
            dashes=VARIANT_STYLE[variant] or (None, None),
            label=name,
        )
        for name, model, variant in PREDICTORS
    ]


def style_axes(ax, log):
    ax.set_facecolor(SURFACE)
    if log:
        ax.set_yscale("log")
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=10)


if __name__ == "__main__":
    main()
