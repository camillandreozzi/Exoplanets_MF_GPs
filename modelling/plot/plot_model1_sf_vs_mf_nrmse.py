"""Per-wavelength NRMSE, single- vs multi-fidelity, for Model 1 LOO CV.

Reads the SF-vs-MF comparison written by modelling/cv/cv_loo.py and draws two
stacked panels sharing a wavelength axis: the two NRMSE curves, and the signed
difference that says which variant wins where.
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


def main():
    comparison = read_comparison()
    figure = plot_sf_vs_mf(comparison)

    output_path = RESULTS_DIR / f"model1_{CV_LABEL}_sf_vs_mf_nrmse.png"
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {output_path}")
    print(summary_line(comparison))


def read_comparison():
    comparison_file = (
        RESULTS_DIR / f"model1_{CV_LABEL}_sf_vs_mf_nrmse_per_wavelength.csv"
    )
    if not comparison_file.exists():
        raise FileNotFoundError(
            f"{comparison_file} not found; run modelling/cv/cv_loo.py first."
        )

    comparison = pd.read_csv(comparison_file).sort_values("wavelength")
    missing = {"wavelength", "sf_nrmse", "mf_nrmse", "delta_mf_minus_sf"} - set(
        comparison.columns
    )
    if missing:
        raise ValueError(f"Comparison file is missing columns: {sorted(missing)}")

    return comparison


def plot_sf_vs_mf(comparison):
    figure, (curves_ax, delta_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12},
    )
    figure.patch.set_facecolor(SURFACE)

    plot_curves(curves_ax, comparison)
    plot_delta(delta_ax, comparison)

    figure.suptitle(
        "Model 1 leave-one-out CV: single- vs multi-fidelity NRMSE",
        x=0.125,
        y=0.965,
        ha="left",
        fontsize=14,
        color=INK_PRIMARY,
    )
    figure.text(
        0.125,
        0.925,
        summary_line(comparison),
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def plot_curves(ax, comparison):
    wavelength = comparison["wavelength"]
    ax.plot(wavelength, comparison["sf_nrmse"], color=SF_COLOR, linewidth=2, label="SF")
    ax.plot(wavelength, comparison["mf_nrmse"], color=MF_COLOR, linewidth=2, label="MF")

    label_line_ends(ax, comparison)
    style_axes(ax)
    ax.set_ylabel("NRMSE", fontsize=11, color=INK_SECONDARY)

    legend = ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=1.8,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def label_line_ends(ax, comparison):
    """Direct labels so identity is never carried by colour alone."""
    last = comparison.iloc[-1]
    span = comparison["sf_nrmse"].max() - comparison["mf_nrmse"].min()
    offsets = {"SF": 0.0, "MF": 0.0}
    if last["mf_nrmse"] > last["sf_nrmse"]:
        offsets = {"MF": 0.02 * span, "SF": -0.02 * span}
    else:
        offsets = {"MF": -0.02 * span, "SF": 0.02 * span}

    for name, column, color in (
        ("SF", "sf_nrmse", SF_COLOR),
        ("MF", "mf_nrmse", MF_COLOR),
    ):
        y = last[column] + offsets[name]
        ax.plot(
            [last["wavelength"] + 0.08],
            [y],
            marker="o",
            markersize=5,
            color=color,
            clip_on=False,
        )
        ax.annotate(
            name,
            xy=(last["wavelength"] + 0.18, y),
            va="center",
            fontsize=10,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def plot_delta(ax, comparison):
    wavelength = comparison["wavelength"]
    delta = comparison["delta_mf_minus_sf"]

    # Colour follows the entity: orange where MF wins, blue where SF wins.
    ax.fill_between(
        wavelength,
        delta,
        0,
        where=delta <= 0,
        interpolate=True,
        color=MF_COLOR,
        alpha=0.85,
        linewidth=0,
    )
    ax.fill_between(
        wavelength,
        delta,
        0,
        where=delta >= 0,
        interpolate=True,
        color=SF_COLOR,
        alpha=0.85,
        linewidth=0,
    )
    ax.axhline(0, color=BASELINE, linewidth=1)

    style_axes(ax)
    ax.set_ylabel("MF - SF", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Wavelength (μm)", fontsize=11, color=INK_SECONDARY)

    limit = max(abs(delta.min()), abs(delta.max())) * 1.25
    ax.set_ylim(-limit, limit)

    annotate_delta_regions(ax, comparison, limit)


def annotate_delta_regions(ax, comparison, limit):
    right = comparison["wavelength"].max()
    ax.annotate(
        "MF better",
        xy=(right, -limit * 0.62),
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.annotate(
        "SF better",
        xy=(right, limit * 0.48),
        ha="right",
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


def summary_line(comparison):
    delta = comparison["delta_mf_minus_sf"]
    mf_wins = int((delta < 0).sum())
    total = len(comparison)
    sf_mean = comparison["sf_nrmse"].mean()
    mf_mean = comparison["mf_nrmse"].mean()
    improvement = (1 - mf_mean / sf_mean) * 100

    return (
        f"MF lower at {mf_wins} of {total} wavelengths  ·  "
        f"mean NRMSE {sf_mean:.4f} SF vs {mf_mean:.4f} MF "
        f"({improvement:+.1f}%)  ·  97 LOO folds per wavelength"
    )


if __name__ == "__main__":
    main()