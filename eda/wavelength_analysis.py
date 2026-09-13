"""Plot the wavelength grid; run with python3 eda/wavelength_analysis.py.

Figures are saved to results/eda, independently of the working directory.
Each spectral channel is counted once, using the project's wavelength map.
"""

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "eda"
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from src.data_load import load_wavelength_map


def plot_histogram(wavelengths):
    """Count channels in equal-width bins so sampling density is comparable."""
    bin_width = 0.5
    start = np.floor(wavelengths[0] / bin_width) * bin_width
    stop = np.ceil(wavelengths[-1] / bin_width) * bin_width
    bins = np.arange(start, stop + bin_width / 2, bin_width)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.hist(wavelengths, bins=bins, color="#2a78d6", edgecolor="white")
    ax.set(
        title=f"Wavelength distribution — {len(wavelengths)} points",
        xlabel="Wavelength (µm)",
        ylabel="Number of points per 0.5 µm bin",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig


def plot_spacing(wavelengths):
    """Show exact channel positions and adjacent gaps on shared linear axes."""
    gaps = np.diff(wavelengths)
    midpoints = (wavelengths[:-1] + wavelengths[1:]) / 2
    fig, (positions_ax, gaps_ax) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        height_ratios=[1, 3], layout="constrained",
    )
    positions_ax.eventplot(wavelengths, colors="#2a78d6", linewidths=0.8)
    positions_ax.set_title("Wavelength positions and spacing")
    positions_ax.set_yticks([])
    positions_ax.set_ylabel("Each tick\nis one point")
    gaps_ax.plot(midpoints, gaps, "o-", color="#2a78d6", markersize=3, linewidth=1)
    gaps_ax.set(
        xlabel="Wavelength (µm)",
        ylabel="Gap between adjacent points (µm)",
        ylim=(0, gaps.max() * 1.2),
        title="Adjacent gaps plotted at the midpoint of each pair",
    )
    gaps_ax.text(
        0.98, 0.96,
        f"Smallest gap: {gaps.min():.4f} µm\nLargest gap: {gaps.max():.4f} µm",
        transform=gaps_ax.transAxes, ha="right", va="top",
    )
    for ax in (positions_ax, gaps_ax):
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    return fig


def main():
    wavelengths = np.sort(np.asarray(list(load_wavelength_map().values())))
    if len(wavelengths) < 2 or not np.all(np.isfinite(wavelengths)):
        raise ValueError("At least two finite wavelength points are required.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, plot in (
        ("wavelength_histogram.png", plot_histogram),
        ("wavelength_spacing.png", plot_spacing),
    ):
        fig = plot(wavelengths)
        output_path = RESULTS_DIR / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved {output_path}")
    print(
        f"{len(wavelengths)} wavelength points: "
        f"{wavelengths[0]:.4f}–{wavelengths[-1]:.4f} µm"
    )


if __name__ == "__main__":
    main()
