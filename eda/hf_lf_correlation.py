"""Pearson HF–LF correlation per wavelength, grouped by instrument band.

Run: python3 eda/hf_lf_correlation.py
Writes a figure, per-wavelength CSV and instrument summary to results/eda.

YHF.csv and YLF.csv are paired by row position at the same atmospheric inputs,
following kernel_comparison/realdata.py. The unpaired YLF10k.csv pool is not
used. Correlations are across atmospheres at each wavelength, using the
original eclipse depths. Instrument intervals follow the requested ranges:
[2.4, 4), [4, 5), [5, 12] µm; boundary points belong to the higher band.
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
import numpy as np
import pandas as pd

from src.data_load import DATA_DIR, load_wavelength_map

BANDS = (
    ("NIRCam F322W2", 2.4, 4.0, "#2166ac"),
    ("NIRCam F444W", 4.0, 5.0, "#d97716"),
    ("MIRI LRS", 5.0, 12.0, "#238b65"),
)


def instrument_labels(wavelengths):
    labels = np.full(len(wavelengths), "", dtype=object)
    for name, low, high, _ in BANDS:
        upper = wavelengths <= high if high == BANDS[-1][2] else wavelengths < high
        labels[(wavelengths >= low) & upper] = name
    if np.any(labels == ""):
        raise ValueError("Some wavelengths fall outside the requested instrument bands.")
    return labels


def load_paired_spectra():
    wavelengths = np.asarray(list(load_wavelength_map().values()))
    hf = pd.read_csv(DATA_DIR / "YHF.csv", index_col=0)
    lf = pd.read_csv(DATA_DIR / "YLF.csv")
    n_inputs = len(pd.read_csv(DATA_DIR / "XHF.csv"))
    if not (len(hf) == len(lf) == n_inputs) or len(hf) < 3:
        raise ValueError("XHF, YHF and YLF must have matching rows (at least three).")
    for name, table in (("HF", hf), ("LF", lf)):
        grid = table.columns.to_numpy(dtype=float)
        if grid.shape != wavelengths.shape or not np.allclose(
            grid, wavelengths, rtol=0, atol=1e-10
        ):
            raise ValueError(f"{name} columns do not match the wavelength map.")
        if not np.all(np.isfinite(table.to_numpy(dtype=float))):
            raise ValueError(f"{name} contains nonfinite eclipse depths.")
    return wavelengths, hf.to_numpy(dtype=float), lf.to_numpy(dtype=float)


def correlation_table(wavelengths, hf, lf):
    """One correlation per column; constant channels have undefined (NaN) r."""
    if hf.shape != lf.shape or hf.ndim != 2 or hf.shape[1] != len(wavelengths):
        raise ValueError("HF and LF must have matching sample-by-wavelength shapes.")
    hf_centered = hf - hf.mean(axis=0)
    lf_centered = lf - lf.mean(axis=0)
    denominator = np.sqrt(
        np.sum(hf_centered**2, axis=0) * np.sum(lf_centered**2, axis=0)
    )
    # Detect exactly constant inputs even if mean subtraction introduces roundoff.
    valid = (np.ptp(hf, axis=0) > 0) & (np.ptp(lf, axis=0) > 0)
    r = np.full(len(wavelengths), np.nan)
    np.divide(
        np.sum(hf_centered * lf_centered, axis=0), denominator,
        out=r, where=valid & (denominator > 0),
    )
    return pd.DataFrame({
        "response_index": np.arange(len(wavelengths)),
        "wavelength_um": wavelengths,
        "instrument": instrument_labels(wavelengths),
        "n_pairs": hf.shape[0],
        "pearson_r": np.clip(r, -1, 1),
    }).sort_values("wavelength_um").reset_index(drop=True)


def instrument_summary(table):
    """Summarise per-wavelength r values, without pooling wavelengths."""
    return table.groupby("instrument", sort=False).agg(
        n_wavelengths=("wavelength_um", "size"),
        n_defined_correlations=("pearson_r", "count"),
        n_pairs_per_wavelength=("n_pairs", "first"),
        wavelength_min_um=("wavelength_um", "min"),
        wavelength_max_um=("wavelength_um", "max"),
        mean_r=("pearson_r", "mean"),
        median_r=("pearson_r", "median"),
        min_r=("pearson_r", "min"),
        max_r=("pearson_r", "max"),
    ).reset_index()


def plot_correlations(table):
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    overview = fig.add_subplot(grid[0, :])
    detail_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    finite_r = table["pearson_r"].dropna()
    detail_low = max(-1, float(finite_r.min()) - 0.04) if len(finite_r) else -1
    detail_high = min(1, float(finite_r.max()) + 0.04) if len(finite_r) else 1

    for ax, (name, low, high, color) in zip(detail_axes, BANDS):
        band = table[table.instrument == name]
        overview.axvspan(low, high, color=color, alpha=0.08)
        overview.plot(
            band.wavelength_um, band.pearson_r, "o-", color=color,
            markersize=3, linewidth=1, label=f"{name} ({low:g}–{high:g} µm)",
        )
        ax.plot(band.wavelength_um, band.pearson_r, "o-", color=color,
                markersize=3, linewidth=1)
        median = band.pearson_r.median()
        if np.isfinite(median):
            ax.axhline(median, color=color, linestyle="--", linewidth=1,
                       label=f"Median r = {median:.3f}")
            ax.legend(loc="lower right", fontsize=9)
        ax.set(
            title=f"{name} · {len(band)} wavelengths",
            xlim=(low, high), ylim=(detail_low, detail_high),
            xlabel="Wavelength (µm)",
        )
    for boundary in (4, 5):
        overview.axvline(boundary, color="0.5", linestyle="--", linewidth=0.8)
    overview.set(
        title=f"HF–LF Pearson correlation · {table.n_pairs.iloc[0]} paired atmospheres per wavelength",
        xlabel="Wavelength (µm)", ylabel="Pearson r",
        xlim=(2.4, 12), ylim=(-1.05, 1.05),
    )
    overview.axhline(0, color="0.5", linewidth=0.7)
    overview.legend(loc="lower center", ncol=3, fontsize=10)
    detail_axes[0].set_ylabel("Pearson r (zoomed; shared limits)")
    for ax in [overview, *detail_axes]:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    note = "Original eclipse depths · YHF.csv paired by row with YLF.csv · Dashed lines in lower panels: band medians"
    undefined = table.pearson_r.isna().sum()
    if undefined:
        note += f"\n{undefined} constant-channel correlations are undefined and omitted."
    fig.supxlabel(note, fontsize=10)
    return fig


def main():
    table = correlation_table(*load_paired_spectra())
    summary = instrument_summary(table)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("hf_lf_correlation_per_wavelength.csv", table),
        ("hf_lf_correlation_by_instrument.csv", summary),
    ):
        path = RESULTS_DIR / filename
        data.to_csv(path, index=False)
        print(f"Saved {path}")
    fig = plot_correlations(table)
    path = RESULTS_DIR / "hf_lf_correlation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
