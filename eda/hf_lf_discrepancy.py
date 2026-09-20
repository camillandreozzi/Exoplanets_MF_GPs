"""Wavelength-resolved HF-LF discrepancy: signed bias and discrepancy RMS.

Run: python3 eda/hf_lf_discrepancy.py
Writes a figure, per-wavelength CSV and instrument summary to results/eda.

Companion to eda/hf_lf_correlation.py: correlation measures how well the LF
model tracks HF variation, this quantifies how far it sits from HF in ppm.
YHF.csv and YLF.csv are paired by row position at the same atmospheric inputs.
The discrepancy is Delta_ij = y^L_ij - y^H_ij; reported per wavelength are the
signed bias mean_i(Delta_ij), the discrepancy RMS [mean_i(Delta_ij^2)]^(1/2)
and that RMS relative to the HF spread s_j (population standard deviation of
the HF depths across atmospheres). Instrument intervals follow the correlation
script: [2.4, 4), [4, 5), [5, 12] micron; boundary points belong to the higher
band.
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

from eda.hf_lf_correlation import BANDS, instrument_labels, load_paired_spectra

PPM = 1e6


def discrepancy_table(wavelengths, hf, lf):
    """Per-wavelength bias, discrepancy RMS and HF spread, all in ppm."""
    if hf.shape != lf.shape or hf.ndim != 2 or hf.shape[1] != len(wavelengths):
        raise ValueError("HF and LF must have matching sample-by-wavelength shapes.")
    import pandas as pd

    delta = (lf - hf) * PPM
    spread = hf.std(axis=0, ddof=0) * PPM
    rms = np.sqrt(np.mean(delta**2, axis=0))
    ratio = np.full(len(wavelengths), np.nan)
    np.divide(rms, spread, out=ratio, where=spread > 0)
    return pd.DataFrame({
        "response_index": np.arange(len(wavelengths)),
        "wavelength_um": wavelengths,
        "instrument": instrument_labels(wavelengths),
        "n_pairs": hf.shape[0],
        "signed_bias_ppm": delta.mean(axis=0),
        "discrepancy_sd_ppm": delta.std(axis=0, ddof=0),
        "discrepancy_rms_ppm": rms,
        "hf_spread_ppm": spread,
        "rms_over_hf_spread": ratio,
    }).sort_values("wavelength_um").reset_index(drop=True)


def instrument_summary(table):
    """Summarise per-wavelength statistics, without pooling wavelengths."""
    return table.groupby("instrument", sort=False).agg(
        n_wavelengths=("wavelength_um", "size"),
        n_pairs_per_wavelength=("n_pairs", "first"),
        wavelength_min_um=("wavelength_um", "min"),
        wavelength_max_um=("wavelength_um", "max"),
        mean_signed_bias_ppm=("signed_bias_ppm", "mean"),
        min_signed_bias_ppm=("signed_bias_ppm", "min"),
        max_signed_bias_ppm=("signed_bias_ppm", "max"),
        mean_discrepancy_rms_ppm=("discrepancy_rms_ppm", "mean"),
        mean_rms_over_hf_spread=("rms_over_hf_spread", "mean"),
    ).reset_index()


PANELS = (
    ("signed_bias_ppm", "$\\bar\\Delta_j$ (ppm)",
     "Signed bias  $\\bar\\Delta_j$", "{:+.1f} ppm"),
    ("discrepancy_rms_ppm", "RMS$(\\Delta_j)$ (ppm)",
     "Discrepancy RMS  $[n_H^{-1}\\sum_i\\Delta_{ij}^2]^{1/2}$", "{:.1f} ppm"),
    ("rms_over_hf_spread", "RMS$(\\Delta_j)\\,/\\,s_j$",
     "Standardised discrepancy  RMS$(\\Delta_j)\\,/\\,s_j$", "{:.2f}"),
)


def plot_discrepancy(table):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    for ax, (column, ylabel, title, fmt) in zip(axes, PANELS):
        for name, low, high, color in BANDS:
            band = table[table.instrument == name]
            ax.axvspan(low, high, color=color, alpha=0.08)
            if column == "signed_bias_ppm":
                ax.fill_between(
                    band.wavelength_um,
                    band.signed_bias_ppm - band.discrepancy_sd_ppm,
                    band.signed_bias_ppm + band.discrepancy_sd_ppm,
                    color=color, alpha=0.18, linewidth=0,
                )
            ax.plot(band.wavelength_um, band[column], "o-", color=color,
                    markersize=3, linewidth=1,
                    label=f"{name} ({low:g}–{high:g} µm)" if ax is axes[0] else None)
            mean = band[column].mean()
            ax.axhline(mean, color=color, linestyle="--", linewidth=1)
            ax.annotate(
                fmt.format(mean), xy=(high - 0.05, mean), xycoords="data",
                xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
            )
        for boundary in (4, 5):
            ax.axvline(boundary, color="0.5", linestyle="--", linewidth=0.8)
        ax.set(title=title, ylabel=ylabel, xlim=(2.4, 12),
               xticks=[2.4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].axhline(0, color="0.4", linewidth=0.9)
    axes[0].legend(loc="upper left", ncol=3, fontsize=10)
    axes[-1].set_xlabel("Wavelength (µm)")
    fig.supxlabel(
        f"$\\Delta_{{ij}}=y^L_{{ij}}-y^H_{{ij}}$ at {table.n_pairs.iloc[0]} matched atmospheric inputs · "
        "original eclipse depths · $s_j$: HF spread across atmospheres · "
        "top ribbon: $\\bar\\Delta_j\\pm\\mathrm{sd}_i(\\Delta_{ij})$ · dashed lines: band means",
        fontsize=10,
    )
    return fig


def main():
    table = discrepancy_table(*load_paired_spectra())
    summary = instrument_summary(table)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("hf_lf_discrepancy_per_wavelength.csv", table),
        ("hf_lf_discrepancy_by_instrument.csv", summary),
    ):
        path = RESULTS_DIR / filename
        data.to_csv(path, index=False)
        print(f"Saved {path}")
    fig = plot_discrepancy(table)
    path = RESULTS_DIR / "hf_lf_discrepancy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
    print(
        f"All bins: mean signed bias {table.signed_bias_ppm.mean():+.1f} ppm, "
        f"range {table.signed_bias_ppm.min():+.1f} to {table.signed_bias_ppm.max():+.1f} ppm, "
        f"mean discrepancy RMS {table.discrepancy_rms_ppm.mean():.1f} ppm"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
