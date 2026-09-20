"""Plot all HF, LF and observed eclipse depths on linear and log10 scales.

Run: python3 eda/spectra_analysis.py
Outputs: results/eda/spectra_linear.png and spectra_log10.png.
Both the 10,000 LF design spectra and the 97 paired LF spectra are included.
Nonpositive measurements remain in the linear plot; their logarithms are
undefined, so they are omitted and counted in the log10 plot.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.data_load import DATA_DIR, first_existing_file, load_wavelength_map


def load_spectra():
    wavelengths = np.asarray(list(load_wavelength_map().values()))
    tables = {
        "HF": pd.read_csv(DATA_DIR / "YHF.csv", index_col=0),
        "LF design": pd.read_csv(
            first_existing_file(DATA_DIR, ["YLF10k.csv", "YLF_10k.csv"])
        ),
        "LF paired": pd.read_csv(DATA_DIR / "YLF.csv"),
    }
    order = np.argsort(wavelengths)
    spectra = {}
    for name, table in tables.items():
        grid = table.columns.to_numpy(dtype=float)
        if grid.shape != wavelengths.shape or not np.allclose(
            grid, wavelengths, rtol=0, atol=1e-10
        ):
            raise ValueError(f"{name} columns do not match the wavelength map.")
        values = table.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains nonfinite eclipse depths.")
        spectra[name] = values[:, order]

    observed = pd.read_csv(DATA_DIR / "Observed_Spectra.csv").sort_values("wavelength")
    if not np.all(np.isfinite(observed[["wavelength", "measured_eclipse_depth"]])):
        raise ValueError("Observed wavelengths or eclipse depths are nonfinite.")
    return wavelengths[order], spectra, observed


def transform_values(values, log_scale):
    if not log_scale:
        return values
    transformed = np.full(values.shape, np.nan, dtype=float)
    np.log10(values, out=transformed, where=values > 0)
    return transformed


def plot_spectra(wavelengths, spectra, observed, log_scale=False):
    fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
    handles = []
    styles = (
        ("LF design", "#d97716", 0.015, 0.5),
        ("LF paired", "#d97716", 0.15, 0.6),
        ("HF", "#2166ac", 0.4, 0.7),
    )
    omitted = []
    for name, color, alpha, linewidth in styles:
        values = spectra[name]
        y = transform_values(values, log_scale)
        segments = np.stack((np.broadcast_to(wavelengths, y.shape), y), axis=-1)
        ax.add_collection(LineCollection(
            segments, colors=color, alpha=alpha, linewidths=linewidth,
        ))
        handles.append(Line2D(
            [], [], color=color, linewidth=1.5,
            label=f"{name}: {len(values):,} spectra",
        ))
        if log_scale and np.any(values <= 0):
            omitted.append(f"{name}: {np.count_nonzero(values <= 0):,}")

    observed_values = observed["measured_eclipse_depth"].to_numpy()
    observed_line, = ax.plot(
        observed["wavelength"], transform_values(observed_values, log_scale),
        "o-", color="#111111", markersize=3, linewidth=1, zorder=5,
        label=f"Observed: {len(observed_values)} points",
    )
    handles.append(observed_line)
    if log_scale and np.any(observed_values <= 0):
        omitted.append(f"observed: {np.count_nonzero(observed_values <= 0)}")
    if omitted:
        fig.supxlabel(
            "Nonpositive values omitted from log₁₀ view (" + "; ".join(omitted)
            + "). All values appear in the linear view.", fontsize=10,
        )

    ax.autoscale_view()
    ax.margins(x=0.01)
    ax.set(
        title="All HF, LF and observed spectra — "
        + ("log₁₀ values" if log_scale else "linear values"),
        xlabel="Wavelength (µm)",
        ylabel="log₁₀(Eclipse depth)" if log_scale else "Eclipse depth",
    )
    # Keep a linear axis after transforming the data: ticks show -4, -3, etc.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    return fig


def main():
    wavelengths, spectra, observed = load_spectra()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for log_scale, filename in (
        (False, "spectra_linear.png"), (True, "spectra_log10.png"),
    ):
        fig = plot_spectra(wavelengths, spectra, observed, log_scale=log_scale)
        output_path = RESULTS_DIR / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved {output_path}")
    for name, values in spectra.items():
        print(f"{name}: {len(values):,} spectra; {np.count_nonzero(values <= 0)} nonpositive values")
    print(
        f"Observed: {len(observed)} points; "
        f"{np.count_nonzero(observed.measured_eclipse_depth <= 0)} "
        "nonpositive values omitted from log10 view"
    )


if __name__ == "__main__":
    main()
