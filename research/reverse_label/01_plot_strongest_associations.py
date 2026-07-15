"""Visualize the strongest inverse single-wavelength associations.

The preceding reverse-label workflow writes one first-order association curve
per physical parameter. This script selects the strongest wavelength for C, f,
O, and N and creates:

1. wavelength-wise association curves with each maximum highlighted;
2. density plots of Y(lambda*) against the corresponding parameter, with the
   quantile-binned conditional mean used by the first-order estimator.

Run after ``00_reverse_input_sensitivity.py`` or as part of:

    make reverse-label
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_XLF_10k, load_YLF_10k
from exoplanets_mf.instruments import INSTRUMENT_MODES
from exoplanets_mf.paths import REVERSE_LABEL_RESULTS_DIR

RESULTS_DIR = REVERSE_LABEL_RESULTS_DIR
SENSITIVITY_CSV = RESULTS_DIR / "00_sensitivity_by_wavelength.csv"

TARGET_PARAMETERS = ("C", "f", "O", "N")
N_BINS = 20


def selected_associations(
    sensitivity: pd.DataFrame,
) -> list[dict[str, float | str]]:
    """Return the peak wavelength and value for each selected parameter."""
    selected = []
    for parameter in TARGET_PARAMETERS:
        column = f"first_order_{parameter}"
        if column not in sensitivity:
            raise ValueError(f"Missing required sensitivity column: {column}")
        row = sensitivity.loc[sensitivity[column].idxmax()]
        selected.append(
            {
                "parameter": parameter,
                "wavelength_um": float(row["wavelength_um"]),
                "association": float(row[column]),
            }
        )
    return selected


def binned_conditional_summary(
    spectral_value: np.ndarray,
    parameter_value: np.ndarray,
    *,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, conditional means, and 10--90% envelopes."""
    edges = np.unique(
        np.quantile(spectral_value, np.linspace(0, 1, n_bins + 1))
    )
    if len(edges) < 2:
        raise ValueError("Cannot bin a constant spectral channel")
    bin_index = np.searchsorted(edges[1:-1], spectral_value, side="right")

    centers = []
    means = []
    lower = []
    upper = []
    for index in range(len(edges) - 1):
        mask = bin_index == index
        centers.append(spectral_value[mask].mean())
        means.append(parameter_value[mask].mean())
        lower.append(np.quantile(parameter_value[mask], 0.10))
        upper.append(np.quantile(parameter_value[mask], 0.90))
    return (
        np.asarray(centers),
        np.asarray(means),
        np.asarray(lower),
        np.asarray(upper),
    )


def _shade_instrument_modes(
    axis,
    wavelengths: np.ndarray,
    *,
    show_labels: bool,
) -> None:
    for mode in INSTRUMENT_MODES:
        lower = max(wavelengths.min(), mode.wavelength_min)
        upper = min(wavelengths.max(), mode.wavelength_max)
        axis.axvspan(
            lower,
            upper,
            color=mode.color,
            alpha=0.55,
            zorder=0,
            label=mode.label if show_labels else None,
        )


def plot_association_spectra(
    sensitivity: pd.DataFrame,
    selected: list[dict[str, float | str]],
    savepath,
) -> None:
    """Plot S(lambda) for each parameter and mark the reported maximum."""
    wavelengths = sensitivity["wavelength_um"].to_numpy()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        sharex=True,
        sharey=True,
    )

    for panel_index, (axis, item) in enumerate(
        zip(axes.flat, selected, strict=True)
    ):
        parameter = str(item["parameter"])
        values = sensitivity[f"first_order_{parameter}"].to_numpy()
        peak_wavelength = float(item["wavelength_um"])
        peak_value = float(item["association"])

        _shade_instrument_modes(
            axis,
            wavelengths,
            show_labels=panel_index == 0,
        )
        axis.plot(wavelengths, values, color="tab:blue", lw=1.5)
        axis.scatter(
            [peak_wavelength],
            [peak_value],
            color="tab:red",
            edgecolor="white",
            linewidth=0.8,
            s=65,
            zorder=3,
        )
        axis.annotate(
            f"{peak_wavelength:.4f} μm\nS = {peak_value:.4f}",
            xy=(peak_wavelength, peak_value),
            xytext=(8, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
        )
        axis.set_title(f"Parameter {parameter}")
        axis.set_ylabel("first-order association")
        axis.grid(alpha=0.2)

    axes[0, 0].legend(loc="upper right", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel(r"spectral predictor wavelength $\lambda$ [$\mu$m]")
    fig.suptitle(
        "Strongest inverse single-wavelength association for each parameter"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_selected_relationships(
    parameter_frame: pd.DataFrame,
    spectrum_frame: pd.DataFrame,
    selected: list[dict[str, float | str]],
    savepath,
) -> None:
    """Show the four selected Y(lambda*) -> X relationships directly."""
    wavelengths = spectrum_frame.columns.to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for axis, item in zip(axes.flat, selected, strict=True):
        parameter = str(item["parameter"])
        peak_wavelength = float(item["wavelength_um"])
        association = float(item["association"])
        wavelength_index = int(np.argmin(np.abs(wavelengths - peak_wavelength)))
        actual_wavelength = wavelengths[wavelength_index]

        spectral_value = spectrum_frame.iloc[
            :, wavelength_index
        ].to_numpy(dtype=float)
        parameter_value = parameter_frame[parameter].to_numpy(dtype=float)
        centers, means, lower, upper = binned_conditional_summary(
            spectral_value,
            parameter_value,
        )

        axis.hexbin(
            spectral_value,
            parameter_value,
            gridsize=45,
            mincnt=1,
            bins="log",
            cmap="Blues",
        )
        axis.fill_between(
            centers,
            lower,
            upper,
            color="tab:orange",
            alpha=0.18,
            label="bin 10–90%",
        )
        axis.plot(
            centers,
            means,
            "o-",
            color="tab:orange",
            lw=2,
            ms=4,
            label="binned conditional mean",
        )
        axis.set_title(
            f"{parameter} ← Y({actual_wavelength:.4f} μm), "
            f"S = {association:.4f}"
        )
        axis.set_xlabel(
            rf"LF spectral value $Y({actual_wavelength:.4f}\,\mu m)$"
        )
        axis.set_ylabel(f"parameter {parameter}")
        axis.ticklabel_format(
            axis="x",
            style="sci",
            scilimits=(-3, 3),
        )
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)

    fig.suptitle(
        "Direct view of the strongest inverse single-wavelength associations"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main() -> None:
    if not SENSITIVITY_CSV.exists():
        raise FileNotFoundError(
            f"{SENSITIVITY_CSV} does not exist. Run "
            "00_reverse_input_sensitivity.py first."
        )

    sensitivity = pd.read_csv(SENSITIVITY_CSV)
    selected = selected_associations(sensitivity)
    parameter_frame = load_XLF_10k()
    spectrum_frame = load_YLF_10k()

    plot_association_spectra(
        sensitivity,
        selected,
        RESULTS_DIR / "01_strongest_association_spectra.png",
    )
    plot_selected_relationships(
        parameter_frame,
        spectrum_frame,
        selected,
        RESULTS_DIR / "01_strongest_association_relationships.png",
    )

    print("\nSelected inverse associations:")
    for item in selected:
        print(
            f"  {item['parameter']:>2}: "
            f"{item['wavelength_um']:.4f} um, "
            f"S = {item['association']:.4f}"
        )
    print(f"\nFigures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
