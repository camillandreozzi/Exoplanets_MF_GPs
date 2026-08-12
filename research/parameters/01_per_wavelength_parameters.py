"""Per-model diagnostic grids of the per-wavelength hyperparameters.

For every per-wavelength model (Models 1A/1B, their log-space and LOO-holdout
refits, GPBoost backends, and the HF-only GPBoost reference) this draws one
figure of every fitted
covariance parameter as a function of wavelength, with JWST instrument bands
shaded and -- for the sklearn models -- the optimiser bounds drawn as dashed
reference lines so a hyperparameter pinned at a bound (e.g. an inert input whose
ARD length scale rests at the upper bound) is obvious at a glance.

Read-only: consumes the already-saved ``*_hyperparameters.csv`` tables, never
refits, and skips any model whose table is absent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exoplanets_mf.parameter_tables import (
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    RHO_BOUNDS,
    SIGNAL_VARIANCE_BOUNDS,
    available_entries,
    input_names,
    length_scale_columns,
    load_parameter_table,
    noise_columns,
    shade_instrument_modes,
    signal_variance_columns,
)
from exoplanets_mf.paths import PARAMETERS_RESULTS_DIR

OUTPUT_DIR = PARAMETERS_RESULTS_DIR / "per_wavelength"


def _bound_lines(ax, bounds) -> None:
    lo, hi = bounds
    for value in (lo, hi):
        ax.axhline(value, color="black", ls=":", lw=0.8, alpha=0.6, zorder=1)


def _missing_panel(ax, message: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="gray",
    )


def _plot_length_scales(ax, wavelengths, df, prefix, names, *, draw_bounds) -> bool:
    columns = length_scale_columns(df, prefix)
    if not columns:
        return False

    cmap = plt.get_cmap("viridis", max(len(columns), 1))
    for i, col in enumerate(columns):
        label = names[i] if i < len(names) else col
        ax.plot(wavelengths, df[col], lw=1.0, color=cmap(i), label=label)
    ax.set_yscale("log")
    if draw_bounds:
        _bound_lines(ax, LENGTH_SCALE_BOUNDS)
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="ARD length scale")
    ax.legend(fontsize=6, ncol=2, loc="best")
    return True


def plot_model(entry, df, savepath) -> None:
    wavelengths = df["wavelength"].to_numpy()
    is_sklearn = entry.backend == "sklearn"

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"{entry.label} -- fitted per-wavelength hyperparameters", fontsize=14)

    # (0,0) rho
    ax = axes[0, 0]
    if "rho" in df.columns:
        shade_instrument_modes(ax, wavelengths)
        ax.plot(wavelengths, df["rho"], ".-", lw=1, ms=3, color=entry.color)
        if is_sklearn:
            _bound_lines(ax, RHO_BOUNDS)
            ax.set_yscale("log")
        ax.set(
            xlabel=r"wavelength $\lambda$ [$\mu$m]",
            ylabel=r"$\rho$",
            title="AR(1) scaling",
        )
    else:
        _missing_panel(ax, "no rho\n(single-fidelity model)")

    # (0,1) signal variances
    ax = axes[0, 1]
    signal = signal_variance_columns(df)
    if signal:
        shade_instrument_modes(ax, wavelengths)
        palette = {
            "signal variance": "tab:purple",
            "LF signal variance": "tab:blue",
            "discrepancy signal variance": "tab:orange",
        }
        for label, col in signal.items():
            ax.plot(wavelengths, df[col], lw=1, color=palette.get(label, "tab:gray"), label=label)
        ax.set_yscale("log")
        if is_sklearn:
            _bound_lines(ax, SIGNAL_VARIANCE_BOUNDS)
        ax.set(
            xlabel=r"wavelength $\lambda$ [$\mu$m]",
            ylabel="signal variance",
            title="Signal variance",
        )
        ax.legend(fontsize=8)
    else:
        _missing_panel(ax, "no signal variance")

    # (0,2) noise
    ax = axes[0, 2]
    shade_instrument_modes(ax, wavelengths)
    noise = noise_columns(df)
    palette = {"LF noise": "tab:green", "HF noise": "tab:red", "error variance": "tab:purple"}
    for label, col in noise.items():
        ax.plot(wavelengths, df[col], lw=1, color=palette.get(label, "tab:gray"), label=label)
    ax.set_yscale("log")
    if is_sklearn:
        _bound_lines(ax, NOISE_LEVEL_BOUNDS)
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="noise", title="Noise")
    ax.legend(fontsize=8)

    # (1,0) LF length scales
    ax = axes[1, 0]
    shade_instrument_modes(ax, wavelengths)
    if _plot_length_scales(ax, wavelengths, df, "low", input_names(), draw_bounds=is_sklearn):
        title = "LF ARD length scales" if "rho" in df.columns else "GP ARD ranges"
        ax.set_title(title)
    else:
        _missing_panel(ax, "no LF/GP length scales")

    # (1,1) discrepancy length scales
    ax = axes[1, 1]
    shade_instrument_modes(ax, wavelengths)
    if _plot_length_scales(ax, wavelengths, df, "delta", input_names(), draw_bounds=is_sklearn):
        ax.set_title("Discrepancy ARD length scales")
    else:
        _missing_panel(ax, "no discrepancy length scales")

    # (1,2) joint LML (sklearn only)
    ax = axes[1, 2]
    if "joint_log_marginal_likelihood" in df.columns:
        shade_instrument_modes(ax, wavelengths)
        ax.plot(wavelengths, df["joint_log_marginal_likelihood"], lw=1, color=entry.color)
        ax.set(
            xlabel=r"wavelength $\lambda$ [$\mu$m]",
            ylabel="joint log marginal likelihood",
            title="Joint LML",
        )
    else:
        _missing_panel(ax, "no joint LML\n(GPBoost backend)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = available_entries(kind="per_wavelength")
    if not entries:
        print("no per-wavelength hyperparameter tables found; nothing to plot")
        return

    for entry in entries:
        df = load_parameter_table(entry)
        if df is None:
            continue
        savepath = OUTPUT_DIR / f"{entry.key}.png"
        plot_model(entry, df, savepath)
        print(f"  {entry.key}: {len(df)} wavelengths -> {savepath.name}")

    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
