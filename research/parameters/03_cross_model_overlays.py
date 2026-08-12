"""Cross-model overlays of the fitted hyperparameters + an at-a-glance table.

Puts every model on shared axes so they can be checked against each other:
rho, signal variances and noise as a function of wavelength (per-wavelength
models overlaid, Model 2's scalar values drawn as reference lines), and a per-
input ARD length-scale comparison. Linear- and log-space fits of Models 1A/1B
appear together in the same overlays, distinguished by colour.

Also writes ``parameter_summary.csv`` -- one row per model summarising rho, the
variance/noise ranges, and how many ARD length scales rest at a bound (a proxy
for inert inputs / degenerate fits) -- as the quickest sanity-check artifact.

Read-only: consumes the saved ``*_hyperparameters.csv`` tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exoplanets_mf.parameter_tables import (
    LENGTH_SCALE_BOUNDS,
    available_entries,
    count_pinned,
    input_names,
    length_scale_columns,
    load_parameter_table,
    noise_columns,
    shade_instrument_modes,
    signal_variance_columns,
)
from exoplanets_mf.paths import PARAMETERS_RESULTS_DIR

OUTPUT_DIR = PARAMETERS_RESULTS_DIR / "overlays"


def _first_wavelengths(tables):
    for entry, df in tables:
        if "wavelength" in df.columns:
            return df["wavelength"].to_numpy()
    return None


def plot_rho(per_wavelength, scalar, savepath) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    wl = _first_wavelengths(per_wavelength)
    if wl is not None:
        shade_instrument_modes(ax, wl)
    for entry, df in per_wavelength:
        if "rho" in df.columns:
            ax.plot(df["wavelength"], df["rho"], lw=1, color=entry.color, label=entry.label)
    for entry, df in scalar:
        if "rho" in df.columns:
            ax.axhline(float(df["rho"].iloc[0]), ls="--", lw=1.2, color=entry.color,
                       label=f"{entry.label} (scalar)")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel=r"$\rho$",
           title="AR(1) scaling across all models")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_family(per_wavelength, scalar, column, title, savepath) -> None:
    def selected_column(df):
        if column in df.columns:
            return column
        if column == "low_signal_variance" and "signal_variance" in df.columns:
            return "signal_variance"
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    wl = _first_wavelengths(per_wavelength)
    if wl is not None:
        shade_instrument_modes(ax, wl)
    for entry, df in per_wavelength:
        col = selected_column(df)
        if col is not None:
            suffix = "" if col == column else " (signal variance)"
            ax.plot(df["wavelength"], df[col], lw=1, color=entry.color,
                    label=f"{entry.label}{suffix}")
    for entry, df in scalar:
        col = selected_column(df)
        if col is not None:
            suffix = "" if col == column else ", signal variance"
            ax.axhline(float(df[col].iloc[0]), ls="--", lw=1.2, color=entry.color,
                       label=f"{entry.label} (scalar{suffix})")
    ax.set_yscale("log")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel=column, title=title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise(per_wavelength, scalar, savepath) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    wl = _first_wavelengths(per_wavelength)
    if wl is not None:
        shade_instrument_modes(ax, wl)
    # Prefer HF noise; fall back to the GPBoost error variance.
    for entry, df in per_wavelength:
        col = "high_noise" if "high_noise" in df.columns else (
            "error_var" if "error_var" in df.columns else None)
        if col is not None:
            ax.plot(df["wavelength"], df[col], lw=1, color=entry.color,
                    label=f"{entry.label} ({col})")
    for entry, df in scalar:
        col = "high_noise" if "high_noise" in df.columns else (
            "error_var" if "error_var" in df.columns else None)
        if col is not None:
            ax.axhline(float(df[col].iloc[0]), ls="--", lw=1.2, color=entry.color,
                       label=f"{entry.label} ({col}, scalar)")
    ax.set_yscale("log")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="noise",
           title="HF noise / error variance across all models")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_length_scale_by_input(all_tables, savepath) -> None:
    """Median ARD length scale per atmospheric input, sklearn models only.

    GPBoost "range" values live in different units, so they are excluded to
    keep the comparison on a common scale.
    """
    names = input_names()
    n_inputs = len(names)
    sklearn = [(e, df) for e, df in all_tables if e.backend == "sklearn"]
    if not sklearn:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, prefix, title in zip(axes, ("low", "delta"),
                                 ("LF ARD length scale", "Discrepancy ARD length scale")):
        positions = np.arange(n_inputs)
        width = 0.8 / max(len(sklearn), 1)
        for k, (entry, df) in enumerate(sklearn):
            cols = length_scale_columns(df, prefix)[:n_inputs]
            medians = np.array([float(np.median(df[c])) for c in cols])
            ax.bar(positions + (k - len(sklearn) / 2 + 0.5) * width, medians,
                   width=width, color=entry.color, label=entry.label)
        ax.set_yscale("log")
        for value in LENGTH_SCALE_BOUNDS:
            ax.axhline(value, color="black", ls=":", lw=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("median length scale (standardized units)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-input ARD length scales across sklearn models", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary(all_tables) -> pd.DataFrame:
    rows = []
    for entry, df in all_tables:
        low_ls = np.concatenate([df[c].to_numpy() for c in length_scale_columns(df, "low")]) \
            if length_scale_columns(df, "low") else np.array([])
        delta_ls = np.concatenate([df[c].to_numpy() for c in length_scale_columns(df, "delta")]) \
            if length_scale_columns(df, "delta") else np.array([])
        pinned = np.nan
        if entry.backend == "sklearn":
            pinned = count_pinned(low_ls, LENGTH_SCALE_BOUNDS) + count_pinned(
                delta_ls, LENGTH_SCALE_BOUNDS)
        noise = noise_columns(df)
        noise_vals = np.concatenate([df[c].to_numpy() for c in noise.values()]) \
            if noise else np.array([np.nan])
        signal = signal_variance_columns(df)

        def min_or_nan(column: str) -> float:
            return float(np.min(df[column])) if column in df.columns else np.nan

        def max_or_nan(column: str) -> float:
            return float(np.max(df[column])) if column in df.columns else np.nan

        if "rho" in df.columns:
            rho_mean = float(np.mean(df["rho"]))
            rho_min = float(np.min(df["rho"]))
            rho_max = float(np.max(df["rho"]))
        else:
            rho_mean = rho_min = rho_max = np.nan
        rows.append({
            "key": entry.key,
            "label": entry.label,
            "kind": entry.kind,
            "backend": entry.backend,
            "space": entry.space,
            "n_rows": len(df),
            "rho_mean": rho_mean,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "signal_var_min": min_or_nan(signal.get("signal variance", "")),
            "signal_var_max": max_or_nan(signal.get("signal variance", "")),
            "low_signal_var_min": min_or_nan(signal.get("LF signal variance", "")),
            "low_signal_var_max": max_or_nan(signal.get("LF signal variance", "")),
            "delta_signal_var_min": min_or_nan(signal.get("discrepancy signal variance", "")),
            "delta_signal_var_max": max_or_nan(signal.get("discrepancy signal variance", "")),
            "noise_min": float(np.nanmin(noise_vals)),
            "noise_max": float(np.nanmax(noise_vals)),
            "length_scales_pinned_at_bound": pinned,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables = []
    for entry in available_entries():
        df = load_parameter_table(entry)
        if df is not None:
            all_tables.append((entry, df))
    if not all_tables:
        print("no hyperparameter tables found; nothing to plot")
        return

    per_wavelength = [(e, df) for e, df in all_tables if e.kind == "per_wavelength"]
    scalar = [(e, df) for e, df in all_tables if e.kind == "scalar"]

    if per_wavelength or scalar:
        plot_rho(per_wavelength, scalar, OUTPUT_DIR / "rho_vs_wavelength.png")
        plot_variance_family(per_wavelength, scalar, "low_signal_variance",
                             "LF / single-fidelity signal variance across all models",
                             OUTPUT_DIR / "low_signal_variance.png")
        plot_variance_family(per_wavelength, scalar, "delta_signal_variance",
                             "Discrepancy signal variance across all models",
                             OUTPUT_DIR / "delta_signal_variance.png")
        plot_noise(per_wavelength, scalar, OUTPUT_DIR / "noise.png")
    plot_length_scale_by_input(all_tables, OUTPUT_DIR / "length_scale_by_input.png")

    summary = build_summary(all_tables)
    summary.to_csv(OUTPUT_DIR / "parameter_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
