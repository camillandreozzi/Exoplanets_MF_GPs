"""Scalar-hyperparameter summaries for the wavelength-augmented Model 2 variants.

Model 2 folds wavelength into the kernel as an extra input dimension, so it has
a single set of hyperparameters for the whole spectrum rather than a per-
wavelength curve. For each available Model 2 variant (linear, log10, GPBoost)
this draws the 9 atmospheric-input ARD length scales (LF vs discrepancy) -- the
key sanity check for which inputs the model treats as informative -- alongside a
scalar summary of the remaining covariance parameters (signal variances, noise,
the wavelength length scale in micrometres, with rho annotated).

Read-only: consumes the saved single-row ``*_hyperparameters.csv`` tables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exoplanets_mf.parameter_tables import (
    LENGTH_SCALE_BOUNDS,
    available_entries,
    input_names,
    length_scale_columns,
    load_parameter_table,
    noise_columns,
    signal_variance_columns,
)
from exoplanets_mf.paths import PARAMETERS_RESULTS_DIR

OUTPUT_DIR = PARAMETERS_RESULTS_DIR / "scalar"


def _atmospheric_length_scales(df, prefix, n_inputs):
    """First ``n_inputs`` ARD dims (the atmospheric inputs; lambda is separate)."""
    columns = length_scale_columns(df, prefix)[:n_inputs]
    return np.array([float(df[col].iloc[0]) for col in columns])


def plot_model(entry, df, savepath) -> None:
    names = input_names()
    n_inputs = len(names)
    low_ls = _atmospheric_length_scales(df, "low", n_inputs)
    delta_ls = _atmospheric_length_scales(df, "delta", n_inputs)
    rho = float(df["rho"].iloc[0]) if "rho" in df.columns else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    rho_text = rf" ($\rho$ = {rho:.4g})" if np.isfinite(rho) else ""
    fig.suptitle(f"{entry.label} -- scalar hyperparameters{rho_text}", fontsize=14)

    # Panel 0: atmospheric ARD length scales, LF vs discrepancy.
    ax = axes[0]
    positions = np.arange(n_inputs)
    ax.bar(positions - 0.2, low_ls, width=0.4, color="tab:blue", label="LF")
    ax.bar(positions + 0.2, delta_ls, width=0.4, color="tab:orange", label="discrepancy")
    ax.set_yscale("log")
    if entry.backend == "sklearn":
        for value in LENGTH_SCALE_BOUNDS:
            ax.axhline(value, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set(ylabel="ARD length scale (standardized units)", title="Atmospheric-input ARD length scales")
    ax.legend()

    # Panel 1: remaining scalar covariance parameters (all strictly positive).
    ax = axes[1]
    summary: dict[str, float] = {
        label.replace("variance", "var"): float(df[col].iloc[0])
        for label, col in signal_variance_columns(df).items()
    }
    for label, col in noise_columns(df).items():
        summary[label] = float(df[col].iloc[0])
    for col, label in (
        ("low_lambda_length_scale_um", r"LF $\lambda$ length scale [$\mu$m]"),
        ("delta_lambda_length_scale_um", r"discrepancy $\lambda$ length scale [$\mu$m]"),
    ):
        if col in df.columns:
            summary[label] = float(df[col].iloc[0])

    labels = list(summary.keys())
    values = np.array([summary[k] for k in labels])
    y = np.arange(len(labels))
    ax.barh(y, np.abs(values), color=entry.color)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, value in zip(y, values):
        ax.text(np.abs(value), yi, f" {value:.3g}", va="center", fontsize=8)
    ax.set(xlabel="value (log scale)", title="Scalar covariance parameters")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = available_entries(kind="scalar")
    if not entries:
        print("no scalar (Model 2) hyperparameter tables found; nothing to plot")
        return

    for entry in entries:
        df = load_parameter_table(entry)
        if df is None:
            continue
        savepath = OUTPUT_DIR / f"{entry.key}.png"
        plot_model(entry, df, savepath)
        print(f"  {entry.key} -> {savepath.name}")

    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
