"""Fit the per-wavelength joint AR(1) multi-fidelity GPs on log10 spectra.

Log-transformed mirror of research/modelling/01_per_wavelength_ar1/
01_fit_joint_mf_gp.py: YHF and YLF_10k are log10-transformed after loading,
so all fitted quantities (rho, kernels, noise) live in log10 space and
predictions must be back-transformed (see research/log_modelling/README.md).

The LF subsample is maximized against a practical time budget (the exact fit
on all 10,097 points is infeasible -- see research/modelling/
01_per_wavelength_ar1/00_benchmark_exact_fit.py, where the cost benchmark
lives; the compute cost here is identical to the linear-scale fit).
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import fit_joint_mf_gp
from exoplanets_mf.paths import LOG_MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

OUTPUT_DIR = LOG_MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "joint_mf_gp"

# Same subsample as the linear-scale fit -- see the sizing notes in
# research/modelling/01_per_wavelength_ar1/01_fit_joint_mf_gp.py
# (TESTING value; ~25 min for the full 195-wavelength run).
SUBSAMPLE_SIZE = 400
SEED = RANDOM_SEED


def hyperparameter_table(layer) -> pd.DataFrame:
    """One row per wavelength of joint-fit covariance parameters."""
    rows = []
    for wl, gp in zip(layer.wavelengths, layer.models):
        kernel = gp.kernel_
        low_amplitude, low_rbf = kernel.low_kernel.k1, kernel.low_kernel.k2
        delta_amplitude, delta_rbf = (
            kernel.discrepancy_kernel.k1,
            kernel.discrepancy_kernel.k2,
        )
        row = {
            "wavelength": wl,
            "rho": kernel.rho,
            "low_signal_variance": low_amplitude.constant_value,
            "delta_signal_variance": delta_amplitude.constant_value,
            "low_noise": kernel.low_noise,
            "high_noise": kernel.high_noise,
            "joint_log_marginal_likelihood": gp.log_marginal_likelihood(),
        }
        if layer.fit_seconds is not None:
            row["fit_seconds"] = layer.fit_seconds[len(rows)]
        row.update(
            {
                f"low_length_scale_{i}": ell
                for i, ell in enumerate(low_rbf.length_scale)
            }
        )
        row.update(
            {
                f"delta_length_scale_{i}": ell
                for i, ell in enumerate(delta_rbf.length_scale)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(table: pd.DataFrame, savepath) -> None:
    n_panels = 4 if "fit_seconds" in table else 3
    fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    ax[0].plot(table["wavelength"], table["rho"], ".-")
    ax[0].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel=r"$\rho$",
        title="Jointly fitted scaling (log10 spectra)",
    )
    ax[1].plot(
        table["wavelength"], table["low_signal_variance"], ".-", label="LF"
    )
    ax[1].plot(
        table["wavelength"],
        table["delta_signal_variance"],
        ".-",
        label="discrepancy",
    )
    ax[1].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="signal variance",
        title="Joint covariance components (log10 spectra)",
    )
    ax[1].legend()
    ax[2].plot(
        table["wavelength"], table["joint_log_marginal_likelihood"], ".-"
    )
    ax[2].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="log marginal likelihood",
        title="Joint LF/HF fit (log10 spectra)",
    )
    if "fit_seconds" in table:
        ax[3].plot(table["wavelength"], table["fit_seconds"], ".-")
        ax[3].set(
            xlabel=r"wavelength $\lambda$ [$\mu$m]",
            ylabel="fit wall time [s]",
            title="Per-wavelength fit time",
        )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = log10_spectra(data["YLF_10k"].to_numpy())
    XHF = data["XHF"].to_numpy()
    YHF = log10_spectra(data["YHF"].to_numpy())
    wavelengths = data["wavelengths"]

    print(
        f"Fitting joint MF-GPs (log10 spectra): {len(wavelengths)} "
        f"wavelengths, {SUBSAMPLE_SIZE} of {len(XLF_10k)} LF + {len(XHF)} HF "
        f"rows per fit"
    )
    t0 = time.perf_counter()
    layer = fit_joint_mf_gp(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        progress_every=20,
    )
    elapsed = time.perf_counter() - t0
    print(f"fit in {elapsed:.1f}s ({elapsed / len(wavelengths):.2f}s/wavelength)")

    table = hyperparameter_table(layer)
    print(
        "joint log-likelihood: "
        f"mean={table['joint_log_marginal_likelihood'].mean():.2f}, "
        f"min={table['joint_log_marginal_likelihood'].min():.2f}"
    )

    table.to_csv(OUTPUT_DIR / "joint_mf_gp_hyperparameters.csv", index=False)
    joblib.dump(layer, OUTPUT_DIR / "joint_mf_gp_layer.joblib")
    plot_diagnostics(table, OUTPUT_DIR / "01_joint_mf_gp_diagnostics.png")
    (OUTPUT_DIR / "timing_summary.json").write_text(
        json.dumps(
            {
                "n_wavelengths": len(wavelengths),
                "n_lf_rows_available": len(XLF_10k),
                "n_lf_subsample": SUBSAMPLE_SIZE,
                "n_hf_rows": len(XHF),
                "total_seconds": round(elapsed, 1),
                "per_wavelength_mean_seconds": round(float(layer.fit_seconds.mean()), 3),
                "per_wavelength_min_seconds": round(float(layer.fit_seconds.min()), 3),
                "per_wavelength_max_seconds": round(float(layer.fit_seconds.max()), 3),
                "seed": SEED,
            },
            indent=2,
        )
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
