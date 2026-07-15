"""Model 1B: per-wavelength joint AR(1) MF-GPs, one free rho_j per wavelength.

195 jointly fitted multi-fidelity GPs on a maximized LF subsample; each
wavelength's rho_j, kernels, and noise levels are optimized inside that
wavelength's own marginal likelihood (see exoplanets_mf.mf_gp).

The exact fit on all 10,097 points is infeasible on this machine (~40.8 GB
per optimizer iteration vs 16 GB RAM; ~1450 h extrapolated even with the
memory -- see 00_benchmark_exact_fit.py). The LF subsample is therefore
maximized against a practical time budget instead: memory would allow
n ~ 5800, but time binds. Wall time is recorded per wavelength.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import fit_joint_mf_gp, hyperparameter_table
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

OUTPUT_DIR = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "model_1b"

# TESTING size for fast iteration: 400 LF + 97 HF measured ~7 s/wavelength,
# ~25 min for the full 195-wavelength run. For a production run, maximize
# against the measured budget instead: the benchmark recommends ~1500 LF for
# an 8 h run (1700 LF ran 100-316 s/wavelength, tracking ~10 h, when tried);
# see benchmark/exact_fit_benchmark.json (recommended_lf_subsample).
# NOTE: 02_fit_model1a.py warm-starts from this layer and must use the SAME
# SUBSAMPLE_SIZE and seed.
SUBSAMPLE_SIZE = 400
SEED = RANDOM_SEED


def plot_diagnostics(table, savepath) -> None:
    n_panels = 4 if "fit_seconds" in table else 3
    fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    ax[0].plot(table["wavelength"], table["rho"], ".-")
    ax[0].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel=r"$\rho_j$",
        title="Model 1B jointly fitted scaling",
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
        title="Joint covariance components",
    )
    ax[1].legend()
    ax[2].plot(
        table["wavelength"], table["joint_log_marginal_likelihood"], ".-"
    )
    ax[2].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="log marginal likelihood",
        title="Joint LF/HF fit",
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
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    wavelengths = data["wavelengths"]

    print(
        f"Fitting Model 1B (per-wavelength rho MF-GPs): {len(wavelengths)} "
        f"wavelengths, {SUBSAMPLE_SIZE} of {len(XLF_10k)} LF + {len(XHF)} HF "
        "rows per fit"
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

    table.to_csv(OUTPUT_DIR / "model_1b_hyperparameters.csv", index=False)
    joblib.dump(layer, OUTPUT_DIR / "model_1b_layer.joblib")
    plot_diagnostics(table, OUTPUT_DIR / "01_model1b_diagnostics.png")
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
