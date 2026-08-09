"""Model 1A: per-wavelength joint AR(1) MF-GPs sharing ONE global rho.

Identical structure and LF/HF design as Model 1B (01_fit_model1b.py); the
only difference is the rho criterion: a single scalar rho shared by all 195
wavelengths, optimized against the summed (i.e. joint) log marginal
likelihood by block-coordinate ascent (see
exoplanets_mf.mf_gp.fit_joint_mf_gp_global_rho).

The fit warm-starts from the saved Model 1B layer when it exists -- SAME
SUBSAMPLE_SIZE and seed required, enforced by the fitter -- which only
changes the starting point, not the criterion.

Both fidelity scales are fit from one run (linear and log10 spectra), each
warm-starting from and writing to its own results root; the log10 fit lives
in log10 space (see the README for back-transforming predictions).
"""

import json
import time

import joblib
import matplotlib.pyplot as plt

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import (
    fit_joint_mf_gp_global_rho,
    hyperparameter_table,
)
from exoplanets_mf.paths import LOG_MODELLING_RESULTS_DIR, MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

# The one canonical LF subsample shared by every model (see
# reproducibility.LF_SUBSAMPLE_SIZE); matches 01_fit_model1b.py for the shared
# subsample and the warm start.
SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE
SEED = RANDOM_SEED

# (name, results root, y transform, print phrase, plot-title suffix).
SCALES = (
    ("linear", MODELLING_RESULTS_DIR, None, "", ""),
    (
        "log10",
        LOG_MODELLING_RESULTS_DIR,
        log10_spectra,
        ", log10 spectra",
        ", log10 spectra",
    ),
)


def load_warm_start_layer(results_root):
    layer_path = (
        results_root / "01_per_wavelength_ar1" / "model_1b" / "model_1b_layer.joblib"
    )
    if layer_path.exists():
        print(f"warm-starting from {layer_path}")
        return joblib.load(layer_path)
    print("no saved Model 1B layer found -- cold start")
    return None


def plot_diagnostics(table, sweeps, savepath, title_suffix="") -> None:
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    ax[0].plot(
        [sweep["sweep"] for sweep in sweeps],
        [sweep["rho_after"] for sweep in sweeps],
        "o-",
    )
    ax[0].set(
        xlabel="block-coordinate sweep",
        ylabel=r"shared $\rho$",
        title=(
            f"Model 1A global scaling{title_suffix} "
            f"(final = {table['rho'].iloc[0]:.4f})"
        ),
    )
    ax[1].plot(
        [sweep["sweep"] for sweep in sweeps],
        [sweep["summed_log_marginal_likelihood"] for sweep in sweeps],
        "o-",
    )
    ax[1].set(
        xlabel="block-coordinate sweep",
        ylabel="summed log marginal likelihood",
        title="Joint likelihood ascent",
    )
    ax[2].plot(
        table["wavelength"], table["low_signal_variance"], ".-", label="LF"
    )
    ax[2].plot(
        table["wavelength"],
        table["delta_signal_variance"],
        ".-",
        label="discrepancy",
    )
    ax[2].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="signal variance",
        title=f"Joint covariance components{title_suffix}",
    )
    ax[2].legend()
    ax[3].plot(
        table["wavelength"], table["joint_log_marginal_likelihood"], ".-"
    )
    ax[3].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="log marginal likelihood",
        title="Per-wavelength LF/HF fit at the shared rho",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pass(data, results_root, y_transform, phrase, title_suffix) -> None:
    output_dir = results_root / "01_per_wavelength_ar1" / "model_1a"
    output_dir.mkdir(parents=True, exist_ok=True)
    XLF_10k = data["XLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    YHF = data["YHF"].to_numpy()
    if y_transform is not None:
        YLF_10k = y_transform(YLF_10k)
        YHF = y_transform(YHF)
    wavelengths = data["wavelengths"]

    print(
        f"Fitting Model 1A (global-rho MF-GPs{phrase}): {len(wavelengths)} "
        f"wavelengths, {SUBSAMPLE_SIZE} of {len(XLF_10k)} LF + {len(XHF)} HF "
        "rows per fit, one shared rho"
    )
    t0 = time.perf_counter()
    layer = fit_joint_mf_gp_global_rho(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        warm_start_layer=load_warm_start_layer(results_root),
        progress_every=40,
    )
    elapsed = time.perf_counter() - t0
    sweeps = layer.global_rho_sweeps
    print(
        f"fit in {elapsed:.1f}s over {len(sweeps)} sweeps; "
        f"shared rho = {layer.rho[0]:.6f}"
    )

    table = hyperparameter_table(layer)
    print(
        "summed joint log-likelihood: "
        f"{table['joint_log_marginal_likelihood'].sum():.2f}"
    )

    table.to_csv(output_dir / "model_1a_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model_1a_layer.joblib")
    plot_diagnostics(table, sweeps, output_dir / "02_model1a_diagnostics.png", title_suffix)
    (output_dir / "timing_summary.json").write_text(
        json.dumps(
            {
                "n_wavelengths": len(wavelengths),
                "n_lf_rows_available": len(XLF_10k),
                "n_lf_subsample": SUBSAMPLE_SIZE,
                "n_hf_rows": len(XHF),
                "total_seconds": round(elapsed, 1),
                "n_sweeps": len(sweeps),
                "shared_rho": float(layer.rho[0]),
                "sweeps": sweeps,
                "seed": SEED,
            },
            indent=2,
        )
    )
    print(f"outputs written to {output_dir}")


def main():
    data = load_all()
    for _name, results_root, y_transform, phrase, title_suffix in SCALES:
        run_pass(data, results_root, y_transform, phrase, title_suffix)


if __name__ == "__main__":
    main()
