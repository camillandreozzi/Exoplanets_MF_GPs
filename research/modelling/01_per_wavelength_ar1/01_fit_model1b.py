"""Model 1B: per-wavelength joint AR(1) MF-GPs, one free rho_j per wavelength.

195 jointly fitted multi-fidelity GPs on a maximized LF subsample; each
wavelength's rho_j, kernels, and noise levels are optimized inside that
wavelength's own marginal likelihood (see exoplanets_mf.mf_gp).

The exact fit on all 10,097 points is infeasible on this machine (~40.8 GB
per optimizer iteration vs 16 GB RAM; ~1450 h extrapolated even with the
memory -- see 00_benchmark_exact_fit.py). The LF subsample is therefore
maximized against a practical time budget instead: memory would allow
n ~ 5800, but time binds. Wall time is recorded per wavelength.

Both fidelity scales are fit from one run: the linear spectra and the
log10-transformed spectra (a mirror fit whose fitted quantities live in log10
space and whose predictions must be back-transformed -- see the README). Each
scale writes to its own results root; this consolidates the former
research/log_modelling/ tree.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import fit_joint_mf_gp, hyperparameter_table
from exoplanets_mf.paths import LOG_MODELLING_RESULTS_DIR, MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

# The one canonical LF subsample, shared by every model and CV (see
# reproducibility.LF_SUBSAMPLE_SIZE). NOTE: 02_fit_model1a.py warm-starts from
# this layer and inherits the SAME size and seed automatically.
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
        " (log10 spectra)",
    ),
)


def plot_diagnostics(table, savepath, title_suffix="") -> None:
    n_panels = 4 if "fit_seconds" in table else 3
    fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    ax[0].plot(table["wavelength"], table["rho"], ".-")
    ax[0].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel=r"$\rho_j$",
        title=f"Model 1B jointly fitted scaling{title_suffix}",
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
        title=f"Joint covariance components{title_suffix}",
    )
    ax[1].legend()
    ax[2].plot(
        table["wavelength"], table["joint_log_marginal_likelihood"], ".-"
    )
    ax[2].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="log marginal likelihood",
        title=f"Joint LF/HF fit{title_suffix}",
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


def run_pass(data, results_root, y_transform, phrase, title_suffix) -> None:
    output_dir = results_root / "01_per_wavelength_ar1" / "model_1b"
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
        f"Fitting Model 1B (per-wavelength rho MF-GPs{phrase}): "
        f"{len(wavelengths)} wavelengths, {SUBSAMPLE_SIZE} of {len(XLF_10k)} "
        f"LF + {len(XHF)} HF rows per fit"
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

    table.to_csv(output_dir / "model_1b_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model_1b_layer.joblib")
    plot_diagnostics(table, output_dir / "01_model1b_diagnostics.png", title_suffix)
    (output_dir / "timing_summary.json").write_text(
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
    print(f"outputs written to {output_dir}")


def main():
    data = load_all()
    for _name, results_root, y_transform, phrase, title_suffix in SCALES:
        run_pass(data, results_root, y_transform, phrase, title_suffix)


if __name__ == "__main__":
    main()
