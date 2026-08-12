"""Model 1, GPBoost variant: per-wavelength AR(1) MF-GPs via GPBoost.

The same per-wavelength joint AR(1) multi-fidelity model as Model 1B, but fitted
with GPBoost's native ``ar1_mf_gaussian_ard`` covariance instead of the sklearn
``AR1MultiFidelityKernel`` (see exoplanets_mf.gpboost_mf). Linear eclipse depth,
all 195 wavelengths, exact inference (no Vecchia), on the same project-wide LF
subsample the sklearn Model 1B production fit uses, so the fitted rho and
hyperparameters are on comparable footing.

The paired GPBoost model comparison lives in
``research/modelling/03_gpboost_comparison``. This script only produces the
Model 1 GPBoost variant's own fit, hyperparameters, timing, and 5-fold CV
metrics.

Skips cleanly (no error) when gpboost is unavailable.
"""

import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt

from exoplanets_mf.cv import cv_metrics
from exoplanets_mf.data import load_all
from exoplanets_mf.gpboost_mf import (
    GPBOOST_COV_FUNCTION,
    cv_predict_model1_gpboost,
    fit_model1_gpboost,
    gpboost_available,
    hyperparameter_table_model1_gpboost,
    predict_hf_model1_gpboost,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR, approximation_suffix
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

OUTPUT_ROOT = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1"
OUTPUT_STEM = "model1_gpboost"

# The one canonical LF subsample shared by every model and CV (see
# reproducibility.LF_SUBSAMPLE_SIZE); mirrors the sklearn Model 1B scripts.
SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE      # production fit (mirrors 01_fit_model1b.py)
CV_SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE   # CV (mirrors 03_compare_model1a_vs_1b.py)
SEED = RANDOM_SEED


def plot_diagnostics(table, savepath) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].axhline(0.0, color="gray", lw=0.6)
    ax[0].plot(table["wavelength"], table["rho"], ".-")
    ax[0].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel=r"$\rho_j$",
        title="GPBoost Model 1 fitted scaling (rho may be < 0)",
    )
    ax[1].plot(table["wavelength"], table["low_signal_variance"], ".-", label="LF")
    ax[1].plot(
        table["wavelength"], table["delta_signal_variance"], ".-", label="discrepancy"
    )
    ax[1].plot(table["wavelength"], table["error_var"], ".-", label="error (nugget)")
    ax[1].set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="variance (standardized y)",
        title="GPBoost covariance components",
    )
    ax[1].set_yscale("log")
    ax[1].legend()
    if "fit_seconds" in table:
        ax[2].plot(table["wavelength"], table["fit_seconds"], ".-")
        ax[2].set(
            xlabel=r"wavelength $\lambda$ [$\mu$m]",
            ylabel="fit wall time [s]",
            title="Per-wavelength fit time",
        )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)




GP_APPROX_CHOICES = (
    "none",
    "vecchia",
    "vecchia_euclidean",
    "full_scale_vecchia",
    "fitc",
    "tapering",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gp-approx",
        choices=GP_APPROX_CHOICES,
        default="none",
        help=(
            "GPBoost large-data approximation (default: exact inference). "
            "A non-default choice writes to its own output directory, e.g. "
            "model1_gpboost_vecchia_k30."
        ),
    )
    parser.add_argument(
        "--num-neighbors",
        default=None,
        help=(
            "Vecchia neighbours: an integer, or 'none' for GPBoost's internal "
            "default. Ignored when --gp-approx is 'none'."
        ),
    )
    args = parser.parse_args()
    if args.gp_approx == "none":
        args.num_neighbors = None
    elif args.num_neighbors is not None:
        if str(args.num_neighbors).strip().lower() in {"none", "null", "default"}:
            args.num_neighbors = None
        else:
            args.num_neighbors = int(args.num_neighbors)
    return args


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / (
        OUTPUT_STEM + approximation_suffix(args.gp_approx, args.num_neighbors)
    )
    if not gpboost_available():
        print("gpboost not available -- skipping Model 1 GPBoost variant.")
        print("  (install gpboost and run tools/fix_gpboost_libomp.py; see README)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    wavelengths = data["wavelengths"]

    print(
        f"Fitting Model 1 (GPBoost {GPBOOST_COV_FUNCTION}): {len(wavelengths)} "
        f"wavelengths, {SUBSAMPLE_SIZE} of {len(XLF_10k)} LF + {len(XHF)} HF rows"
    )
    t0 = time.perf_counter()
    layer = fit_model1_gpboost(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        gp_approx=args.gp_approx,
        num_neighbors=args.num_neighbors,
        progress_every=40,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"fit in {elapsed:.1f}s "
        f"({elapsed / len(wavelengths):.3f}s/wavelength); "
        f"rho mean={layer.rho.mean():.3f}, "
        f"n(rho<0)={(layer.rho < 0).sum()}"
    )

    table = hyperparameter_table_model1_gpboost(layer)
    table.to_csv(output_dir / "model1_gpboost_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model1_gpboost_layer.joblib")
    plot_diagnostics(table, output_dir / "04_model1_gpboost_diagnostics.png")

    print(f"5-fold CV ({CV_SUBSAMPLE_SIZE} LF testing config) ...")
    t0 = time.perf_counter()
    predictions = cv_predict_model1_gpboost(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        subsample_size=CV_SUBSAMPLE_SIZE,
        gp_approx=args.gp_approx,
        num_neighbors=args.num_neighbors,
    )
    cv_seconds = time.perf_counter() - t0
    metrics = cv_metrics(YHF, predictions)
    print(
        f"CV in {cv_seconds:.1f}s: pooled RMSE={metrics.rmse_pooled:.6g}, "
        f"NRMSE={metrics.nrmse_pooled:.4f}, coverage95={metrics.coverage_95:.3f}"
    )

    (output_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "cov_function": layer.cov_function,
                "gp_approx": args.gp_approx,
                "num_neighbors": args.num_neighbors,
                "n_wavelengths": int(len(wavelengths)),
                "n_samples": int(YHF.shape[0]),
                "n_splits": predictions.n_splits,
                "fit_subsample_size": SUBSAMPLE_SIZE,
                "cv_subsample_size": CV_SUBSAMPLE_SIZE,
                "seed": SEED,
                "rho_mean": float(layer.rho.mean()),
                "rho_min": float(layer.rho.min()),
                "rho_max": float(layer.rho.max()),
                "n_rho_negative": int((layer.rho < 0).sum()),
                "rmse_pooled": metrics.rmse_pooled,
                "nrmse_pooled": metrics.nrmse_pooled,
                "coverage_95": metrics.coverage_95,
                "fit_total_seconds": round(elapsed, 1),
                "cv_total_seconds": round(cv_seconds, 1),
            },
            indent=2,
        )
    )
    print(f"outputs written to {output_dir}")


if __name__ == "__main__":
    main()
