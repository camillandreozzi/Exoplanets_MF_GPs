"""Model 2, GPBoost variant: wavelength-augmented AR(1) MF-GP via GPBoost.

The same wavelength-augmented joint AR(1) multi-fidelity model as Model 2, but
fitted with GPBoost's native ``ar1_mf_gaussian_ard`` covariance instead of the
sklearn ``AR1MultiFidelityKernel`` at n_dims=10 (see
exoplanets_mf.gpboost_model2). Linear eclipse depth, exact inference (no
Vecchia), on the same subsampled augmented design the sklearn Model 2 uses.

The paired GPBoost model comparison lives in
``research/modelling/03_gpboost_comparison``. This script produces only the
Model 2 GPBoost variant's own fit, hyperparameters, timing, and 5-fold CV
metrics. Skips cleanly when gpboost is unavailable.
"""

import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np

from exoplanets_mf.cv import cv_metrics
from exoplanets_mf.data import load_all
from exoplanets_mf.gpboost_model2 import (
    cv_predict_model2_gpboost,
    fit_model2_gpboost,
    hyperparameter_table_model2_gpboost,
    predict_hf_model2_gpboost,
)
from exoplanets_mf.gpboost_mf import GPBOOST_COV_FUNCTION, gpboost_available
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.model2 import (
    MODEL2_CV_MAX_AUGMENTED_POINTS,
    MODEL2_MAX_AUGMENTED_POINTS,
    derive_lambda_stride,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR, approximation_suffix
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

OUTPUT_ROOT = MODELLING_RESULTS_DIR / "02_augmented_wavelength"
OUTPUT_STEM = "model2_gpboost"

SEED = RANDOM_SEED
# Match the sklearn Model 2 configuration (01_fit_model2.py / 02_cv_*): the one
# canonical LF subsample (see reproducibility.LF_SUBSAMPLE_SIZE), with the
# wavelength stride derived from the runtime budget in main().
LF_SAMPLE_SIZE = LF_SUBSAMPLE_SIZE
CV_LF_SAMPLE_SIZE = LF_SUBSAMPLE_SIZE


def plot_diagnostics(layer, X_hf, Y_hf, input_names, savepath) -> None:
    cov = layer.model.cov_pars()
    n_dims = len(input_names)
    low = np.array([cov[f"low_GP_range_{i + 1}"] for i in range(n_dims)])
    delta = np.array([cov[f"discrepancy_GP_range_{i + 1}"] for i in range(n_dims)])
    wavelengths = layer.wavelengths

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(n_dims)
    axes[0].bar(positions - 0.2, low, width=0.4, label="LF kernel", color="tab:blue")
    axes[0].bar(
        positions + 0.2, delta, width=0.4, label="discrepancy kernel", color="tab:orange"
    )
    axes[0].set_xticks(positions, input_names, rotation=45)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="GPBoost range (standardized units)", title="Fitted ranges")
    axes[0].legend()

    ax = axes[1]
    for mode, mask in instrument_mode_masks(wavelengths):
        span = wavelengths[mask]
        ax.axvspan(span.min(), span.max(), color=mode.color, alpha=0.4, zorder=0)
    means, _ = predict_hf_model2_gpboost(layer, X_hf)
    residual_rms = np.sqrt(((means - Y_hf) ** 2).mean(axis=0))
    trained = np.zeros(len(wavelengths), dtype=bool)
    trained[layer.hf_lambda_indices] = True
    ax.plot(wavelengths, residual_rms, "-", color="gray", lw=0.8, zorder=1)
    ax.plot(wavelengths[trained], residual_rms[trained], "o", ms=4,
            color="tab:blue", label="HF-trained wavelength")
    ax.plot(wavelengths[~trained], residual_rms[~trained], ".", ms=4,
            color="tab:red", label="interpolated wavelength")
    ax.set_yscale("log")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="in-sample HF residual RMS",
        title=f"GPBoost Model 2 (scalar rho = {layer.rho:.3f})",
    )
    ax.legend()
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
            "model2_gpboost_vecchia_k30."
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
        print("gpboost not available -- skipping Model 2 GPBoost variant.")
        print("  (install gpboost and run tools/fix_gpboost_libomp.py; see README)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    input_names = list(data["XHF"].columns) + ["lambda"]

    # Finest wavelength grid affordable for the shared LF subsample size, at the
    # production budget (single fit) and the tighter CV budget (per fold).
    lambda_stride = derive_lambda_stride(
        LF_SAMPLE_SIZE, len(XHF), len(wavelengths),
        max_points=MODEL2_MAX_AUGMENTED_POINTS,
    )
    cv_lambda_stride = derive_lambda_stride(
        CV_LF_SAMPLE_SIZE, len(XHF), len(wavelengths),
        max_points=MODEL2_CV_MAX_AUGMENTED_POINTS,
    )

    print(
        f"Fitting Model 2 (GPBoost {GPBOOST_COV_FUNCTION}): {LF_SAMPLE_SIZE} of "
        f"{len(XLF_10k)} LF samples (stride {lambda_stride}) + {len(XHF)} HF"
    )
    t0 = time.perf_counter()
    layer = fit_model2_gpboost(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        lf_sample_size=LF_SAMPLE_SIZE,
        lambda_stride=lambda_stride,
        gp_approx=args.gp_approx,
        num_neighbors=args.num_neighbors,
    )
    elapsed = time.perf_counter() - t0
    n_points = layer.model.n_points
    print(
        f"fitted n={n_points} points in {elapsed:.1f}s: scalar rho={layer.rho:.3f}"
    )

    table = hyperparameter_table_model2_gpboost(
        layer, n_points=n_points, lambda_stride=lambda_stride
    )
    table.to_csv(output_dir / "model2_gpboost_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model2_gpboost_layer.joblib")
    plot_diagnostics(
        layer, XHF, YHF, input_names, output_dir / "03_model2_gpboost_diagnostics.png"
    )

    print(f"5-fold CV (stride {cv_lambda_stride}, {CV_LF_SAMPLE_SIZE} LF) ...")
    t0 = time.perf_counter()
    predictions = cv_predict_model2_gpboost(
        XLF_10k,
        YLF_10k,
        XHF,
        YHF,
        wavelengths,
        seed=SEED,
        lf_sample_size=CV_LF_SAMPLE_SIZE,
        lambda_stride=cv_lambda_stride,
        gp_approx=args.gp_approx,
        num_neighbors=args.num_neighbors,
        progress=True,
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
                "n_samples": int(YHF.shape[0]),
                "n_splits": predictions.n_splits,
                "fit_n_points": int(n_points),
                "fit_lf_sample_size": LF_SAMPLE_SIZE,
                "fit_lambda_stride": lambda_stride,
                "cv_lf_sample_size": CV_LF_SAMPLE_SIZE,
                "cv_lambda_stride": cv_lambda_stride,
                "seed": SEED,
                "rho": float(layer.rho),
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
