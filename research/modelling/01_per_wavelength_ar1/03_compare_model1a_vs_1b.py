"""Compare Model 1A (global rho) vs 1B (per-wavelength rho) by cross-validation.

Both models are per-wavelength joint AR(1) multi-fidelity GPs; they differ
ONLY in the rho criterion (one shared rho against the summed marginal
likelihood vs one free rho_j per wavelength). 5-fold CV over the 97 HF
samples with the identical fold assignment: every fold refits both full GP
layers from scratch and predicts the held-out HF spectra from atmospheric
inputs alone (no LF spectrum is consumed at test time).

The CV refits run at a reduced TESTING LF-subsample size because 2 models x
5 folds x 195 GPs is refit from scratch; the production fits
(01_fit_model1b.py / 02_fit_model1a.py) are the reference layers, and their
fitted rho values are what the diagnostic figure shows.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import (
    CV_FULL_MODEL_SPLITS,
    compare_cv,
    cv_predict_joint_mf_gp,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

EXPERIMENT_DIR = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1"
OUTPUT_DIR = EXPERIMENT_DIR / "comparison"

N_SPLITS = CV_FULL_MODEL_SPLITS  # every fold refits both full GP layers
SEED = RANDOM_SEED
# TESTING config for the CV refits (~2 s/wavelength at 200 LF + <97 HF);
# the production layers use SUBSAMPLE_SIZE in the fit scripts.
CV_SUBSAMPLE_SIZE = 200

MODEL_A_LABEL = "Model 1A (global rho MF-GP)"
MODEL_B_LABEL = "Model 1B (per-wavelength rho MF-GP)"


def shade_instrument_modes(ax, wavelengths) -> None:
    for mode, mask in instrument_mode_masks(wavelengths):
        wavelength_range = wavelengths[mask]
        ax.axvspan(
            wavelength_range.min(),
            wavelength_range.max(),
            color=mode.color,
            alpha=0.4,
            zorder=0,
        )


def load_production_rho():
    """Fitted rho of the production layers, for reporting/plots only."""
    rho_b = pd.read_csv(
        EXPERIMENT_DIR / "model_1b" / "model_1b_hyperparameters.csv"
    )["rho"].to_numpy()
    rho_a = float(
        pd.read_csv(
            EXPERIMENT_DIR / "model_1a" / "model_1a_hyperparameters.csv"
        )["rho"].iloc[0]
    )
    return rho_a, rho_b


def plot_comparison(
    wavelengths, YHF, comparison, rho_a, rho_b, savepath,
    predictions_a, predictions_b,
) -> None:
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, rho_b, ".-", color="tab:blue", lw=1,
            label=r"$\rho_j$ (Model 1B)")
    ax.axhline(rho_a, color="tab:red", ls="--",
               label=rf"shared $\rho$ = {rho_a:.3f} (Model 1A)")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel=r"$\rho$",
           title="Jointly fitted scaling (production layers)")
    ax.legend()

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, metrics_a.rmse_per_wavelength, lw=1.2,
            color="tab:red", label=MODEL_A_LABEL)
    ax.plot(wavelengths, metrics_b.rmse_per_wavelength, lw=1.2,
            color="tab:blue", label=MODEL_B_LABEL)
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="held-out RMSE",
           title="Per-wavelength CV RMSE")
    ax.legend()

    ax = axes[1, 0]
    order = np.argsort(comparison.delta_rmse_per_sample)
    colors = np.where(
        comparison.delta_rmse_per_sample[order] < 0, "tab:blue", "tab:red"
    )
    ax.bar(np.arange(len(order)), comparison.delta_rmse_per_sample[order],
           color=colors, width=1.0)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel="sample (sorted)", ylabel="RMSE(B) - RMSE(A)",
           title=(
               "Per-sample paired delta "
               f"(B wins on {comparison.fraction_samples_model2_wins:.0%})"
           ))

    ax = axes[1, 1]
    ax.scatter(YHF.ravel(), predictions_a.y_pred.ravel(), s=2, alpha=0.1,
               color="tab:red", label=MODEL_A_LABEL)
    ax.scatter(YHF.ravel(), predictions_b.y_pred.ravel(), s=2, alpha=0.1,
               color="tab:blue", label=MODEL_B_LABEL)
    limits = [YHF.min(), YHF.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(xlabel="actual HF eclipse depth", ylabel="held-out prediction",
           title="Out-of-fold predictions vs actual")
    ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()

    print(
        f"{N_SPLITS}-fold CV of Model 1A vs 1B MF-GP layers over all "
        f"{YHF.shape[0]} HF samples ({CV_SUBSAMPLE_SIZE} LF testing config, "
        f"seed={SEED})"
    )
    print("Model 1A (global rho) CV ...", flush=True)
    predictions_a = cv_predict_joint_mf_gp(
        XLF_10k, YLF_10k, XHF, YHF, wavelengths,
        per_wavelength_rho=False,
        subsample_size=CV_SUBSAMPLE_SIZE,
        n_splits=N_SPLITS,
        seed=SEED,
    )
    print("Model 1B (per-wavelength rho) CV ...", flush=True)
    predictions_b = cv_predict_joint_mf_gp(
        XLF_10k, YLF_10k, XHF, YHF, wavelengths,
        per_wavelength_rho=True,
        subsample_size=CV_SUBSAMPLE_SIZE,
        n_splits=N_SPLITS,
        seed=SEED,
    )
    if not np.array_equal(
        predictions_a.fold_of_sample, predictions_b.fold_of_sample
    ):
        raise RuntimeError("fold assignments do not match between models")

    comparison = compare_cv(
        YHF, predictions_a, predictions_b, labels=(MODEL_A_LABEL, MODEL_B_LABEL)
    )
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2

    rho_a, rho_b = load_production_rho()

    print(
        f"pooled held-out RMSE: A={metrics_a.rmse_pooled:.4e}, "
        f"B={metrics_b.rmse_pooled:.4e} "
        f"(delta={comparison.delta_rmse_pooled:+.4e})"
    )
    print(
        f"per-sample delta RMSE (B - A): mean={comparison.delta_mean:+.4e} "
        f"+/- {comparison.delta_std:.4e}; "
        f"B wins on {comparison.fraction_samples_model2_wins:.1%} of samples"
    )
    print(
        f"95% CI coverage: A={metrics_a.coverage_95:.1%}, "
        f"B={metrics_b.coverage_95:.1%}"
    )

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rho_b_production": rho_b,
            "rmse_a": metrics_a.rmse_per_wavelength,
            "rmse_b": metrics_b.rmse_per_wavelength,
            "delta_rmse": comparison.delta_rmse_per_wavelength,
        }
    ).to_csv(OUTPUT_DIR / "cv_per_wavelength.csv", index=False)

    pd.DataFrame(
        {
            "sample": np.arange(YHF.shape[0]),
            "fold": predictions_a.fold_of_sample,
            "rmse_a": metrics_a.rmse_per_sample,
            "rmse_b": metrics_b.rmse_per_sample,
            "delta_rmse": comparison.delta_rmse_per_sample,
        }
    ).to_csv(OUTPUT_DIR / "cv_per_sample.csv", index=False)

    (OUTPUT_DIR / "cv_summary.json").write_text(
        json.dumps(
            {
                "model_a": MODEL_A_LABEL,
                "model_b": MODEL_B_LABEL,
                "model_family": "per-wavelength joint AR(1) MF-GP",
                "n_samples": int(YHF.shape[0]),
                "n_splits": N_SPLITS,
                "seed": SEED,
                "cv_lf_subsample_size": CV_SUBSAMPLE_SIZE,
                "rho_global_production": rho_a,
                "rmse_pooled_a": metrics_a.rmse_pooled,
                "rmse_pooled_b": metrics_b.rmse_pooled,
                "nrmse_pooled_a": metrics_a.nrmse_pooled,
                "nrmse_pooled_b": metrics_b.nrmse_pooled,
                "coverage_95_a": metrics_a.coverage_95,
                "coverage_95_b": metrics_b.coverage_95,
                "delta_rmse_pooled": comparison.delta_rmse_pooled,
                "delta_rmse_per_sample_mean": comparison.delta_mean,
                "delta_rmse_per_sample_std": comparison.delta_std,
                "fraction_samples_b_wins": comparison.fraction_samples_model2_wins,
            },
            indent=2,
        )
    )

    plot_comparison(
        wavelengths,
        YHF,
        comparison,
        rho_a,
        rho_b,
        OUTPUT_DIR / "03_model_comparison.png",
        predictions_a,
        predictions_b,
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
