"""Compare Model 1A vs 1B by cross-validation on log10 spectra.

Log-transformed mirror of research/modelling/01_per_wavelength_ar1/
02_compare_rho_models.py: the paired YHF/YLF are log10-transformed after
loading, so rho and the held-out RMSEs live in log10 space (an RMSE of 0.1
means the prediction is off by a factor 10**0.1 ~ 1.26 in linear units).

Leave-one-out over all 97 paired HF samples: every spectrum is predicted from
a fit that never saw it, and the two models are scored on the same held-out
predictions. Purely closed-form (exoplanets_mf.cv rho adapters) -- this script
does not depend on the fitted joint GP layer and runs standalone.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import (
    CV_LOO_SPLITS,
    compare_cv,
    cv_predict_rho_model,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import global_rho, per_wavelength_rho
from exoplanets_mf.paths import LOG_MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

OUTPUT_DIR = LOG_MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "comparison"

N_SPLITS = CV_LOO_SPLITS
SEED = RANDOM_SEED

MODEL_A_LABEL = "Model 1A (global rho)"
MODEL_B_LABEL = "Model 1B (per-wavelength rho)"


def load_paired_spectra():
    data = load_all()
    # Row-paired CV requires input-matched LF -> the 97-row set, not the 10k
    # decoupled design.
    return (
        data["wavelengths"],
        log10_spectra(data["YHF"].to_numpy()),
        log10_spectra(data["YLF"].to_numpy()),
    )


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


def plot_comparison(
    wavelengths, YHF, comparison, rho_b, rho_a, savepath,
    predictions_a, predictions_b,
) -> None:
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, rho_b, ".-", color="tab:blue", lw=1,
            label=r"$\rho_j$ (Model 1B)")
    ax.axhline(rho_a, color="tab:red", ls="--",
               label=rf"global $\rho$ = {rho_a:.3f} (Model 1A)")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel=r"$\rho$",
           title="Fitted scaling coefficients")
    ax.legend()

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, metrics_a.rmse_per_wavelength, lw=1.2,
            color="tab:red", label=MODEL_A_LABEL)
    ax.plot(wavelengths, metrics_b.rmse_per_wavelength, lw=1.2,
            color="tab:blue", label=MODEL_B_LABEL)
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="held-out RMSE",
           title="Per-wavelength LOO RMSE")
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
    ax.set(xlabel="actual HF log10 eclipse depth", ylabel="held-out prediction",
           title="Out-of-fold predictions vs actual (log10)")
    ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wavelengths, YHF, YLF = load_paired_spectra()

    print(
        f"LOO cross-validation over all {YHF.shape[0]} HF samples "
        f"({N_SPLITS} splits, seed={SEED})"
    )
    predictions_a = cv_predict_rho_model(
        YHF, YLF, per_wavelength=False, n_splits=N_SPLITS, seed=SEED
    )
    predictions_b = cv_predict_rho_model(
        YHF, YLF, per_wavelength=True, n_splits=N_SPLITS, seed=SEED
    )
    comparison = compare_cv(
        YHF, predictions_a, predictions_b, labels=(MODEL_A_LABEL, MODEL_B_LABEL)
    )
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2

    # Full-data rho estimates for reporting/plots (the CV itself refits rho
    # inside every fold).
    rho_a = global_rho(YHF, YLF)
    rho_b = per_wavelength_rho(YHF, YLF)

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

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rho_b": rho_b,
            "rmse_a": metrics_a.rmse_per_wavelength,
            "rmse_b": metrics_b.rmse_per_wavelength,
            "delta_rmse": comparison.delta_rmse_per_wavelength,
        }
    ).to_csv(OUTPUT_DIR / "cv_per_wavelength.csv", index=False)

    pd.DataFrame(
        {
            "sample": np.arange(YHF.shape[0]),
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
                "n_samples": int(YHF.shape[0]),
                "n_splits": N_SPLITS,
                "seed": SEED,
                "rho_global": rho_a,
                "rmse_pooled_a": metrics_a.rmse_pooled,
                "rmse_pooled_b": metrics_b.rmse_pooled,
                "nrmse_pooled_a": metrics_a.nrmse_pooled,
                "nrmse_pooled_b": metrics_b.nrmse_pooled,
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
        rho_b,
        rho_a,
        OUTPUT_DIR / "02_model_comparison.png",
        predictions_a,
        predictions_b,
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
