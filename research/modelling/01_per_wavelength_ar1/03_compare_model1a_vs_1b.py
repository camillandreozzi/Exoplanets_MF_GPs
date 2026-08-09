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

The comparison runs on both fidelity scales from one invocation: the linear
spectra and the log10-transformed spectra (on the log scale an RMSE of 0.1
means the prediction is off by a factor 10**0.1 ~ 1.26 in linear units).
Each scale writes to its own results root; this consolidates the former
research/log_modelling/ tree.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.cv import (
    CV_FULL_MODEL_SPLITS,
    compare_cv,
    cv_metrics,
    cv_predict_hf_only_gp,
    cv_predict_joint_mf_gp,
)
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.paths import LOG_MODELLING_RESULTS_DIR, MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

N_SPLITS = CV_FULL_MODEL_SPLITS  # every fold refits both full GP layers
SEED = RANDOM_SEED
# The CV refits use the one canonical LF subsample -- identical rows to the
# production fit scripts (see reproducibility.LF_SUBSAMPLE_SIZE).
CV_SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE

# Per-scale labels and axis text. Both scales run from one invocation.
SCALES = (
    {
        "root": MODELLING_RESULTS_DIR,
        "transform": None,
        "model_a": "Model 1A (global rho MF-GP)",
        "model_b": "Model 1B (per-wavelength rho MF-GP)",
        "hf_only": "HF-only GP (single-fidelity baseline)",
        "print_note": "",
        "pooled_note": "",
        "output_units": "original eclipse-depth units",
        "scaling_title": "Jointly fitted scaling (production layers)",
        "rmse_ylabel": "held-out RMSE",
        "rmse_title": "Per-wavelength CV RMSE",
        "delta_ylabel": "RMSE(B) - RMSE(A)",
        "actual_xlabel": "actual HF eclipse depth",
        "pred_title": "Out-of-fold predictions vs actual",
    },
    {
        "root": LOG_MODELLING_RESULTS_DIR,
        "transform": log10_spectra,
        "model_a": "Model 1A (global rho MF-GP, log10)",
        "model_b": "Model 1B (per-wavelength rho MF-GP, log10)",
        "hf_only": "HF-only GP (single-fidelity baseline, log10)",
        "print_note": " (log10 spectra)",
        "pooled_note": " (log10)",
        "output_units": "log10 eclipse depth",
        "scaling_title": "Jointly fitted scaling, log10 spectra (production layers)",
        "rmse_ylabel": "held-out RMSE [log10]",
        "rmse_title": "Per-wavelength CV RMSE (log10 space)",
        "delta_ylabel": "RMSE(B) - RMSE(A) [log10]",
        "actual_xlabel": "actual HF log10 eclipse depth",
        "pred_title": "Out-of-fold predictions vs actual (log10 space)",
    },
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


def load_production_rho(experiment_dir):
    """Fitted rho of the production layers, for reporting/plots only."""
    rho_b = pd.read_csv(
        experiment_dir / "model_1b" / "model_1b_hyperparameters.csv"
    )["rho"].to_numpy()
    rho_a = float(
        pd.read_csv(
            experiment_dir / "model_1a" / "model_1a_hyperparameters.csv"
        )["rho"].iloc[0]
    )
    return rho_a, rho_b


def plot_comparison(
    wavelengths, YHF, comparison, rho_a, rho_b, savepath,
    predictions_a, predictions_b, metrics_hf, cfg,
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
           title=cfg["scaling_title"])
    ax.legend()

    ax = axes[0, 1]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(wavelengths, metrics_hf.rmse_per_wavelength, lw=1.2,
            color="tab:gray", label=cfg["hf_only"])
    ax.plot(wavelengths, metrics_a.rmse_per_wavelength, lw=1.2,
            color="tab:red", label=cfg["model_a"])
    ax.plot(wavelengths, metrics_b.rmse_per_wavelength, lw=1.2,
            color="tab:blue", label=cfg["model_b"])
    ax.set_yscale("log")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel=cfg["rmse_ylabel"],
           title=cfg["rmse_title"])
    ax.legend()

    ax = axes[1, 0]
    order = np.argsort(comparison.delta_rmse_per_sample)
    colors = np.where(
        comparison.delta_rmse_per_sample[order] < 0, "tab:blue", "tab:red"
    )
    ax.bar(np.arange(len(order)), comparison.delta_rmse_per_sample[order],
           color=colors, width=1.0)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel="sample (sorted)", ylabel=cfg["delta_ylabel"],
           title=(
               "Per-sample paired delta "
               f"(B wins on {comparison.fraction_samples_model2_wins:.0%})"
           ))

    ax = axes[1, 1]
    ax.scatter(YHF.ravel(), predictions_a.y_pred.ravel(), s=2, alpha=0.1,
               color="tab:red", label=cfg["model_a"])
    ax.scatter(YHF.ravel(), predictions_b.y_pred.ravel(), s=2, alpha=0.1,
               color="tab:blue", label=cfg["model_b"])
    limits = [YHF.min(), YHF.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(xlabel=cfg["actual_xlabel"], ylabel="held-out prediction",
           title=cfg["pred_title"])
    ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pass(data, cfg) -> None:
    experiment_dir = cfg["root"] / "01_per_wavelength_ar1"
    output_dir = experiment_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = data["wavelengths"]
    XLF_10k = data["XLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    YHF = data["YHF"].to_numpy()
    if cfg["transform"] is not None:
        YLF_10k = cfg["transform"](YLF_10k)
        YHF = cfg["transform"](YHF)

    print(
        f"{N_SPLITS}-fold CV of Model 1A vs 1B MF-GP layers{cfg['print_note']} "
        f"over all {YHF.shape[0]} HF samples ({CV_SUBSAMPLE_SIZE} LF testing "
        f"config, seed={SEED})"
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
    print("HF-only GP (single-fidelity baseline) CV ...", flush=True)
    predictions_hf = cv_predict_hf_only_gp(
        XHF, YHF, wavelengths, n_splits=N_SPLITS, seed=SEED
    )
    if not (
        np.array_equal(predictions_a.fold_of_sample, predictions_b.fold_of_sample)
        and np.array_equal(predictions_a.fold_of_sample, predictions_hf.fold_of_sample)
    ):
        raise RuntimeError("fold assignments do not match between models")

    comparison = compare_cv(
        YHF, predictions_a, predictions_b, labels=(cfg["model_a"], cfg["model_b"])
    )
    metrics_a, metrics_b = comparison.metrics_1, comparison.metrics_2
    metrics_hf = cv_metrics(YHF, predictions_hf)

    rho_a, rho_b = load_production_rho(experiment_dir)

    print(
        f"pooled held-out RMSE{cfg['pooled_note']}: A={metrics_a.rmse_pooled:.4e}, "
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
            "rmse_hf_only": metrics_hf.rmse_per_wavelength,
            "rmse_a": metrics_a.rmse_per_wavelength,
            "rmse_b": metrics_b.rmse_per_wavelength,
            "delta_rmse": comparison.delta_rmse_per_wavelength,
        }
    ).to_csv(output_dir / "cv_per_wavelength.csv", index=False)

    pd.DataFrame(
        {
            "sample": np.arange(YHF.shape[0]),
            "fold": predictions_a.fold_of_sample,
            "rmse_a": metrics_a.rmse_per_sample,
            "rmse_b": metrics_b.rmse_per_sample,
            "delta_rmse": comparison.delta_rmse_per_sample,
        }
    ).to_csv(output_dir / "cv_per_sample.csv", index=False)

    (output_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "model_a": cfg["model_a"],
                "model_b": cfg["model_b"],
                "model_family": "per-wavelength joint AR(1) MF-GP",
                "output_units": cfg["output_units"],
                "n_samples": int(YHF.shape[0]),
                "n_splits": N_SPLITS,
                "seed": SEED,
                "cv_lf_subsample_size": CV_SUBSAMPLE_SIZE,
                "rho_global_production": rho_a,
                "hf_only_baseline_label": cfg["hf_only"],
                "rmse_pooled_hf_only": metrics_hf.rmse_pooled,
                "nrmse_pooled_hf_only": metrics_hf.nrmse_pooled,
                "coverage_95_hf_only": metrics_hf.coverage_95,
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
        output_dir / "03_model_comparison.png",
        predictions_a,
        predictions_b,
        metrics_hf,
        cfg,
    )
    print(f"outputs written to {output_dir}")


def main() -> None:
    data = load_all()
    for cfg in SCALES:
        run_pass(data, cfg)


if __name__ == "__main__":
    main()
