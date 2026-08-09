"""Predict and score the held-out HF spectrum with Models 1A and 1B."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import predict_hf
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR

EXPERIMENT_DIR = VALIDATION_RESULTS_DIR / "01_loo_holdout_validation"
FIT_DIR = EXPERIMENT_DIR / "fit"
OUTPUT_DIR = EXPERIMENT_DIR / "evaluation"

HOLDOUT_ROW = 80
CI_Z = stats.norm.ppf(0.975)

MODELS = {
    "model_1a": "Model 1A (shared rho)",
    "model_1b": "Model 1B (per-wavelength rho)",
}


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


def evaluate(mean, std, actual, output_range) -> tuple[pd.DataFrame, dict]:
    residual = actual - mean
    standardized_residual = residual / std
    within_95ci = np.abs(standardized_residual) <= CI_Z
    nrmse = np.abs(residual) / output_range
    per_wavelength = pd.DataFrame(
        {
            "y_hf_pred": mean,
            "std": std,
            "residual": residual,
            "standardized_residual": standardized_residual,
            "within_95ci": within_95ci,
            "nrmse": nrmse,
        }
    )
    summary = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_standardized_residual": float(
            np.mean(np.abs(standardized_residual))
        ),
        "fraction_within_95ci": float(np.mean(within_95ci)),
        "mean_nrmse": float(np.mean(nrmse)),
        "mean_predictive_std": float(np.mean(std)),
    }
    return per_wavelength, summary


def plot_comparison(wavelengths, actual, predictions, savepath) -> None:
    figure, (ax_spec, ax_z) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1]
    )
    shade_instrument_modes(ax_spec, wavelengths)
    for color, (name, label) in zip(("C0", "C1"), MODELS.items()):
        mean, std = predictions[name]
        ax_spec.fill_between(
            wavelengths,
            mean - CI_Z * std,
            mean + CI_Z * std,
            color=color,
            alpha=0.25,
            lw=0,
            label=f"{label} 95% CI",
        )
        ax_spec.plot(wavelengths, mean, color=color, lw=1.5, label=label)
        ax_z.plot(
            wavelengths,
            (actual - mean) / std,
            color=color,
            lw=1.0,
            label=label,
        )
    ax_spec.plot(wavelengths, actual, color="black", lw=1.5, label="Actual HF")
    ax_spec.set(
        ylabel="Eclipse depth",
        title="Held-out spectrum 81: Model 1A vs Model 1B",
    )
    ax_spec.legend()
    for bound in (-CI_Z, CI_Z):
        ax_z.axhline(bound, color="gray", lw=0.8, ls="--")
    ax_z.axhline(0.0, color="gray", lw=0.8)
    ax_z.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="standardized residual",
    )
    figure.tight_layout()
    figure.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    actual = YHF[HOLDOUT_ROW]
    train_mask = np.arange(len(XHF)) != HOLDOUT_ROW
    output_range = np.ptp(YHF[train_mask], axis=0)

    predictions = {}
    tables = []
    summaries = {}
    for name, label in MODELS.items():
        layer = joblib.load(FIT_DIR / f"{name}_layer.joblib")
        mean, std = predict_hf(layer, XHF[[HOLDOUT_ROW]])
        predictions[name] = (mean[0], std[0])
        per_wavelength, summary = evaluate(
            mean[0], std[0], actual, output_range
        )
        per_wavelength.insert(0, "model", name)
        per_wavelength.insert(1, "wavelength", wavelengths)
        per_wavelength.insert(2, "y_hf_actual", actual)
        tables.append(per_wavelength)
        summaries[name] = summary
        print(
            f"{label}: RMSE={summary['rmse']:.3e}, "
            f"mean|z|={summary['mean_abs_standardized_residual']:.3f}, "
            f"coverage(95% CI)={summary['fraction_within_95ci']:.3f}, "
            f"mean NRMSE={summary['mean_nrmse']:.4f}"
        )

    summaries["comparison"] = {
        "rmse_ratio_1a_over_1b": (
            summaries["model_1a"]["rmse"] / summaries["model_1b"]["rmse"]
        ),
        "lower_rmse": min(MODELS, key=lambda name: summaries[name]["rmse"]),
        "lower_mean_nrmse": min(
            MODELS, key=lambda name: summaries[name]["mean_nrmse"]
        ),
    }
    print(
        f"lower holdout RMSE: {summaries['comparison']['lower_rmse']} "
        f"(1A/1B RMSE ratio = "
        f"{summaries['comparison']['rmse_ratio_1a_over_1b']:.3f})"
    )

    pd.concat(tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "holdout_prediction_per_wavelength.csv", index=False
    )
    (OUTPUT_DIR / "holdout_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    plot_comparison(
        wavelengths,
        actual,
        predictions,
        OUTPUT_DIR / "01_holdout_1a_vs_1b_spectrum.png",
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
