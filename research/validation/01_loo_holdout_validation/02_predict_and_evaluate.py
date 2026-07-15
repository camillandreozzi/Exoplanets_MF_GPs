"""Predict and score the held-out HF spectrum with the joint MF-GP."""

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()

    layer = joblib.load(FIT_DIR / "joint_mf_gp_layer.joblib")
    mean, std = predict_hf(layer, XHF[[HOLDOUT_ROW]])
    mean, std = mean[0], std[0]
    actual = YHF[HOLDOUT_ROW]

    residual = actual - mean
    standardized_residual = residual / std
    within_95ci = np.abs(standardized_residual) <= CI_Z
    train_mask = np.arange(len(XHF)) != HOLDOUT_ROW
    output_range = np.ptp(YHF[train_mask], axis=0)
    nrmse = np.abs(residual) / output_range

    summary = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_standardized_residual": float(
            np.mean(np.abs(standardized_residual))
        ),
        "fraction_within_95ci": float(np.mean(within_95ci)),
        "mean_nrmse": float(np.mean(nrmse)),
    }
    print(
        f"Joint MF-GP: RMSE={summary['rmse']:.3e}, "
        f"mean|z|={summary['mean_abs_standardized_residual']:.3f}, "
        f"coverage(95% CI)={summary['fraction_within_95ci']:.3f}, "
        f"mean NRMSE={summary['mean_nrmse']:.4f}"
    )

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "y_hf_actual": actual,
            "y_hf_pred": mean,
            "std": std,
            "residual": residual,
            "standardized_residual": standardized_residual,
            "within_95ci": within_95ci,
            "nrmse": nrmse,
        }
    ).to_csv(OUTPUT_DIR / "holdout_prediction_per_wavelength.csv", index=False)
    (OUTPUT_DIR / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    figure, axis = plt.subplots(figsize=(11, 5))
    shade_instrument_modes(axis, wavelengths)
    axis.fill_between(
        wavelengths,
        mean - CI_Z * std,
        mean + CI_Z * std,
        alpha=0.25,
        lw=0,
        label="Joint MF-GP 95% CI",
    )
    axis.plot(wavelengths, mean, lw=1.5, label="Joint MF-GP mean")
    axis.plot(wavelengths, actual, color="black", lw=1.5, label="Actual HF")
    axis.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="Eclipse depth",
        title="Held-out spectrum: joint multi-fidelity prediction",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "01_predicted_vs_actual_spectrum.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
