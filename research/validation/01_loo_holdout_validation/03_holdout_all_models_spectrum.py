"""Held-out spectrum 81: predictions of Models 1A, 1B, 2, and the HF-only GP.

Extends 02_predict_and_evaluate.py (Models 1A/1B only) to all four models used
in the project, each fitted on the identical holdout design -- every HF sample
except spectrum 81 -- and predicting that one held-out spectrum from its
atmospheric inputs alone. All multi-fidelity models draw the one canonical LF
subsample (reproducibility.LF_SUBSAMPLE_SIZE, seed RANDOM_SEED), so they train
on the identical LF rows and the same custom AR(1) sklearn kernel; the HF-only
GP is the single-fidelity floor (it never sees the LF design).

Output space is original eclipse-depth units (linear), matching the existing
holdout plot. Produces one figure overlaying every model's mean +/- 95% CI
against the actual held-out spectrum, plus a standardized-residual panel, and
writes the per-wavelength predictions and a metric summary.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from exoplanets_mf.data import load_all
from exoplanets_mf.hf_only_gp import fit_hf_only_gp, predict_hf_only
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import (
    assert_holdout_row,
    fit_joint_mf_gp,
    fit_joint_mf_gp_global_rho,
    predict_hf,
)
from exoplanets_mf.model2 import (
    MODEL2_MAX_AUGMENTED_POINTS,
    derive_lambda_stride,
    fit_model2,
    predict_hf_model2,
)
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

OUTPUT_DIR = (
    VALIDATION_RESULTS_DIR / "01_loo_holdout_validation" / "evaluation"
)

HOLDOUT_ROW = 80
HOLDOUT_KZZ = 8.47701413791753
HOLDOUT_LABEL = "spectrum 81"

SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE
SEED = RANDOM_SEED
CI_Z = stats.norm.ppf(0.975)

# Consistent with 02_cv_model2_vs_model1.py's colour scheme.
MODELS = {
    "hf_only": ("HF-only GP (single-fidelity)", "tab:gray"),
    "model_1a": ("Model 1A (shared rho)", "tab:green"),
    "model_1b": ("Model 1B (per-wavelength rho)", "tab:blue"),
    "model_2": ("Model 2 (wavelength-augmented)", "tab:red"),
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


def fit_all_models(XLF, YLF, XHF_train, YHF_train, wavelengths):
    """Fit the four models on the holdout design; return {name: layer}."""
    print(
        f"Model 1B excluding row {HOLDOUT_ROW}: {len(wavelengths)} wavelengths, "
        f"{SUBSAMPLE_SIZE} LF + {len(XHF_train)} HF rows",
        flush=True,
    )
    t0 = time.perf_counter()
    layer_1b = fit_joint_mf_gp(
        XLF, YLF, XHF_train, YHF_train, wavelengths,
        seed=SEED, subsample_size=SUBSAMPLE_SIZE, progress_every=50,
    )
    print(f"  Model 1B fit in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"Model 1A excluding row {HOLDOUT_ROW}: warm start from Model 1B", flush=True)
    t0 = time.perf_counter()
    layer_1a = fit_joint_mf_gp_global_rho(
        XLF, YLF, XHF_train, YHF_train, wavelengths,
        seed=SEED, subsample_size=SUBSAMPLE_SIZE,
        warm_start_layer=layer_1b, progress_every=None,
    )
    print(
        f"  Model 1A fit in {time.perf_counter() - t0:.1f}s; "
        f"shared rho = {layer_1a.rho[0]:.6f}",
        flush=True,
    )

    lambda_stride = derive_lambda_stride(
        SUBSAMPLE_SIZE, len(XHF_train), len(wavelengths),
        max_points=MODEL2_MAX_AUGMENTED_POINTS,
    )
    print(
        f"Model 2 excluding row {HOLDOUT_ROW}: lf={SUBSAMPLE_SIZE}, "
        f"stride={lambda_stride}",
        flush=True,
    )
    t0 = time.perf_counter()
    layer_2 = fit_model2(
        XLF, YLF, XHF_train, YHF_train, wavelengths,
        seed=SEED, lf_sample_size=SUBSAMPLE_SIZE, lambda_stride=lambda_stride,
    )
    print(
        f"  Model 2 fit in {time.perf_counter() - t0:.1f}s "
        f"(n={layer_2.model.X_train_.shape[0]}, rho={layer_2.rho:.3f})",
        flush=True,
    )

    print(f"HF-only GP excluding row {HOLDOUT_ROW}", flush=True)
    t0 = time.perf_counter()
    layer_hf = fit_hf_only_gp(
        XHF_train, YHF_train, wavelengths, seed=SEED, progress_every=None,
    )
    print(f"  HF-only GP fit in {time.perf_counter() - t0:.1f}s", flush=True)

    return {
        "model_1b": layer_1b,
        "model_1a": layer_1a,
        "model_2": layer_2,
        "hf_only": layer_hf,
    }


def predict_holdout(layers, X_holdout):
    """{name: (mean, std)} for the single held-out row."""
    predictions = {}
    for name, layer in layers.items():
        if name == "model_2":
            mean, std = predict_hf_model2(layer, X_holdout)
        elif name == "hf_only":
            mean, std = predict_hf_only(layer, X_holdout)
        else:
            mean, std = predict_hf(layer, X_holdout)
        predictions[name] = (mean[0], std[0])
    return predictions


def evaluate(mean, std, actual, output_range) -> dict:
    residual = actual - mean
    z = residual / std
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_standardized_residual": float(np.mean(np.abs(z))),
        "fraction_within_95ci": float(np.mean(np.abs(z) <= CI_Z)),
        "mean_nrmse": float(np.mean(np.abs(residual) / output_range)),
        "mean_predictive_std": float(np.mean(std)),
    }


def plot_all_models(wavelengths, actual, predictions, summaries, savepath) -> None:
    figure, (ax_spec, ax_z) = plt.subplots(
        2, 1, figsize=(12, 8.5), sharex=True, height_ratios=[2, 1]
    )
    shade_instrument_modes(ax_spec, wavelengths)
    for name, (label, color) in MODELS.items():
        mean, std = predictions[name]
        rmse = summaries[name]["rmse"]
        ax_spec.fill_between(
            wavelengths, mean - CI_Z * std, mean + CI_Z * std,
            color=color, alpha=0.12, lw=0,
        )
        ax_spec.plot(
            wavelengths, mean, color=color, lw=1.4,
            label=f"{label}  (RMSE={rmse:.2e})",
        )
        ax_z.plot(wavelengths, (actual - mean) / std, color=color, lw=1.0, label=label)
    ax_spec.plot(
        wavelengths, actual, color="black", lw=2.0, zorder=5, label="Actual HF",
    )
    ax_spec.set(
        ylabel="Eclipse depth",
        title="Held-out spectrum 81: Models 1A, 1B, 2 and HF-only baseline",
    )
    ax_spec.legend(fontsize=8, ncol=2)

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
    XHF_df, YHF_df = data["XHF"], data["YHF"]

    assert_holdout_row(
        XHF_df, YHF_df, HOLDOUT_ROW,
        expected_kzz=HOLDOUT_KZZ, expected_label=HOLDOUT_LABEL,
    )

    XHF = XHF_df.to_numpy()
    YHF = YHF_df.to_numpy()
    actual = YHF[HOLDOUT_ROW]
    train_mask = np.arange(len(XHF)) != HOLDOUT_ROW
    output_range = np.ptp(YHF[train_mask], axis=0)

    XLF = data["XLF_10k"].to_numpy()
    YLF = data["YLF_10k"].to_numpy()

    layers = fit_all_models(
        XLF, YLF, XHF[train_mask], YHF[train_mask], wavelengths
    )
    predictions = predict_holdout(layers, XHF[[HOLDOUT_ROW]])

    summaries = {}
    tables = []
    for name, (label, _color) in MODELS.items():
        mean, std = predictions[name]
        summaries[name] = evaluate(mean, std, actual, output_range)
        print(
            f"{label}: RMSE={summaries[name]['rmse']:.3e}, "
            f"mean|z|={summaries[name]['mean_abs_standardized_residual']:.3f}, "
            f"coverage(95% CI)={summaries[name]['fraction_within_95ci']:.3f}, "
            f"mean NRMSE={summaries[name]['mean_nrmse']:.4f}",
            flush=True,
        )
        tables.append(
            pd.DataFrame(
                {
                    "model": name,
                    "wavelength": wavelengths,
                    "y_hf_actual": actual,
                    "y_hf_pred": mean,
                    "std": std,
                    "standardized_residual": (actual - mean) / std,
                }
            )
        )

    summaries["ranking_by_rmse"] = sorted(
        (name for name in MODELS), key=lambda name: summaries[name]["rmse"]
    )
    print(f"holdout RMSE ranking (best first): {summaries['ranking_by_rmse']}", flush=True)

    pd.concat(tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "holdout_all_models_per_wavelength.csv", index=False
    )
    (OUTPUT_DIR / "holdout_all_models_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    plot_all_models(
        wavelengths, actual, predictions, summaries,
        OUTPUT_DIR / "03_holdout_all_models_spectrum.png",
    )
    print(f"outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
