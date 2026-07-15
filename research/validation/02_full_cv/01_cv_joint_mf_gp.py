"""K-fold cross-validation of the full joint MF-GP over all HF samples.

Generalizes the single sample-81 holdout of 01_loo_holdout_validation: every
one of the 97 HF spectra is held out exactly once, each fold refits the whole
per-wavelength joint layer without the fold's HF rows (the LF data is always
fully included -- only HF availability is being validated), and predictions
come from exoplanets_mf.mf_gp.predict_hf on the held-out atmospheric inputs.

When a Model 2 exists, run its CV with the same exoplanets_mf.cv engine and
compare via exoplanets_mf.cv.compare_cv.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, cv_metrics, cv_predict
from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import fit_joint_mf_gp, predict_hf
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

OUTPUT_DIR = VALIDATION_RESULTS_DIR / "02_full_cv" / "evaluation"

N_SPLITS = CV_FULL_MODEL_SPLITS  # each fold refits the full 195-GP layer
# Smaller LF subsample than the single production fit because the layer is
# refitted N_SPLITS times. TESTING value: 200 LF + 97 HF is ~2 s/wavelength,
# so 5 folds x 195 wavelengths ~ 30-40 min. For production, size against the
# measured budget (e.g. 900 LF ~ 27 s/wavelength -> ~7.3 h for 5 folds).
SUBSAMPLE_SIZE = 200
SEED = RANDOM_SEED
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


def plot_diagnostics(wavelengths, YHF, predictions, metrics, savepath) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.scatter(YHF.ravel(), predictions.y_pred.ravel(), s=2, alpha=0.1,
               color="tab:blue")
    limits = [YHF.min(), YHF.max()]
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set(xlabel="actual HF eclipse depth", ylabel="held-out prediction",
           title="Out-of-fold predictions vs actual")

    ax = axes[1]
    shade_instrument_modes(ax, wavelengths)
    z = (predictions.y_pred - YHF) / predictions.y_std
    coverage_per_wavelength = np.mean(np.abs(z) <= CI_Z, axis=0)
    ax.plot(wavelengths, coverage_per_wavelength, ".-", color="tab:blue", lw=1)
    ax.axhline(0.95, color="black", ls="--", lw=0.8, label="nominal 95%")
    ax.set(xlabel=r"wavelength $\lambda$ [$\mu$m]", ylabel="95% CI coverage",
           title="Coverage vs wavelength")
    ax.legend()

    ax = axes[2]
    ax.bar(np.arange(len(metrics.rmse_per_sample)),
           np.sort(metrics.rmse_per_sample), color="tab:blue", width=1.0)
    ax.set(xlabel="sample (sorted)", ylabel="held-out RMSE",
           title="Per-sample RMSE")

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
        f"{N_SPLITS}-fold CV of the joint MF-GP over all {len(XHF)} HF "
        f"samples (seed={SEED}); each fold refits {len(wavelengths)} GPs"
    )

    def fit_fold(train_idx):
        t0 = time.perf_counter()
        layer = fit_joint_mf_gp(
            XLF_10k,
            YLF_10k,
            XHF[train_idx],
            YHF[train_idx],
            wavelengths,
            seed=SEED,
            subsample_size=SUBSAMPLE_SIZE,
            progress_every=50,
        )
        print(f"  fold fitted in {time.perf_counter() - t0:.1f}s", flush=True)
        return layer

    predictions = cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf(layer, XHF[test_idx]),
        n_samples=YHF.shape[0],
        n_wavelengths=YHF.shape[1],
        n_splits=N_SPLITS,
        seed=SEED,
    )
    metrics = cv_metrics(YHF, predictions)

    z = (predictions.y_pred - YHF) / predictions.y_std
    summary = {
        "n_samples": int(YHF.shape[0]),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "rmse_pooled": metrics.rmse_pooled,
        "nrmse_pooled": metrics.nrmse_pooled,
        "mean_abs_standardized_residual": float(np.mean(np.abs(z))),
        "fraction_within_95ci": metrics.coverage_95,
    }
    print(
        f"Joint MF-GP CV: RMSE={summary['rmse_pooled']:.3e}, "
        f"mean|z|={summary['mean_abs_standardized_residual']:.3f}, "
        f"coverage(95% CI)={summary['fraction_within_95ci']:.3f}, "
        f"NRMSE={summary['nrmse_pooled']:.4f}"
    )

    pd.DataFrame(
        {
            "sample": np.arange(YHF.shape[0]),
            "fold": predictions.fold_of_sample,
            "rmse": metrics.rmse_per_sample,
            "mean_abs_z": np.mean(np.abs(z), axis=1),
            "within_95ci": np.mean(np.abs(z) <= CI_Z, axis=1),
        }
    ).to_csv(OUTPUT_DIR / "cv_per_sample.csv", index=False)

    pd.DataFrame(
        {
            "wavelength": wavelengths,
            "rmse": metrics.rmse_per_wavelength,
            "coverage_95": np.mean(np.abs(z) <= CI_Z, axis=0),
        }
    ).to_csv(OUTPUT_DIR / "cv_per_wavelength.csv", index=False)

    (OUTPUT_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    plot_diagnostics(
        wavelengths, YHF, predictions, metrics,
        OUTPUT_DIR / "01_cv_diagnostics.png",
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
