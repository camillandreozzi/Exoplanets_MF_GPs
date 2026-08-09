"""Fit Models 1A and 1B while excluding only sample 81.

Superseded by research/validation/02_full_cv (all 97 HF samples). Fits both
per-wavelength AR(1) variants on the identical holdout design so
02_predict_and_evaluate.py can compare them on the held-out spectrum:

- Model 1B (fit_joint_mf_gp): one free rho_j per wavelength;
- Model 1A (fit_joint_mf_gp_global_rho): ONE shared rho, warm-started from
  the Model 1B holdout fit (same LF subsample and seed, enforced by the
  fitter; the warm start changes only the starting point, not the criterion).

Uses the same LF subsample size as the modelling fit scripts (derivation in
research/modelling/01_per_wavelength_ar1/01_fit_model1b.py).
"""

import time

import joblib
import numpy as np

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import (
    assert_holdout_row,
    fit_joint_mf_gp,
    fit_joint_mf_gp_global_rho,
    hyperparameter_table,
)
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

OUTPUT_DIR = VALIDATION_RESULTS_DIR / "01_loo_holdout_validation" / "fit"

HOLDOUT_ROW = 80
HOLDOUT_KZZ = 8.47701413791753
HOLDOUT_LABEL = "spectrum 81"

# The one canonical LF subsample shared by every model (see
# reproducibility.LF_SUBSAMPLE_SIZE); matches the fit scripts.
SUBSAMPLE_SIZE = LF_SUBSAMPLE_SIZE
SEED = RANDOM_SEED


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    XHF_df, YHF_df = data["XHF"], data["YHF"]
    wavelengths = data["wavelengths"]

    assert_holdout_row(
        XHF_df,
        YHF_df,
        HOLDOUT_ROW,
        expected_kzz=HOLDOUT_KZZ,
        expected_label=HOLDOUT_LABEL,
    )

    train_mask = np.arange(len(XHF_df)) != HOLDOUT_ROW
    XLF = data["XLF_10k"].to_numpy()
    YLF = data["YLF_10k"].to_numpy()
    XHF_train = XHF_df.to_numpy()[train_mask]
    YHF_train = YHF_df.to_numpy()[train_mask]

    print(
        f"Model 1B excluding row {HOLDOUT_ROW}: {len(wavelengths)} "
        f"wavelengths, {SUBSAMPLE_SIZE} LF + {len(XHF_train)} HF rows"
    )
    t0 = time.perf_counter()
    layer_1b = fit_joint_mf_gp(
        XLF,
        YLF,
        XHF_train,
        YHF_train,
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        progress_every=20,
    )
    rho_1b = layer_1b.rho
    print(
        f"Model 1B fit in {time.perf_counter() - t0:.1f}s: "
        f"rho mean={rho_1b.mean():.4f}, "
        f"min={rho_1b.min():.4f}, max={rho_1b.max():.4f}"
    )

    print(f"Model 1A excluding row {HOLDOUT_ROW}: warm start from Model 1B")
    t0 = time.perf_counter()
    layer_1a = fit_joint_mf_gp_global_rho(
        XLF,
        YLF,
        XHF_train,
        YHF_train,
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        warm_start_layer=layer_1b,
        progress_every=40,
    )
    print(
        f"Model 1A fit in {time.perf_counter() - t0:.1f}s over "
        f"{len(layer_1a.global_rho_sweeps)} sweeps; "
        f"shared rho = {layer_1a.rho[0]:.6f}"
    )

    for name, layer in (("model_1b", layer_1b), ("model_1a", layer_1a)):
        table = hyperparameter_table(layer)
        table.to_csv(OUTPUT_DIR / f"{name}_hyperparameters.csv", index=False)
        joblib.dump(layer, OUTPUT_DIR / f"{name}_layer.joblib")
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
