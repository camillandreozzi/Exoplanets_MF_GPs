"""Fit the joint multi-fidelity model while excluding only sample 81.

Superseded by research/validation/02_full_cv (all 97 HF samples). Uses the
same maximized LF subsample as the production fit (derivation in
research/modelling/01_per_wavelength_ar1/01_fit_joint_mf_gp.py).
"""

import joblib
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import assert_holdout_row, fit_joint_mf_gp
from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

OUTPUT_DIR = VALIDATION_RESULTS_DIR / "01_loo_holdout_validation" / "fit"

HOLDOUT_ROW = 80
HOLDOUT_KZZ = 8.47701413791753
HOLDOUT_LABEL = "spectrum 81"

SUBSAMPLE_SIZE = 400  # matches the fit script's current (testing) size
SEED = RANDOM_SEED


def hyperparameter_table(wavelengths, layer) -> pd.DataFrame:
    rows = []
    for wavelength, model in zip(wavelengths, layer.models):
        kernel = model.kernel_
        rows.append(
            {
                "wavelength": wavelength,
                "rho": kernel.rho,
                "low_signal_variance": (
                    kernel.low_kernel.k1.constant_value
                ),
                "delta_signal_variance": (
                    kernel.discrepancy_kernel.k1.constant_value
                ),
                "low_noise": kernel.low_noise,
                "high_noise": kernel.high_noise,
                "joint_log_marginal_likelihood": (
                    model.log_marginal_likelihood()
                ),
            }
        )
    return pd.DataFrame(rows)


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
    layer = fit_joint_mf_gp(
        data["XLF_10k"].to_numpy(),
        data["YLF_10k"].to_numpy(),
        XHF_df.to_numpy()[train_mask],
        YHF_df.to_numpy()[train_mask],
        wavelengths,
        seed=SEED,
        subsample_size=SUBSAMPLE_SIZE,
        progress_every=20,
    )

    table = hyperparameter_table(wavelengths, layer)
    print(
        f"joint MF-GP excluding row {HOLDOUT_ROW}: "
        f"rho mean={table['rho'].mean():.4f}, "
        f"min={table['rho'].min():.4f}, max={table['rho'].max():.4f}"
    )
    table.to_csv(OUTPUT_DIR / "joint_mf_gp_hyperparameters.csv", index=False)
    joblib.dump(layer, OUTPUT_DIR / "joint_mf_gp_layer.joblib")
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
