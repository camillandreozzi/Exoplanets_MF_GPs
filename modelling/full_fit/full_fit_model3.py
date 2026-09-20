"""Fit Model 3 wavelength-wise on the full LF + HF dataset."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_3 import fit_model3, load_model3, predict_model3, save_model3


OUTPUT_DIR = PROJECT_ROOT / "results/model3/full_fit"
# The closing figure needs a display and 195 model loads; batch jobs skip it.
PLOT_AFTER_FIT = os.environ.get("MODEL3_PLOT", "1") == "1"
# Model 3 tunes automatically inside fit_model3. Use MODEL3_TUNING_KWARGS
# to override tune_model3_parameters defaults, e.g. {"n_trials": 20}.
MODEL3_PARAMS = None
MODEL3_GP_KWARGS = None
MODEL3_TUNING_KWARGS = None
NUM_BOOST_ROUND = None


def main():
    full_data = load_full_data()
    indexes = create_data_indexes(full_data)
    x_columns = indexes["x_columns"]
    wavelength_map = full_data.attrs["wavelength_map"]

    # Hold out sample 81 from training (HF design point used for closure test)
    sample_idx = 81
    hf_rows = full_data.index[full_data["is_hf"] == 1]
    held_out_row = hf_rows[sample_idx]
    training_data = full_data.drop(index=held_out_row).copy()

    for i in range(RESPONSE_DIM):
        response_col = f"response_{i}"
        considered_wavelength = wavelength_map[i]
        train_data = select_scalar_response(training_data, x_columns, response_col)

        for variant, hf_only in (("sf", True), ("mf", False)):
            path = OUTPUT_DIR / f"model3_{variant}_response_{i}.pkl"
            # 390 tuned fits do not finish inside one wall-clock limit, and the
            # loop restarts at 0 on resubmission. Skipping what is already on
            # disk turns a resubmit into a resume.
            if path.exists() and path.with_suffix(path.suffix + ".gp_model.json").exists():
                print(f"Skipping response_{i} {variant}, already fitted", flush=True)
                continue

            fit_model3(
                train_data,
                HF_only=hf_only,
                params=MODEL3_PARAMS,
                num_boost_round=NUM_BOOST_ROUND,
                gp_kwargs=MODEL3_GP_KWARGS,
                tuning_kwargs=MODEL3_TUNING_KWARGS,
            )
            predict_model3(train_data, HF_only=hf_only)
            save_model3(path)

        print(
            f"Finished response_{i}, wavelength={considered_wavelength} nm",
            flush=True,
        )

    if PLOT_AFTER_FIT:
        plot_fitted_spectrum(full_data, x_columns)


def select_scalar_response(data, x_columns, response_col):
    scalar_data = data.loc[:, [*x_columns, "is_hf", response_col]].copy()
    scalar_data.attrs.update(data.attrs)
    return scalar_data


def plot_fitted_spectrum(full_data, x_columns):
    # Imported here, not at module scope: the Euler venv carries no matplotlib,
    # and the batch fit never plots (MODEL3_PLOT=0).
    import matplotlib.pyplot as plt

    sample_idx = 81
    model_kind = "mf"  # use "sf" for HF-only models, "mf" for multi-fidelity models
    hf_only = model_kind == "sf"

    sample_data = full_data.iloc[[sample_idx]].copy()
    wavelength_map = full_data.attrs["wavelength_map"]
    wavelengths = np.array([wavelength_map[i] for i in range(RESPONSE_DIM)])
    actual_spectrum = sample_data.loc[
        :, [f"response_{i}" for i in range(RESPONSE_DIM)]
    ].iloc[0].to_numpy(dtype=float)

    predicted = []
    variance = []

    for i in range(RESPONSE_DIM):
        response_col = f"response_{i}"
        model = load_model3(
            OUTPUT_DIR / f"model3_{model_kind}_response_{i}.pkl",
            HF_only=hf_only,
        )
        validation_data = select_scalar_response(sample_data, x_columns, response_col)
        pred = predict_model3(
            validation_data,
            HF_only=hf_only,
            model=model,
            compute_metrics=False,
        )

        predicted.append(pred["mu"][0])
        variance.append(pred["var"][0])

    predicted = np.array(predicted)
    std = np.sqrt(np.array(variance))

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, actual_spectrum, label="Actual spectrum", color="black")
    plt.plot(wavelengths, predicted, label="Model 3 fitted spectrum", color="tab:green")
    plt.fill_between(
        wavelengths,
        predicted - 1.96 * std,
        predicted + 1.96 * std,
        color="tab:green",
        alpha=0.2,
        label="95% confidence interval",
    )
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title(f"Model 3 {model_kind.upper()} fit for sample {sample_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
