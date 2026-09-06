"""Plot Model 3 in-sample NRMSE per wavelength from full-fit results."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "model3" / "full_fit"
OUTPUT_CSV = RESULTS_DIR / "model3_sf_vs_mf_nrmse_per_wavelength.csv"
OUTPUT_PLOT = RESULTS_DIR / "model3_in_sample_nrmse_per_wavelength.png"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_3 import load_model3, predict_model3


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = compute_model3_nrmse()
    comparison.to_csv(OUTPUT_CSV, index=False)
    plot_nrmse(comparison)

    print(f"Saved Model 3 in-sample NRMSE CSV to {OUTPUT_CSV}")
    print(f"Saved Model 3 in-sample NRMSE plot to {OUTPUT_PLOT}")


def compute_model3_nrmse():
    full_data = load_full_data()
    indexes = create_data_indexes(full_data)
    x_columns = indexes["x_columns"]
    hf_data = full_data.loc[indexes["hf_row_idx"]].copy()
    wavelength_map = full_data.attrs["wavelength_map"]

    rows = []
    for response_index in range(RESPONSE_DIM):
        response_col = f"response_{response_index}"
        validation_data = hf_data.loc[:, [*x_columns, "is_hf", response_col]].copy()
        y_true = validation_data[response_col].to_numpy(dtype=float)
        y_range = np.ptp(y_true)

        sf_rmse = predict_rmse(
            validation_data,
            y_true,
            model_file=RESULTS_DIR / f"model3_sf_response_{response_index}.pkl",
            hf_only=True,
        )
        mf_rmse = predict_rmse(
            validation_data,
            y_true,
            model_file=RESULTS_DIR / f"model3_mf_response_{response_index}.pkl",
            hf_only=False,
        )

        sf_nrmse = normalize_rmse(sf_rmse, y_range)
        mf_nrmse = normalize_rmse(mf_rmse, y_range)
        rows.append(
            {
                "wavelength": wavelength_map[response_index],
                "sf_nrmse": sf_nrmse,
                "mf_nrmse": mf_nrmse,
                "delta_mf_minus_sf": mf_nrmse - sf_nrmse,
                "winner": winner(sf_nrmse, mf_nrmse),
                "sf_rmse": sf_rmse,
                "mf_rmse": mf_rmse,
                "n_hf": len(validation_data),
            }
        )
        print(f"Computed response_{response_index}", flush=True)

    return pd.DataFrame(rows)


def predict_rmse(validation_data, y_true, model_file, hf_only):
    if not model_file.exists():
        raise FileNotFoundError(f"Missing Model 3 full-fit model: {model_file}")

    model = load_model3(model_file, HF_only=hf_only)
    prediction = predict_model3(
        validation_data,
        HF_only=hf_only,
        model=model,
        compute_metrics=False,
    )
    y_pred = np.asarray(prediction["mu"], dtype=float)
    residuals = y_pred - y_true
    return float(np.sqrt(np.mean(residuals**2)))


def normalize_rmse(rmse, y_range):
    if y_range == 0:
        return np.nan
    return float(rmse / y_range)


def winner(sf_nrmse, mf_nrmse):
    if mf_nrmse < sf_nrmse:
        return "MF"
    if sf_nrmse < mf_nrmse:
        return "SF"
    return "Tie"


def plot_nrmse(comparison):
    ordered = comparison.sort_values("wavelength")

    fig, ax = plt.subplots(figsize=(12.1, 6.05), constrained_layout=True)
    ax.plot(
        ordered["wavelength"],
        ordered["sf_nrmse"],
        label="SF NRMSE",
        color="tab:orange",
        linewidth=1.6,
    )
    ax.plot(
        ordered["wavelength"],
        ordered["mf_nrmse"],
        label="MF NRMSE",
        color="tab:blue",
        linewidth=1.6,
    )

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("NRMSE")
    ax.set_title("Model 3 NRMSE per Wavelength (in sample): SF vs MF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
