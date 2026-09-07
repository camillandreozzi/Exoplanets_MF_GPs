"""Fit Model 1 on the full LF + HF dataset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import gpboost as gpb

from src.data_load import load_full_data
from src.model_1 import fit_model1, predict_model1, save_model1

full_data = load_full_data()
wavelength_map = full_data.attrs["wavelength_map"]
output_dir = PROJECT_ROOT / "results/model1/full_fit"

# Model 1
n_input_cols = 10  # 9 physical/input columns + 1 fidelity indicator
sample_idx = 81     # choice of held out sample

hf_rows = full_data.index[full_data["is_hf"] == 1]
held_out_row = hf_rows[sample_idx]

sample_data = full_data.loc[[held_out_row]].copy()
training_data = full_data.drop(index=held_out_row).copy()

# Responses 0--117 have already been fitted. Restart at 118; saving it again
# also replaces the truncated MF file left by the disk-full failure.
start_response = 118

for i in range(start_response, 195):
    considered_wavelength = wavelength_map[i]

    column_indices = list(range(n_input_cols)) + [n_input_cols + i]
    train_data = training_data.iloc[:, column_indices].copy()
    test_data = sample_data.iloc[:, column_indices].copy()

    # Single-fidelity model: train without HF sample 81.
    fit_model1(train_data, HF_only=True)
    predict_model1(test_data, HF_only=True)
    save_model1(output_dir / f"model1_sf_response_{i}.pkl")

    # Multi-fidelity model: train without HF sample 81.
    fit_model1(train_data, HF_only=False)
    predict_model1(test_data, HF_only=False)
    save_model1(output_dir / f"model1_mf_response_{i}.pkl")

    print(
        f"Finished response_{i}, "
        f"wavelength={considered_wavelength} nm"
    )

model_kind = "mf"  # use "sf" for HF-only models, "mf" for multi-fidelity models

sample_data = full_data.iloc[[sample_idx]].copy()
wavelength_map = full_data.attrs["wavelength_map"]
wavelengths = np.array([wavelength_map[i] for i in range(195)])

actual_spectrum = sample_data.iloc[0, n_input_cols:n_input_cols + 195].values.astype(float)

predicted = []
variance = []

for i in range(195):
    gp_model = gpb.GPModel(
        model_file=str(output_dir / f"model1_{model_kind}_response_{i}.pkl")
    )

    x_pred = sample_data.iloc[:, :9].values

    if model_kind == "sf":
        gp_coords_pred = x_pred
    else:
        gp_coords_pred = sample_data.iloc[:, :10].values  # 9 inputs + is_hf

    pred = gp_model.predict(
        gp_coords_pred=gp_coords_pred,
        X_pred=x_pred,
        predict_response=True,
        predict_var=True,
    )

    predicted.append(pred["mu"][0])
    variance.append(pred["var"][0])

predicted = np.array(predicted)
std = np.sqrt(np.array(variance))

plt.figure(figsize=(10, 5))
plt.plot(wavelengths, actual_spectrum, label="Actual spectrum", color="black")
plt.plot(wavelengths, predicted, label="Model 1 fitted spectrum", color="tab:blue")
plt.fill_between(
    wavelengths,
    predicted - 1.96 * std,
    predicted + 1.96 * std,
    color="tab:blue",
    alpha=0.2,
    label="95% confidence interval",
)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title(f"Model 1 {model_kind.upper()} fit for sample {sample_idx}")
plt.legend()
plt.tight_layout()
plt.show()

    
