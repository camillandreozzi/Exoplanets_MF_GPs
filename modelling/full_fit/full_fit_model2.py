import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_load import load_full_data
from src.model_2 import prepare_model2_data
from src.model_1 import fit_model1, predict_model1, save_model1


HELD_OUT_HF_SAMPLE = 81
LF_SAMPLES = 1000
N_WAVELENGTHS = 50
RANDOM_STATE = 42

full_data = load_full_data()
hf_rows = full_data.index[full_data["is_hf"].eq(1)]
held_out_row = hf_rows[HELD_OUT_HF_SAMPLE]
held_out_wide = full_data.loc[[held_out_row]].copy()
training_wide = full_data.drop(index=held_out_row).copy() # Preserve dataframe metadata explicitly.
held_out_wide.attrs.update(full_data.attrs)
training_wide.attrs.update(full_data.attrs)


# ---------------------------------------------------------------
# Prepare the training data
# ---------------------------------------------------------------

train_long = prepare_model2_data(
    training_wide,
    LF_number=LF_SAMPLES,
    n_wavelengths=N_WAVELENGTHS,
    random_state=RANDOM_STATE,
)
held_out_long = prepare_model2_data(
    held_out_wide,
    LF_number=None,
    # Evaluate the held-out spectrum on the complete common wavelength grid.
    wavelength_indices=sorted(full_data.attrs["wavelength_map"]),
)


output_dir = (
    PROJECT_ROOT
    / "results"
    / "model2"
    / f"leave_hf_{HELD_OUT_HF_SAMPLE}_out"
)

fit_model1(train_long,HF_only=True,)
sf_prediction = predict_model1(
    held_out_long,
    HF_only=True,
)
save_model1(output_dir / "model2_SF.pkl")

np.save(
    output_dir / "model2_SF_prediction.npy",
    sf_prediction["mu"],
)

fit_model1(train_long,HF_only=False,)
mf_prediction = predict_model1(held_out_long,HF_only=False,)
save_model1(output_dir / "model2_MF.pkl")

np.save(
    output_dir / "model2_SF_prediction.npy",
    sf_prediction["mu"],
)
np.save(
    output_dir / "model2_MF_prediction.npy",
    mf_prediction["mu"],
)


np.save(
    output_dir / "actual_spectrum.npy",
    held_out_long["response"].to_numpy(),
)

np.save(
    output_dir / "wavelengths.npy",
    held_out_long["wavelength"].to_numpy(),
)

print(f"Held out HF sample: {HELD_OUT_HF_SAMPLE}")
print(f"Original full_data row: {held_out_row}")
print(f"LF samples used: {LF_SAMPLES}")
print(f"Wavelengths sampled per training spectrum: {N_WAVELENGTHS}")
print(f"Held-out wavelengths predicted: {len(held_out_long)}")
print(f"Results saved in: {output_dir}")
