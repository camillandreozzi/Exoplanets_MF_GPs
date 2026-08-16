# Model 2
import numpy as np
from src.model_1 import predict_model1, save_model1

def prepare_model2_data(full_data, LF_number = 1000, random_state = None):
    wavelength_map = full_data.attrs["wavelength_map"]

    if LF_number is not None:
        attrs = full_data.attrs.copy()

        lf_data = full_data[full_data["is_hf"].eq(0)]
        hf_data = full_data[full_data["is_hf"].eq(1)]

        if LF_number > len(lf_data):
            raise ValueError(
                f"LF_number={LF_number} requested, but only {len(lf_data)} LF rows exist."
            )

        lf_sample = lf_data.sample(n=LF_number, random_state=random_state)

        # Keep all HF rows, and use the same reduced LF set before melting.
        full_data = (
            full_data
            .loc[lf_sample.index.union(hf_data.index).sort_values()]
            .reset_index(drop=True)
            .copy()
        )
        full_data.attrs.update(attrs)

    response_items = sorted(wavelength_map.items())
    response_columns = [f"response_{i}" for i, _ in response_items]
    wavelength_by_response = {
        f"response_{i}": float(wavelength)
        for i, wavelength in response_items
    }

    non_input_columns = set(response_columns) | {"is_hf", "source_index"}
    atmospheric_columns = [
        column
        for column in full_data.columns
        if column not in non_input_columns
    ]

    working = full_data.copy()
    working["_row_id"] = range(len(working))

    long_data = working.melt(
        id_vars=["_row_id", *atmospheric_columns, "is_hf"],
        value_vars=response_columns,
        var_name="response_column",
        value_name="response",
    )

    long_data["response_index"] = (
        long_data["response_column"]
        .str.removeprefix("response_")
        .astype(int)
    )
    long_data["wavelength"] = long_data["response_column"].map(wavelength_by_response)

    long_data = (
        long_data
        .sort_values(["_row_id", "response_index"])
        .reset_index(drop=True)
        [atmospheric_columns + ["wavelength", "is_hf", "response"]]
    )

    long_data.attrs.update(full_data.attrs)
    long_data.attrs["x_columns"] = atmospheric_columns + ["wavelength"]
    long_data.attrs["LF_number"] = LF_number

    return long_data