from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESPONSE_DIM = 195


def load_wavelength_map(data_dir=DATA_DIR):
    data_dir = Path(data_dir)

    wavelengths = (
        pd.read_csv(data_dir / "YHF.csv", nrows=0)
        .columns[1:]
        .astype(float)
        .to_numpy()
    )

    if len(wavelengths) != RESPONSE_DIM:
        raise ValueError(
            f"Expected {RESPONSE_DIM} wavelengths, found {len(wavelengths)}"
        )

    return dict(enumerate(wavelengths))


def load_full_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)

    x_hf = pd.read_csv(data_dir / "XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()

    x_lf = pd.read_csv(
        data_dir / "XLF_10k.csv",
        header=None,
        names=x_hf.columns,
    )

    y_hf = pd.read_csv(data_dir / "YHF.csv", index_col=0)
    y_lf = pd.read_csv(data_dir / "YLF_10k.csv")

    response_columns = [f"response_{i}" for i in range(RESPONSE_DIM)]
    wavelength_map = load_wavelength_map(data_dir)
    y_hf.columns = response_columns
    y_lf.columns = response_columns

    x_hf = x_hf.reset_index(drop=True)
    x_hf["is_hf"] = 1
    hf_data = pd.concat(
        [x_hf, y_hf.reset_index(drop=True)],
        axis=1,
    )

    x_lf = x_lf.reset_index(drop=True)
    x_lf["is_hf"] = 0
    lf_data = pd.concat(
        [x_lf, y_lf.reset_index(drop=True)],
        axis=1,
    )

    full_data = pd.concat([lf_data, hf_data], ignore_index=True)
    full_data.attrs["wavelength_map"] = wavelength_map
    return full_data


def create_data_indexes(full_data):
    response_columns = [f"response_{i}" for i in range(RESPONSE_DIM)]
    missing_response_columns = [
        column for column in response_columns if column not in full_data.columns
    ]
    if missing_response_columns:
        raise ValueError("full_data is missing one or more response columns")
    if "is_hf" not in full_data.columns:
        raise ValueError("full_data must include an is_hf column")

    metadata_columns = {"is_hf", "source_index"}
    x_columns = [
        column
        for column in full_data.columns
        if column not in response_columns and column not in metadata_columns
    ]

    indexes = {
        "x_columns": x_columns,
        "y_columns": response_columns,
        "x_column_idx": [full_data.columns.get_loc(column) for column in x_columns],
        "y_column_idx": [
            full_data.columns.get_loc(column) for column in response_columns
        ],
        "lf_row_idx": full_data.index[full_data["is_hf"].eq(0)].tolist(),
        "hf_row_idx": full_data.index[full_data["is_hf"].eq(1)].tolist(),
    }
    return indexes


if __name__ == "__main__":
    full_data = load_full_data()
    indexes = create_data_indexes(full_data)
    print(full_data.shape)
    print(full_data.head())
    print(full_data["is_hf"].value_counts().sort_index())
    print(indexes.keys())
