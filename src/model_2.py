# Model 2
import numpy as np


def prepare_model2_data(
    full_data,
    LF_number=1000,
    n_wavelengths=None,
    wavelength_indices=None,
    random_state=None,
):
    """
    Convert wide spectral data into Model 2's long format.

    Parameters
    ----------
    full_data
        Wide LF/HF dataframe returned by load_full_data().
    LF_number
        Number of LF atmospheric samples to retain. Use None to retain all
        LF samples.
    n_wavelengths
        Number of wavelengths to sample independently for each spectrum.
        Sampling is without replacement. Use None to retain all wavelengths.
    wavelength_indices
        Explicit wavelength indices to use for every spectrum. This takes
        precedence over n_wavelengths and is useful for validation/prediction
        on a common wavelength grid.
    random_state
        Seed used when sampling LF atmospheric samples and wavelengths.
    """
    wavelength_map = full_data.attrs["wavelength_map"]
    attrs = full_data.attrs.copy()
    retained_row_indices = full_data.index.tolist()

    # ---------------------------------------------------------------
    # Select LF atmospheric samples.
    # All HF rows currently present in full_data are always retained.
    # ---------------------------------------------------------------
    if LF_number is not None:
        lf_data = full_data[full_data["is_hf"].eq(0)]
        hf_data = full_data[full_data["is_hf"].eq(1)]

        if LF_number < 0:
            raise ValueError("LF_number must be non-negative or None.")

        if LF_number > len(lf_data):
            raise ValueError(
                f"LF_number={LF_number} requested, but only "
                f"{len(lf_data)} LF rows exist."
            )

        lf_sample = lf_data.sample(
            n=LF_number,
            random_state=random_state,
        )

        selected_rows = lf_sample.index.union(hf_data.index).sort_values()
        retained_row_indices = selected_rows.tolist()

        full_data = (
            full_data.loc[selected_rows]
            .reset_index(drop=True)
            .copy()
        )
        full_data.attrs.update(attrs)

    # ---------------------------------------------------------------
    # Select wavelength indices.
    # ---------------------------------------------------------------
    all_wavelength_indices = np.array(
        sorted(wavelength_map.keys()),
        dtype=int,
    )

    sample_wavelengths_per_spectrum = (
        wavelength_indices is None and n_wavelengths is not None
    )

    if wavelength_indices is not None:
        selected_wavelength_indices = np.asarray(
            wavelength_indices,
            dtype=int,
        )

        invalid_indices = np.setdiff1d(
            selected_wavelength_indices,
            all_wavelength_indices,
        )

        if len(invalid_indices) > 0:
            raise ValueError(
                "Invalid wavelength indices: "
                f"{invalid_indices.tolist()}"
            )

        # Remove duplicates while preserving the requested order.
        selected_wavelength_indices = np.array(
            list(dict.fromkeys(selected_wavelength_indices.tolist())),
            dtype=int,
        )

    elif n_wavelengths is None:
        selected_wavelength_indices = all_wavelength_indices

    else:
        if not 1 <= n_wavelengths <= len(all_wavelength_indices):
            raise ValueError(
                f"n_wavelengths must be between 1 and "
                f"{len(all_wavelength_indices)}."
            )

        # All wavelengths are candidates; each spectrum gets its own subset
        # after the data have been converted to long format.
        selected_wavelength_indices = all_wavelength_indices

    response_items = [
        (i, wavelength_map[i])
        for i in selected_wavelength_indices
    ]

    response_columns = [
        f"response_{i}"
        for i, _ in response_items
    ]

    wavelength_by_response = {
        f"response_{i}": float(wavelength)
        for i, wavelength in response_items
    }

    # ---------------------------------------------------------------
    # Identify the atmospheric input columns.
    # ---------------------------------------------------------------
    all_response_columns = [
        f"response_{i}"
        for i in all_wavelength_indices
    ]

    non_input_columns = (
        set(all_response_columns)
        | {"is_hf", "source_index"}
    )

    atmospheric_columns = [
        column
        for column in full_data.columns
        if column not in non_input_columns
    ]

    # ---------------------------------------------------------------
    # Convert spectra from wide to long format.
    # ---------------------------------------------------------------
    working = full_data.copy()
    working["_row_id"] = range(len(working))

    long_data = working.melt(
        id_vars=[
            "_row_id",
            *atmospheric_columns,
            "is_hf",
        ],
        value_vars=response_columns,
        var_name="response_column",
        value_name="response",
    )

    long_data["response_index"] = (
        long_data["response_column"]
        .str.removeprefix("response_")
        .astype(int)
    )

    long_data["wavelength"] = (
        long_data["response_column"]
        .map(wavelength_by_response)
    )

    wavelength_indices_by_row = None
    if sample_wavelengths_per_spectrum:
        rng = np.random.default_rng(random_state)
        wavelength_indices_by_row = {
            row_id: np.sort(
                rng.choice(
                    all_wavelength_indices,
                    size=n_wavelengths,
                    replace=False,
                )
            ).tolist()
            for row_id in working["_row_id"]
        }

        keep_pairs = {
            (row_id, response_index)
            for row_id, response_indices in wavelength_indices_by_row.items()
            for response_index in response_indices
        }
        keep_mask = [
            (row_id, response_index) in keep_pairs
            for row_id, response_index in zip(
                long_data["_row_id"],
                long_data["response_index"],
            )
        ]
        long_data = long_data.loc[keep_mask]

    long_data = (
        long_data
        .sort_values(["_row_id", "response_index"])
        .reset_index(drop=True)
        [
            atmospheric_columns
            + ["wavelength", "is_hf", "response"]
        ]
    )

    long_data.attrs.update(full_data.attrs)
    long_data.attrs["x_columns"] = (
        atmospheric_columns + ["wavelength"]
    )
    long_data.attrs["LF_number"] = LF_number
    long_data.attrs["wavelength_indices"] = (
        None
        if sample_wavelengths_per_spectrum
        else selected_wavelength_indices.tolist()
    )
    long_data.attrs["wavelength_indices_by_row"] = wavelength_indices_by_row
    long_data.attrs["source_dataframe_indices_by_row"] = retained_row_indices

    return long_data
