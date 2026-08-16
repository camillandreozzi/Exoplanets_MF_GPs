"""Leave-one-out validation splits for the high-fidelity samples."""

try:
    from src.data_load import create_data_indexes, load_full_data
except ModuleNotFoundError:  # Allows running this file directly from src/.
    from data_load import create_data_indexes, load_full_data


def iter_loo_splits(full_data):
    attrs = full_data.attrs.copy()
    full_data = full_data.reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )
    full_data.attrs.update(attrs)

    hf_positions = full_data.index[full_data["is_hf"].eq(1)].tolist()

    if len(hf_positions) != 97:
        raise ValueError(f"Expected 97 HF samples, found {len(hf_positions)}")

    for fold, validation_pos in enumerate(hf_positions, start=1):
        validation_data = full_data.iloc[[validation_pos]].reset_index(drop=True)
        train_data = full_data.drop(index=validation_pos).reset_index(drop=True)
        validation_data.attrs.update(attrs)
        train_data.attrs.update(attrs)

        yield {
            "fold": fold,
            "held_out_source_index": validation_data["source_index"].iloc[0],
            "train_data": train_data,
            "validation_data": validation_data,
        }


if __name__ == "__main__":
    full_data = load_full_data()

    first_split = next(iter_loo_splits(full_data))
    fold_count = sum(1 for _ in iter_loo_splits(full_data))

    print(f"created {fold_count} LOO folds")
    print(f"first validation index: {first_split['held_out_source_index']}")
    print(f"train shape: {first_split['train_data'].shape}")
    print(f"validation shape: {first_split['validation_data'].shape}")
