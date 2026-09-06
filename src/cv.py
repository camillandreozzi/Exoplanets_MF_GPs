"""Validation splits for the high-fidelity samples."""

import numpy as np

try:
    from src.data_load import create_data_indexes, load_full_data
except ModuleNotFoundError:  # Allows running this file directly from src/.
    from data_load import create_data_indexes, load_full_data


EXPECTED_HF_SAMPLES = 97


def with_source_index(full_data):
    attrs = full_data.attrs.copy()
    indexed = full_data.reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )
    indexed.attrs.update(attrs)
    return indexed


def validate_hf_sample_count(full_data):
    n_hf = int(full_data["is_hf"].eq(1).sum())
    if n_hf != EXPECTED_HF_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_HF_SAMPLES} HF samples, found {n_hf}")


def iter_hf_kfold_splits(full_data, n_folds=5, random_state=2026):
    attrs = full_data.attrs.copy()
    full_data = with_source_index(full_data)
    validate_hf_sample_count(full_data)

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    hf_positions = full_data.index[full_data["is_hf"].eq(1)].to_numpy()
    if n_folds > len(hf_positions):
        raise ValueError("n_folds cannot exceed the number of HF samples.")

    rng = np.random.default_rng(random_state)
    shuffled_hf_positions = rng.permutation(hf_positions)

    for fold, validation_positions in enumerate(
        np.array_split(shuffled_hf_positions, n_folds),
        start=1,
    ):
        validation_positions = np.sort(validation_positions)
        validation_data = full_data.loc[validation_positions].reset_index(drop=True)
        train_data = full_data.drop(index=validation_positions).reset_index(drop=True)
        validation_data.attrs.update(attrs)
        train_data.attrs.update(attrs)

        yield {
            "fold": fold,
            "held_out_source_index": validation_data["source_index"].tolist(),
            "train_data": train_data,
            "validation_data": validation_data,
        }


def iter_loo_splits(full_data):
    attrs = full_data.attrs.copy()
    full_data = with_source_index(full_data)
    validate_hf_sample_count(full_data)

    hf_positions = full_data.index[full_data["is_hf"].eq(1)].tolist()

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

    first_split = next(iter_hf_kfold_splits(full_data))
    fold_count = sum(1 for _ in iter_hf_kfold_splits(full_data))

    print(f"created {fold_count} HF k-fold splits")
    print(f"first validation index: {first_split['held_out_source_index']}")
    print(f"train shape: {first_split['train_data'].shape}")
    print(f"validation shape: {first_split['validation_data'].shape}")
