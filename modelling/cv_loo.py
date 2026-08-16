"""Leave-one-out cross-validation for Model 1 and Model 2.

Model 1 is wavelength-wise: each response_i is fit with its own scalar GP.
Model 2 is augmented: one GP per fold uses wavelength as an extra input.

Edit LF_SAMPLE_SIZE to change how many low-fidelity rows are used in each
multi-fidelity training set. Set it to None to use all low-fidelity rows.
"""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.cv import iter_loo_splits
from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_1 import fit_model1, predict_model1
from src.model_2 import prepare_model2_data


LF_SAMPLE_SIZE = 200
LF_SAMPLE_RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results" / "cv"

# None means all 195 wavelengths. Use a list like [0, 10, 50] for a quick run.
MODEL1_RESPONSE_INDEXES = None
MODEL2_RESPONSE_INDEXES = None

VARIANTS = {
    "sf": True,
    "mf": False,
}


def main():
    full_data = load_full_data()
    cv_data = sample_lf_rows(
        full_data,
        lf_sample_size=LF_SAMPLE_SIZE,
        random_state=LF_SAMPLE_RANDOM_STATE,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config(full_data, cv_data)

    prediction_rows = []
    for split in iter_loo_splits(cv_data):
        print(
            "Starting fold "
            f"{split['fold']:03d}, held-out HF source "
            f"{split['held_out_source_index']}"
        )

        prediction_rows.extend(run_model1_fold(split, MODEL1_RESPONSE_INDEXES))
        prediction_rows.extend(run_model2_fold(split, MODEL2_RESPONSE_INDEXES))

    predictions = pd.DataFrame(prediction_rows)
    predictions_file = RESULTS_DIR / "loo_predictions.csv"
    predictions.to_csv(predictions_file, index=False)

    metrics = per_wavelength_metrics(predictions)
    metrics_file = RESULTS_DIR / "loo_metrics_per_wavelength.csv"
    metrics.to_csv(metrics_file, index=False)

    nrmse = nrmse_comparison(metrics)
    nrmse_file = RESULTS_DIR / "loo_nrmse_per_wavelength.csv"
    nrmse.to_csv(nrmse_file, index=False)

    for model in ["model1", "model2"]:
        model_comparison = sf_mf_comparison(metrics, model)
        model_comparison.to_csv(
            RESULTS_DIR / f"{model}_loo_sf_vs_mf_nrmse_per_wavelength.csv",
            index=False,
        )

    summary = overall_summary(metrics)
    summary_file = RESULTS_DIR / "loo_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved CV predictions to {predictions_file}")
    print(f"Saved CV metrics to {RESULTS_DIR}")


def sample_lf_rows(full_data, lf_sample_size, random_state):
    attrs = full_data.attrs.copy()

    if lf_sample_size is None:
        sampled = full_data.copy()
        sampled.attrs.update(attrs)
        return sampled

    if lf_sample_size < 0:
        raise ValueError("LF_SAMPLE_SIZE must be non-negative or None.")

    lf_data = full_data[full_data["is_hf"].eq(0)]
    hf_data = full_data[full_data["is_hf"].eq(1)]

    if lf_sample_size > len(lf_data):
        raise ValueError(
            f"LF_SAMPLE_SIZE={lf_sample_size} requested, "
            f"but only {len(lf_data)} LF rows exist."
        )

    lf_sample = lf_data.sample(n=lf_sample_size, random_state=random_state)
    sampled = full_data.loc[
        lf_sample.index.union(hf_data.index).sort_values()
    ].copy()
    sampled.attrs.update(attrs)
    return sampled


def save_run_config(full_data, cv_data):
    lf_sample = cv_data[cv_data["is_hf"].eq(0)]
    hf_sample = cv_data[cv_data["is_hf"].eq(1)]

    lf_sample_indices = pd.DataFrame({"source_index": lf_sample.index})
    lf_sample_indices.to_csv(RESULTS_DIR / "lf_sample_source_indices.csv", index=False)

    config = {
        "lf_sample_size": LF_SAMPLE_SIZE,
        "lf_sample_random_state": LF_SAMPLE_RANDOM_STATE,
        "n_lf_available": int(full_data["is_hf"].eq(0).sum()),
        "n_hf_available": int(full_data["is_hf"].eq(1).sum()),
        "n_lf_used": int(len(lf_sample)),
        "n_hf_used": int(len(hf_sample)),
        "response_dim": RESPONSE_DIM,
        "model1_response_indexes": response_index_config(MODEL1_RESPONSE_INDEXES),
        "model2_response_indexes": response_index_config(MODEL2_RESPONSE_INDEXES),
    }

    (RESULTS_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def run_model1_fold(split, response_indexes=None):
    """Fit one scalar GP per wavelength for this fold."""
    response_indexes = selected_response_indexes(response_indexes)

    train_data = split["train_data"]
    validation_data = split["validation_data"]
    x_columns = create_data_indexes(train_data)["x_columns"]
    wavelength_map = train_data.attrs["wavelength_map"]

    rows = []
    for response_index in response_indexes:
        response_col = f"response_{response_index}"
        wavelength = float(wavelength_map[response_index])

        train_response = select_scalar_response(
            train_data,
            x_columns=x_columns,
            response_col=response_col,
        )
        validation_response = select_scalar_response(
            validation_data,
            x_columns=x_columns,
            response_col=response_col,
        )
        validate_model1_wavelength_data(train_response, response_col)
        validate_model1_wavelength_data(validation_response, response_col)

        y_true = float(validation_response.iloc[0, -1])

        for variant, hf_only in VARIANTS.items():
            fit_model1(train_response, HF_only=hf_only)
            prediction = predict_model1(validation_response, HF_only=hf_only)

            y_pred = float(prediction["mu"][0])
            variance = float(prediction["var"][0])
            rows.append(
                prediction_record(
                    split=split,
                    model="model1",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=y_true,
                    y_pred=y_pred,
                    variance=variance,
                )
            )

    return rows


def select_scalar_response(data, x_columns, response_col):
    scalar_data = data.loc[:, [*x_columns, "is_hf", response_col]].copy()
    scalar_data.attrs.update(data.attrs)
    return scalar_data


def validate_model1_wavelength_data(data, response_col):
    if list(data.columns[-2:]) != ["is_hf", response_col]:
        raise ValueError(
            "Model 1 wavelength-wise data must end with "
            f"['is_hf', {response_col!r}]."
        )


def run_model2_fold(split, response_indexes=None):
    """Fit one augmented GP over atmospheric inputs plus wavelength."""
    train_long = prepare_model2_fold_data(split["train_data"], response_indexes)
    validation_long = prepare_model2_fold_data(
        split["validation_data"],
        response_indexes,
    )

    wavelength_map = split["train_data"].attrs["wavelength_map"]
    response_index_by_wavelength = {
        float(wavelength): int(response_index)
        for response_index, wavelength in wavelength_map.items()
    }

    rows = []
    for variant, hf_only in VARIANTS.items():
        fit_model1(train_long, HF_only=hf_only)
        prediction = predict_model1(validation_long, HF_only=hf_only)

        validation_rows = validation_long.reset_index(drop=True)
        for row_position, validation_row in validation_rows.iterrows():
            wavelength = float(validation_row["wavelength"])
            response_index = response_index_by_wavelength[wavelength]
            y_true = float(validation_row["response"])
            y_pred = float(prediction["mu"][row_position])
            variance = float(prediction["var"][row_position])

            rows.append(
                prediction_record(
                    split=split,
                    model="model2",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=y_true,
                    y_pred=y_pred,
                    variance=variance,
                )
            )

    return rows


def prepare_model2_fold_data(data, response_indexes=None):
    long_data = prepare_model2_data(data, LF_number=None)

    if response_indexes is None:
        validate_model2_augmented_data(long_data)
        return long_data

    response_indexes = selected_response_indexes(response_indexes)
    wavelength_map = data.attrs["wavelength_map"]
    wavelengths = {float(wavelength_map[i]) for i in response_indexes}
    filtered = long_data[long_data["wavelength"].isin(wavelengths)].copy()
    filtered.attrs.update(long_data.attrs)
    validate_model2_augmented_data(filtered)
    return filtered


def validate_model2_augmented_data(data):
    if list(data.columns[-2:]) != ["is_hf", "response"]:
        raise ValueError("Model 2 augmented data must end with ['is_hf', 'response'].")
    if "wavelength" not in data.columns[:-2]:
        raise ValueError("Model 2 augmented data must include wavelength as an input.")


def prediction_record(
    split,
    model,
    variant,
    response_index,
    wavelength,
    y_true,
    y_pred,
    variance,
):
    return {
        "fold": split["fold"],
        "held_out_source_index": split["held_out_source_index"],
        "model": model,
        "variant": variant,
        "response_index": response_index,
        "wavelength": wavelength,
        "y_true": y_true,
        "y_pred": y_pred,
        "variance": variance,
        "residual": y_pred - y_true,
    }


def per_wavelength_metrics(predictions):
    metrics = (
        predictions
        .groupby(["model", "variant", "response_index", "wavelength"])
        .agg(
            n=("y_true", "size"),
            y_true_min=("y_true", "min"),
            y_true_max=("y_true", "max"),
            rmse=("residual", rmse),
            mae=("residual", mean_absolute),
            bias=("residual", "mean"),
            max_abs_error=("residual", max_absolute),
        )
        .reset_index()
    )

    y_range = metrics["y_true_max"] - metrics["y_true_min"]
    metrics["nrmse"] = metrics["rmse"] / y_range.replace(0, np.nan)

    return metrics[
        [
            "model",
            "variant",
            "response_index",
            "wavelength",
            "n",
            "rmse",
            "nrmse",
            "mae",
            "bias",
            "max_abs_error",
            "y_true_min",
            "y_true_max",
        ]
    ]


def nrmse_comparison(metrics):
    working = metrics.copy()
    working["column"] = (
        working["model"] + "_" + working["variant"] + "_nrmse"
    )

    comparison = (
        working
        .pivot(
            index=["response_index", "wavelength"],
            columns="column",
            values="nrmse",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    expected_columns = [
        "response_index",
        "wavelength",
        "model1_sf_nrmse",
        "model1_mf_nrmse",
        "model2_sf_nrmse",
        "model2_mf_nrmse",
    ]
    present_columns = [
        column for column in expected_columns if column in comparison.columns
    ]
    return comparison[present_columns]


def response_index_config(response_indexes):
    if response_indexes is None:
        return "all"
    return list(response_indexes)


def selected_response_indexes(response_indexes):
    if response_indexes is None:
        return range(RESPONSE_DIM)

    indexes = list(response_indexes)
    if not indexes:
        raise ValueError("Response index selection cannot be empty.")

    non_integer = [i for i in indexes if not isinstance(i, (int, np.integer))]
    if non_integer:
        raise ValueError(f"Response indexes must be integers; got {non_integer}.")

    invalid = [i for i in indexes if i < 0 or i >= RESPONSE_DIM]
    if invalid:
        raise ValueError(
            f"Response indexes must be between 0 and {RESPONSE_DIM - 1}; "
            f"got {invalid}."
        )

    return indexes


def sf_mf_comparison(metrics, model):
    subset = metrics[metrics["model"].eq(model)]
    sf = subset[subset["variant"].eq("sf")][
        ["response_index", "wavelength", "n", "rmse", "nrmse"]
    ].rename(
        columns={
            "n": "n_hf",
            "rmse": "sf_rmse",
            "nrmse": "sf_nrmse",
        }
    )
    mf = subset[subset["variant"].eq("mf")][
        ["response_index", "wavelength", "rmse", "nrmse"]
    ].rename(
        columns={
            "rmse": "mf_rmse",
            "nrmse": "mf_nrmse",
        }
    )

    comparison = sf.merge(mf, on=["response_index", "wavelength"])
    comparison["delta_mf_minus_sf"] = (
        comparison["mf_nrmse"] - comparison["sf_nrmse"]
    )
    comparison["winner"] = np.select(
        [
            comparison["mf_nrmse"] < comparison["sf_nrmse"],
            comparison["sf_nrmse"] < comparison["mf_nrmse"],
        ],
        ["mf", "sf"],
        default="tie",
    )

    return comparison[
        [
            "response_index",
            "wavelength",
            "sf_nrmse",
            "mf_nrmse",
            "delta_mf_minus_sf",
            "winner",
            "sf_rmse",
            "mf_rmse",
            "n_hf",
        ]
    ]


def overall_summary(metrics):
    return (
        metrics
        .groupby(["model", "variant"])
        .agg(
            n_wavelengths=("response_index", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .reset_index()
    )


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def mean_absolute(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values)))


def max_absolute(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values)))


if __name__ == "__main__":
    main()
