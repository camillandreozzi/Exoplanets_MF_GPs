"""5-fold high-fidelity cross-validation for Model 1, Model 2, and Model 3.

Model 1 is wavelength-wise: each response_i is fit with its own scalar GP.
Model 2 is augmented: one GP per fold uses wavelength as an extra input.
Model 3 is wavelength-wise: a tuned boosted fixed effect plus a GP residual.

Edit LF_SAMPLE_SIZE to change how many low-fidelity rows are used in each
multi-fidelity training set. Set it to None to use all low-fidelity rows.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.cv import iter_hf_kfold_splits
from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_1 import fit_model1, predict_model1
from src.model_2 import prepare_model2_data
from src.model_3 import fit_model3, predict_model3

MODEL1_CPU = 8
MODEL2_CPU = 1


LF_SAMPLE_SIZE = 10000
LF_SAMPLE_RANDOM_STATE = 42
N_FOLDS = 5
CV_RANDOM_STATE = 2026
RESULTS_DIR = PROJECT_ROOT / "results" / "cv" / "5fold"
CV_LABEL = f"{N_FOLDS}fold"

# None means all 195 wavelengths. Set N_RESPONSE_SAMPLE to tune the shared
# empirical-density wavelength subset used by both models.
N_RESPONSE_SAMPLE = 15
COMMON_RESPONSE_INDEXES = sorted(
    {
        int(round(index))
        for index in np.linspace(0, RESPONSE_DIM - 1, N_RESPONSE_SAMPLE)
    }
)
MODEL1_RESPONSE_INDEXES = COMMON_RESPONSE_INDEXES
MODEL2_RESPONSE_INDEXES = COMMON_RESPONSE_INDEXES

VARIANTS = {
    "sf": True,
    "mf": False,
}
MODEL3_GP_KWARGS = None
MODEL3_TUNING_KWARGS = None


def log(message):
    print(message, flush=True)


def elapsed_text(start_time):
    return f"{perf_counter() - start_time:.1f}s"


def run_fold(split):
    log(
        "Starting fold "
        f"{split['fold']:03d}, held-out HF samples="
        f"{len(split['validation_data'])}"
    )

    rows = []
    rows.extend(run_model1_fold(split, MODEL1_RESPONSE_INDEXES))
    rows.extend(run_model2_fold(split, MODEL2_RESPONSE_INDEXES))
    return rows


def run_model1_cv_fold(split):
    log(
        "Starting model1 fold "
        f"{split['fold']:03d}, held-out HF samples="
        f"{len(split['validation_data'])}"
    )
    return run_model1_fold(split, MODEL1_RESPONSE_INDEXES)


def run_model2_cv_fold(split):
    log(
        "Starting model2 fold "
        f"{split['fold']:03d}, held-out HF samples="
        f"{len(split['validation_data'])}"
    )
    return run_model2_fold(split, MODEL2_RESPONSE_INDEXES)


def main():
    run_start = perf_counter()
    full_data = load_full_data()
    cv_data = sample_lf_rows(
        full_data,
        lf_sample_size=LF_SAMPLE_SIZE,
        random_state=LF_SAMPLE_RANDOM_STATE,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config(full_data, cv_data)

    log_run_setup(full_data, cv_data)

    prediction_rows = []
    splits = list(
        iter_hf_kfold_splits(
            cv_data,
            n_folds=N_FOLDS,
            random_state=CV_RANDOM_STATE,
        )
    )
    log_fold_setup(splits)

    prediction_rows.extend(
        run_cv_stage(splits, run_model1_cv_fold, MODEL1_CPU, "model1")
    )
    prediction_rows.extend(
        run_cv_stage(splits, run_model2_cv_fold, MODEL2_CPU, "model2")
    )

    log(f"All model stages complete; collected {len(prediction_rows)} prediction rows.")
    log("Sorting predictions and writing CSV outputs...")
    prediction_rows.sort(
        key=lambda row: (
            row["fold"],
            row["held_out_source_index"],
            row["model"],
            row["response_index"],
            row["variant"],
        )
    )

    predictions = pd.DataFrame(prediction_rows)
    predictions_file = RESULTS_DIR / f"{CV_LABEL}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    log(f"Saved CV predictions to {predictions_file} ({len(predictions)} rows)")

    metrics = per_wavelength_metrics(predictions)
    metrics_file = RESULTS_DIR / f"{CV_LABEL}_metrics_per_wavelength.csv"
    metrics.to_csv(metrics_file, index=False)
    log(f"Saved per-wavelength metrics to {metrics_file} ({len(metrics)} rows)")

    nrmse = nrmse_comparison(metrics)
    nrmse_file = RESULTS_DIR / f"{CV_LABEL}_nrmse_per_wavelength.csv"
    nrmse.to_csv(nrmse_file, index=False)
    log(f"Saved NRMSE comparison to {nrmse_file} ({len(nrmse)} rows)")

    for model in ["model1", "model2"]:
        model_comparison = sf_mf_comparison(metrics, model)
        comparison_file = (
            RESULTS_DIR / f"{model}_{CV_LABEL}_sf_vs_mf_nrmse_per_wavelength.csv"
        )
        model_comparison.to_csv(comparison_file, index=False)
        log(f"Saved {model} SF-vs-MF comparison to {comparison_file}")

    summary = overall_summary(metrics)
    summary_file = RESULTS_DIR / f"{CV_LABEL}_summary.csv"
    summary.to_csv(summary_file, index=False)
    log(f"Saved summary to {summary_file} ({len(summary)} rows)")

    log(f"Finished 5-fold CV in {elapsed_text(run_start)}. Results are in {RESULTS_DIR}")


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


def run_cv_stage(splits, fold_runner, cpu, cpu_name):
    if cpu < 1:
        raise ValueError(f"{cpu_name} must be at least 1.")

    stage_start = perf_counter()
    log(f"Starting {cpu_name} stage: folds={len(splits)}, cpu={cpu}")

    rows = []
    if cpu == 1:
        for completed, split in enumerate(splits, start=1):
            fold_start = perf_counter()
            fold_rows = fold_runner(split)
            rows.extend(fold_rows)
            log(
                f"Completed {cpu_name} fold {split['fold']:03d} "
                f"({completed}/{len(splits)}); rows={len(fold_rows)}; "
                f"elapsed={elapsed_text(fold_start)}; stage_rows={len(rows)}"
            )
        log(
            f"Finished {cpu_name} stage in {elapsed_text(stage_start)}; "
            f"rows={len(rows)}"
        )
        return rows

    with ProcessPoolExecutor(max_workers=cpu) as executor:
        future_to_split = {
            executor.submit(fold_runner, split): split
            for split in splits
        }
        future_start_times = {
            future: perf_counter()
            for future in future_to_split
        }
        for completed, future in enumerate(as_completed(future_to_split), start=1):
            split = future_to_split[future]
            fold_rows = future.result()
            rows.extend(fold_rows)
            log(
                f"Completed {cpu_name} fold {split['fold']:03d} "
                f"({completed}/{len(splits)}); rows={len(fold_rows)}; "
                f"elapsed={elapsed_text(future_start_times[future])}; "
                f"stage_rows={len(rows)}"
            )

    log(
        f"Finished {cpu_name} stage in {elapsed_text(stage_start)}; "
        f"rows={len(rows)}"
    )
    return rows


def log_run_setup(full_data, cv_data):
    model1_response_count = len(selected_response_indexes(MODEL1_RESPONSE_INDEXES))
    model2_response_count = len(selected_response_indexes(MODEL2_RESPONSE_INDEXES))
    full_lf = int(full_data["is_hf"].eq(0).sum())
    full_hf = int(full_data["is_hf"].eq(1).sum())
    cv_lf = int(cv_data["is_hf"].eq(0).sum())
    cv_hf = int(cv_data["is_hf"].eq(1).sum())

    log(
        "CV setup: "
        f"folds={N_FOLDS}, variants={list(VARIANTS)}, "
        f"model1_response_indexes={model1_response_count}, "
        f"model2_response_indexes={model2_response_count}, "
        f"model1_cpu={MODEL1_CPU}, model2_cpu={MODEL2_CPU}"
    )
    log(
        "Data setup: "
        f"available LF/HF={full_lf}/{full_hf}, "
        f"used LF/HF={cv_lf}/{cv_hf}, "
        f"results_dir={RESULTS_DIR}"
    )


def log_fold_setup(splits):
    fold_sizes = ", ".join(
        f"{split['fold']:03d}:{len(split['validation_data'])}"
        for split in splits
    )
    log(f"HF validation fold sizes: {fold_sizes}")


def save_run_config(full_data, cv_data):
    lf_sample = cv_data[cv_data["is_hf"].eq(0)]
    hf_sample = cv_data[cv_data["is_hf"].eq(1)]

    lf_sample_indices = pd.DataFrame({"source_index": lf_sample.index})
    lf_sample_indices.to_csv(RESULTS_DIR / "lf_sample_source_indices.csv", index=False)

    config = {
        "cv_method": "hf_kfold",
        "n_folds": N_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "lf_sample_size": LF_SAMPLE_SIZE,
        "lf_sample_random_state": LF_SAMPLE_RANDOM_STATE,
        "n_lf_available": int(full_data["is_hf"].eq(0).sum()),
        "n_hf_available": int(full_data["is_hf"].eq(1).sum()),
        "n_lf_used": int(len(lf_sample)),
        "n_hf_used": int(len(hf_sample)),
        "model1_cpu": MODEL1_CPU,
        "model2_cpu": MODEL2_CPU,
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
    response_indexes = list(selected_response_indexes(response_indexes))

    train_data = split["train_data"]
    validation_data = split["validation_data"]
    x_columns = create_data_indexes(train_data)["x_columns"]
    wavelength_map = train_data.attrs["wavelength_map"]

    rows = []
    progress_interval = max(1, len(response_indexes) // 10)
    for response_number, response_index in enumerate(response_indexes, start=1):
        response_col = f"response_{response_index}"
        wavelength = float(wavelength_map[response_index])
        if should_log_response_progress(response_number, len(response_indexes)):
            log(
                f"Model1 fold {split['fold']:03d}: "
                f"response {response_number}/{len(response_indexes)}, "
                f"response_index={response_index}, wavelength={wavelength}"
            )

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

        source_indices = validation_data["source_index"].to_numpy()
        y_true = validation_response.iloc[:, -1].to_numpy(dtype=float)

        for variant, hf_only in VARIANTS.items():
            variant_start = perf_counter()
            fit_model1(train_response, HF_only=hf_only)
            prediction = predict_model1(validation_response, HF_only=hf_only)
            if response_number == 1 or response_number == len(response_indexes):
                log(
                    f"Model1 fold {split['fold']:03d}: "
                    f"{variant} response_index={response_index} done "
                    f"in {elapsed_text(variant_start)}"
                )

            for row_position, source_index in enumerate(source_indices):
                rows.append(
                    prediction_record(
                        split=split,
                        held_out_source_index=source_index,
                        model="model1",
                        variant=variant,
                        response_index=response_index,
                        wavelength=wavelength,
                        y_true=float(y_true[row_position]),
                        y_pred=float(prediction["mu"][row_position]),
                        variance=float(prediction["var"][row_position]),
                    )
                )

        if response_number % progress_interval == 0:
            log(
                f"Model1 fold {split['fold']:03d}: "
                f"{response_number}/{len(response_indexes)} responses complete; "
                f"rows={len(rows)}"
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
    fold_start = perf_counter()
    response_count = len(selected_response_indexes(response_indexes))
    log(
        f"Model2 fold {split['fold']:03d}: preparing augmented data "
        f"for {response_count} response indexes"
    )
    prep_start = perf_counter()
    train_long = prepare_model2_fold_data(split["train_data"], response_indexes)
    validation_long = prepare_model2_fold_data(
        split["validation_data"],
        response_indexes,
    )
    log(
        f"Model2 fold {split['fold']:03d}: augmented data ready "
        f"in {elapsed_text(prep_start)}; "
        f"train_rows={len(train_long)}, validation_rows={len(validation_long)}, "
        f"train LF/HF rows="
        f"{int(train_long['is_hf'].eq(0).sum())}/"
        f"{int(train_long['is_hf'].eq(1).sum())}"
    )

    wavelength_map = split["train_data"].attrs["wavelength_map"]
    response_index_by_wavelength = {
        float(wavelength): int(response_index)
        for response_index, wavelength in wavelength_map.items()
    }
    validation_rows = validation_long.reset_index(drop=True)
    source_indices = repeated_validation_source_indices(
        split["validation_data"],
        response_indexes,
    )

    rows = []
    for variant, hf_only in VARIANTS.items():
        variant_start = perf_counter()
        log(
            f"Model2 fold {split['fold']:03d}: fitting {variant} "
            f"with train_rows={len(train_long)}, "
            f"validation_rows={len(validation_long)}"
        )
        fit_model1(train_long, HF_only=hf_only)
        log(
            f"Model2 fold {split['fold']:03d}: {variant} fit complete "
            f"in {elapsed_text(variant_start)}; predicting..."
        )
        predict_start = perf_counter()
        prediction = predict_model1(validation_long, HF_only=hf_only)
        log(
            f"Model2 fold {split['fold']:03d}: {variant} prediction complete "
            f"in {elapsed_text(predict_start)}"
        )

        for row_position, validation_row in validation_rows.iterrows():
            wavelength = float(validation_row["wavelength"])
            response_index = response_index_by_wavelength[wavelength]
            y_true = float(validation_row["response"])
            y_pred = float(prediction["mu"][row_position])
            variance = float(prediction["var"][row_position])

            rows.append(
                prediction_record(
                    split=split,
                    held_out_source_index=source_indices[row_position],
                    model="model2",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=y_true,
                    y_pred=y_pred,
                    variance=variance,
                )
            )

        log(
            f"Model2 fold {split['fold']:03d}: {variant} rows appended; "
            f"fold_rows={len(rows)}; variant_elapsed={elapsed_text(variant_start)}"
        )

    log(
        f"Model2 fold {split['fold']:03d}: complete in {elapsed_text(fold_start)}; "
        f"rows={len(rows)}"
    )
    return rows


def run_model3_fold(split, response_indexes=None):
    """Fit one tuned Model 3 GPBoost model per wavelength for this fold."""
    response_indexes = list(selected_response_indexes(response_indexes))

    train_data = split["train_data"]
    validation_data = split["validation_data"]
    x_columns = create_data_indexes(train_data)["x_columns"]
    wavelength_map = train_data.attrs["wavelength_map"]

    rows = []
    progress_interval = max(1, len(response_indexes) // 10)
    for response_number, response_index in enumerate(response_indexes, start=1):
        response_col = f"response_{response_index}"
        wavelength = float(wavelength_map[response_index])
        if should_log_response_progress(response_number, len(response_indexes)):
            log(
                f"Model3 fold {split['fold']:03d}: "
                f"response {response_number}/{len(response_indexes)}, "
                f"response_index={response_index}, wavelength={wavelength}"
            )

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

        source_indices = validation_data["source_index"].to_numpy()
        y_true = validation_response.iloc[:, -1].to_numpy(dtype=float)

        for variant, hf_only in VARIANTS.items():
            variant_start = perf_counter()
            model = fit_model3(
                train_response,
                HF_only=hf_only,
                gp_kwargs=MODEL3_GP_KWARGS,
                tuning_kwargs=MODEL3_TUNING_KWARGS,
                verbose_eval=False,
            )
            prediction = predict_model3(
                validation_response,
                HF_only=hf_only,
                model=model,
            )
            if response_number == 1 or response_number == len(response_indexes):
                log(
                    f"Model3 fold {split['fold']:03d}: "
                    f"{variant} response_index={response_index} done "
                    f"in {elapsed_text(variant_start)}"
                )

            for row_position, source_index in enumerate(source_indices):
                rows.append(
                    prediction_record(
                        split=split,
                        held_out_source_index=source_index,
                        model="model3",
                        variant=variant,
                        response_index=response_index,
                        wavelength=wavelength,
                        y_true=float(y_true[row_position]),
                        y_pred=float(prediction["mu"][row_position]),
                        variance=float(prediction["var"][row_position]),
                    )
                )

        if response_number % progress_interval == 0:
            log(
                f"Model3 fold {split['fold']:03d}: "
                f"{response_number}/{len(response_indexes)} responses complete; "
                f"rows={len(rows)}"
            )

    return rows


def should_log_response_progress(response_number, total_responses):
    progress_interval = max(1, total_responses // 10)
    return (
        response_number == 1
        or response_number == total_responses
        or response_number % progress_interval == 0
    )


def repeated_validation_source_indices(validation_data, response_indexes=None):
    n_responses = len(selected_response_indexes(response_indexes))
    return np.repeat(validation_data["source_index"].to_numpy(), n_responses)


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
    held_out_source_index,
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
        "held_out_source_index": int(held_out_source_index),
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
        "model3_sf_nrmse",
        "model3_mf_nrmse",
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
