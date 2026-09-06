"""5-fold learning-curve CV for current Model 1 and Model 2.

This version checkpoints progressively:
- every completed Model 1 wavelength job is appended to predictions CSVs;
- aggregate metric CSVs are refreshed after each completed fold/sample-size block;
- if the run is interrupted, completed jobs are skipped when rerun.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import modelling.cv.cv_5fold as cv_helpers
from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_1 import fit_model1, predict_model1
from modelling.cv.cv_5fold import (
    sample_lf_rows,
    response_index_config,
    rmse,
    mean_absolute,
    max_absolute,
)


CPU = 8

# Toggle which models and variants are run.
# This default runs only Model 1 SF vs MF.
RUN_MODEL1 = True
RUN_MODEL2 = False
RUN_SF = True
RUN_MF = True

# Set this to a string only if you want to force a specific output folder.
RUN_NAME = "5-fold_1_SF_vs_MF"

N_FOLDS = 5
CV_RANDOM_STATE = 2026

# Same meaning as in cv_5fold.py. Set to None to use all LF rows.
LF_SAMPLE_SIZE = None
LF_SAMPLE_RANDOM_STATE = 42

# Five HF learning-curve points. In 5-fold CV, the smallest full training
# fold has 77 HF rows, so 77 is the largest common leakage-free setting.
HF_TRAIN_SIZES = [10, 20, 40, 70, 77]
HF_SAMPLE_RANDOM_STATE = 123

RUN_LABEL = RUN_NAME or "_".join(
    [
        *[
            label
            for label, enabled in [
                ("model1", RUN_MODEL1),
                ("model2", RUN_MODEL2),
                ("sf", RUN_SF),
                ("mf", RUN_MF),
            ]
            if enabled
        ],
        f"{N_FOLDS}fold",
        "lf_all" if LF_SAMPLE_SIZE is None else f"lf_{LF_SAMPLE_SIZE}",
        "hf_" + "_".join(str(size) for size in HF_TRAIN_SIZES),
    ]
) or "no_models"

RESULTS_DIR = PROJECT_ROOT / "results" / "cv_learning_curve" / RUN_LABEL

# None means all 195 wavelengths. Use e.g. [0, 10, 50] for a quick test.
MODEL1_RESPONSE_INDEXES = None
MODEL2_RESPONSE_INDEXES = None


PREDICTIONS_FILE = RESULTS_DIR / "learning_curve_predictions.csv"
COMPLETED_JOBS_FILE = RESULTS_DIR / "completed_jobs.csv"
METRICS_FILE = RESULTS_DIR / "learning_curve_metrics_per_wavelength.csv"
NRMSE_FILE = RESULTS_DIR / "learning_curve_nrmse_per_wavelength.csv"
SUMMARY_FILE = RESULTS_DIR / "learning_curve_summary.csv"


def main():
    validate_run_selection()

    full_data = load_full_data()
    cv_data = sample_lf_rows(
        full_data,
        lf_sample_size=LF_SAMPLE_SIZE,
        random_state=LF_SAMPLE_RANDOM_STATE,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config(full_data, cv_data)

    splits = list(iter_hf_kfold_splits(cv_data))
    validate_hf_train_sizes(splits)
    completed_jobs = load_completed_jobs()
    jobs = build_jobs(splits, completed_jobs)
    total_jobs = count_total_jobs(splits)
    print(
        "Running "
        f"models={active_model_names()}, "
        f"variants={list(active_variants())}, "
        f"results_dir={RESULTS_DIR}",
        flush=True,
    )
    print(
        f"{len(completed_jobs)} of {total_jobs} checkpoint jobs already complete; "
        f"{len(jobs)} checkpoint jobs left.",
        flush=True,
    )

    if not jobs:
        refresh_progress_outputs()
        print(f"All jobs complete. Results are in {RESULTS_DIR}")
        return

    if CPU < 1:
        raise ValueError("CPU must be at least 1.")

    if CPU == 1:
        for job in jobs:
            rows = run_learning_curve_job(job)
            save_progress(rows, job, total_jobs)
    else:
        with ProcessPoolExecutor(max_workers=CPU) as executor:
            job_iter = iter(jobs)
            future_to_job = {}

            for _ in range(min(CPU, len(jobs))):
                submit_next_job(executor, job_iter, future_to_job)

            while future_to_job:
                for future in as_completed(future_to_job):
                    job = future_to_job.pop(future)
                    rows = future.result()
                    save_progress(rows, job, total_jobs)
                    submit_next_job(executor, job_iter, future_to_job)
                    break

    print(f"Saved learning-curve predictions to {PREDICTIONS_FILE}")
    print(f"Saved learning-curve metrics to {RESULTS_DIR}")


def submit_next_job(executor, job_iter, future_to_job):
    try:
        job = next(job_iter)
    except StopIteration:
        return

    future = executor.submit(run_learning_curve_job, job)
    future_to_job[future] = job


def build_jobs(splits, completed_jobs):
    jobs = []

    for split in splits:
        for n_hf_train in HF_TRAIN_SIZES:
            if RUN_MODEL1:
                for response_index in model1_response_indexes():
                    job = make_job(split, n_hf_train, "model1", response_index)
                    if job_key_from_job(job) not in completed_jobs:
                        jobs.append(job)

            if RUN_MODEL2:
                job = make_job(split, n_hf_train, "model2", "all")
                if job_key_from_job(job) not in completed_jobs:
                    jobs.append(job)

    return jobs


def count_total_jobs(splits):
    jobs_per_fold_and_size = 0
    if RUN_MODEL1:
        jobs_per_fold_and_size += len(model1_response_indexes())
    if RUN_MODEL2:
        jobs_per_fold_and_size += 1
    return len(splits) * len(HF_TRAIN_SIZES) * jobs_per_fold_and_size


def make_job(split, n_hf_train, model, response_index):
    return {
        "split": split,
        "fold": int(split["fold"]),
        "n_hf_train": int(n_hf_train),
        "model": model,
        "response_index": response_index,
    }


def run_learning_curve_job(job):
    split = job["split"]
    n_hf_train = job["n_hf_train"]
    model = job["model"]
    response_index = job["response_index"]
    response_label = (
        f", response {int(response_index):03d}"
        if response_index != "all"
        else ""
    )
    print(
        f"Starting n_hf_train={n_hf_train:02d}, fold {split['fold']:03d}, "
        f"held-out HF samples={len(split['validation_data'])}, "
        f"{model}{response_label}",
        flush=True,
    )

    learning_split = subset_hf_training_rows(split, n_hf_train)
    rows = []
    cv_helpers.VARIANTS = active_variants()

    if model == "model1":
        rows.extend(run_model1_response_fold(learning_split, int(response_index)))
    elif model == "model2":
        rows.extend(run_model2_kfold(learning_split, MODEL2_RESPONSE_INDEXES))
    else:
        raise ValueError(f"Unknown model job {model!r}.")

    for row in rows:
        row["n_hf_train"] = n_hf_train
        row["n_lf_train"] = int(learning_split["train_data"]["is_hf"].eq(0).sum())

    return rows


def validate_run_selection():
    if not RUN_MODEL1 and not RUN_MODEL2:
        raise ValueError("At least one of RUN_MODEL1 or RUN_MODEL2 must be True.")
    if not RUN_SF and not RUN_MF:
        raise ValueError("At least one of RUN_SF or RUN_MF must be True.")


def active_variants():
    variants = {}
    if RUN_SF:
        variants["sf"] = True
    if RUN_MF:
        variants["mf"] = False
    return variants


def active_model_names():
    models = []
    if RUN_MODEL1:
        models.append("model1")
    if RUN_MODEL2:
        models.append("model2")
    return models


def iter_hf_kfold_splits(full_data):
    attrs = full_data.attrs.copy()
    full_data = full_data.reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )
    full_data.attrs.update(attrs)

    hf_positions = full_data.index[full_data["is_hf"].eq(1)].to_numpy()
    if len(hf_positions) != 97:
        raise ValueError(f"Expected 97 HF samples, found {len(hf_positions)}")
    if N_FOLDS < 2:
        raise ValueError("N_FOLDS must be at least 2.")
    if N_FOLDS > len(hf_positions):
        raise ValueError("N_FOLDS cannot exceed the number of HF samples.")

    rng = np.random.default_rng(CV_RANDOM_STATE)
    shuffled_hf_positions = rng.permutation(hf_positions)

    for fold, validation_positions in enumerate(
        np.array_split(shuffled_hf_positions, N_FOLDS),
        start=1,
    ):
        validation_positions = np.sort(validation_positions)
        train_data = full_data.drop(index=validation_positions).reset_index(drop=True)
        validation_data = full_data.loc[validation_positions].reset_index(drop=True)
        train_data.attrs.update(attrs)
        validation_data.attrs.update(attrs)

        yield {
            "fold": fold,
            "held_out_source_index": validation_data["source_index"].tolist(),
            "train_data": train_data,
            "validation_data": validation_data,
        }


def validate_hf_train_sizes(splits):
    min_hf_train = min(
        int(split["train_data"]["is_hf"].eq(1).sum())
        for split in splits
    )
    too_large = [size for size in HF_TRAIN_SIZES if size > min_hf_train]
    if too_large:
        raise ValueError(
            f"HF_TRAIN_SIZES contains values larger than the smallest "
            f"5-fold HF training set ({min_hf_train}): {too_large}."
        )


def model1_response_indexes():
    return list(cv_helpers.selected_response_indexes(MODEL1_RESPONSE_INDEXES))


def model2_response_indexes(response_indexes):
    return list(cv_helpers.selected_response_indexes(response_indexes))


def run_model1_response_fold(split, response_index):
    train_data = split["train_data"]
    validation_data = split["validation_data"]
    x_columns = create_data_indexes(train_data)["x_columns"]
    response_col = f"response_{response_index}"
    wavelength = float(train_data.attrs["wavelength_map"][response_index])

    train_response = cv_helpers.select_scalar_response(
        train_data,
        x_columns=x_columns,
        response_col=response_col,
    )
    validation_response = cv_helpers.select_scalar_response(
        validation_data,
        x_columns=x_columns,
        response_col=response_col,
    )
    cv_helpers.validate_model1_wavelength_data(train_response, response_col)
    cv_helpers.validate_model1_wavelength_data(validation_response, response_col)

    rows = []
    source_indices = validation_data["source_index"].to_numpy()
    y_true = validation_response.iloc[:, -1].to_numpy(dtype=float)

    for variant, hf_only in active_variants().items():
        fit_model1(train_response, HF_only=hf_only)
        prediction = predict_model1(validation_response, HF_only=hf_only)

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

    return rows


def run_model2_kfold(split, response_indexes=None):
    train_long = cv_helpers.prepare_model2_fold_data(
        split["train_data"],
        response_indexes,
    )
    validation_long = cv_helpers.prepare_model2_fold_data(
        split["validation_data"],
        response_indexes,
    )

    wavelength_map = split["train_data"].attrs["wavelength_map"]
    response_index_by_wavelength = {
        float(wavelength): int(response_index)
        for response_index, wavelength in wavelength_map.items()
    }
    source_indices = repeated_validation_source_indices(
        split["validation_data"],
        response_indexes,
    )

    rows = []
    validation_rows = validation_long.reset_index(drop=True)
    for variant, hf_only in active_variants().items():
        fit_model1(train_long, HF_only=hf_only)
        prediction = predict_model1(validation_long, HF_only=hf_only)

        for row_position, validation_row in validation_rows.iterrows():
            wavelength = float(validation_row["wavelength"])
            response_index = response_index_by_wavelength[wavelength]
            rows.append(
                prediction_record(
                    split=split,
                    held_out_source_index=source_indices[row_position],
                    model="model2",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=float(validation_row["response"]),
                    y_pred=float(prediction["mu"][row_position]),
                    variance=float(prediction["var"][row_position]),
                )
            )

    return rows


def repeated_validation_source_indices(validation_data, response_indexes=None):
    n_responses = len(model2_response_indexes(response_indexes))
    return np.repeat(validation_data["source_index"].to_numpy(), n_responses)


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


def subset_hf_training_rows(split, n_hf_train):
    train_data = split["train_data"]
    attrs = train_data.attrs.copy()

    hf_positions = train_data.index[train_data["is_hf"].eq(1)].to_numpy()
    lf_positions = train_data.index[train_data["is_hf"].eq(0)].to_numpy()

    if n_hf_train < 1:
        raise ValueError("n_hf_train must be at least 1.")
    if n_hf_train > len(hf_positions):
        raise ValueError(
            f"Requested n_hf_train={n_hf_train}, but this fold only has "
            f"{len(hf_positions)} HF training rows."
        )

    rng = np.random.default_rng(HF_SAMPLE_RANDOM_STATE + int(split["fold"]))
    shuffled_hf_positions = rng.permutation(hf_positions)
    selected_hf_positions = shuffled_hf_positions[:n_hf_train]

    selected_positions = np.sort(
        np.concatenate([lf_positions, selected_hf_positions])
    )

    sampled_train = train_data.loc[selected_positions].reset_index(drop=True).copy()
    sampled_train.attrs.update(attrs)

    validation_data = split["validation_data"].copy()
    validation_data.attrs.update(split["validation_data"].attrs)

    return {
        **split,
        "train_data": sampled_train,
        "validation_data": validation_data,
    }


def save_progress(rows, job, total_jobs):
    predictions = pd.DataFrame(rows)
    predictions = predictions[
        [
            "n_hf_train",
            "n_lf_train",
            "fold",
            "held_out_source_index",
            "model",
            "variant",
            "response_index",
            "wavelength",
            "y_true",
            "y_pred",
            "variance",
            "residual",
        ]
    ].sort_values(["model", "response_index", "variant"])

    append_csv(predictions, PREDICTIONS_FILE)
    append_csv(
        pd.DataFrame(
            [
                {
                    "fold": job["fold"],
                    "n_hf_train": job["n_hf_train"],
                    "model": job["model"],
                    "response_index": job["response_index"],
                }
            ]
        ),
        COMPLETED_JOBS_FILE,
    )

    completed_jobs = load_completed_jobs()
    refreshed = False
    if fold_size_complete(job, completed_jobs):
        refresh_progress_outputs()
        refreshed = True

    completed = len(completed_jobs)
    refresh_note = " metrics refreshed." if refreshed else ""
    print(
        f"Saved n_hf_train={job['n_hf_train']:02d}, fold {job['fold']:03d}, "
        f"{job['model']} response={job['response_index']} "
        f"({completed}/{total_jobs} jobs complete).{refresh_note}",
        flush=True,
    )


def append_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a",
        index=False,
        header=not path.exists(),
    )


def refresh_progress_outputs():
    if not PREDICTIONS_FILE.exists():
        return

    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions = predictions.drop_duplicates(
        subset=[
            "n_hf_train",
            "fold",
            "held_out_source_index",
            "model",
            "variant",
            "response_index",
        ],
        keep="last",
    )
    predictions = predictions.sort_values(
        [
            "n_hf_train",
            "fold",
            "held_out_source_index",
            "model",
            "response_index",
            "variant",
        ]
    )
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics = per_wavelength_metrics(predictions)
    metrics.to_csv(METRICS_FILE, index=False)

    nrmse = nrmse_comparison(metrics)
    nrmse.to_csv(NRMSE_FILE, index=False)

    if RUN_SF and RUN_MF:
        for model in active_model_names():
            comparison = sf_mf_comparison(metrics, model)
            comparison.to_csv(
                RESULTS_DIR / (
                    f"{model}_learning_curve_sf_vs_mf_nrmse_per_wavelength.csv"
                ),
                index=False,
            )

    summary = overall_summary(metrics)
    summary.to_csv(SUMMARY_FILE, index=False)


def load_completed_jobs():
    if not COMPLETED_JOBS_FILE.exists():
        return set()

    completed = pd.read_csv(COMPLETED_JOBS_FILE)
    if completed.empty:
        return set()

    required_columns = {"fold", "n_hf_train", "model", "response_index"}
    if not required_columns.issubset(completed.columns):
        return set()

    completed = completed.drop_duplicates(
        subset=["fold", "n_hf_train", "model", "response_index"]
    )
    completed.to_csv(COMPLETED_JOBS_FILE, index=False)
    return {
        job_key(row.fold, row.n_hf_train, row.model, row.response_index)
        for row in completed.itertuples(index=False)
    }


def fold_size_complete(job, completed_jobs):
    expected_jobs = fold_size_job_keys(job["fold"], job["n_hf_train"])
    return expected_jobs.issubset(completed_jobs)


def fold_size_job_keys(fold, n_hf_train):
    keys = set()

    if RUN_MODEL1:
        for response_index in model1_response_indexes():
            keys.add(job_key(fold, n_hf_train, "model1", response_index))

    if RUN_MODEL2:
        keys.add(job_key(fold, n_hf_train, "model2", "all"))

    return keys


def job_key_from_job(job):
    return job_key(
        job["fold"],
        job["n_hf_train"],
        job["model"],
        job["response_index"],
    )


def job_key(fold, n_hf_train, model, response_index):
    return (
        int(fold),
        int(n_hf_train),
        str(model),
        normalize_response_index(response_index),
    )


def normalize_response_index(response_index):
    if str(response_index) == "all":
        return "all"
    return str(int(response_index))


def save_run_config(full_data, cv_data):
    lf_sample = cv_data[cv_data["is_hf"].eq(0)]
    hf_sample = cv_data[cv_data["is_hf"].eq(1)]

    config = {
        "cv_method": "hf_kfold",
        "n_folds": N_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "run_label": RUN_LABEL,
        "run_model1": RUN_MODEL1,
        "run_model2": RUN_MODEL2,
        "run_sf": RUN_SF,
        "run_mf": RUN_MF,
        "run_name": RUN_NAME,
        "active_variants": list(active_variants()),
        "active_models": active_model_names(),
        "lf_sample_size": LF_SAMPLE_SIZE,
        "lf_sample_random_state": LF_SAMPLE_RANDOM_STATE,
        "hf_train_sizes": HF_TRAIN_SIZES,
        "hf_sample_random_state": HF_SAMPLE_RANDOM_STATE,
        "n_lf_available": int(full_data["is_hf"].eq(0).sum()),
        "n_hf_available": int(full_data["is_hf"].eq(1).sum()),
        "n_lf_used": int(len(lf_sample)),
        "n_hf_used_for_cv": int(len(hf_sample)),
        "response_dim": RESPONSE_DIM,
        "model1_response_indexes": response_index_config(MODEL1_RESPONSE_INDEXES),
        "model2_response_indexes": response_index_config(MODEL2_RESPONSE_INDEXES),
    }

    config_file = RESULTS_DIR / "run_config.json"
    if config_file.exists():
        existing_config = json.loads(config_file.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError(
                f"{config_file} already exists with different run settings. "
                "Use a different toggle combination, set RUN_NAME, or remove "
                "that results folder before rerunning."
            )

    pd.DataFrame({"source_index": lf_sample.index}).to_csv(
        RESULTS_DIR / "lf_sample_source_indices.csv",
        index=False,
    )

    config_file.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def per_wavelength_metrics(predictions):
    metrics = (
        predictions
        .groupby(["n_hf_train", "model", "variant", "response_index", "wavelength"])
        .agg(
            n_cv=("y_true", "size"),
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
            "n_hf_train",
            "model",
            "variant",
            "response_index",
            "wavelength",
            "n_cv",
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
    working["column"] = working["model"] + "_" + working["variant"] + "_nrmse"

    comparison = (
        working
        .pivot(
            index=["n_hf_train", "response_index", "wavelength"],
            columns="column",
            values="nrmse",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    expected_columns = [
        "n_hf_train",
        "response_index",
        "wavelength",
        "model1_sf_nrmse",
        "model1_mf_nrmse",
        "model2_sf_nrmse",
        "model2_mf_nrmse",
    ]
    return comparison[[c for c in expected_columns if c in comparison.columns]]


def sf_mf_comparison(metrics, model):
    subset = metrics[metrics["model"].eq(model)]

    sf = subset[subset["variant"].eq("sf")][
        ["n_hf_train", "response_index", "wavelength", "n_cv", "rmse", "nrmse"]
    ].rename(columns={"rmse": "sf_rmse", "nrmse": "sf_nrmse"})

    mf = subset[subset["variant"].eq("mf")][
        ["n_hf_train", "response_index", "wavelength", "rmse", "nrmse"]
    ].rename(columns={"rmse": "mf_rmse", "nrmse": "mf_nrmse"})

    comparison = sf.merge(mf, on=["n_hf_train", "response_index", "wavelength"])
    comparison["delta_mf_minus_sf"] = comparison["mf_nrmse"] - comparison["sf_nrmse"]
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
            "n_hf_train",
            "response_index",
            "wavelength",
            "sf_nrmse",
            "mf_nrmse",
            "delta_mf_minus_sf",
            "winner",
            "sf_rmse",
            "mf_rmse",
            "n_cv",
        ]
    ]


def overall_summary(metrics):
    return (
        metrics
        .groupby(["n_hf_train", "model", "variant"])
        .agg(
            completed_validation_rows=("n_cv", "max"),
            n_wavelengths=("response_index", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
