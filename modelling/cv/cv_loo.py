"""Leave-one-out high-fidelity CV for Model 1, Model 2, and Model 3.

Model 1 is wavelength-wise: each response_i is fit with its own scalar GP.
Model 2 is augmented: one GP per fold uses wavelength as an extra input.
Model 3 is wavelength-wise: a tuned boosted fixed effect plus a GP residual.

Scheduling
----------
Model 1 and Model 3 are scheduled as *wavelength lanes*: one task fits a single
wavelength across every LOO fold. Lanes keep task payloads to a two-element
tuple, give 390 well-mixed units of work to balance across the pool, and make
each wavelength independently resumable. A Model 3 lane also tunes once per
fold and reuses that result for both fidelity variants. Model 2 is not
wavelength-separable, so it stays scheduled per fold.

All lanes share one process pool, so the pool never drains between models.
Workers load the data themselves rather than receiving it, so nothing large is
pickled or held in the parent.

Set RUN_MODEL1/RUN_MODEL2/RUN_MODEL3 and RUN_SF/RUN_MF to run models or
variants independently. Edit LF_SAMPLE_SIZE to change how many low-fidelity
rows are used in each multi-fidelity training set. Set it to None to use all
low-fidelity rows. Each completed lane is written immediately under
results/cv/loo/lanes; a rerun skips lanes already on disk unless FRESH_RUN.

Every setting in the configuration block can also come from a CV_LOO_-prefixed
environment variable, which is how the SLURM scripts in euler/ drive this file
without editing it. Those runs also shard: CV_LOO_SHARD_INDEX/SHARD_COUNT give
each array task a stride through the lane list, each task runs with
CV_LOO_WRITE_OUTPUTS=0, and one dependent job with CV_LOO_ASSEMBLE_ONLY=1
assembles every lane file into the summary CSVs.
"""

import os

# Pin the numeric libraries to one thread per process, before numpy or gpboost
# are imported anywhere. GP_THREADS already covers the GPModel, but not
# LightGBM's booster or the TPE tuning CV, which would otherwise start one
# OpenMP thread per core inside every worker.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

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

from modelling.cv import cv_5fold as cv_helpers
from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_1 import fit_model1, model1_cov_pars, predict_model1
from src.model_3 import fit_model3, predict_model3, tune_model3_parameters


def common_response_indexes(n_response_sample):
    if n_response_sample is None:
        return None
    if not isinstance(n_response_sample, (int, np.integer)):
        raise ValueError("N_RESPONSE_SAMPLE must be an integer or None.")
    if n_response_sample < 1:
        raise ValueError("N_RESPONSE_SAMPLE must be at least 1 or None.")

    return sorted(
        {
            int(round(index))
            for index in np.linspace(0, RESPONSE_DIM - 1, n_response_sample)
        }
    )


# ---------------------------------------------------------------------------
# Environment overrides
#
# Every setting below can also be given as an environment variable, so a batch
# job can pick models, fold counts, and output directories without editing this
# file and without a working copy that differs from the one in git. An unset
# variable leaves the literal default alone, so running the script by hand
# behaves exactly as it did before.
# ---------------------------------------------------------------------------

ENV_PREFIX = "CV_LOO_"


def env_value(name):
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_flag(name, default):
    value = env_value(name)
    if value is None:
        return default

    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean, got {value!r}.")


def env_int(name, default):
    """An integer, or None when the variable is set to the string "none"."""
    value = env_value(name)
    if value is None:
        return default
    if value.lower() == "none":
        return None

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"{ENV_PREFIX}{name} must be an integer or 'none', got {value!r}."
        ) from None


def env_path(name, default):
    """A directory. A relative value is taken against the project root."""
    value = env_value(name)
    if value is None:
        return default

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_json(name, default):
    """A JSON object, for the nested keyword-argument settings."""
    value = env_value(name)
    if value is None:
        return default
    if value.lower() == "none":
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{ENV_PREFIX}{name} must be JSON: {error}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{ENV_PREFIX}{name} must be a JSON object.")
    return parsed


# Worker processes. Throughput on this machine saturates around here: the MF
# fits are memory-bandwidth-bound, so 1->4 workers is 3.1x, 4->8 reaches 4.2x,
# and 10 workers only adds another 5% while risking thermal throttling. On a
# cluster this should match the CPUs the job was allocated.
CV_CPU = env_int("CPU", 8)

# Toggle which models and variants are run.
RUN_MODEL1 = env_flag("RUN_MODEL1", False)
RUN_MODEL2 = env_flag("RUN_MODEL2", True)
RUN_MODEL3 = env_flag("RUN_MODEL3", False)
RUN_SF = env_flag("RUN_SF", True)
RUN_MF = env_flag("RUN_MF", True)

WARM_START_MODEL1_MF = env_flag("WARM_START_MODEL1_MF", False)

# Delete existing lane files and start over instead of resuming.
FRESH_RUN = env_flag("FRESH_RUN", False)

# Cap the number of LOO folds. None runs all of them; a small number is useful
# for equivalence checks against a previous run.
MAX_FOLDS = env_int("MAX_FOLDS", None)

LF_SAMPLE_SIZE = env_int("LF_SAMPLE_SIZE", 1000)
LF_SAMPLE_RANDOM_STATE = env_int("LF_SAMPLE_RANDOM_STATE", 42)
# Sample a separate set of wavelengths within every Model 2 training spectrum.
# Held-out HF spectra are still predicted at all 195 wavelengths.
MODEL2_N_WAVELENGTHS = env_int("MODEL2_N_WAVELENGTHS", 50)
MODEL2_WAVELENGTH_RANDOM_STATE = env_int("MODEL2_WAVELENGTH_RANDOM_STATE", 42)
# One results directory per training set. Lane files are named by model, so
# several models can share a directory, but two runs with different LF sample
# sizes must not: they would overwrite each other's run_config.json and leave
# lane files from two different training sets side by side.
RESULTS_DIR = env_path("RESULTS_DIR", PROJECT_ROOT / "results" / "cv" / "loo")
CV_LABEL = "loo"
LANE_RESULTS_DIR = RESULTS_DIR / "lanes"

# Split the lanes across cluster array tasks: shard i of n runs every n-th
# lane. One shard, the default, runs all of them.
SHARD_INDEX = env_int("SHARD_INDEX", 0)
SHARD_COUNT = env_int("SHARD_COUNT", 1)

# An array task fits lanes and leaves assembly to a single dependent job, which
# runs with ASSEMBLE_ONLY and fits nothing.
WRITE_OUTPUTS = env_flag("WRITE_OUTPUTS", True)
ASSEMBLE_ONLY = env_flag("ASSEMBLE_ONLY", False)

# None means all 195 wavelengths. Set N_RESPONSE_SAMPLE to tune the shared
# empirical-density wavelength subset used by all enabled wavelength-wise models.
N_RESPONSE_SAMPLE = env_int("N_RESPONSE_SAMPLE", None)
COMMON_RESPONSE_INDEXES = common_response_indexes(N_RESPONSE_SAMPLE)
MODEL1_RESPONSE_INDEXES = COMMON_RESPONSE_INDEXES
MODEL2_RESPONSE_INDEXES = COMMON_RESPONSE_INDEXES
MODEL3_RESPONSE_INDEXES = COMMON_RESPONSE_INDEXES

# Model 3 tunes automatically inside each lane. Use this to override
# tune_model3_parameters defaults, e.g. {"n_trials": 20, "validation_fraction": 0.2}.
# verbose_eval is forced to 0: fit_model3 does not forward its own verbose_eval
# to the tuner, whose default prints ~100 trial lines per fit.
MODEL3_TUNING_KWARGS = env_json("MODEL3_TUNING_KWARGS", None)
MODEL3_GP_KWARGS = env_json("MODEL3_GP_KWARGS", None)


def log(message):
    print(message, flush=True)


def elapsed_text(start_time):
    return f"{perf_counter() - start_time:.1f}s"


def duration_text(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Worker state
#
# Every worker loads and samples the data itself. Building the 97 splits in the
# parent instead would pin ~1.6 GB there for the whole run and pickle 16.6 MB
# per task; a task payload here is a two-element tuple.
# ---------------------------------------------------------------------------

_X = None
_IS_HF = None
_Y = None
_SOURCE_INDEX = None
_HF_POSITIONS = None
_X_COLUMNS = None
_WAVELENGTH_MAP = None


def init_worker():
    """Load the CV data once per worker process."""
    global _X, _IS_HF, _Y, _SOURCE_INDEX, _HF_POSITIONS, _X_COLUMNS, _WAVELENGTH_MAP

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass

    configure_cv_helpers()

    full_data = load_full_data()
    cv_data = cv_helpers.sample_lf_rows(
        full_data,
        lf_sample_size=LF_SAMPLE_SIZE,
        random_state=LF_SAMPLE_RANDOM_STATE,
    )

    _X_COLUMNS = create_data_indexes(cv_data)["x_columns"]
    _WAVELENGTH_MAP = cv_data.attrs["wavelength_map"]

    # source_index is the row label in the unsampled frame, matching the
    # reset_index(drop=False) that src.cv.with_source_index performs.
    _SOURCE_INDEX = cv_data.index.to_numpy()
    _X = cv_data.loc[:, _X_COLUMNS].to_numpy(dtype=float)
    _IS_HF = cv_data["is_hf"].to_numpy()
    _Y = cv_data.loc[:, [f"response_{i}" for i in range(RESPONSE_DIM)]].to_numpy(
        dtype=float
    )
    _HF_POSITIONS = np.flatnonzero(_IS_HF == 1)

    verify_worker_lf_sample()


def verify_worker_lf_sample():
    """Fail loudly if a worker sampled different LF rows than the parent."""
    recorded_file = RESULTS_DIR / "lf_sample_source_indices.csv"
    if not recorded_file.exists():
        return

    recorded = pd.read_csv(recorded_file)["source_index"].to_numpy()
    sampled = _SOURCE_INDEX[_IS_HF == 0]
    if not np.array_equal(np.sort(recorded), np.sort(sampled)):
        raise RuntimeError(
            "Worker LF sample does not match the sample recorded by the parent; "
            "the run would mix two different training sets."
        )


def iter_folds():
    """(fold number, row position of the held-out HF sample) for each fold.

    Mirrors src.cv.iter_loo_splits: HF rows in frame order, numbered from 1.
    """
    positions = _HF_POSITIONS
    if MAX_FOLDS is not None:
        positions = positions[:MAX_FOLDS]
    return list(enumerate(positions, start=1))


def lane_frame(response_index):
    """The scalar-response frame fit_model1/fit_model3 expect, for one wavelength."""
    frame = pd.DataFrame(
        {name: _X[:, position] for position, name in enumerate(_X_COLUMNS)}
    )
    frame["is_hf"] = _IS_HF
    frame[f"response_{response_index}"] = _Y[:, response_index]
    frame.attrs["wavelength_map"] = _WAVELENGTH_MAP
    return frame


def full_frame():
    """The all-wavelength frame Model 2 needs, with source_index restored."""
    frame = pd.DataFrame({"source_index": _SOURCE_INDEX})
    for position, name in enumerate(_X_COLUMNS):
        frame[name] = _X[:, position]
    frame["is_hf"] = _IS_HF
    for response_index in range(RESPONSE_DIM):
        frame[f"response_{response_index}"] = _Y[:, response_index]
    frame.attrs["wavelength_map"] = _WAVELENGTH_MAP
    return frame


def split_frame(frame, hold_out_position):
    """Drop one row for training and keep it for validation, preserving order."""
    train_data = frame.drop(index=frame.index[hold_out_position])
    validation_data = frame.iloc[[hold_out_position]]
    train_data.attrs.update(frame.attrs)
    validation_data.attrs.update(frame.attrs)
    return train_data, validation_data


# ---------------------------------------------------------------------------
# Lane runners
# ---------------------------------------------------------------------------


def run_model1_lane(response_index):
    """Fit Model 1 for one wavelength across every LOO fold."""
    frame = lane_frame(response_index)
    response_col = f"response_{response_index}"
    wavelength = float(_WAVELENGTH_MAP[response_index])
    folds = iter_folds()

    rows = []
    parameter_rows = []
    warm_start_cov_pars = None
    for fold, hold_out_position in folds:
        train_data, validation_data = split_frame(frame, hold_out_position)
        cv_helpers.validate_model1_wavelength_data(train_data, response_col)
        cv_helpers.validate_model1_wavelength_data(validation_data, response_col)

        source_index = int(_SOURCE_INDEX[hold_out_position])
        y_true = float(validation_data.iloc[0, -1])

        for variant, hf_only in active_variants().items():
            warm_start = None if hf_only else warm_start_cov_pars
            gp_model = fit_model1(
                train_data,
                HF_only=hf_only,
                init_cov_pars=warm_start,
            )
            if not hf_only and WARM_START_MODEL1_MF:
                warm_start_cov_pars = model1_cov_pars(gp_model)

            prediction = predict_model1(validation_data, HF_only=hf_only)
            parameter_rows.append(
                parameter_record(
                    fold=fold,
                    held_out_source_index=source_index,
                    model="model1",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    cov_pars=gp_model.get_cov_pars(std_err=False),
                    neg_log_likelihood=gp_model.get_current_neg_log_likelihood(),
                )
            )
            rows.append(
                cv_helpers.prediction_record(
                    split={"fold": fold},
                    held_out_source_index=source_index,
                    model="model1",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=y_true,
                    y_pred=float(prediction["mu"][0]),
                    variance=float(prediction["var"][0]),
                )
            )

    return rows, parameter_rows


def run_model3_lane(response_index):
    """Fit Model 3 for one wavelength across every LOO fold."""
    frame = lane_frame(response_index)
    response_col = f"response_{response_index}"
    wavelength = float(_WAVELENGTH_MAP[response_index])
    folds = iter_folds()

    rows = []
    parameter_rows = []
    for fold, hold_out_position in folds:
        train_data, validation_data = split_frame(frame, hold_out_position)
        cv_helpers.validate_model1_wavelength_data(train_data, response_col)
        cv_helpers.validate_model1_wavelength_data(validation_data, response_col)

        source_index = int(_SOURCE_INDEX[hold_out_position])
        y_true = float(validation_data.iloc[0, -1])

        # One tuning result per fold, shared by both variants. The boosted mean
        # is trained on HF rows only either way, so tuning does not depend on
        # HF_only and letting fit_model3 tune per variant repeats it verbatim.
        tuning_result = tune_model3_parameters(
            train_data,
            HF_only=True,
            **model3_tuning_options(),
        )

        for variant, hf_only in active_variants().items():
            model = fit_model3(
                train_data,
                HF_only=hf_only,
                tuning_result=tuning_result,
                gp_kwargs=MODEL3_GP_KWARGS,
                verbose_eval=False,
            )
            prediction = predict_model3(
                validation_data,
                HF_only=hf_only,
                model=model,
            )
            parameter_rows.append(
                parameter_record(
                    fold=fold,
                    held_out_source_index=source_index,
                    model="model3",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    cov_pars=model.gp_model.get_cov_pars(std_err=False),
                    neg_log_likelihood=(
                        model.gp_model.get_current_neg_log_likelihood()
                    ),
                    extra=model3_tree_parameters(tuning_result),
                )
            )
            rows.append(
                cv_helpers.prediction_record(
                    split={"fold": fold},
                    held_out_source_index=source_index,
                    model="model3",
                    variant=variant,
                    response_index=response_index,
                    wavelength=wavelength,
                    y_true=y_true,
                    y_pred=float(prediction["mu"][0]),
                    variance=float(prediction["var"][0]),
                )
            )

    return rows, parameter_rows


def run_model2_lane(fold):
    """Fit Model 2 for one LOO fold; it is not separable by wavelength."""
    frame = full_frame()
    hold_out_position = dict(iter_folds())[fold]
    train_data, validation_data = split_frame(frame, hold_out_position)

    split = {
        "fold": fold,
        "held_out_source_index": int(_SOURCE_INDEX[hold_out_position]),
        "train_data": train_data.reset_index(drop=True),
        "validation_data": validation_data.reset_index(drop=True),
    }
    split["train_data"].attrs.update(frame.attrs)
    split["validation_data"].attrs.update(frame.attrs)

    # No parameter rows: run_model2_fold is shared with the 5-fold script and
    # does not hand back the fitted GP, so capturing its covariance parameters
    # would mean changing that shared helper.
    return cv_helpers.run_model2_fold(
        split,
        MODEL2_RESPONSE_INDEXES,
        train_n_wavelengths=MODEL2_N_WAVELENGTHS,
        train_wavelength_random_state=MODEL2_WAVELENGTH_RANDOM_STATE,
    ), []


def cov_par_values(cov_pars):
    """Fitted covariance parameters as {name: value}, whatever the kernel is.

    Names differ by variant: SF Matern gives Error_var/GP_var/GP_range, while
    the MF AR(1) kernel also gives the low-fidelity and discrepancy terms plus
    rho, the coefficient linking the two fidelities.
    """
    if hasattr(cov_pars, "columns"):
        names = [str(name) for name in cov_pars.columns]
    else:
        names = None

    values = np.asarray(cov_pars, dtype=float).ravel()
    if names is None:
        names = [f"cov_par_{position}" for position in range(len(values))]

    return {name: float(value) for name, value in zip(names, values)}


def parameter_record(
    fold,
    held_out_source_index,
    model,
    variant,
    response_index,
    wavelength,
    cov_pars,
    neg_log_likelihood,
    extra=None,
):
    """One row describing the model actually fitted for this fold/wavelength."""
    record = {
        "fold": fold,
        "held_out_source_index": int(held_out_source_index),
        "model": model,
        "variant": variant,
        "response_index": response_index,
        "wavelength": wavelength,
        "neg_log_likelihood": float(neg_log_likelihood),
    }
    record.update(cov_par_values(cov_pars))
    if extra:
        record.update(extra)
    return record


def model3_tree_parameters(tuning_result):
    """The tuned boosting hyperparameters chosen for this fold/wavelength."""
    parameters = {"tree_num_boost_round": tuning_result.get("best_iter")}
    parameters.update(
        {
            f"tree_{name}": value
            for name, value in (tuning_result.get("best_params") or {}).items()
        }
    )
    return parameters


def model3_tuning_options():
    options = dict(MODEL3_TUNING_KWARGS or {})
    options.setdefault("gp_kwargs", MODEL3_GP_KWARGS)
    # fit_model3 does not forward its own verbose_eval to the tuner, whose
    # default prints ~100 trial lines for every fit.
    options["verbose_eval"] = 0
    return options


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

LANE_RUNNERS = {
    "model1": run_model1_lane,
    "model2": run_model2_lane,
    "model3": run_model3_lane,
}


def run_lane_task(task):
    """Run one lane in a worker and write its output before returning."""
    model_name, key = task
    start = perf_counter()
    prediction_rows, parameter_rows = LANE_RUNNERS[model_name](key)

    # Parameters first: resuming keys off the predictions file, so a crash
    # between the two writes must not mark the lane complete.
    save_lane_output(parameter_rows, model_name, key, "parameters", sort=False)
    save_lane_output(prediction_rows, model_name, key, "predictions", sort=True)

    return {
        "task": task,
        "rows": len(prediction_rows),
        "elapsed": perf_counter() - start,
    }


def lane_output_file(model_name, key, kind="predictions"):
    if model_name == "model2":
        return LANE_RESULTS_DIR / f"{CV_LABEL}_model2_fold_{key:03d}_{kind}.csv"
    return LANE_RESULTS_DIR / f"{CV_LABEL}_{model_name}_w{key:03d}_{kind}.csv"


def save_lane_output(rows, model_name, key, kind, sort):
    """Write a lane file atomically, so a kill cannot leave a half-written file."""
    if not rows:
        return

    lane_output = pd.DataFrame(rows)
    if sort:
        sort_predictions(lane_output)

    output_file = lane_output_file(model_name, key, kind)
    temporary_file = output_file.with_name(f"{output_file.name}.{os.getpid()}.tmp")
    lane_output.to_csv(temporary_file, index=False)
    os.replace(temporary_file, output_file)


def all_tasks():
    tasks = []
    # Heaviest first: Model 3 lanes run ~3x longer than Model 1 lanes, so
    # starting them last would leave one straggler holding up the pool.
    if RUN_MODEL3:
        tasks.extend(
            ("model3", response_index)
            for response_index in cv_helpers.selected_response_indexes(
                MODEL3_RESPONSE_INDEXES
            )
        )
    if RUN_MODEL1:
        tasks.extend(
            ("model1", response_index)
            for response_index in cv_helpers.selected_response_indexes(
                MODEL1_RESPONSE_INDEXES
            )
        )
    if RUN_MODEL2:
        tasks.extend(("model2", fold) for fold, _ in parent_folds())
    return tasks


def pending_tasks(tasks):
    return [task for task in tasks if not lane_output_file(*task).exists()]


def parent_folds():
    """Fold numbering as the workers will see it, for Model 2 task creation."""
    folds = list(range(1, hf_sample_count() + 1))
    if MAX_FOLDS is not None:
        folds = folds[:MAX_FOLDS]
    return [(fold, None) for fold in folds]


_HF_SAMPLE_COUNT = None


def hf_sample_count():
    if _HF_SAMPLE_COUNT is None:
        raise RuntimeError("hf_sample_count is only available after main() loads data.")
    return _HF_SAMPLE_COUNT


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main():
    global _HF_SAMPLE_COUNT

    validate_run_selection()

    run_start = perf_counter()
    full_data = load_full_data()
    cv_data = cv_helpers.sample_lf_rows(
        full_data,
        lf_sample_size=LF_SAMPLE_SIZE,
        random_state=LF_SAMPLE_RANDOM_STATE,
    )
    _HF_SAMPLE_COUNT = int(cv_data["is_hf"].eq(1).sum())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LANE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_incremental_outputs()
    save_run_config(full_data, cv_data)

    configure_cv_helpers()
    log_run_setup(full_data, cv_data)
    log_loo_setup(cv_data)

    # The workers load their own copy; nothing here needs it any more.
    del full_data, cv_data

    tasks = all_tasks()
    owned = shard_tasks(tasks)
    outstanding = pending_tasks(owned)
    log(
        f"Lanes: {len(tasks)} total, {len(owned)} owned by this shard, "
        f"{len(owned) - len(outstanding)} already on disk, "
        f"{len(outstanding)} to run on {CV_CPU} workers"
    )

    if ASSEMBLE_ONLY:
        log("ASSEMBLE_ONLY: fitting nothing, assembling the lane files on disk.")
    elif outstanding:
        run_lanes(outstanding)
    else:
        log("Nothing to run for this shard.")

    if not WRITE_OUTPUTS:
        log(
            f"Finished lanes in {elapsed_text(run_start)}. Assembly is left to the "
            f"job that runs with CV_LOO_ASSEMBLE_ONLY=1."
        )
        return

    write_outputs()
    log(f"Finished LOO CV in {elapsed_text(run_start)}. Results are in {RESULTS_DIR}")


def shard_tasks(tasks):
    """The lanes this array task owns: every SHARD_COUNT-th lane from its offset.

    Sharding the full task list rather than the pending one keeps the partition
    identical in every array task, whatever each one finds already on disk when
    it starts. Lanes are ordered heaviest-model-first, so taking a stride
    through the list gives every shard a comparable mix of work.
    """
    if SHARD_COUNT == 1:
        return tasks
    return tasks[SHARD_INDEX::SHARD_COUNT]


def run_lanes(tasks):
    stage_start = perf_counter()
    lane_seconds = []
    # One lane raising must not lose the run: a lane is a self-contained unit
    # that writes its own file, and every other lane in the pool represents
    # hours of work. Failed lanes leave no file behind, so a rerun retries them.
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=CV_CPU, initializer=init_worker) as executor:
        future_to_task = {executor.submit(run_lane_task, task): task for task in tasks}

        for completed, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            model_name, key = task

            try:
                result = future.result()
            except Exception as error:
                failed_tasks.append(task)
                log(
                    f"FAILED {model_name} lane {key:03d} "
                    f"({completed}/{len(tasks)}); {type(error).__name__}: {error}"
                )
                continue

            lane_seconds.append(result["elapsed"])
            log(
                f"Completed {model_name} lane {key:03d} "
                f"({completed}/{len(tasks)}); rows={result['rows']}; "
                f"elapsed={duration_text(result['elapsed'])}; "
                f"eta={eta_text(lane_seconds, len(tasks) - completed)}"
            )

    log(f"Finished all lanes in {duration_text(perf_counter() - stage_start)}")
    if failed_tasks:
        log(
            f"WARNING: {len(failed_tasks)} lane(s) failed and wrote no file: "
            + ", ".join(f"{model_name} {key:03d}" for model_name, key in failed_tasks)
            + ". Rerunning the script retries exactly these."
        )


def eta_text(lane_seconds, remaining):
    if remaining == 0:
        return "0s"

    if not lane_seconds:
        return "unknown"

    mean_lane_seconds = float(np.mean(lane_seconds))
    return duration_text(mean_lane_seconds * remaining / CV_CPU)


def write_outputs():
    predictions = read_lane_files("predictions")
    if predictions.empty:
        raise RuntimeError("No lane predictions were found; nothing to summarise.")

    expected_lanes = {lane_output_file(*task) for task in all_tasks()}
    found_lanes = set(LANE_RESULTS_DIR.glob(f"{CV_LABEL}_*_predictions.csv"))
    unexpected = found_lanes - expected_lanes
    if unexpected:
        log(
            f"WARNING: {len(unexpected)} lane file(s) do not belong to the current "
            f"model/wavelength selection and are still being included, e.g. "
            f"{sorted(unexpected)[0].name}. Set FRESH_RUN = True to discard them."
        )

    sort_predictions(predictions)
    predictions_file = RESULTS_DIR / f"{CV_LABEL}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    log(f"Saved LOO predictions to {predictions_file} ({len(predictions)} rows)")

    parameters = read_lane_files("parameters")
    if not parameters.empty:
        sort_predictions(parameters)
        parameters_file = RESULTS_DIR / f"{CV_LABEL}_fitted_parameters.csv"
        parameters.to_csv(parameters_file, index=False)
        log(f"Saved fitted parameters to {parameters_file} ({len(parameters)} rows)")

    metrics = cv_helpers.per_wavelength_metrics(predictions)
    metrics_file = RESULTS_DIR / f"{CV_LABEL}_metrics_per_wavelength.csv"
    metrics.to_csv(metrics_file, index=False)
    log(f"Saved per-wavelength metrics to {metrics_file} ({len(metrics)} rows)")

    nrmse = cv_helpers.nrmse_comparison(metrics)
    nrmse_file = RESULTS_DIR / f"{CV_LABEL}_nrmse_per_wavelength.csv"
    nrmse.to_csv(nrmse_file, index=False)
    log(f"Saved NRMSE comparison to {nrmse_file} ({len(nrmse)} rows)")

    for model in active_model_names():
        if not has_sf_and_mf(metrics, model):
            log(f"Skipping {model} SF-vs-MF comparison; both variants were not run.")
            continue

        model_comparison = cv_helpers.sf_mf_comparison(metrics, model)
        comparison_file = (
            RESULTS_DIR / f"{model}_{CV_LABEL}_sf_vs_mf_nrmse_per_wavelength.csv"
        )
        model_comparison.to_csv(comparison_file, index=False)
        log(f"Saved {model} SF-vs-MF comparison to {comparison_file}")

    summary = cv_helpers.overall_summary(metrics)
    summary_file = RESULTS_DIR / f"{CV_LABEL}_summary.csv"
    summary.to_csv(summary_file, index=False)
    log(f"Saved summary to {summary_file} ({len(summary)} rows)")


def read_lane_files(kind):
    lane_files = sorted(LANE_RESULTS_DIR.glob(f"{CV_LABEL}_*_{kind}.csv"))
    if not lane_files:
        return pd.DataFrame()

    log(f"Assembling {len(lane_files)} lane {kind} files...")
    return pd.concat(
        (pd.read_csv(lane_file) for lane_file in lane_files),
        ignore_index=True,
    )


def prepare_incremental_outputs():
    if not FRESH_RUN:
        # Leave completed lanes in place; pending_tasks skips them on a rerun.
        # Sharded runs skip the sweep: a sibling array task may be part way
        # through writing one of these temporary files right now.
        if SHARD_COUNT == 1:
            for stale in LANE_RESULTS_DIR.glob("*.tmp"):
                stale.unlink()
        return

    for lane_file in LANE_RESULTS_DIR.glob("*.csv*"):
        lane_file.unlink()
    log(f"FRESH_RUN: cleared lane results in {LANE_RESULTS_DIR}")


def sort_predictions(predictions):
    if predictions.empty:
        return

    predictions.sort_values(
        by=[
            "fold",
            "held_out_source_index",
            "model",
            "response_index",
            "variant",
        ],
        inplace=True,
        ignore_index=True,
    )


def configure_cv_helpers():
    cv_helpers.VARIANTS = active_variants()
    cv_helpers.MODEL3_GP_KWARGS = MODEL3_GP_KWARGS
    cv_helpers.MODEL3_TUNING_KWARGS = MODEL3_TUNING_KWARGS


def validate_run_selection():
    if not RUN_MODEL1 and not RUN_MODEL2 and not RUN_MODEL3:
        raise ValueError(
            "At least one of RUN_MODEL1, RUN_MODEL2, or RUN_MODEL3 must be True."
        )
    if not RUN_SF and not RUN_MF:
        raise ValueError("At least one of RUN_SF or RUN_MF must be True.")
    if CV_CPU < 1:
        raise ValueError("CV_CPU must be at least 1.")
    if SHARD_COUNT < 1:
        raise ValueError("CV_LOO_SHARD_COUNT must be at least 1.")
    if not 0 <= SHARD_INDEX < SHARD_COUNT:
        raise ValueError(
            f"CV_LOO_SHARD_INDEX must be between 0 and {SHARD_COUNT - 1}, "
            f"got {SHARD_INDEX}."
        )
    if FRESH_RUN and SHARD_COUNT > 1:
        raise ValueError(
            "FRESH_RUN with more than one shard would let every array task "
            "delete the lanes its siblings are writing. Clear the lane "
            "directory before submitting instead."
        )
    if RUN_MODEL2 and MODEL2_N_WAVELENGTHS is not None:
        if not isinstance(MODEL2_N_WAVELENGTHS, (int, np.integer)):
            raise ValueError("MODEL2_N_WAVELENGTHS must be an integer or None.")
        if not 1 <= MODEL2_N_WAVELENGTHS <= RESPONSE_DIM:
            raise ValueError(
                f"MODEL2_N_WAVELENGTHS must be between 1 and {RESPONSE_DIM}."
            )
        if MODEL2_RESPONSE_INDEXES is not None:
            raise ValueError(
                "MODEL2_N_WAVELENGTHS samples within each spectrum and cannot "
                "be combined with a common MODEL2_RESPONSE_INDEXES subset."
            )


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
    if RUN_MODEL3:
        models.append("model3")
    return models


def log_run_setup(full_data, cv_data):
    model1_response_count = len(
        cv_helpers.selected_response_indexes(MODEL1_RESPONSE_INDEXES)
    )
    model2_response_count = len(
        cv_helpers.selected_response_indexes(MODEL2_RESPONSE_INDEXES)
    )
    model3_response_count = len(
        cv_helpers.selected_response_indexes(MODEL3_RESPONSE_INDEXES)
    )
    full_lf = int(full_data["is_hf"].eq(0).sum())
    full_hf = int(full_data["is_hf"].eq(1).sum())
    cv_lf = int(cv_data["is_hf"].eq(0).sum())
    cv_hf = int(cv_data["is_hf"].eq(1).sum())

    log(
        "LOO setup: "
        f"models={active_model_names()}, variants={list(active_variants())}, "
        f"model1_response_indexes={model1_response_count}, "
        f"model2_response_indexes={model2_response_count}, "
        f"model3_response_indexes={model3_response_count}, "
        f"model2_n_wavelengths_per_training_spectrum={MODEL2_N_WAVELENGTHS}, "
        f"cv_cpu={CV_CPU}, warm_start_model1_mf={WARM_START_MODEL1_MF}, "
        f"fresh_run={FRESH_RUN}, max_folds={MAX_FOLDS}, "
        f"shard={SHARD_INDEX + 1}/{SHARD_COUNT}, "
        f"assemble_only={ASSEMBLE_ONLY}, write_outputs={WRITE_OUTPUTS}"
    )
    log(
        "Data setup: "
        f"available LF/HF={full_lf}/{full_hf}, "
        f"used LF/HF={cv_lf}/{cv_hf}, "
        f"results_dir={RESULTS_DIR}, "
        f"lane_results_dir={LANE_RESULTS_DIR}"
    )


def log_loo_setup(cv_data):
    held_out = cv_data.index[cv_data["is_hf"].eq(1)].tolist()
    if MAX_FOLDS is not None:
        held_out = held_out[:MAX_FOLDS]
    preview_count = min(5, len(held_out))
    log(
        "HF LOO folds: "
        f"{len(held_out)} held-out samples; "
        f"first={held_out[:preview_count]}, "
        f"last={held_out[-preview_count:]}"
    )


def save_run_config(full_data, cv_data):
    lf_sample = cv_data[cv_data["is_hf"].eq(0)]
    hf_sample = cv_data[cv_data["is_hf"].eq(1)]

    # Written atomically: array tasks all write these, and a worker reading a
    # half-written lf_sample_source_indices.csv would fail its sample check.
    lf_sample_indices = pd.DataFrame({"source_index": lf_sample.index})
    write_atomically(
        RESULTS_DIR / "lf_sample_source_indices.csv",
        lf_sample_indices.to_csv(index=False),
    )

    config = {
        "cv_method": "hf_loo",
        "n_folds": int(len(hf_sample)) if MAX_FOLDS is None else int(MAX_FOLDS),
        "max_folds": MAX_FOLDS,
        "lf_sample_size": LF_SAMPLE_SIZE,
        "lf_sample_random_state": LF_SAMPLE_RANDOM_STATE,
        "model2_n_wavelengths_per_training_spectrum": MODEL2_N_WAVELENGTHS,
        "model2_wavelength_random_state": MODEL2_WAVELENGTH_RANDOM_STATE,
        "n_lf_available": int(full_data["is_hf"].eq(0).sum()),
        "n_hf_available": int(full_data["is_hf"].eq(1).sum()),
        "n_lf_used": int(len(lf_sample)),
        "n_hf_used": int(len(hf_sample)),
        "run_model1": RUN_MODEL1,
        "run_model2": RUN_MODEL2,
        "run_model3": RUN_MODEL3,
        "run_sf": RUN_SF,
        "run_mf": RUN_MF,
        "cv_cpu": CV_CPU,
        "warm_start_model1_mf": WARM_START_MODEL1_MF,
        "response_dim": RESPONSE_DIM,
        "n_response_sample": N_RESPONSE_SAMPLE,
        "model1_response_indexes": cv_helpers.response_index_config(
            MODEL1_RESPONSE_INDEXES
        ),
        "model2_response_indexes": cv_helpers.response_index_config(
            MODEL2_RESPONSE_INDEXES
        ),
        "model3_response_indexes": cv_helpers.response_index_config(
            MODEL3_RESPONSE_INDEXES
        ),
        "model3_tuning_kwargs": MODEL3_TUNING_KWARGS,
        "model3_gp_kwargs": MODEL3_GP_KWARGS,
        "lane_results_dir": str(LANE_RESULTS_DIR),
    }

    write_atomically(RESULTS_DIR / "run_config.json", json.dumps(config, indent=2))


def write_atomically(output_file, content):
    """Write text through a shard-private temporary file, then rename it."""
    temporary_file = output_file.with_name(f"{output_file.name}.{os.getpid()}.tmp")
    temporary_file.write_text(content, encoding="utf-8")
    os.replace(temporary_file, output_file)


def has_sf_and_mf(metrics, model):
    variants = set(metrics.loc[metrics["model"].eq(model), "variant"])
    return {"sf", "mf"}.issubset(variants)


if __name__ == "__main__":
    main()
