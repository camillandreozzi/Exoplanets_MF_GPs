"""Summarize full-fit HF in-sample RMSE/NRMSE for all models."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "full_fit_summary"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import gpboost as gpb
import numpy as np
import pandas as pd

from src.data_load import RESPONSE_DIM, create_data_indexes, load_full_data
from src.model_3 import load_model3, predict_model3


MODEL_NAMES = ("model1", "model2", "model3")
VARIANT_HF_ONLY = {
    "sf": True,
    "mf": False,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_data = load_full_data()
    indexes = create_data_indexes(full_data)
    hf_data = full_data.loc[indexes["hf_row_idx"]].copy()
    x_columns = indexes["x_columns"]
    response_columns = [f"response_{i}" for i in range(RESPONSE_DIM)]
    wavelengths = np.array(
        [full_data.attrs["wavelength_map"][i] for i in range(RESPONSE_DIM)],
        dtype=float,
    )
    y_true = hf_data.loc[:, response_columns].to_numpy(dtype=float)

    wavelength_frames = []
    sample_frames = []
    global_rows = []

    for model_name in MODEL_NAMES:
        for variant, hf_only in VARIANT_HF_ONLY.items():
            print(f"Predicting {model_name} {variant.upper()}", flush=True)
            y_pred = predict_full_fit_model(
                model_name=model_name,
                variant=variant,
                hf_only=hf_only,
                hf_data=hf_data,
                x_columns=x_columns,
                response_columns=response_columns,
                wavelengths=wavelengths,
            )
            residuals = y_pred - y_true

            wavelength_frames.append(
                per_wavelength_metrics(
                    model_name=model_name,
                    variant=variant,
                    wavelengths=wavelengths,
                    y_true=y_true,
                    residuals=residuals,
                )
            )
            sample_frames.append(
                per_sample_metrics(
                    model_name=model_name,
                    variant=variant,
                    full_data_index=hf_data.index.to_numpy(),
                    y_true=y_true,
                    residuals=residuals,
                )
            )
            global_rows.append(
                global_metrics(
                    model_name=model_name,
                    variant=variant,
                    y_true=y_true,
                    residuals=residuals,
                )
            )

    per_wavelength = pd.concat(wavelength_frames, ignore_index=True)
    per_sample = pd.concat(sample_frames, ignore_index=True)
    global_summary = pd.DataFrame(global_rows)
    distribution_summary = pd.concat(
        [
            summarize_distribution(per_wavelength, scope="per_wavelength"),
            summarize_distribution(per_sample, scope="per_sample"),
        ],
        ignore_index=True,
    )

    per_wavelength.to_csv(OUTPUT_DIR / "full_fit_metrics_per_wavelength.csv", index=False)
    per_sample.to_csv(OUTPUT_DIR / "full_fit_metrics_per_sample.csv", index=False)
    global_summary.to_csv(OUTPUT_DIR / "full_fit_metrics_global.csv", index=False)
    distribution_summary.to_csv(
        OUTPUT_DIR / "full_fit_metrics_distribution_summary.csv",
        index=False,
    )

    per_wavelength_wide(per_wavelength).to_csv(
        OUTPUT_DIR / "full_fit_metrics_per_wavelength_wide.csv",
        index=False,
    )
    per_sample_wide(per_sample).to_csv(
        OUTPUT_DIR / "full_fit_metrics_per_sample_wide.csv",
        index=False,
    )

    print(f"Saved full-fit metric summaries to {OUTPUT_DIR}", flush=True)


def predict_full_fit_model(
    model_name,
    variant,
    hf_only,
    hf_data,
    x_columns,
    response_columns,
    wavelengths,
):
    if model_name == "model1":
        return predict_model1_full_fit(
            variant=variant,
            hf_only=hf_only,
            hf_data=hf_data,
            x_columns=x_columns,
            response_columns=response_columns,
        )
    if model_name == "model2":
        return predict_model2_full_fit(
            variant=variant,
            hf_only=hf_only,
            hf_data=hf_data,
            x_columns=x_columns,
            response_columns=response_columns,
            wavelengths=wavelengths,
        )
    if model_name == "model3":
        return predict_model3_full_fit(
            variant=variant,
            hf_only=hf_only,
            hf_data=hf_data,
            x_columns=x_columns,
            response_columns=response_columns,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict_model1_full_fit(variant, hf_only, hf_data, x_columns, response_columns):
    model_dir = PROJECT_ROOT / "results" / "model1" / "full_fit"
    predictions = np.empty((len(hf_data), RESPONSE_DIM), dtype=float)

    for response_index, response_col in enumerate(response_columns):
        model_file = model_dir / f"model1_{variant}_response_{response_index}.pkl"
        model = load_gp_model(model_file)
        validation_data = hf_data.loc[:, [*x_columns, "is_hf", response_col]].copy()
        predictions[:, response_index] = predict_gp_model1_like(
            model=model,
            validation_data=validation_data,
            hf_only=hf_only,
        )

    return predictions


def predict_model2_full_fit(variant, hf_only, hf_data, x_columns, response_columns, wavelengths):
    model_file = PROJECT_ROOT / "results" / "model2" / "full_fit" / (
        f"model2_{variant.upper()}.pkl"
    )
    model = load_gp_model(model_file)
    predictions = np.empty((len(hf_data), RESPONSE_DIM), dtype=float)

    for response_index, response_col in enumerate(response_columns):
        validation_data = hf_data.loc[:, x_columns].copy()
        validation_data["wavelength"] = wavelengths[response_index]
        validation_data["is_hf"] = hf_data["is_hf"].to_numpy()
        validation_data["response"] = hf_data[response_col].to_numpy()
        predictions[:, response_index] = predict_gp_model1_like(
            model=model,
            validation_data=validation_data,
            hf_only=hf_only,
        )
        if (response_index + 1) % 25 == 0 or response_index + 1 == RESPONSE_DIM:
            print(
                f"  {variant.upper()} Model 2 wavelengths: "
                f"{response_index + 1}/{RESPONSE_DIM}",
                flush=True,
            )

    return predictions


def predict_model3_full_fit(variant, hf_only, hf_data, x_columns, response_columns):
    model_dir = PROJECT_ROOT / "results" / "model3" / "full_fit"
    predictions = np.empty((len(hf_data), RESPONSE_DIM), dtype=float)

    for response_index, response_col in enumerate(response_columns):
        model_file = model_dir / f"model3_{variant}_response_{response_index}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing Model 3 full-fit model: {model_file}")

        model = load_model3(model_file, HF_only=hf_only)
        validation_data = hf_data.loc[:, [*x_columns, "is_hf", response_col]].copy()
        prediction = predict_model3(
            validation_data,
            HF_only=hf_only,
            model=model,
            compute_metrics=False,
        )
        predictions[:, response_index] = np.asarray(prediction["mu"], dtype=float)

    return predictions


def load_gp_model(model_file):
    if not model_file.exists():
        raise FileNotFoundError(f"Missing full-fit model: {model_file}")
    return gpb.GPModel(model_file=str(model_file))


def predict_gp_model1_like(model, validation_data, hf_only):
    x_validation = validation_data.iloc[:, :-2].to_numpy(dtype=float)
    if hf_only:
        gp_coords_pred = x_validation
    else:
        gp_coords_pred = validation_data.iloc[:, :-1].to_numpy(dtype=float)

    prediction = model.predict(
        gp_coords_pred=gp_coords_pred,
        X_pred=x_validation,
        predict_response=True,
        predict_var=False,
    )
    return np.asarray(prediction["mu"], dtype=float)


def per_wavelength_metrics(model_name, variant, wavelengths, y_true, residuals):
    y_ranges = np.ptp(y_true, axis=0)
    rows = []
    for response_index, wavelength in enumerate(wavelengths):
        rows.append(
            metric_row(
                model_name=model_name,
                variant=variant,
                response_index=response_index,
                wavelength=wavelength,
                sample_index=None,
                full_data_index=None,
                n=len(y_true),
                y_range=y_ranges[response_index],
                residuals=residuals[:, response_index],
            )
        )
    return pd.DataFrame(rows)


def per_sample_metrics(model_name, variant, full_data_index, y_true, residuals):
    y_ranges = np.ptp(y_true, axis=1)
    rows = []
    for sample_index, source_index in enumerate(full_data_index):
        rows.append(
            metric_row(
                model_name=model_name,
                variant=variant,
                response_index=None,
                wavelength=None,
                sample_index=sample_index,
                full_data_index=source_index,
                n=RESPONSE_DIM,
                y_range=y_ranges[sample_index],
                residuals=residuals[sample_index, :],
            )
        )
    return pd.DataFrame(rows)


def metric_row(
    model_name,
    variant,
    response_index,
    wavelength,
    sample_index,
    full_data_index,
    n,
    y_range,
    residuals,
):
    residuals = np.asarray(residuals, dtype=float)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    max_abs_error = float(np.max(np.abs(residuals)))
    nrmse = np.nan if y_range == 0 else float(rmse / y_range)

    return {
        "model": model_name,
        "variant": variant,
        "response_index": response_index,
        "wavelength": wavelength,
        "sample_index": sample_index,
        "full_data_index": full_data_index,
        "n": n,
        "y_range": float(y_range),
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "bias": bias,
        "max_abs_error": max_abs_error,
    }


def global_metrics(model_name, variant, y_true, residuals):
    y_range = np.ptp(y_true)
    row = metric_row(
        model_name=model_name,
        variant=variant,
        response_index=None,
        wavelength=None,
        sample_index=None,
        full_data_index=None,
        n=residuals.size,
        y_range=y_range,
        residuals=residuals.reshape(-1),
    )
    return {
        key: value
        for key, value in row.items()
        if key
        in {
            "model",
            "variant",
            "n",
            "y_range",
            "rmse",
            "nrmse",
            "mae",
            "bias",
            "max_abs_error",
        }
    }


def summarize_distribution(metrics, scope):
    summary = (
        metrics.groupby(["model", "variant"])
        .agg(
            n_units=("rmse", "size"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            std_rmse=("rmse", "std"),
            min_rmse=("rmse", "min"),
            max_rmse=("rmse", "max"),
            mean_nrmse=("nrmse", "mean"),
            median_nrmse=("nrmse", "median"),
            std_nrmse=("nrmse", "std"),
            min_nrmse=("nrmse", "min"),
            max_nrmse=("nrmse", "max"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
            max_abs_error=("max_abs_error", "max"),
        )
        .reset_index()
    )
    summary.insert(0, "scope", scope)
    return summary


def per_wavelength_wide(per_wavelength):
    return metric_wide(
        per_wavelength,
        index_columns=["response_index", "wavelength", "n", "y_range"],
    )


def per_sample_wide(per_sample):
    return metric_wide(
        per_sample,
        index_columns=["sample_index", "full_data_index", "n", "y_range"],
    )


def metric_wide(metrics, index_columns):
    working = metrics.copy()
    working["column"] = working["model"] + "_" + working["variant"]
    wide = (
        working.pivot_table(
            index=index_columns,
            columns="column",
            values=["rmse", "nrmse"],
            aggfunc="first",
        )
        .sort_index(axis=1, level=[1, 0])
        .reset_index()
    )
    wide.columns = [
        "_".join([part for part in column if part])
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]
    return wide


if __name__ == "__main__":
    main()
