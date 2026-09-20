# Model 1

import json
from pathlib import Path

import gpboost as gpb
import numpy as np


_model1 = None
_model1_HF_only = None
_model1_metadata = {}
_model1_last_prediction = None
_model1_input_mean = None
_model1_input_std = None
GP_THREADS = 1
# Squared exponential with Automatic Relevance Determination: one range per
# input dimension, so the nine physical parameters are free to act on different
# length scales instead of sharing one. The multi-fidelity form wraps the same
# base in the two-level AR(1) structure. No cov_fct_shape goes with these -- it
# is a Matern smoothness and has no meaning for a Gaussian covariance.
SF_COV_FUNCTION = "gaussian_ard"
MF_COV_FUNCTION = "ar1_mf_gaussian_ard"

def _fit_input_scaler(x, fidelity):
    x_hf = x[fidelity == 1]
    if len(x_hf) == 0:
        raise ValueError("Model 1 requires HF training rows for input scaling.")

    mean = x_hf.mean(axis=0)
    std = x_hf.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _scaled_gp_coords(x, fidelity, mean, std, HF_only):
    scaled = (x - mean) / std
    if HF_only:
        return scaled
    return np.column_stack([scaled, fidelity])

# the matrix that passes as training data must have 11 columns
# 9 atmospheric parameters, 1 column for the fidelity indicator, and 1 column for the response variable (scalar)
def fit_model1(train_data, HF_only=False, init_cov_pars=None):
    global _model1, _model1_HF_only
    global _model1_metadata, _model1_last_prediction
    global _model1_input_mean, _model1_input_std

    x = train_data.iloc[:, :-2].to_numpy(dtype=float)
    fidelity = train_data.iloc[:, -2].to_numpy()
    y = train_data.iloc[:, -1].to_numpy(dtype=float)

    # Learn scaling from this fit's training HF rows only.
    mean, std = _fit_input_scaler(x, fidelity)

    if HF_only:
        keep = fidelity == 1
        x = x[keep]
        fidelity = fidelity[keep]
        y = y[keep]

    coords = _scaled_gp_coords(
        x, fidelity, mean, std, HF_only=HF_only
    )

    gp_model = gpb.GPModel(
        gp_coords=coords,
        cov_function=SF_COV_FUNCTION if HF_only else MF_COV_FUNCTION,
        gp_approx="vecchia",
        num_neighbors=20,
        likelihood="gaussian",
        num_parallel_threads=GP_THREADS,
    )

    # Preserve the raw-input linear mean.
    gp_model.fit(
        y=y,
        X=x,
        params=_gp_fit_params(init_cov_pars),
    )

    _model1 = gp_model
    _model1_HF_only = HF_only
    _model1_input_mean = mean
    _model1_input_std = std
    _model1_metadata = _get_model1_metadata(train_data, HF_only)
    _model1_metadata["input_scaling"] = {
        "method": "standard",
        "fitted_on": "training_hf",
        "applied_to": "gp_coordinates_only",
        "columns": list(train_data.columns[:-2]),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    _model1_last_prediction = None

    return gp_model

def predict_model1(
    validation_data,
    HF_only=False,
    compute_metrics=True,
):
    global _model1_last_prediction

    if _model1 is None:
        raise RuntimeError(
            "fit_model1 must be called before predict_model1."
        )

    if HF_only != _model1_HF_only:
        raise ValueError(
            "predict_model1 was called with a different HF_only "
            "value than fit_model1."
        )

    x = validation_data.iloc[:, :-2].to_numpy(dtype=float)
    fidelity = validation_data.iloc[:, -2].to_numpy()

    # Reuse training statistics; never fit a scaler on validation data.
    coords = _scaled_gp_coords(
        x,
        fidelity,
        _model1_input_mean,
        _model1_input_std,
        HF_only=HF_only,
    )

    prediction = _model1.predict(
        gp_coords_pred=coords,
        X_pred=x,
        predict_response=True,
        predict_var=True,
    )

    metrics = None
    if compute_metrics:
        y_true = validation_data.iloc[:, -1].to_numpy(dtype=float)
        metrics = _prediction_metrics(y_true, prediction["mu"])

    _model1_last_prediction = {
        "metrics": metrics,
        "n_validation": len(validation_data),
        "predict_var": True,
    }

    return prediction


def model1_cov_pars(gp_model):
    """Fitted covariance parameters, in the order ``init_cov_pars`` expects."""
    return np.asarray(gp_model.get_cov_pars(std_err=False), dtype=float).ravel()


def _gp_fit_params(init_cov_pars):
    params = {"trace": False}
    if init_cov_pars is not None:
        params["init_cov_pars"] = np.asarray(init_cov_pars, dtype=float)
    return params


def save_model1(filename):
    if _model1 is None:
        raise RuntimeError("fit_model1 must be called before save_model1.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    _model1.save_model(str(filename))

    diagnostics = {
        **_model1_metadata,
        "model_file": str(filename),
        "kernel_parameters": _get_kernel_parameters(_model1),
        "prediction": _model1_last_prediction,
    }

    diagnostics_file = filename.with_suffix(filename.suffix + ".diagnostics.json")
    diagnostics_file.write_text(
        json.dumps(_to_jsonable(diagnostics), indent=2),
        encoding="utf-8",
    )

    return {
        "model_file": str(filename),
        "diagnostics_file": str(diagnostics_file),
    }


def _get_model1_metadata(train_data, HF_only):
    response_column = train_data.columns[-1]
    response_index = None
    wavelength = None

    if isinstance(response_column, str) and response_column.startswith("response_"):
        response_index = int(response_column.removeprefix("response_"))
        wavelength_map = train_data.attrs.get("wavelength_map", {})
        wavelength = wavelength_map.get(response_index)

    return {
        "model": "model1",
        "HF_only": HF_only,
        "response_column": response_column,
        "response_index": response_index,
        "wavelength": wavelength,
    }


def _get_kernel_parameters(gp_model):
    try:
        cov_pars = gp_model.get_cov_pars(std_err=True)
    except Exception:
        cov_pars = gp_model.get_cov_pars(std_err=False)

    if hasattr(cov_pars, "to_dict"):
        return cov_pars.to_dict(orient="index")

    return cov_pars


def _prediction_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    return {
        "rmse": np.sqrt(np.mean(residuals**2)),
        "mae": np.mean(np.abs(residuals)),
        "bias": np.mean(residuals),
        "max_abs_error": np.max(np.abs(residuals)),
    }


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
