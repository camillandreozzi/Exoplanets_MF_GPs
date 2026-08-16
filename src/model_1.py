# Model 1

import json
from pathlib import Path

import gpboost as gpb
import numpy as np


_model1 = None
_model1_HF_only = None
_model1_metadata = {}
_model1_last_prediction = None


# the matrix that passes as training data must have 11 columns
# 9 atmospheric parameters, 1 column for the fidelity indicator, and 1 column for the response variable (scalar)
def fit_model1(train_data, HF_only= False):
    global _model1, _model1_HF_only, _model1_metadata, _model1_last_prediction

    # Input coordinates only: exclude fidelity indicator and response column
    x_train = train_data.iloc[:, :-2].values

    # Fidelity indicator only: second-to-last column
    fidelity = train_data.iloc[:, -2].values

    # Response/output: last column
    y_train = train_data.iloc[:, -1].values

    # Multi-fidelity coordinates: inputs + fidelity indicator
    coords_train_mf = train_data.iloc[:, :-1].values  # 9 inputs + fidelity

    if HF_only:
        # Subset x_train and y_train to only include high-fidelity samples
        hf_indices = np.where(fidelity == 1)[0]
        x_train = x_train[hf_indices]
        y_train = y_train[hf_indices]

        # Gaussian Process for high-fidelity data only
        gp_model = gpb.GPModel(gp_coords= x_train, cov_function = "matern", 
                       cov_fct_shape= 1.5, gp_approx = "vecchia", num_neighbors=20, 
                       likelihood="gaussian")


    else:
        # MF AR(1) model for both high-fidelity and low-fidelity data
        gp_model = gpb.GPModel(gp_coords= coords_train_mf, cov_function = "ar1_mf_matern", 
                       cov_fct_shape= 1.5, gp_approx = "vecchia", num_neighbors=20, 
                       likelihood="gaussian")
        
    gp_model.fit(y=y_train, X=x_train)
    _model1 = gp_model
    _model1_HF_only = HF_only
    _model1_metadata = _get_model1_metadata(train_data, HF_only)
    _model1_last_prediction = None
    return gp_model
   

def predict_model1(validation_data, HF_only= False, compute_metrics=True):
    global _model1_last_prediction

    if _model1 is None:
        raise RuntimeError("fit_model1 must be called before predict_model1.")

    if HF_only != _model1_HF_only:
        raise ValueError(
            "predict_model1 was called with a different HF_only value than fit_model1."
        )

    # Input coordinates only: exclude fidelity indicator and response column
    x_validation = validation_data.iloc[:, :-2].values

    # Multi-fidelity coordinates: inputs + fidelity indicator
    coords_validation_mf = validation_data.iloc[:, :-1].values

    if HF_only:
        gp_coords_pred = x_validation
    else:
        gp_coords_pred = coords_validation_mf

    prediction = _model1.predict(
        gp_coords_pred=gp_coords_pred,
        X_pred=x_validation,
        predict_response=True,
        predict_var=True,
    )

    metrics = None
    if compute_metrics:
        y_true = validation_data.iloc[:, -1].values
        metrics = _prediction_metrics(y_true, prediction["mu"])

    _model1_last_prediction = {
        "metrics": metrics,
        "n_validation": len(validation_data),
    }

    return prediction


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
