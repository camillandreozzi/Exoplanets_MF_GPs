"""Model 3: GPBoost helpers for the exoplanet pipeline.

The expected input shape matches ``model_1``:
- input columns
- ``is_hf`` fidelity indicator
- one scalar response column

For the multi-fidelity variant, low-fidelity samples are used only in the
AR(1) GP covariance component. The boosted fixed-effect mean is trained on
high-fidelity rows only.
"""

import json
from pathlib import Path

import gpboost as gpb
import numpy as np


GP_THREADS = 1
DEFAULT_LIKELIHOOD = "gaussian"
# Vecchia replaces the dense n x n Cholesky with a sparse ordered-conditional
# factorization, which only pays off well above the single-fidelity row count.
# At the 96 high-fidelity rows the SF GP sees, "none" (GPBoost's own default,
# exact Cholesky) agreed with vecchia to ~5 significant figures while costing
# less; inside tuning that saving is multiplied by every boosting iteration x
# CV fold x trial. The MF GP spans ~10k rows, where the dense factorization is
# 0.82 GB and several minutes per fit, so it keeps the approximation.
DEFAULT_GP_APPROX_SF = "none"
DEFAULT_GP_APPROX_MF = "vecchia"
DEFAULT_BOOSTING_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "verbose": 0,
    # One thread per process: the CV scripts get their parallelism from running
    # many independent fits at once, so LightGBM must not also fan out per core.
    "num_threads": 1,
}

_model3 = None
_model3_HF_only = None
_model3_metadata = {}
_model3_last_prediction = None
_model3_last_tuning = None


class Model3Fit:
    """Fitted Model 3: HF-trained boosted mean plus GP residual model."""

    def __init__(
        self,
        booster,
        gp_model,
        HF_only,
        n_tree_train=None,
        n_gp_train=None,
    ):
        self.booster = booster
        self.gp_model = gp_model
        self.HF_only = HF_only
        self.n_tree_train = n_tree_train
        self.n_gp_train = n_gp_train

    def predict(
        self,
        data,
        gp_coords_pred,
        predict_var=True,
        pred_latent=False,
        num_iteration=None,
        predict_cov_mat=False,
        sample_posterior=False,
        num_post_samples=100,
        **predict_kwargs,
    ):
        """Predict response means using the tree as GP fixed-effect offset."""
        ignore_gp_model = bool(predict_kwargs.pop("ignore_gp_model", False))
        tree_kwargs, gp_kwargs = _split_model3_predict_kwargs(predict_kwargs)
        fixed_effect = _predict_tree_fixed_effect(
            self.booster,
            data=data,
            num_iteration=num_iteration,
            predict_kwargs=tree_kwargs,
        )

        if ignore_gp_model:
            return fixed_effect

        if pred_latent:
            random_effect = self.gp_model.predict(
                gp_coords_pred=gp_coords_pred,
                predict_response=False,
                predict_var=predict_var,
                predict_cov_mat=predict_cov_mat,
                sample_posterior=sample_posterior,
                num_post_samples=num_post_samples,
                **gp_kwargs,
            )
            posterior_samples = random_effect.get("posterior_samples")
            if posterior_samples is not None:
                posterior_samples = posterior_samples + fixed_effect[:, np.newaxis]
            return {
                "fixed_effect": fixed_effect,
                "random_effect_mean": random_effect["mu"],
                "random_effect_cov": (
                    random_effect["cov"] if predict_cov_mat else random_effect["var"]
                ),
                "response_mean": None,
                "response_var": None,
                "posterior_samples": posterior_samples,
            }

        response = self.gp_model.predict(
            gp_coords_pred=gp_coords_pred,
            predict_response=True,
            predict_var=predict_var,
            predict_cov_mat=predict_cov_mat,
            sample_posterior=sample_posterior,
            num_post_samples=num_post_samples,
            offset_pred=fixed_effect,
            **gp_kwargs,
        )
        return {
            "fixed_effect": None,
            "random_effect_mean": None,
            "random_effect_cov": None,
            "response_mean": response["mu"],
            "response_var": response["cov"] if predict_cov_mat else response["var"],
            "posterior_samples": response.get("posterior_samples"),
        }

    def save_model(self, filename):
        """Save the tree booster and GP model using a shared file stem."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        self.booster.save_model(str(filename))
        gp_model_file = filename.with_suffix(filename.suffix + ".gp_model.json")
        self.gp_model.save_model(str(gp_model_file))
        return {
            "model_file": str(filename),
            "tree_model_file": str(filename),
            "gp_model_file": str(gp_model_file),
        }


def tune_model3_parameters(
    train_data,
    HF_only=False,
    method="tpe",
    search_space=None,
    param_grid=None,
    params=None,
    n_trials=100,
    num_try_random=100,
    max_num_boost_round=1000,
    early_stopping_rounds=20,
    nfold=5,
    folds=None,
    validation_fraction=None,
    metric="mse",
    cv_seed=4,
    tpe_seed=1,
    seed=4,
    gp_kwargs=None,
    use_gp_model_for_validation=True,
    train_gp_model_cov_pars=True,
    verbose_eval=1,
):
    """Tune GPBoost tree parameters for one scalar-response training set.

    Parameters
    ----------
    train_data
        DataFrame ending in ``is_hf`` and a scalar response column.
    HF_only
        Both variants tune the boosted fixed-effect term using high-fidelity
        rows only. In the final multi-fidelity fit, low-fidelity rows enter
        only through the AR(1) GP covariance component.
    method
        ``"tpe"`` for Optuna TPE search or ``"grid"`` for GPBoost random
        grid search.
    validation_fraction
        Optional holdout fraction. When set and ``folds`` is not provided,
        one deterministic validation fold is used instead of k-fold CV.
    gp_kwargs
        Overrides for the GP the tuner validates against; see
        ``_make_tuning_gp_model``.
    use_gp_model_for_validation
        Score each trial on tree + GP predictions rather than tree-only ones.
    train_gp_model_cov_pars
        Refit the GP covariance parameters during each trial's boosting run.
    """
    global _model3_last_tuning

    arrays = _prepare_model3_training_arrays(
        train_data,
        HF_only=HF_only,
    )
    _validate_model3_training_arrays(arrays)

    if validation_fraction is not None and folds is None:
        folds = _make_validation_folds(
            n_rows=len(arrays["y_tree"]),
            validation_fraction=validation_fraction,
            random_state=seed,
        )

    method = method.lower()
    tuning_params = _tuning_base_params(params)
    tuning_gp_model = _make_tuning_gp_model(
        arrays,
        gp_kwargs=gp_kwargs,
        train_gp_model_cov_pars=train_gp_model_cov_pars,
    )

    if method == "tpe":
        opt_params = gpb.tune_pars_TPE_algorithm_optuna(
            search_space=(
                default_model3_search_space(len(arrays["y_tree"]))
                if search_space is None
                else search_space
            ),
            n_trials=n_trials,
            X=arrays["x_tree"],
            y=arrays["y_tree"],
            gp_model=tuning_gp_model,
            use_gp_model_for_validation=use_gp_model_for_validation,
            train_gp_model_cov_pars=train_gp_model_cov_pars,
            max_num_boost_round=max_num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            metric=metric,
            folds=folds,
            nfold=nfold,
            cv_seed=cv_seed,
            tpe_seed=tpe_seed,
            params=tuning_params,
            verbose_eval=verbose_eval,
        )
    elif method == "grid":
        train_set = gpb.Dataset(data=arrays["x_tree"], label=arrays["y_tree"])
        opt_params = gpb.grid_search_tune_parameters(
            param_grid=(
                default_model3_param_grid(len(arrays["y_tree"]))
                if param_grid is None
                else param_grid
            ),
            params=tuning_params,
            train_set=train_set,
            gp_model=tuning_gp_model,
            use_gp_model_for_validation=use_gp_model_for_validation,
            train_gp_model_cov_pars=train_gp_model_cov_pars,
            num_try_random=num_try_random,
            nfold=nfold,
            folds=folds,
            num_boost_round=max_num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            metric=metric,
            seed=seed,
        )
    else:
        raise ValueError("method must be either 'tpe' or 'grid'.")

    _model3_last_tuning = {
        "method": method,
        "HF_only": HF_only,
        "metric": metric,
        "n_train": len(arrays["y_tree"]),
        "n_tree_train": len(arrays["y_tree"]),
        "n_gp_train": len(arrays["y_gp"]),
        "nfold": nfold if folds is None else None,
        "tuning_gp_covariance": "matern",
        "use_gp_model_for_validation": use_gp_model_for_validation,
        "train_gp_model_cov_pars": train_gp_model_cov_pars,
        "validation_fraction": validation_fraction,
        "fixed_effect_training": "high_fidelity_only",
        "lf_used_in_fixed_effect": False,
        "result": _to_jsonable(opt_params),
    }
    return opt_params


def fit_model3(
    train_data,
    HF_only=False,
    params=None,
    tuning_result=None,
    num_boost_round=None,
    gp_kwargs=None,
    train_gp_model_cov_pars=True,
    use_gp_model_for_validation=True,
    valid_sets=None,
    valid_names=None,
    early_stopping_rounds=None,
    evals_result=None,
    verbose_eval=True,
    tuning_kwargs=None,
):
    """Fit Model 3 for one scalar-response data set.

    The boosted fixed-effect mean is tuned and trained on high-fidelity rows
    only. The GP component is then fit to all eligible rows with the tree
    predictions as offsets; for ``HF_only=False`` this GP uses an AR(1)
    multi-fidelity covariance. Pass ``tuning_kwargs`` to override
    ``tune_model3_parameters`` defaults.
    """
    global _model3, _model3_HF_only, _model3_metadata, _model3_last_prediction

    arrays = _prepare_model3_training_arrays(
        train_data,
        HF_only=HF_only,
    )
    _validate_model3_training_arrays(arrays)

    if tuning_result is None:
        tuning_options = dict(tuning_kwargs or {})
        for reserved_key in ("train_data", "HF_only"):
            if reserved_key in tuning_options:
                raise ValueError(
                    f"tuning_kwargs cannot override {reserved_key}; "
                    "pass it directly to fit_model3 instead."
                )
        tuning_options.setdefault("params", params)
        tuning_options.setdefault("gp_kwargs", gp_kwargs)
        tuning_options.setdefault(
            "use_gp_model_for_validation",
            use_gp_model_for_validation,
        )
        tuning_options.setdefault(
            "train_gp_model_cov_pars",
            train_gp_model_cov_pars,
        )
        tuning_result = tune_model3_parameters(
            train_data,
            HF_only=HF_only,
            **tuning_options,
        )

    fit_params = _fit_params(params=params, tuning_result=tuning_result)
    if num_boost_round is None:
        num_boost_round = _best_num_boost_round(tuning_result, default=100)

    train_set = gpb.Dataset(data=arrays["x_tree"], label=arrays["y_tree"])
    booster = gpb.train(
        params=fit_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=verbose_eval,
    )

    fixed_effect_gp_train = _predict_tree_fixed_effect(
        booster,
        data=arrays["x_gp_offset"],
        num_iteration=None,
    )
    _validate_fixed_effect(
        fixed_effect_gp_train,
        arrays["y_gp"],
        fit_params,
        num_boost_round,
    )
    gp_model = make_model3_gp_model(
        arrays["coords_gp"],
        HF_only=HF_only,
        gp_kwargs=gp_kwargs,
    )
    init_cov_pars = _initial_cov_pars_for_gp_model(
        gp_model=gp_model,
        y=arrays["y_gp"],
        coords=arrays["coords_gp"],
        offset=fixed_effect_gp_train,
    )
    gp_model.fit(
        y=arrays["y_gp"],
        offset=fixed_effect_gp_train,
        params=_gp_fit_params(
            train_gp_model_cov_pars,
            init_cov_pars=init_cov_pars,
        ),
    )

    model = Model3Fit(
        booster=booster,
        gp_model=gp_model,
        HF_only=HF_only,
        n_tree_train=len(arrays["y_tree"]),
        n_gp_train=len(arrays["y_gp"]),
    )

    _model3 = model
    _model3_HF_only = HF_only
    _model3_metadata = _get_model3_metadata(
        train_data=train_data,
        HF_only=HF_only,
        params=fit_params,
        num_boost_round=num_boost_round,
        gp_kwargs=gp_kwargs,
        n_tree_train=len(arrays["y_tree"]),
        n_gp_train=len(arrays["y_gp"]),
    )
    _model3_last_prediction = None
    return model


def predict_model3(
    validation_data,
    HF_only=False,
    model=None,
    compute_metrics=True,
    predict_var=True,
    pred_latent=False,
    num_iteration=None,
    predict_cov_mat=False,
    sample_posterior=False,
    num_post_samples=100,
    **predict_kwargs,
):
    """Predict with the last fitted Model 3 object or an explicit model."""
    global _model3_last_prediction

    booster = _resolve_model3_booster(model=model, HF_only=HF_only)
    x_validation, coords_validation = _prepare_model3_prediction_arrays(
        validation_data,
        HF_only=HF_only,
    )

    prediction = booster.predict(
        data=x_validation,
        gp_coords_pred=coords_validation,
        predict_var=predict_var,
        pred_latent=pred_latent,
        num_iteration=num_iteration,
        predict_cov_mat=predict_cov_mat,
        sample_posterior=sample_posterior,
        num_post_samples=num_post_samples,
        **predict_kwargs,
    )
    prediction = _with_pipeline_prediction_aliases(prediction, pred_latent=pred_latent)

    metrics = None
    if compute_metrics:
        y_true = validation_data.iloc[:, -1].to_numpy(dtype=float)
        metrics = _prediction_metrics(y_true, prediction["mu"])

    _model3_last_prediction = {
        "metrics": metrics,
        "n_validation": len(validation_data),
        "predict_var": predict_var,
        "pred_latent": pred_latent,
    }
    return prediction


def save_model3(filename):
    """Save the last fitted Model 3 object and diagnostics."""
    if _model3 is None:
        raise RuntimeError("fit_model3 must be called before save_model3.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    saved_model = _model3.save_model(str(filename))
    if isinstance(saved_model, dict):
        model_files = saved_model
    else:
        model_files = {
            "model_file": str(filename),
            "tree_model_file": str(filename),
            "gp_model_file": None,
        }

    diagnostics = {
        **_model3_metadata,
        **model_files,
        "kernel_parameters": _get_kernel_parameters(_model3),
        "prediction": _model3_last_prediction,
        "tuning": _model3_last_tuning,
    }

    diagnostics_file = filename.with_suffix(filename.suffix + ".diagnostics.json")
    diagnostics_file.write_text(
        json.dumps(_to_jsonable(diagnostics), indent=2),
        encoding="utf-8",
    )

    return {
        **model_files,
        "diagnostics_file": str(diagnostics_file),
    }


def load_model3(filename, HF_only=None):
    """Load a saved Model 3 object into the module-level prediction state."""
    global _model3, _model3_HF_only, _model3_metadata, _model3_last_prediction

    filename = Path(filename)
    booster = gpb.Booster(model_file=str(filename))
    gp_model_file = filename.with_suffix(filename.suffix + ".gp_model.json")

    if gp_model_file.exists():
        model = Model3Fit(
            booster=booster,
            gp_model=gpb.GPModel(model_file=str(gp_model_file)),
            HF_only=HF_only,
        )
    else:
        model = booster

    _model3 = model
    _model3_HF_only = HF_only
    _model3_metadata = {
        "model": "model3",
        "HF_only": HF_only,
        "model_file": str(filename),
        "gp_model_file": str(gp_model_file) if gp_model_file.exists() else None,
    }
    _model3_last_prediction = None
    return model


def make_model3_gp_model(gp_coords, HF_only=False, gp_kwargs=None):
    """Create the GP component used by Model 3."""
    gp_coords = np.asarray(gp_coords, dtype=float)
    if gp_coords.ndim != 2:
        raise ValueError("gp_coords must be a 2D array.")

    kwargs = {
        "gp_coords": gp_coords,
        "cov_function": "matern" if HF_only else "ar1_mf_matern",
        "cov_fct_shape": 1.5,
        "gp_approx": DEFAULT_GP_APPROX_SF if HF_only else DEFAULT_GP_APPROX_MF,
        "num_neighbors": 20,
        "likelihood": DEFAULT_LIKELIHOOD,
        "num_parallel_threads": GP_THREADS,
    }
    if gp_kwargs is not None:
        kwargs.update(gp_kwargs)

    if not HF_only and str(kwargs.get("cov_function", "")).startswith("ar1_mf_"):
        kwargs["fidelity_specific_mean"] = False

    if kwargs.get("gp_approx") == "vecchia" and kwargs.get("num_neighbors") is not None:
        kwargs["num_neighbors"] = min(int(kwargs["num_neighbors"]), gp_coords.shape[0] - 1)
        if kwargs["num_neighbors"] < 1:
            raise ValueError("Vecchia GP approximation needs at least two training rows.")
    elif kwargs.get("gp_approx") != "vecchia":
        kwargs.pop("num_neighbors", None)

    return gpb.GPModel(**kwargs)


def _make_tuning_gp_model(arrays, gp_kwargs=None, train_gp_model_cov_pars=True):
    """The GP the tuner validates against.

    Without a ``gp_model`` the tuner scores tree-only CV error, so it selects a
    tree that absorbs the smooth structure the GP exists to model: on one LOO
    fold of response_100 that was learning_rate 1.49 with 243 leaves, against
    0.099 with 4 leaves once the GP is in the loop. For the multi-fidelity
    variant that would leave the AR(1) component nothing to carry.

    The boosted mean is trained on high-fidelity rows only in both variants, so
    this is the single-fidelity Matern GP on those rows either way; it is a
    proxy for the AR(1) GP that the multi-fidelity fit ultimately uses.
    """
    gp_model = make_model3_gp_model(
        arrays["x_tree"],
        HF_only=True,
        gp_kwargs=gp_kwargs,
    )
    init_cov_pars = _initial_cov_pars_for_gp_model(
        gp_model=gp_model,
        y=arrays["y_tree"],
        coords=arrays["x_tree"],
    )
    gp_model.set_optim_params(
        params=_gp_fit_params(
            train_gp_model_cov_pars,
            init_cov_pars=init_cov_pars,
        )
    )
    return gp_model


def default_model3_search_space(n_train):
    """Default Optuna/TPE ranges from the GPBoost template, bounded by data size.

    ``learning_rate`` is capped at 1 rather than the template's 10. The tuner
    scores each trial with the GP in the loop, where a rate above 1 only makes
    the boosting converge badly (huge but finite CV error), so such a trial can
    still win a fold. The final fit trains the tree on its own, where L2
    boosting is only contractive for a rate below 2: at 3.2 the fixed effect
    reaches ~1e154 after 1000 rounds, and the GP fit then dies with "NaN
    occurred in initial negative log-likelihood".
    """
    max_bin = max(63, min(10000, int(n_train)))
    min_data_in_leaf_upper = max(1, min(1000, int(n_train)))

    return {
        "learning_rate": [0.001, 1],
        "min_data_in_leaf": [1, min_data_in_leaf_upper],
        "max_depth": [-1, -1],
        "num_leaves": [2, 1024],
        "lambda_l2": [0, 100],
        "max_bin": [63, max_bin],
        "feature_fraction": [0.5, 1],
        "line_search_step_length": [True, False],
    }


def default_model3_param_grid(n_train):
    """Default random-grid search values from the GPBoost template."""
    max_bin_values = _unique_preserving_order(
        [250, 500, 1000, min(10000, int(n_train))]
    )

    return {
        # 10 is dropped for the reason given in default_model3_search_space.
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "min_data_in_leaf": _unique_preserving_order(
            [1, 10, 100, min(1000, int(n_train))]
        ),
        "max_depth": [-1],
        "num_leaves": [int(value) for value in 2 ** np.arange(1, 10)],
        "lambda_l2": [0, 1, 10, 100],
        "max_bin": max_bin_values,
        "feature_fraction": [0.5, 0.75, 1],
        "line_search_step_length": [True, False],
    }


def _prepare_model3_training_arrays(train_data, HF_only):
    x_all = train_data.iloc[:, :-2].to_numpy(dtype=float)
    fidelity = train_data.iloc[:, -2].to_numpy()
    y_all = train_data.iloc[:, -1].to_numpy(dtype=float)
    coords_mf = train_data.iloc[:, :-1].to_numpy(dtype=float)
    hf_indices = np.where(fidelity == 1)[0]
    if len(hf_indices) == 0:
        raise ValueError("Model 3 requires at least one high-fidelity row.")

    if HF_only:
        return {
            "x_tree": x_all[hf_indices],
            "y_tree": y_all[hf_indices],
            "x_gp_offset": x_all[hf_indices],
            "coords_gp": x_all[hf_indices],
            "y_gp": y_all[hf_indices],
            "tree_row_indices": hf_indices,
        }

    return {
        "x_tree": x_all[hf_indices],
        "y_tree": y_all[hf_indices],
        "x_gp_offset": x_all,
        "coords_gp": coords_mf,
        "y_gp": y_all,
        "tree_row_indices": hf_indices,
    }


def _prepare_model3_prediction_arrays(validation_data, HF_only):
    x_validation = validation_data.iloc[:, :-2].to_numpy(dtype=float)

    if HF_only:
        return x_validation, x_validation

    coords_validation_mf = validation_data.iloc[:, :-1].to_numpy(dtype=float)
    return x_validation, coords_validation_mf


def _validate_model3_training_arrays(arrays):
    _validate_training_arrays(
        arrays["x_tree"],
        arrays["x_tree"],
        arrays["y_tree"],
        model_part="Model 3 fixed-effect tree",
    )
    _validate_training_arrays(
        arrays["x_gp_offset"],
        arrays["coords_gp"],
        arrays["y_gp"],
        model_part="Model 3 GP",
    )


def _validate_training_arrays(x_train, coords_train, y_train, model_part="Model 3"):
    if x_train.ndim != 2:
        raise ValueError(f"{model_part} predictors must be a 2D array.")
    if coords_train.ndim != 2:
        raise ValueError(f"{model_part} GP coordinates must be a 2D array.")
    if len(x_train) != len(coords_train) or len(x_train) != len(y_train):
        raise ValueError(
            f"{model_part} predictors, GP coordinates, and response lengths differ."
        )
    if len(y_train) < 2:
        raise ValueError(f"{model_part} needs at least two training rows.")


def _tuning_base_params(params):
    # Same base as _fit_params, so any key a search space leaves untouched has
    # the same value while tuning as it does in the final fit.
    tuning_params = dict(DEFAULT_BOOSTING_PARAMS)
    if params is not None:
        tuning_params.update(params)
    return tuning_params


def _fit_params(params, tuning_result):
    fit_params = dict(DEFAULT_BOOSTING_PARAMS)

    if tuning_result is not None and "best_params" in tuning_result:
        fit_params.update(tuning_result["best_params"])

    if params is not None:
        fit_params.update(params)

    return fit_params


def _validate_fixed_effect(fixed_effect, y, fit_params, num_boost_round):
    """Fail on a diverged boosted mean before the GP sees it.

    A learning rate at or above 2 makes L2 boosting expand rather than shrink
    the residual, so the fixed effect grows geometrically with the number of
    rounds. The values stay finite for a long while (~1e154 at learning_rate
    3.2 over 1000 rounds) but squaring them inside the likelihood overflows,
    and GPBoost then reports only "NaN occurred in initial negative
    log-likelihood" from inside the GP fit, with no hint of where it came from.
    """
    # Generous: a sane fixed effect sits within the response range, and the
    # diverged one is many orders of magnitude past this bound.
    scale_limit = 1e3 * max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    largest = float(np.max(np.abs(fixed_effect))) if len(fixed_effect) else 0.0

    if np.all(np.isfinite(fixed_effect)) and largest <= scale_limit:
        return

    raise ValueError(
        "Model 3 boosted fixed effect diverged: largest absolute value "
        f"{largest:.3e} against a response scale of {float(np.max(np.abs(y))):.3e}, "
        f"after {num_boost_round} boosting rounds with "
        f"learning_rate={fit_params.get('learning_rate')}, "
        f"num_leaves={fit_params.get('num_leaves')}, "
        f"min_data_in_leaf={fit_params.get('min_data_in_leaf')}. "
        "Lower the learning_rate bound in the tuning search space."
    )


def _gp_fit_params(train_gp_model_cov_pars, init_cov_pars=None):
    fit_params = {"trace": False}
    if init_cov_pars is not None:
        fit_params["init_cov_pars"] = np.asarray(init_cov_pars, dtype=np.float64)
    if not train_gp_model_cov_pars:
        fit_params["maxit"] = 0
    return fit_params


def _initial_cov_pars_for_gp_model(gp_model, y, coords, offset=None):
    """Choose finite positive GP covariance starting values from the data scale."""
    cov_par_names = getattr(gp_model, "cov_par_names", None)
    if not cov_par_names:
        return None

    variance = _variance_scale(y, offset=offset)
    range_scale = _coordinate_range_scale(coords)

    gp_var_names = [
        name
        for name in cov_par_names
        if "var" in name.lower() and name.lower() != "error_var"
    ]
    gp_var = max(0.9 * variance / max(len(gp_var_names), 1), np.finfo(float).tiny)
    error_var = max(0.1 * variance, np.finfo(float).tiny)

    init_cov_pars = []
    for name in cov_par_names:
        lower_name = name.lower()
        if lower_name == "error_var":
            init_cov_pars.append(error_var)
        elif "range" in lower_name:
            init_cov_pars.append(range_scale)
        elif lower_name == "rho":
            init_cov_pars.append(1.0)
        elif "var" in lower_name:
            init_cov_pars.append(gp_var)
        else:
            init_cov_pars.append(variance)

    return np.asarray(init_cov_pars, dtype=np.float64)


def _variance_scale(y, offset=None):
    values = np.asarray(y, dtype=float).reshape(-1)
    if offset is not None:
        values = values - np.asarray(offset, dtype=float).reshape(-1)

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return 1.0

    variance = float(np.var(finite_values))
    magnitude = max(
        float(np.nanmedian(np.abs(finite_values))),
        float(np.nanstd(finite_values)),
        1.0,
    )
    min_variance = max(np.finfo(float).eps * magnitude * magnitude, np.finfo(float).tiny)
    if not np.isfinite(variance) or variance <= 0:
        variance = min_variance
    return max(variance, min_variance)


def _coordinate_range_scale(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return 1.0

    range_coords = coords
    if coords.shape[1] > 1:
        last_col = coords[:, -1]
        finite_last = last_col[np.isfinite(last_col)]
        unique_last = np.unique(finite_last)
        if len(unique_last) <= 2 and np.all(np.isin(unique_last, [0.0, 1.0])):
            range_coords = coords[:, :-1]

    finite_rows = range_coords[np.all(np.isfinite(range_coords), axis=1)]
    if len(finite_rows) < 2:
        return 1.0

    max_rows = min(512, len(finite_rows))
    if len(finite_rows) > max_rows:
        rng = np.random.default_rng(0)
        finite_rows = finite_rows[
            rng.choice(len(finite_rows), size=max_rows, replace=False)
        ]

    diffs = finite_rows[:, np.newaxis, :] - finite_rows[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    upper = distances[np.triu_indices(len(finite_rows), k=1)]
    positive = upper[np.isfinite(upper) & (upper > 0)]
    if len(positive) == 0:
        return 1.0

    range_scale = float(np.median(positive))
    if not np.isfinite(range_scale) or range_scale <= 0:
        return 1.0
    return max(range_scale, np.finfo(float).eps)


def _split_model3_predict_kwargs(predict_kwargs):
    if "offset_pred" in predict_kwargs:
        raise ValueError("Model 3 sets offset_pred from the boosted fixed-effect term.")

    gp_keys = {
        "cov_pars",
        "group_data_pred",
        "group_rand_coef_data_pred",
        "gp_rand_coef_data_pred",
        "cluster_ids_pred",
        "y",
        "offset",
        "use_saved_data",
    }
    tree_kwargs = dict(predict_kwargs)
    gp_kwargs = {}
    for key in list(tree_kwargs):
        if key in gp_keys:
            gp_kwargs[key] = tree_kwargs.pop(key)
    return tree_kwargs, gp_kwargs


def _predict_tree_fixed_effect(
    booster,
    data,
    num_iteration=None,
    predict_kwargs=None,
):
    predict_kwargs = dict(predict_kwargs or {})

    if predict_kwargs.pop("pred_leaf", False):
        raise ValueError("Model 3 response prediction does not support pred_leaf=True.")
    if predict_kwargs.pop("pred_contrib", False):
        raise ValueError("Model 3 response prediction does not support pred_contrib=True.")

    predict_kwargs.pop("raw_score", None)
    predict_kwargs.pop("pred_latent", None)
    fixed_effect = booster.predict(
        data=data,
        num_iteration=num_iteration,
        pred_latent=True,
        **predict_kwargs,
    )
    fixed_effect = np.asarray(fixed_effect, dtype=float)
    return fixed_effect.reshape(-1)


def _best_num_boost_round(tuning_result, default):
    if tuning_result is None:
        return default

    best_iter = tuning_result.get("best_iter")
    if best_iter is None:
        return default

    return int(best_iter)


def _make_validation_folds(n_rows, validation_fraction, random_state):
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    n_valid = max(1, int(round(validation_fraction * n_rows)))
    if n_valid >= n_rows:
        raise ValueError("validation_fraction leaves no training rows.")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_rows)
    valid_idx = np.sort(shuffled[:n_valid])
    train_idx = np.sort(shuffled[n_valid:])
    return [(train_idx, valid_idx)]


def _resolve_model3_booster(model, HF_only):
    if model is not None:
        return model

    if _model3 is None:
        raise RuntimeError("fit_model3 must be called before predict_model3.")

    if _model3_HF_only is not None and HF_only != _model3_HF_only:
        raise ValueError(
            "predict_model3 was called with a different HF_only value than fit_model3."
        )

    return _model3


def _with_pipeline_prediction_aliases(prediction, pred_latent):
    if not isinstance(prediction, dict):
        return {"mu": np.asarray(prediction, dtype=float), "raw_prediction": prediction}

    prediction = dict(prediction)

    if prediction.get("response_mean") is not None:
        prediction["mu"] = prediction["response_mean"]
    elif pred_latent and prediction.get("fixed_effect") is not None:
        prediction["mu"] = (
            np.asarray(prediction["fixed_effect"], dtype=float)
            + np.asarray(prediction["random_effect_mean"], dtype=float)
        )
    else:
        raise KeyError("GPBoost prediction did not include a usable mean.")

    if prediction.get("response_var") is not None:
        prediction["var"] = prediction["response_var"]
    elif pred_latent and prediction.get("random_effect_cov") is not None:
        prediction["var"] = prediction["random_effect_cov"]
    else:
        prediction["var"] = np.full(len(prediction["mu"]), np.nan)

    return prediction


def _get_model3_metadata(
    train_data,
    HF_only,
    params,
    num_boost_round,
    gp_kwargs,
    n_tree_train,
    n_gp_train,
):
    response_column = train_data.columns[-1]
    response_index = None
    wavelength = None

    if isinstance(response_column, str) and response_column.startswith("response_"):
        response_index = int(response_column.removeprefix("response_"))
        wavelength_map = train_data.attrs.get("wavelength_map", {})
        wavelength = wavelength_map.get(response_index)

    return {
        "model": "model3",
        "HF_only": HF_only,
        "response_column": response_column,
        "response_index": response_index,
        "wavelength": wavelength,
        "n_rows": len(train_data),
        "n_train": n_gp_train,
        "n_tree_train": n_tree_train,
        "n_gp_train": n_gp_train,
        "n_hf": int(np.sum(train_data.iloc[:, -2].to_numpy() == 1)),
        "n_lf": int(np.sum(train_data.iloc[:, -2].to_numpy() == 0)),
        "n_lf_in_fixed_effect": 0,
        "fixed_effect_training": "high_fidelity_only",
        "lf_used_in_fixed_effect": False,
        "gp_covariance": "matern" if HF_only else "ar1_mf_matern",
        "x_columns": list(train_data.columns[:-2]),
        "params": params,
        "num_boost_round": num_boost_round,
        "gp_kwargs": gp_kwargs,
    }


def _get_kernel_parameters(booster):
    gp_model = getattr(booster, "gp_model", None)
    if gp_model is None:
        return None

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


def _unique_preserving_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(int(value) if isinstance(value, np.integer) else value)
    return unique


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
