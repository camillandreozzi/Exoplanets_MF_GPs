"""Model 3: GPBoost helpers for the exoplanet pipeline.

The expected input shape matches ``model_1``:
- input columns
- ``is_hf`` fidelity indicator
- one scalar response column

This implements the GPBoost algorithm of Sigrist (2022, JMLR 23:232): the tree
ensemble and the covariance parameters are estimated jointly, in one interleaved
loop. Every boosting iteration first re-optimizes theta against the Gaussian
process negative log-likelihood and then fits a tree that sees the resulting
Psi^-1, either as the whitened pseudo-response or as the weight matrix in the
leaf solve. ``gpb.train`` runs that loop for us; handing it ``gp_model=`` and
``train_gp_model_cov_pars=True`` is what makes it the paper's algorithm rather
than plain L2 boosting followed by a GP on the residuals.

Both variants train the trees and the GP on the same rows, because the
algorithm requires it. For the multi-fidelity variant that means all rows: the
AR(1) covariance spans low- and high-fidelity samples, so the ensemble does too,
with the fidelity indicator appended as a boosting feature and separate LF/HF
marginal means.

The GP uses standardized physical coordinates, with scaling learned from
training HF rows only. Tree inputs remain raw.
"""

import json
from pathlib import Path

import gpboost as gpb
import numpy as np


GP_THREADS = 1
DEFAULT_LIKELIHOOD = "gaussian"
# Squared exponential with Automatic Relevance Determination: one range per
# input dimension, so the nine physical parameters are free to act on different
# length scales instead of sharing one. The multi-fidelity form wraps the same
# base in the two-level AR(1) structure, giving the low-fidelity process and
# the discrepancy their own parameter blocks. No cov_fct_shape here -- it is a
# Matern/powered-exponential smoothness and has no meaning for a Gaussian one.
SF_COV_FUNCTION = "gaussian_ard"
MF_COV_FUNCTION = "ar1_mf_gaussian_ard"
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
    # Not optional at this response scale. The functional gradient the trees
    # fit is Psi^-1 (y - F), and a plain gradient step is not invariant to the
    # scale of the loss: the fluxes here have sd ~4e-4, so the estimated error
    # variance is ~2e-8 and Psi^-1 carries entries of order 1e7, making the raw
    # step overshoot by the same factor. On response_100 that drove the fixed
    # effect to +/-225 against data of size 1e-3. The line search restores the
    # scale invariance that Newton-type steps have by construction, and the
    # same fit then lands in [5.6e-4, 1.3e-3].
    "line_search_step_length": True,
}

_model3 = None
_model3_HF_only = None
_model3_metadata = {}
_model3_last_prediction = None
_model3_last_tuning = None


class Model3Fit:
    """Fitted Model 3: a GPBoost booster carrying its jointly estimated GP.

    Thin wrapper over the booster. Its only job is the coordinate scaling,
    which is ours rather than GPBoost's and so has to travel with the model;
    everything else is delegated, because after a joint fit the booster already
    knows how to combine its trees with the GP.
    """

    def __init__(
        self,
        booster,
        HF_only,
        n_tree_train=None,
        n_gp_train=None,
        gp_input_mean=None,
        gp_input_std=None,
    ):
        self.booster = booster
        self.HF_only = HF_only
        self.n_tree_train = n_tree_train
        self.n_gp_train = n_gp_train
        self.gp_input_mean = gp_input_mean
        self.gp_input_std = gp_input_std

    @property
    def gp_model(self):
        """The GP estimated alongside the trees, owned by the booster."""
        return self.booster.gp_model

    def scale_coordinates(self, gp_coords_pred):
        if self.gp_input_mean is None:
            return gp_coords_pred
        return _scale_gp_coordinates(
            gp_coords_pred, self.gp_input_mean, self.gp_input_std
        )

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
        """Predict through the booster, scaling the GP coordinates first.

        ``data`` carries the bare input columns. For the multi-fidelity variant
        the tree also expects the fidelity indicator, but ``Booster.predict``
        appends it from the last column of ``gp_coords_pred`` whenever the two
        widths differ by one, so callers do not have to.
        """
        _reject_offset_pred(predict_kwargs)
        return self.booster.predict(
            data=data,
            gp_coords_pred=self.scale_coordinates(gp_coords_pred),
            predict_var=predict_var,
            pred_latent=pred_latent,
            num_iteration=num_iteration,
            predict_cov_mat=predict_cov_mat,
            sample_posterior=sample_posterior,
            num_post_samples=num_post_samples,
            **predict_kwargs,
        )

    def save_model(self, filename):
        """Save the booster, which carries the GP, plus our input scaling."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if getattr(self.booster, "train_set", None) is None:
            # Saving a booster that owns a GP serializes its training data
            # alongside, and a booster restored from a file no longer holds
            # any, so GPBoost cannot write it back out. Caught here because it
            # otherwise surfaces as an AttributeError from inside the library.
            raise ValueError(
                "This Model 3 fit was loaded from a file and cannot be saved "
                "again; save the model when it is first fitted."
            )
        # With a gp_model attached the booster writes a JSON document holding
        # the GP and the training residuals, so there is no separate GP file.
        self.booster.save_model(str(filename))
        # Keep preprocessing with the model even when save_model() is called
        # directly, without the module-level diagnostics writer.
        scaling_file = filename.with_suffix(filename.suffix + ".input_scaling.json")
        scaling_file.write_text(json.dumps(_to_jsonable({
            "HF_only": self.HF_only,
            "mean": self.gp_input_mean,
            "std": self.gp_input_std,
        }), indent=2), encoding="utf-8")
        return {
            "model_file": str(filename),
            "tree_model_file": str(filename),
            "gp_model_file": None,
            "input_scaling_file": str(scaling_file),
        }


class LegacyModel3Fit(Model3Fit):
    """A Model 3 fitted before the switch to the real GPBoost algorithm.

    Those runs boosted the trees against plain L2 loss and only then fit the GP
    once, with the frozen tree predictions as a fixed offset, so the booster
    carries no GP and the two parts have to be recombined by hand. Kept so the
    stored runs under ``results/model3`` still load and predict; nothing writes
    this shape any more.
    """

    def __init__(self, booster, gp_model, HF_only, **kwargs):
        super().__init__(booster, HF_only, **kwargs)
        self._gp_model = gp_model

    @property
    def gp_model(self):
        return self._gp_model

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

        gp_coords_pred = self.scale_coordinates(gp_coords_pred)

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
        scaling_file = filename.with_suffix(filename.suffix + ".input_scaling.json")
        scaling_file.write_text(json.dumps(_to_jsonable({
            "HF_only": self.HF_only,
            "mean": self.gp_input_mean,
            "std": self.gp_input_std,
        }), indent=2), encoding="utf-8")
        return {
            "model_file": str(filename),
            "tree_model_file": str(filename),
            "gp_model_file": str(gp_model_file),
            "input_scaling_file": str(scaling_file),
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
        Selects which variant to tune. The two no longer share a result: the
        ensembles differ in both row count and covariance, so the row-count
        bounds in the search space and the GP in the validation loop differ
        with them.
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
    elif folds is None and not HF_only:
        folds = _make_fidelity_stratified_folds(
            fidelity=arrays["coords_gp"][:, -1],
            nfold=nfold,
            random_state=seed,
        )

    method = method.lower()
    tuning_params = _tuning_base_params(params)
    tuning_gp_model = _make_tuning_gp_model(
        arrays,
        HF_only=HF_only,
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
        "tuning_gp_covariance": tuning_gp_model.cov_function,
        "use_gp_model_for_validation": use_gp_model_for_validation,
        "train_gp_model_cov_pars": train_gp_model_cov_pars,
        "validation_fraction": validation_fraction,
        "fixed_effect_training": (
            "high_fidelity_only" if HF_only else "all_fidelities"
        ),
        "lf_used_in_fixed_effect": not HF_only,
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

    Runs the GPBoost algorithm: ``gpb.train`` alternates, within one loop,
    between re-estimating the covariance parameters and adding a tree fitted
    against the current Psi^-1. Trees and GP therefore span the same rows --
    high-fidelity only when ``HF_only``, otherwise every row, with an AR(1)
    multi-fidelity covariance and fidelity-specific marginal means. Pass
    ``tuning_kwargs`` to override ``tune_model3_parameters`` defaults.

    GP coordinates are standardized using the training HF means and standard
    deviations. Trees and their tuning retain unscaled inputs.
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

    # Only the GP is standardized; the trees retain raw inputs. Learn one
    # shared transform from this fit's HF rows, not from every tree row: the
    # multi-fidelity ensemble now trains on low-fidelity rows too, and their
    # spread should not move the coordinates the covariance is measured in.
    gp_input_mean, gp_input_std = _gp_input_scaling(arrays)
    coords_gp = _scale_gp_coordinates(
        arrays["coords_gp"], gp_input_mean, gp_input_std
    )
    gp_model = make_model3_gp_model(
        coords_gp,
        HF_only=HF_only,
        gp_kwargs=gp_kwargs,
    )
    # The starting covariance parameters only reach the fit through the GP's
    # own optimizer config: gpb.train never reads init_cov_pars out of the
    # booster params, so setting them here is not interchangeable with passing
    # them to gpb.train.
    gp_model.set_optim_params(
        params=_gp_fit_params(
            train_gp_model_cov_pars,
            init_cov_pars=_initial_cov_pars_for_gp_model(
                gp_model=gp_model,
                y=arrays["y_gp"],
                coords=coords_gp,
            ),
        )
    )

    # free_raw_data=False is required, not a preference: the fidelity-specific
    # mean appends a column to the retained Dataset data, and saving a booster
    # that owns a GP fails outright once that data has been freed.
    train_set = gpb.Dataset(
        data=arrays["x_tree"],
        label=arrays["y_tree"],
        free_raw_data=False,
    )
    # Handing over the gp_model is what makes this the GPBoost algorithm: each
    # iteration re-estimates the covariance parameters and then fits its tree
    # against the resulting Psi^-1.
    booster = gpb.train(
        params=fit_params,
        train_set=train_set,
        gp_model=gp_model,
        train_gp_model_cov_pars=train_gp_model_cov_pars,
        use_gp_model_for_validation=use_gp_model_for_validation,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=verbose_eval,
    )
    _validate_fixed_effect(
        _predict_tree_fixed_effect(
            booster,
            data=arrays["x_tree"],
            gp_coords_pred=coords_gp,
        ),
        arrays["y_gp"],
        fit_params,
        num_boost_round,
    )

    model = Model3Fit(
        booster=booster,
        HF_only=HF_only,
        n_tree_train=len(arrays["y_tree"]),
        n_gp_train=len(arrays["y_gp"]),
        gp_input_mean=gp_input_mean,
        gp_input_std=gp_input_std,
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
    _model3_metadata["input_scaling"] = {
        "applied_to": "residual_gp_coordinates_only",
        "fitted_on": "training_hf",
        "mean": gp_input_mean,
        "std": gp_input_std,
    }
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
    """Load a saved Model 3 object into the module-level prediction state.

    Reads both on-disk shapes. A jointly fitted booster carries its GP, so the
    file stands alone; runs saved before the switch to the GPBoost algorithm
    left the GP in a ``.gp_model.json`` sidecar next to a GP-less booster, and
    those load as a ``LegacyModel3Fit``.
    """
    global _model3, _model3_HF_only, _model3_metadata, _model3_last_prediction

    filename = Path(filename)
    booster = gpb.Booster(model_file=str(filename))
    gp_model_file = filename.with_suffix(filename.suffix + ".gp_model.json")
    scaling_file = filename.with_suffix(filename.suffix + ".input_scaling.json")
    scaling = json.loads(scaling_file.read_text(encoding="utf-8")) if scaling_file.exists() else {}
    saved_hf_only = scaling.get("HF_only")
    if HF_only is None:
        HF_only = saved_hf_only
    elif saved_hf_only is not None and HF_only != saved_hf_only:
        raise ValueError("HF_only does not match the saved Model 3 fidelity variant.")

    scaling_kwargs = {
        "gp_input_mean": (np.asarray(scaling["mean"], dtype=float)
                          if scaling.get("mean") is not None else None),
        "gp_input_std": (np.asarray(scaling["std"], dtype=float)
                         if scaling.get("std") is not None else None),
    }

    legacy = not booster.has_gp_model and gp_model_file.exists()
    if booster.has_gp_model:
        model = Model3Fit(booster=booster, HF_only=HF_only, **scaling_kwargs)
    elif legacy:
        model = LegacyModel3Fit(
            booster=booster,
            gp_model=gpb.GPModel(model_file=str(gp_model_file)),
            HF_only=HF_only,
            **scaling_kwargs,
        )
    else:
        model = booster

    _model3 = model
    _model3_HF_only = HF_only
    _model3_metadata = {
        "model": "model3",
        "HF_only": HF_only,
        "model_file": str(filename),
        "gp_model_file": str(gp_model_file) if legacy else None,
        "legacy_two_stage_fit": legacy,
        "input_scaling": scaling,
    }
    _model3_last_prediction = None
    return model


def _scale_gp_coordinates(coords, mean, std):
    """Scale physical coordinates without mutating inputs or MF fidelity flags."""
    scaled = np.array(coords, dtype=float, copy=True)
    n_features = len(mean)
    scaled[:, :n_features] = (scaled[:, :n_features] - mean) / std
    return scaled


def make_model3_gp_model(gp_coords, HF_only=False, gp_kwargs=None):
    """Create the GP component used by Model 3."""
    gp_coords = np.asarray(gp_coords, dtype=float)
    if gp_coords.ndim != 2:
        raise ValueError("gp_coords must be a 2D array.")

    kwargs = {
        "gp_coords": gp_coords,
        "cov_function": SF_COV_FUNCTION if HF_only else MF_COV_FUNCTION,
        "gp_approx": DEFAULT_GP_APPROX_SF if HF_only else DEFAULT_GP_APPROX_MF,
        "num_neighbors": 20,
        "likelihood": DEFAULT_LIKELIHOOD,
        "num_parallel_threads": GP_THREADS,
    }
    if gp_kwargs is not None:
        kwargs.update(gp_kwargs)

    # fidelity_specific_mean is left at the GPBoost default, which is True for
    # an ar1_mf_ covariance: gpb.train then appends the fidelity indicator from
    # the last coordinate column as a boosting feature, so the two fidelities
    # get their own marginal means instead of sharing one.

    if kwargs.get("gp_approx") == "vecchia" and kwargs.get("num_neighbors") is not None:
        kwargs["num_neighbors"] = min(int(kwargs["num_neighbors"]), gp_coords.shape[0] - 1)
        if kwargs["num_neighbors"] < 1:
            raise ValueError("Vecchia GP approximation needs at least two training rows.")
    elif kwargs.get("gp_approx") != "vecchia":
        kwargs.pop("num_neighbors", None)

    return gpb.GPModel(**kwargs)


def _make_tuning_gp_model(
    arrays,
    HF_only=False,
    gp_kwargs=None,
    train_gp_model_cov_pars=True,
):
    """The GP the tuner validates against: the same one the fit will use.

    Without a ``gp_model`` the tuner scores tree-only CV error, so it selects a
    tree that absorbs the smooth structure the GP exists to model: on one LOO
    fold of response_100 that was learning_rate 1.49 with 243 leaves, against
    0.099 with 4 leaves once the GP is in the loop. For the multi-fidelity
    variant that would leave the AR(1) component nothing to carry.

    Known discrepancy, for the multi-fidelity variant only. GPBoost's CV path
    does not reproduce a fidelity-specific mean: it rebuilds each fold's GP
    from an explicit attribute list that omits ``fidelity_specific_mean``, and
    it constructs fold boosters directly rather than through ``gpb.train``,
    which is the only place the fidelity feature is appended. Folds therefore
    behave as if the mean were shared, while the final fit gives each fidelity
    its own -- quietly, with no error. The tuned tree parameters are still
    chosen against the right covariance on the right rows, which is what they
    are for, but they are not chosen against exactly the fitted model.
    """
    gp_model = make_model3_gp_model(
        arrays["coords_gp"],
        HF_only=HF_only,
        gp_kwargs=gp_kwargs,
    )
    init_cov_pars = _initial_cov_pars_for_gp_model(
        gp_model=gp_model,
        y=arrays["y_gp"],
        coords=arrays["coords_gp"],
    )
    gp_model.set_optim_params(
        params=_gp_fit_params(
            train_gp_model_cov_pars,
            init_cov_pars=init_cov_pars,
        )
    )
    return gp_model


def _make_fidelity_stratified_folds(fidelity, nfold, random_state):
    """k-fold indices that keep both fidelities in every fold.

    GPBoost's own splitter shuffles row indices without regard to fidelity, and
    the low-fidelity rows outnumber the high-fidelity ones by two orders of
    magnitude here, so a plain split can hand the AR(1) covariance a fold with
    almost no high-fidelity rows to correlate against.
    """
    fidelity = np.asarray(fidelity).reshape(-1)
    rng = np.random.default_rng(random_state)
    assignment = np.empty(len(fidelity), dtype=int)

    for value in np.unique(fidelity):
        positions = np.where(fidelity == value)[0]
        shuffled = rng.permutation(len(positions))
        assignment[positions[shuffled]] = np.arange(len(positions)) % nfold

    folds = []
    for fold in range(nfold):
        test_idx = np.where(assignment == fold)[0]
        train_idx = np.where(assignment != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def default_model3_search_space(n_train):
    """Default Optuna/TPE ranges from the GPBoost template, bounded by data size.

    ``learning_rate`` spans the template's full range. It used to stop at 1,
    because the fit boosted the trees on their own and plain L2 boosting only
    contracts the residual below a rate of 2 -- above it the fixed effect grew
    geometrically and overflowed the likelihood. Boosting now runs against the
    GP, with an optional line search for the step length, so the template bound
    applies again.
    """
    max_bin = max(63, min(10000, int(n_train)))
    min_data_in_leaf_upper = max(1, min(1000, int(n_train)))

    return {
        "learning_rate": [0.001, 10],
        "min_data_in_leaf": [1, min_data_in_leaf_upper],
        "max_depth": [-1, -1],
        "num_leaves": [2, 1024],
        "lambda_l2": [0, 100],
        "max_bin": [63, max_bin],
        "feature_fraction": [0.5, 1],
    }


def default_model3_param_grid(n_train):
    """Default random-grid search values from the GPBoost template."""
    max_bin_values = _unique_preserving_order(
        [250, 500, 1000, min(10000, int(n_train))]
    )

    return {
        "learning_rate": [0.001, 0.01, 0.1, 1, 10],
        "min_data_in_leaf": _unique_preserving_order(
            [1, 10, 100, min(1000, int(n_train))]
        ),
        "max_depth": [-1],
        "num_leaves": [int(value) for value in 2 ** np.arange(1, 10)],
        "lambda_l2": [0, 1, 10, 100],
        "max_bin": max_bin_values,
        "feature_fraction": [0.5, 0.75, 1],
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
            "coords_gp": x_all[hf_indices],
            "y_gp": y_all[hf_indices],
            "x_hf": x_all[hf_indices],
            "tree_row_indices": hf_indices,
        }

    # Tree rows and GP rows have to coincide for the GPBoost algorithm, so the
    # multi-fidelity ensemble trains on the low-fidelity rows as well. They
    # reach the trees through the fidelity indicator gpb.train appends, which
    # is what gives the two fidelities their own marginal means.
    return {
        "x_tree": x_all,
        "y_tree": y_all,
        "coords_gp": coords_mf,
        "y_gp": y_all,
        "x_hf": x_all[hf_indices],
        "tree_row_indices": np.arange(len(y_all)),
    }


def _gp_input_scaling(arrays):
    """Standardization for the GP coordinates, learned on HF rows only."""
    x_hf = arrays["x_hf"]
    mean = x_hf.mean(axis=0)
    std = x_hf.std(axis=0, ddof=0)
    return mean, np.where(std == 0, 1.0, std)


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
        arrays["x_tree"],
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
    """Fail on a diverged boosted mean rather than let it reach a prediction.

    The trees fit the functional gradient Psi^-1 (y - F), whose scale is that
    of the inverse error variance -- about 1e7 for these fluxes. Without
    line_search_step_length the step inherits that factor and the ensemble runs
    away geometrically: on response_100 it reached +/-225 against data of size
    1e-3, and the covariance parameters chased it up by nine orders of
    magnitude. The values stay finite, so nothing downstream would notice.
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
        f"line_search_step_length={fit_params.get('line_search_step_length')}. "
        "line_search_step_length=False is the usual cause at this response scale."
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


def _reject_offset_pred(predict_kwargs):
    if "offset_pred" in predict_kwargs:
        raise ValueError("Model 3 takes its fixed effect from the boosted trees.")


def _split_model3_predict_kwargs(predict_kwargs):
    """Route prediction kwargs for a legacy fit, whose GP is a separate model."""
    _reject_offset_pred(predict_kwargs)

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
    gp_coords_pred=None,
    predict_kwargs=None,
):
    """The tree ensemble's contribution alone, whether or not it owns a GP.

    ``gp_coords_pred`` is only needed for a booster with a fidelity-specific
    mean, which reads the indicator off its last column to complete the tree
    features; it is ignored otherwise.
    """
    predict_kwargs = dict(predict_kwargs or {})

    if predict_kwargs.pop("pred_leaf", False):
        raise ValueError("Model 3 response prediction does not support pred_leaf=True.")
    if predict_kwargs.pop("pred_contrib", False):
        raise ValueError("Model 3 response prediction does not support pred_contrib=True.")

    predict_kwargs.pop("raw_score", None)
    predict_kwargs.pop("pred_latent", None)
    predict_kwargs.pop("ignore_gp_model", None)
    if gp_coords_pred is not None:
        predict_kwargs["gp_coords_pred"] = gp_coords_pred
    fixed_effect = booster.predict(
        data=data,
        num_iteration=num_iteration,
        pred_latent=True,
        ignore_gp_model=True,
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
        "n_lf_in_fixed_effect": (
            0 if HF_only else int(np.sum(train_data.iloc[:, -2].to_numpy() == 0))
        ),
        "fixed_effect_training": (
            "high_fidelity_only" if HF_only else "all_fidelities"
        ),
        "lf_used_in_fixed_effect": not HF_only,
        "algorithm": "gpboost_joint",
        "gp_covariance": (gp_kwargs or {}).get(
            "cov_function", SF_COV_FUNCTION if HF_only else MF_COV_FUNCTION
        ),
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
