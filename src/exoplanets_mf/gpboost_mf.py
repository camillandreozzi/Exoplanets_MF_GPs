"""GPBoost adapters for per-wavelength AR(1) multi-fidelity GPs.

The rest of the project uses sklearn kernels for the main MF-GP
implementations. This module keeps GPBoost optional and contained: importing
the module succeeds even when GPBoost is absent, while calling the fit helpers
requires the package to be importable.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exoplanets_mf.cv import CV_FULL_MODEL_SPLITS, CVPredictions, cv_predict
from exoplanets_mf.mf_gp import _append_fidelity, select_lf_subsample
from exoplanets_mf.model2 import lf_column_moments

GPBOOST_COV_FUNCTION = "ar1_mf_gaussian_ard"
GPBOOST_COV_FCT_SHAPE = 1.5
GPBOOST_GP_APPROX = "none"
GPBOOST_NUM_NEIGHBORS = None
_VAR_FLOOR = 1e-12

# Model 1A (shared rho) block-coordinate sweep. GPBoost's rho is unrestricted
# and Model 1B routinely fits rho_j < 0 here, so the shared-rho search runs on
# a LINEAR grid over GPBOOST_GLOBAL_RHO_BOUNDS (unlike the sklearn Model 1A,
# whose kernel constrains rho > 0 and searches log rho). The grid is global at
# the first stage so a local mode near rho = 0 cannot capture the update, then
# zooms: 41 points over the full range, then 21-point refinements, giving a
# final resolution of ~(range/40) * 0.1^(stages-2) << GPBOOST_GLOBAL_RHO_TOL.
# GPBoost's own starting values collapse every ARD length-scale to ~0.3 on
# this design, while the closest two HF planets sit 2.2 apart in standardized
# input space: every off-diagonal covariance underflows and the fit converges
# to a degenerate "fidelity-specific mean + nugget" optimum that predicts the
# same spectrum for every new planet. GPBOOST_HEURISTIC_INIT starts the
# optimizer from length-scales on the scale of the actual point spacing (the
# median heuristic: correlation ~ exp(-1) at the median pairwise distance),
# which reaches a far higher likelihood and a model that actually varies with
# its inputs. Pass heuristic_init=False to reproduce GPBoost's own defaults.
GPBOOST_HEURISTIC_INIT = True
GPBOOST_INIT_ERROR_VAR_FRACTION = 0.1
GPBOOST_INIT_LOW_VAR_FRACTION = 0.9
GPBOOST_INIT_DISCREPANCY_VAR_FRACTION = 0.5
GPBOOST_INIT_RHO = 1.0
GPBOOST_INIT_MAX_DISTANCE_ROWS = 600  # rows sampled for the distance heuristic

GPBOOST_GLOBAL_RHO_BOUNDS = (-3.0, 3.0)
GPBOOST_GLOBAL_RHO_GRID_STAGES = 4
GPBOOST_GLOBAL_RHO_MAX_SWEEPS = 25
GPBOOST_GLOBAL_RHO_TOL = 1e-3  # |rho change| below which a sweep has converged


@lru_cache(maxsize=1)
def _gpboost_module():
    import gpboost as gpb

    return gpb


def gpboost_available() -> bool:
    """Return True when the optional GPBoost dependency can be imported."""
    try:
        _gpboost_module()
    except Exception:
        return False
    return True


def _require_gpboost():
    try:
        return _gpboost_module()
    except Exception as exc:  # pragma: no cover - exercised only without GPBoost
        raise ImportError(
            "GPBoost is required for the GPBoost MF-GP adapters. "
            "Install gpboost or let tests skip these optional models."
        ) from exc


def _median_length_scales(
    features: np.ndarray, n_ranges: int, *, seed: int
) -> np.ndarray:
    """Length-scale starting values on the scale of the point spacing.

    The median heuristic: choose length-scales so that two points at the
    median pairwise separation have correlation ~ exp(-1). With ARD, the
    squared distance is a sum over ``n_dims`` coordinates, so each per-axis
    length-scale carries a sqrt(n_dims) factor -- without it a nominally
    "typical" length-scale still drives the summed exponent to a magnitude
    that underflows in this many dimensions.
    """
    rng = np.random.default_rng(seed)
    rows = features
    if len(rows) > GPBOOST_INIT_MAX_DISTANCE_ROWS:
        rows = rows[
            rng.choice(len(rows), GPBOOST_INIT_MAX_DISTANCE_ROWS, replace=False)
        ]
    n_dims = rows.shape[1]
    upper = np.triu_indices(len(rows), k=1)
    if n_ranges == 1:
        distances = np.sqrt(
            ((rows[:, None, :] - rows[None, :, :]) ** 2).sum(axis=-1)
        )[upper]
        scale = np.median(distances[distances > 0.0]) if distances.size else 1.0
        return np.array([max(float(scale), 1e-3)])
    scales = np.empty(n_ranges)
    for k in range(n_ranges):
        column = rows[:, k] if k < n_dims else rows[:, -1]
        gaps = np.abs(column[:, None] - column[None, :])[upper]
        positive = gaps[gaps > 0.0]
        median_gap = float(np.median(positive)) if positive.size else 1.0
        scales[k] = max(median_gap * np.sqrt(n_dims), 1e-3)
    return scales


def heuristic_init_cov_pars(
    coords: np.ndarray,
    y: np.ndarray,
    num_cov_pars: int,
    *,
    seed: int,
    rho_init: float = GPBOOST_INIT_RHO,
) -> np.ndarray:
    """Starting covariance parameters in GPBoost's own order.

    Order for an ``ar1_mf_<base>`` covariance is
    ``[Error_var, low_var, low_ranges..., discrepancy_var,
    discrepancy_ranges..., rho]``, so ``num_cov_pars == 4 + 2 * n_ranges``.
    Variances are anchored on the observed LF/HF spread (the targets are
    standardized per wavelength upstream, so these are O(1)); length-scales
    come from the median heuristic on the fidelity-free coordinates.
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float)
    n_ranges, remainder = divmod(num_cov_pars - 4, 2)
    if remainder or n_ranges < 1:
        raise ValueError(
            f"cannot infer the ar1_mf parameter layout from {num_cov_pars} "
            "covariance parameters"
        )
    features, fidelity = coords[:, :-1], coords[:, -1]
    scales = _median_length_scales(features, n_ranges, seed=seed)

    y_lf, y_hf = y[fidelity == 0.0], y[fidelity == 1.0]
    total_var = max(float(np.var(y)), 1e-8)
    low_var = max(float(np.var(y_lf)) if y_lf.size else total_var, 1e-8)
    high_var = max(float(np.var(y_hf)) if y_hf.size else total_var, 1e-8)

    init = np.empty(num_cov_pars)
    init[0] = GPBOOST_INIT_ERROR_VAR_FRACTION * total_var
    init[1] = GPBOOST_INIT_LOW_VAR_FRACTION * low_var
    init[2 : 2 + n_ranges] = scales
    init[2 + n_ranges] = GPBOOST_INIT_DISCREPANCY_VAR_FRACTION * high_var
    init[3 + n_ranges : 3 + 2 * n_ranges] = scales
    init[-1] = rho_init
    return init


def _fit_gpboost_ar1(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cov_function: str,
    cov_fct_shape: float,
    gp_approx: str,
    num_neighbors: int | None,
    maxit: int,
    trace: bool,
    num_parallel_threads: int | None,
    fidelity_specific_mean: bool,
    init_cov_pars: np.ndarray | None = None,
    estimate_cov_par_index: np.ndarray | None = None,
    heuristic_init: bool = GPBOOST_HEURISTIC_INIT,
) -> Any:
    """Fit one zero-boosting GPBoost AR(1) model on augmented coordinates.

    ``init_cov_pars`` / ``estimate_cov_par_index`` are GPBoost's mechanism for
    holding covariance parameters at prescribed values: entries of
    ``estimate_cov_par_index`` that are 0 stay at their ``init_cov_pars``
    value. Model 1A uses them to pin rho (the last covariance parameter of an
    ``ar1_mf_*`` covariance) during the fixed-rho half of each sweep.

    With no explicit ``init_cov_pars`` and ``heuristic_init=True``, the
    optimizer starts from ``heuristic_init_cov_pars`` instead of GPBoost's own
    defaults -- see GPBOOST_HEURISTIC_INIT for why that matters here.
    """
    if not cov_function.startswith("ar1_mf_"):
        raise ValueError(
            "GPBoost multi-fidelity models require an ar1_mf_* covariance; "
            f"found {cov_function!r}"
        )
    gpb = _require_gpboost()
    model = gpb.GPModel(
        gp_coords=np.asarray(coords, dtype=float),
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
        likelihood="gaussian",
        seed=seed,
        num_parallel_threads=num_parallel_threads,
        fidelity_specific_mean=fidelity_specific_mean,
    )
    # The intercept gives GPBoost an explicit mean term. With
    # fidelity_specific_mean=True, GPBoost expands it into separate LF/HF
    # means, which is important after column standardization because HF can
    # have a shifted standardized mean even when the AR(1) rho is correct.
    intercept = np.ones((coords.shape[0], 1), dtype=float)
    params: dict[str, Any] = {"maxit": maxit, "trace": trace}
    if init_cov_pars is None and heuristic_init:
        init_cov_pars = heuristic_init_cov_pars(
            coords, y, int(model.num_cov_pars), seed=seed
        )
    if init_cov_pars is not None:
        params["init_cov_pars"] = np.asarray(init_cov_pars, dtype=float)
    if estimate_cov_par_index is not None:
        params["estimate_cov_par_index"] = np.asarray(
            estimate_cov_par_index, dtype=np.int32
        )
    model.fit(
        y=np.asarray(y, dtype=float),
        X=intercept,
        params=params,
    )
    _attach_compat_methods(model, n_points=int(coords.shape[0]))
    return model


def _predict_gpboost_ar1(
    model: Any,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and sanitized standard deviation from one GPBoost model."""
    coords = np.asarray(coords, dtype=float)
    prediction = model.predict(
        gp_coords_pred=coords,
        X_pred=np.ones((coords.shape[0], 1), dtype=float),
        predict_var=True,
    )
    mean = np.asarray(prediction["mu"], dtype=float)
    variance = np.asarray(prediction["var"], dtype=float)
    return mean, np.sqrt(np.maximum(variance, _VAR_FLOOR))


def _cov_pars(model: Any) -> dict[str, float]:
    table = model.get_cov_pars(format_pandas=True)
    return {name: float(value) for name, value in table.iloc[0].items()}


def _cov_par_vector(model: Any) -> np.ndarray:
    """Fitted covariance parameters in GPBoost's own order (rho last)."""
    return np.asarray(
        model.get_cov_pars(format_pandas=True).iloc[0].to_numpy(), dtype=float
    )


def _fixed_effects(model: Any, fidelity: np.ndarray) -> np.ndarray:
    """Linear predictor of the fitted intercept, one value per training row.

    ``neg_log_likelihood`` does not profile out the regression coefficients,
    so the mean term has to be supplied explicitly. The design is a single
    intercept column, which ``fidelity_specific_mean=True`` expands into
    separate LF and HF coefficients.
    """
    coef = np.asarray(
        model.get_coef(format_pandas=True).iloc[0].to_numpy(), dtype=float
    )
    if coef.size == 1:
        return np.full(len(fidelity), coef[0], dtype=float)
    if coef.size == 2:
        return np.where(fidelity == 0.0, coef[0], coef[1])
    raise ValueError(
        "expected an intercept-only mean (1 or 2 coefficients); "
        f"found {coef.size}"
    )


def _attach_compat_methods(model: Any, *, n_points: int | None = None) -> Any:
    """Expose the small API older research scripts expect from GPBoost models."""
    if n_points is not None:
        model.n_points = int(n_points)
    if not hasattr(model, "cov_pars"):
        model.cov_pars = lambda: _cov_pars(model)
    return model


def _model_to_payload(model: Any) -> dict[str, Any]:
    return {
        "model_dict": model.model_to_dict(include_response_data=True),
        "n_points": getattr(model, "n_points", None),
    }


def _payload_to_model(payload: dict[str, Any]) -> Any:
    gpb = _require_gpboost()
    model = gpb.GPModel(model_dict=payload["model_dict"])
    return _attach_compat_methods(model, n_points=payload.get("n_points"))


@dataclass
class GPBoostModel1Layer:
    """One GPBoost AR(1) MF-GP per wavelength.

    ``global_rho_sweeps`` is None for the per-wavelength-rho fit (Model 1B)
    and holds the block-coordinate sweep history for the shared-rho fit
    (Model 1A), mirroring mf_gp.JointMFGPLayer.
    """

    wavelengths: np.ndarray
    scaler: StandardScaler
    subsample_indices: np.ndarray | None
    column_mu: np.ndarray | None
    column_sd: np.ndarray | None
    models: list[Any] = field(default_factory=list)
    fit_seconds: np.ndarray | None = None
    cov_function: str = GPBOOST_COV_FUNCTION
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE
    gp_approx: str = GPBOOST_GP_APPROX
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS
    global_rho_sweeps: list[dict] | None = None

    @property
    def rho(self) -> np.ndarray:
        return np.asarray([_cov_pars(model)["rho"] for model in self.models])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["models"] = [_model_to_payload(model) for model in self.models]
        state["_models_are_payloads"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_models_are_payloads", False):
            state["models"] = [_payload_to_model(payload) for payload in state["models"]]
        self.__dict__.update(state)


@dataclass
class _Model1Design:
    """LF-subsampled, fidelity-augmented design shared by Model 1A and 1B."""

    scaler: StandardScaler
    subsample_indices: np.ndarray | None
    lf_rows: np.ndarray
    coords: np.ndarray
    column_mu: np.ndarray | None
    column_sd: np.ndarray | None
    Y_lf_fit: np.ndarray
    Y_hf_fit: np.ndarray

    def targets(self, j: int) -> np.ndarray:
        return np.concatenate((self.Y_lf_fit[self.lf_rows, j], self.Y_hf_fit[:, j]))

    @property
    def fidelity(self) -> np.ndarray:
        return self.coords[:, -1]


def _model1_design(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None,
    standardize_per_wavelength: bool,
) -> _Model1Design:
    if Y_lf.shape[1] != len(wavelengths) or Y_hf.shape[1] != len(wavelengths):
        raise ValueError("Y_lf, Y_hf, and wavelengths must share columns")
    if standardize_per_wavelength:
        column_mu, column_sd = lf_column_moments(Y_lf)
        Y_lf_fit = (Y_lf - column_mu) / column_sd
        Y_hf_fit = (Y_hf - column_mu) / column_sd
    else:
        column_mu = column_sd = None
        Y_lf_fit = Y_lf
        Y_hf_fit = Y_hf

    if subsample_size is None:
        subsample_indices = None
        lf_rows = np.arange(X_lf.shape[0])
    else:
        subsample_indices = select_lf_subsample(
            X_lf.shape[0], subsample_size, seed=seed
        )
        lf_rows = subsample_indices

    scaler = StandardScaler().fit(X_lf)
    coords = np.vstack(
        (
            _append_fidelity(scaler.transform(X_lf[lf_rows]), 0),
            _append_fidelity(scaler.transform(X_hf), 1),
        )
    )
    return _Model1Design(
        scaler=scaler,
        subsample_indices=subsample_indices,
        lf_rows=lf_rows,
        coords=coords,
        column_mu=column_mu,
        column_sd=column_sd,
        Y_lf_fit=Y_lf_fit,
        Y_hf_fit=Y_hf_fit,
    )


def fit_model1_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    cov_function: str = GPBOOST_COV_FUNCTION,
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE,
    gp_approx: str = GPBOOST_GP_APPROX,
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS,
    maxit: int = 1000,
    trace: bool = False,
    num_parallel_threads: int | None = None,
    standardize_per_wavelength: bool = True,
    fidelity_specific_mean: bool = True,
    heuristic_init: bool = GPBOOST_HEURISTIC_INIT,
    progress_every: int | None = None,
) -> GPBoostModel1Layer:
    """Fit GPBoost's native AR(1) MF covariance independently per wavelength."""
    X_lf = np.asarray(X_lf, dtype=float)
    Y_lf = np.asarray(Y_lf, dtype=float)
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    design = _model1_design(
        X_lf,
        Y_lf,
        X_hf,
        Y_hf,
        wavelengths,
        seed=seed,
        subsample_size=subsample_size,
        standardize_per_wavelength=standardize_per_wavelength,
    )
    coords = design.coords

    models = []
    fit_seconds = []
    for j, _ in enumerate(wavelengths):
        y = design.targets(j)
        t0 = time.perf_counter()
        model = _fit_gpboost_ar1(
            coords,
            y,
            seed=seed,
            cov_function=cov_function,
            cov_fct_shape=cov_fct_shape,
            gp_approx=gp_approx,
            num_neighbors=num_neighbors,
            maxit=maxit,
            trace=trace,
            num_parallel_threads=num_parallel_threads,
            fidelity_specific_mean=fidelity_specific_mean,
            heuristic_init=heuristic_init,
        )
        fit_seconds.append(time.perf_counter() - t0)
        models.append(model)
        if progress_every and (j + 1) % progress_every == 0:
            print(
                f"  fit {j + 1}/{len(wavelengths)} GPBoost MF-GPs "
                f"({fit_seconds[-1]:.1f}s last)",
                flush=True,
            )

    return GPBoostModel1Layer(
        wavelengths=wavelengths,
        scaler=design.scaler,
        subsample_indices=design.subsample_indices,
        column_mu=design.column_mu,
        column_sd=design.column_sd,
        models=models,
        fit_seconds=np.asarray(fit_seconds),
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
    )


# ---------------------------------------------------------------------------
# Model 1A (GPBoost): per-wavelength AR(1) MF-GPs sharing ONE rho.
#
# Same block-coordinate ascent on the same joint criterion as the sklearn
# Model 1A (mf_gp.fit_joint_mf_gp_global_rho): wavelengths are conditionally
# independent given the hyperparameters, so the joint log likelihood is the
# SUM of the per-wavelength log marginal likelihoods and is maximized by
# alternating
#   step A: rho fixed -> GPBoost estimates every other covariance parameter
#           and the LF/HF means, independently per wavelength;
#   step B: everything else fixed -> 1-D search of the shared rho against the
#           summed log marginal likelihood.
# Neither step can decrease the joint likelihood.
# ---------------------------------------------------------------------------


def _rho_grid_loglik(
    models: list[Any],
    targets: list[np.ndarray],
    fixed_effects: list[np.ndarray],
    cov_pars: list[np.ndarray],
    rho_grid: np.ndarray,
) -> np.ndarray:
    """Summed log marginal likelihood at every grid rho, everything else fixed."""
    totals = np.zeros(len(rho_grid))
    for model, y, offset, cov in zip(models, targets, fixed_effects, cov_pars):
        trial = np.array(cov, dtype=float)
        for i, rho in enumerate(rho_grid):
            trial[-1] = rho
            nll = float(
                model.neg_log_likelihood(
                    cov_pars=trial, y=y, fixed_effects=offset
                )
            )
            totals[i] += -nll if np.isfinite(nll) else -np.inf
    return totals


def _optimize_shared_rho_gpboost(
    models: list[Any],
    targets: list[np.ndarray],
    fixed_effects: list[np.ndarray],
    cov_pars: list[np.ndarray],
    *,
    bounds: tuple[float, float] = GPBOOST_GLOBAL_RHO_BOUNDS,
    stages: int = GPBOOST_GLOBAL_RHO_GRID_STAGES,
) -> tuple[float, float]:
    """Maximize the summed LML over one shared rho by a staged grid zoom."""
    lo, hi = bounds
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    n_points = 41
    best_rho, best_loglik = center, -np.inf
    for _ in range(stages):
        grid = np.clip(
            np.linspace(center - half_width, center + half_width, n_points), lo, hi
        )
        totals = _rho_grid_loglik(models, targets, fixed_effects, cov_pars, grid)
        best = int(np.argmax(totals))
        best_rho, best_loglik = float(grid[best]), float(totals[best])
        half_width = float(grid[1] - grid[0])
        center, n_points = best_rho, 21
    return best_rho, best_loglik


def fit_model1a_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    cov_function: str = GPBOOST_COV_FUNCTION,
    cov_fct_shape: float = GPBOOST_COV_FCT_SHAPE,
    gp_approx: str = GPBOOST_GP_APPROX,
    num_neighbors: int | None = GPBOOST_NUM_NEIGHBORS,
    maxit: int = 1000,
    trace: bool = False,
    num_parallel_threads: int | None = None,
    standardize_per_wavelength: bool = True,
    fidelity_specific_mean: bool = True,
    heuristic_init: bool = GPBOOST_HEURISTIC_INIT,
    warm_start_layer: GPBoostModel1Layer | None = None,
    rho_init: float = GPBOOST_INIT_RHO,
    rho_bounds: tuple[float, float] = GPBOOST_GLOBAL_RHO_BOUNDS,
    max_sweeps: int = GPBOOST_GLOBAL_RHO_MAX_SWEEPS,
    tol: float = GPBOOST_GLOBAL_RHO_TOL,
    progress_every: int | None = None,
) -> GPBoostModel1Layer:
    """Model 1A, GPBoost variant: per-wavelength MF-GPs sharing ONE rho.

    Same design, standardization and covariance as ``fit_model1_gpboost``
    (Model 1B); the only difference is the rho criterion. The sweep starts
    from ``rho_init`` with every other parameter at its heuristic starting
    value, exactly as the sklearn Model 1A starts its sweeps. Passing a
    ``warm_start_layer`` (a Model 1B layer fitted on the identical design)
    instead initializes each wavelength from that fit and picks the starting
    rho by the same summed-likelihood criterion the sweeps use; it is an
    optional speedup and does not change the criterion being optimized.
    """
    X_lf = np.asarray(X_lf, dtype=float)
    Y_lf = np.asarray(Y_lf, dtype=float)
    X_hf = np.asarray(X_hf, dtype=float)
    Y_hf = np.asarray(Y_hf, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    design = _model1_design(
        X_lf,
        Y_lf,
        X_hf,
        Y_hf,
        wavelengths,
        seed=seed,
        subsample_size=subsample_size,
        standardize_per_wavelength=standardize_per_wavelength,
    )
    coords = design.coords
    fidelity = design.fidelity
    targets = [design.targets(j) for j in range(len(wavelengths))]

    if warm_start_layer is not None:
        if len(warm_start_layer.models) != len(wavelengths):
            raise ValueError(
                f"warm_start_layer has {len(warm_start_layer.models)} models "
                f"but {len(wavelengths)} wavelengths were requested"
            )
        warm_rows = (
            np.arange(X_lf.shape[0])
            if warm_start_layer.subsample_indices is None
            else warm_start_layer.subsample_indices
        )
        if not np.array_equal(warm_rows, design.lf_rows):
            raise ValueError(
                "warm_start_layer was fitted on a different LF subsample; "
                "use the same subsample_size and seed"
            )

    total_seconds = np.zeros(len(wavelengths))

    def fit_all_fixed_rho(
        rho: float, inits: list[np.ndarray]
    ) -> tuple[list[Any], list[np.ndarray]]:
        models = []
        for j in range(len(wavelengths)):
            init = np.array(inits[j], dtype=float)
            init[-1] = rho
            estimate_index = np.ones(len(init), dtype=np.int32)
            estimate_index[-1] = 0  # rho is swept separately in step B
            t0 = time.perf_counter()
            model = _fit_gpboost_ar1(
                coords,
                targets[j],
                seed=seed,
                cov_function=cov_function,
                cov_fct_shape=cov_fct_shape,
                gp_approx=gp_approx,
                num_neighbors=num_neighbors,
                maxit=maxit,
                trace=trace,
                num_parallel_threads=num_parallel_threads,
                fidelity_specific_mean=fidelity_specific_mean,
                init_cov_pars=init,
                estimate_cov_par_index=estimate_index,
                heuristic_init=heuristic_init,
            )
            total_seconds[j] += time.perf_counter() - t0
            models.append(model)
            if progress_every and (j + 1) % progress_every == 0:
                print(
                    f"  fit {j + 1}/{len(wavelengths)} fixed-rho GPBoost MF-GPs",
                    flush=True,
                )
        return models, [_cov_par_vector(model) for model in models]

    if warm_start_layer is None:
        # No fitted models yet: start every wavelength from the same
        # heuristic covariance parameters and let the first sweep do the work.
        n_cov_pars = 4 + 2 * (design.coords.shape[1] - 1)
        inits = [
            heuristic_init_cov_pars(coords, targets[j], n_cov_pars, seed=seed)
            if heuristic_init
            else np.concatenate(
                (np.ones(n_cov_pars - 1), [rho_init])
            )
            for j in range(len(wavelengths))
        ]
        rho = rho_init
    else:
        models = list(warm_start_layer.models)
        inits = [_cov_par_vector(model) for model in models]
        # Initialize rho by the same criterion the sweeps use -- the summed
        # LML over the warm fits -- not by averaging the per-wavelength rho_j,
        # which is dominated by wavelengths whose rho_j is poorly identified.
        rho, _ = _optimize_shared_rho_gpboost(
            models,
            targets,
            [_fixed_effects(model, fidelity) for model in models],
            inits,
            bounds=rho_bounds,
        )

    history: list[dict] = []
    converged = False
    for sweep in range(max_sweeps):
        models, inits = fit_all_fixed_rho(rho, inits)
        new_rho, summed_loglik = _optimize_shared_rho_gpboost(
            models,
            targets,
            [_fixed_effects(model, fidelity) for model in models],
            inits,
            bounds=rho_bounds,
        )
        delta = abs(new_rho - rho)
        history.append(
            {
                "sweep": sweep,
                "rho_before": rho,
                "rho_after": new_rho,
                "summed_log_marginal_likelihood": summed_loglik,
                "abs_delta_rho": delta,
            }
        )
        if progress_every:
            print(
                f"  sweep {sweep}: rho {rho:.5g} -> {new_rho:.5g}, "
                f"summed LML {summed_loglik:.2f}",
                flush=True,
            )
        rho = new_rho
        if delta < tol:
            converged = True
            break
    if not converged:
        print(
            f"  WARNING: shared rho not converged after {max_sweeps} sweeps "
            f"(last |delta rho| = {history[-1]['abs_delta_rho']:.2e})",
            flush=True,
        )

    # Final consistency refit so every wavelength carries the final rho.
    models, _ = fit_all_fixed_rho(rho, inits)
    rho_lo, rho_hi = rho_bounds
    if np.isclose(rho, rho_lo) or np.isclose(rho, rho_hi):
        warnings.warn(
            f"GPBoost Model 1A shared rho={rho:.4g} pins "
            f"rho_bounds={rho_bounds}; widen it",
            stacklevel=2,
        )

    return GPBoostModel1Layer(
        wavelengths=wavelengths,
        scaler=design.scaler,
        subsample_indices=design.subsample_indices,
        column_mu=design.column_mu,
        column_sd=design.column_sd,
        models=models,
        fit_seconds=total_seconds,
        cov_function=cov_function,
        cov_fct_shape=cov_fct_shape,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
        global_rho_sweeps=history,
    )


def predict_hf_model1_gpboost(
    layer: GPBoostModel1Layer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra from atmospheric inputs with all fitted wavelengths."""
    X_scaled = layer.scaler.transform(np.asarray(X_new, dtype=float))
    coords = _append_fidelity(X_scaled, 1)
    means, stds = zip(*(_predict_gpboost_ar1(model, coords) for model in layer.models))
    mean_matrix = np.column_stack(means)
    std_matrix = np.column_stack(stds)
    if layer.column_mu is not None:
        mean_matrix = mean_matrix * layer.column_sd + layer.column_mu
        std_matrix = std_matrix * layer.column_sd
    return mean_matrix, std_matrix


def hyperparameter_table_model1_gpboost(layer: GPBoostModel1Layer) -> pd.DataFrame:
    """One row per wavelength of GPBoost covariance parameters."""
    rows = []
    for j, (wavelength, model) in enumerate(zip(layer.wavelengths, layer.models)):
        cov = _cov_pars(model)
        row = {
            "wavelength": float(wavelength),
            "rho": cov["rho"],
            "error_var": cov.get("Error_var", np.nan),
            "low_signal_variance": cov.get("low_GP_var", np.nan),
            "delta_signal_variance": cov.get("discrepancy_GP_var", np.nan),
        }
        row.update({f"cov_{name}": value for name, value in cov.items()})
        if layer.fit_seconds is not None:
            row["fit_seconds"] = float(layer.fit_seconds[j])
        n_dims = int(model.dim_coords) - 1
        for i in range(1, n_dims + 1):
            row[f"low_range_{i - 1}"] = cov.get(f"low_GP_range_{i}", np.nan)
            row[f"delta_range_{i - 1}"] = cov.get(
                f"discrepancy_GP_range_{i}", np.nan
            )
        if "low_GP_range" in cov:
            row["low_range"] = cov["low_GP_range"]
        if "discrepancy_GP_range" in cov:
            row["delta_range"] = cov["discrepancy_GP_range"]
        rows.append(row)
    return pd.DataFrame(rows)


def cv_predict_model1_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    progress_every: int | None = None,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold GPBoost Model 1 predictions over paired HF samples."""

    def fit_fold(train_idx: np.ndarray) -> GPBoostModel1Layer:
        return fit_model1_gpboost(
            X_lf,
            Y_lf,
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            subsample_size=subsample_size,
            progress_every=progress_every,
            **fit_kwargs,
        )

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model1_gpboost(
            layer, X_hf[test_idx]
        ),
        n_samples=Y_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )


def cv_predict_model1a_gpboost(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    n_splits: int = CV_FULL_MODEL_SPLITS,
    progress: bool = False,
    progress_every: int | None = None,
    **fit_kwargs,
) -> CVPredictions:
    """Out-of-fold GPBoost Model 1A (shared rho) predictions over HF samples.

    Every fold reruns the whole block-coordinate sweep, warm start included,
    on the fold's training HF rows.
    """

    def fit_fold(train_idx: np.ndarray) -> GPBoostModel1Layer:
        t0 = time.perf_counter()
        layer = fit_model1a_gpboost(
            X_lf,
            Y_lf,
            X_hf[train_idx],
            Y_hf[train_idx],
            wavelengths,
            seed=seed,
            subsample_size=subsample_size,
            progress_every=progress_every,
            **fit_kwargs,
        )
        if progress:
            print(
                f"  GPBoost Model 1A fold fitted in "
                f"{time.perf_counter() - t0:.1f}s "
                f"(shared rho={layer.rho[0]:.4f}, "
                f"{len(layer.global_rho_sweeps)} sweeps)",
                flush=True,
            )
        return layer

    return cv_predict(
        fit_fn=fit_fold,
        predict_fn=lambda layer, test_idx: predict_hf_model1_gpboost(
            layer, X_hf[test_idx]
        ),
        n_samples=Y_hf.shape[0],
        n_wavelengths=len(np.asarray(wavelengths)),
        n_splits=n_splits,
        seed=seed,
    )
