"""GPBoost side of the joint AR(1) multi-fidelity kernel comparison.

Nothing here is a custom covariance: the joint AR(1) multi-fidelity model

    f_H(x) = rho * f_L(x) + delta(x)

is GPBoost's own built-in ``ar1_mf_matern``, with the fidelity indicator in the
last column of ``gp_coords`` and covariance parameters ordered
``[Error_var, low_GP_var, low_GP_range, discrepancy_GP_var,
discrepancy_GP_range, rho]``.

The only reason this module exists at all is to run that built-in from the same
starting values as the scikit-learn arm and hand back the result in a shared
record, so the two can be compared.

Exact vs Vecchia
----------------
``fit_gpboost_ar1`` takes ``gp_approx``: ``"none"`` for an exact GP, or
``"vecchia"`` with ``NUM_NEIGHBORS`` neighbours as in ``src/model_1.py``.
Running both separates two things that are otherwise confounded - whether
GPBoost's optimiser matches scikit-learn's (exact vs exact, same function), and
what the approximation costs (vecchia vs exact, same optimiser).

Whatever ``gp_approx`` was used, the returned ``neg_log_likelihood`` is the
*exact* likelihood at the fitted parameters, so every arm is scored on one
common ruler; ``native_neg_log_likelihood`` is what GPBoost itself reported.
"""

import time

import gpboost as gpb
import numpy as np

from kernel_comparison.sklearn_ar1 import (
    MATERN_NU,
    AR1Fit,
    AR1Params,
    ar1_neg_log_likelihood,
    count_params_at_bounds,
)


COV_FUNCTION = "ar1_mf_matern"  # GPBoost built-in; cov_fct_shape below is its nu
NUM_NEIGHBORS = 40              # only used when gp_approx="vecchia"
GP_THREADS = 1                  # comparable timings with the single-threaded sklearn arm

# GPBoost's default convergence tolerance (1e-6) is looser than the default of
# scipy's L-BFGS-B, which scikit-learn uses; leaving it there makes GPBoost look
# systematically (but spuriously) worse by ~1e-3 nats.  Tightening it to 1e-8
# makes the optimisers stop on the same terms.
DELTA_REL_CONV = 1e-8


def fit_gpboost_ar1(
    X,
    fidelity,
    y,
    inits,
    gp_approx="none",
    label=None,
    num_neighbors=NUM_NEIGHBORS,
    max_iter=1000,
) -> AR1Fit:
    """Fit GPBoost's built-in ``ar1_mf_matern`` from each of ``inits``.

    ``inits`` is the same list of :class:`AR1Params` handed to the scikit-learn
    arm, so every optimiser starts from exactly the same points; the run
    reaching the lowest marginal likelihood is returned.
    """
    X = np.asarray(X, dtype=float)
    fidelity = np.asarray(fidelity, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if label is None:
        label = "gpboost_vecchia" if gp_approx.startswith("vecchia") else "gpboost_exact"

    best = None
    started = time.perf_counter()
    for init in inits:
        gp_model = _gp_model(X, fidelity, gp_approx, num_neighbors)
        gp_model.fit(
            y=y,
            params={
                "trace": False,
                "init_cov_pars": init.to_gpboost(),
                "optimizer_cov": "lbfgs",
                "maxit": max_iter,
                "delta_rel_conv": DELTA_REL_CONV,
            },
        )
        params = AR1Params.from_gpboost(
            gp_model.get_cov_pars(std_err=False, format_pandas=False)
        )
        nll = ar1_neg_log_likelihood(params, X, fidelity, y)
        if best is None or nll < best[0]:
            best = (
                nll,
                params,
                float(gp_model.get_current_neg_log_likelihood()),
                int(gp_model._get_num_optim_iter()),
            )
    seconds = time.perf_counter() - started

    nll, params, native_nll, n_iterations = best
    return AR1Fit(
        framework=label,
        params=params,
        neg_log_likelihood=nll,
        seconds=seconds,
        n_inits=len(inits),
        n_iterations=n_iterations,
        native_neg_log_likelihood=native_nll,
        extra={"n_params_at_bound": count_params_at_bounds(params)},
    )


def gpboost_neg_log_likelihood(
    params: AR1Params, X, fidelity, y, gp_approx="none", num_neighbors=NUM_NEIGHBORS
):
    """GPBoost's own objective at ``params``, without fitting.

    ``simulation.py`` uses this twice: to assert that GPBoost's exact
    likelihood and the reference implementation are the same function (which is
    what licenses reading a lower NLL as a better optimum), and to measure how
    far the Vecchia surface sits from it.
    """
    gp_model = _gp_model(
        np.asarray(X, dtype=float),
        np.asarray(fidelity, dtype=float).ravel(),
        gp_approx,
        num_neighbors,
    )
    return float(
        gp_model.neg_log_likelihood(
            cov_pars=params.to_gpboost(), y=np.asarray(y, dtype=float).ravel()
        )
    )


def _gp_model(X, fidelity, gp_approx, num_neighbors):
    """The built-in AR(1) multi-fidelity GPModel; fidelity goes in the last column.

    No ``X`` (covariate) matrix is ever passed to ``fit``, so the model is
    zero-mean.
    """
    if not np.all(np.isin(fidelity, (0.0, 1.0))):
        raise ValueError("fidelity must contain only 0 (low) and 1 (high).")

    kwargs = dict(
        gp_coords=np.hstack([X, fidelity[:, None]]),
        cov_function=COV_FUNCTION,
        cov_fct_shape=MATERN_NU,
        likelihood="gaussian",
        gp_approx=gp_approx,
        num_parallel_threads=GP_THREADS,
    )
    if gp_approx.startswith("vecchia"):
        kwargs["num_neighbors"] = num_neighbors
    return gpb.GPModel(**kwargs)
