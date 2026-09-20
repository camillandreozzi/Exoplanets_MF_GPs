"""scikit-learn counterpart of GPBoost's built-in ``ar1_mf_matern``.

This module provides

* :class:`AR1MultiFidelityKernel`, a scikit-learn kernel implementing the same
  two-level autoregressive multi-fidelity covariance

      f_H(x) = rho * f_L(x) + delta(x),      f_L _||_ delta

  so that, writing ``a_i = rho`` for a high-fidelity row and ``a_i = 1`` for a
  low-fidelity row,

      K_ij = a_i a_j k_L(x_i, x_j) + f_i f_j k_delta(x_i, x_j),

  with ``k_L`` and ``k_delta`` both Matern-nu.  All of ``k_L``, ``k_delta`` and
  ``rho`` are optimised jointly by the ordinary scikit-learn
  marginal-likelihood optimiser.

* a reference implementation of the same covariance (:func:`ar1_covariance`,
  :func:`ar1_neg_log_likelihood`, :func:`ar1_predict`) used to score every arm
  of the comparison on one common objective and one common predictor, and

* :class:`AR1Params` / :class:`AR1Fit`, the parameter and result records shared
  with ``kernel_comparison/gpboost_ar1.py``.
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Hyperparameter,
    Kernel,
    Matern,
    WhiteKernel,
)


MATERN_NU = 1.5  # GPBoost's cov_fct_shape

LENGTH_SCALE_INIT = 1.0
LENGTH_SCALE_BOUNDS = (1e-2, 1e3)
SIGNAL_VARIANCE_BOUNDS = (1e-6, 1e6)
NOISE_VARIANCE_BOUNDS = (1e-10, 1e2)
RHO_BOUNDS = (-5.0, 5.0)

# Added to the diagonal purely for numerical stability; the actual nugget is an
# explicit WhiteKernel so that it maps onto GPBoost's "Error_var".
JITTER = 1e-10

# Comparison tolerances, in nats of negative log marginal likelihood.
# Differences below TIE_TOLERANCE are optimiser round-off, not a real
# difference in the optimum found; MATERIAL_TOLERANCE is the threshold above
# which a difference is large enough to change any conclusion drawn from the
# fit.
TIE_TOLERANCE = 1e-3
MATERIAL_TOLERANCE = 0.5


# ==========================================================================
# The kernel
# ==========================================================================
class AR1MultiFidelityKernel(Kernel):
    """Two-level AR(1) multi-fidelity kernel with a jointly optimised ``rho``.

    The last column of ``X`` is the fidelity indicator (0 = low, 1 = high);
    every preceding column is an input coordinate handed to the sub-kernels.

    ``rho`` is unconstrained: unlike every other kernel hyperparameter it is
    optimised on the natural scale, not the log scale, so that it is allowed to
    be negative (as in GPBoost).
    """

    def __init__(
        self,
        low_kernel,
        discrepancy_kernel,
        rho=1.0,
        rho_bounds=RHO_BOUNDS,
    ):
        self.low_kernel = low_kernel
        self.discrepancy_kernel = discrepancy_kernel
        self.rho = rho
        self.rho_bounds = rho_bounds

    # -- scikit-learn plumbing ---------------------------------------------
    def get_params(self, deep=True):
        params = {
            "low_kernel": self.low_kernel,
            "discrepancy_kernel": self.discrepancy_kernel,
            "rho": self.rho,
            "rho_bounds": self.rho_bounds,
        }
        if deep:
            for prefix, kernel in (
                ("low_kernel", self.low_kernel),
                ("discrepancy_kernel", self.discrepancy_kernel),
            ):
                for name, value in kernel.get_params().items():
                    params[f"{prefix}__{name}"] = value
        return params

    @property
    def hyperparameter_rho(self):
        return Hyperparameter("rho", "numeric", self.rho_bounds)

    @property
    def hyperparameters(self):
        """Sub-kernel hyperparameters (prefixed) followed by ``rho``."""
        hyperparameters = []
        for prefix, kernel in (
            ("low_kernel", self.low_kernel),
            ("discrepancy_kernel", self.discrepancy_kernel),
        ):
            for hyperparameter in kernel.hyperparameters:
                hyperparameters.append(
                    Hyperparameter(
                        f"{prefix}__{hyperparameter.name}",
                        hyperparameter.value_type,
                        hyperparameter.bounds,
                        hyperparameter.n_elements,
                    )
                )
        hyperparameters.append(self.hyperparameter_rho)
        return hyperparameters

    @property
    def theta(self):
        """Sub-kernel thetas (log scale) with ``rho`` appended (natural scale)."""
        return np.concatenate(
            [
                self.low_kernel.theta,
                self.discrepancy_kernel.theta,
                np.array([self.rho], dtype=float),
            ]
        )

    @theta.setter
    def theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        n_low = self.low_kernel.n_dims
        n_discrepancy = self.discrepancy_kernel.n_dims
        expected = n_low + n_discrepancy + 1
        if theta.shape[0] != expected:
            raise ValueError(
                f"theta has {theta.shape[0]} entries, expected {expected}"
            )
        self.low_kernel.theta = theta[:n_low]
        self.discrepancy_kernel.theta = theta[n_low : n_low + n_discrepancy]
        self.rho = float(theta[-1])

    @property
    def bounds(self):
        return np.vstack(
            [
                self.low_kernel.bounds,
                self.discrepancy_kernel.bounds,
                np.atleast_2d(np.asarray(self.rho_bounds, dtype=float)),
            ]
        )

    def is_stationary(self):
        # Not stationary: the covariance depends on the fidelity level itself.
        return False

    def __repr__(self):
        return (
            f"AR1MF(low={self.low_kernel!r}, "
            f"discrepancy={self.discrepancy_kernel!r}, rho={self.rho:.3g})"
        )

    # -- the covariance itself ---------------------------------------------
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        x_train, fidelity_x = X[:, :-1], X[:, -1]
        scale_x = np.where(fidelity_x == 1.0, self.rho, 1.0)

        if Y is None:
            if eval_gradient:
                k_low, grad_low = self.low_kernel(x_train, eval_gradient=True)
                k_discrepancy, grad_discrepancy = self.discrepancy_kernel(
                    x_train, eval_gradient=True
                )
            else:
                k_low = self.low_kernel(x_train)
                k_discrepancy = self.discrepancy_kernel(x_train)
            scale_y, fidelity_y = scale_x, fidelity_x
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = np.atleast_2d(Y)
            y_train, fidelity_y = Y[:, :-1], Y[:, -1]
            scale_y = np.where(fidelity_y == 1.0, self.rho, 1.0)
            k_low = self.low_kernel(x_train, y_train)
            k_discrepancy = self.discrepancy_kernel(x_train, y_train)

        low_mask = np.outer(scale_x, scale_y)
        discrepancy_mask = np.outer(fidelity_x, fidelity_y)
        K = low_mask * k_low + discrepancy_mask * k_discrepancy

        if not eval_gradient:
            return K

        # d/d(theta) of each block; rho enters only through the low-fidelity term.
        gradient_low = low_mask[:, :, np.newaxis] * grad_low
        gradient_discrepancy = discrepancy_mask[:, :, np.newaxis] * grad_discrepancy
        d_low_mask_d_rho = np.outer(fidelity_x, scale_y) + np.outer(scale_x, fidelity_y)
        gradient_rho = (d_low_mask_d_rho * k_low)[:, :, np.newaxis]

        gradient = np.dstack([gradient_low, gradient_discrepancy, gradient_rho])
        return K, gradient

    def diag(self, X):
        X = np.atleast_2d(X)
        x_train, fidelity = X[:, :-1], X[:, -1]
        scale = np.where(fidelity == 1.0, self.rho, 1.0)
        return scale**2 * self.low_kernel.diag(x_train) + fidelity * (
            self.discrepancy_kernel.diag(x_train)
        )


def make_joint_mf_kernel(
    low_var=1.0,
    low_length_scale=LENGTH_SCALE_INIT,
    discrepancy_var=1.0,
    discrepancy_length_scale=LENGTH_SCALE_INIT,
    rho=1.0,
) -> AR1MultiFidelityKernel:
    """The jointly optimised Matern AR(1) kernel, matching ``ar1_mf_matern``."""
    return AR1MultiFidelityKernel(
        low_kernel=ConstantKernel(low_var, SIGNAL_VARIANCE_BOUNDS)
        * Matern(low_length_scale, LENGTH_SCALE_BOUNDS, nu=MATERN_NU),
        discrepancy_kernel=ConstantKernel(discrepancy_var, SIGNAL_VARIANCE_BOUNDS)
        * Matern(discrepancy_length_scale, LENGTH_SCALE_BOUNDS, nu=MATERN_NU),
        rho=rho,
    )


# ==========================================================================
# Parameter / result records shared with the GPBoost side
# ==========================================================================
PARAMETER_NAMES = (
    "noise_var",
    "low_var",
    "low_length_scale",
    "discrepancy_var",
    "discrepancy_length_scale",
    "rho",
)


@dataclass
class AR1Params:
    """AR(1) multi-fidelity parameters, in GPBoost's ``ar1_mf_matern`` order."""

    noise_var: float
    low_var: float
    low_length_scale: float
    discrepancy_var: float
    discrepancy_length_scale: float
    rho: float

    def to_vector(self) -> np.ndarray:
        return np.array(
            [getattr(self, name) for name in PARAMETER_NAMES], dtype=float
        )

    def to_dict(self):
        return dict(zip(PARAMETER_NAMES, self.to_vector().tolist()))

    def to_gpboost(self) -> np.ndarray:
        """GPBoost's ``cov_pars``: same order, same scale (see module docstring)."""
        return self.to_vector()

    @classmethod
    def from_gpboost(cls, cov_pars):
        values = np.asarray(cov_pars, dtype=float).ravel()
        if values.shape[0] != len(PARAMETER_NAMES):
            raise ValueError(
                f"Expected {len(PARAMETER_NAMES)} covariance parameters, "
                f"got {values.shape[0]}"
            )
        return cls(*values)


def classify_winner(values_by_framework, tolerance=TIE_TOLERANCE):
    """Framework with the lowest objective, or ``"tie"``.

    A tie is declared when the best two arms are within ``tolerance`` nats of
    each other, i.e. when the difference is too small to mean anything.
    """
    finite = {
        name: value
        for name, value in values_by_framework.items()
        if np.isfinite(value)
    }
    if not finite:
        return "tie"
    ordered = sorted(finite.items(), key=lambda item: item[1])
    if len(ordered) == 1:
        return ordered[0][0]
    if ordered[1][1] - ordered[0][1] < tolerance:
        return "tie"
    return ordered[0][0]


def count_params_at_bounds(params: AR1Params, tolerance=1e-3) -> int:
    """How many hyperparameters ended up pinned against their bounds.

    A pinned length scale means the data do not constrain that block, so it is
    reported as a diagnostic rather than raised as a convergence warning.
    """
    checks = (
        (params.noise_var, NOISE_VARIANCE_BOUNDS),
        (params.low_var, SIGNAL_VARIANCE_BOUNDS),
        (params.discrepancy_var, SIGNAL_VARIANCE_BOUNDS),
        (params.low_length_scale, LENGTH_SCALE_BOUNDS),
        (params.discrepancy_length_scale, LENGTH_SCALE_BOUNDS),
    )
    at_bound = 0
    for value, (lower, upper) in checks:
        if value <= 0:
            continue
        log_value = np.log(value)
        if (
            abs(log_value - np.log(lower)) < tolerance
            or abs(log_value - np.log(upper)) < tolerance
        ):
            at_bound += 1
    if min(abs(params.rho - bound) for bound in RHO_BOUNDS) < tolerance:
        at_bound += 1
    return at_bound


@dataclass
class AR1Fit:
    """Outcome of one hyperparameter optimisation."""

    framework: str
    params: AR1Params
    neg_log_likelihood: float
    seconds: float
    n_inits: int
    n_iterations: int = -1
    native_neg_log_likelihood: float = float("nan")
    extra: dict = field(default_factory=dict)


# ==========================================================================
# Reference implementation (the common ruler)
# ==========================================================================
def ar1_covariance(params: AR1Params, X1, f1, X2=None, f2=None) -> np.ndarray:
    """AR(1) multi-fidelity Matern covariance, noise excluded."""
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    f1 = np.asarray(f1, dtype=float).ravel()
    symmetric = X2 is None
    if symmetric:
        X2, f2 = X1, f1
    else:
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        f2 = np.asarray(f2, dtype=float).ravel()

    def block(length_scale, variance):
        kernel = Matern(length_scale, nu=MATERN_NU)
        return variance * (kernel(X1) if symmetric else kernel(X1, X2))

    scale1 = np.where(f1 == 1.0, params.rho, 1.0)
    scale2 = np.where(f2 == 1.0, params.rho, 1.0)
    K = np.outer(scale1, scale2) * block(params.low_length_scale, params.low_var)
    K += np.outer(f1, f2) * block(
        params.discrepancy_length_scale, params.discrepancy_var
    )
    return K


def ar1_neg_log_likelihood(params: AR1Params, X, f, y) -> float:
    """Exact zero-mean Gaussian negative log marginal likelihood.

    This is the ruler every arm is scored on.  It agrees with
    ``GPModel.neg_log_likelihood`` under ``gp_approx="none"`` to machine
    precision (asserted by ``simulation.py``); an arm running under Vecchia
    optimises an approximation to it, and the difference is reported rather
    than hidden.
    """
    y = np.asarray(y, dtype=float).ravel()
    K = ar1_covariance(params, X, f)
    K[np.diag_indices_from(K)] += params.noise_var + JITTER
    try:
        factor = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return float("inf")
    alpha = cho_solve((factor, True), y)
    return float(
        0.5 * y @ alpha
        + np.log(np.diag(factor)).sum()
        + 0.5 * len(y) * np.log(2.0 * np.pi)
    )


def ar1_predict(params: AR1Params, X_train, f_train, y_train, X_test, f_test):
    """Exact GP posterior for the given parameters.

    Every arm is scored through this one predictor, so any difference in
    predictive accuracy comes from the fitted hyperparameters and not from
    differences between the two prediction implementations.

    Returns
    -------
    mean, latent_var, observation_var
    """
    K = ar1_covariance(params, X_train, f_train)
    K[np.diag_indices_from(K)] += params.noise_var + JITTER
    factor = cho_factor(K, lower=True)
    alpha = cho_solve(factor, np.asarray(y_train, dtype=float).ravel())

    K_star = ar1_covariance(params, X_test, f_test, X_train, f_train)
    mean = K_star @ alpha

    v = solve_triangular(factor[0], K_star.T, lower=True)
    prior_var = np.diag(ar1_covariance(params, X_test, f_test))
    latent_var = np.maximum(prior_var - (v**2).sum(axis=0), 1e-12)
    return mean, latent_var, latent_var + params.noise_var


def prediction_metrics(y_true, mean, observation_var, y_latent=None):
    """RMSE / NRMSE / MAE / negative log predictive density on held-out data."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    residuals = mean - y_true
    spread = float(np.std(y_true, ddof=0))
    metrics = {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "nrmse": float(np.sqrt(np.mean(residuals**2)) / spread) if spread > 0 else np.nan,
        "mae": float(np.mean(np.abs(residuals))),
        "nlpd": float(
            np.mean(
                0.5 * np.log(2.0 * np.pi * observation_var)
                + residuals**2 / (2.0 * observation_var)
            )
        ),
        "coverage_95": float(
            np.mean(np.abs(residuals) <= 1.959964 * np.sqrt(observation_var))
        ),
    }
    if y_latent is not None:
        latent_residuals = mean - np.asarray(y_latent, dtype=float).ravel()
        metrics["rmse_latent"] = float(np.sqrt(np.mean(latent_residuals**2)))
    return metrics


# ==========================================================================
# Fitting
# ==========================================================================
def fit_sklearn_ar1(X, fidelity, y, inits, label="sklearn", max_iter=1000) -> AR1Fit:
    """Fit the joint AR(1) Matern kernel with scikit-learn from each of ``inits``.

    ``inits`` is the same list of :class:`AR1Params` handed to the GPBoost arms,
    so every optimiser starts from exactly the same points; the run reaching the
    lowest marginal likelihood is returned.
    """
    X = np.asarray(X, dtype=float)
    fidelity = np.asarray(fidelity, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    X_augmented = np.hstack([X, fidelity[:, None]])

    best = None
    started = time.perf_counter()
    for init in inits:
        regressor = GaussianProcessRegressor(
            kernel=_kernel_at(init),
            alpha=JITTER,
            normalize_y=False,
            n_restarts_optimizer=0,
            optimizer="fmin_l_bfgs_b",
        )
        # A parameter pinned at a bound is expected here (an unconstrained
        # block drives its length scale to the ceiling) and is reported via
        # count_params_at_bounds instead of as a warning per fit.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            regressor.fit(X_augmented, y)
        params = _params_from_kernel(regressor.kernel_)
        nll = ar1_neg_log_likelihood(params, X, fidelity, y)
        if best is None or nll < best[0]:
            best = (nll, params, -float(regressor.log_marginal_likelihood_value_))
    seconds = time.perf_counter() - started

    nll, params, native_nll = best
    return AR1Fit(
        framework=label,
        params=params,
        neg_log_likelihood=nll,
        seconds=seconds,
        n_inits=len(inits),
        native_neg_log_likelihood=native_nll,
        extra={"n_params_at_bound": count_params_at_bounds(params)},
    )


def _kernel_at(params: AR1Params):
    """Full scikit-learn kernel (AR(1) + nugget) seeded at ``params``."""
    return make_joint_mf_kernel(
        low_var=params.low_var,
        low_length_scale=params.low_length_scale,
        discrepancy_var=params.discrepancy_var,
        discrepancy_length_scale=params.discrepancy_length_scale,
        rho=params.rho,
    ) + WhiteKernel(params.noise_var, NOISE_VARIANCE_BOUNDS)


def _params_from_kernel(kernel) -> AR1Params:
    """Read parameters back out of a fitted ``AR1MF + WhiteKernel``."""
    ar1, white = kernel.k1, kernel.k2
    return AR1Params(
        noise_var=white.noise_level,
        low_var=ar1.low_kernel.k1.constant_value,
        low_length_scale=float(ar1.low_kernel.k2.length_scale),
        discrepancy_var=ar1.discrepancy_kernel.k1.constant_value,
        discrepancy_length_scale=float(ar1.discrepancy_kernel.k2.length_scale),
        rho=ar1.rho,
    )


def default_init(y=None) -> AR1Params:
    """Neutral starting point: unit length scales, variance split between blocks."""
    scale = 1.0 if y is None else max(float(np.var(np.asarray(y, dtype=float))), 1e-8)
    return AR1Params(
        noise_var=0.1 * scale,
        low_var=0.9 * scale,
        low_length_scale=LENGTH_SCALE_INIT,
        discrepancy_var=0.1 * scale,
        discrepancy_length_scale=LENGTH_SCALE_INIT,
        rho=1.0,
    )


def make_inits(reference: AR1Params, n_inits: int, seed: int, spread=3.0):
    """Shared starting values: ``reference`` plus log-uniform jitter.

    Every arm is handed this same list, so the multi-start budget is matched.
    """
    inits = [reference]
    rng = np.random.default_rng(seed)
    jitter = lambda: float(np.exp(rng.uniform(-np.log(spread), np.log(spread))))
    for _ in range(max(n_inits - 1, 0)):
        inits.append(
            AR1Params(
                noise_var=reference.noise_var * jitter(),
                low_var=reference.low_var * jitter(),
                low_length_scale=reference.low_length_scale * jitter(),
                discrepancy_var=reference.discrepancy_var * jitter(),
                discrepancy_length_scale=reference.discrepancy_length_scale * jitter(),
                rho=float(np.clip(reference.rho + rng.normal(0.0, 0.3), *RHO_BOUNDS)),
            )
        )
    return inits
