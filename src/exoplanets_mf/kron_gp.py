"""Exact GP inference on a complete Cartesian (theta, lambda) grid.

When every atmospheric input theta_i is evaluated at every wavelength
lambda_j and the kernel over the augmented input separates as
k_theta(theta, theta') * k_lambda(lambda, lambda'), the covariance of the
row-major flattened n x m output grid Y is the Kronecker product

    Cov(vec Y) = sigma_f^2 (K_theta (x) K_lambda) + sigma_n^2 I,

so exact inference on ALL n*m points costs O(n^3 + m^3), not O(n^3 m^3):
with eigendecompositions K_theta = Q_t L_t Q_t^T and
K_lambda = Q_l L_l Q_l^T,

    (sigma_f^2 K_theta (x) K_lambda + sigma_n^2 I)^{-1}
        = (Q_t (x) Q_l) (sigma_f^2 L_t (x) L_l + sigma_n^2 I)^{-1}
          (Q_t (x) Q_l)^T,

and every solve reduces to two small dense products via the row-major vec
identity (A (x) B) vec(M) = vec(A M B^T). K_theta is SE-ARD; K_lambda is
pluggable (any callable returning a PSD m x m matrix -- Kronecker structure
does not require stationarity, so a nonstationary NIRCam/MIRI two-block
kernel can be swapped in later).

The module is stage-agnostic: a plain zero-mean GP fits stage L of the
Le Gratiet recursion, and passing ``Y_pair`` fits the stage-delta residual
D = Y - rho * Y_pair with the scalar rho profiled out in closed form (GLS:
rho* = <Y~, P~/D> / <P~, P~/D> in the eigenbasis). rho is recomputed inside
every objective evaluation, so by the envelope theorem the gradient of the
profiled log marginal likelihood with respect to the kernel hyperparameters
is the partial gradient at rho*. rho may be negative and is never
log-parametrized.

Hyperparameters (one signal variance per stage -- per-factor variances are
unidentifiable in a product kernel) are optimized by L-BFGS-B in log space
with analytic gradients; all bounds are shared with mf_gp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from exoplanets_mf.mf_gp import (
    LENGTH_SCALE_BOUNDS,
    LENGTH_SCALE_INIT,
    NOISE_LEVEL_BOUNDS,
    NOISE_LEVEL_INIT,
    SIGNAL_VARIANCE_BOUNDS,
)

KRON_N_RESTARTS = 10
_BOUND_PIN_ATOL = 1e-6  # log-space distance below which an optimum counts as pinned


# ---------------------------------------------------------------------------
# 1. Kernel factors
# ---------------------------------------------------------------------------

def _scaled_sqdist(U: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
    """Pairwise squared Euclidean distances of pre-scaled inputs.

    Dot-product form (no (n, n, p) temporary) so the theta factor stays
    O(n^2) in memory at n ~ 10k.
    """
    if V is None:
        V = U
    uu = np.einsum("ij,ij->i", U, U)
    vv = np.einsum("ij,ij->i", V, V)
    sq = uu[:, None] + vv[None, :] - 2.0 * (U @ V.T)
    np.maximum(sq, 0.0, out=sq)
    return sq


def se_ard(X: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
    """Unit-variance SE-ARD Gram matrix (n, n)."""
    return np.exp(-0.5 * _scaled_sqdist(np.asarray(X, dtype=float) / length_scales))


def se_ard_cross(
    X_new: np.ndarray, X: np.ndarray, length_scales: np.ndarray
) -> np.ndarray:
    """Unit-variance SE-ARD cross matrix (n_new, n)."""
    return np.exp(
        -0.5
        * _scaled_sqdist(
            np.asarray(X_new, dtype=float) / length_scales,
            np.asarray(X, dtype=float) / length_scales,
        )
    )


class SEKernel1D:
    """Unit-variance squared-exponential kernel on the (standardized) lambda axis.

    The lambda factor of the Kronecker product is pluggable: any object with
    this interface (log-parametrized hyperparameters, ``value`` returning a
    PSD m x m matrix, ``gradients`` returning dK/dlog-param) can replace it,
    including nonstationary kernels -- e.g. a future two-block NIRCam/MIRI
    kernel split at the 5 micron break.
    """

    param_names: tuple[str, ...] = ("lambda_length_scale",)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def initial_log_params(self) -> np.ndarray:
        return np.log(np.array([LENGTH_SCALE_INIT]))

    def log_bounds(self) -> np.ndarray:
        return np.log(np.array([LENGTH_SCALE_BOUNDS]))

    def value(self, t: np.ndarray, log_params: np.ndarray) -> np.ndarray:
        ls = np.exp(log_params[0])
        return se_ard(np.asarray(t, dtype=float).reshape(-1, 1), np.array([ls]))

    def cross(
        self, t_new: np.ndarray, t: np.ndarray, log_params: np.ndarray
    ) -> np.ndarray:
        ls = np.exp(log_params[0])
        return se_ard_cross(
            np.asarray(t_new, dtype=float).reshape(-1, 1),
            np.asarray(t, dtype=float).reshape(-1, 1),
            np.array([ls]),
        )

    def gradients(self, t: np.ndarray, log_params: np.ndarray) -> list[np.ndarray]:
        ls = np.exp(log_params[0])
        u = np.asarray(t, dtype=float) / ls
        S = _scaled_sqdist(u.reshape(-1, 1))
        return [self.value(t, log_params) * S]


# ---------------------------------------------------------------------------
# 2. Log marginal likelihood and analytic gradient in the eigenbasis
# ---------------------------------------------------------------------------

def _unpack_log_eta(
    log_eta: np.ndarray, n_theta_dims: int, lambda_kernel: SEKernel1D
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """(signal variance, theta length scales, lambda log-params, noise variance)."""
    log_eta = np.asarray(log_eta, dtype=float)
    expected = 2 + n_theta_dims + lambda_kernel.n_params
    if log_eta.shape != (expected,):
        raise ValueError(f"expected {expected} log-hyperparameters, found {log_eta.shape}")
    signal_variance = float(np.exp(log_eta[0]))
    length_scales = np.exp(log_eta[1 : 1 + n_theta_dims])
    lambda_log_params = log_eta[1 + n_theta_dims : 1 + n_theta_dims + lambda_kernel.n_params]
    noise_variance = float(np.exp(log_eta[-1]))
    return signal_variance, length_scales, lambda_log_params, noise_variance


def _param_names(n_theta_dims: int, lambda_kernel: SEKernel1D) -> list[str]:
    return (
        ["signal_variance"]
        + [f"length_scale_{d}" for d in range(n_theta_dims)]
        + list(lambda_kernel.param_names)
        + ["noise_variance"]
    )


def _log_bounds(n_theta_dims: int, lambda_kernel: SEKernel1D) -> np.ndarray:
    return np.vstack(
        (
            np.log(np.array([SIGNAL_VARIANCE_BOUNDS])),
            np.tile(np.log(np.array([LENGTH_SCALE_BOUNDS])), (n_theta_dims, 1)),
            lambda_kernel.log_bounds(),
            np.log(np.array([NOISE_LEVEL_BOUNDS])),
        )
    )


def kron_lml_and_grad(
    log_eta: np.ndarray,
    X: np.ndarray,
    t: np.ndarray,
    Y: np.ndarray,
    lambda_kernel: SEKernel1D,
    *,
    Y_pair: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float | None]:
    """Exact LML, its gradient in log-hyperparameter space, and profiled rho.

    With ``Y_pair`` the targets are the residuals Y - rho* Y_pair at the
    closed-form GLS optimum rho* of the current hyperparameters; since
    dLML/drho = 0 at rho*, the returned gradient is exact for the profiled
    objective (envelope theorem). Eigenvalues are clipped at zero: at
    n ~ 10k an SE Gram matrix is numerically rank-deficient and eigh can
    return tiny negative values that a large signal variance would push
    into negative log arguments.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    p = X.shape[1]
    signal_variance, length_scales, lambda_log_params, noise_variance = _unpack_log_eta(
        log_eta, p, lambda_kernel
    )

    K_theta = se_ard(X, length_scales)
    K_lambda = lambda_kernel.value(t, lambda_log_params)
    eig_theta, Q_theta = eigh(K_theta)
    eig_lambda, Q_lambda = eigh(K_lambda)
    np.maximum(eig_theta, 0.0, out=eig_theta)
    np.maximum(eig_lambda, 0.0, out=eig_lambda)

    D = signal_variance * np.outer(eig_theta, eig_lambda) + noise_variance

    Y_tilde = Q_theta.T @ Y @ Q_lambda
    rho: float | None = None
    if Y_pair is not None:
        P_tilde = Q_theta.T @ np.asarray(Y_pair, dtype=float) @ Q_lambda
        denominator = float((P_tilde * P_tilde / D).sum())
        if denominator <= 0.0:
            raise ValueError("degenerate GLS denominator while profiling rho")
        rho = float((Y_tilde * P_tilde / D).sum() / denominator)
        Y_tilde = Y_tilde - rho * P_tilde

    alpha_tilde = Y_tilde / D
    lml = float(
        -0.5 * (Y_tilde * alpha_tilde).sum()
        - 0.5 * np.log(D).sum()
        - 0.5 * n * m * np.log(2.0 * np.pi)
    )

    # d(LML)/d(eta) = 0.5 * (alpha^T dK alpha - tr(K^{-1} dK)); for co-diagonal
    # perturbations both terms live on the n x m eigenvalue grid.
    E = alpha_tilde * alpha_tilde - 1.0 / D
    grad = np.empty_like(np.asarray(log_eta, dtype=float))
    grad[0] = 0.5 * float((E * (signal_variance * np.outer(eig_theta, eig_lambda))).sum())
    grad[-1] = 0.5 * noise_variance * float(E.sum())

    # Shared pieces for the off-diagonal (kernel-matrix) perturbations.
    alpha_mat = Q_theta @ alpha_tilde @ Q_lambda.T
    P_alpha = K_theta @ alpha_mat            # (n, m)
    V_alpha = alpha_mat @ K_lambda           # (n, m)
    r_theta = (1.0 / D) @ eig_lambda         # r_i = sum_j eig_lambda_j / D_ij
    s_lambda = (1.0 / D).T @ eig_theta       # s_j = sum_i eig_theta_i / D_ij

    # Theta-side SE-ARD length scales: dK_theta/dlog l_d = K_theta o S_d with
    # S_d = u^2 1^T + 1 u^2^T - 2 u u^T (u = X[:, d] / l_d), i.e.
    # diag(u^2) K + K diag(u^2) - 2 diag(u) K diag(u). Only the diagonal of
    # Q^T (K o S_d) Q enters the trace term; the h-term (one n x n dgemm per
    # dimension) is mandatory -- dropping it fails the finite-difference check.
    for d in range(p):
        u = X[:, d] / length_scales[d]
        u_sq = u * u
        c = np.einsum("a,ai,ai->i", u_sq, Q_theta, Q_theta)
        G = Q_theta.T @ (u[:, None] * Q_theta)
        h = np.einsum("ik,k,ik->i", G, eig_theta, G)
        trace_diag = 2.0 * (eig_theta * c - h)
        trace_term = signal_variance * float((trace_diag * r_theta).sum())
        dK_alpha = (
            u_sq[:, None] * P_alpha
            + K_theta @ (u_sq[:, None] * alpha_mat)
            - 2.0 * u[:, None] * (K_theta @ (u[:, None] * alpha_mat))
        )
        quad_term = signal_variance * float((dK_alpha * V_alpha).sum())
        grad[1 + d] = 0.5 * (quad_term - trace_term)

    for k, dK_lambda in enumerate(lambda_kernel.gradients(t, lambda_log_params)):
        b_diag = np.einsum("jl,jk,lk->k", dK_lambda, Q_lambda, Q_lambda)
        trace_term = signal_variance * float((b_diag * s_lambda).sum())
        quad_term = signal_variance * float((P_alpha * (alpha_mat @ dK_lambda)).sum())
        grad[1 + p + k] = 0.5 * (quad_term - trace_term)

    return lml, grad, rho


# ---------------------------------------------------------------------------
# 3. Fitting with multiple restarts
# ---------------------------------------------------------------------------

@dataclass
class KronGPFit:
    """One fitted Kronecker-grid GP stage.

    The eigendecomposition cache (Q_theta, ...) is dropped on pickling and
    rebuilt lazily from the stored training data on first prediction, so a
    joblib artifact stays ~ (n x m) instead of ~ (n x n).
    """

    X: np.ndarray                       # (n, p) standardized theta grid
    t: np.ndarray                       # (m,) standardized lambda grid
    Y: np.ndarray                       # (n, m) standardized targets
    Y_pair: np.ndarray | None           # (n, m) paired LF targets (stage delta)
    signal_variance: float
    length_scales: np.ndarray           # (p,)
    lambda_kernel: SEKernel1D
    lambda_log_params: np.ndarray
    noise_variance: float
    rho: float | None                   # profiled GLS rho (None for stage L)
    log_marginal_likelihood: float
    pinned_bounds: dict[str, str]       # param name -> "lower" / "upper"
    restart_records: list[dict]         # per-restart (restart, lml, rho, hypers)
    fit_seconds: float
    eigh_seconds: float                 # last eigh(K_theta) wall time
    n_iterations: int
    _cache: dict | None = field(default=None, repr=False, compare=False)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    @property
    def log_eta(self) -> np.ndarray:
        """The fitted hyperparameters as one log-space vector (fit_kron_gp
        layout), e.g. to warm-start another fit."""
        return np.concatenate(
            (
                [np.log(self.signal_variance)],
                np.log(self.length_scales),
                self.lambda_log_params,
                [np.log(self.noise_variance)],
            )
        )

    def _posterior_cache(self) -> dict:
        if self._cache is None:
            K_theta = se_ard(self.X, self.length_scales)
            K_lambda = self.lambda_kernel.value(self.t, self.lambda_log_params)
            eig_theta, Q_theta = eigh(K_theta)
            eig_lambda, Q_lambda = eigh(K_lambda)
            np.maximum(eig_theta, 0.0, out=eig_theta)
            np.maximum(eig_lambda, 0.0, out=eig_lambda)
            D = self.signal_variance * np.outer(eig_theta, eig_lambda) + self.noise_variance
            Y_effective = self.Y
            if self.Y_pair is not None:
                Y_effective = self.Y - self.rho * self.Y_pair
            alpha_tilde = (Q_theta.T @ Y_effective @ Q_lambda) / D
            self._cache = {
                "Q_theta": Q_theta,
                "eig_lambda": eig_lambda,
                "Q_lambda": Q_lambda,
                "D": D,
                "K_lambda": K_lambda,
                "alpha_mat": Q_theta @ alpha_tilde @ Q_lambda.T,
            }
        return self._cache


def _detect_pinned_bounds(
    log_eta: np.ndarray, bounds: np.ndarray, names: list[str]
) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for name, value, (lo, hi) in zip(names, log_eta, bounds):
        if np.isclose(value, lo, atol=_BOUND_PIN_ATOL, rtol=0.0):
            pinned[name] = "lower"
        elif np.isclose(value, hi, atol=_BOUND_PIN_ATOL, rtol=0.0):
            pinned[name] = "upper"
    return pinned


def kron_gp_at(
    X: np.ndarray,
    t: np.ndarray,
    Y: np.ndarray,
    log_eta: np.ndarray,
    *,
    lambda_kernel: SEKernel1D | None = None,
    Y_pair: np.ndarray | None = None,
) -> KronGPFit:
    """A KronGPFit at FIXED log-hyperparameters (no optimization).

    rho is still profiled in closed form when ``Y_pair`` is given. Used by
    the exactness tests and whenever previously fitted hyperparameters are
    reused on new data.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float)
    if lambda_kernel is None:
        lambda_kernel = SEKernel1D()
    p = X.shape[1]
    lml, _, rho = kron_lml_and_grad(log_eta, X, t, Y, lambda_kernel, Y_pair=Y_pair)
    signal_variance, length_scales, lambda_log_params, noise_variance = _unpack_log_eta(
        log_eta, p, lambda_kernel
    )
    return KronGPFit(
        X=X,
        t=t,
        Y=Y,
        Y_pair=None if Y_pair is None else np.asarray(Y_pair, dtype=float),
        signal_variance=signal_variance,
        length_scales=length_scales,
        lambda_kernel=lambda_kernel,
        lambda_log_params=lambda_log_params,
        noise_variance=noise_variance,
        rho=rho,
        log_marginal_likelihood=lml,
        pinned_bounds=_detect_pinned_bounds(
            np.asarray(log_eta, dtype=float),
            _log_bounds(p, lambda_kernel),
            _param_names(p, lambda_kernel),
        ),
        restart_records=[],
        fit_seconds=0.0,
        eigh_seconds=0.0,
        n_iterations=0,
    )


def fit_kron_gp(
    X: np.ndarray,
    t: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int,
    lambda_kernel: SEKernel1D | None = None,
    n_restarts: int = KRON_N_RESTARTS,
    Y_pair: np.ndarray | None = None,
    initial_log_eta: np.ndarray | None = None,
    maxiter: int = 200,
) -> KronGPFit:
    """Maximize the exact grid LML from ``n_restarts`` starts.

    Start 0 is the deterministic default init (all length scales 1, signal
    variance 1, noise 1e-4 -- same inits as mf_gp) unless ``initial_log_eta``
    warm-starts it (used by the screen-then-polish stage-L strategy);
    remaining starts are log-uniform within the bounds.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float)
    if Y.shape != (X.shape[0], t.shape[0]):
        raise ValueError(
            f"Y must be (n_theta, n_lambda) = {(X.shape[0], t.shape[0])}; found {Y.shape}"
        )
    if Y_pair is not None:
        Y_pair = np.asarray(Y_pair, dtype=float)
        if Y_pair.shape != Y.shape:
            raise ValueError(f"Y_pair must match Y shape {Y.shape}; found {Y_pair.shape}")
    if lambda_kernel is None:
        lambda_kernel = SEKernel1D()
    p = X.shape[1]
    bounds = _log_bounds(p, lambda_kernel)
    names = _param_names(p, lambda_kernel)

    default_start = np.concatenate(
        (
            [0.0],                                       # signal variance 1
            np.full(p, np.log(LENGTH_SCALE_INIT)),
            lambda_kernel.initial_log_params(),
            [np.log(NOISE_LEVEL_INIT)],
        )
    )
    rng = np.random.default_rng(seed)
    starts = [
        np.asarray(initial_log_eta, dtype=float)
        if initial_log_eta is not None
        else default_start
    ]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(rng.uniform(bounds[:, 0], bounds[:, 1]))

    eigh_seconds = 0.0

    def negative_objective(log_eta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eigh_seconds
        t0 = time.perf_counter()
        lml, grad, _ = kron_lml_and_grad(log_eta, X, t, Y, lambda_kernel, Y_pair=Y_pair)
        eigh_seconds = time.perf_counter() - t0  # dominated by eigh(K_theta)
        return -lml, -grad

    t_start = time.perf_counter()
    best_result = None
    restart_records: list[dict] = []
    for restart, x0 in enumerate(starts):
        result = minimize(
            negative_objective,
            np.clip(x0, bounds[:, 0], bounds[:, 1]),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        lml, _, rho = kron_lml_and_grad(
            result.x, X, t, Y, lambda_kernel, Y_pair=Y_pair
        )
        record = {"restart": restart, "lml": lml, "rho": rho, "success": bool(result.success)}
        record.update(dict(zip(names, np.exp(result.x))))
        record["nit"] = int(result.nit)
        restart_records.append(record)
        if np.isfinite(lml) and (best_result is None or lml > best_result[0]):
            best_result = (lml, result.x.copy(), rho, int(result.nit))
    fit_seconds = time.perf_counter() - t_start

    if best_result is None:
        raise RuntimeError("all Kronecker GP restarts failed to produce a finite LML")
    best_lml, best_log_eta, best_rho, best_nit = best_result

    pinned = _detect_pinned_bounds(best_log_eta, bounds, names)

    signal_variance, length_scales, lambda_log_params, noise_variance = _unpack_log_eta(
        best_log_eta, p, lambda_kernel
    )
    return KronGPFit(
        X=X,
        t=t,
        Y=Y,
        Y_pair=Y_pair,
        signal_variance=signal_variance,
        length_scales=length_scales,
        lambda_kernel=lambda_kernel,
        lambda_log_params=lambda_log_params,
        noise_variance=noise_variance,
        rho=best_rho,
        log_marginal_likelihood=best_lml,
        pinned_bounds=pinned,
        restart_records=restart_records,
        fit_seconds=fit_seconds,
        eigh_seconds=eigh_seconds,
        n_iterations=best_nit,
    )


# ---------------------------------------------------------------------------
# 4. Posterior prediction
# ---------------------------------------------------------------------------

def posterior_mean_var(
    fit: KronGPFit, X_new: np.ndarray, *, include_noise: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and per-lambda variance at new theta rows, all lambdas.

    Returns (n_new, m) arrays in the (standardized) training output space.
    ``include_noise`` adds the fitted iid noise variance to the marginal --
    parity with Model 2, whose predictive stds include fitted noise.
    """
    cache = fit._posterior_cache()
    X_new = np.asarray(X_new, dtype=float)
    k_star = se_ard_cross(X_new, fit.X, fit.length_scales)

    means = fit.signal_variance * (k_star @ cache["alpha_mat"] @ cache["K_lambda"])

    A_star = k_star @ cache["Q_theta"]                       # (n_new, n)
    weights = cache["eig_lambda"][None, :] ** 2 / cache["D"]  # (n, m)
    T = (A_star * A_star) @ weights                           # (n_new, m)
    reduction = T @ (cache["Q_lambda"] ** 2).T                # (n_new, m)
    variances = fit.signal_variance - fit.signal_variance**2 * reduction
    if include_noise:
        variances = variances + fit.noise_variance
    np.maximum(variances, 0.0, out=variances)
    return means, variances


def posterior_lambda_cov(
    fit: KronGPFit, x_new: np.ndarray, *, include_noise: bool = False
) -> np.ndarray:
    """Full posterior covariance across the lambda grid at ONE new theta."""
    cache = fit._posterior_cache()
    x_new = np.asarray(x_new, dtype=float).reshape(1, -1)
    a = (se_ard_cross(x_new, fit.X, fit.length_scales) @ cache["Q_theta"])[0]
    t_weights = cache["eig_lambda"] ** 2 * ((a * a) @ (1.0 / cache["D"]))
    cov = fit.signal_variance * cache["K_lambda"] - fit.signal_variance**2 * (
        (cache["Q_lambda"] * t_weights) @ cache["Q_lambda"].T
    )
    if include_noise:
        cov = cov + fit.noise_variance * np.eye(len(fit.t))
    return cov
