"""Joint AR(1) multi-fidelity Gaussian processes for exoplanet spectra.

For each wavelength bin j the model is

    f_H,j(theta) = rho_j * f_L,j(theta) + delta_j(theta),

where f_L,j and delta_j are independent SE-ARD Gaussian processes. LF and HF
observations are stacked into one training vector and all hyperparameters,
including rho, are optimized in one marginal likelihood. The corresponding
two-fidelity covariance is

    K_LL = k_L
    K_LH = rho * k_L
    K_HH = rho**2 * k_L + k_delta.

Two fitting criteria for rho are provided; everything else is identical:

- Model 1B, fit_joint_mf_gp: one free rho_j per wavelength, optimized inside
  that wavelength's own marginal likelihood.
- Model 1A, fit_joint_mf_gp_global_rho: ONE rho shared by every wavelength,
  optimized against the sum of the per-wavelength log marginal likelihoods
  (the joint likelihood -- wavelengths are conditionally independent given
  the hyperparameters).

The implementation uses an explicit scikit-learn kernel so it remains
compatible with the project's pinned Python/NumPy stack. Prediction at high
fidelity needs only the atmospheric inputs; no paired LF output is required.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_solve, cholesky
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Hyperparameter,
    Kernel,
)
from sklearn.preprocessing import StandardScaler

N_INPUT_DIMS = 9  # Kzz, Rp, Tint, C, N, O, S, logg, f  #TODO: keep in sync with XHF.shape[1]


# ---------------------------------------------------------------------------
# 1. Joint AR(1) covariance
# ---------------------------------------------------------------------------

SIGNAL_VARIANCE_BOUNDS = (1e-3, 1e5)      # widened from 1e3: fitted sigma^2 were pinning at the upper bound
LENGTH_SCALE_BOUNDS = (1e-3, 1e5)         # widened from 1e3; standardized-input units. Near-inert inputs (6 of 9, per prior sensitivity analysis) drift toward the upper bound, so keep it high enough that the informative dims separate cleanly
LENGTH_SCALE_INIT = 1.0                   # initial ARD length-scale (identical across the 9 dims before optimization)
RHO_BOUNDS = (1e-3, 1e4)                  # widened from 1e3 to match the variance/length-scale headroom
NOISE_LEVEL_BOUNDS = (1e-8, 1e1)
NOISE_LEVEL_INIT = 1e-4

# Log-space tolerance for flagging a fitted hyperparameter that has come to
# rest against one of the bounds above.
BOUND_PIN_ATOL = 1e-6


class AR1MultiFidelityKernel(Kernel):
    """Two-level AR(1) block kernel.

    The final input column is the fidelity indicator: 0 for LF and 1 for HF.
    ``low_kernel`` and ``discrepancy_kernel`` act only on the preceding
    atmospheric-input columns. Separate diagonal noise levels are learned for
    LF and HF observations.
    """

    def __init__(
        self,
        low_kernel: Kernel,
        discrepancy_kernel: Kernel,
        rho: float = 1.0,
        rho_bounds: tuple[float, float] | str = RHO_BOUNDS,
        low_noise: float = NOISE_LEVEL_INIT,
        low_noise_bounds: tuple[float, float] | str = NOISE_LEVEL_BOUNDS,
        high_noise: float = NOISE_LEVEL_INIT,
        high_noise_bounds: tuple[float, float] | str = NOISE_LEVEL_BOUNDS,
    ):
        self.low_kernel = low_kernel
        self.discrepancy_kernel = discrepancy_kernel
        self.rho = rho
        self.rho_bounds = rho_bounds
        self.low_noise = low_noise
        self.low_noise_bounds = low_noise_bounds
        self.high_noise = high_noise
        self.high_noise_bounds = high_noise_bounds

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "low_kernel": self.low_kernel,
            "discrepancy_kernel": self.discrepancy_kernel,
            "rho": self.rho,
            "rho_bounds": self.rho_bounds,
            "low_noise": self.low_noise,
            "low_noise_bounds": self.low_noise_bounds,
            "high_noise": self.high_noise,
            "high_noise_bounds": self.high_noise_bounds,
        }
        if deep:
            params.update(
                (f"low_kernel__{key}", value)
                for key, value in self.low_kernel.get_params().items()
            )
            params.update(
                (f"discrepancy_kernel__{key}", value)
                for key, value in self.discrepancy_kernel.get_params().items()
            )
        return params

    @property
    def hyperparameter_rho(self) -> Hyperparameter:
        return Hyperparameter("rho", "numeric", self.rho_bounds)

    @property
    def hyperparameter_low_noise(self) -> Hyperparameter:
        return Hyperparameter("low_noise", "numeric", self.low_noise_bounds)

    @property
    def hyperparameter_high_noise(self) -> Hyperparameter:
        return Hyperparameter("high_noise", "numeric", self.high_noise_bounds)

    @property
    def hyperparameters(self) -> list[Hyperparameter]:
        nested = [
            Hyperparameter(
                f"low_kernel__{h.name}", h.value_type, h.bounds, h.n_elements
            )
            for h in self.low_kernel.hyperparameters
        ]
        nested.extend(
            Hyperparameter(
                f"discrepancy_kernel__{h.name}",
                h.value_type,
                h.bounds,
                h.n_elements,
            )
            for h in self.discrepancy_kernel.hyperparameters
        )
        return nested + [
            self.hyperparameter_rho,
            self.hyperparameter_low_noise,
            self.hyperparameter_high_noise,
        ]

    @property
    def theta(self) -> np.ndarray:
        values = [self.low_kernel.theta, self.discrepancy_kernel.theta]
        for value, hyperparameter in (
            (self.rho, self.hyperparameter_rho),
            (self.low_noise, self.hyperparameter_low_noise),
            (self.high_noise, self.hyperparameter_high_noise),
        ):
            if not hyperparameter.fixed:
                values.append(np.atleast_1d(np.log(value)))
        return np.concatenate(values)

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        low_dims = self.low_kernel.n_dims
        discrepancy_dims = self.discrepancy_kernel.n_dims
        self.low_kernel.theta = theta[:low_dims]
        start = low_dims
        self.discrepancy_kernel.theta = theta[start : start + discrepancy_dims]
        start += discrepancy_dims
        for name, hyperparameter in (
            ("rho", self.hyperparameter_rho),
            ("low_noise", self.hyperparameter_low_noise),
            ("high_noise", self.hyperparameter_high_noise),
        ):
            if not hyperparameter.fixed:
                setattr(self, name, float(np.exp(theta[start])))
                start += 1
        if start != len(theta):
            raise ValueError("theta has the wrong number of entries")

    @property
    def bounds(self) -> np.ndarray:
        bounds = [self.low_kernel.bounds, self.discrepancy_kernel.bounds]
        bounds.extend(
            np.log(np.asarray(h.bounds, dtype=float).reshape(-1, 2))
            for h in (
                self.hyperparameter_rho,
                self.hyperparameter_low_noise,
                self.hyperparameter_high_noise,
            )
            if not h.fixed
        )
        return np.vstack([value for value in bounds if value.size])

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
    ):
        if eval_gradient and Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None")

        X_features = X[:, :-1]
        X_fidelity = X[:, -1]
        same_inputs = Y is None
        if same_inputs:
            Y_features = X_features
            Y_fidelity = X_fidelity
        else:
            Y_features = Y[:, :-1]
            Y_fidelity = Y[:, -1]

        X_scale = np.where(X_fidelity == 0.0, 1.0, self.rho)
        Y_scale = np.where(Y_fidelity == 0.0, 1.0, self.rho)
        low_scale = np.multiply.outer(X_scale, Y_scale)
        high_scale = np.multiply.outer(X_fidelity, Y_fidelity)

        if eval_gradient:
            low, low_gradient = self.low_kernel(
                X_features, eval_gradient=True
            )
            discrepancy, discrepancy_gradient = self.discrepancy_kernel(
                X_features, eval_gradient=True
            )
        else:
            low = self.low_kernel(X_features, Y_features)
            discrepancy = self.discrepancy_kernel(X_features, Y_features)

        low_component = low * low_scale
        covariance = low_component + discrepancy * high_scale

        if same_inputs:
            diagonal = np.arange(X.shape[0])
            covariance[diagonal, diagonal] += np.where(
                X_fidelity == 0.0, self.low_noise, self.high_noise
            )

        if not eval_gradient:
            return covariance

        gradients = [
            low_gradient * low_scale[:, :, None],
            discrepancy_gradient * high_scale[:, :, None],
        ]
        if not self.hyperparameter_rho.fixed:
            rho_gradient = low_component * (
                X_fidelity[:, None] + X_fidelity[None, :]
            )
            gradients.append(rho_gradient[:, :, None])
        if not self.hyperparameter_low_noise.fixed:
            low_noise_gradient = np.zeros_like(covariance)
            low_indices = np.flatnonzero(X_fidelity == 0.0)
            low_noise_gradient[low_indices, low_indices] = self.low_noise
            gradients.append(low_noise_gradient[:, :, None])
        if not self.hyperparameter_high_noise.fixed:
            high_noise_gradient = np.zeros_like(covariance)
            high_indices = np.flatnonzero(X_fidelity == 1.0)
            high_noise_gradient[high_indices, high_indices] = self.high_noise
            gradients.append(high_noise_gradient[:, :, None])
        return covariance, np.concatenate(gradients, axis=2)

    def diag(self, X: np.ndarray) -> np.ndarray:
        fidelity = X[:, -1]
        scale = np.where(fidelity == 0.0, 1.0, self.rho)
        return (
            self.low_kernel.diag(X[:, :-1]) * scale**2
            + self.discrepancy_kernel.diag(X[:, :-1]) * fidelity
            + np.where(fidelity == 0.0, self.low_noise, self.high_noise)
        )

    def is_stationary(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"AR1MultiFidelityKernel(low={self.low_kernel}, "
            f"delta={self.discrepancy_kernel}, rho={self.rho:.3g}, "
            f"noise=({self.low_noise:.3g}, {self.high_noise:.3g}))"
        )


def make_joint_mf_kernel(n_dims: int = N_INPUT_DIMS) -> AR1MultiFidelityKernel:
    """Create the jointly optimized SE-ARD AR(1) kernel."""
    length_scales = np.full(n_dims, LENGTH_SCALE_INIT)
    return AR1MultiFidelityKernel(
        low_kernel=ConstantKernel(1.0, SIGNAL_VARIANCE_BOUNDS)
        * RBF(length_scales.copy(), LENGTH_SCALE_BOUNDS),
        discrepancy_kernel=ConstantKernel(1.0, SIGNAL_VARIANCE_BOUNDS)
        * RBF(length_scales.copy(), LENGTH_SCALE_BOUNDS),
    )


def _free_hyperparameter_names(kernel: Kernel) -> list[str]:
    """Names aligned to ``kernel.theta`` (non-fixed params, ARD dims expanded)."""
    names: list[str] = []
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue
        if hp.n_elements > 1:
            names.extend(f"{hp.name}_{i}" for i in range(hp.n_elements))
        else:
            names.append(hp.name)
    return names


def detect_pinned_hyperparameters(gp: GaussianProcessRegressor) -> dict[str, str]:
    """Fitted hyperparameters resting against a bound -> "lower"/"upper".

    Compares the fitted log-space ``theta`` to the log-space ``bounds`` of the
    fitted kernel (both exclude 'fixed' params). Infinite bounds are skipped.
    """
    kernel = gp.kernel_
    pinned: dict[str, str] = {}
    names = _free_hyperparameter_names(kernel)
    for name, value, (lo, hi) in zip(names, kernel.theta, kernel.bounds):
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if np.isclose(value, lo, atol=BOUND_PIN_ATOL, rtol=0.0):
            pinned[name] = "lower"
        elif np.isclose(value, hi, atol=BOUND_PIN_ATOL, rtol=0.0):
            pinned[name] = "upper"
    return pinned


def warn_on_pinned_bounds(
    models: list[GaussianProcessRegressor], *, label: str
) -> dict[tuple[str, str], int]:
    """Emit one aggregated warning if any fitted hyperparameter pins a bound.

    Returns a ``{(name, side): count}`` map (how many wavelengths pinned each
    parameter at each bound). No warning is raised when nothing pins, so a
    clean production rerun stays silent -- that silence is the signal the
    widened bounds sufficed.
    """
    counts: dict[tuple[str, str], int] = {}
    for gp in models:
        for name, side in detect_pinned_hyperparameters(gp).items():
            counts[(name, side)] = counts.get((name, side), 0) + 1
    if counts:
        detail = ", ".join(
            f"{name}@{side} x{count}"
            for (name, side), count in sorted(counts.items())
        )
        warnings.warn(
            f"{label}: {len(models)} wavelengths fitted; hyperparameters pinned "
            f"at a bound (widen the corresponding *_BOUNDS): {detail}",
            stacklevel=2,
        )
    return counts


# The default is the EXACT joint fit on all LF + HF rows (no approximation).
# Cost per wavelength: each optimizer iteration factorizes the full n x n
# covariance (cubic time) and materializes the kernel-gradient stack of
# n^2 * n_theta doubles -- see exact_fit_memory_bytes(). fit_joint_mf_gp
# fails fast with the numbers when that stack cannot fit in RAM, instead of
# thrashing for hours; passing an integer subsample_size remains available as
# an explicit, opt-in reduction (it is no longer the default anywhere).
MF_GP_N_RESTARTS_OPTIMIZER = 1
MF_GP_ALPHA = 1e-10
MF_GP_NORMALIZE_Y = True
MF_GP_MEMORY_FRACTION = 0.8  # refuse exact fits above this share of RAM


@dataclass
class JointMFGPLayer:
    """One jointly fitted AR(1) multi-fidelity GP per wavelength.

    ``global_rho_sweeps`` is None for the per-wavelength-rho fit (Model 1B)
    and holds the block-coordinate sweep history for the shared-rho fit
    (Model 1A); read it with getattr when the layer may predate the field.
    """

    wavelengths: np.ndarray
    scaler: StandardScaler
    subsample_indices: np.ndarray | None  # None => exact fit on all LF rows
    models: list[GaussianProcessRegressor] = field(default_factory=list)
    fit_seconds: np.ndarray | None = None  # per-wavelength fit wall time
    global_rho_sweeps: list[dict] | None = None

    @property
    def rho(self) -> np.ndarray:
        return np.array([model.kernel_.rho for model in self.models])


def select_lf_subsample(
    n_available: int, subsample_size: int, *, seed: int
) -> np.ndarray:
    """Draw one fixed LF row subsample shared by all wavelengths."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_available, size=subsample_size, replace=False)


def exact_fit_memory_bytes(n_points: int, n_theta: int) -> int:
    """Estimated peak bytes of one exact hyperparameter-fit iteration.

    Dominated by the kernel-gradient stack (n^2 * n_theta doubles), which
    AR1MultiFidelityKernel.__call__ holds twice at the np.concatenate step
    (per-parameter pieces + the concatenated copy), plus a few n^2 sklearn
    temporaries (K, its Cholesky factor, ...).
    """
    return (2 * n_theta + 4) * n_points**2 * 8


def available_memory_bytes() -> int | None:
    """Physical RAM in bytes, or None when the platform does not report it."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


_QUERY_RAM = object()  # sentinel: assert_exact_fit_fits_in_memory queries the OS


def assert_exact_fit_fits_in_memory(
    n_points: int,
    n_theta: int,
    *,
    available_bytes: int | None | object = _QUERY_RAM,
    memory_fraction: float = MF_GP_MEMORY_FRACTION,
) -> None:
    """Fail fast (MemoryError) when the exact fit cannot fit in RAM.

    Refusing up front beats silently swapping for hours. The error message
    carries the estimate so the decision how to proceed (bigger machine,
    explicit subsample_size, fewer points) stays with the caller. Passing
    ``available_bytes=None`` means "RAM unknown" and skips the check.
    """
    if available_bytes is _QUERY_RAM:
        available_bytes = available_memory_bytes()
    if available_bytes is None:
        return  # cannot check -- proceed and let the OS decide
    required = exact_fit_memory_bytes(n_points, n_theta)
    budget = memory_fraction * available_bytes
    if required > budget:
        raise MemoryError(
            f"exact joint fit on {n_points} points with {n_theta} "
            f"hyperparameters needs ~{required / 1e9:.1f} GB per optimizer "
            f"iteration but only {available_bytes / 1e9:.1f} GB RAM is "
            f"available ({memory_fraction:.0%} budget = {budget / 1e9:.1f} GB). "
            f"Options: run on a machine with more memory, or pass an explicit "
            f"subsample_size to fit_joint_mf_gp."
        )


def _append_fidelity(X: np.ndarray, fidelity: int) -> np.ndarray:
    augmented = np.empty((X.shape[0], X.shape[1] + 1), dtype=float)
    augmented[:, :-1] = X
    augmented[:, -1] = fidelity
    return augmented


def _joint_design(
    X_lf: np.ndarray,
    X_hf: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None,
    n_theta: int,
) -> tuple[StandardScaler, np.ndarray | None, np.ndarray, np.ndarray]:
    """Shared LF-subsample + fidelity-augmented design used by both fits."""
    scaler = StandardScaler().fit(X_lf)
    if subsample_size is None:
        subsample_indices = None
        lf_rows = np.arange(X_lf.shape[0])
        assert_exact_fit_fits_in_memory(
            X_lf.shape[0] + X_hf.shape[0], n_theta
        )
    else:
        subsample_indices = select_lf_subsample(
            X_lf.shape[0], subsample_size, seed=seed
        )
        lf_rows = subsample_indices
    X_joint = np.vstack(
        (
            _append_fidelity(scaler.transform(X_lf[lf_rows]), 0),
            _append_fidelity(scaler.transform(X_hf), 1),
        )
    )
    return scaler, subsample_indices, lf_rows, X_joint


def fit_joint_mf_gp(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    n_restarts_optimizer: int = MF_GP_N_RESTARTS_OPTIMIZER,
    alpha: float = MF_GP_ALPHA,
    normalize_y: bool = MF_GP_NORMALIZE_Y,
    progress_every: int | None = None,
) -> JointMFGPLayer:
    """Model 1B: jointly fit LF, HF, rho_j, and discrepancy per wavelength.

    ``subsample_size=None`` (default) is the exact fit: every LF row enters
    the joint design. An integer subsamples the LF rows once (shared across
    wavelengths) as an explicit, opt-in approximation. The exact path checks
    its memory footprint first and raises MemoryError with the numbers when
    the machine cannot hold it (see assert_exact_fit_fits_in_memory).
    """
    scaler, subsample_indices, lf_rows, X_joint = _joint_design(
        X_lf,
        X_hf,
        seed=seed,
        subsample_size=subsample_size,
        n_theta=len(make_joint_mf_kernel(X_lf.shape[1]).theta),
    )

    models = []
    fit_seconds = []
    for j in range(len(wavelengths)):
        y_joint = np.concatenate(
            (Y_lf[lf_rows, j], Y_hf[:, j])
        )
        model = GaussianProcessRegressor(
            kernel=make_joint_mf_kernel(X_lf.shape[1]),
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=seed,
        )
        t0 = time.perf_counter()
        model.fit(X_joint, y_joint)
        fit_seconds.append(time.perf_counter() - t0)
        models.append(model)
        if progress_every and (j + 1) % progress_every == 0:
            print(
                f"  fit {j + 1}/{len(wavelengths)} joint MF-GPs "
                f"({fit_seconds[-1]:.1f}s last)",
                flush=True,
            )

    warn_on_pinned_bounds(models, label="Model 1B (per-wavelength-rho MF-GP)")
    return JointMFGPLayer(
        wavelengths=np.asarray(wavelengths),
        scaler=scaler,
        subsample_indices=subsample_indices,
        models=models,
        fit_seconds=np.asarray(fit_seconds),
    )


# ---------------------------------------------------------------------------
# 2. Model 1A: one shared rho across all wavelengths, one joint likelihood.
#
#    The joint log likelihood is the SUM of the per-wavelength log marginal
#    likelihoods (wavelengths are conditionally independent given the
#    hyperparameters), so it is maximized by block-coordinate ascent:
#
#      step A: rho fixed -> fit every wavelength's remaining hyperparameters
#              (195 independent sklearn fits, exactly as in Model 1B);
#      step B: all kernels fixed -> 1-D bounded optimization of the shared
#              rho against the summed log marginal likelihood.
#
#    Neither step can decrease the joint likelihood, so the sweep sequence
#    converges to a stationary point of the same single joint criterion.
# ---------------------------------------------------------------------------

GLOBAL_RHO_INIT = 1.0
GLOBAL_RHO_MAX_SWEEPS = 25
GLOBAL_RHO_LOG_TOL = 1e-3  # |log rho change| below which a sweep has converged
# Staged zoom of the shared-rho grid search: a coarse scan over the full
# RHO_BOUNDS range (global, so a rho ~ 0 local mode cannot capture the
# update), then repeated 21-point refinements around the incumbent. Final
# log-rho resolution ~ (range/40) * (0.1)^(stages-1) ~ 3e-4 < GLOBAL_RHO_LOG_TOL.
GLOBAL_RHO_GRID_STAGES = 4


def _fixed_rho_kernel(
    template: Kernel | None, n_dims: int, rho: float
) -> AR1MultiFidelityKernel:
    kernel = clone(template) if template is not None else make_joint_mf_kernel(n_dims)
    kernel.rho = rho
    kernel.rho_bounds = "fixed"
    return kernel


def _summed_lml_on_log_grid(
    models: list[GaussianProcessRegressor], log_rho_grid: np.ndarray
) -> np.ndarray:
    """Summed LML over all models at every grid rho, kernels held fixed.

    Matches sklearn's own LML definition: computed on the (normalized)
    stored training targets with the fitted kernel and the same jitter.
    Loops models in the outer loop so each model's kernel component
    matrices are built once and reused across the whole grid, with O(n^2)
    memory regardless of the number of wavelengths.
    """
    rhos = np.exp(log_rho_grid)
    totals = np.zeros(len(rhos))
    log_2pi = np.log(2.0 * np.pi)
    for gp in models:
        X = gp.X_train_
        features, fidelity = X[:, :-1], X[:, -1]
        kernel = gp.kernel_
        k_low = kernel.low_kernel(features)
        k_delta = kernel.discrepancy_kernel(features)
        high_scale = np.multiply.outer(fidelity, fidelity)
        noise = (
            np.where(fidelity == 0.0, kernel.low_noise, kernel.high_noise)
            + gp.alpha
        )
        y = gp.y_train_
        n = len(y)
        for i, rho in enumerate(rhos):
            scale = np.where(fidelity == 0.0, 1.0, rho)
            K = k_low * np.multiply.outer(scale, scale) + k_delta * high_scale
            K[np.diag_indices_from(K)] += noise
            try:
                L = cholesky(K, lower=True, check_finite=False)
            except LinAlgError:
                totals[i] = -np.inf
                continue
            alpha_vec = cho_solve((L, True), y, check_finite=False)
            totals[i] += (
                -0.5 * float(y @ alpha_vec)
                - float(np.log(np.diag(L)).sum())
                - 0.5 * n * log_2pi
            )
    return totals


def _optimize_shared_rho(
    models: list[GaussianProcessRegressor],
) -> tuple[float, float]:
    """Maximize the summed LML over one shared rho, kernels held fixed."""
    log_lo, log_hi = np.log(RHO_BOUNDS)
    center = 0.5 * (log_lo + log_hi)
    half_width = 0.5 * (log_hi - log_lo)
    n_points = 41
    best_log_rho, best_lml = center, -np.inf
    for _ in range(GLOBAL_RHO_GRID_STAGES):
        grid = np.clip(
            np.linspace(center - half_width, center + half_width, n_points),
            log_lo,
            log_hi,
        )
        totals = _summed_lml_on_log_grid(models, grid)
        best = int(np.argmax(totals))
        best_log_rho, best_lml = float(grid[best]), float(totals[best])
        half_width = float(grid[1] - grid[0])
        center, n_points = best_log_rho, 21
    return float(np.exp(best_log_rho)), best_lml


def fit_joint_mf_gp_global_rho(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    wavelengths: np.ndarray,
    *,
    seed: int,
    subsample_size: int | None = None,
    rho_init: float = GLOBAL_RHO_INIT,
    warm_start_layer: JointMFGPLayer | None = None,
    max_sweeps: int = GLOBAL_RHO_MAX_SWEEPS,
    log_tol: float = GLOBAL_RHO_LOG_TOL,
    n_restarts_optimizer: int = MF_GP_N_RESTARTS_OPTIMIZER,
    alpha: float = MF_GP_ALPHA,
    normalize_y: bool = MF_GP_NORMALIZE_Y,
    progress_every: int | None = None,
) -> JointMFGPLayer:
    """Model 1A: per-wavelength MF-GPs sharing ONE jointly optimized rho.

    Same joint AR(1) structure and LF/HF design as fit_joint_mf_gp; the only
    difference is the rho criterion: a single scalar maximizing the summed
    (i.e. joint) log marginal likelihood instead of one free rho_j per
    wavelength. ``warm_start_layer`` (e.g. the fitted Model 1B layer on the
    identical design) initializes every kernel and rho_init from its fits,
    which typically cuts the first sweep drastically; the criterion being
    optimized is unchanged by initialization.
    """
    scaler, subsample_indices, lf_rows, X_joint = _joint_design(
        X_lf,
        X_hf,
        seed=seed,
        subsample_size=subsample_size,
        n_theta=len(make_joint_mf_kernel(X_lf.shape[1]).theta) - 1,
    )

    templates: list[Kernel | None]
    if warm_start_layer is not None:
        if len(warm_start_layer.models) != len(wavelengths):
            raise ValueError(
                f"warm_start_layer has {len(warm_start_layer.models)} models "
                f"but {len(wavelengths)} wavelengths were requested"
            )
        if not np.array_equal(
            warm_start_layer.subsample_indices
            if warm_start_layer.subsample_indices is not None
            else np.arange(X_lf.shape[0]),
            lf_rows,
        ):
            raise ValueError(
                "warm_start_layer was fitted on a different LF subsample; "
                "use the same subsample_size and seed"
            )
        templates = [gp.kernel_ for gp in warm_start_layer.models]
        # Initialize rho by the same criterion the sweeps use -- the summed
        # LML over the warm kernels -- NOT a mean of the per-wavelength rhos,
        # which is dominated by wavelengths whose rho_j sits at a bound.
        rho_init, _ = _optimize_shared_rho(warm_start_layer.models)
    else:
        templates = [None] * len(wavelengths)

    def fit_all_fixed_rho(
        rho: float, current_templates: list[Kernel | None], restarts: int
    ) -> tuple[list[GaussianProcessRegressor], np.ndarray]:
        models = []
        seconds = []
        for j in range(len(wavelengths)):
            y_joint = np.concatenate((Y_lf[lf_rows, j], Y_hf[:, j]))
            model = GaussianProcessRegressor(
                kernel=_fixed_rho_kernel(
                    current_templates[j], X_lf.shape[1], rho
                ),
                alpha=alpha,
                n_restarts_optimizer=restarts,
                normalize_y=normalize_y,
                random_state=seed,
            )
            t0 = time.perf_counter()
            model.fit(X_joint, y_joint)
            seconds.append(time.perf_counter() - t0)
            models.append(model)
            if progress_every and (j + 1) % progress_every == 0:
                print(
                    f"  fit {j + 1}/{len(wavelengths)} fixed-rho MF-GPs "
                    f"({seconds[-1]:.1f}s last)",
                    flush=True,
                )
        return models, np.asarray(seconds)

    rho = rho_init
    history: list[dict] = []
    total_seconds = np.zeros(len(wavelengths))
    models: list[GaussianProcessRegressor] = []
    converged = False
    for sweep in range(max_sweeps):
        restarts = (
            n_restarts_optimizer
            if sweep == 0 and warm_start_layer is None
            else 0
        )
        models, seconds = fit_all_fixed_rho(rho, templates, restarts)
        total_seconds += seconds
        templates = [gp.kernel_ for gp in models]
        new_rho, summed_lml = _optimize_shared_rho(models)
        delta_log = abs(float(np.log(new_rho / rho)))
        history.append(
            {
                "sweep": sweep,
                "rho_before": rho,
                "rho_after": new_rho,
                "summed_log_marginal_likelihood": summed_lml,
                "abs_delta_log_rho": delta_log,
            }
        )
        if progress_every:
            print(
                f"  sweep {sweep}: rho {rho:.5g} -> {new_rho:.5g}, "
                f"summed LML {summed_lml:.2f}",
                flush=True,
            )
        rho = new_rho
        if delta_log < log_tol:
            converged = True
            break
    if not converged:
        print(
            f"  WARNING: shared rho not converged after {max_sweeps} sweeps "
            f"(last |delta log rho| = {history[-1]['abs_delta_log_rho']:.2e})",
            flush=True,
        )

    # Final consistency refit so every kernel carries the final rho.
    models, seconds = fit_all_fixed_rho(rho, templates, 0)
    total_seconds += seconds

    # rho is 'fixed' in these kernels (swept separately), so check it against
    # RHO_BOUNDS explicitly on top of the per-kernel bound scan.
    warn_on_pinned_bounds(models, label="Model 1A (global-rho MF-GP)")
    rho_lo, rho_hi = RHO_BOUNDS
    if np.isclose(np.log(rho), np.log(rho_lo), atol=BOUND_PIN_ATOL) or np.isclose(
        np.log(rho), np.log(rho_hi), atol=BOUND_PIN_ATOL
    ):
        warnings.warn(
            f"Model 1A shared rho={rho:.4g} pins RHO_BOUNDS={RHO_BOUNDS}; widen it",
            stacklevel=2,
        )

    return JointMFGPLayer(
        wavelengths=np.asarray(wavelengths),
        scaler=scaler,
        subsample_indices=subsample_indices,
        models=models,
        fit_seconds=total_seconds,
        global_rho_sweeps=history,
    )


def predict_fidelity(
    layer: JointMFGPLayer,
    X_new: np.ndarray,
    *,
    fidelity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict all wavelengths at LF (0) or HF (1)."""
    if fidelity not in (0, 1):
        raise ValueError("fidelity must be 0 (LF) or 1 (HF)")
    X_scaled = layer.scaler.transform(X_new)
    X_augmented = _append_fidelity(X_scaled, fidelity)
    means, stds = zip(
        *(model.predict(X_augmented, return_std=True) for model in layer.models)
    )
    return np.column_stack(means), np.column_stack(stds)


def predict_hf(
    layer: JointMFGPLayer,
    X_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict HF spectra directly from atmospheric inputs."""
    return predict_fidelity(layer, X_new, fidelity=1)


def hyperparameter_table(layer: JointMFGPLayer) -> pd.DataFrame:
    """One row per wavelength of joint-fit covariance parameters."""
    rows = []
    for wl, gp in zip(layer.wavelengths, layer.models):
        kernel = gp.kernel_
        low_amplitude, low_rbf = kernel.low_kernel.k1, kernel.low_kernel.k2
        delta_amplitude, delta_rbf = (
            kernel.discrepancy_kernel.k1,
            kernel.discrepancy_kernel.k2,
        )
        row = {
            "wavelength": wl,
            "rho": kernel.rho,
            "low_signal_variance": low_amplitude.constant_value,
            "delta_signal_variance": delta_amplitude.constant_value,
            "low_noise": kernel.low_noise,
            "high_noise": kernel.high_noise,
            "joint_log_marginal_likelihood": gp.log_marginal_likelihood(),
        }
        if layer.fit_seconds is not None:
            row["fit_seconds"] = layer.fit_seconds[len(rows)]
        row.update(
            {
                f"low_length_scale_{i}": ell
                for i, ell in enumerate(low_rbf.length_scale)
            }
        )
        row.update(
            {
                f"delta_length_scale_{i}": ell
                for i, ell in enumerate(delta_rbf.length_scale)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Data-alignment guards.
# ---------------------------------------------------------------------------

def assert_holdout_row(
    XHF: pd.DataFrame,
    YHF: pd.DataFrame,
    row: int,
    *,
    expected_kzz: float,
    expected_label: str,
    atol: float = 1e-9,
) -> None:
    """Guard against silent row misalignment before excluding `row` from a fit.

    Mirrors exoplanets_mf.data's _require_same_length/_require_same_grid
    defensive style: fail loudly and specifically rather than silently
    training on/holding out the wrong sample.
    """
    actual_kzz = float(XHF.iloc[row]["Kzz"])
    if not np.isclose(actual_kzz, expected_kzz, atol=atol):
        raise ValueError(f"row {row}: expected Kzz={expected_kzz}, found {actual_kzz}")
    actual_label = YHF.index[row]
    if actual_label != expected_label:
        raise ValueError(f"row {row}: expected YHF index {expected_label!r}, found {actual_label!r}")
