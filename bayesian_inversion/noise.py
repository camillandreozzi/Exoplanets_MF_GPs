"""The error budget entering the likelihood.

Two independent things make the measured spectrum differ from the emulator's
prediction at the *true* parameters:

1. observational error -- the instrument, given by ``Observed_Spectra.csv``;
2. emulator error -- the GP is an approximation of the simulator, fitted from
   96 high-fidelity runs, so it misses even when theta is exactly right.

Leaving (2) out tells the sampler the emulator is perfect, and it then chases
emulator artefacts as if they were signal. That would be a serious error here
rather than a refinement: at held-out sample 81 the emulator's out-of-sample RMS
error is 1.06e-4 against a median observational error of 8.2e-5, so the emulator
is the *larger* of the two terms.

Emulator error is also correlated across wavelength -- the residuals are a coherent
offset, not white noise -- so ``Sigma_em`` is estimated as a full 195x195 matrix.
Treating 195 correlated errors as independent would shrink the posterior spuriously.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular
from sklearn.covariance import LedoitWolf

from bayesian_inversion.observations import N_WAVELENGTHS, Observation

LOO_PREDICTIONS = PROJECT_ROOT / "results/cv/loo/loo_predictions.csv"

MODEL = "model1"
VARIANT = "mf"


def emulator_residuals(path: Path = LOO_PREDICTIONS) -> np.ndarray:
    """The (97, 195) matrix of leave-one-out residuals for Model 1 MF.

    Each row is one held-out HF sample; each column one wavelength. These are
    genuine out-of-sample errors -- exactly the quantity the likelihood needs -- so
    the emulator error budget is measured, not assumed.
    """
    predictions = pd.read_csv(path)
    subset = predictions[
        (predictions["model"] == MODEL) & (predictions["variant"] == VARIANT)
    ]
    if subset.empty:
        raise ValueError(f"No {MODEL}/{VARIANT} rows found in {path}")

    residuals = subset.pivot_table(
        index="held_out_source_index", columns="response_index", values="residual"
    )
    if residuals.shape[1] != N_WAVELENGTHS:
        raise ValueError(
            f"Expected {N_WAVELENGTHS} wavelengths, found {residuals.shape[1]}"
        )
    if residuals.isna().to_numpy().any():
        raise ValueError("Leave-one-out residual matrix has gaps")

    return residuals.sort_index(axis=1).to_numpy(dtype=float)


def emulator_error_covariance(
    path: Path = LOO_PREDICTIONS, diagonal: bool = False
) -> np.ndarray:
    """Estimate ``Sigma_em`` (195, 195) from the leave-one-out residuals.

    Ledoit-Wolf shrinkage rather than the sample covariance: 97 samples for a
    195x195 target is rank-deficient, and the raw estimate would be singular.

    ``diagonal=True`` keeps only the per-wavelength variances. That is not the
    recommended setting -- it discards the cross-wavelength correlation the docstring
    above argues is essential -- but it is a useful sensitivity check.
    """
    residuals = emulator_residuals(path)

    if diagonal:
        return np.diag(residuals.var(axis=0, ddof=1))

    # assume_centered=False: the residuals carry a real mean offset (the emulator is
    # biased, not just noisy), and we want the covariance about that mean since any
    # systematic component is shared with the prediction we are scoring.
    return LedoitWolf(assume_centered=False).fit(residuals).covariance_


def total_covariance(observation: Observation, emulator_cov: np.ndarray) -> np.ndarray:
    """``Sigma = diag(sigma_obs^2) + Sigma_em``, the full likelihood covariance."""
    if emulator_cov.shape != (N_WAVELENGTHS, N_WAVELENGTHS):
        raise ValueError(f"emulator_cov has shape {emulator_cov.shape}")
    return np.diag(observation.sigma**2) + emulator_cov


def gaussian_loglike_factory(
    y: np.ndarray, cov: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batched Gaussian log-likelihood with the covariance pre-factorised.

    ``Sigma`` does not depend on theta, so its Cholesky factor and log-determinant
    are computed once here and reused for every evaluation. Each call then costs one
    triangular solve on the (B, 195) residual block instead of a fresh factorisation.
    """
    n = len(y)
    factor = cho_factor(cov, lower=True)
    lower = np.tril(factor[0])
    # log|Sigma| = 2 * sum(log(diag(L))) for Sigma = L L^T
    log_det = 2.0 * np.sum(np.log(np.diag(lower)))
    normalisation = -0.5 * (n * np.log(2.0 * np.pi) + log_det)

    def log_likelihood(predictions: np.ndarray) -> np.ndarray:
        """``predictions`` is (B, n_wavelengths); returns (B,)."""
        residuals = np.atleast_2d(predictions) - y
        # solve_triangular on the transposed block: whitens all B residuals at once.
        whitened = solve_triangular(lower, residuals.T, lower=True)
        return normalisation - 0.5 * np.sum(whitened**2, axis=0)

    return log_likelihood


def error_budget_summary(
    observation: Observation, emulator_cov: np.ndarray
) -> pd.DataFrame:
    """Per-wavelength comparison of the two error sources, for the run log."""
    emulator_sd = np.sqrt(np.diag(emulator_cov))
    return pd.DataFrame(
        {
            "wavelength": observation.wavelengths,
            "sigma_obs": observation.sigma,
            "sigma_emulator": emulator_sd,
            "ratio_emulator_to_obs": emulator_sd / observation.sigma,
        }
    )


if __name__ == "__main__":
    from bayesian_inversion.observations import load_observed

    obs = load_observed()
    cov = emulator_error_covariance()
    summary = error_budget_summary(obs, cov)

    print(f"emulator covariance {cov.shape}, condition {np.linalg.cond(cov):.3e}")
    print(summary.describe().T[["mean", "50%", "min", "max"]].to_string())

    total = total_covariance(obs, cov)
    loglike = gaussian_loglike_factory(obs.y, total)
    print("\nlog-likelihood at the observation itself:", loglike(obs.y[None, :])[0])
