"""The log posterior that the sampler explores.

    log p(theta | y)  =  log p(theta)  +  log N(y | S_hat(theta), Sigma)   + const

``S_hat`` is the emulator (``emulator.py``) and ``Sigma`` is the combined
observational + emulator error covariance (``noise.py``). This module is only the
wiring; the substance is in those two.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from bayesian_inversion.emulator import MFEmulator
from bayesian_inversion.noise import gaussian_loglike_factory, total_covariance
from bayesian_inversion.observations import Observation
from bayesian_inversion.priors import log_prior


class LogPosterior:
    """Batched log posterior, shaped for ``emcee``'s ``vectorize=True``.

    Walkers outside the emulator's training box are rejected on the prior alone and
    never reach the emulator. That is not a micro-optimisation: a forward evaluation
    costs hundreds of milliseconds, so skipping the ones whose answer is already
    ``-inf`` is worth real wall-clock time, and it also keeps the GP from being
    queried where it would be extrapolating.
    """

    def __init__(
        self,
        emulator: MFEmulator,
        observation: Observation,
        emulator_cov: np.ndarray,
    ) -> None:
        self.emulator = emulator
        self.observation = observation

        covariance = total_covariance(observation, emulator_cov)
        self.log_likelihood = gaussian_loglike_factory(observation.y, covariance)

        self.n_calls = 0
        self.n_emulator_evaluations = 0

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """``theta`` is (B, 9); returns (B,) log posterior densities."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))

        logp = log_prior(theta)
        finite = np.isfinite(logp)

        self.n_calls += 1
        if not finite.any():
            return logp

        predictions = self.emulator.mean(theta[finite])
        self.n_emulator_evaluations += int(finite.sum())

        logp[finite] += self.log_likelihood(predictions)
        return logp
