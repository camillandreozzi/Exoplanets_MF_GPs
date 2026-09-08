"""Posterior predictive check: push the posterior back through the emulator.

Inference gives us parameters; this asks the follow-up question -- do those
parameters actually reproduce the spectrum we started from? A posterior that fits
the data badly is a posterior that should not be believed, however tidy its corner
plot looks.

Two bands are drawn, and the distinction between them is the point of the figure:

* **parameter spread** -- the emulator's predictions across posterior draws, i.e.
  how much the spectrum moves given what we know about theta. This alone is *not*
  what should bracket the data.
* **full predictive** -- the same, plus a draw from the error budget Sigma
  (observational + emulator error). This is what the data should fall inside, and
  the gap between the two bands shows how much of the scatter is error budget
  rather than parameter uncertainty.

    python3 bayesian_inversion/posterior_predictive.py --target observed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results/bayesian_inversion"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from bayesian_inversion.emulator import DEFAULT_WORKERS, MFEmulator
from bayesian_inversion.noise import emulator_error_covariance, total_covariance
from bayesian_inversion.observations import load_observation

DEFAULT_DRAWS = 200  # ~0.4 s per draw through the emulator
OBSERVED_COLOUR = "black"
PREDICTIVE_COLOUR = "tab:blue"
PARAMETER_COLOUR = "tab:orange"


def predictive_spectra(
    samples: np.ndarray,
    n_draws: int = DEFAULT_DRAWS,
    workers: int = DEFAULT_WORKERS,
    seed: int = 0,
) -> np.ndarray:
    """Emulator predictions for a random subset of posterior draws. Returns (n, 195)."""
    rng = np.random.default_rng(seed)
    n_draws = min(n_draws, len(samples))
    thinned = samples[rng.choice(len(samples), size=n_draws, replace=False)]

    with MFEmulator(n_workers=workers) as emulator:
        return emulator.mean(thinned)


def add_error_budget(
    spectra: np.ndarray, covariance: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Add one draw from the error budget to each predicted spectrum.

    Sampling the full covariance rather than the diagonal keeps the cross-wavelength
    correlation of the emulator error, which is exactly what makes the real band
    wider than a naive per-wavelength one.
    """
    rng = np.random.default_rng(seed + 1)
    upper = cholesky(covariance, lower=False)
    noise = rng.standard_normal(spectra.shape) @ upper
    return spectra + noise


def summarise(
    observation,
    parameter_spectra: np.ndarray,
    predictive_spectra_: np.ndarray,
    covariance: np.ndarray,
) -> pd.DataFrame:
    """Per-wavelength table of observation, prediction, bands and residual."""
    median = np.median(parameter_spectra, axis=0)
    predictive_sd = predictive_spectra_.std(axis=0, ddof=1)

    return pd.DataFrame(
        {
            "wavelength": observation.wavelengths,
            "observed": observation.y,
            "sigma_obs": observation.sigma,
            "predicted_median": median,
            "parameter_lo_95": np.percentile(parameter_spectra, 2.5, axis=0),
            "parameter_hi_95": np.percentile(parameter_spectra, 97.5, axis=0),
            "predictive_lo_95": np.percentile(predictive_spectra_, 2.5, axis=0),
            "predictive_hi_95": np.percentile(predictive_spectra_, 97.5, axis=0),
            "residual": observation.y - median,
            # Standardised by the full predictive spread, so ~N(0,1) if calibrated.
            "z": (observation.y - median) / predictive_sd,
        }
    )


def goodness_of_fit(observation, median: np.ndarray, covariance: np.ndarray) -> dict:
    """Mahalanobis chi-square of the median prediction against the observation.

    Uses the full covariance, so correlated emulator error is accounted for. With
    195 wavelengths a well-calibrated fit gives chi-square near 195 (reduced ~1);
    far above means the model cannot reproduce the data, far below means the error
    budget is too generous.
    """
    residual = observation.y - median
    upper = cholesky(covariance, lower=False)
    whitened = np.linalg.solve(upper.T, residual)
    chi_square = float(whitened @ whitened)
    return {
        "chi_square": chi_square,
        "n_wavelengths": len(residual),
        "reduced_chi_square": chi_square / len(residual),
    }


def plot_predictive(
    observation, table: pd.DataFrame, n_draws: int, title: str
) -> plt.Figure:
    fig, (ax_spectrum, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        height_ratios=[3, 1],
        constrained_layout=True,
    )

    wavelengths = table["wavelength"].to_numpy()

    ax_spectrum.fill_between(
        wavelengths,
        table["predictive_lo_95"],
        table["predictive_hi_95"],
        color=PREDICTIVE_COLOUR,
        alpha=0.20,
        label="95% posterior predictive (incl. error budget)",
    )
    ax_spectrum.fill_between(
        wavelengths,
        table["parameter_lo_95"],
        table["parameter_hi_95"],
        color=PARAMETER_COLOUR,
        alpha=0.35,
        label="95% from parameter uncertainty alone",
    )
    ax_spectrum.plot(
        wavelengths,
        table["predicted_median"],
        color=PREDICTIVE_COLOUR,
        lw=1.8,
        label="posterior median prediction",
    )
    ax_spectrum.errorbar(
        wavelengths,
        table["observed"],
        yerr=observation.sigma,
        fmt="o",
        ms=2.5,
        lw=0.8,
        color=OBSERVED_COLOUR,
        label="observed spectrum",
        zorder=5,
    )

    ax_spectrum.set_ylabel("Eclipse depth")
    ax_spectrum.set_title(title)
    ax_spectrum.legend(fontsize=9, loc="upper left")

    ax_residual.axhspan(-2, 2, color=PREDICTIVE_COLOUR, alpha=0.12)
    ax_residual.axhline(0, color=OBSERVED_COLOUR, lw=0.8)
    ax_residual.plot(wavelengths, table["z"], "o-", ms=2.5, lw=0.7, color=OBSERVED_COLOUR)
    ax_residual.set_xlabel("Wavelength (micron)")
    ax_residual.set_ylabel("residual / sd")
    ax_residual.set_ylim(-5, 5)

    fig.suptitle(f"{n_draws} posterior draws through the emulator", fontsize=9, y=0.005)
    return fig


def run(
    target: str,
    n_draws: int = DEFAULT_DRAWS,
    workers: int = DEFAULT_WORKERS,
    seed: int = 0,
) -> dict:
    output_dir = RESULTS_DIR / target
    samples_path = output_dir / "samples_flat.npy"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No samples at {samples_path}; run run_inversion.py --target {target} first"
        )

    samples = np.load(samples_path)
    observation = load_observation(target)
    covariance = total_covariance(observation, emulator_error_covariance())

    print(f"target      : {target}")
    print(f"posterior   : {len(samples):,} samples, using {min(n_draws, len(samples))}")

    parameter_spectra = predictive_spectra(samples, n_draws, workers, seed)
    predictive = add_error_budget(parameter_spectra, covariance, seed)

    table = summarise(observation, parameter_spectra, predictive, covariance)
    table.to_csv(output_dir / "posterior_predictive.csv", index=False)

    fit = goodness_of_fit(
        observation, table["predicted_median"].to_numpy(), covariance
    )
    within_95 = float(
        (
            (observation.y >= table["predictive_lo_95"])
            & (observation.y <= table["predictive_hi_95"])
        ).mean()
    )
    fit["fraction_within_95"] = within_95
    fit["max_abs_z"] = float(table["z"].abs().max())

    fig = plot_predictive(
        observation,
        table,
        min(n_draws, len(samples)),
        f"Posterior predictive check -- {target}",
    )
    path = output_dir / "posterior_predictive.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"chi-square  : {fit['chi_square']:.1f} on {fit['n_wavelengths']} wavelengths")
    print(f"reduced     : {fit['reduced_chi_square']:.2f}   (1.0 is well calibrated)")
    print(f"within 95%  : {within_95:.1%} of wavelengths (95% expected)")
    print(f"max |z|     : {fit['max_abs_z']:.1f}")
    print(f"written to  : {path}")
    return fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["sample81", "observed"], required=True)
    parser.add_argument("--n-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.target, args.n_draws, args.workers, args.seed)
