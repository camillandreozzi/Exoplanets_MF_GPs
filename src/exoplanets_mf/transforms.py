"""log10 output-space transforms for eclipse-depth spectra.

log10 rather than natural log, following astronomy convention. The simulated spectra (YHF, YLF, YLF_10k) are strictly
positive (minimum ~1.3e-6), so no clamping is applied -- non-positive input
fails loudly instead of being silently clipped. The *observed* spectrum does
contain negative depths (noise around small signals); callers must mask those
before transforming (see research/exploratory/05_log_spectra_exploration.py).
"""

from __future__ import annotations

import numpy as np


def log10_spectra(Y: np.ndarray) -> np.ndarray:
    """Elementwise log10 of strictly positive spectra."""
    Y = np.asarray(Y, dtype=float)
    if not (Y > 0).all():
        raise ValueError(
            f"log10_spectra requires strictly positive values; "
            f"min={Y.min():.3e} ({np.count_nonzero(Y <= 0)} non-positive entries)"
        )
    return np.log10(Y)


def inverse_log10_spectra(Y_log: np.ndarray) -> np.ndarray:
    """Back-transform log10 spectra to original units.

    Applied to a GP predictive mean this yields the predictive *median* of
    the implied log-normal, not its mean.
    """
    return 10.0 ** np.asarray(Y_log, dtype=float)


def log10_observed_errors(
    depth: np.ndarray, err_lo: np.ndarray, err_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform asymmetric error magnitudes to log10 space.

        err_lo' = log10(d) - log10(d - err_lo)
        err_hi' = log10(d + err_hi) - log10(d)

    Requires d > 0 and d - err_lo > 0: rows violating either have no finite
    lower bar in log space, so the caller must mask or clip them first (the
    real observed data contains such rows).
    """
    depth = np.asarray(depth, dtype=float)
    err_lo = np.asarray(err_lo, dtype=float)
    err_hi = np.asarray(err_hi, dtype=float)
    if not (depth > 0).all():
        raise ValueError("log10_observed_errors requires strictly positive depths")
    if not (depth - err_lo > 0).all():
        raise ValueError(
            "log10_observed_errors requires depth - err_lo > 0; "
            "mask or clip the offending rows first"
        )
    return (
        np.log10(depth) - np.log10(depth - err_lo),
        np.log10(depth + err_hi) - np.log10(depth),
    )
