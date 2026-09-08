"""Prior over the nine atmospheric parameters.

The prior is specified by the retrieval setup (``PRIOR_SPEC``): uniform on seven
parameters, Gaussian on ``Rp`` and ``logg`` -- the two constrained by independent
mass/radius measurements rather than by the spectrum.

A second, separate constraint applies here that does not apply to a retrieval run
with the real simulator: the emulator was only trained inside the ``XLF10k`` design
box, so outside it the GP extrapolates and its predictions are meaningless. Every
draw is therefore restricted to that box. For the seven uniform parameters this
costs nothing -- the design box and the prior bounds coincide exactly. For the two
Gaussians the stated bounds (``Rp`` [0.1, 2], ``logg`` [2, 6]) are far wider than
the training range, so they are truncated to the box instead; see ``EMULATOR_BOX``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PARAMETER_NAMES = (
    "Kzz",
    "Rp",
    "Tint",
    "C/O",
    "[N/H]",
    "[O/H]",
    "[S/H]",
    "logg",
    "f",
)
N_PARAMS = len(PARAMETER_NAMES)

# The retrieval's prior specification, in PARAMETER_NAMES order.
# "uniform"  -> (low, high)
# "gaussian" -> (mean, sd)
PRIOR_SPEC: dict[str, tuple[str, tuple[float, float]]] = {
    "Kzz": ("uniform", (8.0, 11.5)),
    "Rp": ("gaussian", (0.999, 0.031)),
    "Tint": ("uniform", (150.0, 450.0)),
    "C/O": ("uniform", (0.1, 1.5)),
    "[N/H]": ("uniform", (-1.0, 2.0)),
    "[O/H]": ("uniform", (-1.0, 2.0)),
    "[S/H]": ("uniform", (-1.0, 2.0)),
    "logg": ("gaussian", (3.145, 0.016)),
    "f": ("uniform", (1 / 4, 2 / 3)),
}

# Support of the prior *and* the region where the emulator is trustworthy.
#
# Rows in PARAMETER_NAMES order. The seven uniform rows are the prior bounds
# verbatim. The two Gaussian rows are the XLF10k design range, which is +-3.85 sd
# on both (0.999 +- 3.85*0.031 = [0.880, 1.118]; 3.145 +- 3.9*0.016 = [3.083, 3.207]),
# so truncating there discards ~1e-4 of the prior mass. Checked against the design
# file by ``check_box_against_design``.
EMULATOR_BOX = np.array(
    [
        [8.0, 11.5],  # Kzz
        [0.881978, 1.120004],  # Rp     (XLF10k range)
        [150.0, 450.0],  # Tint
        [0.1, 1.5],  # C/O
        [-1.0, 2.0],  # [N/H]
        [-1.0, 2.0],  # [O/H]
        [-1.0, 2.0],  # [S/H]
        [3.084744, 3.207999],  # logg   (XLF10k range)
        [1 / 4, 2 / 3],  # f
    ]
)


def _distributions() -> list[stats.rv_continuous]:
    """One frozen scipy distribution per parameter, truncated to ``EMULATOR_BOX``.

    Built once at import: ``truncnorm``'s shape parameters are expressed in
    standardised units, which is fiddly enough that it is worth doing in exactly
    one place.
    """
    dists = []
    for i, name in enumerate(PARAMETER_NAMES):
        kind, params = PRIOR_SPEC[name]
        low, high = EMULATOR_BOX[i]
        if kind == "uniform":
            dists.append(stats.uniform(loc=low, scale=high - low))
        elif kind == "gaussian":
            mean, sd = params
            dists.append(
                stats.truncnorm(
                    (low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd
                )
            )
        else:  # pragma: no cover - guards a typo in PRIOR_SPEC
            raise ValueError(f"Unknown prior type {kind!r} for {name}")
    return dists


_DISTRIBUTIONS = _distributions()


def in_box(theta: np.ndarray) -> np.ndarray:
    """Boolean mask of which rows of ``theta`` (B, 9) lie inside ``EMULATOR_BOX``."""
    theta = np.atleast_2d(theta)
    return np.all(
        (theta >= EMULATOR_BOX[:, 0]) & (theta <= EMULATOR_BOX[:, 1]), axis=1
    )


def log_prior(theta: np.ndarray) -> np.ndarray:
    """Log prior density for a batch of parameter vectors.

    Parameters
    ----------
    theta : (B, 9) array in ``PARAMETER_NAMES`` order.

    Returns
    -------
    (B,) array; ``-inf`` for any row outside ``EMULATOR_BOX``.
    """
    theta = np.atleast_2d(theta)
    inside = in_box(theta)

    logp = np.full(len(theta), -np.inf)
    if not inside.any():
        return logp

    contributions = np.stack(
        [dist.logpdf(theta[inside, i]) for i, dist in enumerate(_DISTRIBUTIONS)],
        axis=1,
    )
    logp[inside] = contributions.sum(axis=1)
    return logp


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` parameter vectors from the prior. Returns (n, 9)."""
    return np.column_stack(
        [dist.rvs(size=n, random_state=rng) for dist in _DISTRIBUTIONS]
    )


def check_box_against_design(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Compare ``EMULATOR_BOX`` to the actual XLF10k design range.

    The 10k low-fidelity design is itself a draw from this prior, so its range is
    the empirical statement of where the emulator was trained. A mismatch means
    either ``PRIOR_SPEC`` is wrong or the design is not what it is assumed to be.
    """
    design = pd.read_csv(data_dir / "XLF10k.csv", skipinitialspace=True)
    design.columns = design.columns.str.strip()
    design = design[list(PARAMETER_NAMES)]

    return pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "box_low": EMULATOR_BOX[:, 0],
            "design_min": design.min().to_numpy(),
            "design_max": design.max().to_numpy(),
            "box_high": EMULATOR_BOX[:, 1],
            "prior_type": [PRIOR_SPEC[n][0] for n in PARAMETER_NAMES],
        }
    )


if __name__ == "__main__":
    print(check_box_against_design().to_string(index=False))

    rng = np.random.default_rng(2026)
    draws = sample_prior(100_000, rng)
    design = pd.read_csv(DATA_DIR / "XLF10k.csv", skipinitialspace=True)
    design.columns = design.columns.str.strip()

    print("\nKS test, prior draws vs XLF10k design:")
    for i, name in enumerate(PARAMETER_NAMES):
        result = stats.ks_2samp(draws[:, i], design[name].to_numpy())
        flag = "ok" if result.pvalue > 0.01 else "MISMATCH"
        print(f"  {name:<7} D={result.statistic:.4f}  p={result.pvalue:.3f}  {flag}")
