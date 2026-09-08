"""Triangle (corner) plot of the posterior over the nine atmospheric parameters.

The diagonal holds the 1-D marginal density of each parameter; the lower triangle
holds the 2-D joint density of every pair. The pairwise panels are the point of the
figure: they show the degeneracies -- directions in parameter space the spectrum
cannot separate -- which a table of marginal error bars hides completely.

Run as a script to (re)plot a saved chain without repeating the sampling:

    python3 bayesian_inversion/plot_posterior.py --target sample81
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
from scipy.stats import gaussian_kde

from bayesian_inversion.priors import EMULATOR_BOX, N_PARAMS, PARAMETER_NAMES, sample_prior

POSTERIOR_COLOUR = "tab:blue"
PRIOR_COLOUR = "tab:gray"
TRUTH_COLOUR = "tab:red"
REFERENCE_COLOUR = "red"

# Published retrieval values for the real planet, read approximately off Wiser et
# al. (2025). Only six of the nine parameters are quoted; the three abundances are
# left unset and simply not drawn. These are a visual reference for the `observed`
# run only -- they say nothing about the synthetic sample81 design point.
WISER_2025 = {
    "Kzz": 9.2,
    "Rp": 0.99,
    "Tint": 380.0,
    "C/O": 0.5,
    "logg": 3.15,
    "f": 0.3,
}


def reference_vector(values: dict[str, float]) -> np.ndarray:
    """Map a partial {parameter: value} dict onto a (9,) array, NaN where unset."""
    return np.array(
        [values.get(name, np.nan) for name in PARAMETER_NAMES], dtype=float
    )

# Credible levels for the 2-D panels. In two dimensions a Gaussian encloses these
# mass fractions at 1 and 2 sigma -- not 68/95, which are the 1-D numbers and are a
# common source of over-confident contours.
CREDIBLE_LEVELS_2D = (0.393, 0.865)
GRID = 128
MAX_KDE_SAMPLES = 20_000  # gaussian_kde is O(n) per evaluation point


def _subsample(samples: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    """Thin to at most ``limit`` rows; MCMC samples are autocorrelated anyway."""
    if len(samples) <= limit:
        return samples
    rng = np.random.default_rng(seed)
    return samples[rng.choice(len(samples), size=limit, replace=False)]


def _density_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        return gaussian_kde(values)(grid)
    except np.linalg.LinAlgError:
        # Degenerate (a parameter pinned to one value); fall back to a histogram.
        counts, edges = np.histogram(values, bins=40, range=(grid[0], grid[-1]), density=True)
        return np.interp(grid, 0.5 * (edges[:-1] + edges[1:]), counts)


def _contour_levels(density: np.ndarray) -> list[float]:
    """Density thresholds enclosing ``CREDIBLE_LEVELS_2D`` of the total mass.

    Sort the grid cells by density, accumulate from the top, and read off the value
    at which the requested fraction of the mass has been covered. This is the
    highest-density-region construction, which is what a credible contour means.
    """
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]

    levels = []
    for target in sorted(CREDIBLE_LEVELS_2D, reverse=True):
        index = np.searchsorted(cumulative, target)
        levels.append(flat[min(index, len(flat) - 1)])

    # Contour levels must be strictly increasing and distinct.
    levels = sorted(set(levels))
    return levels if len(levels) >= 1 else [flat[len(flat) // 2]]


def plot_corner(
    samples: np.ndarray,
    truth: np.ndarray | None = None,
    show_prior: bool = True,
    title: str | None = None,
    seed: int = 0,
    reference: np.ndarray | None = None,
    reference_label: str = "reference",
) -> plt.Figure:
    """Triangle plot of the posterior. ``samples`` is (N, 9).

    ``reference`` is an optional (9,) array of published values to overlay; entries
    that are NaN are simply not drawn, so a partially-quoted result can be shown
    without inventing the missing parameters.
    """
    thinned = _subsample(samples, MAX_KDE_SAMPLES, seed)
    prior_draws = (
        sample_prior(20_000, np.random.default_rng(seed)) if show_prior else None
    )

    fig, axes = plt.subplots(
        N_PARAMS, N_PARAMS, figsize=(16, 16), constrained_layout=True
    )

    for row in range(N_PARAMS):
        for col in range(N_PARAMS):
            ax = axes[row, col]

            if col > row:
                ax.axis("off")
                continue

            low_x, high_x = EMULATOR_BOX[col]

            if row == col:
                grid = np.linspace(low_x, high_x, GRID)
                density = _density_1d(thinned[:, col], grid)
                ax.plot(grid, density, color=POSTERIOR_COLOUR, lw=1.8)
                ax.fill_between(grid, density, color=POSTERIOR_COLOUR, alpha=0.25)
                if prior_draws is not None:
                    ax.plot(
                        grid,
                        _density_1d(prior_draws[:, col], grid),
                        color=PRIOR_COLOUR,
                        lw=1.2,
                        ls="--",
                    )
                if truth is not None:
                    ax.axvline(truth[col], color=TRUTH_COLOUR, lw=1.5)
                if reference is not None and np.isfinite(reference[col]):
                    ax.axvline(reference[col], color=REFERENCE_COLOUR, lw=2.2)
                ax.set_yticks([])
                ax.set_xlim(low_x, high_x)
            else:
                low_y, high_y = EMULATOR_BOX[row]
                x, y = thinned[:, col], thinned[:, row]

                grid_x = np.linspace(low_x, high_x, GRID)
                grid_y = np.linspace(low_y, high_y, GRID)
                mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

                try:
                    density = gaussian_kde(np.vstack([x, y]))(
                        np.vstack([mesh_x.ravel(), mesh_y.ravel()])
                    ).reshape(mesh_x.shape)
                    ax.contourf(
                        grid_x, grid_y, density, levels=24, cmap="Blues"
                    )
                    ax.contour(
                        grid_x,
                        grid_y,
                        density,
                        levels=_contour_levels(density),
                        colors="k",
                        linewidths=0.7,
                        alpha=0.6,
                    )
                except np.linalg.LinAlgError:
                    ax.hist2d(x, y, bins=40, range=[[low_x, high_x], [low_y, high_y]], cmap="Blues")

                if truth is not None:
                    ax.axvline(truth[col], color=TRUTH_COLOUR, lw=1.0, alpha=0.8)
                    ax.axhline(truth[row], color=TRUTH_COLOUR, lw=1.0, alpha=0.8)
                    ax.plot(truth[col], truth[row], "s", color=TRUTH_COLOUR, ms=5)

                if reference is not None:
                    # Draw each axis independently, so a pair with only one quoted
                    # parameter still shows the line it does have. The marker needs
                    # both, so it is drawn only when both are finite.
                    if np.isfinite(reference[col]):
                        ax.axvline(reference[col], color=REFERENCE_COLOUR, lw=1.6)
                    if np.isfinite(reference[row]):
                        ax.axhline(reference[row], color=REFERENCE_COLOUR, lw=1.6)
                    if np.isfinite(reference[col]) and np.isfinite(reference[row]):
                        ax.plot(
                            reference[col],
                            reference[row],
                            "D",
                            color=REFERENCE_COLOUR,
                            ms=5,
                            markeredgecolor="white",
                            markeredgewidth=0.6,
                        )

                ax.set_xlim(low_x, high_x)
                ax.set_ylim(low_y, high_y)

            if row == N_PARAMS - 1:
                ax.set_xlabel(PARAMETER_NAMES[col], fontsize=11)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(PARAMETER_NAMES[row], fontsize=11)
            elif col != 0:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=8)
            for label in ax.get_xticklabels():
                label.set_rotation(45)

    handles = [
        plt.Line2D([], [], color=POSTERIOR_COLOUR, lw=2, label="posterior"),
    ]
    if show_prior:
        handles.append(plt.Line2D([], [], color=PRIOR_COLOUR, lw=1.2, ls="--", label="prior"))
    if truth is not None:
        handles.append(plt.Line2D([], [], color=TRUTH_COLOUR, lw=1.5, label="truth"))
    if reference is not None:
        handles.append(
            plt.Line2D([], [], color=REFERENCE_COLOUR, lw=2.2, label=reference_label)
        )
    fig.legend(handles=handles, loc="upper right", fontsize=13, frameon=False)

    if title:
        fig.suptitle(title, fontsize=15)

    return fig


def plot_from_run(target: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Load a completed run's samples and write its triangle plot."""
    from bayesian_inversion.observations import load_observation

    output_dir = results_dir / target
    samples_path = output_dir / "samples_flat.npy"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No samples at {samples_path}; run run_inversion.py --target {target} first"
        )

    samples = np.load(samples_path)
    truth = load_observation(target).truth

    # Wiser et al. retrieved the real planet, so their values are a meaningful
    # comparison for `observed` only -- overlaying them on the synthetic sample81
    # design point would invite a comparison that means nothing.
    reference = reference_vector(WISER_2025) if target == "observed" else None

    fig = plot_corner(
        samples,
        truth=truth,
        reference=reference,
        reference_label="Wiser et al. (2025), approx.",
        title=f"Posterior over atmospheric parameters -- {target} ({len(samples):,} samples)",
    )

    path = output_dir / "posterior_corner.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["sample81", "observed"], required=True)
    args = parser.parse_args()

    print(f"written to {plot_from_run(args.target)}")
