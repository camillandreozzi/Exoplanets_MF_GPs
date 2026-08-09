"""Exploratory analysis of the INPUT parameters (the Xs) only.

We ignore the spectra here and just look at the atmospheric parameters
    Kzz, Rp, Tint, C, N, O, S, logg, f
to answer three simple questions:

    1. How is each parameter distributed?          -> marginal histograms
    2. How do the parameters relate to each other?  -> pairwise scatter matrix
    3. How correlated are they?                     -> correlation heatmaps
    4. Is the design "spatially" structured?        -> empirical variograms

We look at both designs:
    - XHF      : the 97-point high-fidelity design (small)
    - XLF_10k  : the 10,000-point low-fidelity design (large, decoupled)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from exoplanets_mf.data import load_XHF, load_XLF_10k
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

FIG_DIR = EXPLORATORY_RESULTS_DIR


# ---------------------------------------------------------------------------
# 1. Marginal distributions
# ---------------------------------------------------------------------------
def plot_marginals(X, name, savepath, color="tab:blue"):
    """One histogram per parameter, laid out on a grid."""
    cols = X.columns
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, c in zip(axes.flat, cols):
        ax.hist(X[c], bins=30, color=color, alpha=0.8)
        ax.set_title(c)
    for ax in axes.flat[n:]:          # hide unused panels
        ax.axis("off")
    fig.suptitle(f"Marginal distributions of inputs — {name}  (n={len(X)})")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Pairwise relationships (scatter matrix)
# ---------------------------------------------------------------------------
def plot_pairs(X, name, savepath, max_points=2000):
    """Lower-triangle scatter matrix; diagonal shows histograms.

    For the 10k design we subsample for legibility (a scatter of 10k x 10k
    panels is just an ink blob; 2,000 points show the same shape).
    """
    if len(X) > max_points:
        X = X.sample(max_points, random_state=RANDOM_SEED)
    cols = X.columns
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(X[ci], bins=20, color="tab:gray")
            elif i > j:                # lower triangle only
                ax.scatter(X[cj], X[ci], s=3, alpha=0.25, color="tab:blue")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(cj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Pairwise relationships of inputs — {name}")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Correlation heatmaps
# ---------------------------------------------------------------------------
def plot_correlation(X, name, savepath):
    """Pearson (linear) and Spearman (rank/monotone) correlation, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, method in zip(axes, ["pearson", "spearman"]):
        C = X.corr(method=method)
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(C)))
        ax.set_yticks(range(len(C)))
        ax.set_xticklabels(C.columns, rotation=45, ha="right")
        ax.set_yticklabels(C.columns)
        # annotate each cell with its value
        for i in range(len(C)):
            for j in range(len(C)):
                ax.text(j, i, f"{C.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(C.iloc[i, j]) < 0.5 else "white")
        ax.set_title(f"{method.capitalize()} correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Input correlations — {name}")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return X.corr()


# ---------------------------------------------------------------------------
# 4. Empirical variograms
# ---------------------------------------------------------------------------
def plot_variograms(X, name, savepath, n_bins=15, max_points=2000):
    """Omnidirectional empirical semivariogram of each input parameter.

    For every pair of design points (p, q) we measure:
        - their separation h = Euclidean distance in the STANDARDISED input
          space (all parameters z-scored so each contributes equally), and
        - the squared difference of a single parameter, (x_j(p) - x_j(q))^2.

    The semivariogram is the average squared difference within each distance
    bin:  gamma_j(h) = 0.5 * mean[ (x_j(p) - x_j(q))^2 ]  over pairs at lag h.

    Intuition: gamma starts near 0 (close points are similar) and rises to a
    "sill" equal to the parameter's variance. A flat, fast-rising curve means
    the design is space-filling with no leftover spatial structure (what you
    want for a good GP design); a slow rise or dips signal clustering/trends.
    """
    if len(X) > max_points:
        X = X.sample(max_points, random_state=RANDOM_SEED)

    # z-score so the separation distance is not dominated by large-range params
    Z = (X - X.mean()) / X.std()
    h = pdist(Z.values)                       # pairwise distance in input space

    edges = np.linspace(0, h.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(h, edges) - 1, 0, n_bins - 1)

    cols = X.columns
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))

    for ax, c in zip(axes.flat, cols):
        diff_sq = pdist(X[[c]].values) ** 2   # squared diff of this param only
        gamma = np.array([
            0.5 * diff_sq[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
            for b in range(n_bins)
        ])
        ax.plot(centers, gamma, "o-", color="tab:blue", ms=4)
        ax.axhline(X[c].var(), ls="--", color="tab:red", lw=1, label="variance (sill)")
        ax.set_title(c)
        ax.set_xlabel("separation h (std. input space)")
        ax.set_ylabel(r"$\gamma(h)$")
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Empirical variograms of inputs — {name}")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(X, name, tag):
    """Produce all four figures for one design and print its correlation."""
    print(f"\n=== {name}  (n={len(X)}, {X.shape[1]} params) ===")
    print(X.describe().T[["mean", "std", "min", "max"]].round(3))

    plot_marginals(X, name, FIG_DIR / f"02_{tag}_marginals.png")
    plot_pairs(X, name, FIG_DIR / f"02_{tag}_pairs.png")
    plot_correlation(X, name, FIG_DIR / f"02_{tag}_correlation.png")
    plot_variograms(X, name, FIG_DIR / f"02_{tag}_variograms.png")
    print(f"  -> figures written to {FIG_DIR}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run(load_XHF(), "XHF design", "xhf")
    run(load_XLF_10k(), "XLF_10k design", "xlf10k")


if __name__ == "__main__":
    main()
