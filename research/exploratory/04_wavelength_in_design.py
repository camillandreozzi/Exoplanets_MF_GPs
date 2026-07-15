"""Parallel check: should wavelength go INSIDE the design matrix?

Scripts 02/03 treat wavelength as the OUTPUT index: a spectrum is 195 separate
outputs and lambda never enters as a predictor. The alternative is a single
scalar GP  y = f(X, lambda)  with lambda appended as a 10th input column (rows =
samples x wavelengths). This script gathers the two diagnostics that decide
whether that representation is sensible, as a parallel to 02/03:

    A. Wavelength-wavelength correlation of the spectra.
       If neighbouring wavelengths are highly correlated and the correlation
       decays smoothly with separation, lambda behaves like a continuous input
       with a usable length scale (a kernel over lambda is meaningful). Sharp
       block structure flags molecular bands -> a *non-stationary* lambda kernel.

    B. First-order sensitivity with lambda in the design.
       Pool all (sample, wavelength) pairs into one scalar model y(X, lambda) and
       compute the same model-free first-order index as 03, now including lambda.
       This shows how much of the *pooled* variance lambda carries relative to the
       atmospheric inputs -- i.e. how strong an axis lambda is once flattened in.

Run on the 10k decoupled LF design (XLF_10k / YLF_10k).
"""

import matplotlib.pyplot as plt
import numpy as np

from exoplanets_mf.data import load_XLF_10k, load_YLF_10k, load_all
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

FIG_DIR = EXPLORATORY_RESULTS_DIR

N_BINS = 20
N_SAMPLES_FLAT = 3000   # samples pooled for the flattened sensitivity (x195 rows)


# ---------------------------------------------------------------------------
# A. Wavelength-wavelength correlation
# ---------------------------------------------------------------------------
def wavelength_correlation(Y, wl, savepath):
    """195x195 correlation heatmap + correlation-vs-separation decay."""
    C = np.corrcoef(Y.T)                       # (w, w) across the 10k samples

    # decay of correlation with wavelength separation
    iu = np.triu_indices_from(C, k=1)
    lag = np.abs(wl[iu[0]] - wl[iu[1]])
    rho = C[iu]
    n_bins = 40
    edges = np.linspace(0, lag.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(lag, edges) - 1, 0, n_bins - 1)
    mean_rho = np.array([rho[idx == b].mean() if np.any(idx == b) else np.nan
                         for b in range(n_bins)])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(C, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=[wl.min(), wl.max(), wl.min(), wl.max()])
    ax[0].set(xlabel=r"$\lambda$ [$\mu$m]", ylabel=r"$\lambda'$ [$\mu$m]",
              title="Wavelength-wavelength correlation of spectra")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    ax[1].plot(centers, mean_rho, "o-", color="tab:blue", ms=4)
    ax[1].axhline(0, ls=":", color="gray", lw=1)
    ax[1].axhline(0.5, ls="--", color="tab:red", lw=1, label=r"$\rho=0.5$")
    ax[1].set(xlabel=r"wavelength separation $|\lambda-\lambda'|$ [$\mu$m]",
              ylabel=r"mean correlation $\rho$",
              title="Correlation decay (implied $\\lambda$ length scale)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

    # nearest-neighbour correlation and lag where mean rho first drops below 0.5
    nn = np.diag(C, k=1)
    below = np.where(mean_rho < 0.5)[0]
    lag_half = centers[below[0]] if len(below) else np.nan
    return C, nn.mean(), lag_half


# ---------------------------------------------------------------------------
# B. First-order index with lambda flattened into the design
# ---------------------------------------------------------------------------
def first_order_scalar(X, y, n_bins=N_BINS):
    """Vector of first-order indices S_j = Var(E[y|x_j]) / Var(y), per column."""
    n, p = X.shape
    var_y = y.var()
    S = np.zeros(p)
    for j in range(p):
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(X[:, j], edges) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        sums = np.bincount(idx, weights=y, minlength=n_bins)
        bin_means = sums / counts
        S[j] = (counts * (bin_means - y.mean()) ** 2).sum() / n / var_y
    return S


def flatten_with_lambda(
    X, Y, wl, n_samples=N_SAMPLES_FLAT, seed=RANDOM_SEED
):
    """Build the pooled scalar design [X..., lambda] -> y."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    w = len(wl)
    Xrep = np.repeat(X[sub], w, axis=0)                 # (n*w, p)
    lam = np.tile(wl, len(sub))[:, None]                # (n*w, 1)
    Xflat = np.hstack([Xrep, lam])
    yflat = Y[sub].ravel()                              # row-major matches Xrep
    return Xflat, yflat


def plot_flat_ranking(names, S, savepath):
    order = np.argsort(S)[::-1]
    y = np.arange(len(names))
    colors = ["tab:orange" if names[j] == "lambda" else "tab:blue"
              for j in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y, S[order], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(np.array(names)[order])
    ax.invert_yaxis()
    ax.set_xlabel(r"first-order index $S_j$ (pooled scalar model)")
    ax.set_title(r"Sensitivity with $\lambda$ in the design matrix"
                 "\n(flattened $y(X,\\lambda)$, XLF_10k / YLF_10k)")
    for i, v in enumerate(S[order]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return order


# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X = load_XLF_10k().to_numpy()
    Y = load_YLF_10k().to_numpy()
    wl = load_all()["wavelengths"]
    names = list(load_XLF_10k().columns)

    # A. wavelength correlation
    C, nn_corr, lag_half = wavelength_correlation(
        Y, wl, FIG_DIR / "04_wavelength_correlation.png")
    print(f"A. wavelength-wavelength correlation:")
    print(f"   mean nearest-neighbour corr = {nn_corr:.3f}")
    print(f"   mean corr drops below 0.5 at separation ~ {lag_half:.2f} um")

    # B. flattened sensitivity including lambda
    Xflat, yflat = flatten_with_lambda(X, Y, wl)
    S = first_order_scalar(Xflat, yflat)
    flat_names = names + ["lambda"]
    order = plot_flat_ranking(flat_names, S, FIG_DIR / "04_flat_sensitivity.png")
    print(f"\nB. first-order index in pooled model y(X, lambda)  "
          f"(rows = {len(yflat):,}):")
    for j in order:
        print(f"   {flat_names[j]:7s}  {S[j]:.3f}")
    print(f"\n   sum = {S.sum():.3f}")
    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
