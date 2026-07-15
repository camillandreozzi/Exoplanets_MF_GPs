"""Output-side global sensitivity analysis: which inputs drive the spectra?

The input-side EDA (02_input_exploration.py) showed the 9 parameters are mutually
independent and space-filling, so the input geometry alone offers no way to
reduce dimensionality. The only remaining lever is the *output*: do all 9 inputs
actually move the spectra, or are some effectively inert?

We answer this on the 10,000-point decoupled LF design (XLF_10k, YLF_10k), which
is large, independent and space-filling -- the ideal setting for global
sensitivity analysis (GSA). We ask four questions:

    1. How many numbers does a spectrum really carry?   -> PCA scree
    2. What fraction of spectral variance does each       -> first-order
       input explain on its own (main effect)?              Sobol indices
    3. Does that ranking survive once interactions are    -> random-forest
       allowed?                                              total-effect proxy
    4. Which input drives which part of the spectrum?     -> per-wavelength
                                                             sensitivity heatmap

First-order (main-effect) index, model-free:
    S_j = Var_{x_j}( E[y | x_j] ) / Var(y)
estimated by slicing input j into equal-count bins, taking the per-bin output
mean, and dividing the (count-weighted) variance of those bin means by the total
output variance. With 10k points this is low-bias. Aggregated over wavelengths
with weights Var(y_lambda), it reads as "fraction of total spectral variance
explained by the main effect of input j". The inputs sum to <= 1; the shortfall
is variance carried by interactions / higher-order effects.

NOTE: this characterises the *LF* model. It is the cheap design we have 10k of;
the HF sensitivity structure is expected to be similar but is not measured here
(only 97 HF points -- too few for GSA).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from exoplanets_mf.data import load_XLF_10k, load_YLF_10k, load_all
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

FIG_DIR = EXPLORATORY_RESULTS_DIR

N_BINS = 20          # equal-count slices per input for the main-effect estimator
N_PCA_KEEP = 10      # PCs fed to the random forest (captures ~all the signal)
RF_SUBSAMPLE = 4000  # rows used to fit the forest (speed)


# ---------------------------------------------------------------------------
# 1. Effective output dimensionality
# ---------------------------------------------------------------------------
def pca(Y):
    """Center the spectra and return (eigvals, components, scores)."""
    Yc = Y - Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    eig = (s ** 2) / (len(Y) - 1)        # variance per component
    scores = U * s                       # (n, k) projections
    return eig, Vt, scores, Yc


def plot_scree(eig, savepath):
    frac = eig / eig.sum()
    cum = np.cumsum(frac)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    k = np.arange(1, len(eig) + 1)
    ax[0].bar(k[:15], frac[:15], color="tab:blue")
    ax[0].set(xlabel="principal component", ylabel="variance fraction",
              title="Spectral PCA scree (first 15)")
    ax[1].plot(k[:15], cum[:15], "o-", color="tab:blue")
    ax[1].axhline(0.99, ls="--", color="tab:red", lw=1, label="99%")
    ax[1].set(xlabel="# components", ylabel="cumulative variance",
              title="Cumulative explained variance", ylim=(0, 1.02))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return frac, cum


# ---------------------------------------------------------------------------
# 2. First-order (main-effect) Sobol indices, per wavelength
# ---------------------------------------------------------------------------
def main_effect_per_wavelength(X, Y, n_bins=N_BINS):
    """Return S[input, wavelength] = first-order index of each input per bin.

    Equal-count (quantile) binning of each input; count-weighted variance of the
    per-bin output mean, normalised by the total output variance.
    """
    n, p = X.shape
    w = Y.shape[1]
    var_y = Y.var(axis=0)                          # (w,)
    var_y_safe = np.where(var_y > 0, var_y, np.nan)
    S = np.zeros((p, w))
    for j in range(p):
        # quantile edges -> equal-count bins
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(X[:, j], edges) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins).astype(float)  # (n_bins,)
        sums = np.zeros((n_bins, w))
        np.add.at(sums, idx, Y)
        bin_means = sums / counts[:, None]         # E[y | bin]
        grand = Y.mean(axis=0)
        # count-weighted variance of the conditional mean
        cond_var = (counts[:, None] * (bin_means - grand) ** 2).sum(0) / n
        S[j] = cond_var / var_y_safe
    return S, var_y


def aggregate(S, var_y):
    """Variance-weighted average over wavelengths -> one index per input."""
    return (S * var_y).sum(axis=1) / var_y.sum()


# ---------------------------------------------------------------------------
# 3. Model-based total-effect proxy (captures interactions)
# ---------------------------------------------------------------------------
def rf_total_effect(X, scores, eig, n_keep=N_PCA_KEEP, seed=RANDOM_SEED):
    """Random-forest impurity importance on the top PCs, eigenvalue-weighted.

    Unlike the first-order index, tree ensembles split on interactions too, so
    this is a total-effect-style ranking. Importances per output sum to 1; we
    weight each PC by its variance and aggregate over inputs.
    """
    keep = min(n_keep, scores.shape[1])
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(RF_SUBSAMPLE, len(X)), replace=False)
    # rf.feature_importances_ averages over outputs equally; instead weight PCs
    # by their variance by fitting per-PC importances.
    imp = np.zeros((keep, X.shape[1]))
    for k in range(keep):
        r = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=seed
        )
        r.fit(X[sub], scores[sub, k])
        imp[k] = r.feature_importances_
    w = eig[:keep] / eig[:keep].sum()
    return (w[:, None] * imp).sum(0)               # (p,) aggregated importance


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_rankings(names, S_first, rf_imp, savepath):
    order = np.argsort(S_first)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh(y, S_first[order], color="tab:blue")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(np.array(names)[order])
    ax[0].invert_yaxis()
    ax[0].set_title("First-order index\n(fraction of spectral variance, main effect)")
    ax[0].set_xlabel(r"$S_j$")
    for i, v in enumerate(S_first[order]):
        ax[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    ax[1].barh(y, rf_imp[order], color="tab:green")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(np.array(names)[order])
    ax[1].invert_yaxis()
    ax[1].set_title("Random-forest importance\n(total effect, incl. interactions)")
    ax[1].set_xlabel("relative importance")
    for i, v in enumerate(rf_imp[order]):
        ax[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle("Input -> spectrum sensitivity  (XLF_10k / YLF_10k)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return order


def plot_sensitivity_spectrum(wl, names, S, savepath):
    """Heatmap of first-order index: inputs (rows) x wavelength (cols)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(S, aspect="auto", origin="lower", cmap="magma",
                   extent=[wl.min(), wl.max(), 0, len(names)],
                   vmin=0, vmax=np.nanpercentile(S, 99))
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"wavelength $\lambda$ [$\mu$m]")
    ax.set_title("First-order sensitivity per wavelength  "
                 r"$S_j(\lambda)$  (XLF_10k / YLF_10k)")
    fig.colorbar(im, ax=ax, label=r"$S_j(\lambda)$")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X = load_XLF_10k().to_numpy()
    Y = load_YLF_10k().to_numpy()
    wl = load_all()["wavelengths"]
    names = list(load_XLF_10k().columns)

    print(f"X {X.shape}, Y {Y.shape}")

    # 1. output dimensionality
    eig, Vt, scores, Yc = pca(Y)
    frac, cum = plot_scree(eig, FIG_DIR / "03_pca_scree.png")
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"\nPCA: PC1={frac[0]:.3f}, PC1-3={cum[2]:.3f}, "
          f"#PCs for 99% var = {n99}")

    # 2. first-order indices
    S, var_y = main_effect_per_wavelength(X, Y)
    S_first = aggregate(S, var_y)

    # 3. random-forest total effect
    rf_imp = rf_total_effect(X, scores, eig)

    # ranked report
    order = np.argsort(S_first)[::-1]
    print("\ninput        first-order S_j   RF importance")
    for j in order:
        print(f"  {names[j]:5s}      {S_first[j]:8.3f}        {rf_imp[j]:8.3f}")
    print(f"\n  sum of first-order indices = {S_first.sum():.3f}  "
          f"(remainder = interactions / higher order)")

    plot_rankings(names, S_first, rf_imp, FIG_DIR / "03_sensitivity_ranking.png")
    plot_sensitivity_spectrum(wl, names, S, FIG_DIR / "03_sensitivity_spectrum.png")
    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
