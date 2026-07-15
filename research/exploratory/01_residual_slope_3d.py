"""3D per-wavelength HF residual vs standardized-LF slope diagnostic.

For every wavelength bin j we standardise the input-matched LF values and
look at the local linear relationship between them:

    standardized LF             z_LF[i,j] = (YLF[i,j] - mu_LF[j]) / sigma_LF[j]
    HF residual (LF-std units)   r_HF[i,j] = (YHF[i,j] - mu_HF[j]) / sigma_LF[j]
    per-bin slope                rho[j]    = OLS slope of r_HF on z_LF
                                           = sum(r_HF * z_LF) / sum(z_LF^2)

Because both coordinates are zero-mean per bin, rho[j] is the through-origin
regression slope: the AR(1) / co-kriging scaling that maps standardized LF to
the expected HF residual at that wavelength. Each comb line is z = rho[j] * y
drawn in the (standardized LF, HF residual) plane at x = lambda_j; the scatter
is the raw matched cloud. Both are coloured by rho[j]. The two panels are two
viewing angles of the same 3D axes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from exoplanets_mf.data import load_all
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR

FIG_DIR = EXPLORATORY_RESULTS_DIR

CMAP = plt.get_cmap("plasma")
# Two viewing angles (elev, azim) for the left / right panels.
VIEWS = [(16, -60), (10, -86)]


def residual_slopes(YHF, YLF):
    """Return standardized LF, HF residuals, and per-bin slopes.

    YHF, YLF : (n_samples, n_wavelengths) arrays, input-matched row-by-row.
    """
    mu_LF = YLF.mean(axis=0)
    sd_LF = YLF.std(axis=0)
    mu_HF = YHF.mean(axis=0)

    z_LF = (YLF - mu_LF) / sd_LF                # standardized LF
    r_HF = (YHF - mu_HF) / sd_LF                # LF-std units
    # Through-origin OLS slope per wavelength bin.
    rho = (r_HF * z_LF).sum(axis=0) / (z_LF * z_LF).sum(axis=0)
    return z_LF, r_HF, rho


def _draw(ax, wl, z_LF, r_HF, rho, norm):
    # Faint raw matched cloud, coloured by the bin slope.
    lam = np.repeat(wl[None, :], z_LF.shape[0], axis=0)
    ax.scatter(lam.ravel(), z_LF.ravel(), r_HF.ravel(),
               c=np.repeat(rho[None, :], z_LF.shape[0], axis=0).ravel(),
               cmap=CMAP, norm=norm, s=2, alpha=0.10, lw=0, depthshade=False)

    # One comb line per wavelength bin: z = rho * y across the LF span.
    for j in range(len(wl)):
        y0, y1 = z_LF[:, j].min(), z_LF[:, j].max()
        ys = np.array([y0, y1])
        ax.plot(np.full(2, wl[j]), ys, rho[j] * ys,
                color=CMAP(norm(rho[j])), lw=1.2)

    ax.set_xlabel(r"wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel("standardized LF")
    ax.set_zlabel("HF residual  (LF-std units)")
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)


def plot_residual_slope_3d(wl, YHF, YLF, savepath):
    z_LF, r_HF, rho = residual_slopes(YHF, YLF)
    norm = Normalize(vmin=rho.min(), vmax=rho.max())

    fig = plt.figure(figsize=(15, 6))
    for k, view in enumerate(VIEWS):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        _draw(ax, wl, z_LF, r_HF, rho, norm)
        ax.view_init(elev=view[0], azim=view[1])

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.7, pad=0.02)
    cbar.set_label(r"per-bin slope $\rho_j$")

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wl = data["wavelengths"]
    YHF = data["YHF"].to_numpy()
    YLF = data["YLF"].to_numpy()   # 97 rows, input-matched to HF

    rho = plot_residual_slope_3d(
        wl, YHF, YLF, FIG_DIR / "01_residual_slope_3d.png"
    )
    print(f"per-bin slope rho: mean={rho.mean():.3f}, "
          f"min={rho.min():.3f}, max={rho.max():.3f}")
    print(f"figure saved to {FIG_DIR / '01_residual_slope_3d.png'}")


if __name__ == "__main__":
    main()
