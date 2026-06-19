"""3D per-wavelength HF/LF residual-slope diagnostic for the HF-matched data.

For every wavelength bin j we standardise the input-matched HF/LF residuals and
look at the local linear relationship between them:

    LF residual (std units)      r_LF[i,j] = (YLF[i,j] - mu_LF[j]) / sigma_LF[j]
    HF residual (LF-std units)   r_HF[i,j] = (YHF[i,j] - mu_HF[j]) / sigma_LF[j]
    per-bin slope                rho[j]    = OLS slope of r_HF on r_LF
                                           = sum(r_HF * r_LF) / sum(r_LF^2)

Because both residuals are zero-mean per bin, rho[j] is the through-origin
regression slope: the AR(1) / co-kriging scaling that maps an LF residual to the
expected HF residual at that wavelength. Each comb line is z = rho[j] * y drawn
in the (LF, HF) residual plane at x = lambda_j; the scatter is the raw residual
cloud. Both are coloured by rho[j]. The two panels are two viewing angles of the
same 3D axes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from read_in import load_all

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CMAP = plt.get_cmap("plasma")
# Two viewing angles (elev, azim) for the left / right panels.
VIEWS = [(16, -60), (10, -86)]


def residual_slopes(YHF, YLF):
    """Return standardised residuals and per-bin slopes for matched HF/LF data.

    YHF, YLF : (n_samples, n_wavelengths) arrays, input-matched row-by-row.
    """
    mu_LF = YLF.mean(axis=0)
    sd_LF = YLF.std(axis=0)
    mu_HF = YHF.mean(axis=0)

    r_LF = (YLF - mu_LF) / sd_LF                # std units
    r_HF = (YHF - mu_HF) / sd_LF                # LF-std units
    # Through-origin OLS slope per wavelength bin.
    rho = (r_HF * r_LF).sum(axis=0) / (r_LF * r_LF).sum(axis=0)
    return r_LF, r_HF, rho


def _draw(ax, wl, r_LF, r_HF, rho, norm):
    # Faint raw residual cloud, coloured by the bin slope.
    lam = np.repeat(wl[None, :], r_LF.shape[0], axis=0)
    ax.scatter(lam.ravel(), r_LF.ravel(), r_HF.ravel(),
               c=np.repeat(rho[None, :], r_LF.shape[0], axis=0).ravel(),
               cmap=CMAP, norm=norm, s=2, alpha=0.10, lw=0, depthshade=False)

    # One comb line per wavelength bin: z = rho * y across the LF-residual span.
    for j in range(len(wl)):
        y0, y1 = r_LF[:, j].min(), r_LF[:, j].max()
        ys = np.array([y0, y1])
        ax.plot(np.full(2, wl[j]), ys, rho[j] * ys,
                color=CMAP(norm(rho[j])), lw=1.2)

    ax.set_xlabel(r"wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel("LF residual (std units)")
    ax.set_zlabel("HF residual  (LF-std units)")
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)


def plot_residual_slope_3d(wl, YHF, YLF, savepath):
    r_LF, r_HF, rho = residual_slopes(YHF, YLF)
    norm = Normalize(vmin=rho.min(), vmax=rho.max())

    fig = plt.figure(figsize=(15, 6))
    for k, view in enumerate(VIEWS):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        _draw(ax, wl, r_LF, r_HF, rho, norm)
        ax.view_init(elev=view[0], azim=view[1])

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.7, pad=0.02)
    cbar.set_label(r"per-bin slope $\rho_j$")

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho


def main():
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
