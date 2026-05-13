"""Exploratory plots and correlation between high- and low-fidelity spectra."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from read_in import load_all

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_hf_lf(wl, YHF, YLF, savepath):
    """Overlay HF and LF spectra (all samples translucent + mean bold)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl, YLF.T, color="tab:orange", alpha=0.15, lw=0.6)
    ax.plot(wl, YHF.T, color="tab:blue",   alpha=0.15, lw=0.6)
    ax.plot(wl, YLF.mean(axis=0), color="tab:orange", lw=2, label="LF mean")
    ax.plot(wl, YHF.mean(axis=0), color="tab:blue",   lw=2, label="HF mean")
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Eclipse depth")
    ax.set_title("High- vs low-fidelity spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_hf_lf_observed(wl, YHF, YLF, obs, savepath):
    """HF + LF means with observed spectrum and its asymmetric error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl, YLF.T, color="tab:orange", alpha=0.10, lw=0.5)
    ax.plot(wl, YHF.T, color="tab:blue",   alpha=0.10, lw=0.5)
    ax.plot(wl, YLF.mean(axis=0), color="tab:orange", lw=2, label="LF mean")
    ax.plot(wl, YHF.mean(axis=0), color="tab:blue",   lw=2, label="HF mean")

    # measured_err_lo / measured_err_hi are asymmetric error magnitudes
    # (always positive), not absolute bounds.
    yerr = np.vstack([obs["measured_err_lo"], obs["measured_err_hi"]])
    ax.errorbar(
        obs["wavelength"], obs["measured_eclipse_depth"], yerr=yerr,
        fmt="o", ms=3, color="black", ecolor="gray", elinewidth=0.8, capsize=2,
        label="Observed",
    )
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Eclipse depth")
    ax.set_title("HF / LF / Observed spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def correlation_hf_lf(YHF, YLF):
    """Pearson correlation between HF and LF, computed per-sample and per-wavelength."""
    # Per-sample: how well does the LF prediction track HF across wavelength
    # for the same input X?
    per_sample = np.array([
        np.corrcoef(YHF[i], YLF[i])[0, 1] for i in range(YHF.shape[0])
    ])
    # Per-wavelength: at a given wavelength, do HF/LF agree across the sample
    # set?
    per_wavelength = np.array([
        np.corrcoef(YHF[:, j], YLF[:, j])[0, 1] for j in range(YHF.shape[1])
    ])
    # Global: flatten everything.
    global_corr = np.corrcoef(YHF.ravel(), YLF.ravel())[0, 1]
    return per_sample, per_wavelength, global_corr


def plot_correlation(wl, per_sample, per_wavelength, global_corr, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(per_sample, bins=20, color="tab:blue", edgecolor="white")
    axes[0].set_xlabel("Pearson r (HF vs LF) per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Per-sample correlation\nmean={per_sample.mean():.3f}")

    axes[1].plot(wl, per_wavelength, color="tab:purple")
    axes[1].set_xlabel("Wavelength [um]")
    axes[1].set_ylabel("Pearson r (HF vs LF) across samples")
    axes[1].set_title(f"Per-wavelength correlation\nglobal r = {global_corr:.3f}")

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    data = load_all()
    wl = data["wavelengths"]
    YHF = data["YHF"].to_numpy()
    YLF = data["YLF"].to_numpy()
    obs = data["observed"]

    print(f"HF spectra: {YHF.shape},  LF spectra: {YLF.shape}")

    plot_hf_lf(wl, YHF, YLF, FIG_DIR / "00_hf_vs_lf.png")
    plot_hf_lf_observed(wl, YHF, YLF, obs, FIG_DIR / "00_hf_lf_observed.png")

    per_sample, per_wavelength, global_corr = correlation_hf_lf(YHF, YLF)
    plot_correlation(wl, per_sample, per_wavelength, global_corr,
                     FIG_DIR / "00_hf_lf_correlation.png")

    print(f"global  r(HF, LF) = {global_corr:.4f}")
    print(f"per-sample      r : mean={per_sample.mean():.4f}, "
          f"min={per_sample.min():.4f}, max={per_sample.max():.4f}")
    print(f"per-wavelength  r : mean={per_wavelength.mean():.4f}, "
          f"min={per_wavelength.min():.4f}, max={per_wavelength.max():.4f}")
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
