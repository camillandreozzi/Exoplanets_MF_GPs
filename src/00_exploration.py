"""Exploratory plots and correlation between high- and low-fidelity spectra."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from read_in import load_all

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _lf_band(ax, wl, YLF, color="tab:orange", band=(5, 95)):
    """Draw the 10k LF distribution as a shaded percentile band + median line.

    With 10k samples, plotting every spectrum is an opaque blob; a percentile
    envelope conveys the same distributional spread far more legibly.
    """
    lo, med, hi = np.percentile(YLF, [band[0], 50, band[1]], axis=0)
    ax.fill_between(wl, lo, hi, color=color, alpha=0.25, lw=0,
                    label=f"LF {band[0]}–{band[1]}% (n={YLF.shape[0]})")
    ax.plot(wl, med, color=color, lw=2, label="LF median")


def plot_hf_lf(wl, YHF, YLF, savepath):
    """Overlay HF samples (translucent + mean) with the 10k LF percentile band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _lf_band(ax, wl, YLF)
    ax.plot(wl, YHF.T, color="tab:blue", alpha=0.15, lw=0.6)
    ax.plot(wl, YHF.mean(axis=0), color="tab:blue", lw=2, label="HF mean")
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Eclipse depth")
    ax.set_title("High- vs low-fidelity spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_hf_lf_observed(wl, YHF, YLF, obs, savepath):
    """HF samples + 10k LF band with observed spectrum and asymmetric error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _lf_band(ax, wl, YLF)
    ax.plot(wl, YHF.T, color="tab:blue", alpha=0.10, lw=0.5)
    ax.plot(wl, YHF.mean(axis=0), color="tab:blue", lw=2, label="HF mean")

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
    """Pearson correlation between HF and LF, computed per-sample and per-wavelength.

    Requires YHF and YLF to be *input-matched* (row i = same atmosphere). Use the
    paired 97-row LF set here, not the decoupled 10k design.
    """
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


def detect_channel_boundaries(wl):
    """Return wavelengths where the sampling step changes (channel breaks).

    Detected as midpoints between consecutive wavelengths whose spacing
    differs from the previous step by more than 50%.
    """
    d = np.diff(wl)
    breaks = []
    for i in range(1, len(d)):
        if abs(d[i] - d[i - 1]) / max(d[i - 1], 1e-12) > 0.5:
            breaks.append(0.5 * (wl[i] + wl[i + 1]))
    return np.array(breaks)


def annotate_channels(ax, wl, boundaries, labels=("Ch1", "Ch2")):
    """Draw vertical separators and shaded backgrounds for each channel."""
    edges = np.concatenate(([wl.min()], boundaries, [wl.max()]))
    colors = ["#e8f1fb", "#fbecdc", "#eaf6ea", "#f4e8f7"]
    for i in range(len(edges) - 1):
        ax.axvspan(edges[i], edges[i + 1],
                   color=colors[i % len(colors)], alpha=0.4, zorder=0)
        if i < len(labels):
            ax.text(0.5 * (edges[i] + edges[i + 1]), 0.97, labels[i],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="dimgray")
    for b in boundaries:
        ax.axvline(b, color="k", lw=1.0, ls="--", alpha=0.7)


def plot_correlation(wl, YHF, YLF, per_sample, per_wavelength, global_corr,
                     savepath):
    boundaries = detect_channel_boundaries(wl)
    # Per-sample correlation split by channel.
    edges = np.concatenate(([wl.min() - 1e-9], boundaries, [wl.max() + 1e-9]))
    channel_masks = [
        (wl > edges[i]) & (wl <= edges[i + 1]) for i in range(len(edges) - 1)
    ]
    channel_labels = [f"Ch{i + 1}" for i in range(len(channel_masks))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- per-sample histogram, split by channel ---
    bins = np.linspace(min(0.0, per_sample.min()), 1.0, 25)
    axes[0].hist(per_sample, bins=bins, color="lightgray",
                 edgecolor="white", label=f"All wl (mean={per_sample.mean():.3f})")
    for mask, label, color in zip(channel_masks, channel_labels,
                                  ["tab:blue", "tab:orange", "tab:green"]):
        r_ch = np.array([np.corrcoef(YHF[i, mask], YLF[i, mask])[0, 1]
                         for i in range(YHF.shape[0])])
        axes[0].hist(r_ch, bins=bins, histtype="step", lw=2, color=color,
                     label=f"{label} (mean={r_ch.mean():.3f})")
    axes[0].set_xlabel("Pearson r (HF vs LF) per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-sample correlation by channel")
    axes[0].legend(fontsize=8)

    # --- per-wavelength correlation, one dot per wavelength ---
    annotate_channels(axes[1], wl, boundaries, labels=channel_labels)
    axes[1].plot(wl, per_wavelength, color="tab:purple", lw=1, alpha=0.6,
                 zorder=2)
    axes[1].scatter(wl, per_wavelength, s=18, color="tab:purple",
                    edgecolor="white", linewidth=0.4, zorder=3)
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
    YLF = data["YLF"].to_numpy()          # 97 rows, input-matched to HF
    YLF_10k = data["YLF_10k"].to_numpy()  # 10k decoupled LF design
    obs = data["observed"]

    print(f"HF spectra: {YHF.shape},  LF spectra (paired): {YLF.shape},  "
          f"LF spectra (10k): {YLF_10k.shape}")

    # Overlay / distribution plots use the large 10k LF design.
    plot_hf_lf(wl, YHF, YLF_10k, FIG_DIR / "00_hf_vs_lf.png")
    plot_hf_lf_observed(wl, YHF, YLF_10k, obs, FIG_DIR / "00_hf_lf_observed.png")

    # Row-paired correlation requires input-matched LF -> use the 97-row set.
    per_sample, per_wavelength, global_corr = correlation_hf_lf(YHF, YLF)
    plot_correlation(wl, YHF, YLF, per_sample, per_wavelength, global_corr,
                     FIG_DIR / "00_hf_lf_correlation.png")

    print(f"global  r(HF, LF) = {global_corr:.4f}")
    print(f"per-sample      r : mean={per_sample.mean():.4f}, "
          f"min={per_sample.min():.4f}, max={per_sample.max():.4f}")
    print(f"per-wavelength  r : mean={per_wavelength.mean():.4f}, "
          f"min={per_wavelength.min():.4f}, max={per_wavelength.max():.4f}")
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
