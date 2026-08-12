"""Exploratory plots and correlation between high- and low-fidelity spectra."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR

FIG_DIR = EXPLORATORY_RESULTS_DIR

# Round wavelengths (µm) that make natural major ticks on a log axis; the
# helper keeps only those inside the plotted range.
_WAVELENGTH_TICKS = np.array(
    [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]
)


def _decimal_tick_label(value, _pos=None) -> str:
    """Plain-decimal tick label: 0.001, 0.01, 1e-06 -> '0.000001'.

    ScalarFormatter cannot do this below 1e-4 (it rounds every such tick to
    '0.0000'), so small-valued log axes need their own formatter.
    """
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _log_axis_limits(values, margin=0.05):
    """Data limits padded by `margin` of the log10 span, as matplotlib would.

    Log axes here are set *after* the artists are drawn, and matplotlib's
    autoscale then mixes log-space and data-space extents whenever a
    Collection (our fill_between band) sits on an already-log axis -- on a
    shared x-axis that silently stretched the range down by a decade. Setting
    the limits explicitly from the data makes the framing independent of draw
    order and of axis sharing.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = margin * np.log10(hi / lo)
    return lo * 10.0**-pad, hi * 10.0**pad


def set_log_wavelength_axis(ax, wl) -> None:
    """Log-scale the wavelength x-axis with plain µm tick labels (not 10^x).

    The standard emission-spectrum presentation: the wide µm range reads far
    better on a log axis, but the labels stay original-scale numbers. Minor
    ticks keep their marks but drop their labels to avoid clutter.
    """
    ax.set_xscale("log")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ticks = _WAVELENGTH_TICKS[
        (_WAVELENGTH_TICKS >= np.min(wl)) & (_WAVELENGTH_TICKS <= np.max(wl))
    ]
    if ticks.size:
        ax.set_xticks(ticks)
    ax.set_xlim(*_log_axis_limits(wl))


def set_log_depth_axis(ax) -> None:
    """Log-scale the eclipse-depth y-axis, keeping original-unit tick labels.

    Depths span ~4 orders of magnitude, so the log view is the informative
    one -- but the reader should read 0.001 off the axis, not a log10 value
    of -3 that they have to exponentiate in their head. Plot the data in
    original units and let the axis do the transform.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_tick_label))
    ax.yaxis.set_minor_formatter(NullFormatter())


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
    set_log_wavelength_axis(ax, wl)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Eclipse depth")
    ax.set_title("High- vs low-fidelity spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def draw_hf_lf_observed(ax, wl, YHF, YLF, obs, *, ylabel="Eclipse depth",
                        title="HF / LF / Observed spectra",
                        hf_center=None, hf_label="HF mean", log_depth=False):
    """Draw one HF/LF/observed panel onto a provided Axes.

    Data are always plotted in original eclipse-depth units; `log_depth`
    switches the *axis* to log scale rather than transforming the values
    (see research/exploratory/05_log_spectra_exploration.py). `hf_center`
    overrides the HF summary line -- the log view centres on the geometric
    mean, which the caller computes -- keeping transformation out of the
    plotting code.
    """
    _lf_band(ax, wl, YLF)
    ax.plot(wl, YHF.T, color="tab:blue", alpha=0.10, lw=0.5)
    if hf_center is None:
        hf_center = YHF.mean(axis=0)
    ax.plot(wl, hf_center, color="tab:blue", lw=2, label=hf_label)

    # measured_err_lo / measured_err_hi are asymmetric error magnitudes
    # (always positive), not absolute bounds.
    yerr = np.vstack([obs["measured_err_lo"], obs["measured_err_hi"]])
    ax.errorbar(
        obs["wavelength"], obs["measured_eclipse_depth"], yerr=yerr,
        fmt="o", ms=3, color="black", ecolor="gray", elinewidth=0.8, capsize=2,
        label="Observed",
    )
    set_log_wavelength_axis(ax, wl)
    if log_depth:
        set_log_depth_axis(ax)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_hf_lf_observed(wl, YHF, YLF, obs, savepath):
    """HF samples + 10k LF band with observed spectrum and asymmetric error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_hf_lf_observed(ax, wl, YHF, YLF, obs)
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


def correlations_by_instrument(wl, YHF, YLF):
    """Return per-sample HF/LF correlations for the three instrument modes."""
    correlations = []
    for mode, mask in instrument_mode_masks(wl):
        r_mode = np.array([
            np.corrcoef(YHF[i, mask], YLF[i, mask])[0, 1]
            for i in range(YHF.shape[0])
        ])
        correlations.append((mode, r_mode))
    return tuple(correlations)


def annotate_instrument_modes(ax, wl):
    """Shade and label the three nominal instrument-mode wavelength ranges."""
    wl_min, wl_max = np.min(wl), np.max(wl)
    assignments = instrument_mode_masks(wl)

    for mode, _ in assignments:
        left = max(mode.wavelength_min, wl_min)
        right = min(mode.wavelength_max, wl_max)
        ax.axvspan(left, right, color=mode.color, alpha=0.4, zorder=0)
        ax.text(
            0.5 * (left + right),
            0.97,
            mode.plot_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
        )

    for mode, _ in assignments[1:]:
        ax.axvline(
            mode.wavelength_min, color="k", lw=1.0, ls="--", alpha=0.7
        )


def plot_correlation(wl, YHF, YLF, per_sample, per_wavelength, global_corr,
                     savepath):
    mode_correlations = correlations_by_instrument(wl, YHF, YLF)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- per-sample histogram, split by instrument mode ---
    correlation_min = min(
        per_sample.min(), *(values.min() for _, values in mode_correlations)
    )
    bins = np.linspace(min(0.0, correlation_min), 1.0, 30)
    axes[0].hist(per_sample, bins=bins, color="lightgray",
                 edgecolor="white", label=f"All wl (mean={per_sample.mean():.3f})")
    line_colors = ["tab:blue", "tab:orange", "tab:green"]
    for (mode, values), color in zip(mode_correlations, line_colors):
        axes[0].hist(values, bins=bins, histtype="step", lw=2, color=color,
                     label=f"{mode.label} (mean={values.mean():.3f})")
    axes[0].set_xlabel("Pearson r (HF vs LF) per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-sample correlation by instrument mode")
    axes[0].legend(fontsize=8)

    # --- per-wavelength correlation, one dot per wavelength ---
    annotate_instrument_modes(axes[1], wl)
    axes[1].plot(wl, per_wavelength, color="tab:purple", lw=1, alpha=0.6,
                 zorder=2)
    axes[1].scatter(wl, per_wavelength, s=18, color="tab:purple",
                    edgecolor="white", linewidth=0.4, zorder=3)
    set_log_wavelength_axis(axes[1], wl)
    axes[1].set_xlabel("Wavelength [um]")
    axes[1].set_ylabel("Pearson r (HF vs LF) across samples")
    axes[1].set_title(f"Per-wavelength correlation\nglobal r = {global_corr:.3f}")

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return mode_correlations


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
    mode_correlations = plot_correlation(
        wl,
        YHF,
        YLF,
        per_sample,
        per_wavelength,
        global_corr,
        FIG_DIR / "00_hf_lf_correlation.png",
    )

    print(f"global  r(HF, LF) = {global_corr:.4f}")
    print(f"per-sample      r : mean={per_sample.mean():.4f}, "
          f"min={per_sample.min():.4f}, max={per_sample.max():.4f}")
    for mode, values in mode_correlations:
        print(f"  {mode.label:15s}: mean={values.mean():.4f}, "
              f"min={values.min():.4f}, max={values.max():.4f}")
    print(f"per-wavelength  r : mean={per_wavelength.mean():.4f}, "
          f"min={per_wavelength.min():.4f}, max={per_wavelength.max():.4f}")
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
