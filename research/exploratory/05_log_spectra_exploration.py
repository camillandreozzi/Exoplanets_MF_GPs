"""Quick statistics and plots for log10-transformed spectra.

Motivates the research/log_modelling pipeline: eclipse depths span ~4 orders
of magnitude with strong right skew, which log10 largely removes. Prints and
saves summary statistics, and places the familiar hf_lf_observed panel side
by side with its log10 counterpart.

The observed spectrum contains negative depths (noise around small signals),
so the log panel drops those points and clips lower error bars that would
cross zero; the simulated spectra are strictly positive and transform cleanly.

Both panels plot depths in original units: the log panel differs only in its
log-scaled y-axis (ticks read 0.001, 0.01, ... rather than -3, -2) and in
summarising HF by the geometric mean, the centre the log10 models actually
fit.
"""

import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from exoplanets_mf.data import load_all
from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR
from exoplanets_mf.transforms import inverse_log10_spectra, log10_spectra

# "00_spectra_exploration" starts with a digit, so a plain import statement is
# invalid -- same importlib pattern as tests/test_mf_gp.py. Import-safe: the
# module only defines functions at import time.
spectra00 = importlib.import_module("research.exploratory.00_spectra_exploration")

FIG_DIR = EXPLORATORY_RESULTS_DIR


def spectra_stats(Y, label):
    """Linear- and log10-scale summary statistics of one spectra matrix."""
    Y_log = log10_spectra(Y)
    return {
        "label": label,
        "shape": list(Y.shape),
        "linear": {
            "min": float(Y.min()),
            "max": float(Y.max()),
            "mean": float(Y.mean()),
            "std": float(Y.std()),
            "skew": float(stats.skew(Y.ravel())),
        },
        "log10": {
            "min": float(Y_log.min()),
            "max": float(Y_log.max()),
            "mean": float(Y_log.mean()),
            "std": float(Y_log.std()),
            "skew": float(stats.skew(Y_log.ravel())),
        },
    }


def print_stats(entry) -> None:
    print(f"{entry['label']} {tuple(entry['shape'])}:")
    for scale in ("linear", "log10"):
        block = entry[scale]
        print(
            f"  {scale:6s}: min={block['min']:.4g}, max={block['max']:.4g}, "
            f"mean={block['mean']:.4g}, std={block['std']:.4g}, "
            f"skew={block['skew']:+.3f}"
        )


def log_plottable_observed_frame(obs, floor):
    """Observed data restricted to what a log-scaled y-axis can draw.

    Depths stay in original units -- the axis does the log transform -- but
    non-positive depths are dropped (no position on a log axis) and lower
    error bars that would cross zero are clipped to `floor` (the minimum
    positive simulated depth) so the bar remains drawable while visibly
    saturated.
    """
    positive = obs["measured_eclipse_depth"].to_numpy() > 0
    frame = obs.loc[positive].copy()
    depth = frame["measured_eclipse_depth"].to_numpy()
    frame["measured_err_lo"] = np.minimum(
        frame["measured_err_lo"].to_numpy(), depth - floor
    )
    return frame, int((~positive).sum())


def plot_linear_vs_log(wl, YHF, YLF_10k, obs, floor, savepath):
    # No sharex: each panel autoscales from the same wavelengths anyway, and
    # sharing lets one panel's log-space extents leak into the other's limits.
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    spectra00.draw_hf_lf_observed(axes[0], wl, YHF, YLF_10k, obs)
    obs_log, n_dropped = log_plottable_observed_frame(obs, floor)
    spectra00.draw_hf_lf_observed(
        axes[1],
        wl,
        YHF,
        YLF_10k,
        obs_log,
        # Percentiles commute with log10, so the LF band is unchanged; the HF
        # centre is not -- in log space it is the geometric mean.
        hf_center=inverse_log10_spectra(log10_spectra(YHF).mean(axis=0)),
        hf_label="HF geometric mean",
        log_depth=True,
        ylabel="Eclipse depth (log scale)",
        title=(
            "HF / LF / Observed spectra (log scale)\n"
            f"{n_dropped} non-positive observed points dropped"
        ),
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return n_dropped


def plot_correlation_linear_vs_log(wl, per_wavelength, per_wavelength_log,
                                   savepath):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    spectra00.annotate_instrument_modes(ax, wl)
    ax.plot(wl, per_wavelength, ".-", color="tab:purple", lw=1, alpha=0.7,
            label=f"linear (mean={per_wavelength.mean():.3f})")
    ax.plot(wl, per_wavelength_log, ".-", color="tab:green", lw=1, alpha=0.7,
            label=f"log10 (mean={per_wavelength_log.mean():.3f})")
    spectra00.set_log_wavelength_axis(ax, wl)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Pearson r (HF vs LF) across samples")
    ax.set_title("Per-wavelength HF/LF correlation: linear vs log10 scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wl = data["wavelengths"]
    YHF = data["YHF"].to_numpy()
    YLF = data["YLF"].to_numpy()          # 97 rows, input-matched to HF
    YLF_10k = data["YLF_10k"].to_numpy()  # 10k decoupled LF design
    obs = data["observed"]

    stats_entries = [
        spectra_stats(YHF, "YHF"),
        spectra_stats(YLF, "YLF (paired)"),
        spectra_stats(YLF_10k, "YLF (10k)"),
    ]
    for entry in stats_entries:
        print_stats(entry)

    # Row-paired correlation requires input-matched LF -> the 97-row set.
    _, per_wavelength, global_corr = spectra00.correlation_hf_lf(YHF, YLF)
    _, per_wavelength_log, global_corr_log = spectra00.correlation_hf_lf(
        log10_spectra(YHF), log10_spectra(YLF)
    )
    print(f"global  r(HF, LF): linear={global_corr:.4f}, log10={global_corr_log:.4f}")
    print(f"per-wavelength  r : linear mean={per_wavelength.mean():.4f} "
          f"[{per_wavelength.min():.4f}, {per_wavelength.max():.4f}], "
          f"log10 mean={per_wavelength_log.mean():.4f} "
          f"[{per_wavelength_log.min():.4f}, {per_wavelength_log.max():.4f}]")

    depth = obs["measured_eclipse_depth"].to_numpy()
    err_lo = obs["measured_err_lo"].to_numpy()
    n_nonpositive_depth = int((depth <= 0).sum())
    n_lower_bar_crosses_zero = int(((depth - err_lo) <= 0).sum())
    print(f"observed: {n_nonpositive_depth} non-positive depths, "
          f"{n_lower_bar_crosses_zero} lower error bars crossing zero "
          f"(of {len(depth)})")

    floor = min(YHF.min(), YLF_10k.min())
    n_dropped = plot_linear_vs_log(
        wl, YHF, YLF_10k, obs, floor,
        FIG_DIR / "05_hf_lf_observed_linear_vs_log.png",
    )
    plot_correlation_linear_vs_log(
        wl, per_wavelength, per_wavelength_log,
        FIG_DIR / "05_correlation_linear_vs_log.png",
    )

    (FIG_DIR / "05_log_stats.json").write_text(
        json.dumps(
            {
                "spectra": stats_entries,
                "correlation": {
                    "global_linear": float(global_corr),
                    "global_log10": float(global_corr_log),
                    "per_wavelength_linear_mean": float(per_wavelength.mean()),
                    "per_wavelength_log10_mean": float(per_wavelength_log.mean()),
                },
                "observed": {
                    "n_nonpositive_depth": n_nonpositive_depth,
                    "n_lower_bar_crosses_zero": n_lower_bar_crosses_zero,
                    "n_dropped_in_log_plot": n_dropped,
                    "log_plot_error_bar_floor": float(floor),
                },
            },
            indent=2,
        )
    )
    print(f"figures and 05_log_stats.json saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
