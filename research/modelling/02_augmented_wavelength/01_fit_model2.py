"""Fit Model 2 (wavelength-augmented joint MF-GP) on linear and log10 spectra.

One scalar-valued AR(1) MF-GP over z = (theta, lambda) per output scale: all
97 HF samples on a stride-LAMBDA_STRIDE wavelength subgrid plus an LF sample
subset on the half-stride-offset subgrid, one joint marginal likelihood, one
scalar rho. The full augmented grid (~1.97M points) is far beyond the exact
fit memory gate (see 00_benchmark_model2_fit.py), so the LF sample count is
maximized against the measured time budget instead.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.mf_gp import per_wavelength_rho
from exoplanets_mf.model2 import fit_model2, predict_hf_model2
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02_augmented_wavelength"

SEED = RANDOM_SEED
# Sizing pinned from benchmark/model2_fit_benchmark.json (memory cap
# n=5640; timing ladder anchor: n=2955 fitted in ~19 min). MODERATE size for
# iteration: 40 LF samples -> n = 97*25 + 40*24 = 3385 points, ~30-45 min
# per scale by cubic extrapolation. For a production run, raise toward the
# benchmark recommendation of 120 LF samples (n=5305, ~1.8 h per scale,
# recommended_production_lf_sample_size).
LAMBDA_STRIDE = 8       # 25 HF + 24 offset LF wavelengths of 195
LF_SAMPLE_SIZE = 40


def hyperparameter_table(layer, *, n_points: int) -> pd.DataFrame:
    """Single-row table of the jointly fitted covariance parameters."""
    kernel = layer.model.kernel_
    low_amplitude, low_rbf = kernel.low_kernel.k1, kernel.low_kernel.k2
    delta_amplitude, delta_rbf = (
        kernel.discrepancy_kernel.k1,
        kernel.discrepancy_kernel.k2,
    )
    # Length scales live in standardized augmented-input units; the lambda
    # column (last) is converted back to micrometres for interpretation.
    lambda_scale_um = float(layer.scaler.scale_[-1])
    row = {
        "rho": layer.rho,
        "low_signal_variance": low_amplitude.constant_value,
        "delta_signal_variance": delta_amplitude.constant_value,
        "low_noise": kernel.low_noise,
        "high_noise": kernel.high_noise,
        "joint_log_marginal_likelihood": layer.model.log_marginal_likelihood(),
        "fit_seconds": layer.fit_seconds,
        "n_points": n_points,
        "lf_sample_size": len(layer.lf_sample_indices),
        "lambda_stride": LAMBDA_STRIDE,
        "low_lambda_length_scale_um": low_rbf.length_scale[-1] * lambda_scale_um,
        "delta_lambda_length_scale_um": delta_rbf.length_scale[-1] * lambda_scale_um,
    }
    row.update(
        {f"low_length_scale_{i}": ell for i, ell in enumerate(low_rbf.length_scale)}
    )
    row.update(
        {f"delta_length_scale_{i}": ell for i, ell in enumerate(delta_rbf.length_scale)}
    )
    return pd.DataFrame([row])


def shade_instrument_modes(ax, wavelengths) -> None:
    for mode, mask in instrument_mode_masks(wavelengths):
        wavelength_range = wavelengths[mask]
        ax.axvspan(
            wavelength_range.min(),
            wavelength_range.max(),
            color=mode.color,
            alpha=0.4,
            zorder=0,
        )


def plot_diagnostics(
    layer,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    rho_1b: np.ndarray,
    input_names: list[str],
    *,
    output_units: str,
    savepath,
) -> None:
    kernel = layer.model.kernel_
    wavelengths = layer.wavelengths
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    positions = np.arange(len(input_names))
    ax.bar(
        positions - 0.2,
        kernel.low_kernel.k2.length_scale,
        width=0.4,
        color="tab:blue",
        label="LF kernel",
    )
    ax.bar(
        positions + 0.2,
        kernel.discrepancy_kernel.k2.length_scale,
        width=0.4,
        color="tab:orange",
        label="discrepancy kernel",
    )
    ax.set_xticks(positions, input_names, rotation=45)
    ax.set_yscale("log")
    ax.set(
        ylabel="ARD length scale (standardized units)",
        title=f"Fitted length scales ({output_units})",
    )
    ax.legend()

    ax = axes[1]
    shade_instrument_modes(ax, wavelengths)
    ax.plot(
        wavelengths, rho_1b, ".-", color="tab:blue", lw=1,
        label="rho_j (Model 1B closed form)",
    )
    ax.axhline(
        layer.rho, color="tab:red", ls="--",
        label=f"Model 2 scalar rho = {layer.rho:.3f}",
    )
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="rho",
        title="Scalar rho vs per-wavelength diagnostic",
    )
    ax.legend()

    ax = axes[2]
    shade_instrument_modes(ax, wavelengths)
    means, _ = predict_hf_model2(layer, X_hf)
    residual_rms = np.sqrt(((means - Y_hf) ** 2).mean(axis=0))
    trained = np.zeros(len(wavelengths), dtype=bool)
    trained[layer.hf_lambda_indices] = True
    ax.plot(wavelengths, residual_rms, "-", color="gray", lw=0.8, zorder=1)
    ax.plot(
        wavelengths[trained], residual_rms[trained], "o", ms=4,
        color="tab:blue", label="HF-trained wavelength",
    )
    ax.plot(
        wavelengths[~trained], residual_rms[~trained], ".", ms=4,
        color="tab:red", label="interpolated wavelength",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="in-sample HF residual RMS",
        title="Wavelength interpolation quality (in-sample)",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_and_report(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    Y_lf_paired: np.ndarray,
    wavelengths: np.ndarray,
    input_names: list[str],
    *,
    output_units: str,
    output_dir,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    layer = fit_model2(
        X_lf,
        Y_lf,
        X_hf,
        Y_hf,
        wavelengths,
        seed=SEED,
        lf_sample_size=LF_SAMPLE_SIZE,
        lambda_stride=LAMBDA_STRIDE,
    )
    elapsed = time.perf_counter() - t0
    n_points = layer.model.X_train_.shape[0]
    print(
        f"  [{output_units}] fitted n={n_points} points in {elapsed:.1f}s: "
        f"rho={layer.rho:.3f}, "
        f"logML={layer.model.log_marginal_likelihood():.1f}"
    )

    table = hyperparameter_table(layer, n_points=n_points)
    table.to_csv(output_dir / "model2_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model2_layer.joblib")
    plot_diagnostics(
        layer,
        X_hf,
        Y_hf,
        per_wavelength_rho(Y_hf, Y_lf_paired),
        input_names,
        output_units=output_units,
        savepath=output_dir / "01_model2_diagnostics.png",
    )
    (output_dir / "timing_summary.json").write_text(
        json.dumps(
            {
                "output_units": output_units,
                "n_points": int(n_points),
                "lf_sample_size": LF_SAMPLE_SIZE,
                "lambda_stride": LAMBDA_STRIDE,
                "n_hf_wavelengths": int(len(layer.hf_lambda_indices)),
                "n_lf_wavelengths": int(len(layer.lf_lambda_indices)),
                "fit_seconds": round(layer.fit_seconds, 1),
                "total_seconds": round(elapsed, 1),
                "seed": SEED,
            },
            indent=2,
        )
    )
    print(f"  outputs written to {output_dir}")


def main() -> None:
    data = load_all()
    wavelengths = data["wavelengths"]
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    YLF_paired = data["YLF"].to_numpy()
    input_names = list(data["XHF"].columns) + ["lambda"]

    print(
        f"Fitting Model 2: {LF_SAMPLE_SIZE} of {len(XLF_10k)} LF samples "
        f"(lambda stride {LAMBDA_STRIDE}, offset {LAMBDA_STRIDE // 2}) + "
        f"{len(XHF)} HF samples (stride {LAMBDA_STRIDE}), both scales"
    )
    fit_and_report(
        XLF_10k, YLF_10k, XHF, YHF, YLF_paired, wavelengths, input_names,
        output_units="original eclipse-depth units",
        output_dir=OUTPUT_DIR / "linear",
    )
    fit_and_report(
        XLF_10k,
        log10_spectra(YLF_10k),
        XHF,
        log10_spectra(YHF),
        log10_spectra(YLF_paired),
        wavelengths,
        input_names,
        output_units="log10 eclipse-depth units",
        output_dir=OUTPUT_DIR / "log10",
    )


if __name__ == "__main__":
    main()
