"""Fit Model 2 (wavelength-augmented joint MF-GP) on linear and log10 spectra.

One scalar-valued AR(1) MF-GP over z = (theta, lambda) per output scale: all
97 HF samples on a stride wavelength subgrid plus the shared LF subsample on
the half-stride-offset subgrid, one joint marginal likelihood, one scalar
rho. The LF count is the project-wide LF_SUBSAMPLE_SIZE (the one canonical LF
subsample shared by every model); the full augmented grid (~1.97M points) is
far beyond the exact-fit memory gate, so the wavelength stride is derived from
a runtime point budget (derive_lambda_stride) to keep the design affordable.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.model2 import (
    MODEL2_MAX_AUGMENTED_POINTS,
    derive_lambda_stride,
    fit_model2,
    predict_hf_model2,
)
from exoplanets_mf.paths import (
    LOG_MODELLING_RESULTS_DIR,
    MODELLING_RESULTS_DIR,
)
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02_augmented_wavelength"
LOG_OUTPUT_DIR = LOG_MODELLING_RESULTS_DIR / "02_augmented_wavelength"

SEED = RANDOM_SEED
# The one canonical LF subsample shared by every model and CV (see
# reproducibility.LF_SUBSAMPLE_SIZE) -- identical LF rows to Model 1. The
# wavelength stride is derived from this size (in main, once the data shape is
# known) so the augmented design stays within the measured runtime budget.
LF_SAMPLE_SIZE = LF_SUBSAMPLE_SIZE


def hyperparameter_table(layer, *, n_points: int, lambda_stride: int) -> pd.DataFrame:
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
        "lambda_stride": lambda_stride,
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
    if rho_1b is not None:
        ax.plot(
            wavelengths, rho_1b, ".-", color="tab:blue", lw=1,
            label="rho_j (Model 1B MF-GP)",
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


def load_model1b_rho(model1b_dir) -> np.ndarray | None:
    """Fitted Model 1B GP rho_j for the diagnostic overlay, if available."""
    csv_path = model1b_dir / "model_1b_hyperparameters.csv"
    if not csv_path.exists():
        print(f"  no Model 1B layer at {csv_path} -- overlay skipped")
        return None
    return pd.read_csv(csv_path)["rho"].to_numpy()


def fit_and_report(
    X_lf: np.ndarray,
    Y_lf: np.ndarray,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    model1b_dir,
    wavelengths: np.ndarray,
    input_names: list[str],
    *,
    lambda_stride: int,
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
        lambda_stride=lambda_stride,
    )
    elapsed = time.perf_counter() - t0
    n_points = layer.model.X_train_.shape[0]
    print(
        f"  [{output_units}] fitted n={n_points} points in {elapsed:.1f}s: "
        f"rho={layer.rho:.3f}, "
        f"logML={layer.model.log_marginal_likelihood():.1f}"
    )

    table = hyperparameter_table(layer, n_points=n_points, lambda_stride=lambda_stride)
    table.to_csv(output_dir / "model2_hyperparameters.csv", index=False)
    joblib.dump(layer, output_dir / "model2_layer.joblib")
    plot_diagnostics(
        layer,
        X_hf,
        Y_hf,
        load_model1b_rho(model1b_dir),
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
                "lambda_stride": lambda_stride,
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
    input_names = list(data["XHF"].columns) + ["lambda"]

    # Derive the wavelength stride for the shared LF subsample size: the finest
    # grid whose augmented design stays within the production runtime budget.
    lambda_stride = derive_lambda_stride(
        LF_SAMPLE_SIZE,
        len(XHF),
        len(wavelengths),
        max_points=MODEL2_MAX_AUGMENTED_POINTS,
    )

    print(
        f"Fitting Model 2: {LF_SAMPLE_SIZE} of {len(XLF_10k)} LF samples "
        f"(lambda stride {lambda_stride}, offset {lambda_stride // 2}) + "
        f"{len(XHF)} HF samples (stride {lambda_stride}), both scales"
    )
    fit_and_report(
        XLF_10k, YLF_10k, XHF, YHF,
        MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "model_1b",
        wavelengths, input_names,
        lambda_stride=lambda_stride,
        output_units="original eclipse-depth units",
        output_dir=OUTPUT_DIR / "linear",
    )
    fit_and_report(
        XLF_10k,
        log10_spectra(YLF_10k),
        XHF,
        log10_spectra(YHF),
        LOG_MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "model_1b",
        wavelengths,
        input_names,
        lambda_stride=lambda_stride,
        output_units="log10 eclipse-depth units",
        output_dir=LOG_OUTPUT_DIR / "log10",
    )


if __name__ == "__main__":
    main()
