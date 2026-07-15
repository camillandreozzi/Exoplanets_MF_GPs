"""Fit Model 2K (exact Kronecker two-stage MF-GP) on linear and log10 spectra.

Stage L trains on the COMPLETE stacked LF grid -- 10,097 theta rows
(XLF_10k + the 97 paired HF inputs) x 195 wavelengths = 1,968,915 points,
exactly; stage delta trains on the complete 97 x 195 HF residual grid with
the scalar rho profiled in closed form. No subsampling anywhere: the
Kronecker eigen-identity makes exact inference O(n^3 + m^3) per stage.

Stage-L hyperparameters use screen-then-polish restarts: N_RESTARTS
L-BFGS-B starts on a SCREEN_SIZE-row theta subset (~1 s per objective
evaluation), then one warm-started polish on the full grid (~2.5 min per
evaluation, dominated by eigh of the 10,097^2 theta factor). Only the
restart initialization is screened; the reported fit is exact on all rows.

Restart robustness (restart_robustness.csv): seeds 1-4 rerun the FULL
stage-delta fit (seconds each -> complete spread of rho and the delta
lambda length scale) and the stage-L SCREENING fit (spread of the LF
lambda length scale at n=SCREEN_SIZE); full-grid stage-L refits per seed
would multiply hours by five for a warm-start-only difference.
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.kron_gp import fit_kron_gp
from exoplanets_mf.mf_gp import per_wavelength_rho
from exoplanets_mf.model2_kron import (
    STAGE_L_SCREEN_SIZE,
    Model2KronLayer,
    fit_model2_kron,
    fit_stage_l,
    predict_hf_model2_kron,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED
from exoplanets_mf.transforms import log10_spectra

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02b_kronecker"

SEED = RANDOM_SEED
N_RESTARTS = 10
SCREEN_SIZE = STAGE_L_SCREEN_SIZE
ROBUSTNESS_SEEDS = (1, 2, 3, 4)


def _pinned(fit) -> str:
    return ";".join(f"{name}:{side}" for name, side in sorted(fit.pinned_bounds.items()))


def hyperparameter_table(layer: Model2KronLayer) -> pd.DataFrame:
    """Single-row table of both stages' fitted covariance parameters."""
    stage_l, delta = layer.stage_l, layer.stage_delta
    gp_l = stage_l.gp
    # Theta length scales live in standardized input units; the lambda length
    # scale is converted back to micrometres for interpretation.
    row = {
        "rho": layer.rho,
        "low_signal_variance": gp_l.signal_variance,
        "delta_signal_variance": delta.signal_variance,
        "low_noise": gp_l.noise_variance,
        "delta_noise": delta.noise_variance,
        "lml_stage_l": gp_l.log_marginal_likelihood,
        "lml_delta": delta.log_marginal_likelihood,
        "low_lambda_length_scale_um": float(
            np.exp(gp_l.lambda_log_params[0]) * stage_l.lambda_sd
        ),
        "delta_lambda_length_scale_um": float(
            np.exp(delta.lambda_log_params[0]) * stage_l.lambda_sd
        ),
        "pinned_bounds_low": _pinned(gp_l),
        "pinned_bounds_delta": _pinned(delta),
        "n_theta_stage_l": gp_l.X.shape[0],
        "n_grid_points_stage_l": gp_l.X.shape[0] * len(gp_l.t),
        "n_hf_samples": delta.X.shape[0],
        "screen_size": SCREEN_SIZE,
        "n_restarts": N_RESTARTS,
        "stage_l_screen_seconds": stage_l.screen_seconds,
        "stage_l_fit_seconds": stage_l.fit_seconds,
        "stage_l_polish_iterations": gp_l.n_iterations,
        "eigh_seconds_stage_l": gp_l.eigh_seconds,
        "delta_fit_seconds": layer.fit_seconds,
        "seed": SEED,
    }
    row.update(
        {f"low_length_scale_{i}": ell for i, ell in enumerate(gp_l.length_scales)}
    )
    row.update(
        {f"delta_length_scale_{i}": ell for i, ell in enumerate(delta.length_scales)}
    )
    return pd.DataFrame([row])


def restart_robustness_table(layer: Model2KronLayer, X_hf: np.ndarray, Y_hf: np.ndarray, Y_lf_paired: np.ndarray) -> pd.DataFrame:
    """(rho, lambda length scale) spread across seeds 0-4.

    stage == "delta_full": complete stage-delta refits (rho + delta lambda
    scale). stage == "L_screen": stage-L screening refits on seed-specific
    theta subsets (LF lambda scale at n=SCREEN_SIZE; the seed-0 row is the
    production polish on the FULL grid, marked n_theta_rows accordingly).
    """
    stage_l = layer.stage_l
    t = stage_l.gp.t
    X_hf_std = stage_l.standardize_theta(X_hf)
    Y_hf_std = stage_l.standardize_columns(Y_hf)
    Y_lfp_std = stage_l.standardize_columns(Y_lf_paired)

    def row(seed: int, stage: str, fit, n_theta_rows: int) -> dict:
        return {
            "seed": seed,
            "stage": stage,
            "n_theta_rows": n_theta_rows,
            "lml": fit.log_marginal_likelihood,
            "rho": np.nan if fit.rho is None else fit.rho,
            "lambda_length_scale_um": float(
                np.exp(fit.lambda_log_params[0]) * stage_l.lambda_sd
            ),
            "signal_variance": fit.signal_variance,
            "noise_variance": fit.noise_variance,
            "pinned_bounds": _pinned(fit),
        }

    rows = [
        row(SEED, "delta_full", layer.stage_delta, X_hf.shape[0]),
        row(SEED, "L_screen", stage_l.gp, stage_l.gp.X.shape[0]),
    ]
    for seed in ROBUSTNESS_SEEDS:
        delta_fit = fit_kron_gp(
            X_hf_std, t, Y_hf_std, Y_pair=Y_lfp_std, seed=seed, n_restarts=N_RESTARTS
        )
        rows.append(row(seed, "delta_full", delta_fit, X_hf.shape[0]))

        X_full, Y_full = stage_l.gp.X, stage_l.gp.Y
        subset = np.random.default_rng(seed).choice(
            X_full.shape[0], size=SCREEN_SIZE, replace=False
        )
        print(f"    robustness seed {seed}: stage-L screening fit ...", flush=True)
        screen_fit = fit_kron_gp(
            X_full[subset], t, Y_full[subset], seed=seed, n_restarts=N_RESTARTS
        )
        rows.append(row(seed, "L_screen", screen_fit, SCREEN_SIZE))
    return pd.DataFrame(rows)


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
    layer: Model2KronLayer,
    X_hf: np.ndarray,
    Y_hf: np.ndarray,
    rho_1b: np.ndarray,
    input_names: list[str],
    *,
    output_units: str,
    savepath,
) -> None:
    wavelengths = layer.wavelengths
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Both stages' ARD length scales; lambda (last position) is the separate
    # Kronecker factor's length scale, shown in the same standardized units.
    ax = axes[0]
    low_scales = np.append(
        layer.stage_l.gp.length_scales, np.exp(layer.stage_l.gp.lambda_log_params[0])
    )
    delta_scales = np.append(
        layer.stage_delta.length_scales, np.exp(layer.stage_delta.lambda_log_params[0])
    )
    positions = np.arange(len(input_names))
    ax.bar(positions - 0.2, low_scales, width=0.4, color="tab:blue", label="LF kernel")
    ax.bar(
        positions + 0.2, delta_scales, width=0.4, color="tab:orange",
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
        label=f"Model 2K scalar rho = {layer.rho:.3f}",
    )
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="rho",
        title="Scalar rho vs per-wavelength diagnostic",
    )
    ax.legend()

    ax = axes[2]
    shade_instrument_modes(ax, wavelengths)
    means, _ = predict_hf_model2_kron(layer, X_hf)
    residual_rms = np.sqrt(((means - Y_hf) ** 2).mean(axis=0))
    ax.plot(wavelengths, residual_rms, "o-", ms=3, lw=0.8, color="tab:blue")
    ax.axvline(
        5.0, color="black", ls=":", lw=1.2, label=r"5 $\mu$m NIRCam/MIRI break"
    )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"wavelength $\lambda$ [$\mu$m]",
        ylabel="in-sample HF residual RMS",
        title="Per-wavelength residuals (all 195 trained -- no interpolation)",
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
    print("  fitting stage L (screen + full-grid polish) ...", flush=True)
    stage_l = fit_stage_l(
        X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
        seed=SEED, n_restarts=N_RESTARTS, screen_size=SCREEN_SIZE,
    )
    print(
        f"  stage L done in {stage_l.fit_seconds:.0f}s "
        f"(screen {stage_l.screen_seconds:.0f}s, polish "
        f"{stage_l.gp.n_iterations} iterations, last objective eval "
        f"{stage_l.gp.eigh_seconds:.0f}s), lml={stage_l.gp.log_marginal_likelihood:.1f}",
        flush=True,
    )
    layer = fit_model2_kron(
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths,
        seed=SEED, n_restarts=N_RESTARTS, stage_l=stage_l,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  [{output_units}] exact fit on {layer.stage_l.gp.X.shape[0]}x195 LF + "
        f"97x195 HF grids in {elapsed:.0f}s: rho={layer.rho:.3f}, "
        f"lml_L={stage_l.gp.log_marginal_likelihood:.1f}, "
        f"lml_delta={layer.stage_delta.log_marginal_likelihood:.1f}",
        flush=True,
    )

    hyperparameter_table(layer).to_csv(
        output_dir / "model2k_hyperparameters.csv", index=False
    )
    print("  restart robustness (seeds 1-4) ...", flush=True)
    restart_robustness_table(layer, X_hf, Y_hf, Y_lf_paired).to_csv(
        output_dir / "restart_robustness.csv", index=False
    )
    joblib.dump(layer, output_dir / "model2k_layer.joblib")
    plot_diagnostics(
        layer,
        X_hf,
        Y_hf,
        per_wavelength_rho(Y_hf, Y_lf_paired),
        input_names,
        output_units=output_units,
        savepath=output_dir / "01_model2k_diagnostics.png",
    )
    (output_dir / "timing_summary.json").write_text(
        json.dumps(
            {
                "output_units": output_units,
                "n_theta_stage_l": int(layer.stage_l.gp.X.shape[0]),
                "n_grid_points_stage_l": int(
                    layer.stage_l.gp.X.shape[0] * len(wavelengths)
                ),
                "n_grid_points_delta": int(X_hf.shape[0] * len(wavelengths)),
                "screen_size": SCREEN_SIZE,
                "n_restarts": N_RESTARTS,
                "stage_l_screen_seconds": round(stage_l.screen_seconds, 1),
                "stage_l_fit_seconds": round(stage_l.fit_seconds, 1),
                "stage_l_polish_iterations": int(stage_l.gp.n_iterations),
                "stage_l_last_objective_eval_seconds": round(
                    stage_l.gp.eigh_seconds, 1
                ),
                "delta_fit_seconds": round(layer.fit_seconds, 1),
                "total_seconds": round(elapsed, 1),
                "seed": SEED,
            },
            indent=2,
        )
    )
    print(f"  outputs written to {output_dir}", flush=True)


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
        f"Fitting Model 2K exactly on the complete grids: stage L "
        f"{len(XLF_10k) + len(XHF)} theta x {len(wavelengths)} lambda, "
        f"stage delta {len(XHF)} x {len(wavelengths)}; "
        f"{N_RESTARTS} restarts (stage-L screening at n={SCREEN_SIZE}), both scales"
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
