"""Full-data GPBoost fits for HF-only, Model 1A, Model 1B, and Model 2.

This writes parameter tables for the four GPBoost variants on the full HF data
and the project-wide LF subsample. It is separate from the paired CV workflow:
these fits are for fitted parameters and diagnostics, not held-out scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from exoplanets_mf.data import load_all
from exoplanets_mf.gpboost_hf_only import (
    fit_hf_only_gpboost,
    hyperparameter_table_hf_only_gpboost,
)
from exoplanets_mf.gpboost_mf import (
    GPBOOST_COV_FCT_SHAPE,
    GPBOOST_COV_FUNCTION,
    fit_model1_gpboost,
    fit_model1a_gpboost,
    gpboost_available,
    hyperparameter_table_model1_gpboost,
)
from exoplanets_mf.gpboost_model2 import (
    fit_model2_gpboost,
    hyperparameter_table_model2_gpboost,
)
from exoplanets_mf.model2 import MODEL2_MAX_AUGMENTED_POINTS, derive_lambda_stride
from exoplanets_mf.paths import MODELLING_RESULTS_DIR, approximation_suffix
from exoplanets_mf.provenance import write_run_metadata
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED

OUTPUT_ROOT = MODELLING_RESULTS_DIR / "03_gpboost_comparison"
OUTPUT_STEM = "full_fit"

GP_APPROX_CHOICES = (
    "none",
    "vecchia",
    "vecchia_euclidean",
    "full_scale_vecchia",
    "fitc",
    "tapering",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gp-approx",
        choices=GP_APPROX_CHOICES,
        default="none",
        help="GPBoost approximation for all four full-data fits.",
    )
    parser.add_argument(
        "--num-neighbors",
        default=None,
        help=(
            "Vecchia neighbours: an integer, or 'none' for GPBoost's internal "
            "default. Ignored when --gp-approx is 'none'."
        ),
    )
    parser.add_argument(
        "--model1-lf-sample-size",
        type=int,
        default=LF_SUBSAMPLE_SIZE,
        help="LF rows for Model 1A/1B full fits.",
    )
    parser.add_argument(
        "--model2-lf-sample-size",
        type=int,
        default=LF_SUBSAMPLE_SIZE,
        help="LF rows for the Model 2 full fit.",
    )
    parser.add_argument(
        "--model2-max-points",
        type=int,
        default=MODEL2_MAX_AUGMENTED_POINTS,
        help="Augmented scalar-row budget used to derive Model 2's wavelength stride.",
    )
    return parser.parse_args()


def parse_num_neighbors(gp_approx: str, raw: str | None) -> int | None:
    if gp_approx == "none":
        return None
    if raw is None:
        return None
    if raw.strip().lower() in {"none", "null", "default"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"--num-neighbors must be positive; found {value}")
    return value


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(float(seconds), 60.0)
    if minutes < 1:
        return f"{secs:.1f}s"
    return f"{int(minutes)}m {secs:04.1f}s"


def validate_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive; found {value}")
    return value


def output_stem(args: argparse.Namespace) -> str:
    if (
        args.model1_lf_sample_size == LF_SUBSAMPLE_SIZE
        and args.model2_lf_sample_size == LF_SUBSAMPLE_SIZE
        and args.model2_max_points == MODEL2_MAX_AUGMENTED_POINTS
    ):
        return OUTPUT_STEM
    return (
        f"{OUTPUT_STEM}_m1lf{args.model1_lf_sample_size}"
        f"_m2lf{args.model2_lf_sample_size}"
        f"_m2max{args.model2_max_points}"
    )


def main() -> None:
    args = parse_args()
    num_neighbors = parse_num_neighbors(args.gp_approx, args.num_neighbors)
    model1_lf_sample_size = validate_positive(
        "--model1-lf-sample-size", args.model1_lf_sample_size
    )
    model2_lf_sample_size = validate_positive(
        "--model2-lf-sample-size", args.model2_lf_sample_size
    )
    model2_max_points = validate_positive("--model2-max-points", args.model2_max_points)
    output_dir = OUTPUT_ROOT / (
        output_stem(args) + approximation_suffix(args.gp_approx, num_neighbors)
    )
    if not gpboost_available():
        print("gpboost not available -- skipping all-GPBoost full fits.")
        print("  (install gpboost and run tools/fix_gpboost_libomp.py; see README)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    wavelengths = data["wavelengths"]
    X_lf = data["XLF_10k"].to_numpy()
    Y_lf = data["YLF_10k"].to_numpy()
    X_hf = data["XHF"].to_numpy()
    Y_hf = data["YHF"].to_numpy()
    lambda_stride = derive_lambda_stride(
        model2_lf_sample_size,
        len(X_hf),
        len(wavelengths),
        max_points=model2_max_points,
    )

    runtime_seconds: dict[str, float] = {}

    def timed(key: str, label: str, fn):
        print(f"{label} full fit ...", flush=True)
        t0 = time.perf_counter()
        layer = fn()
        runtime_seconds[key] = time.perf_counter() - t0
        print(f"  {label} fit in {format_duration(runtime_seconds[key])}", flush=True)
        return layer

    run_started = time.perf_counter()

    hf_layer = timed(
        "hf_only_gpboost",
        "HF-only GPBoost",
        lambda: fit_hf_only_gpboost(
            X_hf,
            Y_hf,
            wavelengths,
            seed=RANDOM_SEED,
            gp_approx=args.gp_approx,
            num_neighbors=num_neighbors,
            progress_every=40,
        ),
    )
    hyperparameter_table_hf_only_gpboost(hf_layer).to_csv(
        output_dir / "hf_only_gpboost_hyperparameters.csv", index=False
    )
    joblib.dump(hf_layer, output_dir / "hf_only_gpboost_layer.joblib")

    model1b_layer = timed(
        "model_1b_gpboost",
        "Model 1B GPBoost",
        lambda: fit_model1_gpboost(
            X_lf,
            Y_lf,
            X_hf,
            Y_hf,
            wavelengths,
            seed=RANDOM_SEED,
            subsample_size=model1_lf_sample_size,
            gp_approx=args.gp_approx,
            num_neighbors=num_neighbors,
            progress_every=40,
        ),
    )
    hyperparameter_table_model1_gpboost(model1b_layer).to_csv(
        output_dir / "model_1b_gpboost_hyperparameters.csv", index=False
    )
    joblib.dump(model1b_layer, output_dir / "model_1b_gpboost_layer.joblib")

    model1a_layer = timed(
        "model_1a_gpboost",
        "Model 1A GPBoost",
        lambda: fit_model1a_gpboost(
            X_lf,
            Y_lf,
            X_hf,
            Y_hf,
            wavelengths,
            seed=RANDOM_SEED,
            subsample_size=model1_lf_sample_size,
            gp_approx=args.gp_approx,
            num_neighbors=num_neighbors,
            warm_start_layer=model1b_layer,
            progress_every=40,
        ),
    )
    hyperparameter_table_model1_gpboost(model1a_layer).to_csv(
        output_dir / "model_1a_gpboost_hyperparameters.csv", index=False
    )
    joblib.dump(model1a_layer, output_dir / "model_1a_gpboost_layer.joblib")

    model2_layer = timed(
        "model_2_gpboost",
        "Model 2 GPBoost",
        lambda: fit_model2_gpboost(
            X_lf,
            Y_lf,
            X_hf,
            Y_hf,
            wavelengths,
            seed=RANDOM_SEED,
            lf_sample_size=model2_lf_sample_size,
            lambda_stride=lambda_stride,
            gp_approx=args.gp_approx,
            num_neighbors=num_neighbors,
        ),
    )
    hyperparameter_table_model2_gpboost(
        model2_layer,
        n_points=getattr(model2_layer.model, "n_points", None),
        lambda_stride=lambda_stride,
    ).to_csv(output_dir / "model_2_gpboost_hyperparameters.csv", index=False)
    joblib.dump(model2_layer, output_dir / "model_2_gpboost_layer.joblib")

    runtime_seconds["total"] = time.perf_counter() - run_started
    summary = {
        "result_set": output_dir.name,
        "n_samples": int(Y_hf.shape[0]),
        "n_wavelengths": int(len(wavelengths)),
        "seed": RANDOM_SEED,
        "cov_function_model_1": GPBOOST_COV_FUNCTION,
        "cov_fct_shape": GPBOOST_COV_FCT_SHAPE,
        "gp_approx": args.gp_approx,
        "num_neighbors": num_neighbors,
        "model_1_lf_sample_size": model1_lf_sample_size,
        "model_2_lf_sample_size": model2_lf_sample_size,
        "model_2_max_points": model2_max_points,
        "model_2_lambda_stride": lambda_stride,
        "model_2_n_points": int(getattr(model2_layer.model, "n_points", 0)),
        "model_1a_shared_rho": float(model1a_layer.rho[0]),
        "model_1a_sweeps": model1a_layer.global_rho_sweeps,
        "model_1b_rho_mean": float(model1b_layer.rho.mean()),
        "model_2_rho": float(model2_layer.rho),
        "runtime_seconds": {
            key: round(value, 1) for key, value in runtime_seconds.items()
        },
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_run_metadata(
        output_dir / "run_metadata.json",
        workflow="modelling/03_gpboost_comparison",
        scripts=[Path(__file__).name, "full-fit"],
    )
    print(
        "runtime: "
        + ", ".join(
            f"{key}={format_duration(value)}"
            for key, value in runtime_seconds.items()
        )
    )
    print(f"outputs written to {output_dir}")


if __name__ == "__main__":
    main()
