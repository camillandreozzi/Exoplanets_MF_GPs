"""Benchmark the Model 2 joint fit cost and size the augmented subsample.

Model 2 stacks (theta, lambda) rows of BOTH fidelities into one design and
fits a single joint marginal likelihood, so the relevant size is the total
scalar point count n = n_lf_samples * n_lf_lambdas + n_hf_samples *
n_hf_lambdas. Two questions, answered from measurements rather than guesses:

1. How large can n be? (Memory of one optimizer iteration vs RAM -- the same
   exact_fit_memory_bytes gate as Model 1, but with the 25-hyperparameter
   augmented kernel.)
2. Given a wavelength stride, what is the LARGEST LF sample count that keeps
   one production fit inside FIT_TIME_BUDGET_HOURS, and one CV fold inside
   CV_FOLD_TIME_BUDGET_HOURS? The recommendation is advisory; the fit and CV
   scripts pin their constants with the derivation in a comment so
   production runs stay reproducible.

Report-only -- it never launches the full fit itself.
"""

import json
import resource
import time

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import (
    MF_GP_MEMORY_FRACTION,
    available_memory_bytes,
    exact_fit_memory_bytes,
)
from exoplanets_mf.model2 import (
    fit_model2,
    model2_n_theta,
    select_wavelength_subgrid,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

OUTPUT_DIR = MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "benchmark"

SEED = RANDOM_SEED
# Timing ladder of (lf_sample_size, lambda_stride) with growing totals; kept
# well inside RAM so the benchmark itself is safe to run.
BENCHMARK_CONFIGS = ((25, 39), (50, 24), (100, 13))
# Wavelength stride proposed for real runs: stride 8 keeps 25 of 195 HF
# wavelengths (plus 24 offset LF wavelengths) and lets the lambda kernel
# interpolate the remaining 170.
TARGET_LAMBDA_STRIDE = 8
# Practical wall-time budgets: one production fit, and one fold of the
# 5-fold CV (which refits the joint GP from scratch).
FIT_TIME_BUDGET_HOURS = 2.0
CV_FOLD_TIME_BUDGET_HOURS = 0.5


def design_size(lf_sample_size: int, lambda_stride: int, n_hf: int, n_lambdas: int) -> int:
    n_hf_lambdas = len(select_wavelength_subgrid(n_lambdas, stride=lambda_stride))
    n_lf_lambdas = len(
        select_wavelength_subgrid(
            n_lambdas, stride=lambda_stride, offset=(lambda_stride // 2) % lambda_stride
        )
    )
    return lf_sample_size * n_lf_lambdas + n_hf * n_hf_lambdas


def largest_lf_size_within(n_cap: float, lambda_stride: int, n_hf: int, n_lambdas: int) -> int:
    """Largest LF sample count with total design size <= n_cap (multiple of 10)."""
    n_hf_lambdas = len(select_wavelength_subgrid(n_lambdas, stride=lambda_stride))
    n_lf_lambdas = len(
        select_wavelength_subgrid(
            n_lambdas, stride=lambda_stride, offset=(lambda_stride // 2) % lambda_stride
        )
    )
    return max(int(n_cap - n_hf * n_hf_lambdas) // n_lf_lambdas // 10 * 10, 0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    wavelengths = data["wavelengths"]
    n_hf = len(XHF)
    n_lambdas = len(wavelengths)

    n_theta = model2_n_theta(XLF_10k.shape[1])
    n_full = (len(XLF_10k) + n_hf) * n_lambdas
    ram_bytes = available_memory_bytes()
    if ram_bytes is None:
        n_memory = n_full
    else:
        budget_bytes = MF_GP_MEMORY_FRACTION * ram_bytes
        n_memory = int((budget_bytes / ((2 * n_theta + 4) * 8)) ** 0.5)

    print(
        f"full augmented grid: n={n_full} scalar points "
        f"({len(XLF_10k)} LF + {n_hf} HF samples x {n_lambdas} wavelengths), "
        f"{n_theta} hyperparameters"
    )
    print(
        f"memory cap: n={n_memory} points "
        f"(~{exact_fit_memory_bytes(n_memory, n_theta) / 1e9:.1f} GB per "
        f"iteration at {MF_GP_MEMORY_FRACTION:.0%} of "
        f"{'unknown' if ram_bytes is None else f'{ram_bytes / 1e9:.1f} GB'} RAM)"
    )

    # Timing ladder: full production-like joint fits at growing sizes.
    ladder = []
    for lf_size, stride in BENCHMARK_CONFIGS:
        n_points = design_size(lf_size, stride, n_hf, n_lambdas)
        t0 = time.perf_counter()
        fit_model2(
            XLF_10k,
            YLF_10k,
            XHF,
            YHF,
            wavelengths,
            seed=SEED,
            lf_sample_size=lf_size,
            lambda_stride=stride,
        )
        elapsed = time.perf_counter() - t0
        ladder.append(
            {
                "lf_sample_size": lf_size,
                "lambda_stride": stride,
                "n_points": n_points,
                "fit_seconds": round(elapsed, 3),
            }
        )
        print(f"  n={n_points} (lf={lf_size}, stride={stride}): {elapsed:.1f}s")

    # Hyperparameter optimization is cubic in n per iteration; assume the
    # iteration count itself does not shrink with n. The largest benchmarked
    # size gives the most trustworthy extrapolation anchor.
    anchor = ladder[-1]

    def n_within_hours(hours: float) -> float:
        return anchor["n_points"] * (
            hours * 3600.0 / anchor["fit_seconds"]
        ) ** (1.0 / 3.0)

    n_time_fit = n_within_hours(FIT_TIME_BUDGET_HOURS)
    n_time_cv = n_within_hours(CV_FOLD_TIME_BUDGET_HOURS)
    recommended_production_lf = largest_lf_size_within(
        min(n_time_fit, n_memory), TARGET_LAMBDA_STRIDE, n_hf, n_lambdas
    )
    recommended_cv_lf = largest_lf_size_within(
        min(n_time_cv, n_memory), TARGET_LAMBDA_STRIDE, n_hf, n_lambdas
    )
    n_production = design_size(
        recommended_production_lf, TARGET_LAMBDA_STRIDE, n_hf, n_lambdas
    )
    n_cv = design_size(recommended_cv_lf, TARGET_LAMBDA_STRIDE, n_hf, n_lambdas)
    production_hours = (
        anchor["fit_seconds"] * (n_production / anchor["n_points"]) ** 3 / 3600.0
    )
    cv_fold_hours = anchor["fit_seconds"] * (n_cv / anchor["n_points"]) ** 3 / 3600.0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

    print(
        f"stride {TARGET_LAMBDA_STRIDE} recommendation: production "
        f"lf_sample_size ~{recommended_production_lf} (n={n_production}, "
        f"~{production_hours:.1f} h/fit), CV lf_sample_size "
        f"~{recommended_cv_lf} (n={n_cv}, ~{cv_fold_hours * 60:.0f} min/fold)"
    )

    verdict = (
        f"EXACT FIT ON THE FULL AUGMENTED GRID IS INFEASIBLE "
        f"(n={n_full} vs memory cap n={n_memory}). Subsample both "
        f"fidelities: at stride {TARGET_LAMBDA_STRIDE}, production "
        f"lf_sample_size ~{recommended_production_lf} "
        f"(~{production_hours:.1f} h), CV lf_sample_size "
        f"~{recommended_cv_lf} (~{cv_fold_hours * 60:.0f} min per fold, "
        f"x5 folds x2 scales)."
    )
    print(f"\nVERDICT: {verdict}")

    (OUTPUT_DIR / "model2_fit_benchmark.json").write_text(
        json.dumps(
            {
                "n_points_full_grid": n_full,
                "n_hyperparameters": n_theta,
                "ram_gb": None if ram_bytes is None else round(ram_bytes / 1e9, 2),
                "memory_budget_fraction": MF_GP_MEMORY_FRACTION,
                "memory_cap_n_points": n_memory,
                "timing_ladder": ladder,
                "target_lambda_stride": TARGET_LAMBDA_STRIDE,
                "fit_time_budget_hours": FIT_TIME_BUDGET_HOURS,
                "cv_fold_time_budget_hours": CV_FOLD_TIME_BUDGET_HOURS,
                "recommended_production_lf_sample_size": recommended_production_lf,
                "recommended_production_n_points": n_production,
                "extrapolated_production_fit_hours": round(production_hours, 2),
                "recommended_cv_lf_sample_size": recommended_cv_lf,
                "recommended_cv_n_points": n_cv,
                "extrapolated_cv_fold_hours": round(cv_fold_hours, 3),
                "benchmark_peak_rss_gb": round(peak_rss / 1e9, 3),
                "seed": SEED,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
