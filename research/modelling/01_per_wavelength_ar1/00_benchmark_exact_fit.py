"""Benchmark the joint MF-GP fit cost and size the LF subsample.

Two questions, answered from measurements rather than guesses:

1. Is the exact fit on all 10,000 LF + 97 HF rows feasible here? (Memory of
   one optimizer iteration vs RAM; cubic runtime extrapolated from timed
   single-wavelength fits at growing sizes.)
2. If not, what is the LARGEST LF subsample that keeps the full
   195-wavelength run inside TIME_BUDGET_HOURS? This recommendation is
   advisory; the fit scripts pin their SUBSAMPLE_SIZE constant with the
   derivation in a comment so production runs stay reproducible.

Report-only -- it never launches the full fit itself.
"""

import json
import resource
import time

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import (
    available_memory_bytes,
    exact_fit_memory_bytes,
    fit_joint_mf_gp,
    make_joint_mf_kernel,
    MF_GP_MEMORY_FRACTION,
)
from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

OUTPUT_DIR = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1" / "benchmark"

SEED = RANDOM_SEED
# LF subsample sizes for the timing ladder (plus the 97 HF rows each). Kept
# well inside RAM so the benchmark itself is safe to run.
BENCHMARK_LF_SIZES = (400, 900, 1900)
# Practical wall-time budget for one full 195-wavelength production run.
TIME_BUDGET_HOURS = 8.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    XLF_10k = data["XLF_10k"].to_numpy()
    YLF_10k = data["YLF_10k"].to_numpy()
    XHF = data["XHF"].to_numpy()
    YHF = data["YHF"].to_numpy()
    wavelengths = data["wavelengths"]

    n_full = len(XLF_10k) + len(XHF)
    n_wavelengths = len(wavelengths)
    n_theta = len(make_joint_mf_kernel(XLF_10k.shape[1]).theta)
    required_bytes = exact_fit_memory_bytes(n_full, n_theta)
    ram_bytes = available_memory_bytes()
    memory_feasible = (
        ram_bytes is not None
        and required_bytes <= MF_GP_MEMORY_FRACTION * ram_bytes
    )

    print(
        f"exact fit target: n={n_full} points x {n_wavelengths} wavelengths, "
        f"{n_theta} hyperparameters"
    )
    print(
        f"memory: needs ~{required_bytes / 1e9:.1f} GB per optimizer "
        f"iteration, RAM = "
        f"{'unknown' if ram_bytes is None else f'{ram_bytes / 1e9:.1f} GB'} "
        f"-> {'OK' if memory_feasible else 'EXCEEDS BUDGET'}"
    )

    # Timing ladder: single-wavelength fits at growing joint sizes.
    ladder = []
    for lf_size in BENCHMARK_LF_SIZES:
        n_points = lf_size + len(XHF)
        t0 = time.perf_counter()
        fit_joint_mf_gp(
            XLF_10k,
            YLF_10k[:, :1],
            XHF,
            YHF[:, :1],
            wavelengths[:1],
            seed=SEED,
            subsample_size=lf_size,
        )
        elapsed = time.perf_counter() - t0
        # Hyperparameter optimization is cubic in n per iteration; assume the
        # iteration count itself does not shrink with n.
        extrapolated_one = elapsed * (n_full / n_points) ** 3
        ladder.append(
            {
                "n_points": n_points,
                "fit_seconds": round(elapsed, 3),
                "extrapolated_seconds_at_full_n": round(extrapolated_one, 1),
            }
        )
        print(
            f"  n={n_points}: {elapsed:.1f}s/wavelength "
            f"-> ~{extrapolated_one:.0f}s at n={n_full} (cubic)"
        )

    # The largest benchmarked size gives the most trustworthy extrapolation.
    seconds_per_wavelength = ladder[-1]["extrapolated_seconds_at_full_n"]
    full_run_hours = seconds_per_wavelength * n_wavelengths / 3600.0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

    # Largest subsample within the time budget (cubic from the largest ladder
    # anchor) and within the memory budget; rounded down to a multiple of 100.
    anchor = ladder[-1]
    budget_per_wavelength = TIME_BUDGET_HOURS * 3600.0 / n_wavelengths
    n_time = anchor["n_points"] * (
        budget_per_wavelength / anchor["fit_seconds"]
    ) ** (1.0 / 3.0)
    if ram_bytes is None:
        n_memory = n_full
    else:
        budget_bytes = MF_GP_MEMORY_FRACTION * ram_bytes
        n_memory = int((budget_bytes / ((2 * n_theta + 4) * 8)) ** 0.5)
    recommended_lf_subsample = min(
        int(min(n_time, n_memory) - len(XHF)) // 100 * 100,
        len(XLF_10k),
    )
    print(
        f"largest LF subsample within {TIME_BUDGET_HOURS:.0f}h budget: "
        f"~{recommended_lf_subsample} (time cap n={n_time:.0f}, "
        f"memory cap n={n_memory})"
    )

    if not memory_feasible:
        verdict = (
            f"EXACT FIT INFEASIBLE ON THIS MACHINE: needs "
            f"~{required_bytes / 1e9:.1f} GB per iteration "
            f"(RAM {0 if ram_bytes is None else ram_bytes / 1e9:.1f} GB); "
            f"even with enough memory the full run would take "
            f"~{full_run_hours:.0f} h. Largest LF subsample within the "
            f"{TIME_BUDGET_HOURS:.0f} h budget: ~{recommended_lf_subsample}."
        )
    elif full_run_hours > 12:
        verdict = (
            f"MEMORY OK BUT SLOW: ~{full_run_hours:.0f} h extrapolated for "
            f"{n_wavelengths} wavelengths (~{seconds_per_wavelength:.0f}s "
            f"each). Consider running overnight or reducing scope."
        )
    else:
        verdict = (
            f"FEASIBLE: ~{full_run_hours:.1f} h extrapolated for the full "
            f"exact run."
        )
    print(f"\nVERDICT: {verdict}")

    (OUTPUT_DIR / "exact_fit_benchmark.json").write_text(
        json.dumps(
            {
                "n_points_full": n_full,
                "n_wavelengths": n_wavelengths,
                "n_hyperparameters": n_theta,
                "required_memory_gb_per_iteration": round(required_bytes / 1e9, 2),
                "ram_gb": None if ram_bytes is None else round(ram_bytes / 1e9, 2),
                "memory_budget_fraction": MF_GP_MEMORY_FRACTION,
                "memory_feasible": memory_feasible,
                "timing_ladder": ladder,
                "extrapolated_seconds_per_wavelength": round(seconds_per_wavelength, 1),
                "extrapolated_full_run_hours": round(full_run_hours, 2),
                "time_budget_hours": TIME_BUDGET_HOURS,
                "recommended_lf_subsample": recommended_lf_subsample,
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
