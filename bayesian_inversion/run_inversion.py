"""End-to-end Bayesian inversion of a spectrum onto the nine atmospheric parameters.

    python3 bayesian_inversion/run_inversion.py --target sample81
    python3 bayesian_inversion/run_inversion.py --target observed

Run ``sample81`` first. It inverts the held-out high-fidelity design point, whose
true parameters are known, so it answers "does this machinery recover an answer we
already have?" before any effort goes into the real observation.

These runs take many hours, so the chain is checkpointed periodically and
``--resume`` restarts from the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import emcee
import numpy as np
import pandas as pd

from bayesian_inversion.emulator import DEFAULT_WORKERS, MFEmulator, benchmark
from bayesian_inversion.noise import emulator_error_covariance, error_budget_summary
from bayesian_inversion.observations import load_observation
from bayesian_inversion.plot_posterior import plot_from_run
from bayesian_inversion.posterior import LogPosterior
from bayesian_inversion.priors import (
    EMULATOR_BOX,
    N_PARAMS,
    PARAMETER_NAMES,
    PRIOR_SPEC,
    sample_prior,
)

RESULTS_DIR = PROJECT_ROOT / "results/bayesian_inversion"


def initial_walkers(n_walkers: int, rng: np.random.Generator) -> np.ndarray:
    """Start every walker at an independent prior draw.

    Deliberately not a tight ball around a MAP estimate: the posterior is expected
    to have degenerate ridges (C/O against the abundances), and a dispersed start
    makes it obvious if separate walkers settle into separate regions.
    """
    return sample_prior(n_walkers, rng)


def checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "chain": output_dir / "chain.npy",
        "log_prob": output_dir / "log_prob.npy",
        "state": output_dir / "checkpoint_state.npy",
    }


def save_checkpoint(
    output_dir: Path,
    sampler: emcee.EnsembleSampler,
    previous_chain: np.ndarray | None,
    previous_log_prob: np.ndarray | None,
) -> None:
    """Persist the full chain: everything from earlier runs plus this one.

    A resumed run gets a fresh ``EnsembleSampler`` holding only the new steps, so
    the earlier chain has to be carried explicitly and concatenated here. Writing
    ``sampler.get_chain()`` alone would silently discard every step before the
    resume.
    """
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()

    if previous_chain is not None:
        chain = np.concatenate([previous_chain, chain], axis=0)
        log_prob = np.concatenate([previous_log_prob, log_prob], axis=0)

    paths = checkpoint_paths(output_dir)
    np.save(paths["chain"], chain)
    np.save(paths["log_prob"], log_prob)
    np.save(paths["state"], sampler.get_last_sample().coords)


def load_checkpoint(
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Return (last positions, chain, log_prob, steps done), or None if absent."""
    paths = checkpoint_paths(output_dir)
    if not all(path.exists() for path in paths.values()):
        return None
    chain = np.load(paths["chain"])
    log_prob = np.load(paths["log_prob"])
    return np.load(paths["state"]), chain, log_prob, chain.shape[0]


def posterior_summary(
    samples: np.ndarray, truth: np.ndarray | None = None
) -> pd.DataFrame:
    """Per-parameter marginal summary, plus where the truth falls if we know it."""
    quantiles = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)

    summary = pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "prior": [
                f"{PRIOR_SPEC[name][0][:4]} {PRIOR_SPEC[name][1]}"
                for name in PARAMETER_NAMES
            ],
            "median": quantiles[2],
            "lo_68": quantiles[1],
            "hi_68": quantiles[3],
            "lo_95": quantiles[0],
            "hi_95": quantiles[4],
        }
    )

    if truth is not None:
        # Percentile rank of the truth within the marginal: ~50 is ideal, and
        # values pinned near 0 or 100 mean the posterior has missed it.
        summary["truth"] = truth
        summary["truth_percentile"] = [
            100.0 * (samples[:, i] < truth[i]).mean() for i in range(N_PARAMS)
        ]
        summary["within_95"] = (truth >= quantiles[0]) & (truth <= quantiles[4])

    # How much the data actually told us, relative to the prior range.
    summary["width_95_vs_prior"] = (quantiles[4] - quantiles[0]) / (
        EMULATOR_BOX[:, 1] - EMULATOR_BOX[:, 0]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["sample81", "observed"], required=True)
    parser.add_argument("--nwalkers", type=int, default=96)
    parser.add_argument("--nsteps", type=int, default=4000)
    parser.add_argument("--burn-frac", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagonal-emulator-error",
        action="store_true",
        help="ignore cross-wavelength correlation in the emulator error (sensitivity check)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.nwalkers < 2 * N_PARAMS:
        parser.error(f"--nwalkers must be at least {2 * N_PARAMS} for {N_PARAMS} parameters")

    output_dir = RESULTS_DIR / args.target
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    observation = load_observation(args.target)

    print(f"target        : {observation.name}")
    print(f"wavelengths   : {len(observation.y)}")

    emulator_cov = emulator_error_covariance(diagonal=args.diagonal_emulator_error)
    budget = error_budget_summary(observation, emulator_cov)
    budget.to_csv(output_dir / "error_budget.csv", index=False)
    print(
        "error budget  : emulator/observational sd ratio, median "
        f"{budget['ratio_emulator_to_obs'].median():.2f}"
    )

    with MFEmulator(n_workers=args.workers) as emulator:
        ms_per_theta = benchmark(emulator, batch_size=args.nwalkers)
        seconds_per_step = ms_per_theta * args.nwalkers / 1000.0
        print(
            f"throughput    : {ms_per_theta:.0f} ms per parameter vector, "
            f"{seconds_per_step:.0f} s per step "
            f"-> {seconds_per_step * args.nsteps / 3600:.1f} h for {args.nsteps} steps"
        )

        log_posterior = LogPosterior(emulator, observation, emulator_cov)

        start_positions = initial_walkers(args.nwalkers, rng)
        previous_chain: np.ndarray | None = None
        previous_log_prob: np.ndarray | None = None
        steps_done = 0

        if args.resume:
            checkpoint = load_checkpoint(output_dir)
            if checkpoint is None:
                print("resume        : no checkpoint found, starting fresh")
            else:
                start_positions, previous_chain, previous_log_prob, steps_done = checkpoint
                if start_positions.shape[0] != args.nwalkers:
                    parser.error(
                        f"checkpoint has {start_positions.shape[0]} walkers but "
                        f"--nwalkers is {args.nwalkers}; they must match to resume"
                    )
                print(f"resume        : continuing from step {steps_done}")

        remaining = args.nsteps - steps_done
        if remaining <= 0:
            print(f"nothing to do: {steps_done} steps already complete")
            return

        sampler = emcee.EnsembleSampler(
            args.nwalkers, N_PARAMS, log_posterior, vectorize=True
        )

        print(f"sampling      : {remaining} steps x {args.nwalkers} walkers\n")
        started = time.time()
        for completed, _ in enumerate(
            sampler.sample(start_positions, iterations=remaining, progress=False), start=1
        ):
            if completed % args.checkpoint_every == 0 or completed == remaining:
                save_checkpoint(output_dir, sampler, previous_chain, previous_log_prob)
                elapsed = time.time() - started
                print(
                    f"  step {completed:>5}/{remaining}  "
                    f"acceptance {np.mean(sampler.acceptance_fraction):.3f}  "
                    f"elapsed {elapsed / 3600:.2f} h  "
                    f"eta {elapsed / completed * (remaining - completed) / 3600:.2f} h",
                    flush=True,
                )

    # Analyse the full chain (earlier runs included), not just this invocation's.
    full_chain = np.load(checkpoint_paths(output_dir)["chain"])
    burn_in = int(args.burn_frac * full_chain.shape[0])
    post_burn_in = full_chain[burn_in:]
    samples = post_burn_in.reshape(-1, N_PARAMS)
    np.save(output_dir / "samples_flat.npy", samples)

    try:
        autocorr = emcee.autocorr.integrated_time(post_burn_in)
    except emcee.autocorr.AutocorrError as error:
        # Expected on short runs; the chain is still saved and resumable.
        print(f"\nwarning: {error}")
        autocorr = emcee.autocorr.integrated_time(post_burn_in, quiet=True)

    summary = posterior_summary(samples, observation.truth)
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)

    config = {
        "target": args.target,
        "model_source": "results/model1/full_fit (MF, HF sample 81 held out)",
        "nwalkers": args.nwalkers,
        "nsteps": args.nsteps,
        "steps_in_chain": int(full_chain.shape[0]),
        "burn_in": burn_in,
        "n_samples": int(len(samples)),
        "seed": args.seed,
        "diagonal_emulator_error": bool(args.diagonal_emulator_error),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": {
            name: (None if not np.isfinite(tau) else float(tau))
            for name, tau in zip(PARAMETER_NAMES, autocorr)
        },
        "emulator_evaluations": log_posterior.n_emulator_evaluations,
        "prior_spec": {name: list(PRIOR_SPEC[name]) for name in PARAMETER_NAMES},
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    print(f"\nacceptance fraction : {np.mean(sampler.acceptance_fraction):.3f}")
    if np.isfinite(autocorr).any():
        tau_max = np.nanmax(autocorr)
        # emcee's rule of thumb: the chain should be at least 50 tau long.
        verdict = "ok" if post_burn_in.shape[0] >= 50 * tau_max else "TOO SHORT"
        print(
            f"max autocorr time   : {tau_max:.1f} steps "
            f"({post_burn_in.shape[0] / tau_max:.0f} tau in chain, {verdict})"
        )
    else:
        print("max autocorr time   : not estimable, chain far too short")
    print(f"posterior samples   : {len(samples)}\n")
    print(summary.to_string(index=False))

    if observation.truth is not None:
        covered = int(summary["within_95"].sum())
        print(
            f"\nclosed-loop test: truth inside the 95% interval for "
            f"{covered}/{N_PARAMS} parameters"
        )

    plot_from_run(args.target)
    print(f"\nwritten to {output_dir}")


if __name__ == "__main__":
    main()
