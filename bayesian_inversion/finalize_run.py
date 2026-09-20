"""Write the summary products for a chain that stopped before run_inversion did.

A chain killed by its wall clock leaves ``chain.npy`` on disk but never reaches
the analysis block at the end of ``run_inversion.main``, so it has no
``samples_flat.npy``, no ``posterior_summary.csv`` and no triangle plot.
Resuming does not help: with ``--nsteps`` already satisfied, main() prints
"nothing to do" and returns before the analysis.

    python3 bayesian_inversion/finalize_run.py --target observed
    python3 bayesian_inversion/finalize_run.py --target sample81 --model model3

This reads the checkpoint only -- it never loads the emulator -- so it is cheap
and runs anywhere. It records the convergence diagnostics as measured, flagging
the chain as unconverged when it is, rather than quietly presenting the
quantiles as if the sampler had finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import emcee
import numpy as np

from bayesian_inversion.emulator import DEFAULT_FAMILY, MODEL_FAMILIES
from bayesian_inversion.noise import LOO_PREDICTIONS
from bayesian_inversion.observations import load_observation
from bayesian_inversion.plot_posterior import plot_from_run
from bayesian_inversion.priors import N_PARAMS, PARAMETER_NAMES, PRIOR_SPEC
from bayesian_inversion.run_inversion import RESULTS_DIR, posterior_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["sample81", "observed"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_FAMILIES), default=DEFAULT_FAMILY)
    parser.add_argument("--burn-frac", type=float, default=0.5)
    parser.add_argument(
        "--acceptance",
        type=float,
        default=None,
        help="acceptance fraction from the run log; recorded if given",
    )
    args = parser.parse_args()

    results_dir = RESULTS_DIR / args.model
    output_dir = results_dir / args.target
    chain_path = output_dir / "chain.npy"
    if not chain_path.exists():
        parser.error(f"no chain at {chain_path}")

    chain = np.load(chain_path)
    steps, walkers, ndim = chain.shape
    burn_in = int(args.burn_frac * steps)
    post_burn_in = chain[burn_in:]
    samples = post_burn_in.reshape(-1, ndim)
    np.save(output_dir / "samples_flat.npy", samples)

    autocorr = emcee.autocorr.integrated_time(post_burn_in, quiet=True)
    tau_max = float(np.nanmax(autocorr))
    n_tau = post_burn_in.shape[0] / tau_max
    converged = bool(n_tau >= 50)

    observation = load_observation(args.target)
    summary = posterior_summary(samples, observation.truth)
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)

    config = {
        "target": args.target,
        "model": args.model,
        "model_source": f"results/{args.model}/full_fit (MF, HF sample 81 held out)",
        "emulator_residual_source": str(LOO_PREDICTIONS.relative_to(PROJECT_ROOT)),
        "nwalkers": walkers,
        "steps_in_chain": steps,
        "burn_in": burn_in,
        "n_samples": int(len(samples)),
        "acceptance_fraction": args.acceptance,
        "autocorr_time": {
            name: (None if not np.isfinite(t) else float(t))
            for name, t in zip(PARAMETER_NAMES, autocorr)
        },
        "tau_max": tau_max,
        "chain_length_in_tau": n_tau,
        "converged": converged,
        "convergence_note": (
            "post-burn-in length is {:.0f} tau against emcee's 50 tau rule of "
            "thumb; intervals below are NOT converged and tau estimated from a "
            "chain this short is itself biased low".format(n_tau)
            if not converged
            else "post-burn-in length is {:.0f} tau, at or above the 50 tau "
            "rule of thumb".format(n_tau)
        ),
        "finalized_from_checkpoint": True,
        "prior_spec": {name: list(PRIOR_SPEC[name]) for name in PARAMETER_NAMES},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )

    print(f"target        : {args.target}  ({args.model})")
    print(f"chain         : {steps:,} steps x {walkers} walkers, burn-in {burn_in:,}")
    print(f"posterior     : {len(samples):,} samples")
    print(f"max autocorr  : {tau_max:.1f} steps ({n_tau:.0f} tau in chain)")
    print(f"converged     : {'yes' if converged else 'NO -- see convergence_note'}")
    print()
    print(summary.to_string(index=False))

    if observation.truth is not None:
        covered = int(summary["within_95"].sum())
        print(
            f"\nclosed-loop test: truth inside the 95% interval for "
            f"{covered}/{N_PARAMS} parameters"
        )

    plot_from_run(args.target, results_dir)
    print(f"\nwritten to {output_dir}")


if __name__ == "__main__":
    main()
