#!/usr/bin/env bash
#SBATCH --job-name=inversion
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.out
#SBATCH --ntasks=1
#
# Runs Bayesian inversion (MCMC sampling) for one target. Checkpoints
# periodically so resumed runs pick up where they left off.
#
# Submitted by euler/submit_inversion.sh, which sets the resource flags and
# exports the INVERSION_* settings.

set -euo pipefail

PROJECT_DIR="${INVERSION_PROJECT_DIR:?INVERSION_PROJECT_DIR must be set by submit_inversion.sh}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/euler/cluster_env.sh"

TARGET="${INVERSION_TARGET:?INVERSION_TARGET must be set}"
MODEL="${INVERSION_MODEL:-model3}"
NSTEPS="${INVERSION_NSTEPS:-30000}"
NWALKERS="${INVERSION_NWALKERS:-48}"
CPU="${SLURM_CPUS_PER_TASK:-${INVERSION_WORKERS:-8}}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} target=${TARGET} model=${MODEL} steps=${NSTEPS} walkers=${NWALKERS} workers=${CPU}"
echo "started $(date "+%Y-%m-%dT%H:%M:%S")"
echo

RUN_ARGS=(
    --target "${TARGET}"
    --model "${MODEL}"
    --nsteps "${NSTEPS}"
    --nwalkers "${NWALKERS}"
    --workers "${CPU}"
)

# Chains are keyed by model family, so this only ever resumes a chain built
# against the same forward model.
if [ -f "results/bayesian_inversion/${MODEL}/${TARGET}/chain.npy" ]; then
    echo "Resuming from checkpoint..."
    RUN_ARGS+=(--resume)
else
    echo "Starting fresh..."
fi

python3 -u bayesian_inversion/run_inversion.py "${RUN_ARGS[@]}"

echo
echo "finished $(date "+%Y-%m-%dT%H:%M:%S")"
