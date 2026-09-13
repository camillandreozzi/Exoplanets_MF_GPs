#!/usr/bin/env bash
#SBATCH --job-name=loo_lanes
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --ntasks=1
#
# One array task fits its own stride of the lane list. Lanes write themselves
# to disk as they finish, so a task that hits its time limit still leaves every
# lane it completed behind; resubmitting picks up from there.
#
# Submitted by euler/submit_loo.sh, which sets the resource flags and exports
# the CV_LOO_* settings. Everything below reads from the environment.

set -euo pipefail

PROJECT_DIR="${CV_LOO_PROJECT_DIR:?CV_LOO_PROJECT_DIR must be set by submit_loo.sh}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/euler/cluster_env.sh"

# SLURM_ARRAY_TASK_ID is 0-based here because submit_loo.sh asks for 0-(n-1),
# which is exactly the shard numbering cv_loo.py expects.
export CV_LOO_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export CV_LOO_CPU="${SLURM_CPUS_PER_TASK:-${CV_LOO_CPU:-1}}"

# The array tasks only fit lanes; the dependent assemble job writes the
# summary CSVs once, after every shard has finished.
export CV_LOO_WRITE_OUTPUTS=0

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} shard=${CV_LOO_SHARD_INDEX}/${CV_LOO_SHARD_COUNT:-1} cpus=${CV_LOO_CPU}"
echo "started $(date "+%Y-%m-%dT%H:%M:%S")"
echo

python3 -u modelling/cv/cv_loo.py

echo
echo "finished $(date "+%Y-%m-%dT%H:%M:%S")"
