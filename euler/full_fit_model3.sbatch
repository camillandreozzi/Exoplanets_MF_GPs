#!/usr/bin/env bash
#SBATCH --job-name=model3_full_fit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --ntasks=1
#
# Fits the 195 Model 3 wavelength models (SF and MF) on LF + HF with HF design
# point 81 held out, matching the hold-out the inversion's closed-loop test
# needs. Submitted by euler/submit_full_fit_model3.sh.
#
# The script skips any wavelength already on disk, so a job that runs out of
# time is resumed by resubmitting the same command.

set -euo pipefail

PROJECT_DIR="${MODEL3_PROJECT_DIR:?MODEL3_PROJECT_DIR must be set by submit_full_fit_model3.sh}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/euler/cluster_env.sh"

export MODEL3_PLOT=0
export MPLBACKEND=Agg

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "fitting Model 3 SF+MF, 195 wavelengths, HF sample 81 held out"
echo "started $(date "+%Y-%m-%dT%H:%M:%S")"
echo

python3 -u modelling/full_fit/full_fit_model3.py

echo
echo "finished $(date "+%Y-%m-%dT%H:%M:%S")"
