#!/usr/bin/env bash
#SBATCH --job-name=loo_assemble
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8g
#
# Runs after the lane array with --dependency=afterany, so it still assembles
# whatever finished even if some array tasks hit their time limit. Reports the
# lanes that are still missing rather than failing on them.

set -euo pipefail

PROJECT_DIR="${CV_LOO_PROJECT_DIR:?CV_LOO_PROJECT_DIR must be set by submit_loo.sh}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/euler/cluster_env.sh"

export CV_LOO_ASSEMBLE_ONLY=1
export CV_LOO_WRITE_OUTPUTS=1
export CV_LOO_SHARD_INDEX=0
export CV_LOO_SHARD_COUNT=1
export CV_LOO_CPU=1

echo "assembling in ${CV_LOO_RESULTS_DIR:-results/cv/loo}"
echo

python3 -u modelling/cv/cv_loo.py

echo
python3 -u euler/check_lanes.py
