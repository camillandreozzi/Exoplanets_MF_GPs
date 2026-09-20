#!/usr/bin/env bash
# Submit the Model 3 full fit on Euler. Run on a login node, from the project
# directory.
#
#   ./euler/submit_full_fit_model3.sh
#   ./euler/submit_full_fit_model3.sh --cores 24 --time 120:00:00
#
# Prints the job id on the last line, so the inversion can be chained onto it:
#
#   ./euler/submit_inversion.sh --after <job id>

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

CORES="${EULER_MODEL3_CORES:-16}"
MEM_PER_CPU="${EULER_MEM_PER_CPU:-4g}"
TIME_LIMIT="${EULER_MODEL3_TIME:-120:00:00}"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --cores)   CORES="$2"; shift 2 ;;
        --mem)     MEM_PER_CPU="$2"; shift 2 ;;
        --time)    TIME_LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

CHECK_VENV="${EULER_VENV:-$HOME/venvs/exoplanets}"
if [ ! -f "${CHECK_VENV}/bin/activate" ] && [ "${DRY_RUN}" = "0" ]; then
    echo "No virtual environment at ${CHECK_VENV}." >&2
    echo "Build it first, from your laptop:  ./euler/euler.sh setup" >&2
    exit 1
fi

LOG_DIR="${PROJECT_DIR}/results/model3/full_fit/logs"
[ "${DRY_RUN}" = "1" ] || mkdir -p "${LOG_DIR}"

# nullglob, not `ls | wc -l`: an unmatched glob makes ls exit 2, and under
# `set -e` with pipefail that kills the script before it prints anything.
shopt -s nullglob
existing=("${PROJECT_DIR}"/results/model3/full_fit/model3_mf_response_*.pkl)
shopt -u nullglob
done_count=${#existing[@]}
echo "Resources    : ${CORES} cores, ${MEM_PER_CPU}/core, ${TIME_LIMIT}"
echo "Already done : ${done_count}/195 MF wavelengths (these are skipped)"
echo "Logs         : ${LOG_DIR}"
echo

SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="model3_full_fit"
    --cpus-per-task="${CORES}"
    --mem-per-cpu="${MEM_PER_CPU}"
    --time="${TIME_LIMIT}"
    --output="${LOG_DIR}/%x_%j.out"
    --error="${LOG_DIR}/%x_%j.out"
    --export="ALL,MODEL3_PROJECT_DIR=${PROJECT_DIR}"
    "${PROJECT_DIR}/euler/full_fit_model3.sbatch"
)

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${SBATCH_CMD[@]}"; echo
    exit 0
fi

JOB_ID="$("${SBATCH_CMD[@]}")"
echo "Submitted Model 3 full fit as job ${JOB_ID}"
echo
echo "Chain the inversion onto it:"
echo "  ./euler/submit_inversion.sh --after ${JOB_ID}"
echo "${JOB_ID}"
