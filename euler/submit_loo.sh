#!/usr/bin/env bash
# Submit a LOO CV run on Euler. Run this on a login node, from the project
# directory. The local driver (euler/euler.sh run) calls it over ssh for you.
#
#   ./euler/submit_loo.sh --models 1,3
#   ./euler/submit_loo.sh --models all --lf 10000 --cores 24 --time 48:00:00
#   ./euler/submit_loo.sh --models 3 --max-folds 2 --array 1 --time 01:00:00
#
# Submits a job array whose tasks each fit a stride of the lane list, plus one
# dependent job that assembles every lane file into the summary CSVs. Lanes
# already on disk are skipped, so resubmitting the same command resumes a run
# that ran out of time.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

MODELS="1,2,3"
VARIANTS="sf,mf"
LF_SAMPLE_SIZE=""
MAX_FOLDS=""
N_RESPONSE_SAMPLE=""
MODEL2_N_WAVELENGTHS=""
RESULTS_DIR=""
RUN_TAG=""
CORES="${EULER_CORES:-16}"
MEM_PER_CPU="${EULER_MEM_PER_CPU:-4g}"
TIME_LIMIT="${EULER_TIME:-24:00:00}"
ARRAY_SIZE="${EULER_ARRAY_SIZE:-10}"
FRESH=0
DRY_RUN=0

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'HELP'

Options:
  --models LIST      Models to run: 1, 2, 3, a comma list, or "all" (default 1,2,3)
  --variants LIST    sf, mf, or both (default sf,mf)
  --lf N             Low-fidelity training rows, or "none" for all 10000
  --max-folds N      Cap the LOO folds; use a small number for a pilot run
  --wavelengths N    Fit an N-wavelength subset instead of all 195
  --model2-wl N      Wavelengths sampled per Model 2 training spectrum
  --results-dir DIR  Results directory (default results/cv/loo_<tag>)
  --tag NAME         Name for the default results directory and job names
  --cores N          Cores, and therefore workers, per array task
  --mem SIZE         Memory per core (default 4g)
  --time HH:MM:SS    Wall-clock limit per array task
  --array N          Number of array tasks
  --fresh            Delete the lane directory before submitting
  --dry-run          Print the sbatch commands without submitting
HELP
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models)       MODELS="$2"; shift 2 ;;
        --variants)     VARIANTS="$2"; shift 2 ;;
        --lf)           LF_SAMPLE_SIZE="$2"; shift 2 ;;
        --max-folds)    MAX_FOLDS="$2"; shift 2 ;;
        --wavelengths)  N_RESPONSE_SAMPLE="$2"; shift 2 ;;
        --model2-wl)    MODEL2_N_WAVELENGTHS="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --tag)          RUN_TAG="$2"; shift 2 ;;
        --cores)        CORES="$2"; shift 2 ;;
        --mem)          MEM_PER_CPU="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --array)        ARRAY_SIZE="$2"; shift 2 ;;
        --fresh)        FRESH=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

contains() { case ",$1," in *",$2,"*) return 0 ;; *) return 1 ;; esac; }

if [ "${MODELS}" = "all" ]; then MODELS="1,2,3"; fi
if [ "${VARIANTS}" = "both" ]; then VARIANTS="sf,mf"; fi

export CV_LOO_RUN_MODEL1=$(contains "${MODELS}" 1 && echo 1 || echo 0)
export CV_LOO_RUN_MODEL2=$(contains "${MODELS}" 2 && echo 1 || echo 0)
export CV_LOO_RUN_MODEL3=$(contains "${MODELS}" 3 && echo 1 || echo 0)
export CV_LOO_RUN_SF=$(contains "${VARIANTS}" sf && echo 1 || echo 0)
export CV_LOO_RUN_MF=$(contains "${VARIANTS}" mf && echo 1 || echo 0)

if [ "${CV_LOO_RUN_MODEL1}${CV_LOO_RUN_MODEL2}${CV_LOO_RUN_MODEL3}" = "000" ]; then
    echo "No models selected; --models takes 1, 2, 3, or a comma list." >&2
    exit 2
fi
if [ "${CV_LOO_RUN_SF}${CV_LOO_RUN_MF}" = "00" ]; then
    echo "No variants selected; --variants takes sf, mf, or both." >&2
    exit 2
fi

[ -n "${LF_SAMPLE_SIZE}" ] && export CV_LOO_LF_SAMPLE_SIZE="${LF_SAMPLE_SIZE}"
[ -n "${MAX_FOLDS}" ] && export CV_LOO_MAX_FOLDS="${MAX_FOLDS}"
[ -n "${N_RESPONSE_SAMPLE}" ] && export CV_LOO_N_RESPONSE_SAMPLE="${N_RESPONSE_SAMPLE}"
[ -n "${MODEL2_N_WAVELENGTHS}" ] && export CV_LOO_MODEL2_N_WAVELENGTHS="${MODEL2_N_WAVELENGTHS}"

# One results directory per training set. The LF sample size goes in the
# default name because mixing two sample sizes in one directory is what
# silently invalidates a run: the lane files look interchangeable but were fit
# against different training data, and run_config.json records only the last
# run to touch it.
if [ -z "${RUN_TAG}" ]; then
    RUN_TAG="lf${LF_SAMPLE_SIZE:-1000}"
fi
if [ -z "${RESULTS_DIR}" ]; then
    RESULTS_DIR="results/cv/loo_${RUN_TAG}"
fi
export CV_LOO_RESULTS_DIR="${RESULTS_DIR}"
export CV_LOO_SHARD_COUNT="${ARRAY_SIZE}"
export CV_LOO_PROJECT_DIR="${PROJECT_DIR}"

ABSOLUTE_RESULTS_DIR="${RESULTS_DIR}"
case "${ABSOLUTE_RESULTS_DIR}" in /*) ;; *) ABSOLUTE_RESULTS_DIR="${PROJECT_DIR}/${RESULTS_DIR}" ;; esac

if [ "${FRESH}" = "1" ]; then
    # cv_loo.py refuses FRESH_RUN across shards, since every array task would
    # delete the lanes its siblings are writing. Clear it once, here.
    echo "Clearing ${ABSOLUTE_RESULTS_DIR}/lanes"
    [ "${DRY_RUN}" = "1" ] || rm -rf "${ABSOLUTE_RESULTS_DIR}/lanes"
fi

# Check the environment here rather than letting every array task discover it
# independently: a missing venv otherwise queues N doomed tasks that each fail
# a few seconds after starting, plus an assemble job that finds no lanes.
CHECK_VENV="${EULER_VENV:-$HOME/venvs/exoplanets}"
if [ ! -f "${CHECK_VENV}/bin/activate" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
        echo "WARNING: no virtual environment at ${CHECK_VENV} (ignored for --dry-run)" >&2
    else
        echo "No virtual environment at ${CHECK_VENV}." >&2
        echo "Build it first, from your laptop:  ./euler/euler.sh setup" >&2
        echo "or on a login node:                ./euler/setup_euler.sh" >&2
        exit 1
    fi
fi

LOG_DIR="${ABSOLUTE_RESULTS_DIR}/logs"
[ "${DRY_RUN}" = "1" ] || mkdir -p "${LOG_DIR}"

JOB_NAME="loo_${RUN_TAG}_m${MODELS//,/}"

SBATCH_LANES=(
    sbatch
    --parsable
    --job-name="${JOB_NAME}"
    --array="0-$((ARRAY_SIZE - 1))"
    --cpus-per-task="${CORES}"
    --mem-per-cpu="${MEM_PER_CPU}"
    --time="${TIME_LIMIT}"
    --output="${LOG_DIR}/%x_%A_%a.out"
    --error="${LOG_DIR}/%x_%A_%a.out"
    --export=ALL
    "${PROJECT_DIR}/euler/loo_lanes.sbatch"
)

echo "Models       : ${MODELS}   variants: ${VARIANTS}"
echo "Results dir  : ${ABSOLUTE_RESULTS_DIR}"
echo "Array        : ${ARRAY_SIZE} tasks x ${CORES} cores, ${MEM_PER_CPU}/core, ${TIME_LIMIT}"
echo "Totals       : $((ARRAY_SIZE * CORES)) cores at ${MEM_PER_CPU} each, held at once"
echo "LF rows      : ${LF_SAMPLE_SIZE:-1000 (default)}   max folds: ${MAX_FOLDS:-all}"
echo "Logs         : ${LOG_DIR}"
echo

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${SBATCH_LANES[@]}"; echo
    echo "(dry run: nothing submitted)"
    exit 0
fi

LANES_JOB_ID="$("${SBATCH_LANES[@]}")"
echo "Submitted lane array ${LANES_JOB_ID}"

ASSEMBLE_JOB_ID="$(sbatch \
    --parsable \
    --job-name="${JOB_NAME}_assemble" \
    --dependency="afterany:${LANES_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.out" \
    --export=ALL \
    "${PROJECT_DIR}/euler/loo_assemble.sbatch")"
echo "Submitted assemble job ${ASSEMBLE_JOB_ID} (runs after the array finishes)"

# A request above the shareholder group's QOS cap is accepted at submission and
# then pends forever with a QOSMax... reason, which looks exactly like ordinary
# queueing. Say so now instead of letting it burn an afternoon.
sleep 5
blocked="$(squeue -u "${USER}" -r -h -t PENDING -o '%R' 2>/dev/null | grep -c 'QOSMax' || true)"
if [ "${blocked:-0}" -gt 0 ]; then
    echo
    echo "WARNING: ${blocked} task(s) are blocked by your QOS limits, not by demand:"
    squeue -u "${USER}" -r -h -t PENDING -o '%R' | grep 'QOSMax' | sort | uniq -c | sed 's/^/    /'
    echo "    You are holding $((ARRAY_SIZE * CORES)) cores at ${MEM_PER_CPU} each."
    echo "    See your cap with:  sacctmgr -n show qos public format=Name,MaxTRESPU%60 -P"
    echo "    Then scancel and resubmit with a smaller --array, --cores or --mem."
fi

echo
echo "Watch    : squeue -u \$USER"
echo "Logs     : tail -f ${LOG_DIR}/${JOB_NAME}_${LANES_JOB_ID}_0.out"
echo "Fetch    : ./euler/euler.sh down   (from your laptop)"
