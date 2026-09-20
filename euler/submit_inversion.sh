#!/usr/bin/env bash
# Submit the Bayesian inversion MCMC runs on Euler. Run on a login node, from
# the project directory.
#
#   ./euler/submit_inversion.sh
#   ./euler/submit_inversion.sh --after 12345678      wait for the full-fit job
#   ./euler/submit_inversion.sh --targets observed --nsteps 40000
#
# Submits one job per target. `observed` is submitted at default priority and
# `sample81` is niced behind it, so when the two compete for the same cores the
# real observation runs first.
#
# Chains checkpoint every --checkpoint-every steps and the job script resumes
# from the checkpoint, so a job that hits its wall clock is resubmitted with the
# same command and carries on from where it stopped.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# 4000 steps left the observed chain at tau_max = 158, i.e. 12.7 tau post
# burn-in against emcee's 50 tau rule of thumb. 30000 steps puts 15000 steps
# past the burn-in, ~95 tau at that tau and still ~60 tau if it drifts to 250.
NSTEPS="${EULER_INVERSION_NSTEPS:-30000}"
TARGETS="observed,sample81"
MODEL="${EULER_INVERSION_MODEL:-model3}"
# Per-step cost is linear in walkers (ms_per_theta * nwalkers), so 48 rather
# than emcee's default 96 halves the wall clock. 48 is still 5.3x the 9
# parameters, well above the 2x floor, and tau is measured in steps, so the
# shorter step does not cost convergence.
NWALKERS="${EULER_INVERSION_NWALKERS:-48}"
# 195 wavelengths stride across the workers, so cores buy throughput almost
# linearly until each worker holds one model. 48 is the ceiling: Euler's
# cli_filter rejects anything above it for guest users, at submission time.
CORES="${EULER_INVERSION_CORES:-48}"
# 48 x 2g = 96 GiB, under the guest ceiling. Every worker currently loads all
# 195 models rather than just its own stride, so total memory grows with cores.
MEM_PER_CPU="${EULER_MEM_PER_CPU:-2g}"
TIME_LIMIT="${EULER_INVERSION_TIME:-48:00:00}"

# Euler's guest limits, enforced by a cli_filter at submission.
MAX_CORES=48
MAX_MEM_GIB=128
AFTER_JOB=""
DRY_RUN=0

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'HELP'

Options:
  --targets LIST     observed, sample81, or a comma list (default both)
  --model NAME       Forward model: model3 (default) or model1
  --nwalkers N       Ensemble walkers (default 48); cost is linear in this
  --nsteps N         Total steps in the chain, resume included (default 30000)
  --after JOBID      Start only once JOBID has finished successfully
  --cores N          Cores per job; also the emulator worker count (default 48, the cap)
  --mem SIZE         Memory per core (default 4g)
  --time HH:MM:SS    Wall-clock limit per job (default 48:00:00)
  --dry-run          Print the sbatch commands without submitting
HELP
}

while [ $# -gt 0 ]; do
    case "$1" in
        --targets)  TARGETS="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --nwalkers) NWALKERS="$2"; shift 2 ;;
        --nsteps)   NSTEPS="$2"; shift 2 ;;
        --after)    AFTER_JOB="$2"; shift 2 ;;
        --cores)    CORES="$2"; shift 2 ;;
        --mem)      MEM_PER_CPU="$2"; shift 2 ;;
        --time)     TIME_LIMIT="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

contains() { case ",$1," in *",$2,"*) return 0 ;; *) return 1 ;; esac; }

CHECK_VENV="${EULER_VENV:-$HOME/venvs/exoplanets}"
if [ ! -f "${CHECK_VENV}/bin/activate" ] && [ "${DRY_RUN}" = "0" ]; then
    echo "No virtual environment at ${CHECK_VENV}." >&2
    echo "Build it first, from your laptop:  ./euler/euler.sh setup" >&2
    exit 1
fi

# Check the guest caps here. sbatch enforces them too, but only after the
# script has printed a full settings banner, which reads like a successful
# submission right up until the rejection.
mem_gib="${MEM_PER_CPU%[gG]}"
total_gib=$((CORES * mem_gib))
if [ "${CORES}" -gt "${MAX_CORES}" ]; then
    echo "--cores ${CORES} exceeds the guest cap of ${MAX_CORES}." >&2
    exit 2
fi
if [ "${total_gib}" -gt "${MAX_MEM_GIB}" ]; then
    echo "${CORES} cores x ${MEM_PER_CPU} = ${total_gib} GiB, over the guest" \
         "cap of ${MAX_MEM_GIB} GiB. Lower --mem or --cores." >&2
    exit 2
fi

LOG_DIR="${PROJECT_DIR}/results/bayesian_inversion/logs"
[ "${DRY_RUN}" = "1" ] || mkdir -p "${LOG_DIR}"

echo "Model        : ${MODEL} (MF)"
echo "Targets      : ${TARGETS}"
echo "Steps        : ${NSTEPS} (total in chain, resume included)"
echo "Walkers      : ${NWALKERS}"
echo "Resources    : ${CORES} cores, ${MEM_PER_CPU}/core, ${TIME_LIMIT}"
[ -n "${AFTER_JOB}" ] && echo "Dependency   : afterok:${AFTER_JOB}"
echo "Logs         : ${LOG_DIR}"
echo

submit_target() {
    local target="$1" nice="$2"

    local sbatch_cmd=(
        sbatch
        --parsable
        --job-name="inv_${target}"
        --cpus-per-task="${CORES}"
        --mem-per-cpu="${MEM_PER_CPU}"
        --time="${TIME_LIMIT}"
        --nice="${nice}"
        --output="${LOG_DIR}/%x_%j.out"
        --error="${LOG_DIR}/%x_%j.out"
        # --export carries the settings; sbatch's -e is --error, not an env flag.
        --export="ALL,INVERSION_TARGET=${target},INVERSION_MODEL=${MODEL},INVERSION_NSTEPS=${NSTEPS},INVERSION_NWALKERS=${NWALKERS},INVERSION_PROJECT_DIR=${PROJECT_DIR},MPLBACKEND=Agg"
    )
    [ -n "${AFTER_JOB}" ] && sbatch_cmd+=(
        --dependency="afterok:${AFTER_JOB}" --kill-on-invalid-dep=yes
    )
    sbatch_cmd+=("${PROJECT_DIR}/euler/inversion.sbatch")

    if [ "${DRY_RUN}" = "1" ]; then
        printf '%q ' "${sbatch_cmd[@]}"; echo; echo
    else
        local job_id
        job_id="$("${sbatch_cmd[@]}")"
        echo "Submitted ${target} as job ${job_id} (nice ${nice})"
    fi
}

# nice 0 outranks nice 100: the real observation wins any contest for cores.
contains "${TARGETS}" observed && submit_target observed 0
contains "${TARGETS}" sample81 && submit_target sample81 100

echo
echo "Watch  : squeue -u \$USER"
echo "Logs   : tail -f ${LOG_DIR}/inv_observed_*.out"
echo "Fetch  : ./euler/euler.sh down   (from your laptop)"
