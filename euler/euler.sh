#!/usr/bin/env bash
# Local driver for the Euler LOO runs. Everything happens from your laptop:
#
#   ./euler/euler.sh setup                  build the cluster environment (once)
#   ./euler/euler.sh run --models 1,2,3     sync the code up and submit
#   ./euler/euler.sh status                 what is queued or running
#   ./euler/euler.sh logs                   tail the newest job log
#   ./euler/euler.sh check --models 3       which lanes are still missing
#   ./euler/euler.sh down                   copy results back into ./results
#
# Settings live in euler/config.sh.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/euler/config.sh"

# Code and data only. Results are pulled the other way, and data/lit is 2.4 GB
# of literature the models never read.
RSYNC_UP_EXCLUDES=(
    --exclude '.git/'
    --exclude 'results/'
    --exclude 'data/lit/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.DS_Store'
    --exclude '*.png'
)

remote_dir() {
    # EULER_REMOTE_DIR may contain $SCRATCH, which only the cluster can expand.
    ssh "${EULER_HOST}" "echo ${EULER_REMOTE_DIR}"
}

remote_run() {
    # Passes the module/venv settings through so config.sh stays the one place
    # they are defined.
    ssh "${EULER_HOST}" \
        "export EULER_VENV=\"${EULER_VENV}\" EULER_MODULES=\"${EULER_MODULES}\"; \
         cd ${EULER_REMOTE_DIR} && $*"
}

sync_up() {
    local target
    target="$(remote_dir)"
    echo "Syncing ${PROJECT_DIR}/ -> ${EULER_HOST}:${target}/"
    ssh "${EULER_HOST}" "mkdir -p ${EULER_REMOTE_DIR}"
    rsync -az --delete-after --stats \
        "${RSYNC_UP_EXCLUDES[@]}" \
        "${PROJECT_DIR}/" "${EULER_HOST}:${target}/"
}

case "${1:-}" in
    setup)
        sync_up
        echo
        remote_run "./euler/setup_euler.sh"
        ;;

    up)
        sync_up
        ;;

    run)
        shift
        sync_up
        echo
        # The submit script needs the resource defaults from config.sh too.
        ssh "${EULER_HOST}" \
            "export EULER_VENV=\"${EULER_VENV}\" EULER_MODULES=\"${EULER_MODULES}\" \
                    EULER_CORES=\"${EULER_CORES}\" EULER_MEM_PER_CPU=\"${EULER_MEM_PER_CPU}\" \
                    EULER_TIME=\"${EULER_TIME}\" EULER_ARRAY_SIZE=\"${EULER_ARRAY_SIZE}\"; \
             cd ${EULER_REMOTE_DIR} && ./euler/submit_loo.sh $*"
        ;;

    status)
        ssh "${EULER_HOST}" \
            "squeue -u \$USER -o '%.10i %.24j %.9T %.10M %.10l %.6D %R'"
        ;;

    logs)
        shift
        # Newest log under any results directory, unless one is named. Uses
        # find rather than a glob: logs sit at <results>/cv/<run>/logs/*.out,
        # and a glob that misses makes `ls` list the working directory instead
        # of reporting that there is nothing there yet.
        ssh -t "${EULER_HOST}" "cd ${EULER_REMOTE_DIR} && "'
            latest=$(find '"${1:-results}"' -path "*/logs/*.out" -type f \
                         -printf "%T@ %p\n" 2>/dev/null |
                     sort -rn | head -1 | cut -d" " -f2-)
            if [ -z "$latest" ]; then
                echo "No job logs yet."
                echo "The array is probably still pending; check with:"
                echo "  ./euler/euler.sh status"
            else
                echo "== $latest =="
                tail -n 40 -f "$latest"
            fi'
        ;;

    check)
        shift
        remote_run "source euler/cluster_env.sh && python3 euler/check_lanes.py" "$@"
        ;;

    down)
        shift
        target="$(remote_dir)"
        subpath="${1:-results}"
        echo "Syncing ${EULER_HOST}:${target}/${subpath}/ -> ${PROJECT_DIR}/${subpath}/"
        mkdir -p "${PROJECT_DIR}/${subpath}"
        # No --delete: local results the cluster never saw are left alone.
        rsync -az --stats \
            "${EULER_HOST}:${target}/${subpath}/" "${PROJECT_DIR}/${subpath}/"
        ;;

    shell)
        ssh -t "${EULER_HOST}" "cd ${EULER_REMOTE_DIR} && exec \$SHELL -l"
        ;;

    *)
        sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
