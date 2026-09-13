#!/usr/bin/env bash
# Sourced on Euler by setup_euler.sh, submit_loo.sh, and both sbatch scripts.
# Puts the right Python on PATH and pins the numeric libraries to one thread.

set -euo pipefail

EULER_VENV="${EULER_VENV:-$HOME/venvs/exoplanets}"
EULER_MODULES="${EULER_MODULES:-stack/2024-06 gcc/12.2.0 python/3.11.6}"

# `module` is a shell function, and the profile script that defines it is only
# sourced by login shells. Neither `ssh host "cmd"` nor a SLURM job script is
# one, so bootstrap it here rather than relying on the caller's shell.
if ! command -v module >/dev/null 2>&1; then
    for lmod_init in \
        /cluster/apps/lmod/lmod/init/bash \
        "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash" \
        /etc/profile.d/lmod.sh \
        /cluster/apps/local/env2lmod.sh
    do
        if [ -f "${lmod_init}" ]; then
            # shellcheck disable=SC1090
            source "${lmod_init}"
            break
        fi
    done
fi

if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load ${EULER_MODULES}
elif [ "${EULER_ALLOW_NO_MODULES:-0}" = "1" ]; then
    echo "No module system found; using the python3 already on PATH." >&2
else
    echo "Could not find the module system on this host." >&2
    echo "Run this on Euler, or set EULER_ALLOW_NO_MODULES=1 to use PATH." >&2
    exit 1
fi

if [ "${EULER_SKIP_VENV:-0}" != "1" ]; then
    if [ ! -f "${EULER_VENV}/bin/activate" ]; then
        echo "No virtual environment at ${EULER_VENV}." >&2
        echo "Run euler/setup_euler.sh on a login node first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${EULER_VENV}/bin/activate"
fi

# One thread per process. cv_loo.py sets these too, but a worker inherits the
# job's environment first, and BLAS reads them at import time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optuna and joblib otherwise scribble into $HOME, which has a small quota.
export TMPDIR="${TMPDIR:-${SCRATCH:-/tmp}/tmp}"
mkdir -p "${TMPDIR}"
