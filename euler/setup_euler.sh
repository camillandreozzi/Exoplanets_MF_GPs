#!/usr/bin/env bash
# Build the Python environment on Euler. Run once, on a login node:
#
#   ./euler/setup_euler.sh
#
# Safe to rerun: it reuses an existing virtual environment and reinstalls the
# pinned requirements into it.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EULER_VENV="${EULER_VENV:-$HOME/venvs/exoplanets}"
export EULER_VENV

# The venv does not exist yet on a first run, so load the modules without it.
EULER_SKIP_VENV=1 source "${PROJECT_DIR}/euler/cluster_env.sh"

echo "Python: $(command -v python3) ($(python3 --version))"

if [ ! -f "${EULER_VENV}/bin/activate" ]; then
    echo "Creating virtual environment at ${EULER_VENV}"
    mkdir -p "$(dirname "${EULER_VENV}")"
    python3 -m venv "${EULER_VENV}"
fi

# shellcheck disable=SC1091
source "${EULER_VENV}/bin/activate"

python3 -m pip install --upgrade pip wheel
python3 -m pip install -r "${PROJECT_DIR}/euler/requirements.txt"

echo
echo "Import check:"
python3 - <<'PYTHON'
import gpboost, numpy, optuna, pandas, scipy, sklearn

for module in (gpboost, numpy, pandas, scipy, sklearn, optuna):
    print(f"  {module.__name__:12s} {module.__version__}")
PYTHON

echo
echo "Data check:"
cd "${PROJECT_DIR}"
python3 - <<'PYTHON'
from src.data_load import load_full_data

data = load_full_data()
low_fidelity = int(data["is_hf"].eq(0).sum())
high_fidelity = int(data["is_hf"].eq(1).sum())
print(f"  loaded {len(data)} rows: {low_fidelity} LF, {high_fidelity} HF")
PYTHON

echo
echo "Environment ready at ${EULER_VENV}"
