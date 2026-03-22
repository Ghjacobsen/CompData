#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${COMPDATA_ROOT:-}" ]]; then
  echo "COMPDATA_ROOT is not set. Export it to your CompData path on HPC." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

cd "${COMPDATA_ROOT}"

# Try to get a Python 3.12 interpreter in non-interactive job shells.
if ! command -v python3.12 >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load python3/3.12.11 >/dev/null 2>&1 || \
      module load python3/3.12.9 >/dev/null 2>&1 || \
      module load python3/3.12.7 >/dev/null 2>&1 || \
      true
  fi
fi

PYTHON_BIN="$(command -v python3.12 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python3.12 is not available in PATH." >&2
  echo "Load a python3/3.12.* module or update Case1/hpc/common_env.sh." >&2
  exit 1
fi

if [[ -d ".venv" ]]; then
  VENV_PY_MM="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [[ "${VENV_PY_MM}" != "3.12" ]]; then
    echo "Existing .venv uses Python ${VENV_PY_MM:-unknown}; recreating with Python 3.12." >&2
    rm -rf .venv
  fi
fi

if [[ ! -d ".venv" ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi

source .venv/bin/activate

# Install dependencies only when explicitly requested to avoid parallel pip races.
if [[ "${CASE1_INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn statsmodels
fi

python -c "import numpy, pandas, scipy, sklearn" >/dev/null

export PYTHONPATH="${COMPDATA_ROOT}/Case1/src:${PYTHONPATH:-}"
