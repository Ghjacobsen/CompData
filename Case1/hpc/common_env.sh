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

if [[ ! -d ".venv" ]]; then
  python3.12 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pandas scipy

export PYTHONPATH="${COMPDATA_ROOT}/Case1/src:${PYTHONPATH:-}"
