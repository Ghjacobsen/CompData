#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: run_family.sh <elastic_net|tree_ensembles|pls_pcr> [seed] [outer_splits] [inner_splits]" >&2
  exit 1
fi

APPROACH="$1"
SEED="${2:-42}"
OUTER_SPLITS="${3:-10}"
INNER_SPLITS="${4:-5}"
N_JOBS="${N_JOBS:--1}"

source "$(dirname "$0")/common_env.sh"

RUN_ROOT="Case1/results/runs/${APPROACH}"
mkdir -p "${RUN_ROOT}/nested" "${RUN_ROOT}/refit"

python -m case1_comp.approaches.${APPROACH}.run_nested_cv \
  --data-dir Case1/data \
  --out-dir "${RUN_ROOT}/nested" \
  --seed "${SEED}" \
  --outer-splits "${OUTER_SPLITS}" \
  --inner-splits "${INNER_SPLITS}" \
  --n-jobs "${N_JOBS}"

python -m case1_comp.approaches.${APPROACH}.run_refit_predict \
  --data-dir Case1/data \
  --out-dir "${RUN_ROOT}/refit" \
  --seed "${SEED}" \
  --inner-splits "${INNER_SPLITS}" \
  --n-jobs "${N_JOBS}"
