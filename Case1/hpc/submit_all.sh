#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPDATA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${COMPDATA_ROOT}"

if [[ -n "${LSB_JOBID:-}" ]]; then
	echo "ERROR: Do not submit submit_all.sh with bsub." >&2
	echo "Run it directly from a login shell: bash ${COMPDATA_ROOT}/Case1/hpc/submit_all.sh" >&2
	exit 1
fi

mkdir -p "${COMPDATA_ROOT}/Case1/results/logs"

# Bootstrap venv and dependencies once before launching parallel jobs.
export COMPDATA_ROOT
export CASE1_INSTALL_DEPS=1
source "${COMPDATA_ROOT}/Case1/hpc/common_env.sh"
deactivate >/dev/null 2>&1 || true
unset CASE1_INSTALL_DEPS

en_out=$(bsub -env "all,COMPDATA_ROOT=${COMPDATA_ROOT}" < "${COMPDATA_ROOT}/Case1/hpc/run_elastic_net.lsf")
tree_out=$(bsub -env "all,COMPDATA_ROOT=${COMPDATA_ROOT}" < "${COMPDATA_ROOT}/Case1/hpc/run_tree_ensembles.lsf")
pls_out=$(bsub -env "all,COMPDATA_ROOT=${COMPDATA_ROOT}" < "${COMPDATA_ROOT}/Case1/hpc/run_pls_pcr.lsf")

en_id=$(echo "$en_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')
tree_id=$(echo "$tree_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')
pls_id=$(echo "$pls_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')

if [[ -z "$en_id" || -z "$tree_id" || -z "$pls_id" ]]; then
	echo "ERROR: Failed to parse one or more LSF job IDs from bsub output." >&2
	echo "EN output:   $en_out" >&2
	echo "Tree output: $tree_out" >&2
	echo "PLS output:  $pls_out" >&2
	exit 1
fi

echo "Submitted EN job: ${en_id}"
echo "Submitted Tree job: ${tree_id}"
echo "Submitted PLS/PCR job: ${pls_id}"

dep="done(${en_id}) && done(${tree_id}) && done(${pls_id})"
agg_out=$(bsub -env "all,COMPDATA_ROOT=${COMPDATA_ROOT}" -w "$dep" < "${COMPDATA_ROOT}/Case1/hpc/run_aggregate.lsf")
agg_id=$(echo "$agg_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')

if [[ -z "$agg_id" ]]; then
	echo "ERROR: Failed to parse aggregate LSF job ID from bsub output." >&2
	echo "Aggregate output: $agg_out" >&2
	exit 1
fi

echo "Submitted aggregate job: ${agg_id}"
