#!/usr/bin/env bash
set -euo pipefail

mkdir -p Case1/results/logs

en_out=$(bsub < Case1/hpc/run_elastic_net.lsf)
tree_out=$(bsub < Case1/hpc/run_tree_ensembles.lsf)
pls_out=$(bsub < Case1/hpc/run_pls_pcr.lsf)

en_id=$(echo "$en_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')
tree_id=$(echo "$tree_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')
pls_id=$(echo "$pls_out" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p')

echo "Submitted EN job: ${en_id}"
echo "Submitted Tree job: ${tree_id}"
echo "Submitted PLS/PCR job: ${pls_id}"

dep="done(${en_id}) && done(${tree_id}) && done(${pls_id})"
bsub -w "$dep" < Case1/hpc/run_aggregate.lsf
