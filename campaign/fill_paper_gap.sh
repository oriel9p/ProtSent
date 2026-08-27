#!/usr/bin/env bash
# The two ProtSent-paper tasks the 21-task run missed, so the suite matches the paper's 23.
#
# PAPER_TASKS in run_late_bench.sh omits both: ppi_bernett lives only in TASKS=full, and
# rhla_enzyme_mutations was never in the list at all even though the paper reports it (Table 2,
# "RhlA Enzyme Mutations", the largest 35M regression gain at +77.2%). go_mf stays dropped -- it is
# excluded from ProtBench's FAST_TASKS by name as "too slow for the default sweep", and the paper
# counts CAFA5 / Mol. Function as multilabel linear-probe extras, not among the 23 KNN tasks.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
OUT=$(pwd)/results/late_interaction/r2_final/paper_suite
log(){ echo "$(date +%H:%M) $*"; }

log "waiting for ProteinGym to release the GPUs"
while pgrep -f "[e]val.py proteingym" > /dev/null 2>&1; do sleep 60; done
log "GPUs free"

declare -a G0 G1 G2 G3
G0=("vanilla35m_clean_s10000=$M/vanilla35m_clean/snapshots/step-10000-dense"
    "esm2_35m_frozen=facebook/esm2_t12_35M_UR50D")
G1=("late-r2-protsentv2-35m_s10000=$M/late-r2-protsentv2-35m/snapshots/step-10000-dense"
    "protsent_v2_35m_frozen=GrimSqueaker/ProtSent-V2-35M")
G2=("late-r2-esm2-150m_s10000=$M/late-r2-esm2-150m/snapshots/step-10000-dense"
    "esm2_150m_frozen=facebook/esm2_t30_150M_UR50D")
G3=("late-r2-protsentv2-150m_s10000=$M/late-r2-protsentv2-150m/snapshots/step-10000-dense"
    "protsent_v2_150m_frozen=GrimSqueaker/ProtSent-V2-150M")

pids=()
for g in 0 1 2 3; do
  eval "arms=(\"\${G$g[@]}\")"
  CUDA_VISIBLE_DEVICES=$g TASKS="ppi_bernett rhla_enzyme_mutations" OUT="$OUT" \
    ./run_late_bench.sh "${arms[@]}" > "logs/papergap_gpu$g.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
log "FILL_PAPER_GAP COMPLETE (fail=$fail)"
