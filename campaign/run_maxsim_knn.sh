#!/usr/bin/env bash
# MaxSim as the kNN metric on ProtBench's own task setups -- full train gallery, k=3, same splits
# and metrics as the pooled-cosine knn probe already in results/. That makes MaxSim-vs-cosine a
# controlled swap rather than a separate experiment.
#
# Partial coverage on purpose. Four tasks spanning the problem types ProtBench's knn probe covers:
# one multiclass fold task (the paper's headline), one binary, one multiclass localisation, one
# regression. EC is excluded -- it is multilabel and ProtBench auto-switches it to a linear probe,
# so a k-vote there would not be comparable to anything.
#
# One task per GPU; each runs all 6 models serially so the per-task gallery is encoded once per model.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

log "waiting for the GPUs"
while pgrep -f "[e]val.py proteingym|[r]un_late_bench.sh|[e]val.py fewshot_rh" > /dev/null 2>&1; do sleep 60; done
log "GPUs free"

MODELS=(
  --models "esm2_35m_frozen=zeroshot:facebook/esm2_t12_35M_UR50D"
  --models "esm2_150m_frozen=zeroshot:facebook/esm2_t30_150M_UR50D"
  --models "protsent_v2_35m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-35M"
  --models "protsent_v2_150m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-150M"
  --models "late-r2-esm2-150m_s10000=late:$M/late-r2-esm2-150m/snapshots/step-10000"
  --models "late-r2-protsentv2-150m_s10000=late:$M/late-r2-protsentv2-150m/snapshots/step-10000"
)
TASKS=(remote_homology metal_ion_binding subcellular_loc fluorescence)

pids=()
for i in "${!TASKS[@]}"; do
  t=${TASKS[$i]}
  CUDA_VISIBLE_DEVICES=$i uv run --no-sync python maxsim_knn_bench.py \
    --task "$t" "${MODELS[@]}" --knn_k 3 --batch_size 32 --device cuda:0 \
    > "logs/maxsim_knn_$t.log" 2>&1 &
  pids+=($!)
  log "  $t -> gpu $i"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
log "RUN_MAXSIM_KNN COMPLETE (fail=$fail)"
