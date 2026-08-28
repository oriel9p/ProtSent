#!/usr/bin/env bash
# Regression coverage for the MaxSim-kNN comparison, on tasks whose cost is proportionate.
#
# fluorescence was the original pick and was killed after one model: 21,446 gallery x 27,217 eval
# is 584M pairs and 48,663 encodings, which ran 42 min for the 35M model alone and projected to 6-8
# hours across six models -- 15x the other three tasks combined, for one more point of the same kind.
# beta_lactamase_peer (2M pairs) and optimal_ph (14M) give TWO regression tasks for a fraction of
# that. stability is excluded for the same reason as fluorescence: 689M pairs.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

gpus_busy(){ [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; }
while gpus_busy; do sleep 30; done
log "GPUs free; two regression tasks, one per GPU"

MODELS=(
  --models "esm2_35m_frozen=zeroshot:facebook/esm2_t12_35M_UR50D"
  --models "esm2_150m_frozen=zeroshot:facebook/esm2_t30_150M_UR50D"
  --models "protsent_v2_35m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-35M"
  --models "protsent_v2_150m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-150M"
  --models "late-r2-esm2-150m_s10000=late:$M/late-r2-esm2-150m/snapshots/step-10000"
  --models "late-r2-protsentv2-150m_s10000=late:$M/late-r2-protsentv2-150m/snapshots/step-10000"
)
TASKS=(beta_lactamase_peer optimal_ph)
pids=()
for i in "${!TASKS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run --no-sync python maxsim_knn_bench.py \
    --task "${TASKS[$i]}" "${MODELS[@]}" --knn_k 3 --batch_size 32 --device cuda:0 \
    > "logs/maxsim_knn_${TASKS[$i]}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || log "  a task failed"; done
log "MAXSIM_KNN_REGRESSION COMPLETE"
