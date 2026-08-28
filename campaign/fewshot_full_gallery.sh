#!/usr/bin/env bash
# Answer "is MaxSim better than a probe?" fairly, by giving MaxSim the same labels the probes get.
#
# The existing few-shot rows cap the gallery at 1,000 while ProtBench's knn/linear probes fit on all
# 12,312 fold_prediction training sequences. Comparing 0.24 F1_M (MaxSim @1000) against 0.44
# (linear @12312) measures the 12x label gap, not the scoring function. Same test split (3,244),
# same k=3, same metric -- only the gallery size differed.
#
# Adding budget 12312 makes MaxSim 3-NN and the knn probe an apples-to-apples pair: identical data,
# identical vote, the only difference being MaxSim vs pooled-cosine similarity.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

log "waiting for the GPUs"
while pgrep -f "[e]val.py proteingym|[r]un_late_bench.sh" > /dev/null 2>&1; do sleep 60; done
log "GPUs free"

uv run --no-sync python late_interaction_eval.py fewshot_rh \
  --models "esm2_zeroshot=zeroshot:facebook/esm2_t12_35M_UR50D" \
  --models "esm2_150m_zeroshot=zeroshot:facebook/esm2_t30_150M_UR50D" \
  --models "protsent_v2_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-35M" \
  --models "protsent_v2_150m_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-150M" \
  --models "protsent_v2_150m_dense=dense:GrimSqueaker/ProtSent-V2-150M" \
  --models "vanilla35m_clean_s10000=late:$M/vanilla35m_clean/snapshots/step-10000" \
  --models "late-r2-protsentv2-35m_s10000=late:$M/late-r2-protsentv2-35m/snapshots/step-10000" \
  --models "late-r2-esm2-150m_s10000=late:$M/late-r2-esm2-150m/snapshots/step-10000" \
  --models "late-r2-protsentv2-150m_s10000=late:$M/late-r2-protsentv2-150m/snapshots/step-10000" \
  --budgets 12312 --knn_k 3 \
  --out_dir "$(pwd)/results/late_interaction/r2_final/benchmarks" \
  --batch_size 32 --device cuda:0 --n_boot 2000 \
  >> logs/fewshot_full.log 2>&1 && log "FULL GALLERY done" || log "FULL GALLERY FAILED"
