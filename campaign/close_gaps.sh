#!/usr/bin/env bash
# Two measurements that close the two structural holes in the late-interaction section.
#   1. ESM2-150M zero-shot MaxSim: the missing cell of the pretraining 2x2, and a real test of the
#      residue-geometry mechanism. That mechanism predicts it will NOT beat ESM2-35M's 0.4735,
#      because ESM2-150M has LOWER effective residue rank (10.54) than ESM2-35M (12.10).
#   2. Two-stage rerank at every SCOPe level (was family-only) for the arms the paper would ship:
#      pooled ProtSent-V2-150M shortlist, reranked by MaxSim. This is the answer to "your index is
#      40-60x bigger", and it was the least-measured claim in the section.
# Serial on one card: the other three are on ProteinGym.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

log "1/2 ESM2-150M zero-shot MaxSim (prediction test: should NOT beat 0.4735)"
uv run --no-sync python late_interaction_eval.py scope \
  --models "esm2_150m_zeroshot=zeroshot:facebook/esm2_t30_150M_UR50D" \
  --out_dir "$(pwd)/results/late_interaction/pilot_35m/scope" \
  --batch_size 32 --device cuda:3 && log "1/2 done" || log "1/2 FAILED"

log "2/2 two-stage rerank, all SCOPe levels"
uv run --no-sync python analyze_maxsim_cost.py \
  --models "protsent_v2_150m_dense=dense:GrimSqueaker/ProtSent-V2-150M" \
  --models "protsent_v2_150m_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-150M" \
  --models "late-r2-esm2-150m_s10000=late:$M/late-r2-esm2-150m/snapshots/step-10000" \
  --rerank "protsent_v2_150m_zeroshot=protsent_v2_150m_dense" \
  --rerank "late-r2-esm2-150m_s10000=protsent_v2_150m_dense" \
  --out_dir "$(pwd)/results/late_interaction/r2_final" \
  --batch_size 32 --device cuda:3 && log "2/2 done" || log "2/2 FAILED"
log "CLOSE_GAPS COMPLETE"
