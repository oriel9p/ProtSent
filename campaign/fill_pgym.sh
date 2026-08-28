#!/usr/bin/env bash
# ProteinGym (MaxSim) for every model that lacks it: the one trained arm plus the four FROZEN
# references. Frozen arms use kind `zeroshot:`, which is raw residue MaxSim with no projection --
# the same scorer as the trained arms, so the rows are directly comparable. This is what lets
# ProteinGym speak to the frozen-vs-trained claim that SCOPe and few-shot already test.
#
# Batch size is probed rather than guessed. maxsim_against_one holds one _QUERY_CHUNK (8192) of
# query embeddings at a time, so peak memory is dominated by that residency rather than by the
# encoder batch, and the 256 used so far may be leaving throughput on the table. Three short runs
# (--max_assays 3) time 256/512/1024 on the largest model; the fastest that does not OOM is used
# for the full sweep. Costs ~5 minutes against a multi-hour sweep.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
log(){ echo "$(date +%H:%M) $*"; }

log "waiting for the paper-suite children to finish"
while pgrep -f "[r]un_late_bench.sh" > /dev/null 2>&1; do sleep 60; done
log "GPUs free"

PROBE_MODEL="probe=zeroshot:facebook/esm2_t30_150M_UR50D"
BEST=256; BEST_T=999999
for BS in 256 512 1024; do
  t0=$SECONDS
  if uv run --no-sync python late_interaction_eval.py proteingym \
       --models "$PROBE_MODEL" --variant dms_substitutions --max_assays 3 \
       --max_seq_length 1024 --batch_size "$BS" --device cuda:0 \
       --out_dir "$(pwd)/results/late_interaction/r2_final/_probe" \
       > "logs/pgym_probe_$BS.log" 2>&1; then
    dt=$(( SECONDS - t0 )); log "  batch $BS: ${dt}s"
    (( dt < BEST_T )) && { BEST_T=$dt; BEST=$BS; }
  else
    log "  batch $BS: FAILED (likely OOM) -- not considered"
  fi
done
log "using batch_size $BEST (${BEST_T}s on the 3-assay probe)"
rm -rf "$(pwd)/results/late_interaction/r2_final/_probe"

M=$(pwd)/models/late_interaction
ARMS=(
  "late-r2-protsentv2-35m_s10000=late:$M/late-r2-protsentv2-35m/snapshots/step-10000"
  "esm2_zeroshot=zeroshot:facebook/esm2_t12_35M_UR50D"
  "esm2_150m_zeroshot=zeroshot:facebook/esm2_t30_150M_UR50D"
  "protsent_v2_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-35M"
  "protsent_v2_150m_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-150M"
)
for v in dms_substitutions dms_indels; do
  PGYM_ARGS="--max_seq_length 1024 --batch_size $BEST" \
    ./run_proteingym_parallel.sh "$v" "$(pwd)/results/late_interaction/r2_final" "${ARMS[@]}" \
    >> logs/pgym_fill.log 2>&1 && log "PGym $v done" || log "PGym $v FAILED"
done
log "FILL_PGYM COMPLETE"
