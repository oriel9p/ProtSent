#!/usr/bin/env bash
# Remaining campaign, run explicitly rather than through the queue whose byte offsets my in-place
# edit shifted. Three phases, each waiting on the previous:
#   1. esm2-150m finishes at its 10,000 gate (already armed) and gets its cheap sweep
#   2. protsentv2-35m trains to 10,000 (it crashed at step 10 last time) and gets the same sweep
#   3. CATH + ProteinGym at 10,000 for all four arms -- the two benchmarks no r2 arm has yet
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

wait_gone(){ while ps -p "$1" >/dev/null 2>&1; do sleep 60; done; }

# --- phase 1: wait for the in-flight esm2-150m gate to finish its stop + sweep
GATE=$(ps -eo pid,cmd | grep "stop_at_and_bench.sh" | grep -v "grep\|snapshot-bash\|eval " | awk '{print $1}' | head -1)
[[ -n "${GATE:-}" ]] && { log "waiting on esm2-150m gate (pid $GATE)"; wait_gone "$GATE"; }
log "phase 1 done"

# --- phase 2: the 35M arm that crashed at step 10. --tee 3 so a repeat is diagnosable: last time
# it died with exitcode -6 and error_file: <N/A> on every rank, which says nothing.
RUN=late-r2-protsentv2-35m
if [[ ! -f "logs/stage_${RUN}.done" ]]; then
  log "training $RUN to 10,000"
  setsid nohup uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
    --main_process_port 29537 --tee 3 train_late_interaction.py \
    --model GrimSqueaker/ProtSent-V2-35M \
    --files /storage/users/ddofer/data/protsent-data-dc40/pfam_sorted.parquet \
            /storage/users/ddofer/data/protsent-data-dc40/afdb_sorted.parquet \
            /storage/users/ddofer/data/protsent-data-dc40/stringdb_train_15M.parquet \
    --output_dir "$M/$RUN" --run_name "$RUN" \
    --proj_dim 128 --batch_size 256 --mini_batch_num_tokens 32768 --score_mini_batch_size 32 \
    --lr 5e-5 --proj_lr 0 --lr_scheduler constant_with_warmup --warmup_steps 500 \
    --multi_dataset_sampler proportional --max_pairs_per_file 0 --string_max_pairs 15000000 \
    --seed 42 --compile --save_steps 1000 --save_total_limit 2 \
    --max_steps 30000 --max_minutes 0 --dataloader_num_workers 4 \
    >> "logs/queue_${RUN}.log" 2>&1 &
  for i in $(seq 90); do
    sleep 20
    PID=$(pgrep -f "bin/python -u train_late_interaction.py.*$RUN" | head -1)
    [[ -n "${PID:-}" ]] && break
  done
  if [[ -n "${PID:-}" ]]; then
    log "$RUN training pid $PID"
    setsid nohup ./snapshot_checkpoints.sh "$M/$RUN" > "logs/snap_${RUN}.log" 2>&1 &
    RUN="$RUN" TARGET=10000 MARKS="1000 4000 10000" \
      OUTDIR="$(pwd)/results/late_interaction/clean_35m/benchmarks" TRAIN_PID="$PID" \
      ./stop_at_and_bench.sh > "logs/gate_${RUN}.log" 2>&1 && touch "logs/stage_${RUN}.done"
  else
    log "$RUN trainer never appeared; skipping"
  fi
fi
log "phase 2 done"

# --- phase 3: CATH + ProteinGym at 10,000 for every arm. These read the MULTI-VECTOR snapshot
# (MaxSim), not the dense view, which is why they were never covered by the cheap sweep.
ARMS=""
for a in vanilla35m_clean late-r2-protsentv2-35m late-r2-protsentv2-150m late-r2-esm2-150m; do
  s="$M/$a/snapshots/step-10000"
  [[ -d "$s" ]] && ARMS="$ARMS ${a}_s10000=late:$s"
done
log "phase 3 arms:$ARMS"
[[ -n "$ARMS" ]] || { log "no step-10000 snapshots; nothing to score"; exit 0; }

log "CATH"
uv run --no-sync python late_interaction_eval.py cath $(for s in $ARMS; do echo -n "--models $s "; done) \
  --out_dir "$(pwd)/results/late_interaction/r2_final" --device cuda:0 >> logs/r2_cath.log 2>&1 \
  && log "CATH done" || log "CATH FAILED"

log "ProteinGym (substitutions + indels)"
for v in dms_substitutions dms_indels; do
  PGYM_ARGS="--max_seq_length 1024 --batch_size 256" \
    ./run_proteingym_parallel.sh "$v" "$(pwd)/results/late_interaction/r2_final" $ARMS \
    >> logs/r2_pgym.log 2>&1 && log "PGym $v done" || log "PGym $v FAILED"
done
log "CAMPAIGN COMPLETE"
