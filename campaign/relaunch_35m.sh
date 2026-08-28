#!/usr/bin/env bash
# Relaunch late-r2-protsentv2-35m, the one arm still missing. Both prior attempts died at step 10
# with an NCCL ALLREDUCE watchdog timeout (SeqNum=314, NumelIn=984000) because they passed
# --mini_batch_num_tokens, which ST's gradcache packs per-rank: chunk counts differ across ranks,
# so the number of gradient allreduces differs and the collective mismatches. Use --mini_batch_size
# 64, exactly what the three arms that completed used.
#
# Waits for finish_campaign.sh so training never contends with the phase-3 ProteinGym run.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
RUN=late-r2-protsentv2-35m
log(){ echo "$(date +%H:%M) $*"; }

CAMP=$(ps -eo pid,cmd | grep "finish_campaign.sh" | grep -v "grep\|snapshot-bash\|eval " | awk '{print $1}' | head -1)
if [[ -n "${CAMP:-}" ]]; then
  log "waiting for finish_campaign.sh (pid $CAMP) so the GPUs are free"
  while ps -p "$CAMP" >/dev/null 2>&1; do sleep 60; done
fi
log "GPUs free; launching $RUN"

setsid nohup uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
  --main_process_port 29541 --tee 3 train_late_interaction.py \
  --model GrimSqueaker/ProtSent-V2-35M \
  --files /storage/users/ddofer/data/protsent-data-dc40/pfam_sorted.parquet \
          /storage/users/ddofer/data/protsent-data-dc40/afdb_sorted.parquet \
          /storage/users/ddofer/data/protsent-data-dc40/stringdb_train_15M.parquet \
  --output_dir "$M/$RUN" --run_name "$RUN" \
  --proj_dim 128 --batch_size 256 --mini_batch_size 64 --score_mini_batch_size 32 \
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
[[ -n "${PID:-}" ]] || { log "trainer never appeared; aborting"; exit 1; }
log "$RUN training pid $PID"

# Past step 10 means the deadlock did not recur. Say so in the log either way.
sleep 900
if ps -p "$PID" >/dev/null 2>&1; then
  log "alive at +15min: $(tr '\r' '\n' < logs/queue_${RUN}.log | grep -oE '[0-9]+/30000' | tail -1)"
else
  log "DIED within 15 min again -- check logs/queue_${RUN}.log"; exit 1
fi

setsid nohup ./snapshot_checkpoints.sh "$M/$RUN" > "logs/snap_${RUN}.log" 2>&1 &
setsid nohup uv run --no-sync python late_interaction_eval.py watch_curve \
  --run_dir "$M/$RUN" --name "$RUN" \
  --out_dir "$(pwd)/results/late_interaction/clean_35m/scope" \
  --batch_size 32 --device cuda:1 --follow_pid "$PID" --max_hours 40 \
  > "logs/watch_${RUN}.log" 2>&1 &

RUN="$RUN" TARGET=10000 MARKS="1000 4000 10000" \
  OUTDIR="$(pwd)/results/late_interaction/clean_35m/benchmarks" TRAIN_PID="$PID" \
  ./stop_at_and_bench.sh > "logs/gate_${RUN}.log" 2>&1 && touch "logs/stage_${RUN}.done"
log "$RUN COMPLETE"
