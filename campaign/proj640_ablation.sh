#!/usr/bin/env bash
# proj_dim ablation, one arm. We already have 128-d at 150M from vanilla ESM2
# (late-r2-esm2-150m); this is the matched 640-d control, identical in every other flag, so the
# only difference is the output dimension. That isolates "128-d indexes 5x smaller at parity"
# from the training confound the current comparison carries (our 128-d arms are trained, the
# 640-d reference is not).
#
# 640 = ESM2-150M's hidden size, so this is a same-width projection, NOT --proj_dim 0. Keeping the
# projection layer present in both arms is what makes dimension the only variable.
#
# 2000 steps: the 128-d curve is flat from step 1000 (0.7383@1k vs 0.7403@8k, CI half-width
# 0.0126), so marks at 1000 and 2000 are enough to read the comparison. ~2 h.
#
# Waits for the 35M relaunch so nothing contends.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
RUN=late-r2-esm2-150m-proj640
log(){ echo "$(date +%H:%M) $*"; }

for name in relaunch_35m.sh finish_campaign.sh; do
  P=$(ps -eo pid,cmd | grep "$name" | grep -v "grep\|snapshot-bash\|eval " | awk '{print $1}' | head -1)
  if [[ -n "${P:-}" ]]; then
    log "waiting for $name (pid $P)"
    while ps -p "$P" >/dev/null 2>&1; do sleep 60; done
  fi
done
log "GPUs free; launching $RUN (proj_dim 640)"

setsid nohup uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
  --main_process_port 29545 --tee 3 train_late_interaction.py \
  --model facebook/esm2_t30_150M_UR50D \
  --files /storage/users/ddofer/data/protsent-data-dc40/pfam_sorted.parquet \
          /storage/users/ddofer/data/protsent-data-dc40/afdb_sorted.parquet \
          /storage/users/ddofer/data/protsent-data-dc40/stringdb_train_15M.parquet \
  --output_dir "$M/$RUN" --run_name "$RUN" \
  --proj_dim 640 --batch_size 256 --mini_batch_size 64 --score_mini_batch_size 32 \
  --lr 5e-5 --proj_lr 0 --lr_scheduler constant_with_warmup --warmup_steps 500 \
  --multi_dataset_sampler proportional --max_pairs_per_file 0 --string_max_pairs 15000000 \
  --seed 42 --compile --save_steps 1000 --save_total_limit 3 \
  --max_steps 30000 --max_minutes 0 --dataloader_num_workers 4 \
  >> "logs/queue_${RUN}.log" 2>&1 &

for i in $(seq 90); do
  sleep 20
  PID=$(pgrep -f "bin/python -u train_late_interaction.py.*$RUN" | head -1)
  [[ -n "${PID:-}" ]] && break
done
[[ -n "${PID:-}" ]] || { log "trainer never appeared; aborting"; exit 1; }
log "$RUN training pid $PID"

setsid nohup ./snapshot_checkpoints.sh "$M/$RUN" > "logs/snap_${RUN}.log" 2>&1 &
setsid nohup uv run --no-sync python late_interaction_eval.py watch_curve \
  --run_dir "$M/$RUN" --name "$RUN" \
  --out_dir "$(pwd)/results/late_interaction/clean_150m/scope" \
  --batch_size 32 --device cuda:1 --follow_pid "$PID" --max_hours 8 \
  > "logs/watch_${RUN}.log" 2>&1 &

RUN="$RUN" TARGET=2000 MARKS="1000 2000" \
  OUTDIR="$(pwd)/results/late_interaction/clean_150m/benchmarks" TRAIN_PID="$PID" \
  ./stop_at_and_bench.sh > "logs/gate_${RUN}.log" 2>&1 && touch "logs/stage_${RUN}.done"
log "$RUN COMPLETE"
