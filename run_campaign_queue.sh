#!/usr/bin/env bash
# Overnight campaign queue: train -> stop at target -> cheap knn sweep, one stage at a time.
# Survives disconnect (setsid the whole thing). Resumable: a stage whose runtime.json exists is
# skipped, so re-running the queue after a crash picks up where it stopped.
#
#   setsid nohup ./run_campaign_queue.sh > logs/campaign_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}" TOKENIZERS_PARALLELISM=false
DATA=/storage/users/ddofer/data/protsent-data-dc40
FILES="$DATA/pfam_sorted.parquet $DATA/afdb_sorted.parquet $DATA/stringdb_train_15M.parquet"
M="models/late_interaction"
R="$(pwd)/results/late_interaction"

# Wait for a pid to disappear (the in-flight vanilla-35M sweep, on the first pass).
wait_for_pid() {
  [[ -n "${1:-}" ]] || return 0
  while kill -0 "$1" 2>/dev/null; do sleep 120; done
}

# run_stage <run_name> <base_model> <target_steps> <marks> <results_dir> <port>
# max_steps == target, so training ends on its own and exports late/ + dense_view/ + runtime.json;
# nothing has to be killed. Flash + compile are on (attn defaults to auto, which verifies the
# backend at load time and logs it); gather_across_devices is deliberately NOT passed, so
# in-batch negatives stay per-rank at 255.
run_stage() {
  local RUN="$1" MODEL="$2" TARGET="$3" MARKS="$4" RESDIR="$5" PORT="$6" \
        CHUNK="${7:---mini_batch_size 64}" MAXSTEPS="${8:-$3}"
  local done="logs/stage_${RUN}.done"
  if [[ -f "$done" ]]; then echo "[skip] $RUN already finished"; return 0; fi
  echo "=== $(date +%H:%M) $RUN: $MODEL -> stop at $TARGET of $MAXSTEPS budgeted (chunking: $CHUNK)"

  setsid nohup uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
    --main_process_port "$PORT" train_late_interaction.py \
    --model "$MODEL" --files $FILES \
    --output_dir "$M/$RUN" --run_name "$RUN" \
    --proj_dim 128 --batch_size 256 $CHUNK --score_mini_batch_size 32 \
    --lr 5e-5 --proj_lr 0 --lr_scheduler constant_with_warmup --warmup_steps 500 \
    --multi_dataset_sampler proportional --max_pairs_per_file 0 --string_max_pairs 15000000 \
    --seed 42 --compile --save_steps 1000 --save_total_limit 2 \
    --max_steps "$MAXSTEPS" --max_minutes 0 --dataloader_num_workers 4 \
    >> "logs/queue_${RUN}.log" 2>&1 &

  local pid="" i
  for i in $(seq 90); do
    sleep 20
    pid=$(pgrep -f "bin/python -u train_late_interaction.py.*$RUN" | head -1)
    [[ -n "$pid" ]] && break
  done
  [[ -n "$pid" ]] || { echo "$RUN: trainer never appeared; skipping stage"; return 1; }
  echo "$(date +%H:%M) $RUN training pid $pid"

  setsid nohup ./snapshot_checkpoints.sh "$M/$RUN" > "logs/snap_${RUN}.log" 2>&1 &
  setsid nohup uv run --no-sync python late_interaction_eval.py watch_curve \
      --run_dir "$M/$RUN" --name "$RUN" --out_dir "$RESDIR/scope" --batch_size 32 \
      --device cuda:1 --follow_pid "$pid" --max_hours 40 > "logs/watch_${RUN}.log" 2>&1 &

  wait_for_pid "$pid"
  sleep 30
  grep -m1 -oE "attention backend:.*" "logs/queue_${RUN}.log" || echo "(no attention-backend line logged)"

  # The endpoint mark comes from the run's own final export; the snapshotter may not have caught
  # the last checkpoint before the trainer cleaned it up. A symlink, so the sweep's own cleanup
  # (rm -rf *-dense) removes the link and leaves dense_view/ intact.
  if [[ -d "$M/$RUN/dense_view" && ! -e "$M/$RUN/snapshots/step-$TARGET-dense" ]]; then
    mkdir -p "$M/$RUN/snapshots"
    ln -s ../dense_view "$M/$RUN/snapshots/step-$TARGET-dense"
  fi

  RUN="$RUN" TARGET="$TARGET" MARKS="$MARKS" OUTDIR="$RESDIR/benchmarks" TRAIN_PID="$pid" \
    ./stop_at_and_bench.sh || echo "$(date +%H:%M) $RUN: sweep reported failures (see its logs)"
  touch "$done"
  echo "=== $(date +%H:%M) $RUN complete"
}

# Stage 0: let the in-flight vanilla-35M stop-and-sweep finish first.
echo "$(date +%H:%M) waiting for the in-flight vanilla35m_clean sweep (pid ${WAIT_PID:-none})"
wait_for_pid "${WAIT_PID:-}"

# Marks share 1000/4000/10000 across every arm so all four runs are comparable at matched steps;
# the 150M pair adds 15000 as its endpoint.
# Stage 1 is a clean A/B against stage 0's fixed --mini_batch_size 64: same architecture, same
# data, same 4 GPUs, back to back with nothing else on the cards. It budgets by tokens instead,
# at 32768 = the 64x512 worst case stage 0 already survives, so every chunk becomes the size of
# stage 0's unluckiest one rather than its average -- more work per chunk at the same peak VRAM.
# GradCache is gradient-equivalent across chunkings, so this moves speed and memory only and the
# arms stay comparable. runtime.json reports steps_per_s and peak_vram_bytes for both.
run_stage late-r2-protsentv2-35m  GrimSqueaker/ProtSent-V2-35M  10000 "1000 4000 10000" "$R/clean_35m" 29527 "--mini_batch_num_tokens 32768"

# The 150M pair reads its mini_batch at stage start rather than having it baked in, so the value
# can be set from stage 1's measurement without editing this script while bash is executing it
# (bash reads scripts lazily by byte offset -- editing a live one corrupts it).
CHUNK150="$(cat .chunk150 2>/dev/null || echo '--mini_batch_size 64')"
echo "$(date +%H:%M) 150M stages chunking: $CHUNK150 (from .chunk150; default is the known-safe fixed 64)"
# 10k, not 15k: the marks then match the 35M arms exactly (1000/4000/10000), so every arm in the
# campaign is comparable at the same three steps. Resumable to any longer budget later --
# --resume finds the latest checkpoint and the constant LR means no schedule distortion.
run_stage late-r2-protsentv2-150m GrimSqueaker/ProtSent-V2-150M 10000 "1000 4000 10000" "$R/clean_150m" 29528 "$CHUNK150"

# The vanilla 150M arm stops at the SECOND mark and is benchmarked there as a decision gate.
# The parsimonious prior is that it does worse than the ProtSent-V2 base (that is what every 35M
# comparison has shown); 4,000 steps is enough to confirm or refute that against the V2 arm's own
# 4,000 mark, and costs ~3 h instead of ~8. If it wins, resume it -- nothing is thrown away.
run_stage late-r2-esm2-150m       facebook/esm2_t30_150M_UR50D   4000 "1000 4000"       "$R/clean_150m" 29529 "$CHUNK150"


# resume_stage <run> <from_step> <to_step> <marks> <resdir> <port> <chunk>
# A true continuation: the arm was killed at its gate rather than finishing, so checkpoint-N still
# has optimizer.pt and AdamW's moments carry over. That matters -- a fresh optimizer on a converged
# model is the restart shock this campaign already suspects of costing phase 2 ~0.015 sfam.
resume_stage() {
  local RUN="$1" FROM="$2" TO="$3" MARKS="$4" RESDIR="$5" PORT="$6" CHUNK="$7"
  local done="logs/stage_${RUN}_resume.done"
  [[ -f "$done" ]] && { echo "[skip] $RUN resume already done"; return 0; }
  [[ -d "$M/$RUN/checkpoint-$FROM" ]] || { echo "$RUN: no checkpoint-$FROM to resume from"; return 1; }
  echo "=== $(date +%H:%M) $RUN: resuming $FROM -> $TO"

  setsid nohup uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
    --main_process_port "$PORT" train_late_interaction.py \
    --model "$M/$RUN" --files $FILES \
    --output_dir "$M/$RUN" --run_name "$RUN" --resume \
    --proj_dim 128 --batch_size 256 $CHUNK --score_mini_batch_size 32 \
    --lr 5e-5 --proj_lr 0 --lr_scheduler constant_with_warmup --warmup_steps 500 \
    --multi_dataset_sampler proportional --max_pairs_per_file 0 --string_max_pairs 15000000 \
    --seed 42 --compile --save_steps 1000 --save_total_limit 2 \
    --max_steps "$TO" --max_minutes 0 --dataloader_num_workers 4 \
    >> "logs/queue_${RUN}.log" 2>&1 &

  local pid="" i
  for i in $(seq 90); do
    sleep 20
    pid=$(pgrep -f "bin/python -u train_late_interaction.py.*$RUN" | head -1)
    [[ -n "$pid" ]] && break
  done
  [[ -n "$pid" ]] || { echo "$RUN: resume never started"; return 1; }

  setsid nohup ./snapshot_checkpoints.sh "$M/$RUN" > "logs/snap_${RUN}_resume.log" 2>&1 &
  setsid nohup uv run --no-sync python late_interaction_eval.py watch_curve \
      --run_dir "$M/$RUN" --name "$RUN" --out_dir "$RESDIR/scope" --batch_size 32 \
      --device cuda:1 --follow_pid "$pid" --max_hours 40 > "logs/watch_${RUN}_resume.log" 2>&1 &

  wait_for_pid "$pid"
  sleep 30
  if [[ -d "$M/$RUN/dense_view" && ! -e "$M/$RUN/snapshots/step-$TO-dense" ]]; then
    ln -s ../dense_view "$M/$RUN/snapshots/step-$TO-dense"
  fi
  RUN="$RUN" TARGET="$TO" MARKS="$MARKS" OUTDIR="$RESDIR/benchmarks" TRAIN_PID="$pid" \
    ./stop_at_and_bench.sh || echo "$(date +%H:%M) $RUN resume: sweep reported failures"
  touch "$done"
  echo "=== $(date +%H:%M) $RUN resume complete"
}

# Decide on the shared 4,000-step mark -- the only step both 150M arms reached under an identical
# recipe -- then carry the winner to 15,000.
WINNER="$(uv run --no-sync python gate_pick_winner.py "$R/clean_150m/scope/scope_checkpoint_curve.csv")"
echo "$(date +%H:%M) GATE: 4,000-step winner is $WINNER"

if [[ "$WINNER" == "late-r2-esm2-150m" ]]; then
  resume_stage late-r2-esm2-150m 4000 15000 "10000 15000" "$R/clean_150m" 29530 "$CHUNK150"
else
  resume_stage late-r2-protsentv2-150m 10000 15000 "15000" "$R/clean_150m" 29531 "$CHUNK150"
fi

cat <<'GATE'

=== GATE RESOLVED ===============================================================
late-r2-esm2-150m stopped at its 4,000-step gate. Compare it against
late-r2-protsentv2-150m @4000 -- same recipe, same data, same step count -- on
the cheap knn sweep (results/late_interaction/clean_150m/benchmarks/) and the
SCOPe curve (clean_150m/scope/scope_checkpoint_curve.csv).

Resolved automatically on SCOPe superfamily eligible_MAP at the shared 4,000-step
mark; the winner has been resumed to 15,000 with its optimizer state intact.
=================================================================================
GATE

echo "$(date +%H:%M) campaign queue finished"
