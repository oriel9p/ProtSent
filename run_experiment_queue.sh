#!/usr/bin/env bash
# Resumable experiment queue for the late-interaction campaign.
#
#   ./run_experiment_queue.sh start     # launch the three per-GPU queues (detached; survives SSH loss)
#   ./run_experiment_queue.sh status    # one-line-per-job state
#   ./run_experiment_queue.sh stop      # stop everything (safe: rerun `start` to resume)
#
# Every job is idempotent: a finished job is skipped (done-marker = runtime.json / CSV rows), and an
# interrupted training job resumes from its own checkpoint. Re-running `start` after any crash, reboot
# or Ctrl-C picks up exactly where the queue stopped.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
RES="$ROOT/results/late_interaction"
MODELS="$ROOT/models/late_interaction"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}" TOKENIZERS_PARALLELISM=false
FILES=("$DATA/pfam_sorted.parquet" "$DATA/afdb_sorted.parquet" "$DATA/stringdb_train_15M.parquet")
mkdir -p logs "$MODELS" "$RES/pilot_150m/scope" "$RES/pilot_35m/scope"

# ---------------------------------------------------------------- job helpers
train() {  # train <gpu> <name> <model> <steps> <batch> <save_steps> <extra args...>
  local gpu="$1" name="$2" model="$3" steps="$4" batch="$5" save="$6"; shift 6
  local dir="$MODELS/$name"
  if [[ -f "$dir/runtime.json" ]]; then echo "[skip] train $name"; return 0; fi
  echo "[run ] train $name (gpu $gpu, $steps steps)"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python train_late_interaction.py \
    --model "$model" --files "${FILES[@]}" --output_dir "$dir" \
    --max_steps "$steps" --batch_size "$batch" --mini_batch_size 64 --score_mini_batch_size 32 \
    --save_steps "$save" --save_total_limit 1 --max_minutes 0 --resume \
    --dataloader_num_workers 4 --run_name "$name" "$@"
}

watch_curve() {  # watch_curve <gpu> <name> <out_dir> — background, follows a training job's checkpoints
  local gpu="$1" name="$2" out="$3"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python late_interaction_eval.py watch_curve \
    --run_dir "$MODELS/$name" --name "$name" --out_dir "$out" --batch_size 32 --max_hours 24 &
}

scope_rows() {  # scope_rows <gpu> <out_dir> <reference|-> <specs...>
  local gpu="$1" out="$2" ref="$3"; shift 3
  local args=()
  for spec in "$@"; do args+=(--models "$spec"); done
  [[ "$ref" != "-" ]] && args+=(--reference "$ref")
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python late_interaction_eval.py scope \
    "${args[@]}" --n_boot 1000 --out_dir "$out"
}

# ---------------------------------------------------------------- queues
queue_a() {  # GPU A: ProtSent-V2-150M late
  local gpu="$1"
  watch_curve "$gpu" protsent_late_150m "$RES/pilot_150m/scope"
  train "$gpu" protsent_late_150m GrimSqueaker/ProtSent-V2-150M 5000 128 500
  wait
}

queue_b() {  # GPU B: vanilla ESM2-150M late
  local gpu="$1"
  watch_curve "$gpu" esm2_late_150m "$RES/pilot_150m/scope"
  train "$gpu" esm2_late_150m Synthyra/ESM2-150M 5000 128 500
  wait
}

queue_c() {  # GPU C: cheap 35M ablations first, then the long runs
  local gpu="$1"
  # These run on ONE GPU, where a step consumes batch_size pairs; the pilot arms ran
  # on two, where a step consumes 2 x batch_size. 4,000 single-GPU steps therefore match
  # the pilot's 512,000-pair budget, and the in-batch negative count (128) is identical
  # either way because gather_across_devices is off. Comparing at equal steps instead of
  # equal pairs would confound the ablation with half the training data.
  # C1 head size: 128-D projection vs the pilot's 64-D
  train "$gpu" protsent_late_proj128 GrimSqueaker/ProtSent-V2-35M 4000 128 500 --proj_dim 128
  # C1b symmetry: identical recipe, pair order randomly swapped
  train "$gpu" protsent_late_swap GrimSqueaker/ProtSent-V2-35M 4000 128 500 --swap_pair_order
  scope_rows "$gpu" "$RES/pilot_35m/scope" - \
    "protsent_late_proj128=late:$MODELS/protsent_late_proj128/late" \
    "protsent_late_swap=late:$MODELS/protsent_late_swap/late"
}

# The long runs live in their own queues so they can take the two GPUs the 150M arms
# free up, rather than running back to back on one card.
#
# 18,219 steps is one complete ROUND_ROBIN pass: the sampler cycles until the smallest
# source is exhausted, Pfam is smallest at 777,306 pairs = 6,073 batches, so a pass over
# three sources is 3x that = 2.33M pairs. Stopping at a round 10k would end mid-pass with
# the sources unevenly represented.
#
# 128-D, not the 64-D default: the head-size ablation measured 128-D at +.034 MAP over
# 64-D and +.007 over scoring the backbone's own 480-D residues, both with paired CIs
# clear of zero, so the 64-D head these runs originally inherited is the wrong one to
# spend 8 GPU-h on.
LONG_STEPS="${LONG_STEPS:-18219}"
LONG_ARGS=(--proj_dim 128 --max_pairs_per_file 0 --string_max_pairs 6000000)

queue_d() {  # vanilla ESM-2, long
  local gpu="$1"
  watch_curve "$gpu" esm2_late_long "$RES/pilot_35m/scope"
  train "$gpu" esm2_late_long Synthyra/ESM2-35M "$LONG_STEPS" 128 2000 "${LONG_ARGS[@]}"
  wait
}

queue_e() {  # ProtSent-V2, long
  local gpu="$1"
  watch_curve "$gpu" protsent_late_long "$RES/pilot_35m/scope"
  train "$gpu" protsent_late_long GrimSqueaker/ProtSent-V2-35M "$LONG_STEPS" 128 2000 "${LONG_ARGS[@]}"
  wait
}

# ---------------------------------------------------------------- driver
case "${1:-start}" in
  start)
    for q in ${QUEUES:-a b c}; do
      gpu_var="GPU_${q^^}"; gpu="${!gpu_var:-}"
      [[ -z "$gpu" ]] && gpu=$(( $(printf '%s' "$q" | tr 'abcde' '01201') ))
      if pgrep -f "run_experiment_queue.sh __run $q" > /dev/null; then echo "queue $q already running"; continue; fi
      setsid nohup "$ROOT/run_experiment_queue.sh" __run "$q" "$gpu" >> "logs/queue_$q.log" 2>&1 < /dev/null &
      echo "queue $q -> gpu $gpu (log logs/queue_$q.log)"
    done
    ;;
  __run) queue_"$2" "$3" ;;
  status)
    echo "--- running"
    pgrep -af "run_experiment_queue.sh __run|train_late_interaction.py|late_interaction_eval.py" | cut -c1-140 || echo "(nothing)"
    echo "--- training jobs"
    for d in "$MODELS"/*/; do
      n=$(basename "$d")
      [[ "$n" == sweep_* ]] && continue
      if [[ -f "$d/runtime.json" ]]; then
        python3 -c "import json;d=json.load(open('$d/runtime.json'));print(f'  done    $n  {d[\"steps\"]} steps, {d[\"wall_time_s\"]/3600:.1f}h, {d[\"pairs_seen\"]:,} pairs')"
      else
        last=$(ls -d "$d"checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
        echo "  running $n  (checkpoint ${last:-none})"
      fi
    done
    echo "--- result files"
    find "$RES" -name "*.csv" -newermt "-7 days" -printf "  %p (%s bytes)\n" 2>/dev/null | sort
    ;;
  stop)
    pkill -f "run_experiment_queue.sh __run"; pkill -f train_late_interaction.py; pkill -f late_interaction_eval.py
    echo "stopped (rerun 'start' to resume)"
    ;;
  *) echo "usage: $0 {start|status|stop}"; exit 1 ;;
esac
