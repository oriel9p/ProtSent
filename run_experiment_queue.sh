#!/usr/bin/env bash
# Resumable experiment queue for the late-interaction campaign.
#
#   ./run_experiment_queue.sh start     # launch the three per-GPU queues (detached; survives SSH loss)
#   ./run_experiment_queue.sh status    # one-line-per-job state
#   ./run_experiment_queue.sh bench     # pooled benchmarks over every finished arm's dense view
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
train() {  # train <gpu> <name> <model> <steps> <save_steps> <extra args...>
  local gpu="$1" name="$2" model="$3" steps="$4" save="$5"; shift 5
  local dir="$MODELS/$name"
  if [[ -f "$dir/runtime.json" ]]; then echo "[skip] train $name"; return 0; fi
  echo "[run ] train $name (gpu $gpu, $steps steps)"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python train_late_interaction.py \
    --model "$model" --files "${FILES[@]}" --output_dir "$dir" \
    --max_steps "$steps" --batch_size 128 --mini_batch_size 64 --score_mini_batch_size 32 \
    --save_steps "$save" --save_total_limit 1 --max_minutes 0 --resume \
    --dataloader_num_workers 4 --run_name "$name" "$@"
}

train_ddp() {  # train_ddp <cards> <port> <name> <model> <steps> <save_steps> <extra args...>
  local cards="$1" port="$2" name="$3" model="$4" steps="$5" save="$6"; shift 6
  local dir="$MODELS/$name"
  if [[ -f "$dir/runtime.json" ]]; then echo "[skip] train $name"; return 0; fi
  echo "[run ] train $name (cards $cards, $steps steps, 2-way DDP)"
  # --compile comes from the callers. Fair A/B (both arms concurrent, one per card pair):
  # 1.14x faster, 0.586 vs 0.516 steps/s. Unequal-load measurements said the opposite twice.
  CUDA_VISIBLE_DEVICES="$cards" uv run --no-sync accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port "$port" \
    train_late_interaction.py \
    --model "$model" --files "${FILES[@]}" --output_dir "$dir" \
    --max_steps "$steps" --batch_size 128 --mini_batch_size 64 --score_mini_batch_size 32 \
    --save_steps "$save" --save_total_limit 1 --max_minutes 0 --resume \
    --dataloader_num_workers 4 --run_name "$name" "$@"
}

watch_curve() {  # watch_curve <gpu> <name> <out_dir> — background, dies with its trainer
  local gpu="$1" name="$2" out="$3"
  # --follow_pid was added so a crashed trainer cannot leave this polling for 24 h with the
  # queue blocked on `wait`; it has to actually be passed. $$ is the queue driver, which
  # exits when its jobs do.
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python late_interaction_eval.py watch_curve \
    --run_dir "$MODELS/$name" --name "$name" --out_dir "$out" --batch_size 32 \
    --follow_pid $$ --max_hours 24 &
}

scope_rows() {  # scope_rows <gpu> <out_dir> <specs...>
  local gpu="$1" out="$2"; shift 2
  local args=()
  for spec in "$@"; do args+=(--models "$spec"); done
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python late_interaction_eval.py scope \
    "${args[@]}" --n_boot 1000 --out_dir "$out"
}

# ---------------------------------------------------------------- queues
# ---------------------------------------------------------------- phase 2: the 8-hour campaign
# Two arms, two cards each, continuing existing late training. Data is the V2 paper recipe: whole
# dc40 corpus, k=8, uncapped, PROPORTIONAL -- 34.76M pairs (Pfam 0.78M / AFDB 19.0M / STRING 15M).
# Step budgets are set from measured rates so the cosine schedule anneals inside ~8 h.
PHASE2_ARGS=(--proj_dim 128 --max_pairs_per_file 0 --string_max_pairs 15000000 --seed 42
             --multi_dataset_sampler proportional --compile)
P2_STEPS="${P2_STEPS:-31000}"          # 35M:  x 256 pairs/step = 7.9M pairs, ~8.0 h
P2_STEPS_150M="${P2_STEPS_150M:-15000}"  # x 256 = 3.8M pairs, ~8.7 h at 2.10 s/it

queue_a() {  # ProtSent 35M, cards 0+1
  watch_curve 0 protsent_late_35m_prop "$RES/pilot_35m/scope"
  train_ddp 0,1 29521 protsent_late_35m_prop "$MODELS/protsent_late_proj128/late" \
    "$P2_STEPS" 5000 "${PHASE2_ARGS[@]}"
  wait
}

queue_b() {  # ProtSent 150M, cards 2+3. Continues a 64-D model, so backbone carries over but
             # the projection restarts at 128-D (logged as a shape mismatch).
  watch_curve 2 protsent_late_150m_prop "$RES/pilot_150m/scope"
  train_ddp 2,3 29522 protsent_late_150m_prop "$MODELS/protsent_late_150m/late" \
    "$P2_STEPS_150M" 5000 "${PHASE2_ARGS[@]}"
  wait
}

# ---------------------------------------------------------------- driver
# Default card per queue. Override any of them with GPU_P=3 etc.
declare -A DEFAULT_GPU=( [a]=0 [b]=2 )  # each arm takes this card and the next

case "${1:-start}" in
  start)
    for q in ${QUEUES:-a b}; do
      gpu_var="GPU_${q^^}"; gpu="${!gpu_var:-${DEFAULT_GPU[$q]:-0}}"
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
  bench)
    # Dense views only: the [L x d] late models cannot be fed to a pooled benchmark.
    # TASKS=full for the 20-task set; default is ProtBench's curated very-fast subset.
    shift
    ./run_late_bench.sh "$@"
    ;;
  stop)
    pkill -f "run_experiment_queue.sh __run"; pkill -f train_late_interaction.py; pkill -f late_interaction_eval.py
    echo "stopped (rerun 'start' to resume)"
    ;;
  *) echo "usage: $0 {start|status|bench|stop}"; exit 1 ;;
esac
