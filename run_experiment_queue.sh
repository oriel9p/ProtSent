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
# ---------------------------------------------------------------- phase 2: the 50k campaign
#
# Data matches the ProtSent-V2 paper recipe: the WHOLE dc40 corpus at k=8 pairs per cluster, no
# caps. That builds Pfam 777,306 + AFDB 18,987,468 + STRING 15,000,000 = 34.76M pairs, against
# V2's own 34.8M.
#
# PROPORTIONAL samples each source in proportion to its size, as V2 did: AFDB 54.6%, STRING
# 43.1%, **Pfam 2.2%**. That last number is the risk in this design -- round-robin gave Pfam 33%,
# and Pfam is the family-level signal SCOPe rewards most. queue_pr is the control that measures
# it: same start point, same steps, ROUND_ROBIN.
#
# 50,000 steps x 128 = 6.4M pairs, 12.5x the pilot arms, ~9.4 h per 35M card. The 150M arm gets
# 25,000 steps to land in the same wallclock. All arms resume, so a stop costs nothing.
PHASE2_ARGS=(--proj_dim 128 --max_pairs_per_file 0 --string_max_pairs 15000000 --seed 42 --compile)
P2_STEPS="${P2_STEPS:-50000}"
P2_STEPS_150M="${P2_STEPS_150M:-25000}"
# The 35M ProtSent arms continue from the best validated 128-D model rather than from V2 itself:
# 4,000 fp32 steps, fully annealed, .7057 SCOPe superfamily eligible MAP. build_multivector_encoder
# reuses that checkpoint's trained projection head instead of randomising a fresh one.
P2_BASE_35M="${P2_BASE_35M:-$MODELS/protsent_late_proj128/late}"

queue_pp() {  # ProtSent 35M, PROPORTIONAL -- the priority arm
  local gpu="$1"
  watch_curve "$gpu" protsent_late_prop50k "$RES/pilot_35m/scope"
  train "$gpu" protsent_late_prop50k "$P2_BASE_35M" "$P2_STEPS" 5000 \
    "${PHASE2_ARGS[@]}" --multi_dataset_sampler proportional
  wait
}

queue_pr() {  # identical start and budget, ROUND_ROBIN -- isolates the sampler
  local gpu="$1"
  watch_curve "$gpu" protsent_late_rr50k "$RES/pilot_35m/scope"
  train "$gpu" protsent_late_rr50k "$P2_BASE_35M" "$P2_STEPS" 5000 \
    "${PHASE2_ARGS[@]}" --multi_dataset_sampler round_robin
  wait
}

queue_vp() {  # vanilla ESM-2 35M -- isolates the starting point
  local gpu="$1"
  watch_curve "$gpu" esm2_late_prop50k "$RES/pilot_35m/scope"
  train "$gpu" esm2_late_prop50k facebook/esm2_t12_35M_UR50D "$P2_STEPS" 5000 \
    "${PHASE2_ARGS[@]}" --multi_dataset_sampler proportional
  wait
}

queue_lp() {  # ProtSent 150M -- isolates scale. No 128-D 150M model exists to continue from,
              # so this one starts from V2-150M with a fresh head.
  local gpu="$1"
  watch_curve "$gpu" protsent_late_150m_prop25k "$RES/pilot_150m/scope"
  train "$gpu" protsent_late_150m_prop25k GrimSqueaker/ProtSent-V2-150M "$P2_STEPS_150M" 5000 \
    "${PHASE2_ARGS[@]}" --multi_dataset_sampler proportional
  wait
}

# ---------------------------------------------------------------- driver
# Default card per queue. Override any of them with GPU_P=3 etc.
declare -A DEFAULT_GPU=( [pp]=0 [pr]=1 [vp]=2 [lp]=3 )

case "${1:-start}" in
  start)
    for q in ${QUEUES:-pp pr vp lp}; do
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
