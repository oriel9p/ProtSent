#!/usr/bin/env bash
# Wait for a target checkpoint, stop training (resumably), then benchmark several checkpoints
# in parallel -- one per GPU -- so the trend across training is measured, not just the endpoint.
#
#   TRAIN_PID=<rank0 pid> TARGET=10000 MARKS="1000 4000 10000" ./stop_at_and_bench.sh
#
# Exits non-zero if any benchmark failed, and leaves the dense views in place when it does, so a
# retry does not have to rebuild them. Reporting success after a sweep that produced nothing is
# the failure mode that actually costs time here.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
RUN="${RUN:-vanilla35m_clean}"
TARGET="${TARGET:-10000}"
MARKS="${MARKS:-1000 4000 10000}"
PROBES="${PROBES:-knn}"   # knn is the primary probe; linear doubles the sweep for a view nothing reads
SETTLE="${SETTLE:-60}"    # grace after the checkpoint files appear, before killing the writer
SWEEP_WARN_S="${SWEEP_WARN_S:-4500}"   # 1.25 h: marks run in parallel, so this is wall clock for all of them
POLL="${POLL:-120}"
TRAIN_PID="${TRAIN_PID:?rank0 pid of the training process}"
D="models/late_interaction/$RUN"
# clean_35m names the RECIPE and SIZE (r2, 35M), so both r2 35M runs share it deliberately --
# co-located arms are what makes the comparison hit identical references. Override for anything
# that is not an r2 35M run.
OUTDIR="${OUTDIR:-$(pwd)/results/late_interaction/clean_35m/benchmarks}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NGPU" -ge 1 ]] 2>/dev/null || NGPU=1

echo "$(date +%H:%M) waiting for checkpoint-$TARGET (train pid $TRAIN_PID)"
while true; do
  # trainer_state.json + optimizer.pt are the last files a complete save writes; both must exist
  # or a kill here would leave a checkpoint that cannot be resumed from.
  if [[ -f "$D/checkpoint-$TARGET/trainer_state.json" && -f "$D/checkpoint-$TARGET/optimizer.pt" ]]; then
    echo "$(date +%H:%M) checkpoint-$TARGET complete; settling ${SETTLE}s before stop"
    sleep "$SETTLE"
    break
  fi
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "$(date +%H:%M) training exited before $TARGET"; break; }
  sleep "$POLL"
done

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "$(date +%H:%M) stopping training (resume later with --resume, which finds the latest checkpoint-*)"
  pkill -f "train_late_interaction.py.*$RUN" 2>/dev/null
  sleep 15
  pkill -9 -f "train_late_interaction.py.*$RUN" 2>/dev/null
  sleep 5
  # A rank wedged in D-state on NCCL survives SIGKILL until its collective times out. Benchmarking
  # around it yields contention-skewed numbers that look perfectly valid, so refuse instead.
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "$(date +%H:%M) ABORT: training pid $TRAIN_PID survived the stop; not benchmarking on busy GPUs"
    exit 1
  fi
fi

# The sequential mark-follower would otherwise claim a GPU for its own mark mid-sweep.
pkill -f "bench_marks_clean.sh" 2>/dev/null && echo "$(date +%H:%M) stopped the sequential mark-follower"
sleep 3
echo "$(date +%H:%M) GPUs free ($NGPU); benchmarking marks: $MARKS"

# Dense views first (CPU, cheap) so the GPU stage is pure benchmarking.
for N in $MARKS; do
  snap="$D/snapshots/step-$N"; dense="$snap-dense"
  [[ -d "$snap" ]] || { echo "no snapshot for $N, skipping"; continue; }
  [[ -d "$dense" ]] && continue
  uv run --no-sync python - "$snap" "$dense" <<'PY'
import sys
import late_interaction as li
snap, dense = sys.argv[1], sys.argv[2]
mve, st = li.build_multivector_encoder(snap, proj_dim=128, max_seq_length=512, device="cpu")
st.save(dense)
print(f"dense view -> {dense}")
PY
done

# One checkpoint per GPU, identical task set and probes so the marks stay comparable.
sweep_start=$(date +%s)
pids=(); marks_run=(); gpu=0
for N in $MARKS; do
  dense="$D/snapshots/step-$N-dense"
  [[ -d "$dense" ]] || continue
  CUDA_VISIBLE_DEVICES=$gpu TASKS=cheap PROBES="$PROBES" OUT="$OUTDIR" \
    ./run_late_bench.sh "${RUN}_s${N}=$(pwd)/$dense" > "logs/bench_${RUN}_s${N}.log" 2>&1 &
  pids+=("$!"); marks_run+=("$N")
  gpu=$(( (gpu + 1) % NGPU ))
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "$(date +%H:%M) mark ${marks_run[$i]} done"
  else
    echo "$(date +%H:%M) mark ${marks_run[$i]} FAILED (see logs/bench_${RUN}_s${marks_run[$i]}.log)"
    fail=$((fail + 1))
  fi
done

elapsed=$(( $(date +%s) - sweep_start ))
echo "$(date +%H:%M) sweep wall clock: $((elapsed / 60)) min for ${#pids[@]} marks in parallel"
if [[ $elapsed -gt $SWEEP_WARN_S ]]; then
  echo "$(date +%H:%M) WARN: sweep took $((elapsed / 60)) min, over the $((SWEEP_WARN_S / 60)) min threshold." \
       "Marks already run one-per-GPU, so this is a per-mark cost, not a scheduling problem;" \
       "ss3 is the residue-level task and the usual culprit."
fi

if [[ $fail -gt 0 ]]; then
  echo "$(date +%H:%M) $fail of ${#pids[@]} marks failed; keeping dense views so a retry can reuse them"
  exit 1
fi

# Disposable adapter -- 129 MB per checkpoint whose only job is to give the pooled benchmark
# something it can consume. The snapshots they were derived from stay.
rm -rf "$D"/snapshots/*-dense
echo "$(date +%H:%M) all marks benchmarked; results in $OUTDIR"
