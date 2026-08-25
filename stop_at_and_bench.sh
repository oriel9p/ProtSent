#!/usr/bin/env bash
# Wait for a target checkpoint, stop training (resumably), then benchmark several checkpoints
# in parallel -- one per GPU -- so the trend across training is measured, not just the endpoint.
#
#   TRAIN_PID=<rank0> TARGET=20000 MARKS="5000 10000 15000 20000" ./stop_at_and_bench.sh
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
RUN="${RUN:-vanilla35m_clean}"
TARGET="${TARGET:-10000}"
MARKS="${MARKS:-1000 4000 7000 10000}"  # even spacing: early / mid / late trend
PROBES="${PROBES:-knn}"   # knn is the primary probe; linear doubles the sweep for a second view we do not need here
TRAIN_PID="${TRAIN_PID:?rank0 pid of the training process}"
D="models/late_interaction/$RUN"
OUTDIR="$(pwd)/results/late_interaction/clean_35m/benchmarks"

echo "$(date +%H:%M) waiting for checkpoint-$TARGET (train pid $TRAIN_PID)"
while true; do
  # trainer_state.json + optimizer.pt are the last files a complete save writes; both must exist
  # or a kill here would leave a checkpoint that cannot be resumed from.
  if [[ -f "$D/checkpoint-$TARGET/trainer_state.json" && -f "$D/checkpoint-$TARGET/optimizer.pt" ]]; then
    echo "$(date +%H:%M) checkpoint-$TARGET complete; settling 60s before stop"
    sleep 60
    break
  fi
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "$(date +%H:%M) training exited before $TARGET"; break; }
  sleep 120
done

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "$(date +%H:%M) stopping training (resume later with --resume, which finds the latest checkpoint-*)"
  pkill -f "train_late_interaction.py.*$RUN" 2>/dev/null
  sleep 15
  pkill -9 -f "train_late_interaction.py.*$RUN" 2>/dev/null
  sleep 5
fi
# The sequential mark-follower would otherwise grab a GPU for its own 10000 mark mid-sweep.
pkill -f "bench_marks_clean.sh" 2>/dev/null && echo "$(date +%H:%M) stopped the sequential mark-follower"
sleep 3
echo "$(date +%H:%M) GPUs free; benchmarking marks: $MARKS"

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

# One checkpoint per GPU, all with identical task set and probes so the marks are comparable.
gpu=0
for N in $MARKS; do
  dense="$D/snapshots/step-$N-dense"
  [[ -d "$dense" ]] || continue
  ( CUDA_VISIBLE_DEVICES=$gpu TASKS=cheap PROBES="$PROBES" OUT="$OUTDIR" \
      ./run_late_bench.sh "${RUN}_s${N}=$(pwd)/$dense" \
      > "logs/bench_${RUN}_s${N}.log" 2>&1
    echo "$(date +%H:%M) mark $N done" ) &
  gpu=$(( (gpu + 1) % 4 ))
done
wait

# The dense views are a disposable adapter -- 129 MB per checkpoint whose only job is to give
# the pooled benchmark something it can consume. The snapshots they were derived from stay.
rm -rf "$D"/snapshots/*-dense
echo "$(date +%H:%M) all marks benchmarked; results in $OUTDIR"
