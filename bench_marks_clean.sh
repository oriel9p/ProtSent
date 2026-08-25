#!/usr/bin/env bash
# Very-fast pooled benchmarks at fixed checkpoint marks of the clean runs.
# For each mark: wait for the weights-only snapshot, export its dense view, run the cheap
# task set (knn only -- the fast probe) against it. SCOPe per checkpoint is watch_curve's job.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
RUN="${RUN:-vanilla35m_clean}"
MARKS="${MARKS:-1000 5000 10000 30000}"
GPU="${GPU:-1}"
TRAIN_PID="${TRAIN_PID:?pid of the training process to follow}"
OUTDIR="results/late_interaction/clean_35m/benchmarks"

for N in $MARKS; do
  snap="models/late_interaction/$RUN/snapshots/step-$N"
  while [[ ! -d "$snap" ]]; do
    kill -0 "$TRAIN_PID" 2>/dev/null || { echo "training gone before step $N; stopping"; exit 0; }
    sleep 120
  done
  dense="$snap-dense"
  if [[ ! -d "$dense" ]]; then
    uv run --no-sync python - "$snap" "$dense" <<'PY'
import sys
import late_interaction as li
snap, dense = sys.argv[1], sys.argv[2]
mve, st = li.build_multivector_encoder(snap, proj_dim=128, max_seq_length=512, device="cpu")
st.save(dense)
print(f"dense view -> {dense}")
PY
  fi
  echo "=== mark $N: cheap knn bench $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU TASKS=cheap PROBES=knn OUT="$OUTDIR" \
    ./run_late_bench.sh "${RUN}_s${N}=$(pwd)/$dense"
done
echo "all marks done"
