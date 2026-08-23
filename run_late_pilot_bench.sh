#!/usr/bin/env bash
# Pooled-embedding benchmarks for the late-interaction pilot: identical ProtBench
# invocation for every arm (same tasks, seed, split, probe). KNN is the primary
# probe; linear is optional and reported separately. Run AFTER training, with
# dense views exported.
set -euo pipefail
cd "$(dirname "$0")"

PROTBENCH_DIR="${PROTBENCH_DIR:-/opt/hpc/ddofer/ProtBench}"
OUT="${OUT:-$(pwd)/results/late_interaction/pilot_35m/benchmarks}"
PY="$(pwd)/.venv/bin/python"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# arm name -> model path/id (dense views must exist for the late arms)
declare -A MODELS=(
  [esm2_35m]="Synthyra/ESM2-35M"
  [protsent_v2_35m]="GrimSqueaker/ProtSent-V2-35M"
  [esm2_late_dense]="$(pwd)/models/late_interaction/esm2_late/dense_view"
  [protsent_late_dense]="$(pwd)/models/late_interaction/protsent_late/dense_view"
)

for probe in knn linear; do
  for arm in "${!MODELS[@]}"; do
    model="${MODELS[$arm]}"
    [[ -e "$model" || "$model" != /* ]] || { echo "SKIP $arm (missing $model)"; continue; }
    echo "=== $arm ($probe) $model"
    (cd "$PROTBENCH_DIR" && "$PY" protein_benchmark_suite.py \
        -m "$model" --fast --eval_split test -p "$probe" --knn_k 3 \
        --seed 42 --output_dir "$OUT/$probe")
  done
done
echo "collect with: $PY $PROTBENCH_DIR/collect_bench_results.py $OUT/knn (and .../linear)"
