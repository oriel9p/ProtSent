#!/usr/bin/env bash
# Pooled-embedding benchmarks for late-interaction arms: identical ProtBench invocation for
# every arm (same tasks, seed, split, probe), so arms stay comparable. KNN is the primary
# probe; linear is reported separately and never averaged with it.
#
# The late models themselves are [L x d] and cannot be fed to a pooled benchmark. What is
# scored here is each arm's DENSE VIEW -- the same trained backbone with mean pooling -- which
# train_late_interaction.py exports next to every model it saves.
#
# Usage:
#   ./run_late_bench.sh                          # the long-run arms, cheap task set
#   ./run_late_bench.sh name=path [name=path ...]
#   TASKS=full ./run_late_bench.sh               # everything --fast covers
#   TASKS="remote_homology ec_classification" ./run_late_bench.sh
#   PROBES=knn ./run_late_bench.sh
set -euo pipefail
cd "$(dirname "$0")"

PROTBENCH_DIR="${PROTBENCH_DIR:-/opt/hpc/ddofer/ProtBench}"
OUT="${OUT:-$(pwd)/results/late_interaction/pilot_35m/benchmarks}"
PY="$(pwd)/.venv/bin/python"
PROBES="${PROBES:-knn linear}"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Task presets are ProtBench's own, not a third curation maintained here:
#   cheap -> --very-fast   (its curated low-variance scout subset)
#   full  -> --fast        (the 20-task set the pilot arms were scored on)
# SCOPe and CATH are deliberately absent from both: retrieval is already covered per-checkpoint
# by `late_interaction_eval.py watch_curve` (MaxSim) and `scope` (dense cosine), which score the
# multi-vector model itself rather than its pooled view.
case "${TASKS:-cheap}" in
  cheap) TASK_ARGS=(--very-fast) ;;
  full)  TASK_ARGS=(--fast) ;;
  *)     TASK_ARGS=(-t ${TASKS}) ;;
esac

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  M="$(pwd)/models/late_interaction"
  ARMS=(
    "esm2_35m=facebook/esm2_t12_35M_UR50D"
    "protsent_v2_35m=GrimSqueaker/ProtSent-V2-35M"
    "protsent_late_long=$M/protsent_late_long/dense_view"
    "esm2_late_long=$M/esm2_late_long/dense_view"
    "protsent_late_pool_control=$M/protsent_late_pool_control/dense_view"
  )
fi

for probe in $PROBES; do
  for arm in "${ARMS[@]}"; do
    model="${arm#*=}"
    # A local path that does not exist means the arm has not finished training yet; a bare HF id
    # (no leading /) is always allowed through.
    [[ -e "$model" || "$model" != /* ]] || { echo "SKIP ${arm%%=*} (missing $model)"; continue; }
    echo "=== ${arm%%=*} ($probe) $model"
    (cd "$PROTBENCH_DIR" && "$PY" protein_benchmark_suite.py \
        -m "$model" "${TASK_ARGS[@]}" --eval_split test -p "$probe" --knn_k 3 \
        --seed 42 --output_dir "$OUT/$probe")
  done
done
echo "collect with: $PY $PROTBENCH_DIR/collect_bench_results.py $OUT/knn (and .../linear)"
