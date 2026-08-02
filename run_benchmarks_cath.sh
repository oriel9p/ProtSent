#!/usr/bin/env bash
# CATH v4.3 midnight-zone annotation transfer (Heinzinger 2022, ProtTucker/EAT)
# for our ESM2 arms: vanilla base, ProtSent V1 (the submitted model), ProtSent V2.
#
# Why this benchmark: our own rebuttal concedes ProtTucker is the closest
# analogue to ProtSent's protocol -- both contrastively post-train a frozen
# pLM's per-protein embeddings and score by 1-NN annotation transfer -- and the
# comparison we would most want to add. This is that comparison.
#
# The task lives in ~/ProtBench (the refactored benchmark repo), not here: it
# ships the cath_eat TaskConfig and the GrimSqueaker/cath43-eat dataset built
# from Rostlab/EAT's own splits. We use ProtBench ONLY for CATH. Migrating the
# existing ProtSent sweeps onto it would risk silently changing every already
# published number, since the two suites have diverged.
#
# -p knn --knn_k 1 is REQUIRED, not a default. ProtBench's knn probe is
# KNeighborsClassifier(n_neighbors=1, metric="euclidean", algorithm="brute")
# with NO StandardScaler, which is literally EAT's method. The linear probe
# standardises features AND would fit 6.5k classes over 69k rows -- a different
# experiment that is not comparable to the paper.
#
# -e test is REQUIRED. The suite defaults to --eval_split validation, and this
# dataset HAS a validation split (EAT's val200), so a bare run silently scores
# the wrong thing. With -e test it resolves to test_h: the 150 of 219 queries
# whose superfamily exists in the lookup set at all. That matches the paper's
# H-level denominator, which is also 150.
#
# Paper Table 1 for reference (accuracy at H, test219 -> lookup69k):
#   MMseqs2 35 | raw ProtT5 64 | ProtTucker(ProtT5) 76 | HMMER profiles 77
# ProtBench measured on this same task: 3-mers 0.0, ESM2-8M 21.3, ESM2-650M 42.7.
# The ESM2 family sits far below ProtT5 here, so the number that matters is each
# ProtSent arm against ITS OWN frozen base, not against 64.
set -uo pipefail

BENCH_REPO="${BENCH_REPO:-/home/ddofer/ProtBench}"
PY="${PY:-/home/ddofer/ProtSent/.venv/bin/python}"
# ProtBench is a flat script repo with no venv of its own (pyproject sets
# package=false); the ProtSent venv already has every dependency it declares.
cd "$BENCH_REPO"

OUT="${OUT:-/home/ddofer/ProtSent/results/benchmarks/cath_eat}"
LOGS="${LOGS:-/home/ddofer/ProtSent/logs/bench_cath}"
STATUS="${STATUS:-/home/ddofer/ProtSent/bench_arm_status.py}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
# Only GPU 2 is idle; GPUs 0,1,3-7 are running protJepa. Do not widen this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
# The models are fp32-resident and the suite defaults to --amp_dtype fp32, but
# from_pretrained_with_flash auto-selects flash_attention_2 whenever flash-attn
# is importable. FA2 accepts only bf16 and raises "'flash_attention_2' supports
# only manifest-declared dtype(s) bfloat16; received float32". sdpa is also what
# our existing ProtSent numbers effectively ran under, so this keeps the arms
# consistent with the rest of the paper rather than merely working.
export PROTEIN_BENCH_ATTN_IMPLEMENTATION=sdpa
# 256 cores but OpenBLAS is precompiled for far fewer threads; uncapped, large
# probe fits SIGABRT with "corrupted size vs. prev_size".
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

mkdir -p "$OUT" "$LOGS"

# tag <TAB> model. V1 arms are not optional: the paper under review used V1, so
# the reviewer-facing number has to be V1's. The /final suffix on V2 paths is
# load-bearing -- SentenceTransformerTrainer writes there, not to the parent.
ARMS=(
  "esm2_35m|/storage/models/ESM2-35M"
  "protsent_v1_35m|oriel9p/protsent-esm2-35M"
  "protsent_v2_35m|/home/ddofer/ProtSent/models/protsent_esm2_35m_v3/final"
  "esm2_150m|Synthyra/ESM2-150M"
  "protsent_v1_150m|oriel9p/protsent-esm2-150M"
  "protsent_v2_150m|/home/ddofer/ProtSent/models/protsent_esm2_150m_v2/final"
)

# A completed arm = the task has at least one row with an empty Error column.
# Row count is the wrong test: the suite APPENDS, so a rerun leaves two rows and
# a failed rerun of a good arm still has more rows than tasks.
arm_is_complete() {
  local dir="$1" csv
  csv=$(ls "$dir"/*.csv 2>/dev/null | head -1)
  [[ -n "$csv" ]] || return 1
  "$PY" "$STATUS" "$csv" 1 >/dev/null 2>&1
}

run_one() {
  local tag="$1" model="$2"
  local log="$LOGS/${tag}.log"
  if [[ "${FORCE:-0}" != "1" ]] && arm_is_complete "$OUT/$tag"; then
    echo "=== $(date +%H:%M:%S) $tag -- already complete, skipping ==="
    return 0
  fi
  echo "=== $(date +%H:%M:%S) $tag  ($model) ==="
  local t0=$SECONDS
  "$PY" protein_benchmark_suite.py \
    -m "$model" \
    --tasks cath_eat \
    -p knn --knn_k 1 \
    -e test \
    --cache_embeddings \
    --bootstrap "$BOOTSTRAP" \
    -b "$BATCH" \
    --device "$DEVICE" \
    -o "$OUT/$tag" \
    >"$log" 2>&1
  local rc=$? el=$((SECONDS - t0))
  # Exit code alone is NOT enough: the suite catches per-task exceptions, writes
  # them to an "Error" column, and still exits 0. Read the CSV.
  local csv status
  csv=$(ls "$OUT/$tag"/*.csv 2>/dev/null | head -1)
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED rc=$rc after ${el}s -- see $log"
  elif [[ -z "$csv" ]]; then
    echo "    FAILED: no results CSV written -- see $log"
  elif status=$("$PY" "$STATUS" "$csv" 1 2>&1); then
    echo "    ok in ${el}s ($status)"
    "$PY" - "$csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
d = d[d.get("Error", pd.Series([float("nan")] * len(d))).isna()]
if len(d):
    r = d.iloc[-1]
    print(f"    H-level accuracy = {r['Accuracy']:.4f}  (n={int(r['Samples'])})")
EOF
  else
    echo "    FAILED: $status -- see $csv"
  fi
  return 0   # never let one arm kill the sweep
}

for arm in "${ARMS[@]}"; do
  run_one "${arm%%|*}" "${arm#*|}"
done

echo
echo "=== done $(date) -- results under $OUT ==="
