#!/usr/bin/env bash
# Post-training benchmark sweep: ProtSent-v2-35M vs the ESM2-35M starting point.
#
# Runs kNN first, then linear probe, for each model. Both probes reuse the same
# cached embeddings (--cache_embeddings is on by default), so the second probe
# costs only the classifier fit -- ordering kNN first is therefore free.
#
# --eval_split test is REQUIRED for comparability with mmseqs_baseline.py, which
# scores each task's declared test split. The suite otherwise defaults to
# `validation`, which silently falls back to 4-fold CV on train for tasks with
# no validation split -- a different protocol, not a different seed.
set -uo pipefail
cd ~/ProtSent

MODEL_NEW="${MODEL_NEW:-models/protsent_esm2_35m_v3}"
MODEL_BASE="${MODEL_BASE:-/storage/models/ESM2-35M}"
OUT="${OUT:-results/benchmarks/v3}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-cuda}"

export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=flash_attention_2

mkdir -p "$OUT"

# scope40_retrieval is in RETRIEVAL_TASKS, which is opt-in and excluded from the
# defaults -- it must be named explicitly or it silently will not run.
TASKS="${TASKS:-scope40_retrieval remote_homology}"

run_one() {
  local model="$1" tag="$2" probe="$3"
  local log="logs/bench_v3/${tag}_${probe}.log"
  echo "=== $(date +%H:%M:%S) $tag / $probe ==="
  uv run --no-sync python protein_benchmark_suite.py \
    -m "$model" \
    -t $TASKS \
    -p "$probe" \
    -e test \
    -b "$BATCH" \
    --device "$DEVICE" \
    -o "$OUT/${tag}_${probe}" \
    >"$log" 2>&1
  local rc=$?
  # Exit code alone is NOT enough: the suite catches per-task exceptions, writes
  # them into an "Error" column, and still exits 0. A whole sweep can report "ok"
  # while every single row is a failure -- which is exactly what happened when
  # embed_dataset()'s signature changed. Check the CSV too.
  local csv errs
  csv=$(ls "$OUT/${tag}_${probe}"/*.csv 2>/dev/null | head -1)
  errs=0
  if [[ -n "$csv" ]] && head -1 "$csv" | grep -q "Error"; then
    errs=$(awk -F, 'NR>1 && $NF != "" {n++} END{print n+0}' "$csv")
  fi
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED rc=$rc -- see $log"
  elif [[ $errs -gt 0 ]]; then
    echo "    FAILED: $errs task(s) errored despite rc=0 -- see $csv"
  elif [[ -z "$csv" ]]; then
    echo "    FAILED: no results CSV written -- see $log"
  else
    echo "    ok ($(($(wc -l < "$csv") - 1)) rows)"
  fi
  return 0   # never let one task kill the sweep
}

mkdir -p logs/bench_v3

# A bare directory is not a model: the trainer creates $MODEL_NEW/debug_traces
# at startup, so `-d` alone is true within seconds of launch and the sweep will
# happily run against a checkpoint that does not exist. Require actual weights.
if ! ls "$MODEL_NEW"/*.safetensors "$MODEL_NEW"/pytorch_model.bin >/dev/null 2>&1; then
  echo "ERROR: no model weights under $MODEL_NEW -- training did not finish" >&2
  exit 1
fi

for probe in knn linear; do
  run_one "$MODEL_NEW"  protsent_v3 "$probe"
  run_one "$MODEL_BASE" esm2_35m    "$probe"
done

echo
echo "=== done $(date) -- results under $OUT ==="
grep -rh "Recall@10\|AUC" "$OUT" 2>/dev/null | head -20 || true
