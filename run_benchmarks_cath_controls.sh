#!/usr/bin/env bash
# Controls for the CATH midnight-zone result. These do not add models we care
# about; they exist to attack the headline number before a reviewer does.
#
# 1. L2 CONTROL (the important one). ProtSent is trained with a cosine objective
#    (CachedMultipleNegativesRankingLoss) and its SentenceTransformer head emits
#    embeddings that are effectively direction-only. The probe scores EUCLIDEAN
#    distance on RAW embeddings, and Euclidean distance between unnormalised
#    vectors is dominated by norm differences. So part of the ProtSent gain over
#    vanilla could be "the embeddings are normalised", not "the representation is
#    better" -- and we already know from the whitening control on SCOPe-40 that
#    this mechanism explains much of ProtSent's kNN gain there.
#    Running vanilla WITH --l2_normalize_embeddings gives vanilla the same
#    geometry for free. Whatever gap survives is the part that is representation.
#
# 2. SCALING CHECK. ProtBench's docs record ESM2-650M at 42.7 on this task,
#    which is BELOW our ESM2-150M (43.3) -- scaling says that should not happen.
#    Those reference numbers used facebook/esm2_*, which loads via the plain HF
#    path, while our arms load via FastPLM. Same architecture, different code.
#    Running 650M and 35M through facebook/* here separates "our arms are high"
#    from "that reference number is low".
#
# 3. FLOOR. k-mer frequencies should score ~0, confirming the task has no
#    composition shortcut. ProtBench reports 0.0; reproduce it here.
set -uo pipefail

BENCH_REPO="${BENCH_REPO:-/home/ddofer/ProtBench}"
PY="${PY:-/home/ddofer/ProtSent/.venv/bin/python}"
cd "$BENCH_REPO"

OUT="${OUT:-/home/ddofer/ProtSent/results/benchmarks/cath_eat_controls}"
LOGS="${LOGS:-/home/ddofer/ProtSent/logs/bench_cath}"
STATUS="${STATUS:-/home/ddofer/ProtSent/bench_arm_status.py}"
BATCH="${BATCH:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
export PROTEIN_BENCH_ATTN_IMPLEMENTATION=sdpa
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

mkdir -p "$OUT" "$LOGS"

# tag|model|extra-flags
ARMS=(
  "esm2_35m_l2|/storage/models/ESM2-35M|--l2_normalize_embeddings"
  "esm2_150m_l2|Synthyra/ESM2-150M|--l2_normalize_embeddings"
  "protsent_v2_35m_l2|/home/ddofer/ProtSent/models/protsent_esm2_35m_v3/final|--l2_normalize_embeddings"
  "protsent_v2_150m_l2|/home/ddofer/ProtSent/models/protsent_esm2_150m_v2/final|--l2_normalize_embeddings"
  "hf_esm2_35m|facebook/esm2_t12_35M_UR50D|"
  "hf_esm2_650m|facebook/esm2_t33_650M_UR50D|"
  "kmer3|kmer|"
)

run_one() {
  local tag="$1" model="$2" extra="$3"
  local log="$LOGS/ctrl_${tag}.log"
  local csv
  csv=$(ls "$OUT/$tag"/*.csv 2>/dev/null | head -1)
  if [[ "${FORCE:-0}" != "1" && -n "$csv" ]] && "$PY" "$STATUS" "$csv" 1 >/dev/null 2>&1; then
    echo "=== $(date +%H:%M:%S) $tag -- already complete, skipping ==="
    return 0
  fi
  echo "=== $(date +%H:%M:%S) $tag  ($model $extra) ==="
  local t0=$SECONDS
  # shellcheck disable=SC2086  # $extra is an intentional flag list, may be empty
  "$PY" protein_benchmark_suite.py \
    -m "$model" --tasks cath_eat -p knn --knn_k 1 -e test \
    --cache_embeddings --bootstrap 1000 -b "$BATCH" --device cuda \
    $extra -o "$OUT/$tag" >"$log" 2>&1
  local rc=$? el=$((SECONDS - t0))
  csv=$(ls "$OUT/$tag"/*.csv 2>/dev/null | head -1)
  if [[ $rc -ne 0 || -z "$csv" ]]; then
    echo "    FAILED rc=$rc after ${el}s -- see $log"
  else
    "$PY" - "$csv" "$tag" "$el" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
if "Error" in d.columns:
    bad = d[d["Error"].notna()]
    d = d[d["Error"].isna()]
    if len(bad) and not len(d):
        print(f"    FAILED: {bad.iloc[-1]['Error']}")
        raise SystemExit
if len(d):
    r = d.iloc[-1]
    print(f"    ok in {sys.argv[3]}s  H={100 * r['Accuracy']:.2f}  norm={r.get('EmbeddingNorm', '?')}")
EOF
  fi
  return 0
}

for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag model extra <<<"$arm"
  run_one "$tag" "$model" "$extra"
done

echo
echo "=== done $(date) -- results under $OUT ==="
