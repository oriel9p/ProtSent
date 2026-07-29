#!/usr/bin/env bash
# Reviewer HNXd asked for a variability analysis over multiple random seeds, and
# Yi1G flagged single-run results. --seed_list runs every seed inside one process
# while reusing the loaded model and the embedding cache, so seeds after the first
# cost only the probe refit -- the whole sweep is a few minutes per arm.
#
# scope40_retrieval is omitted deliberately: retrieval has no probe randomness, so
# every seed returns an identical number. Its uncertainty is quantified by
# bootstrap_ci.py instead, which resamples queries.
set -uo pipefail
cd ~/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=flash_attention_2

SEEDS="${SEEDS:-0,1,2,3,4}"
TASKS="${TASKS:-remote_homology solubility stability thermostability fluorescence metal_ion_binding subcellular_loc variant_effect}"

run() {
  local tag="$1" model="$2"
  echo "=== $(date +%H:%M:%S) $tag ==="
  uv run --no-sync python protein_benchmark_suite.py \
    -m "$model" -t $TASKS -p knn -e test -b 64 --device cuda \
    --cache_embeddings --seed_list "$SEEDS" \
    -o "results/benchmarks/seeds/${tag}" >"logs/seeds/${tag}.log" 2>&1
  echo "    rc=$? $(ls results/benchmarks/seeds/${tag}/*.csv 2>/dev/null | head -1)"
}

run esm2_35m    /storage/models/ESM2-35M
run protsent_v1 oriel9p/protsent-esm2-35M
run protsent_v2 models/protsent_esm2_35m_v3/final
echo "=== done $(date) ==="
