#!/usr/bin/env bash
# ProtSent-trained ESM-C 300M V2 on the same 23-task suite as
# results/benchmarks/ism/{esmc,ismc}_300m_{knn,linear}, so it is directly
# comparable to the vanilla ESM-C-300M and ISM-C-300M rows already measured
# there. Mirrors run_benchmarks_ism.sh's run_one(), single model only.
set -uo pipefail
cd ~/ProtSent

MODEL="${MODEL:-/storage/users/ddofer/protsent_models/protsent_esmc_300m_v2/final}"
TAG="${TAG:-protsent_esmc_300m_v2}"
OUT="${OUT:-results/benchmarks/ism}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=flash_attention_2
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

mkdir -p "$OUT" logs/bench_ism

TASKS="${TASKS:-aav_flip antibiotic_resistance beta_lactamase_peer \
binary_subcellular_localization cloning_clf ec_classification \
enzyme_catalytic_efficiency fluorescence go_mf material_production \
metal_ion_binding optimal_ph peptide_hla profet_np_sp_cleaved remote_homology \
scope40_retrieval signalp_binary solubility stability subcellular_loc \
temperature_stability thermostability variant_effect}"

n_tasks() { wc -w <<<"$TASKS"; }

arm_is_complete() {
  local dir="$1" want="$2"
  local csv
  csv=$(ls "$dir"/*.csv 2>/dev/null | head -1)
  [[ -n "$csv" ]] || return 1
  uv run --no-sync python bench_arm_status.py "$csv" "$want" >/dev/null 2>&1
}

run_one() {
  local model="$1" tag="$2" probe="$3"
  local log="logs/bench_ism/${tag}_${probe}.log"
  if [[ "${FORCE:-0}" != "1" ]] && arm_is_complete "$OUT/${tag}_${probe}" "$(n_tasks)"; then
    echo "=== $(date +%H:%M:%S) $tag / $probe -- already complete, skipping ==="
    return 0
  fi
  echo "=== $(date +%H:%M:%S) $tag / $probe ==="
  uv run --no-sync python protein_benchmark_suite.py \
    -m "$model" \
    -t $TASKS \
    -p "$probe" \
    -e test \
    --cache_embeddings \
    -b "$BATCH" \
    --device "$DEVICE" \
    -o "$OUT/${tag}_${probe}" \
    >"$log" 2>&1
  local rc=$?
  local csv status
  csv=$(ls "$OUT/${tag}_${probe}"/*.csv 2>/dev/null | head -1)
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED rc=$rc -- see $log"
  elif [[ -z "$csv" ]]; then
    echo "    FAILED: no results CSV written -- see $log"
  elif status=$(uv run --no-sync python bench_arm_status.py "$csv" "$(n_tasks)" 2>&1); then
    echo "    ok ($status)"
  else
    echo "    FAILED: $status -- see $csv"
  fi
  return 0
}

if [[ ! -f "$MODEL/model.safetensors" ]]; then
  echo "ERROR: no model at $MODEL." >&2
  exit 1
fi

for probe in knn linear; do
  run_one "$MODEL" "$TAG" "$probe"
done

echo
echo "=== done $(date) -- results under $OUT ==="
