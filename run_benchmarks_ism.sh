#!/usr/bin/env bash
# ISM-C-300M vs vanilla ESM-C-300M on the 23-task suite plus SCOPe-40 retrieval.
#
# Reviewer jVGf asked us to position ProtSent against structure-informed protein
# LMs (ESM-S, S-PLM, ISM, Magneton). ISM is the only one of the four with usable
# public weights, so this measures it on our own benchmark rather than describing
# it. ISM-C-300M is a structure-distilled ESM-C-300M, so `Synthyra/ESMplusplus_small`
# -- which IS vanilla ESM-C-300M -- is the matched control: same architecture,
# same parameter count, same tokenizer, differing only in the distillation. Run
# alone, the ISM-C numbers would be uninterpretable.
#
# ISM-C ships as a bare .pth with no config or tokenizer; convert_ismc_to_hf.py
# builds $MODEL_ISM from it and gates the conversion. Run that first.
#
# This is deliberately NOT a wrapper over run_benchmarks_v3.sh. That script
# hard-exits when $MODEL_NEW/*.safetensors is absent and then runs
# make_checkpoint_loadable.py, both of which are specific to our own ESM2
# training checkpoints and wrong for a third-party model.
#
# --eval_split test is REQUIRED for comparability with the mmseqs2/HMMER
# baselines and with the 35M and 150M arms, all of which scored each task's
# declared test split.
set -uo pipefail
cd ~/ProtSent

MODEL_ISM="${MODEL_ISM:-/storage/models/ISM-C-300M}"
MODEL_ESMC="${MODEL_ESMC:-Synthyra/ESMplusplus_small}"
TAG_ISM="${TAG_ISM:-ismc_300m}"
TAG_ESMC="${TAG_ESMC:-esmc_300m}"
OUT="${OUT:-results/benchmarks/ism}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-cuda}"
# Only GPU 2 is idle; the other seven are running someone else's job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=flash_attention_2
# This box has 256 cores but OpenBLAS is precompiled for far fewer threads. Left
# uncapped, the probe fits abort with "corrupted size vs. prev_size" (SIGABRT,
# rc=134) once a task is large enough -- aav_flip, at 50,430 x 22,186 sequences,
# reproduces it every time.
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

mkdir -p "$OUT" logs/bench_ism

# Identical to run_benchmarks_v3.sh: the 23 tasks that have a paired MMseqs2
# baseline row. scope40_retrieval is in RETRIEVAL_TASKS, which is opt-in and
# excluded from the suite defaults -- it must be named explicitly or it silently
# will not run. rhla_enzyme_mutations is omitted: its sequences are 6-residue
# mutation-site strings, too short for MMseqs2 k-mers.
TASKS="${TASKS:-aav_flip antibiotic_resistance beta_lactamase_peer \
binary_subcellular_localization cloning_clf ec_classification \
enzyme_catalytic_efficiency fluorescence go_mf material_production \
metal_ion_binding optimal_ph peptide_hla profet_np_sp_cleaved remote_homology \
scope40_retrieval signalp_binary solubility stability subcellular_loc \
temperature_stability thermostability variant_effect}"

n_tasks() { wc -w <<<"$TASKS"; }

# A completed arm = every requested task has at least one row with an empty Error
# column. Row count is the wrong test: the suite appends, so a rerun of 23 tasks
# leaves 46 rows, and a partial arm can still have more rows than tasks.
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
  # Exit code alone is NOT enough: the suite catches per-task exceptions, writes
  # them into an "Error" column, and still exits 0. A whole sweep can report "ok"
  # while every single row is a failure. Check the CSV too.
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
  return 0   # never let one task kill the sweep
}

if [[ ! -f "$MODEL_ISM/model.safetensors" ]]; then
  echo "ERROR: no converted model at $MODEL_ISM." >&2
  echo "       Run: uv run --no-sync python convert_ismc_to_hf.py" >&2
  exit 1
fi

for probe in knn linear; do
  run_one "$MODEL_ISM"  "$TAG_ISM"  "$probe"
  run_one "$MODEL_ESMC" "$TAG_ESMC" "$probe"
done

echo
echo "=== done $(date) -- results under $OUT ==="
