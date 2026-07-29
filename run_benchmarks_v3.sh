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
MODEL_OLD="${MODEL_OLD:-oriel9p/protsent-esm2-35M}"
MODEL_BASE="${MODEL_BASE:-/storage/models/ESM2-35M}"
# The LR schedule is a 3-cycle cosine (--lr_num_cycles 3.0), so the final step
# sits at the full 2e-4 peak rather than annealed. checkpoint-4000 (LR 5.5e-5) is
# the saved checkpoint nearest the last trough at step 4,208, kept aside by
# snapshot_ckpt.sh because save_total_limit=2 would otherwise delete it. Run as a
# fourth arm so the final model is never quoted as a converged optimum without a
# settled-LR comparison. Skipped if the snapshot is absent.
MODEL_TROUGH="${MODEL_TROUGH:-models/protsent_esm2_35m_v3_snapshots/checkpoint-4000}"
OUT="${OUT:-results/benchmarks/v3}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-cuda}"

export HF_HOME=/storage/models/hf_home
export TOKENIZERS_PARALLELISM=false
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=flash_attention_2
# This box has 256 cores but OpenBLAS is precompiled for far fewer threads.
# Left uncapped, the kNN/linear probe fits abort with "corrupted size vs.
# prev_size" (SIGABRT, rc=134) once a task is large enough -- aav_flip, at
# 50,430 x 22,186 sequences, reproduces it every time. Nested joblib/loky
# parallelism on top makes it worse. Capping fixes it and costs nothing here,
# since the probe fits are not the bottleneck.
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

mkdir -p "$OUT"

# The 23 tasks that have a paired MMseqs2 baseline row, so every model number has
# an alignment reference. scope40_retrieval is in RETRIEVAL_TASKS, which is opt-in
# and excluded from the suite defaults -- it must be named explicitly or it
# silently will not run. rhla_enzyme_mutations is omitted: its sequences are
# 6-residue mutation-site strings, too short for MMseqs2 k-mers, so no baseline
# exists to compare against.
TASKS="${TASKS:-aav_flip antibiotic_resistance beta_lactamase_peer \
binary_subcellular_localization cloning_clf ec_classification \
enzyme_catalytic_efficiency fluorescence go_mf material_production \
metal_ion_binding optimal_ph peptide_hla profet_np_sp_cleaved remote_homology \
scope40_retrieval signalp_binary solubility stability subcellular_loc \
temperature_stability thermostability variant_effect}"

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
# Check each candidate separately: `ls a b` exits non-zero when EITHER is absent,
# so testing both in one ls rejects a perfectly good safetensors-only checkpoint.
# SentenceTransformerTrainer writes the finished model to <output_dir>/final, not
# to <output_dir> itself, so resolve that before deciding training failed. The
# first version of this guard checked only the parent and aborted the whole sweep
# on a run that had completed normally.
if ! compgen -G "$MODEL_NEW"/*.safetensors >/dev/null \
   && [[ ! -f "$MODEL_NEW/pytorch_model.bin" ]]; then
  if compgen -G "$MODEL_NEW/final"/*.safetensors >/dev/null; then
    MODEL_NEW="$MODEL_NEW/final"
    echo "note: using finished model at $MODEL_NEW"
  else
    echo "ERROR: no model weights under $MODEL_NEW or $MODEL_NEW/final" >&2
    exit 1
  fi
fi

# Training saves ESM2 checkpoints with FastPLM's custom-code identity
# (model_type "fast_esm", tokenizer_class "FastEsmTokenizer"), which
# SentenceTransformer cannot load: "Unrecognized processing class". Rewrite the
# metadata to the plain-ESM form the published checkpoint uses. Weights untouched,
# idempotent, originals kept as *.fastplm.
uv run --no-sync python make_checkpoint_loadable.py "$MODEL_NEW" || {
  echo "ERROR: could not normalise $MODEL_NEW for loading" >&2; exit 1; }

HAVE_TROUGH=0
if compgen -G "$MODEL_TROUGH"/*.safetensors >/dev/null; then
  uv run --no-sync python make_checkpoint_loadable.py "$MODEL_TROUGH" && HAVE_TROUGH=1
else
  echo "note: no snapshot at $MODEL_TROUGH -- skipping the near-trough arm"
fi

# Three arms so the +/- decontamination effect is directly visible:
#   esm2_35m     the untuned starting point
#   protsent_old the published paper model (trained on the UNfiltered corpus)
#   protsent_v3  retrained on the 40%/80%-decontaminated corpus
for probe in knn linear; do
  run_one "$MODEL_NEW"  protsent_v3  "$probe"
  run_one "$MODEL_OLD"  protsent_old "$probe"
  run_one "$MODEL_BASE" esm2_35m     "$probe"
  [[ $HAVE_TROUGH -eq 1 ]] && run_one "$MODEL_TROUGH" protsent_v3_ckpt4000 "$probe"
done

echo
echo "=== done $(date) -- results under $OUT ==="
grep -rh "Recall@10\|AUC" "$OUT" 2>/dev/null | head -20 || true
