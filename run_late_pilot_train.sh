#!/usr/bin/env bash
# The two late-interaction training arms. Defaults follow the pilot plan:
# <=2h45m wall each, ~2000 steps target, round-robin over the three dc40 sources.
# Batch size: run the throughput sweep first (SWEEP=1) and pick best pairs/sec.
#
#   GPUS=2,3 ARM=protsent ./run_late_pilot_train.sh
#   GPUS=2,3 ARM=esm2     ./run_late_pilot_train.sh
#   SWEEP=1 GPUS=2 ARM=protsent ./run_late_pilot_train.sh   # 15-step sweep @ bs 16/32/64
set -euo pipefail
cd "$(dirname "$0")"

ARM="${ARM:-protsent}"           # protsent | esm2
GPUS="${GPUS:-2,3}"
DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-32}"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
export TOKENIZERS_PARALLELISM=false

case "$ARM" in
  protsent) MODEL="GrimSqueaker/ProtSent-V2-35M" ;;
  esm2)     MODEL="Synthyra/ESM2-35M" ;;
  *) echo "ARM must be protsent|esm2" >&2; exit 1 ;;
esac

NPROC=$(( $(tr -cd , <<<"$GPUS" | wc -c) + 1 ))
# NOTE: train_late_interaction.py caps each cluster file at --max_pairs_per_file
# (default 2M) built from the *prefix* of the group-sorted corpus (~500k distinct
# AFDB clusters at k=8). A <=2.5k-step pilot sees <<2M pairs per source, so this
# only trades corpus randomness for build time; drop the cap for longer runs.
FILES=("$DATA/pfam_sorted.parquet" "$DATA/afdb_sorted.parquet" "$DATA/stringdb_train_15M.parquet")

run() {  # run <steps> <batch> <outdir> [extra args...]
  local steps="$1" batch="$2" outdir="$3"; shift 3
  CUDA_VISIBLE_DEVICES="$GPUS" uv run --no-sync accelerate launch \
    --num_processes "$NPROC" --mixed_precision bf16 --main_process_port 0 \
    train_late_interaction.py --model "$MODEL" --files "${FILES[@]}" \
    --output_dir "$outdir" --max_steps "$steps" --batch_size "$batch" \
    --run_name "${ARM}_late" "$@"
}

if [[ "${SWEEP:-0}" == "1" ]]; then
  for bs in 16 32 64; do
    echo "=== sweep bs=$bs"
    run 15 "$bs" "models/late_interaction/sweep_${ARM}_bs${bs}" \
      --save_steps 0 --max_minutes 20 --skip_step0_export || echo "bs=$bs failed (OOM?)"
    grep -hE '"pairs_per_s"|"peak_vram_bytes"' "models/late_interaction/sweep_${ARM}_bs${bs}/runtime.json" || true
  done
  exit 0
fi

run "$STEPS" "$BATCH" "models/late_interaction/${ARM}_late" --max_minutes 165
