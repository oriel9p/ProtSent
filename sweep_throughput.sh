#!/usr/bin/env bash
# 15-step throughput probes. Usage: MODEL=... GPU=3 OUT=models/late_interaction/sweep_150m ./sweep_throughput.sh
# Each point is skipped if its runtime.json already exists, so the script is re-runnable.
set -uo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-GrimSqueaker/ProtSent-V2-150M}"
GPU="${GPU:-3}"
OUT="${OUT:-models/late_interaction/sweep_150m}"
DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
STEPS="${STEPS:-15}"
PROJ="${PROJ:-64}"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}" TOKENIZERS_PARALLELISM=false
FILES=("$DATA/pfam_sorted.parquet" "$DATA/afdb_sorted.parquet" "$DATA/stringdb_train_15M.parquet")

point() {  # point <tag> <extra args...>
  local tag="$1"; shift
  local dir="$OUT/$tag"
  if [[ -f "$dir/runtime.json" ]]; then echo "skip $tag (done)"; return; fi
  echo "=== $tag"
  CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python train_late_interaction.py \
    --model "$MODEL" --files "${FILES[@]}" --output_dir "$dir" --proj_dim "$PROJ" \
    --max_steps "$STEPS" --save_steps 0 --max_minutes 25 --skip_step0_export \
    --dataloader_num_workers 4 --warmup_steps 1 "$@" >> "logs/sweep_$(basename "$OUT").log" 2>&1 \
    || { echo "$tag FAILED (OOM?)"; return; }
  python3 -c "
import json; d=json.load(open('$dir/runtime.json'))
print(f\"$tag pairs/s {d['pairs_per_s']} steps/s {d['steps_per_s']} peakGB {d['peak_vram_bytes']/1e9:.1f}\")"
}

mkdir -p logs "$OUT"
point bs128_mini64   --batch_size 128 --mini_batch_size 64 --score_mini_batch_size 32
point bs256_mini64   --batch_size 256 --mini_batch_size 64 --score_mini_batch_size 32
point bs256_tok16k   --batch_size 256 --mini_batch_num_tokens 16384 --score_mini_batch_size 32
point bs128_tok8k    --batch_size 128 --mini_batch_num_tokens 8192 --score_mini_batch_size 32
point bs128_tok12k   --batch_size 128 --mini_batch_num_tokens 12288 --score_mini_batch_size 16
