#!/usr/bin/env bash
# Batch-size probe for ProtSent v2 on free GPUs 4-7 (B300 275GB).
# Option A: contrastive-only (pfam only for probe speed), GIST+8M guide, NO DMS.
# Tries a ladder without grad-checkpointing, then with it, reporting FIT / OOM / ERROR.
set -uo pipefail
cd ~/ProtSent

export HF_HOME=/storage/models/hf_home
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=sdpa
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# 35M guide is corrupt (149GB declared, 406MB present); use cached 8M instead
GUIDE=$(ls -d /storage/models/hf_home/hub/models--facebook--esm2_t6_8M_UR50D/snapshots/c731040fcd8d73dceaa04b0a8e6329b345b0f5df)
echo "GUIDE=$GUIDE"
LOGDIR=/tmp/claude-2003/-home-ddofer-ProtSent/08526552-9c53-402f-b1d1-75ad89ca306d/scratchpad/probe_logs
mkdir -p "$LOGDIR"

run_probe () {
  local CKPT_FLAG="$1"; local BS="$2"; local TAG="$3"
  local LOG="$LOGDIR/probe_${TAG}.log"
  echo "=== PROBE $TAG : per-device batch=$BS  ckpt='${CKPT_FLAG:-off}' ==="
  timeout 900 uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
    protein_pipeline.py train \
    --model Synthyra/ESMplusplus_large \
    --files /storage/users/ddofer/data/pfam_sorted.parquet \
    --loss_mode multi --multi_primary_loss gist \
    --gist_guide_model "$GUIDE" \
    --max_seq_length 1024 --no-compile --dataloader_num_workers 0 \
    $CKPT_FLAG \
    --batch_size "$BS" \
    --fast --epochs 1 --run_name "probe_${TAG}" > "$LOG" 2>&1
  local rc=$?
  local peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 2>/dev/null | paste -sd, -)
  if grep -qiE "out of memory|OutOfMemoryError|CUDA error: out of memory" "$LOG"; then
    echo "  -> OOM at BS=$BS ($TAG)"
  elif grep -qiE "gradient_checkpointing_enable|does not support gradient|checkpoint" "$LOG" && [ -n "$CKPT_FLAG" ] && [ $rc -ne 0 ]; then
    echo "  -> GRAD-CKPT UNSUPPORTED (see $LOG)"
  elif [ $rc -eq 0 ]; then
    echo "  -> FIT ✅ BS=$BS ($TAG)  [mem now: $peak MiB]"
  else
    echo "  -> ERROR rc=$rc ($TAG) — tail:"; tail -6 "$LOG" | sed 's/^/     /'
  fi
  sleep 5
}

echo "########## Phase 1: NO gradient checkpointing ##########"
run_probe ""  96  "nockpt_bs96"
run_probe ""  128 "nockpt_bs128"
run_probe ""  192 "nockpt_bs192"

echo "########## Phase 2: WITH gradient checkpointing ##########"
run_probe "--gradient_checkpointing" 256  "ckpt_bs256"
run_probe "--gradient_checkpointing" 512  "ckpt_bs512"
run_probe "--gradient_checkpointing" 768  "ckpt_bs768"
run_probe "--gradient_checkpointing" 1024 "ckpt_bs1024"

echo "########## PROBE SUMMARY ##########"
echo "(logs in $LOGDIR)"
