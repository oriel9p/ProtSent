#!/usr/bin/env bash
# ProtSent v2 stage run — ESM-C 600M, 4x B300 (GPUs 4-7).
# Option A: contrastive-only (pfam+afdb+stringdb, MNRL+Matryoshka).
# No DMS in joint interleave (avoids CoSENT/gather deadlock under DDP).
# Defaults are set from measured B300 probes: remove GIST guide overhead,
# use cached contrastive loss for large global batches, and keep the data
# path on standard HF/SentenceTransformers samplers.
set -euo pipefail
cd ~/ProtSent

# ---- filled in after the batch-size probe ----
BATCH_SIZE="${BATCH_SIZE:-128}"
CKPT="${CKPT:---gradient_checkpointing}"                 # "" or "--gradient_checkpointing"
MAX_MAP_ROWS="${MAX_MAP_ROWS:-0}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
MODEL="${MODEL:-Synthyra/ESMplusplus_large}"
RUN_NAME="${RUN_NAME:-protsent_esmc600m_v2}"
MNRL_MINI_BATCH_SIZE="${MNRL_MINI_BATCH_SIZE:-64}"
USE_LENGTH_BUCKETS="${USE_LENGTH_BUCKETS:-0}"
LENGTH_BUCKET_SIZE="${LENGTH_BUCKET_SIZE:-64}"
NO_GATHER_ACROSS_DEVICES="${NO_GATHER_ACROSS_DEVICES:-0}"
# ----------------------------------------------

export HF_HOME=/storage/models/hf_home
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-sdpa}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
PRIMARY_LOSS="${PRIMARY_LOSS:-mnrl}"

EXTRA_ARGS=()
if [[ "$MAX_STEPS" -gt 0 ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ "$USE_LENGTH_BUCKETS" == "1" ]]; then
  EXTRA_ARGS+=(--length_bucketed_batches --length_bucket_size "$LENGTH_BUCKET_SIZE")
fi
if [[ "$NO_GATHER_ACROSS_DEVICES" == "1" ]]; then
  EXTRA_ARGS+=(--no_gather_across_devices)
fi

echo "$(date) launching v2: loss=$PRIMARY_LOSS BS=$BATCH_SIZE ckpt='${CKPT:-off}' max_rows=$MAX_MAP_ROWS max_seq=$MAX_SEQ_LENGTH backend=$PROTSENT_ESMPLUSPLUS_ATTN_BACKEND"

uv run --no-sync accelerate launch --num_processes 4 --mixed_precision bf16 \
  protein_pipeline.py train \
  --model "$MODEL" \
  --files /storage/users/ddofer/data/pfam_sorted.parquet \
          /storage/users/ddofer/data/afdb_sorted.parquet \
          /storage/users/ddofer/data/stringdb_train.parquet \
  --loss_mode multi --multi_primary_loss "$PRIMARY_LOSS" \
  --gor_weight 0.0 \
  --matryoshka --matryoshka_dims 64 128 512 \
  --max_seq_length "$MAX_SEQ_LENGTH" --max_pairs_per_cluster 500 --no-compile \
  $CKPT \
  --batch_size "$BATCH_SIZE" \
  --mnrl_mini_batch_size "$MNRL_MINI_BATCH_SIZE" \
        --max_map_rows "$MAX_MAP_ROWS" \
        --epochs "$EPOCHS" --learning_rate 8e-5 --warmup_steps 1000 \
  "${EXTRA_ARGS[@]}" \
  --dataloader_num_workers 0 \
      --no_resume \
  --run_name "$RUN_NAME"
