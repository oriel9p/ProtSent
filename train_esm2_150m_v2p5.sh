#!/usr/bin/env bash
# ProtSent-V2.5-150M — a continuation pass on top of ProtSent-V2-150M.
#
# Starts from models/protsent_150m_v2p5_init, which is
# models/protsent_esm2_150m_v2/final with FastPLM's config.json and
# tokenizer_config.json restored from the .fastplm backups that
# make_checkpoint_loadable.py left behind. Verified real: all 515 tensors, 147.7M
# parameters, max absolute difference 0.0 against the published V2-150M weights
# (huggingface.co/GrimSqueaker/ProtSent-V2-150M).
#
# Sized by probe_150m_v2p5.sh on 4 B300s, 10 steps per arm:
#   mini 256          -> OOM at 260 GiB of 267.   mini 128 fits.
#   len 768           -> OOM at mini 128.          512 stays.
#   Matryoshka [128,640] -> hung at step 6 of 10, twice, killed at 20 min. Dropped.
#   batch 2048        -> 1.11x the step time for 2x the data, and 2x the in-batch
#                        negatives. Taken.
#
# Deltas from the V2-150M run, beyond the 35M V2.5 recipe:
#   * Symmetric --mnrl_directions. V2-150M trained one-directional because ST
#     defaults to the asymmetric query/document case; both columns here are
#     proteins drawn the same way, and the reverse term needs no extra forward.
#     NOT NO_DUPLICATES: see the --batch_sampler note below.
#   * GOR at the paper's 1.0 rather than the 35M run's arbitrary 0.1, which moved
#     geometry but no task metric. mean_weight is left at 1.0. The 35M ablation
#     says GOR buys nothing at that scale, but 35M had no headroom left: whitening
#     already put its k-NN on top of its linear probe. At 150M whitening is still
#     worth +0.0555 and leaves +0.0154 on the table, so the channel is open here.
#   * DMS CoSENT auxiliary target, as at 35M.
#   * A different draw: k=8 (V2-150M used 5) and seeds off 40/41.
#
# gather_across_devices stays OFF and is not negotiable here. RUNS.md records a
# reproduced DDP deadlock between gather and a CoSENT target -- both ranks spin at
# ~91% CPU and no step ever completes -- and this run has a CoSENT target.
#
# Be precise about what that costs: with gather off each rank runs its own
# BATCH_SIZE-way contrastive task, so the negative pool stays at BATCH_SIZE and
# does NOT become 4 x BATCH_SIZE. What scales with rank count is the optimizer's
# effective batch (4 x BATCH_SIZE = 4096 here), because gradients are averaged.
# More ranks therefore buy steps-per-hour and gradient quality, not harder
# negatives. Raising BATCH_SIZE is the only lever on the negative pool.
set -euo pipefail
cd ~/ProtSent

DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
MODEL="${MODEL:-models/protsent_150m_v2p5_init}"
RUN_NAME="${RUN_NAME:-protsent_esm2_150m_v2p5}"

# NOT decontaminated -- predates protsent-data-dc40. Four suite tasks are
# DMS-derived, so treat V2.5-150M's Fluorescence / Stability / beta-lactamase /
# Variant Effect numbers as contaminated until checked at sequence identity.
DMS_FILE="${DMS_FILE:-/storage/users/ddofer/data/dms_cosent.parquet}"
DMS_MAX_ROWS="${DMS_MAX_ROWS:-1000000}"

BATCH_SIZE="${BATCH_SIZE:-2048}"       # per device; 4 ranks, gather off
MINI_BATCH="${MINI_BATCH:-128}"        # CachedMNRL chunk, and the CoSENT cap
PRIMARY_LOSS="${PRIMARY_LOSS:-cached_mnrl}"
GOR_WEIGHT="${GOR_WEIGHT:-1.0}"
GOR_MAX_SAMPLES="${GOR_MAX_SAMPLES:-128}"
MAX_PAIRS_PER_CLUSTER="${MAX_PAIRS_PER_CLUSTER:-8}"
STRING_FILE="${STRING_FILE:-stringdb_train_15M.parquet}"
MAX_MAP_ROWS="${MAX_MAP_ROWS:-120000000}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
LR_CYCLES="${LR_CYCLES:-0.5}"
WARMUP="${WARMUP:-200}"
# 512 keeps 93.3% of AFDB residues and 77.2% of STRING; 768 takes STRING's
# truncation from 29.7% of sequences to 13.9%. Decided by probe_150m_v2p5.sh.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
WORKERS="${WORKERS:-8}"
SHUFFLE_SEED="${SHUFFLE_SEED:-17}"
SEED="${SEED:-11}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_MINUTES="${MAX_MINUTES:-930}"      # hard 15.5 h stop, safety net

export HF_HOME=/storage/models/hf_home
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ddofer/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-flash_attention_2}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=32
NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

MATRYOSHKA="${MATRYOSHKA:-0}"
MATRYOSHKA_DIMS="${MATRYOSHKA_DIMS:-128 640}"
RESUME="${RESUME:-0}"

EXTRA_ARGS=()
# none, not the new NO_DUPLICATES default: that sampler scans row-by-row building
# Python string sets, and afdb/pfam parquets are cluster-sorted, so consecutive rows
# share proteins and it rejects long runs. Measured: 0 steps in 23 min at 0% GPU.
EXTRA_ARGS+=(--batch_sampler "${BATCH_SAMPLER:-none}")
[[ "$RESUME" == "1" ]] || EXTRA_ARGS+=(--no_resume)
[[ "$MAX_STEPS" -gt 0 ]] && EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
EXTRA_ARGS+=(--no_gather_across_devices)   # see the header; not a tunable here
if [[ "$MATRYOSHKA" == "1" ]]; then
  # shellcheck disable=SC2086
  EXTRA_ARGS+=(--matryoshka --matryoshka_dims $MATRYOSHKA_DIMS)
else
  EXTRA_ARGS+=(--no-matryoshka)
fi

for f in pfam_sorted.parquet afdb_sorted.parquet "$STRING_FILE"; do
  [[ -f "$DATA/$f" ]] || { echo "MISSING: $DATA/$f" >&2; exit 1; }
done
[[ -f "$DMS_FILE" ]] || { echo "MISSING: $DMS_FILE" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "MISSING: $MODEL — see the header" >&2; exit 1; }
if [[ -e "models/$RUN_NAME/final" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "models/$RUN_NAME/final exists; set ALLOW_OVERWRITE=1 to replace it" >&2
  exit 1
fi

echo "$(date) ProtSent v2.5-150M: model=$MODEL bs=$BATCH_SIZE mini=$MINI_BATCH" \
     "k=$MAX_PAIRS_PER_CLUSTER lr=$LR gor=$GOR_WEIGHT len=$MAX_SEQ_LENGTH" \
     "matryoshka=$MATRYOSHKA gpus=$CUDA_VISIBLE_DEVICES"

uv run --no-sync accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  protein_pipeline.py train \
  --model "$MODEL" \
  --files "$DATA/pfam_sorted.parquet" \
          "$DATA/afdb_sorted.parquet" \
          "$DATA/$STRING_FILE" \
  --dms_file "$DMS_FILE" \
  --dms_max_rows "$DMS_MAX_ROWS" \
  --loss_mode multi --multi_primary_loss "$PRIMARY_LOSS" \
  --mnrl_mini_batch_size "$MINI_BATCH" \
  --multi_dataset_sampler proportional \
  --gor_weight "$GOR_WEIGHT" \
  --gor_max_samples "$GOR_MAX_SAMPLES" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_pairs_per_cluster "$MAX_PAIRS_PER_CLUSTER" \
  --batch_size "$BATCH_SIZE" \
  --max_map_rows "$MAX_MAP_ROWS" \
  --epochs "$EPOCHS" --learning_rate "$LR" --warmup_steps "$WARMUP" \
  --lr_scheduler_type cosine_with_min_lr --lr_num_cycles "$LR_CYCLES" \
  --max_minutes "$MAX_MINUTES" \
  --pair_dataset_shuffle_seed "$SHUFFLE_SEED" --seed "$SEED" \
  --dataloader_num_workers "$WORKERS" \
  --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --no-compile \
  "${EXTRA_ARGS[@]}" \
  --run_name "$RUN_NAME"
