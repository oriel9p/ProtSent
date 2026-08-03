#!/usr/bin/env bash
# ProtSent-V2.5-35M — a continuation pass on top of ProtSent-V2-35M.
#
# Starts from models/protsent_esm2_35m_v3/final (paper name: ProtSent-V2-35M),
# copied to models/protsent_v2p5_init with FastPLM's config.json restored from
# the config.json.fastplm that make_checkpoint_loadable.py left behind. The
# vanilla-ESM rewrite is what lets SentenceTransformer(path) load the model for
# benchmarking; training wants the FastPLM identity back so it goes down the
# same code path the original run used. Verified equivalent: embeddings from
# both forms match to 0.0 max absolute difference.
#
# Three deltas from the V2 run, all requested:
#   * DMS CoSENT as an auxiliary target (--dms_file).
#   * Global Orthogonal Regularization on the contrastive losses
#     (--gor_weight), against the anisotropy that made whitening help the
#     linear probe so much.
#   * A different draw of the data: k drops 8 -> 3, which changes the
#     Dataset.from_generator fingerprint and so re-samples every cluster, and
#     the shuffle//global seeds move off 40/41.
#
# Sized for ONE B300 in under 16 h. Measured end to end on a free B300:
#
#   | config                                  | samples/s | peak GiB |
#   |-----------------------------------------|-----------|----------|
#   | bs 512, mini 128, GOR + DMS + Matryoshka |     145.5 |        - |
#   | bs 512, mini 256, GOR + DMS + Matryoshka |     146.8 |    121.6 |
#   | bs 1024, mini 256, same                  |     234.0 |    109.7 |
#
# Two things had to be fixed before any of that was true, both recorded here so
# the numbers above are not mistaken for something the old code could reach:
#
#   * --max_seq_length was never applied. Every model branch passes it to
#     models.Transformer, but the FastPLM tokenizer that then replaces it
#     declares model_max_length = 1e24, and sentence-transformers reads
#     truncation off the tokenizer. Batches were padded to the longest sequence
#     present -- 1,561 tokens on a 512-pair batch was measured -- which is what
#     made the GOR step cost 150 GiB. load_model_for_training now enforces the
#     limit after assembly. NOTE: V2 and V2-150M trained under the same bug, so
#     they saw untruncated sequences despite the flag.
#   * GOR sliced batches by first dimension, which silently matches nothing
#     under DataCollatorWithFlattening and embedded the whole batch with grad.
#     It now slices with sentence-transformers' own minibatch helpers.
#
# CoSENT has no gradient cache and is capped at MINI_BATCH rows per batch
# (SubsampledLoss); sentence-transformers uses one per_device_train_batch_size
# for every dataset in a multi-task dict, so that cap is what lets the
# contrastive batch stay at V2's 1024 while a DMS target rides along.
#
# A full epoch is not on the table at this budget: V2 was 34.8M pairs over
# 7 GPUs for 10 h 53 m, i.e. ~76 GPU-hours. This pass is ~15M rows, roughly 43%
# of an epoch, sized from the step time measured in the real run: 2.62 s/it at
# bs 1024, so ~14,700 steps is ~10.7 h and the 900-minute stop is slack, not the
# plan. Note V2's own step time is not the reference here — it was 8.08 s/it for
# the same 1024 rows, because it was padding to 1,561 tokens.
set -euo pipefail
cd ~/ProtSent

DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
MODEL="${MODEL:-models/protsent_v2p5_init}"
RUN_NAME="${RUN_NAME:-protsent_esm2_35m_v2p5}"

# NOT decontaminated: this file predates protsent-data-dc40 and was never
# filtered against the benchmark test sequences. Several suite tasks are
# themselves DMS-derived (Fluorescence, Stability, beta-lactamase, Variant
# Effect), so treat V2.5's numbers on those four as contaminated until checked.
DMS_FILE="${DMS_FILE:-/storage/users/ddofer/data/dms_cosent.parquet}"
# Kept auxiliary. The parquet is already interleaved across assays (179k length
# changes in the first 200k rows), so a prefix cap is an unbiased sample.
# 1.0M against ~14M contrastive pairs is ~7% of steps under the proportional
# sampler; the full 2.17M would be ~13%, which starts to stop being auxiliary.
DMS_MAX_ROWS="${DMS_MAX_ROWS:-1000000}"

BATCH_SIZE="${BATCH_SIZE:-1024}"       # V2's contrastive batch, kept
MINI_BATCH="${MINI_BATCH:-256}"        # CachedMNRL chunk; also the CoSENT cap
PRIMARY_LOSS="${PRIMARY_LOSS:-cached_mnrl}"
GOR_WEIGHT="${GOR_WEIGHT:-0.1}"
# k=5 -> C(5,2)=10 pairs per cluster, against V2's k=8 -> 28.
MAX_PAIRS_PER_CLUSTER="${MAX_PAIRS_PER_CLUSTER:-5}"
STRING_FILE="${STRING_FILE:-stringdb_train_15M.parquet}"
# Split evenly across the 3 --files, so this is a 7.0M cap per corpus.
#
# It was intended to bind only on STRING, which is a flat pair table with no
# clusters and so is not bounded by --max_pairs_per_cluster at all. It does not:
# AFDB yields more than 7.0M at k=5 and is capped too. That matters because the
# two corpora are capped by different mechanisms — STRING is *sampled* under the
# cap ("row-group sample, seed=13"), while a clustered corpus is truncated at the
# first N pairs of a group-sorted file, so the tail clusters are never visited.
# Measured coverage for the run of 2026-08-03 is recorded in RUNS.md; raise this
# and lower MAX_PAIRS_PER_CLUSTER together if you want AFDB whole.
MAX_MAP_ROWS="${MAX_MAP_ROWS:-21000000}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
# 5e-5, a quarter of V2's 2e-4, and a single half-cosine to a floor rather than
# V2's 3 cycles. V2 ended at peak LR, which RUNS.md had to control for with a
# near-trough checkpoint; this one ends where it should.
LR="${LR:-5e-5}"
LR_CYCLES="${LR_CYCLES:-0.5}"
WARMUP="${WARMUP:-200}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
WORKERS="${WORKERS:-8}"
# Different draw of the data than V2 (40/41).
SHUFFLE_SEED="${SHUFFLE_SEED:-13}"
SEED="${SEED:-7}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_MINUTES="${MAX_MINUTES:-900}"      # hard 15 h stop, safety net only

export HF_HOME=/storage/models/hf_home
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ddofer/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-flash_attention_2}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=32
NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

MATRYOSHKA="${MATRYOSHKA:-1}"
EXTRA_ARGS=()
[[ "$MAX_STEPS" -gt 0 ]] && EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
if [[ "$MATRYOSHKA" == "1" ]]; then
  EXTRA_ARGS+=(--matryoshka --matryoshka_dims 64 128 256)
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

echo "$(date) ProtSent v2.5: model=$MODEL bs=$BATCH_SIZE k=$MAX_PAIRS_PER_CLUSTER lr=$LR" \
     "gor=$GOR_WEIGHT dms_rows=$DMS_MAX_ROWS gpus=$CUDA_VISIBLE_DEVICES"

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
  --no_gather_across_devices \
  "${EXTRA_ARGS[@]}" \
  --no_resume \
  --run_name "$RUN_NAME"
