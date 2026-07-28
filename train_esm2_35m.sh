#!/usr/bin/env bash
# ProtSent v3 — ESM2-35M (Synthyra FastPLM) on decontaminated pretraining data.
#
# Data: /storage/users/ddofer/data/protsent-data-dc40 — every corpus filtered at
# 40% identity / 80% coverage of the benchmark test sequence (see the dataset
# card in that folder for per-corpus method and measured recall).
#
# Differences from the ESM-C v2 run, all measured rather than assumed:
#   * No gradient checkpointing. At 35M it is not needed, and it is what forced
#     dataloader_num_workers=0 (protein_pipeline.py:2242-2247), which cost ~270ms
#     per step in NFS Arrow fetch + single-threaded tokenization.
#   * flash_attention_2. FA3 is impossible here: the pinned
#     kernels-community/flash-attn3 build ships sm_80/sm_90a only and dies on
#     these B300s (sm_103) with "no kernel image is available for execution on
#     the device" -- FA3 is Hopper-only, Blackwell wants FA4, which FastPLM's
#     AttentionBackend enum does not have. FA2 does load (pinned snapshot
#     db6b517, cached) and is a large win on this workload: measured in the real
#     training loop at bs1024/mini_batch256, FA2 = 10.48 s/it vs sdpa = 16.79
#     s/it, i.e. sdpa is 60% slower. Note a synthetic fixed-length fwd/bwd probe
#     shows the two as equivalent -- that benchmark is misleading, because it has
#     no padding variance and so never exercises FA2's variable-length path.
#   * cached_mnrl, not mnrl. Plain MNRL embeds the whole batch at once and
#     backprops through it, so memory scales with batch: under accelerate's bf16
#     autocast (weights fp32, so MLP activations are fp32 too) even bs256 OOMs a
#     267 GiB B300 at ~260 GiB, inside feed_forward_chunk's gelu. CachedMNRL
#     gradient-caches: it embeds in MINI_BATCH chunks, so peak memory is set by
#     MINI_BATCH while the contrastive batch -- the thing that actually matters
#     for a contrastive objective -- stays large. This is what lets BATCH_SIZE go
#     up rather than down. Note --mnrl_mini_batch_size IS live here
#     (protein_pipeline.py:2751-2756); it is only inert for plain mnrl.
#   * matryoshka_dims below the 480 native dim (512 would be silently dropped).
#   * --max_pairs_per_cluster is the real data-budget knob now that
#     --max_map_rows 0 genuinely means "no limit" (protein_pipeline.py:2665).
set -euo pipefail
cd ~/ProtSent

DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
MODEL="${MODEL:-/storage/models/ESM2-35M}"
RUN_NAME="${RUN_NAME:-protsent_esm2_35m_v3}"

BATCH_SIZE="${BATCH_SIZE:-1024}"       # per-device (protein_pipeline.py:2287)
MINI_BATCH="${MINI_BATCH:-256}"        # CachedMNRL chunk; sets peak memory
PRIMARY_LOSS="${PRIMARY_LOSS:-cached_mnrl}"
# k sequences sampled per cluster -> C(k,2) pairs per cluster, NOT k pairs
# (protein_pipeline.py:1846-1848). k=8 gives 28 pairs/cluster: AFDB 817,282
# clusters -> ~22.9M pairs, Pfam 29,368 -> ~0.8M. Every cluster is still visited,
# unlike the old round-robin run which reached only ~2% of them.
MAX_PAIRS_PER_CLUSTER="${MAX_PAIRS_PER_CLUSTER:-8}"
# STRING is a flat pair table with no clusters, so --max_pairs_per_cluster does
# not bound it and --max_map_rows splits uniformly across files (would starve
# AFDB too). Subsampled once to 15M pairs instead, seed 42.
STRING_FILE="${STRING_FILE:-stringdb_train_15M.parquet}"
MAX_MAP_ROWS="${MAX_MAP_ROWS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
WORKERS="${WORKERS:-8}"
CKPT="${CKPT:-}"                       # "" or "--gradient_checkpointing"
COMPILE="${COMPILE:---no-compile}"     # "" enables torch.compile
MATRYOSHKA="${MATRYOSHKA:---matryoshka}"
# Gather OFF. Under CachedMNRL peak memory is set by MINI_BATCH, not BATCH_SIZE,
# so per-device batch is simply raised to the paper's 1024 in-batch negatives on
# EVERY rank, without paying the cross-device allgather. Same contrastive signal
# as the paper, no communication overhead.
NO_GATHER_ACROSS_DEVICES="${NO_GATHER_ACROSS_DEVICES:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"          # ~1h at the measured step time
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

export HF_HOME=/storage/models/hf_home
# /storage is NFS4 (50.50.50.1:/flex_vol). Dataset.from_generator() writes the
# generated pair corpus as Arrow and the dataloader then random-reads it every
# step -- over NFS that starved the GPUs to 0-3% utilisation. /dev/md0 is local
# ext4 with ~1.3 TB free, which is ample for the ~100 GB pair cache. Model
# weights stay on HF_HOME; only the dataset cache moves.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ddofer/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-flash_attention_2}"
NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
export TOKENIZERS_PARALLELISM=false
# NVLink SHARP multicast cannot be allocated while another job holds the
# NVSwitch resources ("Failed to bind NVLink SHARP (NVLS) Multicast memory ...
# CUDA error 401"). Falls back to ring/tree allreduce; at 35M params the
# gradient allreduce is tiny, so the cost is negligible.
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8

EXTRA_ARGS=()
[[ "$MAX_STEPS" -gt 0 ]] && EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
[[ "$NO_GATHER_ACROSS_DEVICES" == "1" ]] && EXTRA_ARGS+=(--no_gather_across_devices)
[[ -n "$MATRYOSHKA" ]] && EXTRA_ARGS+=(--matryoshka --matryoshka_dims 64 128 256)

for f in pfam_sorted.parquet afdb_sorted.parquet "$STRING_FILE"; do
  [[ -f "$DATA/$f" ]] || { echo "MISSING: $DATA/$f — decontamination not finished" >&2; exit 1; }
done

echo "$(date) ProtSent v3: model=$MODEL bs=$BATCH_SIZE k=$MAX_PAIRS_PER_CLUSTER lr=$LR" \
     "ckpt='${CKPT:-off}' compile='${COMPILE:-on}' backend=$PROTSENT_ESMPLUSPLUS_ATTN_BACKEND"

uv run --no-sync accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  protein_pipeline.py train \
  --model "$MODEL" \
  --files "$DATA/pfam_sorted.parquet" \
          "$DATA/afdb_sorted.parquet" \
          "$DATA/$STRING_FILE" \
  --loss_mode multi --multi_primary_loss "$PRIMARY_LOSS" \
  --mnrl_mini_batch_size "$MINI_BATCH" \
  --multi_dataset_sampler proportional \
  --gor_weight 0.0 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_pairs_per_cluster "$MAX_PAIRS_PER_CLUSTER" \
  --batch_size "$BATCH_SIZE" \
  --max_map_rows "$MAX_MAP_ROWS" \
  --epochs "$EPOCHS" --learning_rate "$LR" --warmup_steps 1000 \
  --dataloader_num_workers "$WORKERS" \
  --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  $CKPT $COMPILE \
  "${EXTRA_ARGS[@]}" \
  --no_resume \
  --run_name "$RUN_NAME"
