#!/usr/bin/env bash
# ProtSent-V2-150M — Synthyra FastPLM ESM2-150M on the DECONTAMINATED corpus.
#
# Sibling of train_esm2_35m.sh. Every setting below was measured on this hardware
# during the 35M run rather than assumed; see that script for the evidence behind
# each one. Only the deltas forced by model size are re-derived here.
#
# WHAT THIS RUN IS
#   The 150M counterpart of ProtSent-V2-35M. Same decontaminated data, same
#   objective, same configuration knobs. It is NOT a replacement for the 35M
#   model and it does not touch it: separate RUN_NAME, separate output directory,
#   separate results directory. Nothing from the 35M run is overwritten.
#
# DATA — filtered, and checked at launch
#   /storage/users/ddofer/data/protsent-data-dc40 is the 40% identity / 80%
#   coverage decontaminated corpus (pfam 27,929,772 rows; afdb 126,301,607;
#   stringdb_train_15M 15,000,000 pairs = 169,231,379 total). verify_training_corpus.py
#   confirmed zero flagged sequences survive in all three files. The guard below
#   refuses to start if any of the three filtered parquets is missing, so this
#   cannot silently fall back to the unfiltered data.
#
# DELTAS FROM THE 35M RUN
#   * MODEL: Synthyra/ESM2-150M (fast_esm, hidden 640, 30 layers) vs 480/12.
#     ~3.3x the activation memory per sequence, which is what MINI_BATCH has to
#     absorb.
#   * MINI_BATCH=64, measured. Under CachedMNRL peak memory is set by the
#     mini-batch, so this is the knob, and at 150M / seq 512 the frontier sits
#     between 64 and 128. Probed on 6x B300 (267 GiB each), 6 steps each:
#         mini 512, bs 1024            OOM at 262 GiB
#         mini 256, bs 1024            OOM at 262 GiB
#         mini 128, bs 1024            OOM at 262 GiB
#         mini 128, bs  512            OOM at 262 GiB  <- not a batch-size problem
#         mini  64, bs 1024            OK, 245 GiB peak, 34.0 s/it
#         mini 256, bs 1024 + ckpt     OK, 118 GiB peak, 36.6 s/it
#     So mini-batch 64 without gradient checkpointing is both the fastest option
#     and the one that keeps the paper's 1024 in-batch negatives. Gradient
#     checkpointing is NOT used: it is slower here and CachedMNRL already bounds
#     memory by the mini-batch. Headroom at mini 64 is ~22 GiB, which is thin --
#     if a long-sequence batch OOMs mid-run, resume with RESUME=1 MINI_BATCH=48.
#   * Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. It breaks the
#     dataloader workers here with "pidfd_getfd: Operation not permitted".
#   * BATCH_SIZE stays 1024, measured rather than assumed. Halving it to 512
#     raises throughput only ~8% (195.7 vs 180.7 sequences/s across 6 GPUs:
#     15.7 s/it over 11,318 steps vs 34.0 s/it over 5,659) because the encoder
#     forward/backward is O(B) and only the similarity matrix is O(B^2). That
#     buys ~4 hours off a ~53 hour run at the cost of halving the in-batch
#     negatives, which is the paper's headline contrastive setting. Not worth it.
#
# EXPECTED WALL CLOCK: 5,659 steps at ~34 s/it = ~53 hours on 6 GPUs.
# To cut it, lower MAX_PAIRS_PER_CLUSTER (8 -> 4 roughly quarters the AFDB and
# Pfam pair counts) or set MAX_STEPS. Do not cut BATCH_SIZE; see above.
#   * 6 GPUs, so the corpus divides into more steps than the 7-GPU 35M run.
#
# UNCHANGED, AND WHY
#   * flash_attention_2. FA3 remains impossible on these B300s (sm_103): the
#     pinned kernels-community/flash-attn3 build ships sm_80/sm_90a only. FA2
#     measured 10.48 s/it vs sdpa 16.79 s/it at 35M -- a 60% win from the
#     variable-length path, which matters more at 30 layers, not less.
#   * torch.compile off. Measured 8.87 vs 8.89 s/it at 35M, i.e. nothing, and
#     these are variable-length inputs which compile handles poorly.
#   * cached_mnrl. Plain MNRL OOMs a 267 GiB B300 even at bs256 for the 35M
#     model; at 150M it is hopeless.
#   * gather_across_devices OFF. Each rank already carries the paper's 1024
#     in-batch negatives, so the allgather buys combinations at the cost of
#     communication on every step.
#   * proportional sampling, no synthetic hard negatives -- the configuration the
#     paper's own ablations favour (20/23 tasks at +7.9% without hard negatives
#     vs 16/23 at +6.7% with).
#   * matryoshka dims 64/128/256. Kept identical to the 35M run so the two models
#     are comparable; 512 would now be legal under the 640 native dim but would
#     make the two runs differ in a second place.
set -euo pipefail
cd ~/ProtSent

DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
MODEL="${MODEL:-Synthyra/ESM2-150M}"
RUN_NAME="${RUN_NAME:-protsent_esm2_150m_v2}"

BATCH_SIZE="${BATCH_SIZE:-1024}"       # per-device contrastive batch
MINI_BATCH="${MINI_BATCH:-512}"        # CachedMNRL chunk; sets peak memory
PRIMARY_LOSS="${PRIMARY_LOSS:-cached_mnrl}"
MAX_PAIRS_PER_CLUSTER="${MAX_PAIRS_PER_CLUSTER:-8}"
STRING_FILE="${STRING_FILE:-stringdb_train_15M.parquet}"
MAX_MAP_ROWS="${MAX_MAP_ROWS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
# 5% of the 5,659-step epoch. The 35M run used a flat 1000, which was 20.6% of
# its schedule -- far more warmup than this model needs, and it delays the point
# at which the run is doing useful work.
WARMUP_STEPS="${WARMUP_STEPS:-300}"
WORKERS="${WORKERS:-8}"
CKPT="${CKPT:-}"
COMPILE="${COMPILE:---no-compile}"
MATRYOSHKA="${MATRYOSHKA:---matryoshka}"
NO_GATHER_ACROSS_DEVICES="${NO_GATHER_ACROSS_DEVICES:-1}"
# A 150M step is several times a 35M step, so 500 steps is well over an hour.
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

export HF_HOME=/storage/models/hf_home
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ddofer/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
# The 150M weights are already in HF_HOME (snapshot fetched before launch), so
# offline is safe and keeps a transient Hub outage from killing a multi-day run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-flash_attention_2}"
NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8

OUTPUT_ROOT="${OUTPUT_ROOT:-models}"
OUT="$OUTPUT_ROOT/$RUN_NAME"
# --no_resume by default so a stale checkpoint from an aborted attempt is never
# picked up silently. Set RESUME=1 to continue an interrupted run on purpose.
RESUME_FLAG="${RESUME:+}"
[[ "${RESUME:-0}" == "1" ]] || RESUME_FLAG="--no_resume"
# Never clobber a finished model. Resuming is a deliberate act, not a default.
if [[ -e "$OUT/final" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "REFUSING: $OUT/final already exists. Set ALLOW_OVERWRITE=1 or change RUN_NAME." >&2
  exit 1
fi

EXTRA_ARGS=()
[[ "$MAX_STEPS" -gt 0 ]] && EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
[[ "$NO_GATHER_ACROSS_DEVICES" == "1" ]] && EXTRA_ARGS+=(--no_gather_across_devices)
[[ -n "$MATRYOSHKA" ]] && EXTRA_ARGS+=(--matryoshka --matryoshka_dims 64 128 256)

for f in pfam_sorted.parquet afdb_sorted.parquet "$STRING_FILE"; do
  [[ -f "$DATA/$f" ]] || { echo "MISSING: $DATA/$f — decontamination not finished" >&2; exit 1; }
done
echo "$(date) ProtSent-V2-150M: model=$MODEL data=$DATA out=$OUT"
echo "  bs=$BATCH_SIZE mini=$MINI_BATCH gpus=$NUM_PROCESSES ($CUDA_VISIBLE_DEVICES)" \
     "k=$MAX_PAIRS_PER_CLUSTER lr=$LR backend=$PROTSENT_ESMPLUSPLUS_ATTN_BACKEND" \
     "max_steps=${MAX_STEPS:-auto}"

uv run --no-sync accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  protein_pipeline.py train \
  --model "$MODEL" \
  --files "$DATA/pfam_sorted.parquet" \
          "$DATA/afdb_sorted.parquet" \
          "$DATA/$STRING_FILE" \
  --loss_mode multi --multi_primary_loss "$PRIMARY_LOSS" \
  `# pinned: the defaults changed after this run finished. --batch_sampler auto now
   # resolves to NO_DUPLICATES for multi-task runs and --mnrl_directions defaults to
   # symmetric, so without these two the script no longer reproduces the run RUNS.md
   # attributes to it.` \
  --batch_sampler none --mnrl_directions query_to_doc \
  --mnrl_mini_batch_size "$MINI_BATCH" \
  --multi_dataset_sampler proportional \
  --gor_weight 0.0 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" --learning_rate "$LR" --warmup_steps "$WARMUP_STEPS" \
  --max_pairs_per_cluster "$MAX_PAIRS_PER_CLUSTER" \
  --max_map_rows "$MAX_MAP_ROWS" \
  --dataloader_num_workers "$WORKERS" \
  --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  $CKPT $COMPILE \
  "${EXTRA_ARGS[@]}" \
  $RESUME_FLAG
