#!/usr/bin/env bash
# ProtSent-V2 ESM-C 300M — the V2 recipe on Synthyra/ESMplusplus_small.
#
# WHAT THIS RUN IS
#   A third backbone for the V2 line (35M and 150M are ESM-2). One epoch of the
#   decontaminated V2 corpus plus the DMS/ProteinGym CoSENT target, from the
#   vanilla ESM-C 300M weights. Not a continuation of anything.
#
# DELTAS FROM train_esm2_150m.sh, and why
#   * k=10 with the disjoint pair sampler (now the default). At k=10 the budget
#     per cluster is C(min(n,10),2)=45 rather than 28, and the budget is spent on
#     disjoint pairs, so it covers up to 90 distinct sequences per cluster instead
#     of 10. Corpus as the builder actually emitted it: pfam 1,230,551 + afdb
#     27,105,179 + string 15,000,000 + DMS 1,000,000 = 44,335,730 pairs, so
#     21,646 global batches of 2048.
#
#     AFDB is well above the 18,016,192 that sum(C(min(n,10),2)) over its distinct
#     group_id values predicts, while pfam matches its prediction exactly. The
#     builder groups by CONSECUTIVE RUNS of group_id, not by distinct value, so a
#     file sorted by some other key fragments each cluster into several runs and
#     every fragment draws its own budget. Pairs stay inside one group_id either
#     way -- every member of a buffer shares grp -- so this over-weights large
#     clusters rather than producing wrong positives.
#   * DMS IS in the interleave, which multi-GPU only tolerates because of the
#     sampler fix. accelerate hands rank i batches i, i+N, ..., so ranks run
#     different datasets on the same step; a CoSENT batch and a CachedMNRL batch
#     then produce different DDP collective sequences -- CachedMNRL calls
#     .backward() inside its own forward, CoSENT does not -- the ranks
#     desynchronise, and NCCL's watchdog kills the job after 30 minutes. Caught
#     five times before the fix, always at step 3-20, and the flight recorder
#     names it under one sequence number:
#       Rank 0: SeqNum=14603, ALLREDUCE, NumelIn=9285184   <- DDP gradient bucket
#       Rank 1: SeqNum=14603, ALLGATHER, NumelIn=1, NumelOut=3
#       Rank 2: SeqNum=14603, ALLGATHER, NumelIn=1, NumelOut=3
#     Pfam, AFDB and STRING all use CachedMNRL, so a step only breaks when it
#     mixes DMS with the rest -- 6.5% of steps at 3 ranks, measured. That is why
#     it reached step 11-20 rather than failing at step 1, and why a short smoke
#     test does NOT clear it: budget 30+ steps before believing a fix.
#
#     The fix is _align_proportional_sampler_to_world_size in protein_pipeline.py,
#     applied automatically when the sampler is PROPORTIONAL and dms_cosent is
#     present. It emits batches in blocks of world_size from one dataset so every
#     rank's slice of a block is that same dataset. Verified by
#     tests/test_sampler_alignment.py: 0 mixed steps at 2, 3 and 4 ranks.
#
#     It is NOT contention, though contention was the obvious suspect and costs
#     about 40% of throughput separately: the five failures span load 107 to 722
#     and never moved off step 3-20. A slow rank changes when a collective is
#     issued, never which one, and here the ranks differ in collective TYPE at
#     the same sequence number.
#
#   * gather_across_devices OFF. Independent of the above -- it was on for the
#     first attempt and off for the next two, and all three died identically. Its
#     cost, precisely: each rank runs its own BATCH_SIZE-way contrastive task, so
#     the negative pool is BATCH_SIZE, not NUM_PROCESSES x BATCH_SIZE. Ranks buy
#     steps-per-hour and gradient quality; raising BATCH_SIZE is the only lever
#     on the negative pool.
#   * No GOR. The 35M ablation found it bought nothing; there is no reason to pay
#     for it on an untested backbone.
#   * Symmetric --mnrl_directions, the current default. The V2 ESM-2 runs pinned
#     query_to_doc because that was ST's default at the time; both columns here
#     are proteins drawn the same way, and the reverse term costs no extra forward.
#   * No Matryoshka, unlike the ESM-2 V2 runs. It is dropped on the 150M V2.5
#     precedent (hung at step 6 of 10, twice) rather than on evidence from this
#     backbone: the one run here that had it enabled also had gather on, and it
#     died of the gather fault above, so Matryoshka's own cost is unmeasured. If
#     it is ever wanted back, note that MatryoshkaLoss re-runs the whole cached
#     mini-batch loop once per dim from inside CachedMNRL's gradient-caching
#     backward (matryoshka.py:110 inside
#     cached_multiple_negatives_ranking.py:438), so price it before trusting it.
#
# ATTENTION BACKEND: flash_attention_2, not 3. FA3 is unavailable on these B300s
#   and it is not a packaging problem — flash-attn-3 3.0.0+cu130torch2.13 is
#   installed and its kernels are Hopper-only, so a forward pass dies with
#     CUDA error (.../hopper/flash_fwd_launch_template.h:192):
#     no kernel image is available for execution on the device
#   on sm_103. The newest prebuilt wheel is the same git rev. Worse, ESM++ accepts
#   "flash_attention_3" at assignment and only fails in the forward pass, so the
#   backend ladder in model_utils.py cannot fall back for you. Do not set it.
#
# DISK: both the datasets cache and the checkpoints go to /storage. The root
#   filesystem was at 100% (497 MB free) when this run was set up.
set -euo pipefail
cd ~/ProtSent

DATA="${DATA:-/storage/users/ddofer/data/protsent-data-dc40}"
MODEL="${MODEL:-Synthyra/ESMplusplus_small}"
RUN_NAME="${RUN_NAME:-protsent_esmc_300m_v2}"

# NOT decontaminated -- predates protsent-data-dc40. Four suite tasks are
# DMS-derived, so treat this model's Fluorescence / Stability / beta-lactamase /
# Variant Effect numbers as contaminated until checked at sequence identity.
DMS_FILE="${DMS_FILE:-/storage/users/ddofer/data/dms_cosent.parquet}"
DMS_MAX_ROWS="${DMS_MAX_ROWS:-1000000}"

BATCH_SIZE="${BATCH_SIZE:-2048}"       # per device; x NUM_PROCESSES with gather on
# CachedMNRL chunk, and the CoSENT cap. Peak memory rides on this, not on
# BATCH_SIZE. Measured at 128: rank 0 sits at 227.6 GiB of 275, the other ranks
# at ~123. If a long-sequence batch OOMs mid-run, resume with MINI_BATCH=64.
MINI_BATCH="${MINI_BATCH:-128}"
PRIMARY_LOSS="${PRIMARY_LOSS:-cached_mnrl}"
MAX_PAIRS_PER_CLUSTER="${MAX_PAIRS_PER_CLUSTER:-10}"
STRING_FILE="${STRING_FILE:-stringdb_train_15M.parquet}"
MAX_MAP_ROWS="${MAX_MAP_ROWS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
# Between the two measured V2 arms: 2e-4 at ESM-2 150M, 8e-5 at ESM-C 600M.
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
# Safety net, not the plan: 71 h against a 72 h budget. On 3 ranks the epoch is
# 21,646/3 = 7,215 steps at the measured ~34 s/it = ~68 h, so this only fires if
# the machine slows down -- and when it does the run stops and saves rather than
# overrunning silently. 4 ranks would be 5,412 steps and ~51 h.
MAX_MINUTES="${MAX_MINUTES:-4260}"
MATRYOSHKA="${MATRYOSHKA:-0}"          # off; see the header
GATHER="${GATHER:-0}"                  # off, and unsafe to flip while DMS is in
RESUME="${RESUME:-0}"

export HF_HOME=/storage/models/hf_home
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/storage/users/ddofer/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export PROTSENT_ESMPLUSPLUS_ATTN_BACKEND="${PROTSENT_ESMPLUSPLUS_ATTN_BACKEND:-flash_attention_2}"
export PROTEIN_PROGRESS_BARS="${PROTEIN_PROGRESS_BARS:-on}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=32
# NOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, whatever the OOM message
# suggests: expandable segments do not support CUDA IPC and every dataloader
# worker then dies with "pidfd_getfd: Operation not permitted".
NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/users/ddofer/protsent_models}"
OUT="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"

EXTRA_ARGS=()
[[ "$RESUME" == "1" ]] || EXTRA_ARGS+=(--no_resume)
[[ "$MAX_STEPS" -gt 0 ]] && EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
[[ "$GATHER" == "1" ]] || EXTRA_ARGS+=(--no_gather_across_devices)
if [[ "$MATRYOSHKA" == "1" ]]; then
  EXTRA_ARGS+=(--matryoshka --matryoshka_dims 64 128 256)
else
  EXTRA_ARGS+=(--no-matryoshka)
fi

for f in pfam_sorted.parquet afdb_sorted.parquet "$STRING_FILE"; do
  [[ -f "$DATA/$f" ]] || { echo "MISSING: $DATA/$f" >&2; exit 1; }
done
if [[ -n "$DMS_FILE" ]]; then
  [[ -f "$DMS_FILE" ]] || { echo "MISSING: $DMS_FILE" >&2; exit 1; }
  DMS_ARGS=(--dms_file "$DMS_FILE" --dms_max_rows "$DMS_MAX_ROWS")
else
  DMS_ARGS=()   # DMS_FILE="" runs MNRL-only; see the CoSENT/DDP note in the header
fi
if [[ -e "$OUT/final" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "REFUSING: $OUT/final already exists. Set ALLOW_OVERWRITE=1 or change RUN_NAME." >&2
  exit 1
fi

echo "$(date) ProtSent-V2 ESM-C 300M: model=$MODEL out=$OUT"
echo "  bs=$BATCH_SIZE mini=$MINI_BATCH gpus=$NUM_PROCESSES ($CUDA_VISIBLE_DEVICES)" \
     "k=$MAX_PAIRS_PER_CLUSTER lr=$LR len=$MAX_SEQ_LENGTH gather=$GATHER" \
     "matryoshka=$MATRYOSHKA backend=$PROTSENT_ESMPLUSPLUS_ATTN_BACKEND"

uv run --no-sync accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  protein_pipeline.py train \
  --model "$MODEL" \
  --files "$DATA/pfam_sorted.parquet" \
          "$DATA/afdb_sorted.parquet" \
          "$DATA/$STRING_FILE" \
  "${DMS_ARGS[@]}" \
  --loss_mode multi --multi_primary_loss "$PRIMARY_LOSS" \
  --batch_sampler none \
  --mnrl_mini_batch_size "$MINI_BATCH" \
  --multi_dataset_sampler proportional \
  --gor_weight 0.0 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_pairs_per_cluster "$MAX_PAIRS_PER_CLUSTER" \
  --batch_size "$BATCH_SIZE" \
  --max_map_rows "$MAX_MAP_ROWS" \
  --epochs "$EPOCHS" --learning_rate "$LR" --warmup_steps "$WARMUP_STEPS" \
  --pair_dataset_shuffle_seed "$SHUFFLE_SEED" --seed "$SEED" \
  --dataloader_num_workers "$WORKERS" \
  --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --max_minutes "$MAX_MINUTES" \
  --output_root "$OUTPUT_ROOT" \
  --progress_bars on \
  --no-compile \
  "${EXTRA_ARGS[@]}" \
  --run_name "$RUN_NAME"
