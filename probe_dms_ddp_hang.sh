#!/usr/bin/env bash
# Why does a DMS/CoSENT batch sharing a step with CachedMNRL batches kill ESM-C
# under DDP, when the identical four-dataset interleave ran 3,600 clean steps on
# ESM-2 150M (logs/v150m/train.log)?
#
# Established already, so not re-tested here:
#   * Ranks really do get different datasets on the same step. accelerate's
#     BatchSamplerShard(split_batches=False) hands rank i batches i, i+N, ...
#     Simulated over the real batch counts: 6.5% of steps mix CoSENT with MNRL,
#     first at step 2. So mixing is common and, on ESM-2, survivable -- it cannot
#     be a sufficient cause on its own.
#   * The observed hazard (~11, 19, 20 steps to failure over three attempts)
#     matches that 6.5% rate, so the trigger IS a mixed step, on ESM-C.
#   * Not orphaned placeholder parameters: the assembled ESM-C SentenceTransformer
#     has 332,997,184 params and all of them are the backbone.
#
# Each arm changes exactly one thing against the run that hung. An arm "survives"
# by reaching $STEPS; a hang shows up as a step counter that stops advancing and
# then a 30-minute NCCL watchdog SIGABRT.
set -uo pipefail
cd ~/ProtSent

GPUS="${GPUS:-5,6,7}"
STEPS="${STEPS:-60}"
mkdir -p logs/dmshang

run () {
  local name="$1"; shift
  rm -rf "/storage/users/ddofer/protsent_models/hang_$name"
  echo "=== $(date +%H:%M) $name: $*"
  env "$@" CUDA_VISIBLE_DEVICES="$GPUS" RUN_NAME="hang_$name" \
      MAX_MAP_ROWS=2000000 DMS_MAX_ROWS=120000 MAX_STEPS="$STEPS" ALLOW_OVERWRITE=1 \
      timeout 2700 bash "$SCRIPT" > "logs/dmshang/$name.log" 2>&1
  local rc=$?
  local reached
  reached=$(tr '\r' '\n' < "logs/dmshang/$name.log" \
            | grep -oE "[0-9]+/$STEPS \[" | tail -1 | grep -oE '^[0-9]+')
  local verdict="SURVIVED"
  [[ "${reached:-0}" -lt "$STEPS" ]] && verdict="HUNG at step ${reached:-0}"
  grep -q "OutOfMemoryError" "logs/dmshang/$name.log" && verdict="OOM"
  echo "    -> $verdict (rc=$rc, reached ${reached:-0}/$STEPS)"
  rm -rf "/storage/users/ddofer/protsent_models/hang_$name"
}

# Arm 1: the exact configuration that hung, to confirm it reproduces on demand.
SCRIPT=train_esmc_300m_v2.sh
run esmc_baseline BATCH_SIZE=2048 MINI_BATCH=128

# Arm 2: ESM-C, but with GOR on -- the one loss-shaping flag the 3,600-step
# ESM-2 run had and this one does not. LossWithGOR wraps BOTH losses, so it may
# be equalising their backward structure.
run esmc_gor BATCH_SIZE=2048 MINI_BATCH=128 GOR_WEIGHT=1.0 GOR_MAX_SAMPLES=64

# Arm 3: ESM-2 150M at THIS run's settings (GOR off), to separate "backbone" from
# "settings". If this hangs, the 150M production run survived because of GOR or
# its batch geometry, not because of its backbone.
SCRIPT=train_esm2_150m_v2p5.sh
run esm2_nogor BATCH_SIZE=1024 MINI_BATCH=64 GOR_WEIGHT=0.0 MATRYOSHKA=0

echo; echo "=== $(date +%H:%M) done; verdicts above ==="
