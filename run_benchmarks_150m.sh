#!/usr/bin/env bash
# Post-training benchmark sweep for ProtSent-V2-150M.
#
# Thin wrapper over run_benchmarks_v3.sh, whose MODEL_* defaults are all 35M. The
# arms mirror the 35M sweep so the two scales are read the same way:
#   protsent_v2_150m          the model trained here on the decontaminated corpus
#   protsent_v2_150m_ckpt3250 near-trough control (the LR schedule ends at peak)
#   protsent_v1_150m          the published/submitted 150M
#   esm2_150m                 the untuned backbone
#
# Every arm is measured through one code path, kNN then linear, --eval_split test.
# Arms already complete are skipped (FORCE=1 overrides), and the embedding cache is
# on, so the linear pass is a classifier refit rather than a second forward pass.
set -uo pipefail
cd ~/ProtSent

export MODEL_NEW="${MODEL_NEW:-models/protsent_esm2_150m_v2}"
export MODEL_OLD="${MODEL_OLD:-oriel9p/protsent-esm2-150M}"
export MODEL_BASE="${MODEL_BASE:-Synthyra/ESM2-150M}"
export MODEL_TROUGH="${MODEL_TROUGH:-models/protsent_esm2_150m_v2_snapshots/checkpoint-3250}"
export OUT="${OUT:-results/benchmarks/v2_150m}"
export BATCH="${BATCH:-32}"   # 150M: half the 35M batch to stay clear of OOM
export TAG_NEW="${TAG_NEW:-protsent_v2_150m}"
export TAG_OLD="${TAG_OLD:-protsent_v1_150m}"
export TAG_BASE="${TAG_BASE:-esm2_150m}"
export TAG_TROUGH="${TAG_TROUGH:-protsent_v2_150m_ckpt3250}"

exec bash run_benchmarks_v3.sh
