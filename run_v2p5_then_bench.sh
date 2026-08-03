#!/usr/bin/env bash
# Wait for the running V2.5 training to exit, then benchmark it.
#
# The benchmark sweep reuses run_benchmarks_v3.sh against results/benchmarks/v3,
# where the ESM-2 35M, ProtSent-V1-35M and ProtSent-V2-35M arms are already
# complete. arm_is_complete() skips those, so only the two V2.5 arms actually
# run, and V2.5 lands in the same directory as the model it continues from.
set -uo pipefail
cd ~/ProtSent

TRAIN_PID="${TRAIN_PID:?set TRAIN_PID to the accelerate launcher pid}"
RUN_NAME="${RUN_NAME:-protsent_esm2_35m_v2p5}"
FINAL="models/$RUN_NAME/final"

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "$(date) training pid $TRAIN_PID exited"

if ! compgen -G "$FINAL"/*.safetensors >/dev/null; then
  echo "ERROR: no weights at $FINAL — training did not finish; not benchmarking" >&2
  exit 1
fi

echo "$(date) benchmarking $FINAL"
MODEL_NEW="models/$RUN_NAME" TAG_NEW=protsent_v2p5 DEVICE=cuda \
  CUDA_VISIBLE_DEVICES="${BENCH_GPU:-5}" \
  bash run_benchmarks_v3.sh
echo "$(date) benchmark sweep finished"
