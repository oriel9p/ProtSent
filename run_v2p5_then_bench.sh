#!/usr/bin/env bash
# Wait for the running V2.5 training to exit, resume it once if it was cut short,
# then benchmark whatever finished.
#
# The benchmark sweep reuses run_benchmarks_v3.sh against results/benchmarks/v3,
# where the ESM-2 35M, ProtSent-V1-35M and ProtSent-V2-35M arms are already
# complete. arm_is_complete() skips those, so only the two V2.5 arms actually
# run, and V2.5 lands in the same directory as the model it continues from.
#
# The resume exists because MAX_MINUTES=900 is a wall-clock stop, not a step
# target: if the box is contended enough that 14,924 steps do not fit in 15 h,
# training is killed mid-cosine and the saved model sits at an elevated LR, which
# is worse than a shorter completed schedule. Resuming is safe here ONLY because
# nothing changes -- same single GPU, same data knobs -- so the step count and LR
# schedule are re-derived identically and pick up from global_step (RUNS.md).
# Bounded to one attempt: a resume loop that keeps failing would burn the night.
set -uo pipefail
cd ~/ProtSent

TRAIN_PID="${TRAIN_PID:?set TRAIN_PID to the training launcher pid}"
RUN_NAME="${RUN_NAME:-protsent_esm2_35m_v2p5}"
BENCH_GPU="${BENCH_GPU:-5}"
FINAL="models/$RUN_NAME/final"

wait_for() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }

wait_for "$TRAIN_PID"
echo "$(date) training pid $TRAIN_PID exited"

if ! compgen -G "$FINAL"/*.safetensors >/dev/null; then
  ckpt=$(ls -d "models/$RUN_NAME"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: no $FINAL and no checkpoint to resume from" >&2
    exit 1
  fi
  echo "$(date) no final model; resuming once from $ckpt"
  # Trap 1: checkpoints are written with tokenizer_class "FastEsmTokenizer" and no
  # AutoTokenizer entry in auto_map, so SentenceTransformer(dir) dies with
  # "Unrecognized processing class" before training starts. Rewrite that ONE field.
  # Deliberately not make_checkpoint_loadable.py, which also rewrites config.json to
  # plain ESM: leaving config.json alone keeps the model on FastPLM's auto_map and
  # its attention backend, so the resumed half runs the same code as the first half.
  uv run --no-sync python -c '
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "tokenizer_config.json"
cfg = json.loads(p.read_text())
if cfg.get("tokenizer_class") != "EsmTokenizer":
    cfg["tokenizer_class"] = "EsmTokenizer"
    p.write_text(json.dumps(cfg, indent=2))
    print("rewrote tokenizer_class in", p)
' "$ckpt" || { echo "ERROR: tokenizer rewrite failed on $ckpt" >&2; exit 1; }
  # Trap 2: the saved optimizer carries the MaskedLM head's parameters, which the
  # model rebuilt from a checkpoint directory does not have, so resume dies on a
  # parameter-group size mismatch ~8 minutes in without this.
  uv run --no-sync python fix_resume_optimizer.py "$ckpt" || {
    echo "ERROR: fix_resume_optimizer.py failed on $ckpt" >&2; exit 1; }
  RESUME=1 CUDA_VISIBLE_DEVICES="$BENCH_GPU" nohup bash train_esm2_35m_v2p5.sh \
    >> "logs/v2p5/${RUN_NAME}_resume.log" 2>&1 &
  wait_for $!
  echo "$(date) resumed run exited"
fi

if ! compgen -G "$FINAL"/*.safetensors >/dev/null; then
  echo "ERROR: still no weights at $FINAL — not benchmarking" >&2
  exit 1
fi

echo "$(date) benchmarking $FINAL"
MODEL_NEW="models/$RUN_NAME" TAG_NEW=protsent_v2p5 DEVICE=cuda \
  CUDA_VISIBLE_DEVICES="$BENCH_GPU" \
  bash run_benchmarks_v3.sh
echo "$(date) benchmark sweep finished"
