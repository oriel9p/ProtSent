#!/usr/bin/env bash
# Wait for the 150M V2.5 training to exit, then benchmark it against the arms
# already scored in results/benchmarks/v2_150m (ESM-2 150M, ProtSent-V2-150M),
# so only the new arm actually runs.
set -uo pipefail
cd ~/ProtSent
RUN_NAME="${RUN_NAME:-protsent_esm2_150m_v2p5}"
BENCH_GPU="${BENCH_GPU:-0}"
FINAL="models/$RUN_NAME/final"

# Wait on the pattern, not a pid: the launcher shell exits as soon as accelerate
# takes over, so a captured $! goes stale within seconds.
sleep 300
while pgrep -f "protein_pipeline[.]py train.*${RUN_NAME}" >/dev/null; do sleep 120; done
echo "$(date) training exited"

if ! compgen -G "$FINAL"/*.safetensors >/dev/null; then
  echo "ERROR: no weights at $FINAL — not benchmarking" >&2; exit 1
fi

# Checkpoints are written with FastPLM's tokenizer_class, which plain
# SentenceTransformer(dir) cannot resolve. Rewrite that one field for scoring.
uv run --no-sync python -c '
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "tokenizer_config.json"
cfg = json.loads(p.read_text())
if cfg.get("tokenizer_class") != "EsmTokenizer":
    cfg["tokenizer_class"] = "EsmTokenizer"; p.write_text(json.dumps(cfg, indent=2))
    print("rewrote tokenizer_class")' "$FINAL"

echo "$(date) benchmarking $FINAL"
MODEL_NEW="models/$RUN_NAME" TAG_NEW=protsent_v2p5_150m DEVICE=cuda \
  CUDA_VISIBLE_DEVICES="$BENCH_GPU" bash run_benchmarks_150m.sh
echo "$(date) benchmark sweep finished"
