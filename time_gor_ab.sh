#!/usr/bin/env bash
# What does the GOR term actually cost per step?
#
# Neither existing measurement answers this. The V2.5 run (GOR on) and the
# GOR-off ablation ran on different GPUs under different box contention, and the
# pre-fix smoke logs are unusable -- they show GOR *faster*, because back then
# --max_seq_length was not applied and batch length dominated everything else.
#
# So: same GPU, back to back, identical data knobs, only --gor_weight changes.
# Data knobs are deliberately NOT shrunk to make this quicker: the proportional
# sampler's corpus mix sets the mean sequence length per batch, which is what
# step time is made of, so a smaller corpus would time a different batch.
# Costs ~7 min of dataset build per arm; the timed part is 60 steps.
set -uo pipefail
cd ~/ProtSent

GPU="${GPU:-3}"
STEPS="${STEPS:-60}"
mkdir -p logs/v2p5

for w in 0.1 0; do
  name="timing_gor${w/./p}"
  rm -rf "models/$name"
  echo "$(date) === gor_weight=$w ==="
  GOR_WEIGHT="$w" RUN_NAME="$name" MAX_STEPS="$STEPS" CUDA_VISIBLE_DEVICES="$GPU" \
    ALLOW_OVERWRITE=1 bash train_esm2_35m_v2p5.sh > "logs/v2p5/${name}.log" 2>&1
  rm -rf "models/$name"
done

echo "$(date) === results ==="
uv run --no-sync python - <<'PY'
import re, statistics, pathlib
# tqdm prints "<elapsed><remaining, X.XXs/it]"; the last-40-of-60 window skips
# warmup, the LR ramp's first steps, and any lazy CUDA init.
# Glob rather than re-deriving the mangled names: the weight list lives in one
# place (the bash loop above), and a hardcoded list here would print "no log"
# after any edit to it.
for p in sorted(pathlib.Path("logs/v2p5").glob("timing_gor*.log")):
    r = [float(x) for x in re.findall(r"(\d+\.\d+)s/it", p.read_text())][-40:]
    if not r:
        print(f"{p.stem}: no timings -- run failed, check the log"); continue
    print(f"{p.stem:>16}  median {statistics.median(r):.3f} s/it  "
          f"n={len(r)}  min {min(r):.3f}  max {max(r):.3f}")
PY
