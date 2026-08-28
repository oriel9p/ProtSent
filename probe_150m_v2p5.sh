#!/usr/bin/env bash
# Size the 150M V2.5 run. Two stages, because the first attempt got this wrong
# in both directions: it spent 20 min per arm generating pairs to measure step
# time, and then OOMed at 260 GiB of 267 on the very first arm.
#
# Stage 1 (fit): find the largest MINI_BATCH that survives. CachedMNRL bounds
#   peak activation memory by the mini-batch, not the batch, so this is the knob
#   that decides whether batch 1024 is reachable at all on a 30-layer backbone.
#   GOR and CoSENT both embed with grad and ride on the same budget.
# Stage 2 (speed): only among configs that fit, measure the three tradeoffs --
#   Matryoshka (15% budget), 768 tokens, batch 2048.
#
# The corpus is deliberately tiny here. Step time is set by the batch's sequence
# mix, which the proportional sampler keeps roughly stable, not by corpus size;
# the real run's 24M rows would cost 20 min of pair generation per arm and
# measure the same thing.
#
# gather stays off in every arm, always: RUNS.md records a reproduced DDP
# deadlock between gather and a CoSENT target, and this run has one.
set -uo pipefail
cd ~/ProtSent

GPUS="${GPUS:-0,1,4,7}"
STEPS="${STEPS:-10}"
# Keep the 8:8:1 corpus ratio of the real run so the batch mix is comparable.
SMALL=(MAX_MAP_ROWS=3000000 DMS_MAX_ROWS=150000)
mkdir -p logs/v150m

# NOT expandable_segments, despite the OOM message recommending it: expandable
# segments do not support CUDA IPC, and the DataLoader workers rely on it. Setting
# it turns every rank into "RuntimeError: pidfd_getfd: Operation not permitted"
# before step 0 -- which looks like a permissions problem and is not one.

run () {
  local name="$1"; shift
  rm -rf "models/probe_$name"
  echo "=== $(date +%H:%M) $name: $*"
  env "${SMALL[@]}" "$@" CUDA_VISIBLE_DEVICES="$GPUS" RUN_NAME="probe_$name" \
      MAX_STEPS="$STEPS" ALLOW_OVERWRITE=1 \
      timeout 1800 bash train_esm2_150m_v2p5.sh > "logs/v150m/probe_$name.log" 2>&1
  local rc=$?
  local verdict="ok"
  grep -q "OutOfMemoryError" "logs/v150m/probe_$name.log" && verdict="OOM"
  [[ $rc -ne 0 && $verdict == "ok" ]] && verdict="fail(rc=$rc)"
  echo "    -> $verdict"
  rm -rf "models/probe_$name"
}

echo "##### stage 1: what fits at batch 1024, len 512 #####"
run fit_mb32   MINI_BATCH=32  GOR_MAX_SAMPLES=32  MATRYOSHKA=0
run fit_mb64   MINI_BATCH=64  GOR_MAX_SAMPLES=64  MATRYOSHKA=0
run fit_mb128  MINI_BATCH=128 GOR_MAX_SAMPLES=128 MATRYOSHKA=0

echo "##### stage 2: tradeoffs, at the largest mini-batch that fit #####"
MB="${MB:-64}"
run matryoshka MINI_BATCH=$MB GOR_MAX_SAMPLES=$MB MATRYOSHKA=1
run len768     MINI_BATCH=$MB GOR_MAX_SAMPLES=$MB MATRYOSHKA=0 MAX_SEQ_LENGTH=768
run bs2048     MINI_BATCH=$MB GOR_MAX_SAMPLES=$MB MATRYOSHKA=0 BATCH_SIZE=2048

echo; echo "=== $(date +%H:%M) summary ==="
uv run --no-sync python - <<'PY'
import pathlib, re, statistics
rows = []
for p in sorted(pathlib.Path("logs/v150m").glob("probe_*.log")):
    name = p.stem.replace("probe_", "")
    if name == "all":
        continue
    txt = p.read_text()
    if "OutOfMemoryError" in txt:
        rows.append((name, None, None, "OOM")); continue
    its = [float(x) for x in re.findall(r"(\d+\.\d+)s/it", txt)]
    its = its[len(its) // 2:] if len(its) >= 4 else its   # drop warmup half
    if not its:
        rows.append((name, None, None, "no steps")); continue
    bs = 2048 if "bs2048" in name else 1024
    sit = statistics.median(its)
    rows.append((name, sit, bs * 4 / sit, ""))            # 4 ranks

base = next((r[1] for r in rows if r[0].startswith("fit_mb64") and r[1]), None)
print(f"{'arm':12} {'s/it':>8} {'samples/s':>11} {'vs mb64':>9}  note")
for name, sit, sps, note in rows:
    if sit is None:
        print(f"{name:12} {'-':>8} {'-':>11} {'-':>9}  {note}"); continue
    d = f"{(sit / base - 1) * 100:+.1f}%" if base else "-"
    print(f"{name:12} {sit:8.3f} {sps:11.1f} {d:>9}  {note}")
print("\nMatryoshka keeps its slot only if its row is under +15%.")
print("Steps for a 15 h budget = 15*3600 / s_per_it.")
PY
