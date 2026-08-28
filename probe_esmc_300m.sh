#!/usr/bin/env bash
# Size the ESM-C 300M V2 run on 3 B300s at len 512: how large a CachedMNRL
# mini-batch pays off, and whether the per-device batch can double.
#
# Mini 64 peaked at 66-117 GiB of 275, so the question is how much of that
# headroom converts into speed, not whether a larger chunk fits. Batch is its own
# arm because gather_across_devices is off (see train_esmc_300m_v2.sh), which
# makes the per-device batch the only lever on the negative pool.
#
# Neither Matryoshka nor gather is an arm here; both are disabled in the train
# script for reasons its header records.
#
# The corpus is deliberately tiny. Step time is set by the batch's sequence mix,
# which the proportional sampler keeps roughly stable, not by corpus size; the
# real 35M-pair corpus would cost ~20 min of pair generation per arm.
set -uo pipefail
cd ~/ProtSent

GPUS="${GPUS:-5,6,7}"
STEPS="${STEPS:-16}"
SMALL=(MAX_MAP_ROWS=3000000 DMS_MAX_ROWS=150000)
mkdir -p logs/esmc300

run () {
  local name="$1"; shift
  rm -rf "/storage/users/ddofer/protsent_models/probe_$name"
  echo "=== $(date +%H:%M) $name: $*"
  env "${SMALL[@]}" "$@" CUDA_VISIBLE_DEVICES="$GPUS" RUN_NAME="probe_$name" \
      MAX_STEPS="$STEPS" ALLOW_OVERWRITE=1 \
      timeout 2700 bash train_esmc_300m_v2.sh > "logs/esmc300/probe_$name.log" 2>&1
  local rc=$?
  local verdict="ok"
  grep -q "OutOfMemoryError" "logs/esmc300/probe_$name.log" && verdict="OOM"
  [[ $rc -ne 0 && $verdict == "ok" ]] && verdict="fail(rc=$rc)"
  echo "    -> $verdict"
  rm -rf "/storage/users/ddofer/protsent_models/probe_$name"
}

run mb64        MINI_BATCH=64
run mb128       MINI_BATCH=128
run mb256       MINI_BATCH=256
run bs4096_mb128 MINI_BATCH=128 BATCH_SIZE=4096

echo; echo "=== $(date +%H:%M) summary ==="
uv run --no-sync python - <<'PY'
import pathlib, re, statistics

TOTAL_PAIRS = 34_246_743 + 1_000_000   # k=10 budget + DMS
PROBE_RANKS = 3      # what the arms below were measured on
RUN_RANKS = 4        # what the real run gets

# s/it is per rank and barely moves with rank count -- with gather off each rank
# runs its own independent batch, and only the DDP gradient allreduce grows. So a
# 3-rank measurement carries to 4 ranks by shrinking the step count, not the step.

rows = []
for p in sorted(pathlib.Path("logs/esmc300").glob("probe_*.log")):
    name = p.stem.replace("probe_", "")
    batch = 4096 if "bs4096" in name else 2048
    txt = p.read_text()
    if "OutOfMemoryError" in txt:
        rows.append((name, batch, None, "OOM")); continue
    # Marginal seconds per step over the second half of the run, NOT tqdm's
    # "s/it" field. That field is a running average, and step 1 here costs 2-4x a
    # steady-state step, so the average stays wrong for the whole probe: at mini
    # 64 it read 148 s/it at step 1 and was still reading 38.9 at step 10 while
    # actual steps were taking 37.
    marks = [(int(n), int(m) * 60 + int(s))
             for n, m, s in re.findall(r"(\d+)/\d+ \[(\d+):(\d+)<", txt)]
    sit = None
    if len(marks) >= 4:
        (n0, t0), (n1, t1) = marks[len(marks) // 2], marks[-1]
        sit = (t1 - t0) / (n1 - n0) if n1 > n0 else None
    rows.append((name, batch, sit, "" if sit else "too few steps"))

print(f"{'arm':14} {'batch':>6} {'s/it':>8} {'neg':>6} "
      f"{'steps@4':>8} {'epoch h@3':>10} {'epoch h@4':>10}  note")
for name, batch, sit, note in rows:
    if sit is None:
        print(f"{name:14} {batch:6} {'-':>8} {batch:6} {'-':>8} {'-':>10} {'-':>10}  {note}")
        continue
    h3 = TOTAL_PAIRS / (batch * PROBE_RANKS) * sit / 3600
    s4 = TOTAL_PAIRS / (batch * RUN_RANKS)
    print(f"{name:14} {batch:6} {sit:8.2f} {batch:6} {s4:8,.0f} "
          f"{h3:10.1f} {s4*sit/3600:10.1f}  {note}")
print(f"\n{TOTAL_PAIRS:,} pairs. Budget is 72 h. 'neg' is the in-batch negative "
      f"pool, which equals the per-device batch because gather is off.")
PY
