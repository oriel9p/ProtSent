#!/usr/bin/env bash
# Keep a weights-only copy of every training checkpoint before save_total_limit rotates it out.
#
# The long runs keep one checkpoint on disk and delete all of them at the end, so the only
# dense views that survive a run are step0 and final. Two points cannot tell "still improving"
# from "peaked at 6k steps and drifted" -- and with a cosine schedule over 18,219 steps, the
# second is the failure mode worth catching. watch_curve already scores each checkpoint on SCOPe
# before it disappears; this does the same preservation job for the pooled benchmarks, which are
# too slow to run inline on the training GPU.
#
# Optimizer/scheduler/RNG state is excluded: these snapshots are for scoring, not for resuming.
# ~140 MB per 35M point, ~600 MB per 150M point.
#
# Usage: ./snapshot_checkpoints.sh [run_dir ...]     (default: the four long-run dirs)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
MODELS="${MODELS:-models/late_interaction}"
INTERVAL="${INTERVAL:-120}"
MAX_HOURS="${MAX_HOURS:-24}"

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=("$MODELS/protsent_late_long" "$MODELS/protsent_late_150m_long" \
        "$MODELS/esm2_late_long" "$MODELS/protsent_late_pool_control")
fi

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "snapshotting: ${RUNS[*]}"
while [[ $(date +%s) -lt $deadline ]]; do
  live=0
  for run in "${RUNS[@]}"; do
    [[ -d "$run" ]] || continue
    [[ -f "$run/runtime.json" ]] || live=1   # run still going
    for ckpt in "$run"/checkpoint-*; do
      [[ -d "$ckpt" ]] || continue
      # modules.json is written last, so its presence means the save finished.
      [[ -f "$ckpt/modules.json" ]] || continue
      step="${ckpt##*checkpoint-}"
      dest="$run/snapshots/step-$step"
      [[ -d "$dest" ]] && continue
      mkdir -p "$dest.partial"
      if rsync -a --exclude 'optimizer.pt' --exclude 'scheduler.pt' --exclude 'rng_state*' \
               --exclude 'trainer_state.json' --exclude 'training_args.bin' \
               "$ckpt/" "$dest.partial/" 2>/dev/null; then
        # Trainer checkpoints bypass save_late_and_dense, so they carry the POISONED
        # token_dropout that disable_esm2_token_dropout writes at load time (false), while every
        # exported late/ carries the runtime truth (true). EsmEmbeddings caches the flag in
        # __init__, so a snapshot reloaded as-is silently loses ESM's 0.88 embedding scaling and
        # is not comparable to any other arm. Measured: max|diff| 0.005, cosine 0.999941.
        if [[ -f "$run/late/config.json" && -f "$dest.partial/config.json" ]]; then
          python3 - "$run/late/config.json" "$dest.partial/config.json" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
want = json.load(open(src)).get("token_dropout")
cfg = json.load(open(dst))
if want is not None and cfg.get("token_dropout") != want:
    cfg["token_dropout"] = want
    json.dump(cfg, open(dst, "w"), indent=2)
    print(f"  synced token_dropout={want} into {dst}")
PYEOF
        fi
        mv "$dest.partial" "$dest"   # atomic: a half-copied snapshot never looks complete
        echo "$(date +%H:%M:%S) kept $(basename "$run")@$step ($(du -sh "$dest" | cut -f1))"
      else
        rm -rf "$dest.partial"
      fi
    done
  done
  [[ $live -eq 0 ]] && { echo "all runs finished"; break; }
  sleep "$INTERVAL"
done
