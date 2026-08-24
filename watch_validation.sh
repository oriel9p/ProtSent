#!/usr/bin/env bash
# Run validate_run.py against each arm as soon as it has something to check, then once more at
# the end. Exists because the bf16 bug ran for hours behind a healthy-looking progress bar; a
# run is not "fine" because it is producing steps.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
GPU="${GPU:-0}"
RUNS=("$@")
[[ ${#RUNS[@]} -eq 0 ]] && RUNS=(protsent_late_long protsent_late_150m_long esm2_late_long protsent_late_pool_control)
deadline=$(( $(date +%s) + ${MAX_HOURS:-12} * 3600 ))
declare -A done_once
while [[ $(date +%s) -lt $deadline ]]; do
  pending=0
  for name in "${RUNS[@]}"; do
    dir="models/late_interaction/$name"
    # A run whose directory does not exist yet has not started; treating that as "finished"
    # would end the watcher before it ever checked anything.
    [[ -d "$dir" ]] || { pending=1; continue; }
    [[ -f "$dir/runtime.json" ]] || pending=1
    # Re-check after the run finishes, since the final export is what gets reported.
    key="$name:$([[ -f "$dir/runtime.json" ]] && echo final || echo live)"
    [[ -n "${done_once[$key]:-}" ]] && continue
    compgen -G "$dir/checkpoint-*" > /dev/null || [[ -d "$dir/late" ]] || continue
    echo "=== $(date +%H:%M:%S) $key"
    CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python validate_run.py "$dir" 2>&1 \
      | grep -E "backbone dtypes|cosine|last logged loss|^  (dtype|loss|drift)|OK:|FAIL:|NOT TRAINING|INCONCLUSIVE"
    done_once[$key]=1
  done
  [[ $pending -eq 0 ]] && { echo "all runs finished and validated"; break; }
  sleep 300
done
