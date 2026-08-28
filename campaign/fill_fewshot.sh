#!/usr/bin/env bash
# Fill the few-shot column for arms that do not exist yet.
#
# late_interaction_eval.py fewshot_rh appends per invocation, so the arms benchmarked tonight
# (three r2 arms + references) leave a hole wherever an arm was still training. This waits for
# each remaining arm's step-10000/step-2000 snapshot and scores it into the same CSV, so the
# table is complete without a full rerun.
#
# Waits on the snapshot itself, not on a pid: the gate writes the snapshot before the trainer
# exits, and a pid that never appears (a crashed relaunch) would hang this forever.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
OUT=$(pwd)/results/late_interaction/r2_final/benchmarks
log(){ echo "$(date +%H:%M) $*"; }

# arm_name:snapshot_subdir:max_wait_hours
TARGETS=(
  "late-r2-protsentv2-35m:step-10000:20"
  "late-r2-esm2-150m-proj640:step-2000:26"
)

for t in "${TARGETS[@]}"; do
  IFS=: read -r arm snap hours <<<"$t"
  path="$M/$arm/snapshots/$snap"
  deadline=$(( SECONDS + hours * 3600 ))
  log "waiting for $arm/$snap (up to ${hours}h)"
  while [[ ! -d "$path" ]]; do
    if (( SECONDS > deadline )); then log "TIMEOUT waiting for $arm; skipping"; break; fi
    sleep 120
  done
  [[ -d "$path" ]] || continue
  # The gate writes the snapshot dir before its contents settle; wait for the weights file.
  for _ in $(seq 30); do
    compgen -G "$path/*/model.safetensors" > /dev/null && break
    compgen -G "$path/model.safetensors" > /dev/null && break
    sleep 20
  done
  log "scoring few-shot for $arm"
  uv run --no-sync python late_interaction_eval.py fewshot_rh \
    --models "${arm}_${snap/step-/s}=late:$path" \
    --out_dir "$OUT" --batch_size 32 --device cuda:3 --n_boot 2000 \
    >> logs/fill_fewshot.log 2>&1 && log "$arm few-shot done" || log "$arm few-shot FAILED"
done
log "FILL_FEWSHOT COMPLETE"
