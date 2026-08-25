#!/usr/bin/env bash
# Restart an arm at its next checkpoint so a changed --max_steps costs no work.
# Restarting immediately would discard everything since the last save; --resume picks the run up
# from the checkpoint either way, so waiting is free.
#
#   ./retarget_at_checkpoint.sh <queue-letter> <run-name>
# The new step budget comes from run_experiment_queue.sh's own defaults, so set it there.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
Q="${1:?queue letter}" NAME="${2:?run name}"
D="models/late_interaction/$NAME"
LOG="logs/retarget_$NAME.log"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

echo "$(date +%H:%M:%S) waiting for a checkpoint in $D" >> "$LOG"
while [[ $(date +%s) -lt $deadline ]]; do
  if [[ -f "$D/runtime.json" ]]; then
    echo "$(date +%H:%M:%S) $NAME finished before a checkpoint appeared; nothing to do" >> "$LOG"; exit 0
  fi
  if compgen -G "$D/checkpoint-*" > /dev/null; then
    echo "$(date +%H:%M:%S) checkpoint present: $(ls -d $D/checkpoint-* | tr '\n' ' ')" >> "$LOG"
    ps -eo pid,cmd | grep -E "[_]_run $Q|[t]rain_late_interaction.*$NAME" | awk '{print $1}' | xargs -r kill -TERM
    sleep 15
    ps -eo pid,cmd | grep "[t]rain_late_interaction.*$NAME" | awk '{print $1}' | xargs -r kill -KILL
    sleep 5
    echo "$(date +%H:%M:%S) relaunching queue $Q (resumes from the checkpoint, new budget from the queue)" >> "$LOG"
    QUEUES="$Q" ./run_experiment_queue.sh start >> "$LOG" 2>&1
    exit 0
  fi
  # Heartbeat: a waiter that dies silently is worse than no waiter, because it looks armed.
  echo "$(date +%H:%M:%S) still waiting ($(tr '\r' '\n' < logs/queue_$Q.log | grep -oE '[0-9]+/[0-9]+' | tail -1))" >> "$LOG"
  sleep 120
done
echo "$(date +%H:%M:%S) timed out" >> "$LOG"
