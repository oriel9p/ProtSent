#!/usr/bin/env bash
# The 150M uses the repo-default 3-cycle cosine (--lr_num_cycles 3.0) with 300
# warmup steps over 3,890, so it TROUGHS at steps 898 / 2,095 / 3,292 and returns
# to the full 2e-4 peak at the final step. checkpoint-3250 is the saved checkpoint
# nearest the last trough and is therefore the settled-LR counterpart to the final
# model -- exactly the control that made the 35M result quotable.
#
# save_total_limit=2 deletes it as soon as 3,750 and the final save land, so copy it
# aside while it exists. ~1.8 GB per snapshot at 150M; disk is not a constraint.
set -uo pipefail
cd ~/ProtSent
OUT=models/protsent_esm2_150m_v2_snapshots
mkdir -p "$OUT"
while tmux has-session -t protsent150 2>/dev/null; do
  for s in 3250 3500; do
    src="models/protsent_esm2_150m_v2/checkpoint-$s"
    if [[ -d "$src" && ! -d "$OUT/checkpoint-$s" ]]; then
      cp -r "$src" "$OUT/" && echo "$(date +%H:%M) snapshotted checkpoint-$s"
    fi
  done
  sleep 120
done
echo "training ended; snapshots:"; ls "$OUT" 2>/dev/null
