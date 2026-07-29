#!/usr/bin/env bash
# The LR schedule is a 3-cycle cosine (--lr_num_cycles 3.0), so the run ENDS at
# peak LR (2e-4) rather than settled. Troughs are at steps 1642 / 2925 / 4208.
# save_total_limit=2 would delete the trough-adjacent checkpoints before we can
# use them, so copy them aside as they appear. Cheap: ~150MB each at 35M params.
set -uo pipefail
cd ~/ProtSent
OUT=models/protsent_esm2_35m_v3_snapshots
mkdir -p "$OUT"
while tmux has-session -t protsent_v3 2>/dev/null; do
  for s in 3000 4000 4500; do
    src="models/protsent_esm2_35m_v3/checkpoint-$s"
    [[ -d "$src" && ! -d "$OUT/checkpoint-$s" ]] && cp -r "$src" "$OUT/" && echo "$(date +%H:%M) snapshotted checkpoint-$s"
  done
  sleep 120
done
echo "training ended; snapshots:"; ls "$OUT" 2>/dev/null
