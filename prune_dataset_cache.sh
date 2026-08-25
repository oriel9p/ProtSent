#!/usr/bin/env bash
# Delete stale HF generator-cache entries. Each distinct pair-build config makes a new entry and
# nothing ever reclaims them; they reached 349 GB here.
#
# Keeps: entries any live trainer has memory-mapped (Arrow files show in /proc/PID/maps, NOT in
# /proc/PID/fd -- checking fd alone looks safe and is not), and anything touched recently.
set -uo pipefail
G="${G:-/storage/models/hf_home/datasets/generator}"
KEEP_HOURS="${KEEP_HOURS:-2}"
[[ -d "$G" ]] || { echo "no cache at $G"; exit 0; }

mapfile -t keep < <(for p in $(pgrep -f train_late_interaction.py 2>/dev/null); do
    grep -oE 'generator/default-[0-9a-f]+' "/proc/$p/maps" 2>/dev/null | cut -d/ -f2
done | sort -u)
echo "in use, keeping: ${keep[*]:-none}"

freed=0 n=0
for d in "$G"/default-*; do
  e=$(basename "$d")
  for k in "${keep[@]}"; do [[ "$e" == "$k" ]] && continue 2; done      # exact match, not substring
  [[ -n "$(find "$d" -maxdepth 1 -newermt "-${KEEP_HOURS} hours" -print -quit 2>/dev/null)" ]] && continue
  sz=$(du -sm "$d" 2>/dev/null | cut -f1)
  rm -rf "$d" && n=$((n+1)) && freed=$((freed+sz))
done
echo "deleted $n entries, freed ${freed} MB; $G is now $(du -sh "$G" 2>/dev/null | cut -f1)"
