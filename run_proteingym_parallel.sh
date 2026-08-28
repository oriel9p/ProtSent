#!/usr/bin/env bash
# Score ProteinGym arms in parallel, one arm per idle GPU.
#
#   ./run_proteingym_parallel.sh <variant> <out_dir> name=kind:path [name=kind:path ...]
#
# Per-arm .npz names are unique but the CSV is read-rewrite, so each arm writes to its own out_dir
# and the rows are merged at the end. One arm or one idle card degrades to the serial path.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/storage/models/hf_home} TOKENIZERS_PARALLELISM=false

VARIANT="${1:?variant}"; OUT="${2:?out_dir}"; shift 2
ARGS=${PGYM_ARGS:-"--max_seq_length 1024 --batch_size 256"}
IDLE_UTIL=${IDLE_UTIL:-10} IDLE_MB=${IDLE_MB:-2000}

# Sharing a card is slower than queueing for one, and the training queue holds cards for hours.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
  awk -F', *' -v u="$IDLE_UTIL" -v m="$IDLE_MB" '$2 < u && $3 < m {print $1}')
[[ ${#FREE[@]} -eq 0 ]] && { echo "no idle GPU; refusing to contend" >&2; exit 1; }
echo "idle GPUs: ${FREE[*]}   arms: $#"

declare -A busy                       # gpu -> pid of the arm running on it
failed=0
for spec in "$@"; do
  # Take the first card with no live job. Waiting on a whole wave would let one slow arm (the
  # cosine arms take ~2.5x the maxsim ones) hold up every card behind it.
  while :; do
    for g in "${FREE[@]}"; do
      [[ -z ${busy[$g]:-} ]] && { slot=$g; break 2; }
      kill -0 "${busy[$g]}" 2>/dev/null || { wait "${busy[$g]}" || failed=1; unset "busy[$g]"; slot=$g; break 2; }
    done
    sleep 2
  done
  name="${spec%%=*}"; d="$OUT/_parallel/$name"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$slot" uv run --no-sync python late_interaction_eval.py proteingym \
    --models "$spec" --variant "$VARIANT" $ARGS --out_dir "$d" > "$d/log" 2>&1 &
  busy[$slot]=$!
  echo "  $name -> gpu $slot"
done
for pid in "${busy[@]}"; do wait "$pid" || failed=1; done
(( failed )) && echo "WARNING: at least one arm exited non-zero; check */log" >&2

mkdir -p "$OUT"
find "$OUT/_parallel" -name '*.npz' -exec mv -t "$OUT" {} +
uv run --no-sync python - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1]); target = out / "proteingym_maxsim.csv"
rows = list(csv.DictReader(target.open())) if target.exists() else []
for f in sorted(out.glob("_parallel/*/proteingym_maxsim.csv")):
    rows += list(csv.DictReader(f.open()))
# Last row per (variant, model) wins: fresh per-arm rows are appended after any existing ones.
rows = list({(r["variant"], r["model"]): r for r in rows}.values())
if rows:
    keys = sorted({k for r in rows for k in r})
    with target.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"merged {len(rows)} rows into {target}")
PY
exit $failed
