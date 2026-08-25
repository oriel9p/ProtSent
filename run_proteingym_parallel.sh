#!/usr/bin/env bash
# Score ProteinGym arms in parallel, one arm per free GPU.
#
# Assays are independent and so are arms, but the per-arm .npz files are uniquely named while the
# CSV is read-rewrite -- concurrent writers would clobber it. So each arm gets its own out_dir and
# the results are merged at the end. That keeps the Python single-threaded and unchanged.
#
#   ./run_proteingym_parallel.sh <variant> <out_dir> name=kind:path [name=kind:path ...]
#
# With one arm, or one free GPU, this degrades to exactly the serial behaviour.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/storage/models/hf_home} TOKENIZERS_PARALLELISM=false

VARIANT="${1:?variant}"; OUT="${2:?out_dir}"; shift 2
ARGS=${PGYM_ARGS:-"--max_seq_length 1024 --batch_size 256"}

# Only cards that are actually idle: the campaign's training arms and bench stages hold others, and
# sharing a card is slower than queueing for one.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
  awk -F', *' '$2 < 10 && $3 < 2000 {print $1}')
[[ ${#FREE[@]} -eq 0 ]] && { echo "no idle GPU; refusing to contend" >&2; exit 1; }
echo "idle GPUs: ${FREE[*]}   arms: $#"

i=0
for spec in "$@"; do
  name="${spec%%=*}"
  gpu="${FREE[$((i % ${#FREE[@]}))]}"
  d="$OUT/_parallel/$name"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python late_interaction_eval.py proteingym \
    --models "$spec" --variant "$VARIANT" $ARGS --out_dir "$d" \
    > "$d/log" 2>&1 &
  echo "  $name -> gpu $gpu"
  i=$((i + 1))
  # Keep at most one job per idle card in flight.
  (( i % ${#FREE[@]} == 0 )) && wait
done
wait

# Merge: .npz names are already unique per arm, the CSV needs its rows concatenated under one header.
mkdir -p "$OUT"
find "$OUT/_parallel" -name '*.npz' -exec mv -t "$OUT" {} +
uv run --no-sync python - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1]); target = out / "proteingym_maxsim.csv"
rows = list(csv.DictReader(target.open())) if target.exists() else []
for f in sorted(out.glob("_parallel/*/proteingym_maxsim.csv")):
    rows += list(csv.DictReader(f.open()))
if rows:
    keys = sorted({k for r in rows for k in r})
    with target.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"merged {len(rows)} rows into {target}")
PY
echo "done"
