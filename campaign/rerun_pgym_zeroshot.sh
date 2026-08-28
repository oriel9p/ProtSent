#!/usr/bin/env bash
# Rerun the three zero-shot ProteinGym arms that OOMed, ONE PER GPU.
#
# Why they failed: zero-shot arms are unprojected (proj_dim=0), so each residue is 480/640-d
# instead of the trained arms' 128-d. maxsim_against_one holds a _QUERY_CHUNK (8192) of query
# embeddings at a time, which at 1024 residues and 640 dims is ~21 GB -- 5x what a 128-d arm needs.
# run_proteingym_parallel.sh saw only 3 idle GPUs (the batch probe still held GPU 0) and packed two
# arms onto GPUs 2 and 3; two unprojected arms on one 80 GB card cannot both fit.
#
# So: one arm per GPU, and batch 256 rather than 512. The probe measured 256/512/1024 at 222/221/221s
# -- statistically identical -- so the larger batch bought nothing and only raised the memory floor.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
OUT=$(pwd)/results/late_interaction/r2_final
log(){ echo "$(date +%H:%M) $*"; }

log "waiting for the in-flight ProteinGym pass to finish"
while pgrep -f "[e]val.py proteingym" > /dev/null 2>&1; do sleep 60; done
log "GPUs free"

run_one(){  # name spec gpu variant
  local name=$1 spec=$2 gpu=$3 v=$4
  local d="$OUT/_parallel/$name"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$gpu uv run --no-sync python late_interaction_eval.py proteingym \
    --models "$name=$spec" --variant "$v" --max_seq_length 1024 --batch_size 256 \
    --out_dir "$d" > "$d/log_$v" 2>&1 \
    && log "  $name $v done" || log "  $name $v FAILED"
}

for v in dms_substitutions dms_indels; do
  log "$v: three zero-shot arms, one per GPU"
  run_one esm2_zeroshot            zeroshot:facebook/esm2_t12_35M_UR50D    0 "$v" &
  p0=$!
  run_one esm2_150m_zeroshot       zeroshot:facebook/esm2_t30_150M_UR50D   1 "$v" &
  p1=$!
  run_one protsent_v2_zeroshot     zeroshot:GrimSqueaker/ProtSent-V2-35M   2 "$v" &
  p2=$!
  wait $p0 $p1 $p2
done

log "merging into $OUT/proteingym_maxsim.csv (dedup on variant+model, keep last)"
uv run --no-sync python - "$OUT" <<'PY'
import glob, os, sys
import pandas as pd
out = sys.argv[1]
frames = [pd.read_csv(f) for f in glob.glob(os.path.join(out, "_parallel", "*", "proteingym_maxsim.csv"))]
merged = os.path.join(out, "proteingym_maxsim.csv")
if os.path.exists(merged):
    frames.append(pd.read_csv(merged))
d = pd.concat(frames, ignore_index=True).drop_duplicates(["variant", "model"], keep="last")
d.to_csv(merged, index=False)
print(f"{len(d)} rows -> {merged}")
PY
log "RERUN_PGYM_ZEROSHOT COMPLETE"
