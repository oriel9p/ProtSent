#!/usr/bin/env bash
# Everything still outstanding, run in one chain instead of four scripts that each poll for the
# others.
#
# Those four deadlocked. Each waited with `pgrep -f "[e]val.py proteingym"`, and the agent shell that
# WROTE each script has that text in its own command line (the script arrives as a heredoc), so
# pgrep matched the authoring shells and every waiter blocked on a process that was never going to
# run a benchmark. The [e] bracket trick only stops a pattern from matching the grep that carries it;
# it does nothing about a third process that happens to contain the same literal text.
#
# The predicate here asks the GPUs directly. nvidia-smi reports compute apps, which no shell can
# spoof, and "are the GPUs busy" is the actual question anyway.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
log(){ echo "$(date +%H:%M) $*"; }

gpus_busy(){ [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; }
wait_gpus(){ while gpus_busy; do sleep 30; done; }

MODELS=(
  --models "esm2_35m_frozen=zeroshot:facebook/esm2_t12_35M_UR50D"
  --models "esm2_150m_frozen=zeroshot:facebook/esm2_t30_150M_UR50D"
  --models "protsent_v2_35m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-35M"
  --models "protsent_v2_150m_frozen=zeroshot:GrimSqueaker/ProtSent-V2-150M"
  --models "late-r2-esm2-150m_s10000=late:$M/late-r2-esm2-150m/snapshots/step-10000"
  --models "late-r2-protsentv2-150m_s10000=late:$M/late-r2-protsentv2-150m/snapshots/step-10000"
)

# --- 1. MaxSim as the kNN metric on ProtBench task setups: the comparison actually asked for.
wait_gpus; log "1/3 MaxSim-kNN, one task per GPU"
TASKS=(remote_homology metal_ion_binding subcellular_loc fluorescence)
pids=()
for i in "${!TASKS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i uv run --no-sync python maxsim_knn_bench.py \
    --task "${TASKS[$i]}" "${MODELS[@]}" --knn_k 3 --batch_size 32 --device cuda:0 \
    > "logs/maxsim_knn_${TASKS[$i]}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || log "  a MaxSim-kNN task failed"; done
log "1/3 done"

# --- 2. The two paper tasks the 21-task suite missed.
wait_gpus; log "2/3 paper gap: ppi_bernett + rhla_enzyme_mutations"
OUT=$(pwd)/results/late_interaction/r2_final/paper_suite
declare -a G0 G1 G2 G3
G0=("vanilla35m_clean_s10000=$M/vanilla35m_clean/snapshots/step-10000-dense"
    "esm2_35m_frozen=facebook/esm2_t12_35M_UR50D")
G1=("late-r2-protsentv2-35m_s10000=$M/late-r2-protsentv2-35m/snapshots/step-10000-dense"
    "protsent_v2_35m_frozen=GrimSqueaker/ProtSent-V2-35M")
G2=("late-r2-esm2-150m_s10000=$M/late-r2-esm2-150m/snapshots/step-10000-dense"
    "esm2_150m_frozen=facebook/esm2_t30_150M_UR50D")
G3=("late-r2-protsentv2-150m_s10000=$M/late-r2-protsentv2-150m/snapshots/step-10000-dense"
    "protsent_v2_150m_frozen=GrimSqueaker/ProtSent-V2-150M")
pids=()
for g in 0 1 2 3; do
  eval "arms=(\"\${G$g[@]}\")"
  CUDA_VISIBLE_DEVICES=$g TASKS="ppi_bernett rhla_enzyme_mutations" OUT="$OUT" \
    ./run_late_bench.sh "${arms[@]}" > "logs/papergap_gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || log "  a paper-gap shard failed"; done
log "2/3 done"

# --- 3. ProteinGym substitutions for the frozen arms that are still missing. Slowest, so last.
wait_gpus; log "3/3 ProteinGym substitutions, frozen arms, one per GPU"
OUTP=$(pwd)/results/late_interaction/r2_final
run_pg(){ local name=$1 spec=$2 gpu=$3
  local d="$OUTP/_parallel/$name"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$gpu uv run --no-sync python late_interaction_eval.py proteingym \
    --models "$name=$spec" --variant dms_substitutions --max_seq_length 1024 --batch_size 256 \
    --out_dir "$d" > "$d/log_subs" 2>&1 && log "  $name done" || log "  $name FAILED"; }
run_pg esm2_zeroshot            zeroshot:facebook/esm2_t12_35M_UR50D  0 &
run_pg esm2_150m_zeroshot       zeroshot:facebook/esm2_t30_150M_UR50D 1 &
run_pg protsent_v2_zeroshot     zeroshot:GrimSqueaker/ProtSent-V2-35M 2 &
run_pg protsent_v2_150m_zeroshot zeroshot:GrimSqueaker/ProtSent-V2-150M 3 &
wait
uv run --no-sync python - "$OUTP" <<'PY'
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
log "RUN_REMAINING COMPLETE"
