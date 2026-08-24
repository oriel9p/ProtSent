#!/usr/bin/env bash
# Wait for the training arms to finish, then benchmark them. Survives disconnect; every stage is
# idempotent (its own done-marker or an append-only CSV), so re-running resumes rather than repeats.
#
#   ./run_after_training.sh                    # wait, then run every stage
#   STAGES="scope cath" ./run_after_training.sh
#   ARMS="protsent_late_35m_prop" ./run_after_training.sh
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}" TOKENIZERS_PARALLELISM=false
M="$(pwd)/models/late_interaction"
RES="$(pwd)/results/late_interaction"
ARMS="${ARMS:-protsent_late_35m_prop protsent_late_150m_prop}"
STAGES="${STAGES:-scope cath bench_cheap proteingym bench_full}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"

wait_for_runs() {
  local deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    local pending=0
    for a in $ARMS; do [[ -f "$M/$a/runtime.json" ]] || pending=1; done
    [[ $pending -eq 0 ]] && { echo "$(date +%H:%M:%S) all arms finished"; return 0; }
    sleep 300
  done
  echo "timed out waiting for: $ARMS"; return 1
}

# Every arm is scored against the same references, so a delta is never a cross-protocol artifact.
specs_for() {  # specs_for <arm>  -> the maxsim and dense views of that arm
  echo "$a=late:$M/$a/late" "${a}_dense=dense:$M/$a/dense_view"
}

stage_scope() {   # SCOPe-40 at fold/superfamily/family, MaxSim and pooled cosine
  local out="$1"; shift
  local args=(--models "protsent_v2_dense=dense:GrimSqueaker/ProtSent-V2-35M"
              --models "protsent_v2_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-35M")
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python late_interaction_eval.py scope \
    "${args[@]}" --n_boot 1000 --out_dir "$out"
}

stage_cath() {    # CATH v4.3 midnight zone, with the paired McNemar over arms
  local out="$1"
  local args=()
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=1 uv run --no-sync python late_interaction_eval.py cath \
    "${args[@]}" --mcnemar --out_dir "$out"
}

stage_proteingym() {  # MaxSim(mutant, WT); the pooled cosine arm comes from `bench`
  local out="$1"
  local args=()
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=2 uv run --no-sync python late_interaction_eval.py proteingym \
    "${args[@]}" --variant dms_substitutions --max_variants_per_assay 2000 --out_dir "$out"
}

bench_pairs() {
  echo "esm2_35m=facebook/esm2_t12_35M_UR50D" "protsent_v2_35m=GrimSqueaker/ProtSent-V2-35M"
  for a in $ARMS; do echo "$a=$M/$a/dense_view"; done
}
stage_bench_cheap() { GPU=3 TASKS=cheap ./run_late_bench.sh $(bench_pairs); }
stage_bench_full()  { GPU=3 TASKS=full  ./run_late_bench.sh $(bench_pairs); }

mkdir -p logs
wait_for_runs || exit 1
for a in $ARMS; do
  echo "=== $a: $(python3 -c "import json;d=json.load(open('$M/$a/runtime.json'));print(f\"{d['steps']} steps, {d['pairs_seen']:,} pairs, {d['wall_time_s']/3600:.1f} h\")" 2>/dev/null || echo '?')"
done

run_stage() {  # skips if its done-marker exists, so a re-run resumes
  local s="$1" mk="logs/after_$1.done"
  [[ -f "$mk" ]] && { echo "[skip] $s"; return 0; }
  ( echo "[run ] $s $(date +%H:%M:%S)"
    case "$s" in
      scope)       stage_scope      "$RES/pilot_35m/scope" ;;
      cath)        stage_cath       "$RES/pilot_35m/benchmarks" ;;
      proteingym)  stage_proteingym "$RES/pilot_35m/benchmarks" ;;
      bench_cheap) stage_bench_cheap ;;
      bench_full)  stage_bench_full ;;
    esac && touch "$mk" && echo "[done] $s $(date +%H:%M:%S)" ) >> "logs/after_$s.log" 2>&1
}
in_stages() { [[ " $STAGES " == *" $1 "* ]]; }

# Wave 1: independent, one card each.
for s in scope cath bench_cheap; do in_stages "$s" && run_stage "$s" & done
wait
# Wave 2: ProteinGym runs after the cheap pooled benchmark, so the quick pooled numbers land first
# and ProteinGym is not competing with them for a card.
for s in proteingym bench_full; do in_stages "$s" && run_stage "$s" & done
wait
echo "all stages finished; markers in logs/after_*.done"
