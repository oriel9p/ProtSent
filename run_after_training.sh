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
# Both sizes score into pilot_35m/: the 35M-vs-150M comparison only means anything if every arm
# hit the same references in the same run, so they share one output dir despite the legacy name.
ARMS="${ARMS:-protsent_late_35m_prop protsent_late_150m_prop}"
# proteingym is deliberately NOT in the default stages. The existing run is quarantined as PARTIAL
# (results/late_interaction/pilot_35m/benchmarks/proteingym_partial/) and the full-coverage rerun
# (~1.0 h/arm) is deferred while the scoring path is optimised in a separate session. Opt in only
# when you own that work:
#   STAGES="proteingym" ./run_after_training.sh
STAGES="${STAGES:-scope cath bench_cheap bench_full}"
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
  # Baselines at both sizes, each in its dense (pooled cosine) and zeroshot (MaxSim, no projection,
  # no late training) view -- so "MaxSim scores better" and "late training helped" stay separable
  # at 35M and at 150M rather than only at 35M.
  local args=(--models "protsent_v2_dense=dense:GrimSqueaker/ProtSent-V2-35M"
              --models "protsent_v2_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-35M"
              --models "protsent_v2_150m_dense=dense:GrimSqueaker/ProtSent-V2-150M"
              --models "protsent_v2_150m_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-150M")
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=2 uv run --no-sync python late_interaction_eval.py scope \
    "${args[@]}" --n_boot 1000 --out_dir "$out"
}

stage_cath() {    # CATH v4.3 midnight zone, with the paired McNemar over arms
  local out="$1"
  local args=()
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=3 uv run --no-sync python late_interaction_eval.py cath \
    "${args[@]}" --mcnemar --out_dir "$out"
}

stage_proteingym() {  # MaxSim(mutant, WT) for the late view, pooled cosine for the dense view
  # Substitutions only; indels are where raw MaxSim would be a length artifact. The cap is left at
  # the 500 default deliberately: measured at 28.2M residues, that is ~23 min for the 35M pair and
  # ~59 min for the 150M pair on a contended card, roughly half that on free ones. Passing 2000
  # here (as this did) would have made it ~4.7 h to buy ~0.002 on the mean.
  local out="$1"
  local args=()
  for a in $ARMS; do for s in $(specs_for "$a"); do args+=(--models "$s"); done; done
  CUDA_VISIBLE_DEVICES=2 uv run --no-sync python late_interaction_eval.py proteingym \
    "${args[@]}" --variant dms_substitutions --out_dir "$out"
}

# Baselines at BOTH sizes: a 150M dense view scored only against 35M baselines would confound
# "late training helped" with "the model is 4x bigger".
bench_pairs() {
  echo "esm2_35m=facebook/esm2_t12_35M_UR50D"   "protsent_v2_35m=GrimSqueaker/ProtSent-V2-35M"
  echo "esm2_150m=facebook/esm2_t30_150M_UR50D" "protsent_v2_150m=GrimSqueaker/ProtSent-V2-150M"
  for a in $ARMS; do echo "$a=$M/$a/dense_view"; done
}
# cheap = the scout set, lands in minutes. full = the paper's 22 tasks + ppi_bernett, ~45 min per
# arm per probe, so ~6 h wallclock for 6 arms x 2 probes on one card.
stage_bench_cheap() { CUDA_VISIBLE_DEVICES=1 TASKS=cheap ./run_late_bench.sh $(bench_pairs); }
stage_bench_full()  { CUDA_VISIBLE_DEVICES=1 TASKS=full  ./run_late_bench.sh $(bench_pairs); }

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
