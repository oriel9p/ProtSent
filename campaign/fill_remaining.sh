#!/usr/bin/env bash
# Fill the two benchmark gaps left after the r2 campaign, four GPUs at a time.
#
#   Phase 0  regenerate two dense views. stop_at_and_bench.sh:129 deletes snapshots/*-dense after
#            each sweep to save disk, so late-r2-protsentv2-35m and late-r2-esm2-150m no longer
#            have one and the pooled suite cannot score them.
#   Phase 1  the 22-task paper suite, minus go_mf, on 8 models (4 trained dense views + 4 frozen).
#            go_mf is dropped because benchmark_tasks.py excludes it from FAST_TASKS by name:
#            "large multilabel task, too slow for the default sweep". Nothing else in the paper
#            set is flagged slow.
#   Phase 2  SUPERSEDED, never ran. Its driver was stopped after phase 1 so the frozen references
#            could be re-scoped; campaign/fill_pgym.sh replaced it and campaign/rerun_pgym_zeroshot.sh
#            fixed the OOM that pass hit. Kept because phase 0 and phase 1 DID run and produced
#            results/late_interaction/r2_final/paper_suite.
#   Phase 2  ProteinGym MaxSim for the 5 models that lack it -- the 35M ProtSent arm, and the four
#            FROZEN references. The frozen ones are the point: the section's central claim is that
#            frozen ProtSent matches or beats the trained arms, and ProteinGym currently has no
#            frozen model at all to test that against.
#
# Passing an explicit task list keeps --max_samples unset (see run_late_bench.sh:56), so nothing is
# subsampled and these rows stay comparable with the cheap sweep.
set -uo pipefail
cd /opt/hpc/ddofer/ProtSent
export HF_HOME=/storage/models/hf_home TOKENIZERS_PARALLELISM=false
M=$(pwd)/models/late_interaction
OUT=$(pwd)/results/late_interaction/r2_final/paper_suite
log(){ echo "$(date +%H:%M) $*"; }

PAPER_NO_GOMF="aav_flip antibiotic_resistance binary_subcellular_localization cloning_clf \
ec_classification enzyme_catalytic_efficiency fluorescence material_production metal_ion_binding \
optimal_ph peptide_hla profet_np_sp_cleaved remote_homology signalp_binary solubility stability \
subcellular_loc temperature_stability thermostability variant_effect beta_lactamase_peer"

log "phase 0: regenerating missing dense views"
for a in late-r2-protsentv2-35m late-r2-esm2-150m; do
  snap="$M/$a/snapshots/step-10000"; dense="$snap-dense"
  [[ -d "$dense" ]] && { log "  $a already has one"; continue; }
  uv run --no-sync python - "$snap" "$dense" <<'PY' || log "  $a dense export FAILED"
import sys
import late_interaction as li
snap, dense = sys.argv[1], sys.argv[2]
mve, st = li.build_multivector_encoder(snap, proj_dim=128, max_seq_length=512, device="cpu")
st.save(dense)
print(f"dense view -> {dense}")
PY
done

# Two models per GPU. Trained arms first so a partial run still yields the comparison that matters.
declare -a G0 G1 G2 G3
G0=("vanilla35m_clean_s10000=$M/vanilla35m_clean/snapshots/step-10000-dense"
    "esm2_35m_frozen=facebook/esm2_t12_35M_UR50D")
G1=("late-r2-protsentv2-35m_s10000=$M/late-r2-protsentv2-35m/snapshots/step-10000-dense"
    "protsent_v2_35m_frozen=GrimSqueaker/ProtSent-V2-35M")
G2=("late-r2-esm2-150m_s10000=$M/late-r2-esm2-150m/snapshots/step-10000-dense"
    "esm2_150m_frozen=facebook/esm2_t30_150M_UR50D")
G3=("late-r2-protsentv2-150m_s10000=$M/late-r2-protsentv2-150m/snapshots/step-10000-dense"
    "protsent_v2_150m_frozen=GrimSqueaker/ProtSent-V2-150M")

log "phase 1: paper suite (21 tasks, go_mf dropped) x 8 models, 2 per GPU"
pids=()
for g in 0 1 2 3; do
  eval "arms=(\"\${G$g[@]}\")"
  CUDA_VISIBLE_DEVICES=$g TASKS="$PAPER_NO_GOMF" OUT="$OUT" \
    ./run_late_bench.sh "${arms[@]}" > "logs/paper_gpu$g.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
log "phase 1 done (fail=$fail); results in $OUT"

log "phase 2: ProteinGym MaxSim for the 5 models that lack it"
PG_ARMS=(
  "late-r2-protsentv2-35m_s10000=late:$M/late-r2-protsentv2-35m/snapshots/step-10000"
  "protsent_v2_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-35M"
  "esm2_zeroshot=zeroshot:facebook/esm2_t12_35M_UR50D"
  "protsent_v2_150m_zeroshot=zeroshot:GrimSqueaker/ProtSent-V2-150M"
  "esm2_150m_zeroshot=zeroshot:facebook/esm2_t30_150M_UR50D"
)
for v in dms_substitutions dms_indels; do
  PGYM_ARGS="--max_seq_length 1024 --batch_size 256" \
    ./run_proteingym_parallel.sh "$v" "$(pwd)/results/late_interaction/r2_final" "${PG_ARMS[@]}" \
    >> logs/pgym_fill.log 2>&1 && log "PGym $v done" || log "PGym $v FAILED"
done
log "FILL_REMAINING COMPLETE"
