#!/usr/bin/env bash
# Pooled-embedding benchmarks for late-interaction arms: identical ProtBench invocation for
# every arm (same tasks, seed, split, probe), so arms stay comparable. KNN is the primary
# probe; linear is reported separately and never averaged with it.
#
# The late models themselves are [L x d] and cannot be fed to a pooled benchmark. What is
# scored here is each arm's DENSE VIEW -- the same trained backbone with mean pooling -- which
# train_late_interaction.py exports next to every model it saves.
#
# Usage:
#   ./run_late_bench.sh                          # the long-run arms, cheap task set
#   ./run_late_bench.sh name=path [name=path ...]
#   TASKS=paper ./run_late_bench.sh              # the paper's 22 suite tasks
#   TASKS=full ./run_late_bench.sh               # paper + ppi_bernett
#   TASKS=proteingym ./run_late_bench.sh          # the 8 ProteinGym tasks (slow)
#   TASKS="remote_homology ec_classification" ./run_late_bench.sh
#
# ProteinGym here scores the DENSE VIEW, so its zero-shot arm is cosine(mutant, WT). For the
# MaxSim equivalent on the multi-vector model itself:
#   late_interaction_eval.py proteingym --models NAME=late:PATH --variant dms_substitutions
#   PROBES=knn ./run_late_bench.sh
set -euo pipefail
cd "$(dirname "$0")"

# Branch scope-hierarchy-main = ProtBench origin/main + our SCOPe hierarchy commit. The older
# scope-hierarchy branch forked off a squashed export with an unrelated root and lacked ~115
# upstream commits. Needs skorch + tabulate in the venv.
PROTBENCH_DIR="${PROTBENCH_DIR:-/opt/hpc/ddofer/ProtBench}"
OUT="${OUT:-$(pwd)/results/late_interaction/pilot_35m/benchmarks}"
PY="$(pwd)/.venv/bin/python"
PROBES="${PROBES:-knn linear}"
# ProtBench defaults --embed_cache_dir to a RELATIVE "embed_cache", which lands inside the
# ProtBench checkout on the 96%-full /opt/hpc. Point it at the per-user cache on /storage, which
# already holds this exact per-model layout.
EMBED_CACHE="${EMBED_CACHE:-/storage/users/ddofer/protbench_cache}"
export HF_HOME="${HF_HOME:-/storage/models/hf_home}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Task presets:
#   cheap  -> ProtBench --very-fast minus conservation_flip (its curated low-variance scout set)
#   paper  -> the 22 suite tasks the ProtSent paper reports (SCOPe, its 23rd, is scored by
#             late_interaction_eval.py scope instead, on the multi-vector model itself)
#   full   -> paper + ppi_bernett (pair-level, the closest task to what late interaction trains)
#   fast   -> ProtBench's own --fast; kept as an escape hatch, not used by the campaign
# conservation_flip and disprot are dropped everywhere: residue-level, ~30 min/arm each, and
# absent from the paper's 23. ss3 is the one residue-level probe kept -- late interaction trains
# per-residue representations and token_classification is the only family that reads them
# directly instead of collapsing them into a pooled vector.
CHEAP_TASKS="remote_homology solubility metal_ion_binding fluorescence stability beta_lactamase_peer ss3"
PAPER_TASKS="aav_flip antibiotic_resistance binary_subcellular_localization cloning_clf \
ec_classification enzyme_catalytic_efficiency fluorescence go_mf material_production \
metal_ion_binding optimal_ph peptide_hla profet_np_sp_cleaved remote_homology signalp_binary \
solubility stability subcellular_loc temperature_stability thermostability variant_effect \
beta_lactamase_peer"

case "${TASKS:-cheap}" in
  cheap)      TASK_ARGS=(-t $CHEAP_TASKS) ;;
  paper)      TASK_ARGS=(-t $PAPER_TASKS) ;;
  full)       TASK_ARGS=(-t $PAPER_TASKS ppi_bernett) ;;
  fast)       TASK_ARGS=(--fast) ;;
  proteingym) TASK_ARGS=(--proteingym) ;;   # the 8 ProteinGym tasks; large and slow
  *)          TASK_ARGS=(-t ${TASKS}) ;;
esac

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  M="$(pwd)/models/late_interaction"
  ARMS=(
    "esm2_35m=facebook/esm2_t12_35M_UR50D"
    "protsent_v2_35m=GrimSqueaker/ProtSent-V2-35M"
    "protsent_late_long=$M/protsent_late_long/dense_view"
    "esm2_late_long=$M/esm2_late_long/dense_view"
    "protsent_late_pool_control=$M/protsent_late_pool_control/dense_view"
  )
fi

for probe in $PROBES; do
  for arm in "${ARMS[@]}"; do
    model="${arm#*=}"
    # A local path that does not exist means the arm has not finished training yet; a bare HF id
    # (no leading /) is always allowed through.
    [[ -e "$model" || "$model" != /* ]] || { echo "SKIP ${arm%%=*} (missing $model)"; continue; }
    echo "=== ${arm%%=*} ($probe) $model"
    (cd "$PROTBENCH_DIR" && "$PY" protein_benchmark_suite.py \
        -m "$model" "${TASK_ARGS[@]}" --eval_split test -p "$probe" --knn_k 3 \
        --seed 42 --embed_cache_dir "$EMBED_CACHE" --output_dir "$OUT/$probe")
  done
done
echo "collect with: $PY $PROTBENCH_DIR/collect_bench_results.py $OUT/knn (and .../linear)"
