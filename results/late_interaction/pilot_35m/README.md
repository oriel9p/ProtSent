# Late-interaction pilot (35M) — results

Branch `LateInteractions`; plan and definitions in the repo root scripts:
`late_interaction.py`, `train_late_interaction.py`, `late_interaction_eval.py`,
`run_experiment_queue.sh` (training queue), `run_late_bench.sh` (pooled benchmarks).

Layout (files appear as stages run):

    scope/scope_hierarchy.csv          all-vs-all SCOPe-40 at fold/superfamily/family
                                       (dense cosine vs zero-shot vs trained MaxSim)
    scope/scope_checkpoint_curve.csv   MaxSim SCOPe over training checkpoints
    scope/scope_pairwise_bootstrap.json paired eligible-query deltas vs reference
    scope/per_query_*.npz              per-query hit/AP vectors
    training/                          runtime.json + train_log.csv per arm
    benchmarks/knn|linear/             ProtBench --fast, --eval_split test, seed 42
    ../RESULTS.md                      generated from these CSVs; the current source

Ground rules baked into the tooling:
- `scope40_retrieval` (family, legacy) is separate from `_superfamily` / `_fold`;
  eligible-query metrics are reported next to all-query metrics.
- KNN and linear probe results live in separate tables, never averaged.
- Dense views (mean-pooled 480-D from the late-trained backbone) are what the
  pooled benchmarks consume; the `[L x 64]` late model is only scored with MaxSim.
- Step-0 dense view must reproduce the base model's embeddings (test in
  `tests/test_late_interaction.py`); training arms export `step0/` for the same
  check on the real backbones.
