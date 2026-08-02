# ISM-C-300M vs vanilla ESM-C-300M

Four arms, produced by `../../../run_benchmarks_ism.sh`, 23 tasks each, `--eval_split test`,
zero error rows.

| directory | model | what it is |
|---|---|---|
| `ismc_300m_knn`, `ismc_300m_linear` | `/storage/models/ISM-C-300M` | structure-distilled ESM-C, converted from `jozhang97/ismc-300m-2024-12` by `convert_ismc_to_hf.py` |
| `esmc_300m_knn`, `esmc_300m_linear` | `Synthyra/ESMplusplus_small` | vanilla ESM-C-300M — the matched control, identical architecture and tokenizer |

Read the joined tables in `../ISM_COMPARISON.md`, not these CSVs directly. Confidence
intervals for the retrieval comparisons are in `../scope40_bootstrap_ci_ism.json`.
Interpretation, including two things that must NOT be claimed from these numbers, is in
the `## ISM-C-300M` section of `../../../RUNS.md`.

The CSVs append across runs and record per-task failures in an `Error` column rather than
exiting non-zero, so check completeness with `bench_arm_status.py <csv> 23`.
