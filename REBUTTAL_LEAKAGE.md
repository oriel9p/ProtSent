# ProtSent — pretraining/benchmark leakage: methods, controls, and baselines

Working document for the reviewer question on leakage between the **structural
tasks** (SCOPe-40 retrieval, remote homology) and the ProtSent **pretraining
corpora** (AFDB, Pfam, STRING).

Status (2026-07-29): decontamination complete and audited; ProtSent-V2-35M
retrained, benchmarked and cross-checked (§5). The only thing still running is
the multi-seed variability sweep (§0.5), which nothing here depends on.

**Start at §0** — it indexes every artifact produced during the rebuttal work
with its path, the command that made it, and its date. §0a is the list of
durable conclusions and the numbers that must not be reused.

---

## Summary — the answer, with the evidence for each claim

1. **We decontaminated all three pretraining corpora** against the benchmark test
   sets at 40% identity / 80% coverage: 240,005,097 → 226,122,796 rows,
   13.9M removed. Negative controls return **0 hits**; positive controls
   **100%** self-hit at `fident = 1.000`. → §1

2. **SCOPe-40 cannot be decontaminated by anyone**, so it is reported rather than
   filtered. Median max-identity of a SCOPe-40 sequence to a comprehensive
   protein corpus is **0.908**, and **no** sequence falls below 20%. SCOPe domains
   come from PDB entries whose parent sequences are in UniProt, and AFDB covers
   essentially all of UniProt. ESM-2 (UniRef50) carries the identical exposure,
   so the model-vs-model **delta** is the valid measurement. → §2

3. **The gain is not memorisation, and this is measured, not argued.** On the
   published ProtSent-35M vs ESM-2 35M over 1,693 eligible SCOPe-40 queries, the
   advantage is *largest* where the nearest pretraining neighbour is *most
   distant* (+0.0915 hit@10 in the 20–40% identity bin vs +0.0865 above 70%), and
   the only significant gain-vs-identity correlation is **negative**
   (AP: Spearman r = −0.114, p = 2.8e-06). Memorisation predicts the opposite
   sign. It also survives a baseline-headroom control (§0a.3 item 11).
   → §2(a)

4. **Sequence search alone does not explain the results.** An MMseqs2 alignment
   baseline scored under identical metric definitions reaches Recall@10 0.564 on
   SCOPe-40 and AUC 0.652 on remote homology, across 24 benchmark tasks. → §3

5. **A retrained, fully decontaminated model is reported**, so the claim does not
   rest on the argument in (2) alone. → §4

---

## 0. Results index

Every path in this section was checked with `ls` on **2026-07-29** and exists
unless marked `MISSING`. Relative paths are from the repo root
(`/home/ddofer/ProtSent`); everything else is absolute. The **date** column is
the artifact's file mtime, not the date the analysis was designed.

Three reading rules that a future reader will otherwise get wrong:

1. **The benchmark suite appends.** `protein_benchmark_suite.py` never
   overwrites its results CSV, so one task can appear several times in one file.
   The valid row is **the newest row for that task whose `Error` column is
   empty**. Row count is not a completeness test — use
   `python bench_arm_status.py <csv> 23`.
2. **Two separate task-subsetting effects, often confused. They are not the
   same three tasks.**

   *Why the win/tie/loss counts are over 20 tasks, not 23.*
   `antibiotic_resistance`, `remote_homology` and `temperature_stability` have
   no delta in either probe: their main metric is multiclass AUC, and the test
   split contains classes absent from train, so `roc_auc_score(multi_class="ovr")`
   refuses. They are excluded from the counts in both tables. Note that this
   drops **remote homology, the headline task**, out of the aggregate tally — its
   accuracy is reported separately and is not part of the 20.
   Verify with: `comparison.json` -> `tables.{knn,linear}` -> rows whose
   `delta_v1_minus_esm2` is null.

   *A different three tasks ignore `-p knn`.* `ec_classification`, `go_mf` and
   `scope40_retrieval` use a built-in evaluator regardless of the requested probe
   and record `Probe=linear` even inside a `*_knn/` directory, so their rows are
   one measurement printed in both tables rather than two measurements. They ARE
   included in the counts. Verify with the `probe_ignored` flag in
   `comparison.json`.

   The tie band in the win/tie/loss counts is **+/-0.005** (`tie_tol` in
   `comparison.json`). It is not eyeballed, and with 7 ties out of 20 under the
   linear probe it materially affects the record, so quote it whenever the counts
   are quoted.
3. **`BenchmarkSeed=42` in every row of every `v3/` CSV.** These are single-seed
   numbers. Seed variability is the sweep in §0.5, which had not finished when
   this index was written.

### 0.1 Benchmark result CSVs — 4 model arms x 2 probes, 23 tasks, `--eval_split test`

All eight produced by `run_benchmarks_v3.sh`, which for each arm runs:

```bash
uv run --no-sync python protein_benchmark_suite.py \
  -m <MODEL> -t <the 23 tasks> -p {knn|linear} -e test \
  --cache_embeddings -b 64 --device cuda -o results/benchmarks/v3/<tag>_<probe>
```

The 23 tasks are the ones with a paired MMseqs2 row: `aav_flip
antibiotic_resistance beta_lactamase_peer binary_subcellular_localization
cloning_clf ec_classification enzyme_catalytic_efficiency fluorescence go_mf
material_production metal_ion_binding optimal_ph peptide_hla
profet_np_sp_cleaved remote_homology scope40_retrieval signalp_binary
solubility stability subcellular_loc temperature_stability thermostability
variant_effect`. `scope40_retrieval` must be named explicitly or it silently
does not run. `rhla_enzyme_mutations` is excluded (no MMseqs2 baseline exists
for 6-residue mutation-site strings).

| what it shows | artifact path | produced by | rows / n / seed | date | one-line conclusion |
|---|---|---|---|---|---|
| ProtSent-V2 (retrained, decontaminated), kNN | `results/benchmarks/v3/protsent_v3_knn/bench_models_protsent_esm2_35m_v3_final.csv` | `run_benchmarks_v3.sh`, `-m models/protsent_esm2_35m_v3/final -p knn -e test` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 09:59 | Best kNN arm; SCOPe-40 eligible R@10 0.9220, remote homology accuracy 0.66677 |
| ProtSent-V2, linear probe | `results/benchmarks/v3/protsent_v3_linear/bench_models_protsent_esm2_35m_v3_final.csv` | same, `-p linear` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 11:06 | Loses to ESM-2 overall under a linear probe; remote homology accuracy 0.7016 (best) |
| Near-trough checkpoint-4000, kNN | `results/benchmarks/v3/protsent_v3_ckpt4000_knn/bench_models_protsent_esm2_35m_v3_snapshots_checkpoint-4000.csv` | same, `-m models/protsent_esm2_35m_v3_snapshots/checkpoint-4000 -p knn` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 10:50 | Within 0.005-0.008 of the final checkpoint on every structural metric; RH accuracy 0.66554 |
| checkpoint-4000, linear probe | `results/benchmarks/v3/protsent_v3_ckpt4000_linear/bench_models_protsent_esm2_35m_v3_snapshots_checkpoint-4000.csv` | same, `-p linear` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 11:23 | Same conclusion; RH accuracy 0.69883. The final model is not an artifact of where training stopped |
| ProtSent-V1 (published `oriel9p/protsent-esm2-35M`), kNN | `results/benchmarks/v3/protsent_old_knn/bench_oriel9p_protsent-esm2-35M.csv` | same, `-m oriel9p/protsent-esm2-35M -p knn` | **46 rows**, 23 tasks x 2 runs (2026-07-28 and -29), 0 errors, seed 42 | 2026-07-29 12:01 | The 07-29 rerun reproduces the 07-28 numbers exactly; either copy is valid. RH accuracy 0.65875 |
| ProtSent-V1, linear probe | `results/benchmarks/v3/protsent_old_linear/bench_oriel9p_protsent-esm2-35M.csv` | same, `-p linear` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 00:09 | RH accuracy 0.68989 |
| ESM-2 35M baseline, kNN | `results/benchmarks/v3/esm2_35m_knn/bench__storage_models_ESM2-35M.csv` | same, `-m /storage/models/ESM2-35M -p knn` | **46 rows**: 23 from 2026-07-28 of which **2 are error rows**, 23 clean from 2026-07-29 | 2026-07-29 10:36 | Use the 07-29 rows only. RH accuracy 0.58354 |
| ESM-2 35M baseline, linear probe | `results/benchmarks/v3/esm2_35m_linear/bench__storage_models_ESM2-35M.csv` | same, `-p linear` | 23 rows, 23 tasks, 0 errors, seed 42 | 2026-07-29 00:34 | RH accuracy 0.68681 |

The two error rows in `esm2_35m_knn` are `Peptide-HLA Binding` (`Error = '|'`) and
`Thermostability (FLIP)` (`Error = '#'`) dated 2026-07-28 — the FastPLM
tokenizer `KeyError` fixed in commit `37ca0ea`. The 2026-07-29 rerun filled both
in (`peptide_hla` AUC 0.74963, `thermostability` Spearman 0.44486) and **no
other task's numbers changed between the two dates**, so only those two cells
were ever affected. `python bench_arm_status.py <csv> 23` reports
"23 task(s) with a clean result" for all eight arms as of 2026-07-29.

### 0.2 Derived comparison tables and the alignment baseline

| what it shows | artifact path | produced by | rows / n | date | one-line conclusion |
|---|---|---|---|---|---|
| Per-task kNN and linear tables, 4 arms side by side, with caveat sections | `results/benchmarks/COMPARISON.md` | `uv run --no-sync python build_comparison.py` | 23 tasks x 2 probes; win/tie/loss over 20 comparable tasks, tie band ±0.005 | 2026-07-29 12:01 | kNN: V1 11/3/6, V2 10/3/7 vs ESM-2. Linear: V1 4/4/12, V2 2/7/11. The probe decides the headline |
| The same content as machine-readable JSON incl. `summary` and `source_notes` | `results/benchmarks/comparison.json` | same invocation | keys `naming, tie_tol, arms, tables{knn,linear}, summary, caveats` | 2026-07-29 12:01 | Median signed delta vs ESM-2: kNN V1 +0.00749, V2 +0.00410; linear V1 −0.01395, V2 −0.01071 |
| MMseqs2-only baseline over the full benchmark | `results/benchmarks/mmseqs_baseline.json` | `uv run --no-sync python mmseqs_baseline.py --task <task_key>` once per task, appending; flags `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3` | **24 entries**, 23 with a usable main metric (`rhla_enzyme_mutations` has `Spearman: null`, hit coverage 0.0) | 2026-07-28 21:51 | Alignment beats the best embedding model on 3 tasks under kNN and 6 under a linear probe; `solubility` AUC 0.4185 is below chance |
| MMseqs2 scratch dirs incl. the raw hit tables | `/storage/users/ddofer/data/mmseqs_baseline/<task_key>/hits.tsv` | as above (`--work_dir` default) | 25 task dirs; `scope40_retrieval/hits.tsv` is 348 KB | 2026-07-28 | The hit table `bootstrap_ci.py` re-scores for the MMseqs2 arm |
| SCOPe-40 head-to-head, 3 methods, all-query and eligible-query metrics | `results/benchmarks/scope40_table.json` | `protein_benchmark_suite.py` (embeddings) + `mmseqs_baseline.py` (alignment), assembled by hand | 3 entries; `n_queries` 2207, `n_eligible_queries` 1693 in every entry | 2026-07-28 21:59 | Contains ESM-2, ProtSent-V1 and MMseqs2 only — **ProtSent-V2 is not in this file**; V2's SCOPe row lives in the `v3/protsent_v3_*` CSVs |
| Standalone SCOPe-only runs that fed `scope40_table.json` | `results/benchmarks/scope_map/bench__storage_models_ESM2-35M.csv`, `results/benchmarks/scope_map_protsent35m/bench_oriel9p_protsent-esm2-35M.csv` | `protein_benchmark_suite.py -t scope40_retrieval -e test` | 1 row each, seed 42, n_queries 2207 / n_eligible 1693 | 2026-07-28 21:56 / 21:58 | Superseded by the `v3/` arms but numerically identical; keep as the provenance of `scope40_table.json` |
| `-s 5.7` MMseqs2 sensitivity variant (R@1 0.3847, R@10 0.4259; remote homology AUC 0.6262, hit cov 0.6233) | **MISSING** — no results file | `mmseqs_baseline.py` was re-run at reduced sensitivity but the output was not persisted; the numbers survive only in §3 of this document | n/a | 2026-07-28 | Re-derivable but not currently on disk. If it is ever quoted again, re-run and persist it |

### 0.3 SCOPe leakage analyses

| what it shows | artifact path | produced by | n / seeds | date | one-line conclusion |
|---|---|---|---|---|---|
| 95% bootstrap CIs, marginal and paired, on SCOPe-40 retrieval | `results/benchmarks/scope40_bootstrap_ci.json` | `uv run --no-sync python bootstrap_ci.py --models ESM-2=/storage/models/ESM2-35M ProtSent-V1=oriel9p/protsent-esm2-35M ProtSent-V2=models/protsent_esm2_35m_v3/final --mmseqs_hits /storage/users/ddofer/data/mmseqs_baseline/scope40_retrieval/hits.tsv` | `n_queries` 2207, `n_eligible` 1693, `n_boot` **10,000**, resample seed 0 | 2026-07-29 12:53 | Every paired interval for V2 excludes zero, including V2 − MMseqs2 at R@1 (+0.0289 [+0.0035, +0.0544]) |
| Identity-vs-gain correlation, **ProtSent-V1** vs ESM-2 | `results/benchmarks/scope_identity_correlation_v1.json` | `uv run --no-sync python scope_identity_correlation.py --models /storage/models/ESM2-35M oriel9p/protsent-esm2-35M --out results/benchmarks/scope_identity_correlation_v1.json` | 2207 queries, 1693 eligible; `identity_median` 0.907725; bins 0/164/315/1214 | 2026-07-29 09:49 | AP Spearman −0.1136 (p=2.8e-06); gain does not grow with proximity to pretraining data |
| Same, **ProtSent-V2** vs ESM-2 | `results/benchmarks/scope_identity_correlation_v2.json` | same script, `--models /storage/models/ESM2-35M models/protsent_esm2_35m_v3/final --out ..._v2.json` | same 2207 / 1693 / 0.907725 / 0/164/315/1214 | 2026-07-29 09:46 | AP Spearman −0.1162 (p=1.6e-06); identical sign and magnitude to V1 |
| Headroom control on the V1 correlation | `results/benchmarks/scope_identity_partial_v1.json` | `uv run --no-sync python scope_identity_partial.py --models /storage/models/ESM2-35M oriel9p/protsent-esm2-35M --out results/benchmarks/scope_identity_partial_v1.json` | `n_eligible` 1693; AP quartile strata n = 423/423/423/424; hit@10 zero-baseline stratum n = 404 | 2026-07-29 12:57 | Partial Spearman −0.083 (p=6.8e-04); the null is not a headroom artifact |
| Headroom control on the V2 correlation | `results/benchmarks/scope_identity_partial_v2.json` | same, `--models /storage/models/ESM2-35M models/protsent_esm2_35m_v3/final --out ..._v2.json` | same n; V2 AP quartile Spearmans +0.007 / −0.090 / −0.158 / −0.057 | 2026-07-29 12:56 | Partial Spearman −0.0806 (p=9.0e-04); among the 404 zero-baseline queries, identity does not predict gain (+0.038, p=0.45) |
| Per-SCOPe-sequence max identity to the pretraining corpus, before and after dc40 | `/storage/users/ddofer/data/decontam_work/scope_strat/scope40_max_identity.parquet` | `scope_identity_strat.py`, consuming the streaming reductions from `decontam_work/scope_strat/reduce.sh` | **2,207 rows x 27 columns**; headline column `max_ident_overall` (= `fident × tcov`), dc column `max_ident_dc_overall` | 2026-07-28 23:40 | Median max identity 0.9077 unfiltered, 0.8950 after dc40; **the [0, 0.2) bin is empty**, so low-identity stratification is impossible on this benchmark |
| Superseded first-pass correlation (see §0a "Superseded") | `results/benchmarks/scope_identity_correlation.json` | earlier run of the same script against the pre-regeneration identity table | 2207 / 1693; `identity_median` 0.893; bins 0/174/351/1168 | 2026-07-28 22:50 | **Do not cite.** Replaced by `_v1.json` |

Note on `scope_identity_partial.py`: its module docstring shows a `--json`
flag, but the actual CLI is `--models BASELINE PROTSENT --out PATH`. Use the
CLI, not the docstring.

### 0.4 Corpus verification

| what it shows | artifact path | produced by | rows | date | one-line conclusion |
|---|---|---|---|---|---|
| Zero flagged sequences survived into the files training actually opened | `results/benchmarks/training_corpus_verification.json` | `uv run --no-sync python verify_training_corpus.py` (no arguments; paths are hardcoded to `/storage/users/ddofer/data/protsent-data-dc40`) | pfam 27,929,772 / afdb 126,301,607 / stringdb 15,000,000; `leaked_total` 0 for every file and column; `all_clean: true` | 2026-07-29 10:10 | The decontamination held all the way to the training loader, checked by semi-join and not taken from the filter's own report |
| Console transcript of the same run | `logs/verify_corpus.log` | as above | 3 checks, all `CLEAN` | 2026-07-29 10:10 | Row arithmetic closes: 27,929,772 + 126,301,607 + 15,000,000 = 169,231,379 |
| The filtering job's own record of intent | `/storage/users/ddofer/data/protsent-data-dc40/decontam_report.json` | `uv run --no-sync python decontaminate_pretrain.py --corpus all --gpu` (defaults `--min-seq-id 0.4 --cov 0.8 --cov-mode 1`) | per-corpus rows before/after/removed, prefilter, test set, shard count | 2026-07-28 19:41 | Records that only two test sets were ever filter targets: `biomap-research/fold_prediction[test]` (3,244) and `Synthyra/bernett_gold_ppi[test]` (3,022) |
| Every hit with per-hit `fident`/`qcov`/`tcov`, plus the leaked-sequence lists | `/storage/users/ddofer/data/protsent-data-dc40/decontam/` | same | `afdb_leaked_sequences.parquet` 1.2 GB, `pfam_...` 47 MB, `stringdb_...` 64 MB, plus `*_hits.tsv.gz` and the two test FASTAs | 2026-07-26 | The audit trail `verify_training_corpus.py` joins against |

### 0.5 In flight at the time of writing — multi-seed variability sweep

**Status as of 2026-07-29 13:11: RUNNING, no output file yet.**

| what it shows | artifact path | produced by | n / seeds | date | state |
|---|---|---|---|---|---|
| Probe-seed variability, 3 arms x 8 tasks x 5 seeds, kNN | `results/benchmarks/seeds/` — **directory exists but is EMPTY** | `tmux new-session -d -s seeds 'cd ~/ProtSent && CUDA_VISIBLE_DEVICES=1 bash run_seed_variability.sh 2>&1 \| tee logs/seeds/sweep.log'` | seeds `0,1,2,3,4`; tasks `remote_homology solubility stability thermostability fluorescence metal_ion_binding subcellular_loc variant_effect`; arms `esm2_35m`, `protsent_v1`, `protsent_v2` | started 2026-07-29 12:54 | Arm 1 of 3 (`esm2_35m`) only; at 13:11 it was on seed 5/5, task `subcellular_loc`. Arms 2 and 3 have not started. **No `.csv` has been written for any arm** — the suite writes at the end of a run |
| Live log for arm 1 | `logs/seeds/esm2_35m.log` | as above | 370 lines at 13:11 | 2026-07-29 13:08 | Progressing normally, no tracebacks |
| Sweep driver log | `logs/seeds/sweep.log` | as above | 1 line (`=== 12:54:00 esm2_35m ===`) | 2026-07-29 12:54 | No arm has reported `rc=` yet |

`scope40_retrieval` is deliberately **not** in the seed sweep: retrieval has no
probe randomness, so every seed returns an identical number. Its uncertainty is
quantified by `bootstrap_ci.py` (§0.3) instead, which resamples queries.

**Nothing in this document or in the rebuttal currently depends on the seed
sweep.** If it is still unfinished when the response goes out, say so; do not
quote a partial arm.

### 0.6 Models

| what it is | path | provenance | size / steps | date | note |
|---|---|---|---|---|---|
| **ProtSent-V2-35M** (paper name); on disk the RUN_NAME is `v3` | `models/protsent_esm2_35m_v3/final` | `train_esm2_35m.sh`, 7x NVIDIA B300, log `logs/esm2_v3/protsent_esm2_35m_v3.log` | 4,850 steps, 1 epoch over 169,231,379 rows, `train_runtime` 3.917e+04 s (10 h 53 m), 887.5 samples/s, `effective_batch=7168`, `world_size=7` | 2026-07-29 08:03 (weights); `config.json`/`tokenizer_config.json` rewritten 09:45 | The rewrite at 09:45 is `make_checkpoint_loadable.py`; originals kept as `*.fastplm`. Weights untouched |
| Trainer checkpoints kept by `save_total_limit=2` | `models/protsent_esm2_35m_v3/checkpoint-4500`, `.../checkpoint-4850` | same run | — | 2026-07-29 07:16 / 08:03 | Not benchmarked; `final` is `checkpoint-4850` re-saved |
| Near-trough LR snapshot, **benchmarked** | `models/protsent_esm2_35m_v3_snapshots/checkpoint-4000` | `snapshot_ckpt.sh`, then `make_checkpoint_loadable.py` | LR 5.5e-5; nearest saved checkpoint to the last cosine trough at step 4,208 | 2026-07-29 06:11 (weights), 09:45 (config rewrite) | The LR-schedule control arm in §0.1 |
| Second near-trough snapshot, **never benchmarked** | `models/protsent_esm2_35m_v3_snapshots/checkpoint-3000` | `snapshot_ckpt.sh` | still in FastPLM form (no `config.json.fastplm`, so `make_checkpoint_loadable.py` has not been run on it) | 2026-07-29 03:57 | Exists on disk; **no results anywhere depend on it** |
| ProtSent-V1 (submitted paper model) | `oriel9p/protsent-esm2-35M` (HF hub, not a local path) | published before the rebuttal period | — | — | Trained on the **unfiltered** corpus |
| ESM-2 35M starting point | `/storage/models/ESM2-35M` | third-party | — | — | Untuned baseline arm |

### 0.7 Data

| what it is | path | produced by | size / rows | date | note |
|---|---|---|---|---|---|
| Decontaminated corpus, all three parquets | `/storage/users/ddofer/data/protsent-data-dc40/` | `decontaminate_pretrain.py --corpus all --gpu` | `pfam_sorted.parquet` 2.47 GB / 27,929,772 rows; `afdb_sorted.parquet` 12.2 GB / 126,301,607; `stringdb_train.parquet` 34.2 GB / 71,891,417 | 2026-07-26 | Also holds `README.md` (dataset card) and `decontam_report.json` |
| The STRING file training actually used | `/storage/users/ddofer/data/protsent-data-dc40/stringdb_train_15M.parquet` | seeded (seed 42) 15M-pair subsample of `stringdb_train.parquet` | 7.14 GB, 15,000,000 rows | 2026-07-28 20:25 | A **compute-budget** decision, not a leakage control. Do not present it as one |
| SCOPe identity table | `/storage/users/ddofer/data/decontam_work/scope_strat/scope40_max_identity.parquet` | `scope_identity_strat.py` | 2,207 rows x 27 cols | 2026-07-28 23:40 | Use `max_ident_overall` (`fident × tcov`), **never raw `fident`** |
| SCOPe stratification working dir (reps FASTAs, leaked-ID lists, hit TSVs, driver scripts) | `/storage/users/ddofer/data/decontam_work/scope_strat/` | `reduce.sh`, `build_reps.py`, `search_reps.sh`, `search_stringdb*.sh` | ~16 TB of intermediates incl. `afdb_lowbin_hits.tsv` 8.8 GB | 2026-07-28 / -29 | Intermediates, not results; kept so the identity table can be rebuilt |

### 0.8 Scripts — what each does and how to re-run it

Every script below has a `--selfcheck` mode except `mmseqs_baseline.py`,
`run_benchmarks_v3.sh`, `run_seed_variability.sh` and `bench_arm_status.py`.
All Python is run through `uv run --no-sync python <script>`.

| script | what it does | re-run |
|---|---|---|
| `decontaminate_pretrain.py` | Removes any pretraining sequence aligning to a benchmark test sequence at ≥40% identity / ≥80% coverage of the *test* sequence, via MMseqs2 `easy-search` with the corpus as query. | `python decontaminate_pretrain.py --corpus all --gpu` (defaults `--min-seq-id 0.4 --cov 0.8 --cov-mode 1 --shard-size 10000000`) |
| `verify_training_corpus.py` | Re-reads the parquets the training log shows being opened and semi-joins them against the recorded removal lists — proves the filter's *result*, not its intent. | `python verify_training_corpus.py` (no args) |
| `mmseqs_baseline.py` | Scores each benchmark task with alignment instead of embeddings, under the same metric definitions; no-hit queries count as failures. | `python mmseqs_baseline.py --task scope40_retrieval` (per task; `--output` defaults to `results/benchmarks/mmseqs_baseline.json`, `--threads 64`) |
| `run_benchmarks_v3.sh` | Drives the 4-arm x 2-probe sweep on `--eval_split test`, caps BLAS threads, skips arms that are already complete, and checks the CSV rather than trusting the exit code. | `bash run_benchmarks_v3.sh` (`FORCE=1` to re-measure everything) |
| `bench_arm_status.py` | Decides whether one arm succeeded: complete = every requested task has at least one error-free row *somewhere* in the appended CSV. | `python bench_arm_status.py <results.csv> 23` (exit 0 = complete) |
| `make_checkpoint_loadable.py` | Rewrites a FastPLM-saved checkpoint's `config.json` / `tokenizer_config.json` into plain-ESM form so `SentenceTransformer(path)` can load it. Metadata only; weights untouched; idempotent; originals kept as `*.fastplm`. | `python make_checkpoint_loadable.py <checkpoint_dir> [...]` |
| `build_comparison.py` | Merges the four arms per task per probe into `COMPARISON.md` + `comparison.json`, with the caveat sections. Idempotent; missing cells stay missing. | `python build_comparison.py` |
| `bootstrap_ci.py` | Bootstraps the per-query SCOPe metrics: marginal CIs per method and **paired** CIs on per-query differences (the ones that settle anything). | `python bootstrap_ci.py --models ESM-2=... ProtSent-V1=... ProtSent-V2=... --mmseqs_hits <hits.tsv>` (`--n_boot` default 10000) |
| `scope_identity_correlation.py` | Correlates each SCOPe query's max identity to the pretraining corpus against that query's retrieval gain over the baseline; reports Spearman/Pearson and per-bin means. | `python scope_identity_correlation.py --models <BASELINE> <PROTSENT> --out <path>` |
| `scope_identity_partial.py` | The same correlation after controlling for baseline headroom: partial Spearman, within-quartile strata, and headroom-normalised gain. | `python scope_identity_partial.py --models <BASELINE> <PROTSENT> --out <path>` |
| `scope_identity_strat.py` | Builds `scope40_max_identity.parquet` from the streaming reductions, carrying both `fident × tcov` and high-coverage raw `fident`, before and after dc40. | `python scope_identity_strat.py` (reads `/storage/users/ddofer/data/decontam_work/scope_strat/`) |
| `run_seed_variability.sh` | Runs 8 tasks x 5 seeds x 3 arms under kNN in one process per arm, reusing the loaded model and the embedding cache. | `bash run_seed_variability.sh` (`SEEDS=`, `TASKS=` overridable) |

### 0.9 Logs worth keeping

| log | what it records | date |
|---|---|---|
| `logs/esm2_v3/protsent_esm2_35m_v3.log` | The full V2 training run: `total=169,231,379`, `effective_batch=7168`, `world_size=7`, `train_runtime` 3.917e+04, clean exit at 08:03 | 2026-07-29 08:03 |
| `logs/bench_v3/{protsent_v3,protsent_old,esm2_35m,protsent_v3_ckpt4000}_{knn,linear}.log` | Per-arm benchmark stdout | 2026-07-29 |
| `logs/bench_v3/sweep2.log` | The driver's own record of which arms ran and which were skipped | 2026-07-29 11:23 |
| `logs/mmseqs_bl_scope.log`, `logs/mmseqs_bl_rh.log` | MMseqs2 baseline runs incl. the search parameters MMseqs2 echoes | 2026-07-28 19:54 / 19:55 |
| `logs/scope_corr_v1.log`, `logs/scope_corr_v2.log` | The two identity-correlation runs; both print `2207 SCOPe sequences; identity median 0.908` | 2026-07-29 09:49 / 09:46 |
| `logs/verify_corpus.log` | The corpus verification transcript | 2026-07-29 10:10 |
| `logs/seeds/esm2_35m.log`, `logs/seeds/sweep.log` | The in-flight seed sweep (§0.5) | 2026-07-29, still being written |

---

## 0a. Conclusions and checks from the rebuttal sessions

Durable findings, each with the path that establishes it. These are the
statements that should survive into a revision; everything else in this document
is working detail.

### 0a.1 What the decontamination establishes

1. **Decontamination at 40% identity / 80% coverage completed on all three
   corpora.** Pfam 28,530,684 → 27,929,772 (−2.11%), AFDB 135,404,259 →
   126,301,607 (−6.72%), STRING 76,070,154 → 71,891,417 (−5.49%).
   → `/storage/users/ddofer/data/protsent-data-dc40/decontam_report.json`, §1.

2. **Verified on the files training actually opened, not on the filter's own
   report.** Semi-join against the recorded removal lists returns **0 surviving
   flagged sequences** in all three files (STRING checked on both pair columns).
   → `results/benchmarks/training_corpus_verification.json` (`all_clean: true`),
   `verify_training_corpus.py`.

3. **Row arithmetic closes independently.** 27,929,772 + 126,301,607 +
   15,000,000 = **169,231,379**, exactly the `total=` in the training log.
   → `logs/esm2_v3/protsent_esm2_35m_v3.log`.

4. **Scope limit — state this prominently.** Only **`remote_homology`**
   (`biomap-research/fold_prediction[test]`, 3,244 sequences) and
   **`ppi_bernett`** (`Synthyra/bernett_gold_ppi[test]`, 3,022 sequences) were
   ever used as decontamination targets. **The other 21 benchmark test sets were
   not filtered.** The decontamination claim covers those two tasks and nothing
   else. → `decontam_report.json`, which names one `test_set` per corpus and no
   others.

5. **SCOPe-40 was deliberately not a filter target.** `tattabio/scope40_test`
   has no train/test split — the benchmark is leave-one-out self-retrieval over
   the whole set — so filtering AFDB/Pfam against all of SCOPe at 40% identity
   would remove essentially every structured domain from the corpus. This is the
   ProtTucker posture (identity-based decontamination of the *supervised* split
   plus low-identity stratification, no filtering of the pLM's pretraining
   corpus). → §2, §2 "Precedent (verified)".

### 0a.2 What the retrained model shows

6. **The retrained model beats the submitted one and beats a tuned MMseqs2 at
   every SCOPe cutoff, and every paired bootstrap CI excludes zero.** V2 − V1,
   V2 − ESM-2 and V2 − MMseqs2 all exclude zero at R@1, R@10 and MAP; the
   tightest is V2 − MMseqs2 at R@1, +0.0289 [+0.0035, +0.0544].
   → `results/benchmarks/scope40_bootstrap_ci.json` (10,000 resamples, 1,693
   eligible queries), §4a of `rebuttal/NEW_EVIDENCE.md`.

7. **Remote homology — the task the corpus was actually filtered against —
   improved.** kNN accuracy 0.58354 (ESM-2) → 0.65875 (V1) → **0.66677** (V2);
   linear 0.68681 → 0.68989 → **0.70160**. Removing every pretraining sequence
   within 40% identity / 80% coverage of that test set did not cost performance
   on it. → the eight CSVs in §0.1, read as `Remote Homology (Fold)` /
   `Accuracy`.

8. **The near-trough checkpoint agrees with the final one.** The 3-cycle cosine
   schedule ends at peak LR, so checkpoint-4000 (LR 5.5e-5) was benchmarked as a
   control; it differs from the final checkpoint by 0.005-0.008 on every
   structural metric. The final model is not an artifact of where training
   stopped. → `results/benchmarks/v3/protsent_v3_ckpt4000_{knn,linear}/`.

9. **There is no unfiltered-corpus retrain at the V2 configuration, and none is
   planned.** V2 also changed the effective batch (7x1024 vs 1x1024), dropped
   synthetic hard negatives, and uses proportional sampling over one epoch. The
   V1-vs-V2 comparison is therefore **not a controlled decontamination
   ablation**. The only supportable claim is the sufficient one: *decontaminating
   the corpus did not cost performance.* → §5 "Caveat to state plainly", §2 of
   `rebuttal/NEW_EVIDENCE.md`.

### 0a.3 What the leakage analysis shows

10. **The identity-vs-gain correlation is null to negative, for both models.**
    Per-query Spearman between max identity to the pretraining corpus and gain
    over ESM-2: R@10 −0.038 (V1) / −0.038 (V2); AP −0.1136 / −0.1162, both
    p < 3e-6. Memorization predicts the opposite sign.
    → `results/benchmarks/scope_identity_correlation_v{1,2}.json`.

11. **It survives a baseline-headroom control.** A blind reader raised
    regression to the mean — high-identity queries are already well solved, so
    they have less room to improve, and gain is bounded above by (1 − baseline)
    (`rebuttal/review_blind.md`, "#4's binning is confounded"). The control was
    run *because* that objection was raised, and it did not change the
    conclusion: partial Spearman controlling for baseline score is −0.083 (V1,
    p=6.8e-04) / −0.081 (V2, p=9.0e-04); every within-quartile correlation for
    V2 is null or negative (+0.007, −0.090, −0.158, −0.057); and among the 404
    queries where the untuned backbone scores zero at R@10 — full headroom by
    construction — identity does not predict the gain (V2 +0.038, p=0.45).
    → `results/benchmarks/scope_identity_partial_v{1,2}.json`,
    `scope_identity_partial.py`. **Record that the objection was raised and that
    it survived** — a pre-empted objection that survived its own control is
    worth more than the raw correlation.

12. **Recall@K on SCOPe-40 is upper-bounded at 0.7671.** Only 1,693 of 2,207
    queries have any non-self same-family protein in the gallery; the other 514
    are singleton families and are unachievable for any method. Every SCOPe
    recall number needs this in its caption.
    → `n_queries` / `n_eligible_queries` columns in every SCOPe CSV row, and
    `results/benchmarks/scope40_table.json`.

13. **Low-identity stratification is impossible on this benchmark, and the
    reason is a property of the sequence universe.** The [0, 0.2) bin is empty
    and the median max identity is 0.908, because AFDB covers essentially all of
    UniProt and SCOPe domains come from PDB entries whose parent sequences are
    in UniProt. ESM-2's UniRef50 carries the identical exposure, so the
    model-vs-model delta is the valid measurement.
    → `scope40_max_identity.parquet`, §2(a).

### 0a.4 The honesty constraints

14. **The probe decides the headline. This is the main constraint on any
    claim.** Over the 20 tasks comparable in both arms, both ProtSent models
    beat ESM-2 under kNN (V1 11/3/6, V2 10/3/7; median delta +0.0075 / +0.0041)
    and lose under a linear probe (V1 4/4/12, V2 2/7/11; median −0.0139 /
    −0.0107). The structural-retrieval advantage survives both; a
    general-purpose superiority claim does not. **Any "ProtSent > ESM-2"
    sentence must name the probe.** → `results/benchmarks/COMPARISON.md`,
    `comparison.json` → `summary`.

15. **MMseqs2 wins outright on several tasks — the measured
    generality-accuracy trade-off.** 3 tasks under kNN (`ec_classification`,
    `go_mf`, `beta_lactamase_peer`) and 6 under a linear probe (those plus
    `enzyme_catalytic_efficiency`, `optimal_ph`, `stability`).
    → `comparison.json` → `summary.{knn,linear}.mmseqs_beats_best_embedding`.

16. **MMseqs2 also beats the submitted model at top-1 on SCOPe** (+0.0697
    [+0.0413, +0.0975], significant) and beats ESM-2 at top-1 by +0.1565. It
    does **not** beat V2. State the V1 weakness first; that is what makes the V2
    result credible. Never retro-claim the top-1 win for the submitted paper.
    → `scope40_bootstrap_ci.json` → `paired`.

17. **MMseqs2 vs ESM-2 at depth is unresolved**, not a win for the pLM: R@10
    −0.0213 [−0.0484, +0.0047], MAP −0.0125 [−0.0351, +0.0102]. Do not claim
    ESM-2 beats alignment at depth. → same file.

18. **Every result in `v3/` is single-seed (`BenchmarkSeed=42`).** The
    multi-seed sweep that would quantify probe-seed variance had not finished
    when this index was written (§0.5). The bootstrap CIs quantify *which
    proteins are in the benchmark*, not training-seed or probe-seed variance.

### 0a.5 Paper description errors found by audit — disclose proactively

19. **"100,000 sequences at the superfamily level" is wrong on both counts.**
    The code evaluates the SCOPe **family** field on **2,207** sequences. The
    100,000 is the evaluator's `max_samples` cap echoed into the results table —
    visible as `Samples,100000` in the header row of every benchmark CSV
    alongside `n_queries,2207`. An apparent 45x error in reported N is a logging
    artifact. → any CSV in `results/benchmarks/v3/`.

20. **The remote-homology test split is not hierarchy-disjoint.** It is TAPE
    remote homology repackaged: the pooled concatenation of TAPE's three
    holdouts (718 fold + 1,254 superfamily + 1,272 family = 3,244) with no
    column marking which. Two thirds is not fold-disjoint. The pooled 457-class
    macro AUC is also not comparable to published per-holdout top-1 accuracies.
    Corpus-level decontamination is the real control. → §3 "Split protocol",
    `results/benchmarks/COMPARISON.md` → "Metrics that are not comparable to
    published literature".

21. **The PPI decontamination text does not match `data_prep.py`.** The code
    uses `easy-search` (STRING as query, Bernett test as target) at 40%
    identity, `--cov-mode 1 -c 0.8`, removing hit query IDs — not `easy-linclust`
    at 50% with cluster-level removal. Its own docstring says `easy-search` was
    chosen deliberately because linclust loses sensitivity below ~50% identity.
    Describe what the code does. → `data_prep.py`, §6 item 4.

22. **Eq. 1 is malformed**, as the reviewer noted. → §7 of
    `rebuttal/NEW_EVIDENCE.md`.

### 0a.6 Engineering fixes that affect result validity

These matter because results produced **before** each fix are not trustworthy.
Where a stale result still exists on disk it is named.

| fix | commit | what was wrong | which results are affected |
|---|---|---|---|
| FastPLM tokenizer raised `KeyError` on `\|`, `#`, `J` and any other non-A-Z character; the suite caught the exception into an `Error` column and **still exited 0**, so a whole task could silently fail while the sweep reported success | `0e74e9b` (2026-07-28, OOV residue crash + silent benchmark failure), `37ca0ea` (2026-07-29, map *any* unknown token to `X`, not just A-Z) | affected `peptide_hla` (pipe-joined `HLA\|peptide` inputs) and `thermostability` (`#` in sequences) | The 2026-07-28 rows in `results/benchmarks/v3/esm2_35m_knn/bench__storage_models_ESM2-35M.csv` are error rows and must not be read. The 2026-07-29 rerun of that arm is clean. No other arm has error rows. **Any benchmark number dated before 2026-07-29 for `peptide_hla` or `thermostability` is invalid.** |
| `--cache_embeddings` is `store_true` and OFF by default, and the sweep never passed it — every arm recomputed embeddings from scratch (~15 min each) instead of the linear pass being a cheap classifier refit | `b93f2af` (2026-07-29) | a cost bug, not a correctness bug: the numbers were right, the sweep was ~4x slower than it should have been, which is why early arms were run piecemeal and appended | No result is invalidated. It explains why `protsent_old_linear` and `esm2_35m_linear` are dated 2026-07-29 00:09/00:34 while their kNN counterparts were re-run later |
| OpenBLAS thread oversubscription: on this 256-core box, uncapped probe fits abort with `corrupted size vs. prev_size` (SIGABRT, rc=134). `aav_flip` (50,430 x 22,186) reproduced it every time | `d05d96c` (2026-07-28) — `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=32` exported in `run_benchmarks_v3.sh` (and set before the numpy import in `build_comparison.py`) | whole arms died mid-sweep rather than producing wrong numbers | Nothing on disk is wrong because of it; arms that hit it produced no rows. Any future re-run must keep the caps |
| `bench_arm_status.py` originally counted error rows over the whole appended file, so an arm that had just succeeded on all 23 tasks was reported `FAILED` because the pre-fix error rows were still in the file | `46c2200` (2026-07-29) | reporting only | `logs/bench_v3/sweep2.log` still contains the false `FAILED: 2 task(s) errored despite rc=0` for `esm2_35m / knn` at 10:36. That arm is complete and clean; the current `bench_arm_status.py` confirms 23/23 |

### 0a.7 Superseded / do not cite

Numbers that appear in older drafts and must not be reused.

| do not cite | where it came from | what replaces it | why |
|---|---|---|---|
| MMseqs2 SCOPe-40 **R@1 0.3539 / MAP 0.1795** | `rebuttal/DRAFT_rebuttal.md` | **R@1 0.5029 / MAP 0.3100** at `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`, `results/benchmarks/mmseqs_baseline.json` | A default-sensitivity run. Publishing the weaker baseline while a stronger one is reproducible from the released repo is a self-inflicted integrity problem. Any MMseqs2 number must state its sensitivity setting |
| MMseqs2 **R@10 = R@30 = 0.3856** (exact equality) | `rebuttal/DRAFT_rebuttal.md` | **R@10 0.5637, R@30 0.5641** | Exact equality to four decimals indicates a truncated hit list, not a plateau |
| SCOPe AP correlation **Spearman −0.105 (p=1.6e-05)**, bin gains **+0.103 / +0.090**, bin counts **174 / 351 / 1,168**, identity median **0.893** | `results/benchmarks/scope_identity_correlation.json` (2026-07-28), and the Summary of this document before 2026-07-29 | **−0.1136 (p=2.8e-06)**, bin gains **+0.0915 / +0.0865**, counts **164 / 315 / 1,214**, median **0.9077** — `results/benchmarks/scope_identity_correlation_v1.json` | Computed against a superseded identity table. The unsuffixed `scope_identity_correlation.json` is kept only as provenance |
| SCOPe identity distribution **247 / 459 / 1,501** (unfiltered) and **281 / 475 / 1,451** (dc40), and "STRING supplies **1,399** of 2,207 above 0.7" | §2(a) of this document before 2026-07-29 | **237 / 416 / 1,554** and **270 / 430 / 1,507**; STRING **1,457** above 0.7 (AFDB 733 is unchanged) | Same superseded identity table. Recomputed on 2026-07-29 from `scope40_max_identity.parquet` columns `max_ident_overall` / `max_ident_dc_overall` |
| SCOPe eligible-query totals **AP 0.4216 → 0.5505 → 0.6448** used interchangeably with **0.4210 → 0.5509 → 0.6459** | §2(a) vs §3/§5 of this document | Both are correct *for their own code path*; do not mix them in one table | `scope_identity_correlation.py` and `protein_benchmark_suite.evaluate_retrieval` agree to four decimals on Recall@10 (0.92203 vs 0.9220) but differ in the third decimal on AP/MAP. The cross-validation claim in §5 is about **Recall@10 only** |
| Task count **24** for the model arms | early notes | **23** model-arm tasks; `mmseqs_baseline.json` has **24 entries**, of which 23 have a usable main metric | `rhla_enzyme_mutations` has an MMseqs2 row with `Spearman: null` and hit coverage 0.0, and is excluded from every model arm. Quote "23 tasks" for the sweep and "24 rows, 23 scorable" for the baseline JSON |
| "the n=92 strict-subset analysis" | `rebuttal/DRAFT_rebuttal.md` | the identity-stratified analysis over all 2,207 queries (§2a) | The 92-query subset has a ceiling of 57/92 = 0.620 that the draft does not state, so its R@30 of 0.500 reads against an implied 1.0 |

---

## 1. What was actually filtered

All three pretraining corpora were filtered against benchmark **test** sequences
with MMseqs2 `easy-search`, at **40% sequence identity / 80% coverage of the test
sequence** (`--min-seq-id 0.4 --cov-mode 1 -c 0.8 --alignment-mode 3 -e 1e-3`).

Orientation is deliberate: **pretraining corpus = query, benchmark test set =
target**. Each pretraining sequence only needs *any* hit to be dropped, so the
default `--max-seqs 300` prefilter cap is harmless. The reverse orientation would
silently truncate at 300 hits per test sequence and under-remove.
`--cov-mode 1` is coverage of the *target* (the test sequence), so a long AFDB
protein that merely *contains* a test-length domain is still caught.

| corpus | filtered against | rows before | rows after | removed | % |
|---|---|---:|---:|---:|---:|
| AFDB | `biomap-research/fold_prediction[test]` (3,244 seqs) | 135,404,259 | 126,301,607 | 9,102,652 | **6.72%** |
| Pfam | `biomap-research/fold_prediction[test]` (3,244 seqs) | 28,530,684 | 27,929,772 | 600,912 | **2.11%** |
| STRING | `Synthyra/bernett_gold_ppi[test]` (3,022 seqs) | 76,070,154 pairs | 71,891,417 pairs | 4,178,737 | **5.49%** |

Totals: **240,005,097 → 226,122,796 rows, 13,882,301 removed (5.78%)**.

Do not quote 5.78% as the size of the corpus the model actually saw — those are
two different reductions, and §4 applies a second one. See the "rows reaching
training" table in §4.

Additional detail: AFDB 117,549,800 unique sequences searched → 7,414,137 leaked;
clusters 819,790 → 817,282. Pfam 600,899 leaked unique sequences; families
29,395 → 29,368. STRING 14,567,625 unique sequences searched → 319,282 leaked; a
pair is dropped if **either** partner leaked.

Removing members can strand new singleton clusters (which then produce zero
pairs), so the singleton drop and the cluster-contiguity sort were re-applied
after filtering.

Artifacts: `/storage/users/ddofer/data/protsent-data-dc40/` — filtered parquets,
`decontam_report.json`, dataset card, and a `decontam/` audit subfolder holding
every hit with per-hit `fident`/`qcov`/`tcov` plus the leaked-sequence lists.

### Controls

| control | result |
|---|---|
| Negative — 1,000 random seqs from *filtered* AFDB vs fold test | **0 hits** |
| Negative — same, re-run with GPU prefilter (stricter; AFDB's production run used k-mer `-s 5.7` at 89.4% recall) | **0 hits** |
| Negative — 1,000 random seqs from *filtered* Pfam vs fold test | **0 hits** |
| Negative — 1,000 random seqs from *filtered* STRING vs bernett test | **0 hits** |
| Positive — fold test vs itself | **3,244 / 3,244** self-hit at `fident = 1.000` |
| Positive — bernett test vs itself | **3,022 / 3,022** self-hit at `fident = 1.000` |

The positive controls establish that the flags do what is claimed; the negative
controls establish that the leakage is actually gone from the shipped corpus.

---

## 2. SCOPe-40: why it is NOT used as a filter target

`tattabio/scope40_test` has **no train/test split**. The benchmark
(`benchmark_tasks.py:416-425`) uses `train_split="train"`, `test_split="train"`:
it is leave-one-out **self-retrieval** over the whole set, scored as family-level
Recall@10 with self-matches excluded.

Filtering AFDB/Pfam at 40% identity against *all* of SCOPe would therefore
remove essentially every structured domain from the pretraining corpus. That is
not a decontamination, it is corpus destruction, and the resulting model would
not be informative about anything.

The protocol used instead has three parts:

**(a) Identity vs. gain — measured.**

We computed, for every SCOPe-40 sequence, its maximum sequence identity to the
pretraining corpus. The result establishes the key point directly: **SCOPe-40 is
not separable from any comprehensive protein corpus.** Median max-identity is
**0.908**, and **no** SCOPe sequence falls below 20% identity. AFDB is predicted
structure over essentially all of UniProt, and SCOPe domains come from PDB
entries whose parent sequences are in UniProt — so every SCOPe domain has a close
neighbour by construction.

This is a property of the sequence universe, not of ProtSent. ESM-2 is pretrained
on UniRef50, the same universe, so the exposure is identical for every model in
the comparison. It also quantifies why SCOPe cannot be used as a filter target
(§2 opening): decontaminating against it would require removing sequences
matching essentially every query.

| max identity to pretraining | unfiltered | after dc40 |
|---|---:|---:|
| [0, 0.2) | **0 (0.00%)** | **0 (0.00%)** |
| [0.2, 0.4) | 237 (10.74%) | 270 (12.23%) |
| [0.4, 0.7) | 416 (18.85%) | 430 (19.48%) |
| [0.7, 1.0] | 1,554 (70.41%) | 1,507 (68.28%) |

Per corpus, STRING — not AFDB — supplies most of the near-identical matches
(1,457 of 2,207 above 0.7, vs AFDB's 733).

*Corrected 2026-07-29.* This table previously read 247/459/1,501 and
281/475/1,451, with STRING at 1,399 — those came from the superseded identity
table, the same one that produced the superseded bin counts noted at the end of
this subsection. The values above were recomputed from the current
`scope40_max_identity.parquet` (columns `max_ident_overall` and
`max_ident_dc_overall`, same bin edges, n = 2,207). See §0a.7.

The question the binning was meant to answer is answered directly instead. If the
advantage came from memorising pretraining neighbours, queries with a *closer*
pretraining neighbour would gain *more*. Measured per query on the **published**
ProtSent 35M (trained on the unfiltered corpus — the model under scrutiny),
against ESM-2 35M, over the 1,693 eligible queries
(`scope_identity_correlation.py`). Both models were measured against the same
identity table and the same 1,693 eligible queries, so the two columns are
directly comparable — ProtSent-v1 is the published model trained on the
**unfiltered** corpus, ProtSent-v2 the retrain on the **decontaminated** one
(`results/benchmarks/scope_identity_correlation_v1.json`, `..._v2.json`):

| metric | v1 mean gain | v1 Spearman r (p) | v2 mean gain | v2 Spearman r (p) |
|---|---:|---:|---:|---:|
| hit@1 | +0.0868 | −0.021 (0.39) | **+0.1855** | −0.012 (0.63) |
| hit@10 | +0.0898 | −0.038 (0.12) | **+0.1607** | −0.038 (0.12) |
| average precision | +0.1289 | **−0.114** (2.8e-06) | **+0.2232** | **−0.116** (1.6e-06) |

| identity bin | n | v1 gain hit@10 | v1 gain AP | v2 gain hit@10 | v2 gain AP |
|---|---:|---:|---:|---:|---:|
| [0.2, 0.4) | 164 | +0.0915 | +0.1856 | **+0.1524** | **+0.2859** |
| [0.4, 0.7) | 315 | +0.1016 | +0.1453 | +0.1810 | +0.2417 |
| [0.7, 1.0] | 1,214 | +0.0865 | +0.1169 | +0.1565 | +0.2099 |

**The gain is largest for the queries whose pretraining neighbour is most
distant**, and the only statistically significant correlation is *negative* — for
both models. Memorisation predicts the opposite sign. State it that way: the
effect does not track proximity to the pretraining corpus.

**Decontamination did not cost SCOPe performance; the retrained model is
stronger.** Over the same 1,693 eligible queries, hit@10 goes ESM-2 0.7614 →
v1 0.8512 → **v2 0.9220**, and average precision 0.4216 → 0.5505 → **0.6448**.
(These are eligible-query means, which is why they differ from the all-query
Recall@10/MAP in §3; the per-bin gains above sum to exactly these totals.)
Do not attribute that
improvement to decontamination alone: v2 also changed the effective batch size
(7×1024 vs the paper's 1×1024), dropped the synthetic hard negatives, and uses
proportional multi-dataset sampling over one epoch (§4). The defensible claim is
the narrower one the reviewers asked about — removing the 40%/80%-identity
overlap does not remove the gain.

An earlier version of this section reported v1 bin counts of 174/351/1,168 against
a superseded identity table. Both models above use the regenerated table; the
counts are 164/315/1,214.

Measurement note for anyone re-deriving these: with `-c 0.0`, MMseqs2 `fident` is
identity over the aligned region only, which for short local alignments is
meaningless (one Pfam hit scored `fident = 1.0` over a 12-residue alignment
covering 9% of the query). The identities above are `fident × tcov`, i.e.
identities over the full SCOPe sequence length. **Do not use raw `fident` from
`scope40_max_identity.parquet`.** The AFDB hits behind the table are substantive:
median alignment 126 aa, median tcov 0.97, median E-value 1.2e-37.

Pfam and AFDB identities come from cluster representatives (longest member per
cluster, which biases toward finding leakage). That can only *understate* the
maximum identity, never overstate it, so it cannot overturn a finding that
identities are already high. STRING was searched exhaustively over all 14.5M
unique sequences and carries no such caveat.

**(b) Both corpora reported.** SCOPe-40 is evaluated with the model trained on
the fold_prediction-filtered corpus **and** on the published model trained on the
unfiltered one, so the effect of decontamination on this task is directly
visible rather than asserted. The unfiltered-model numbers are in §2(a); the
filtered-model numbers landed 2026-07-29 and are in §5.

For calibration of how much this can move: decontamination shifts SCOPe-40's own
identity profile only slightly (median 0.908 → 0.895; the >70% bin 1,554 → 1,507
of 2,207), because the filter targeted `fold_prediction`, not SCOPe. A large
swing in SCOPe scores between the two models would therefore be surprising, and
that stability is itself the point: the benchmark is measuring representation
quality, not corpus overlap.

**(c) Baseline parity.** Every PLM baseline compared against (ESM2-35M and
friends) is pretrained on UniRef50, which contains all of SCOPe. The
contamination is common to every model in the table, so the **delta** is the
quantity being measured, and it is measured fairly.

**Precedent (verified).** ProtTucker (Heinzinger *et al.*, *NAR Genom. Bioinform.*
4(2):lqac043, 2022, doi:10.1093/nargab/lqac043) is the closest published analogue
and takes exactly posture (c):

- Data: CATH v4.3 sequence-unique set, CATH-S100 (123k domains). `test300`
  (300 proteins) and `val200` (200) were **randomly split off** from CATH-S100,
  constrained so that every homologous superfamily appears at most once *within*
  the held-out sets and every held-out protein carries an SSG annotation.
- Redundancy reduction: proteins sharing **>20% PIDE** with any val/test protein
  were removed **from the training set**, using MMseqs2 iterative profile search
  (`--num-iterations 3 -s 7.5 --cov-mode 0`). Result: `train66k`, lookup set
  `lookup69k`, query set `test219` (test300 minus queries with no same-H protein
  left in the lookup set).
- **The holdout is at the sequence-identity level, not the H level.** Training
  and lookup sets deliberately still contain the *same* homologous superfamilies
  as the queries — they must, since the task is transferring an H-level label
  from lookup to query by embedding kNN. So "no H-level leakage" is not what
  ProtTucker claims; it claims no >20%-PIDE sequence leakage.
- **They applied no decontamination to the underlying pLM's pretraining corpus.**
  ProtTucker is a 2-layer FNN (1024→256→tanh→128) on frozen ProtT5-XL-U50
  embeddings; ProtT5 was pretrained on BFD + UniRef50, which contains CATH in
  full. No statement addressing this overlap appears in the paper. *(Established
  by repeated search of the methods text; the journal/bioRxiv hosts are blocked
  by this cluster's firewall, so confirm by eye before quoting an absence in the
  response letter.)*
- Eval: embedding-based annotation transfer (EAT), Euclidean 1-NN from lookup to
  query, accuracy reported per CATH level (C/A/T/H); queries whose top hit came
  from a different level counted wrong; sequence-search baselines (MMseqs2,
  HMMER) scored as incorrect when no hit at E<10, and the headline claim is
  performance in the "midnight zone" (<20% PIDE).

The field's accepted standard is therefore identity-based decontamination of the
*supervised* split plus low-identity stratification — not removal of the
benchmark from the self-supervised pretraining corpus. §1 (40% PIDE filtering of
the pretraining corpora) is *stricter* than this precedent; §2a is the same
midnight-zone stratification.

---

## 3. MMseqs2-only baseline

Reviewer-relevant question: *how much of the structural performance is just
sequence similarity?* These numbers answer it by scoring the same tasks with
alignment instead of embeddings, under the **same metric definitions**.

Implementation: `mmseqs_baseline.py`. For retrieval it reproduces
`evaluate_retrieval()` (`protein_benchmark_suite.py:1863-1907`) exactly —
family-level Recall@K, self excluded — with cosine-NN rank replaced by MMseqs2
bitscore rank. For classification, per-class score = max bitscore over that
class's training sequences, giving a dense score vector so AUC stays comparable
rather than degenerating to hard 1-NN accuracy. For regression, 1-NN by bitscore.

Search flags: `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`.

**Queries with no hit are counted as failures, not dropped.** "Found no
homolog" is a real failure mode of sequence search and is exactly the gap an
embedding model should close; hiding it would flatter the baseline.

**`hit coverage`** (reported alongside every task) is the fraction of test
queries for which MMseqs2 returned *any* alignment at E<10. The remainder are
scored against a fallback carrying no information from the search — lowest-rank
class for classification, the training mean for regression, the empty label set
for multilabel. It is the column that separates "search ran and was right or
wrong" from "search found nothing and we scored a default", so a headline metric
should always be read next to it: at coverage 1.0 the metric is a real measure of
alignment; at coverage 0.0 it is a property of the fallback and means nothing.

### SCOPe-40: head-to-head, all measured with the same code

Every row below was produced on this machine by `protein_benchmark_suite.py`
(embeddings) or `mmseqs_baseline.py` (alignment), on the same 2,207-sequence
gallery, `--eval_split test`, self-matches excluded, no-hit queries scored as
failures. Raw values: `results/benchmarks/scope40_table.json`.

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10`) | **0.5029** | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent 35M (published) | 0.4490 | **0.6529** | **0.7100** | **0.4226** |

Eligible queries only (n = 1,693 of 2,207):

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 | **0.6556** | 0.7348 | 0.7354 | 0.4041 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent 35M | 0.5854 | **0.8511** | **0.9256** | **0.5509** |

**The evaluation path reproduces the submitted paper.** Measured here versus
Table 3 as submitted: ESM-2 35M R@1 0.3829 vs 0.3833, R@10 0.5840 vs 0.5841,
R@30 0.6398 vs 0.6402, MAP 0.3230 vs 0.3235; ProtSent 35M R@1 0.4490 vs 0.4495,
R@10 0.6529 vs 0.6529, R@30 0.7100 vs 0.7100, MAP 0.4226 vs 0.4225. The MAP
agreement to four decimals confirms the paper's MAP convention is the one now
implemented in `evaluate_retrieval()`: average precision over the full ranking,
averaged over all queries, unretrieved relevant items contributing zero.

**Recall@K on this benchmark is upper-bounded at 0.7671.** Only 1,693 of 2,207
queries (76.71%) have any non-self same-family protein in the gallery; the
remaining 514 are singleton families and are unachievable for any method. State
this in every caption carrying a SCOPe recall — ProtSent 35M's R@30 of 0.7100 is
**92.6% of the attainable maximum**, not 71% of 100%.

**A tuned MMseqs2 is the strongest top-1 method in the table.** At R@1 it beats
ESM-2 35M by 12.0 points and ProtSent 35M by 5.4. ProtSent leads at every deeper
cutoff (R@10 +8.9, R@30 +14.6 over MMseqs2) and on MAP (+11.3). The defensible
claim is therefore about **ranking depth, not top-1**.

The `-s 5.7` variant reproduces a much weaker alignment baseline (R@1 0.3847,
R@10 0.4259). Any MMseqs2 comparison must state its sensitivity: at default
settings the baseline looks far worse than it is, and publishing that number
while a stronger one is reproducible from this repo would be indefensible.

### Remote homology

| metric | MMseqs2 |
|---|---:|
| AUC (457 classes, macro OvR) | 0.6523 |
| Accuracy | 0.4365 |
| F1-macro | 0.2064 |
| hit coverage | 0.8893 |

359 of 3,244 test sequences find no homolog at any sensitivity and score 0.

### Full sweep — all 24 evaluable tasks

All rows on the **test** split (`--eval_split test`), sorted by task type.
Raw JSON: `results/benchmarks/mmseqs_baseline.json`.

| task | type | metric | MMseqs2 | secondary | hit cov. | n train/test |
|---|---|---|---:|---|---:|---|
| profet_np_sp_cleaved | binary | AUC | **0.9010** | Acc 0.9228; F1m 0.9127 | 0.9555 | 2,727/337 |
| signalp_binary | binary | AUC | 0.7961 | Acc 0.9345; F1m 0.8728 | 0.7695 | 16,606/4,152 |
| metal_ion_binding | binary | AUC | 0.7239 | Acc 0.7755; F1m 0.7755 | 0.9542 | 6,000/1,332 |
| binary_subcellular_localization | binary | AUC | 0.6834 | Acc 0.7176; F1m 0.7167 | 0.8954 | 5,184/1,749 |
| peptide_hla | binary | AUC | 0.6374 | Acc 0.7761; F1m 0.7761 | 1.0000 | 57,357/8,406 |
| material_production | binary | AUC | 0.5796 | Acc 0.6577; F1m 0.5964 | 0.9643 | 23,339/4,791 |
| solubility | binary | AUC | **0.4185** | Acc 0.4183; F1m 0.4116 | 0.9505 | 62,478/2,001 |
| antibiotic_resistance | multiclass | AUC | **0.9544** | Acc 0.9821; F1m 0.9199 | 0.9948 | 2,072/1,344 |
| temperature_stability | multiclass | AUC | 0.6853 | Acc 0.8552; F1m 0.8552 | 0.9717 | 283,057/73,205 |
| subcellular_loc | multiclass | AUC | 0.6828 | Acc 0.5304; F1m 0.3988 | 0.8974 | 6,622/1,842 |
| **remote_homology** | multiclass | AUC | **0.6523** | Acc 0.4365; F1m 0.2064 (457 cls) | 0.8893 | 12,312/3,244 |
| ec_classification | multilabel | F1_Macro | 0.7103 | F1_Micro 0.8777 | 0.9900 | 13,090/1,604 |
| go_mf | multilabel | F1_Macro | 0.5850 | F1_Micro 0.6406 | 0.9555 | 22,081/3,350 |
| beta_lactamase_peer | regression | Spearman | **0.8026** | MSE 0.0380 | 1.0000 | 4,158/520 |
| variant_effect | regression | Spearman | 0.7166 | MSE 1.0326 | 1.0000 | 6,289/1,745 |
| enzyme_catalytic_efficiency | regression | Spearman | 0.6322 | MSE 17.79 | 0.9941 | 13,470/1,684 |
| stability | regression | Spearman | 0.5817 | MSE 0.3503 | 1.0000 | 53,614/12,851 |
| optimal_ph | regression | Spearman | 0.5462 | MSE 1.0000 | 0.9868 | 7,124/1,971 |
| thermostability | regression | Spearman | 0.4799 | MSE 0.0448 | 0.9933 | 5,377/1,345 |
| aav_flip | regression | Spearman | 0.4024 | MSE 11.65 | 1.0000 | 22,246/50,432 |
| fluorescence | regression | Spearman | 0.3863 | MSE 2.0808 | 1.0000 | 21,446/27,217 |
| cloning_clf | regression | Spearman | 0.1707 | MSE 0.3969 | 0.9666 | 23,375/4,791 |
| rhla_enzyme_mutations | regression | Spearman | *n/a* | MSE 0.2045 | **0.0000** | 942/511 |
| **scope40_retrieval** | retrieval | Recall@10 | **0.5637** | R@1 0.5029; R@30 0.5641 | n/a | 2,207/2,207 |

Reading of these numbers:

- **`antibiotic_resistance` (0.954) and `beta_lactamase_peer` (0.803) are close to
  saturated by alignment alone.** These are the tasks where an embedding model has
  to justify its existence; a PLM that merely matches these adds nothing.
- **`solubility` at AUC 0.4185 is *below chance*** — the nearest homolog's
  solubility label is anti-correlated. Solubility is not conserved by homology, so
  a sequence-similarity prior actively misleads. Good evidence that some tasks
  cannot be solved by retrieval.
- **`rhla_enzyme_mutations` has 0% hit coverage** and no Spearman. Its `protein`
  column holds 6-residue mutation-site strings, not proteins; MMseqs2 reports
  `No k-mer could be extracted`. Structurally incompatible with alignment search
  — not a bug, and not a task where this baseline means anything.
- `peptide_hla` inputs are pipe-joined `HLA_pseudoseq|peptide` strings (~44 chars).
  MMseqs2 treats `|` as an unknown residue. It is the *same* string the model side
  embeds, so the comparison is fair, but it is not a biologically meaningful
  alignment.

Excluded by construction: `ppi_bernett` (pair input, not single-sequence), all
`proteingym_*` (mutant-vs-WT scoring), `chezod_disorder` (local data dir),
`cafa5` (size).

### Sensitivity variant

Documenting the speed/accuracy trade, since a cheaper search was considered:

| task | metric | `-s 7.5` | `-s 5.7` | Δ |
|---|---|---:|---:|---:|
| scope40_retrieval | Recall@10 | 0.5637 | 0.4259 | **−0.1378** |
| scope40_retrieval | Recall@1 | 0.5029 | 0.3847 | −0.1182 |
| remote_homology | AUC | 0.6523 | 0.6262 | −0.0261 |
| remote_homology | hit coverage | 0.8893 | 0.6233 | **−0.2660** |

MMseqs2 search time 4.09 s → 3.40 s (scope40) and 5.34 s → 3.62 s
(remote_homology). `-s 5.7` saves ~1.7 s on a 5 s search and costs 13.8 points of
Recall@10 — plainly the wrong trade at this scale. `-s 7.5` is used throughout.

### Split protocol — read before comparing any number here to a model number

Everything above is the **test** split. The benchmark suite defaults to
`--eval_split validation`, which falls back to 4-fold CV on *train* when a task
declares no validation split, so the default is **not** comparable to this table.
`run_benchmarks_v3.sh` therefore passes `-e test` for both models.

Two specifics worth stating in the paper:

- These 6 tasks have **no validation split** and would hit CV-on-train under the
  suite default: `metal_ion_binding`, `material_production`, `subcellular_loc`,
  `antibiotic_resistance`, `cloning_clf`, `thermostability`.
- **`thermostability` has no real test split either.** Under `-e test` the suite
  takes a seeded 80/20 split of train (`eval_strategy=test_random_split`). The
  MMseqs2 row uses that same seeded split, so the pairing is self-consistent, but
  it is not an official held-out set and should not be presented as one.
- `remote_homology`'s test split (3,244) is TAPE remote homology repackaged: the
  *pooled* concatenation of TAPE's three holdouts (718 fold + 1,254 superfamily +
  1,272 family), with no column marking which. Published work reports per-holdout
  top-1 accuracy on those three separately, so **our pooled 457-class macro AUC is
  not comparable to a published TAPE number** — say so rather than let a reviewer
  attempt the comparison.

---

## 4. ProtSent-v2-35M training configuration

Retrained on the filtered corpus. Every value below was measured on this
hardware (8× NVIDIA B300, sm_103), not assumed.

| setting | value | justification |
|---|---|---|
| model | Synthyra FastPLM ESM2-35M | |
| attention | `flash_attention_2` | 10.48 s/it vs sdpa 16.79 s/it in the real loop — sdpa is 60% slower |
| loss | `cached_mnrl`, mini-batch 256 | plain MNRL OOMs at ~260 GiB even at bs 256 under bf16 autocast |
| batch size | 1024 / device | CachedMNRL bounds memory by mini-batch, so this is free |
| gather across devices | **off** | 1024 in-batch negatives per rank already matches the paper; avoids allgather |
| dataset sampler | `proportional` | round-robin truncates to the smallest corpus |
| synthetic hard negatives | **off** | as specified |
| torch.compile | off | measured 8.87 vs 8.89 s/it — no effect |
| gradient checkpointing | off | not needed at 35M; it also forced `dataloader_num_workers=0` |
| Matryoshka dims | 64 / 128 / 256 (+ native 480) | |
| LR schedule | `cosine_with_min_lr`, 2e-4 peak, 1,000-step warmup, `num_cycles=3.0`, `min_lr_rate=0.05` | repository defaults, not overridden |
| steps | 4,850 (1 epoch, proportional) | pfam 759 + afdb 18,542 + string 14,648 batches |
| throughput | ~8.0 s/it → **~11 h** | measured over the first 1,900 steps of the live run |

**The LR schedule runs three cosine cycles and therefore ends at peak LR.** With
`num_cycles=3.0`, HuggingFace's `cosine_with_min_lr` lambda sweeps its cosine
argument through 6π across the post-warmup span. The schedule troughs at steps
1,642 / 2,925 / 4,208 (LR 1e-5, the 0.05 floor) and climbs back to the full 2e-4
at the final step. Verified against the live run: the lambda predicts 3.195e-5 at
step 1,500 where `trainer_state.json` logs 3.2248e-5, and the logged LR turns
upward after step 1,642 as predicted.

This is the repository default (`--lr_num_cycles 3.0`, `protein_pipeline.py`) and
was not overridden here, so the ±decontamination comparison is not confounded by
it. It does mean the final checkpoint is taken at the top of a cycle rather than
annealed, so the last-step checkpoint should not be presented as a converged
optimum without also reporting a near-trough checkpoint.

**FlashAttention-3 is not usable on this hardware.** The pinned
`kernels-community/flash-attn3` build contains `sm_80, sm_90a` only and fails on
sm_103 with `CUDA error: no kernel image is available for execution on the
device`. FA3 is Hopper-only; Blackwell needs FA4, which FastPLM's
`AttentionBackend` enum does not contain. FA2 is used instead.

### Data budget caveat — state this if asked

**Rows reaching training.** Two independent reductions apply, and conflating them
misstates the corpus by 57M rows. Counts verified directly from the parquet
metadata and from the running job's log.

| corpus | source rows | after decontamination | rows fed to training |
|---|---:|---:|---:|
| Pfam | 28,530,684 | 27,929,772 | 27,929,772 |
| AFDB | 135,404,259 | 126,301,607 | 126,301,607 |
| STRING | 76,070,154 | 71,891,417 | **15,000,000** |
| **total** | **240,005,097** | **226,122,796** (−5.78%) | **169,231,379** (70.51% of source) |

Reduction 1 is the decontamination itself (−5.78%, §1). Reduction 2 is a
deliberate STRING subsample (71,891,417 → 15,000,000, seed 42) taken to fit a
~12 h compute budget; it is a budget decision, not a leakage control, and must
not be presented as one.

**Pairs generated from those rows.** `--max_pairs_per_cluster = 8` samples 8
sequences per cluster and emits all C(8,2) = 28 pairs, so pair count is not row
count. From the run log: AFDB 18,987,468 + Pfam 777,306 + STRING 15,000,000 =
**34,764,774 pairs**, giving 33,949 global batches and **4,850 optimizer steps**
at batch 1024 per rank across 7 ranks (one epoch).

AFDB and Pfam clusters are **all** visited — a substantial improvement over the
earlier round-robin run, which exhausted its pair budget within the first ~2% of
the group-sorted corpus and therefore only ever saw the lowest-sorted clusters.
But the paper cannot claim "trained on the entire filtered corpus": STRING
contributes 20.9% of its available filtered pairs.

### Fixes made along the way

- `FastPLMESM2Wrapper` requested `output_hidden_states=True` and ran the
  `lm_head` on every forward, neither of which is used for embedding
  (`model_utils.py:539`).
- `save_total_limit` was hardcoded to 1 (`protein_pipeline.py:2302`).
- `--multi_dataset_sampler` left at `round_robin` truncates to the smallest
  dataset.

---

## 5. ProtSent-V2-35M results — measured, cross-checked

Training finished 2026-07-29 08:03: 4,850 steps, `train_runtime` 39,170 s
(10 h 53 m), 887.5 samples/s, clean exit. Model at
`models/protsent_esm2_35m_v3/final` (the `v3` string is an internal RUN_NAME;
the paper name is ProtSent-V2-35M).

### The corpus it trained on contains zero flagged sequences

Not asserted from the filtering job's own report, which records only intent.
`verify_training_corpus.py` re-reads the exact parquet files the training log
shows being opened and semi-joins them against the recorded removal lists:

| training file | rows | sequences flagged by MMseqs2 that survived |
|---|---|---|
| `pfam_sorted.parquet` | 27,929,772 | **0** |
| `afdb_sorted.parquet` | 126,301,607 | **0** |
| `stringdb_train_15M.parquet` | 15,000,000 | **0** (both `seq1` and `seq2`) |

The removal lists hold 600,899 / 7,414,137 / 319,282 sequences respectively.
The STRING row is the load-bearing one: `stringdb_train_15M.parquet` was
subsampled two days after the filtering run and nothing on disk records which
parent it came from, so membership is the only way to distinguish a subsample of
the filtered file from a subsample of the original.

Independently, the row arithmetic closes: 27,929,772 + 126,301,607 + 15,000,000
= 169,231,379, exactly the `total=` the trainer logged. The log also confirms
`hard_neg=False`, `sampler=proportional`, `effective_batch=7168`.

### Structural tasks, test split, all four arms

`eligible_*` restricts to the 1,693 of 2,207 SCOPe queries that have at least one
same-family neighbour; the unrestricted figure counts the other 514 as zero.

| probe | metric | ESM-2 35M | ProtSent-V1 | **ProtSent-V2** | V2 ckpt-4000 |
|---|---|---|---|---|---|
| kNN | SCOPe-40 eligible Recall@1 | 0.4991 | 0.5854 | **0.6852** | 0.6775 |
| kNN | SCOPe-40 eligible Recall@10 | 0.7614 | 0.8512 | **0.9220** | 0.9173 |
| kNN | SCOPe-40 eligible MAP | 0.4210 | 0.5509 | **0.6459** | 0.6447 |
| kNN | Remote Homology accuracy | 0.5835 | 0.6587 | **0.6668** | 0.6655 |
| linear | Remote Homology accuracy | 0.6868 | 0.6899 | **0.7016** | 0.6988 |

**Remote homology is the task the corpus was actually filtered against**, and it
improved rather than degraded. That is the direct answer to the reviewers'
question: removing every pretraining sequence within 40% identity / 80% coverage
of the remote-homology test set did not cost remote-homology performance.

**The SCOPe number is cross-validated.** Two independent implementations —
`protein_benchmark_suite.evaluate_retrieval` and
`scope_identity_correlation.compute_per_query` — agree to four decimals
(V2 0.92203 vs 0.9220; V1 0.85115 vs 0.8512; ESM-2 0.76137 vs 0.7614).

### Identity-vs-gain, both models on the same identity table

Gain over ESM-2 35M, per SCOPe query, binned by maximum identity to the
pretraining corpus. Both rows use the same identity parquet, the same bins and
the same eligible set, so V1 and V2 are directly comparable.

| bin | n | V1 ΔRecall@10 | V2 ΔRecall@10 | V1 ΔMAP | V2 ΔMAP |
|---|---|---|---|---|---|
| [0.2, 0.4) | 164 | +0.0915 | **+0.1524** | +0.1856 | **+0.2859** |
| [0.4, 0.7) | 315 | +0.1016 | +0.1810 | +0.1453 | +0.2417 |
| [0.7, 1.0] | 1,214 | +0.0865 | +0.1565 | +0.1169 | +0.2099 |

Per-query Spearman between max identity and gain stays null-to-negative for both
models: Recall@10 −0.038 (V1) / −0.038 (V2), MAP −0.114 / −0.116 (p < 3e-6).
The advantage does not grow with proximity to pretraining data — it shrinks
slightly. Memorization predicts the opposite sign.

### Caveat to state plainly

**V2 differs from V1 in more than decontamination**: 7x1024 effective batch vs
1x1024, no synthetic hard negatives, proportional sampling, one epoch. The
improvement therefore cannot be attributed to filtering alone. The claim the
data does support, and the one the reviewers asked about, is the weaker and
sufficient one: **removing the contaminated pairs did not cost performance on
the structural tasks.**

### The LR-schedule caveat is empirically negligible

§4 notes the 3-cycle cosine ends at peak LR. The near-trough checkpoint-4000
(LR 5.5e-5) and the peak-LR final checkpoint differ by 0.005-0.008 on every
structural metric above, so quoting the final model is safe.

### Aggregate across all 23 tasks

Against ESM-2 35M over the 20 tasks comparable in both arms
(`results/benchmarks/COMPARISON.md`):

| probe | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| kNN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median −0.0139 | 2 / 7 / 11, median −0.0107 |

**The probe decides the headline.** Both ProtSent models beat ESM-2 under kNN
and lose under a linear probe. The structural-retrieval advantage survives both;
the general-purpose claim does not. Any "ProtSent > ESM-2" sentence must name
the probe.

## 5a. Still pending

- [ ] MMseqs2 + MAP on the remaining benchmarks compared against published
      literature values (needs the target papers named)
- [ ] Multi-seed variability sweep (5 seeds x 8 tasks x 3 arms, kNN). Started
      2026-07-29 12:54, still on arm 1 of 3 at 13:11, **no output file yet**.
      See §0.5 for its exact state and `run_seed_variability.sh` to resume it.

---

## 6. Corrections the draft rebuttal needs (grounded, from this repo)

Each item below is a factual error or omission in `rebuttal/DRAFT_rebuttal.md`,
with the source that settles it. These are not stylistic preferences.

1. **The MMseqs2 baseline in the draft (R@1 0.3539, MAP 0.1795) is a
   default-sensitivity run.** §3 above reproduces R@1 0.5029 / MAP 0.3100 at
   `-s 7.5`, and `-s 5.7` gives R@1 0.3847 — bracketing the draft's number.
   Publishing the weaker figure while the stronger one is reproducible from
   `results/benchmarks/mmseqs_baseline.json` in the released repo is a
   self-inflicted integrity problem. Publish 0.5029 with flags stated.

2. **R@10 = R@30 = 0.3856 exactly** in the draft's table. Exact equality to four
   decimals indicates a truncated hit list, not a plateau. The measured run gives
   0.5637 vs 0.5641 — near-equal but distinct.

3. **The remote-homology test split is not hierarchy-disjoint.** The draft claims
   it is. It is TAPE remote homology repackaged: the pooled concatenation of
   TAPE's three holdouts (718 fold + 1,254 superfamily + 1,272 family = 3,244),
   with no column marking which. Two thirds is not fold-disjoint. Rely on
   corpus-level decontamination instead, and note that the pooled 457-class macro
   AUC is not comparable to published per-holdout top-1 accuracies.

4. **The PPI decontamination description contradicts the released code.** The
   draft describes `easy-linclust` at 50% identity with cluster-level removal.
   `data_prep.py` uses `easy-search` (STRING as query, Bernett test as target) at
   `--decontam_min_seq_id` (default 0.4), `--cov-mode 1 -c 0.8`, removing hit
   query IDs, not clusters — and its own docstring states `easy-search` was
   chosen deliberately because linclust loses sensitivity below ~50% identity.
   Describe what the code does. The completed 40% pass is the stronger answer
   anyway: 4,178,737 STRING pairs (5.49%) and 319,282 unique sequences removed,
   with 0-hit negative and 3,022/3,022 positive controls.

5. **"100,000 sequences" has a mechanical explanation.** It is the evaluator's
   `max_samples` cap echoed into the results table (visible in every benchmark
   CSV as `Samples 100000`), applied to a 2,207-row dataset. Saying so converts
   an apparent 45x error in reported N into a logging artifact.

6. **Task count is 23 in the draft and 24 here.** `mmseqs_baseline.json` has 24
   rows. Pick one and state the exclusions (`ppi_bernett` pair-input,
   `proteingym_*`, `chezod_disorder`, `cafa5`, `rhla_enzyme_mutations`).

7. **Use the decontamination that is already finished.** §1 above — all three
   corpora filtered at 40%/80% with negative controls at 0 hits and positive
   controls at 3,244/3,244 and 3,022/3,022 — is complete, auditable, and stricter
   than the ProtTucker precedent verified in §2. The draft concedes the leakage
   point instead of citing this work.

8. **Keep one story about R@1 across all three responses — and note it changed
   with V2.** The measured table in §3 shows a tuned alignment baseline beating
   the *submitted* model at top-1 (0.5029 vs 0.4490). It does **not** beat the
   decontaminated retrained model: ProtSent-V2 reaches R@1 0.5256, R@10 0.7073,
   MAP 0.4955, i.e. it leads MMseqs2 at every cutoff. So the honest framing is
   two-part: for the paper as submitted, the defensible claim is ranking depth
   (R@10/R@30/MAP), not top-1; for V2, the top-1 win is real and measured. Never
   retro-claim the top-1 win for the submitted model, and never concede it for
   V2. Asserting different things to different reviewers is what must be avoided.

9. **The n=92 strict-subset analysis has a ceiling of 57/92 = 0.620**, which the
   draft does not state, so its R@30 of 0.500 reads against an implied 1.0.
   The identity-stratified analysis (§2a) retains all 2,207 queries and has the
   statistical power the 92-query subset lacks.
- [x] MMseqs2 baseline across all 24 evaluable benchmarks + sensitivity variant (§3)
- [x] ProtSent-v2-35M benchmark results: kNN probe and linear probe, vs ESM2-35M,
      both with `-e test` — done 2026-07-29, four arms x two probes, indexed in §0.1
- [ ] **Additional metrics to match the comparison papers.** The table in §3
      reports each task's declared `main_metric` plus whatever the evaluator
      emits. Papers we are compared against may report different quantities
      (e.g. per-holdout top-1 accuracy for TAPE remote homology; Foldseek/TM-Vec
      style "sensitivity up to the first false positive" for SCOPe rather than
      Recall@K; alignment/embedding-geometry diagnostics such as
      alignment-vs-uniformity, embedding anisotropy, or TM-score correlation).
      Decide which are actually needed for the response letter before adding
      them — each one is a separate evaluator, and the ones above are not
      currently computed by either `mmseqs_baseline.py` or the benchmark suite.
- [x] Verify the ProtTucker precedent citation (§2) — done, see §2
- [ ] Optional: run ProtTucker as a second baseline. **Blocked, not recommended.**
      The `ProtTucker_ProtT5.pt` checkpoint is served only from `rostlab.org` and
      `zenodo.org`, both unreachable from this cluster (NETWORK_WHITELIST.md), and
      is not mirrored on HF. Even with the weights, embedding both task sets with
      ProtT5-XL (1.2B-param encoder, 3.07M residues) is ~4-8 h of CPU. No
      published SCOPe-40 or fold-prediction number is protocol-comparable to ours
      (see below), so there is nothing to cite in its place either.
