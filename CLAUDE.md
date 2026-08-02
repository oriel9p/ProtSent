# ProtSent — orientation

Contrastive post-training (SentenceTransformers + MNRL) of ESM-2 protein language models.
Read [RUNS.md](RUNS.md) before touching any number: it lists every trained model, what it
was trained on, and where its results live.

## Environment

Python 3.14 in `.venv`, managed by `uv`. **Always** `uv run --no-sync python ...` — the
scripts assume it, and `pip` is not installed inside the venv.

Set `HF_HOME=/storage/models/hf_home`. Most entry points also want the BLAS thread caps
(`OPENBLAS_NUM_THREADS=32` and friends): uncapped, the sklearn probe fits abort with
`corrupted size vs. prev_size` on the larger tasks. `run_benchmarks_*.sh` export all of it.

## Which number is authoritative

Several quantities were measured more than once. Using the wrong one has produced a
published false claim before.

- **HMMER on SCOPe-40**: use `results/benchmarks/hmmer_maxsens.json`, the **filters-off** run (eligible R@1
  0.7525). The default-filter run is weaker; quoting it once produced a claim that ProtSent
  beats profile search at top-1, which it does not, at either scale.
- **MMseqs2 on SCOPe-40**: use the 2026-07-31 scoring in `results/benchmarks/scope40_bootstrap_ci_150m.json`
  (eligible 0.6556 / 0.7401 / 0.4098). `results/benchmarks/mmseqs_baseline.json` is an earlier, different
  scoring — hit30 differs by 0.021. `results/benchmarks/mmseqs_baseline.json` is still correct for the other
  22 tasks.
- **SCOPe-40 retrieval**: report the **eligible-query** figures (n=1,693 of 2,207).
  Unrestricted values are exactly `eligible x 1693/2207`, since a query with no same-family
  protein in the gallery cannot succeed at any k.
- **Point estimate vs bootstrap mean**: the CSVs hold point estimates, the
  `scope40_bootstrap_ci*.json` files hold resample means. They differ in the fourth decimal.
  Do not mix them within one table row.
- **23-task suite vs 20-task tally**: three tasks (Antibiotic Resistance, Remote Homology,
  Temperature Stability) have an undefined one-vs-rest AUC, and the published tallies are
  over the remaining 20 with those three reported separately. `ism_comparison.py` can score
  all 23 via an Accuracy fallback, but that shifts arms asymmetrically — see the caveat in
  RUNS.md before using it.

## Claims the evidence does not support

- **No inferential claim from the 23-task aggregate.** A sign test resolves almost none of
  the win/tie/loss records; the only significant ones are ProtSent's *losses* under a linear
  probe. Comparative adjectives over these tallies ("milder trade", "smaller cost") are not
  supportable, and `rebuttal/FINAL_rebuttal.md` states publicly that no such claim is drawn.
- **ProtSent vs ESM-C / ISM-C is not controlled.** It crosses both model family and scale,
  and raw mean-pooled ESM-C is weak at retrieval to begin with — below ESM-2 35M. It does
  not establish that contrastive post-training beats structure distillation.
- **Alignment leads at top-1**, at both scales. The supportable retrieval claim is ranking
  depth and MAP.

## Verifying anything

- The benchmark suite **catches per-task exceptions, writes them to an `Error` column, and
  still exits 0**. A sweep can report success with every row a failure. Always check with
  `bench_arm_status.py <csv> <n_tasks>`.
- The suite **appends** to a stable per-model CSV and dedups keeping the newest, so a rerun
  leaves stale rows behind. Row count is not a completeness test.
- Most analysis scripts take `--selfcheck`. Run it after editing one.
- `--eval_split test` is required for comparability. The default is `validation`, which
  silently falls back to 4-fold CV on train for tasks with no validation split — a different
  protocol, not a different seed.
- `scope40_retrieval` is opt-in and excluded from the suite defaults. It must be named
  explicitly on `-t` or it silently does not run.

## Layout

| path | what |
|---|---|
| `protein_pipeline.py train` | training entry point; `train_esm2_{35m,150m}.sh` wrap it |
| `protein_benchmark_suite.py` | the 23-task benchmark; `run_benchmarks_*.sh` wrap it |
| `model_utils.py` | model detection, loading, and the compatibility patches |
| `ism_comparison.py`, `build_comparison.py` | join finished results into tables; run no models |
| `bootstrap_ci.py` | paired bootstrap CIs on SCOPe-40 retrieval |
| `results/benchmarks/` | all measured numbers, with READMEs per subdirectory |
| `private/` | **gitignored**, and must stay that way |

Model loading dispatches on `detect_model_type` in `model_utils.py`, which routes ESM-2,
FastPLM, ESM++/ESM-C, AMPLIFY, DPLM2 and E1 down separate branches. A new backbone usually
needs no code — check which branch it lands in before writing any.
