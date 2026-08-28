# ProtSent — orientation

Contrastive post-training (SentenceTransformers + MNRL) of ESM-2 protein language models.
Read [RUNS.md](RUNS.md) before touching any number: lists every trained model, training data, results location.

## Environment

Python 3.14 in `.venv`, managed by `uv`. **Always** `uv run --no-sync python ...` — scripts assume it, `pip` not installed in venv.

Set `HF_HOME=/storage/models/hf_home`. Most entry points also want BLAS thread caps (`OPENBLAS_NUM_THREADS=32` and friends): uncapped, sklearn probe fits abort with `corrupted size vs. prev_size` on larger tasks. `run_benchmarks_*.sh` export all of it.

## Which number is authoritative

Several quantities measured more than once. Wrong one produced published false claim before.

- **HMMER on SCOPe-40**: use `results/benchmarks/hmmer_maxsens.json`, **filters-off** run (eligible R@1 0.7525). Default-filter run weaker; quoting it produced claim that ProtSent beats profile search at top-1 — false at both scales.
- **MMseqs2 on SCOPe-40**: use 2026-07-31 scoring in `results/benchmarks/scope40_bootstrap_ci_150m.json` (eligible 0.6556 / 0.7401 / 0.4098). `results/benchmarks/mmseqs_baseline.json` is earlier, different scoring — hit30 differs by 0.021. `results/benchmarks/mmseqs_baseline.json` still correct for other 22 tasks.
- **SCOPe-40 retrieval**: report **eligible-query** figures (n=1,693 of 2,207). Unrestricted values exactly `eligible x 1693/2207` — query with no same-family protein in gallery cannot succeed at any k.
- **Point estimate vs bootstrap mean**: CSVs hold point estimates, `scope40_bootstrap_ci*.json` files hold resample means. Differ in fourth decimal. Never mix within one table row.
- **23-task suite vs 20-task tally**: three tasks (Antibiotic Resistance, Remote Homology, Temperature Stability) have undefined one-vs-rest AUC; published tallies cover remaining 20, those three reported separately. `ism_comparison.py` can score all 23 via Accuracy fallback, but shifts arms asymmetrically — see caveat in RUNS.md before using.

## Claims the evidence does not support

- **No inferential claim from the 23-task aggregate.** Sign test resolves almost none of win/tie/loss records; only significant ones are ProtSent's *losses* under linear probe. Comparative adjectives over these tallies ("milder trade", "smaller cost") not supportable; `rebuttal/FINAL_rebuttal.md` states publicly no such claim drawn.
- **ProtSent vs ESM-C / ISM-C is not controlled.** Crosses both model family and scale, and raw mean-pooled ESM-C weak at retrieval to begin with — below ESM-2 35M. Does not establish contrastive post-training beats structure distillation.
- **Alignment leads at top-1**, both scales. Supportable retrieval claim is ranking depth and MAP.

## Verifying anything

- Benchmark suite **catches per-task exceptions, writes them to `Error` column, still exits 0**. Sweep can report success with every row a failure. Always check with `bench_arm_status.py <csv> <n_tasks>`.
- Suite **appends** to stable per-model CSV and dedups keeping newest, so rerun leaves stale rows behind. Row count not a completeness test.
- Most analysis scripts take `--selfcheck`. Run after editing one.
- `--eval_split test` required for comparability. Default is `validation`, which silently falls back to 4-fold CV on train for tasks with no validation split — different protocol, not different seed.
- `scope40_retrieval` opt-in, excluded from suite defaults. Must be named explicitly on `-t` or silently does not run.

## Layout

| path | what |
|---|---|
| `protein_pipeline.py train` | training entry point; `train_esm2_{35m,150m}.sh` wrap it |
| `protein_benchmark_suite.py` | the 23-task benchmark; `run_benchmarks_*.sh` wrap it |
| `model_utils.py` | model detection, loading, compatibility patches |
| `ism_comparison.py`, `build_comparison.py` | join finished results into tables; run no models |
| `bootstrap_ci.py` | paired bootstrap CIs on SCOPe-40 retrieval |
| `results/benchmarks/` | all measured numbers, READMEs per subdirectory |
| `private/` | **gitignored**, must stay that way |

Model loading dispatches on `detect_model_type` in `model_utils.py`, routing ESM-2, FastPLM, ESM++/ESM-C, AMPLIFY, DPLM2, E1 down separate branches. New backbone usually needs no code — check which branch it lands in before writing any.
