# ProtSent: Protein Sentence Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2605.06830-b31b1b.svg)](https://arxiv.org/abs/2605.06830)

Code for ["ProtSent: Protein Sentence Transformers"](https://arxiv.org/abs/2605.06830).

ProtSent applies contrastive fine-tuning (SentenceTransformers + MNRL) to ESM-2 protein language models, producing fixed-length embeddings where biological similarity maps to embedding proximity. Training uses five complementary data sources: Pfam families, structurally derived hard negatives, AlphaFold DB structural pairs, StringDB interactions, and deep mutational scanning fitness data.

**Models:** [ProtSent-ESM2-35M](https://huggingface.co/oriel9p/protsent-esm2-35M) | [ProtSent-ESM2-150M](https://huggingface.co/oriel9p/protsent-esm2-150M) | [ProtSent-V2-35M](https://huggingface.co/GrimSqueaker/ProtSent-V2-35M) | [ProtSent-V2-150M](https://huggingface.co/GrimSqueaker/ProtSent-V2-150M)

## Where the results are

Start with **[RUNS.md](RUNS.md)** — one section per trained model, what it was trained on,
where its numbers live, and which figures are authoritative when two exist.

| you want | go to |
|---|---|
| Which model is which, and its config | [RUNS.md](RUNS.md), top table |
| ProtSent-V2 vs the ESM-C / ISM-C baselines, summarised | [RUNS.md](RUNS.md) § `ISM-C-300M`, "TLDR" |
| Per-task numbers for every arm, both probes | [results/benchmarks/ISM_COMPARISON.md](results/benchmarks/ISM_COMPARISON.md) |
| SCOPe-40 retrieval with confidence intervals | `results/benchmarks/scope40_bootstrap_ci*.json` |
| Raw per-arm CSVs | `results/benchmarks/{v3,v2_150m,ism}/` — see the READMEs there |
| What was said to reviewers | [rebuttal/FINAL_rebuttal.md](rebuttal/FINAL_rebuttal.md) |

Reproduce a comparison without retraining anything:

```bash
uv run --no-sync python ism_comparison.py    # -> results/benchmarks/ISM_COMPARISON.md
uv run --no-sync python build_comparison.py  # -> results/benchmarks/COMPARISON.md
```

## Setup

Requires Python 3.13+, NVIDIA GPU with CUDA, and `uv` (recommended) or `pip`.

```bash
# With uv (recommended)
uv venv .venv && source .venv/bin/activate && uv sync

# Or with pip
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Verify the install:

```bash
python protein_pipeline.py train --help
python protein_benchmark_suite.py --help
```

## Data preparation

Each training dataset is prepared separately. Run only the ones you need.

```bash
# Core dataset (start here)
python data_prep.py --dataset pfam

# Additional datasets
python data_prep.py --dataset afdb --limit_gb 5
python data_prep.py --dataset stringdb        # requires MMseqs2
python data_prep.py --dataset dms
python data_prep.py --dataset pfam_hard_negatives
```

Quick smoke test with Pfam only:

```bash
python data_prep.py --dataset pfam --fast
```

## Training

Smoke test (single GPU, small data):

```bash
CUDA_VISIBLE_DEVICES=0 python protein_pipeline.py train \
  --files data/pfam_sorted.parquet \
  --model facebook/esm2_t12_35M_UR50D \
  --loss_mode cached_mnrl \
  --fast --no_resume --run_name smoke_test
```

Full multi-dataset training (reproduces paper):

```bash
python protein_pipeline.py train \
  --files data/pfam_sorted.parquet data/afdb_sorted.parquet \
         data/stringdb/stringdb_train.parquet data/dms_cosent.parquet \
  --hard_negatives data/pfam_hard_negatives.parquet \
  --model facebook/esm2_t12_35M_UR50D \
  --loss_mode cached_mnrl \
  --batch_size 1024 \
  --run_name protsent_35m
```

For ESM-2 150M, replace the model name with `facebook/esm2_t30_150M_UR50D`.

Checkpoints are saved to `models/<run_name>/checkpoint-*/` with the final export at `models/<run_name>/final`.

## Ablation experiments

The ablation runner trains leave-one-out configurations:

```bash
python run_ablation_v2.py --config no_pfam
python run_ablation_v2.py --config no_afdb
python run_ablation_v2.py --config no_stringdb
python run_ablation_v2.py --config no_hardneg
python run_ablation_v2.py --config no_dms
python run_ablation_v2.py --config proportional
```

## Benchmarking

Evaluate any model on 23+ downstream tasks:

```bash
# Fast mode (core tasks, capped samples)
python protein_benchmark_suite.py -m facebook/esm2_t12_35M_UR50D

# Full suite (all tasks)
python protein_benchmark_suite.py -m models/protsent_35m/final --no-fast

# KNN probe (as used in paper)
python protein_benchmark_suite.py -m models/protsent_35m/final --probe_type knn --no-fast

# SCOPe-40 retrieval (opt-in)
python protein_benchmark_suite.py -m models/protsent_35m/final -t scope40_retrieval

# Compare two models
python protein_benchmark_suite.py --compare \
  --compare_model1 results/baseline_bench \
  --compare_model2 results/protsent_bench
```

### Benchmark tasks (23 used in paper)

| Category | Tasks |
|---|---|
| Binary (8) | PPI, solubility, peptide-HLA, metal ion binding, signal peptide, neuropeptide, binary subcellular localization, material production |
| Multiclass (4) | Remote homology (fold), subcellular localization, antibiotic resistance, temperature stability |
| Multilabel (1) | EC classification |
| Regression (10) | Variant effect (GB1), fluorescence, stability, thermostability, optimal pH, enzyme catalytic efficiency, cloning, beta-lactamase, AAV fitness, RhlA mutations |

## Few-shot evaluation

```bash
python protein_benchmark_suite.py \
  -m models/protsent_35m/final \
  --probe_type knn \
  --max_samples 100 \
  -t remote_homology fluorescence ec_classification
```

## Late interaction (ColBERT-style MaxSim)

An experimental second scoring mode: instead of comparing two mean-pooled vectors,
keep one vector per residue and score a pair by MaxSim — for each query residue take
its best match in the document, then sum. Built on Sentence Transformers v6
(`MultiVectorEncoder`), sharing the same backbone and the same dc40 pair corpus as
ordinary ProtSent training, so any late-trained model also exports a normal
mean-pooled "dense view" that every existing benchmark can consume unchanged.

```bash
# Continue an existing ProtSent model with a residue-level contrastive objective
python train_late_interaction.py \
  --model GrimSqueaker/ProtSent-V2-35M \
  --files /storage/users/ddofer/data/protsent-data-dc40/{pfam_sorted,afdb_sorted,stringdb_train_15M}.parquet \
  --output_dir models/late_interaction/protsent_late \
  --proj_dim 128 --max_steps 18219

# SCOPe-40 all-vs-all: dense cosine vs zero-shot MaxSim vs trained MaxSim
python late_interaction_eval.py scope --models models/late_interaction/protsent_late/late

# The long-run queue (survives disconnect; idempotent; --resume)
QUEUES="p q" ./run_experiment_queue.sh start
./run_experiment_queue.sh status
```

Outputs land in `--output_dir` as `late/` (the multi-vector model), `dense_view/`
(its mean-pooled equivalent), `step0/` (both, before training, as a parity control),
plus `runtime.json` and `train_log.csv`.

**Defaults worth knowing.** `--proj_dim 128` (measured better than 64-D and better
than no projection at all); `--attn_implementation auto` tries flash attention and
falls back to sdpa, logging which it got and recording it in `runtime.json`.
Flash needs `0.15.2 <= kernels < 0.16` — outside that window transformers rejects the
kernel and the run silently trains on sdpa, so check the logged backend after any
environment change. `--compile` is off by default (it helps short Pfam pairs and
hurts longer STRING ones). FastPLM checkpoints (`Synthyra/*`) always take sdpa,
because the flash path loads a plain `EsmModel` instead of the FastPLM wrapper.

Results, method notes and the caveats that go with them:
[`results/late_interaction/pilot_35m/SUMMARY.md`](results/late_interaction/pilot_35m/SUMMARY.md).

## Tests

```bash
pytest tests/ -m "not slow"
```

## Project structure

```
protein_pipeline.py          # Main training entry point
data_prep.py                 # Dataset download and preprocessing
protein_benchmark_suite.py   # Benchmark evaluation suite
run_ablation_v2.py           # Ablation experiment runner
model_utils.py               # Model compatibility utilities
benchmark_tasks.py           # Task definitions and configs
benchmark_utils.py           # Shared benchmark helpers
attention_pooling.py         # Custom pooling modules
benchmark_comparison.py      # Model comparison utilities
benchmark_relative_plot.py   # Relative performance plots
benchmark_ablation_report.py # Ablation summary generation
benchmark_plotting.py        # Shared plot helpers
umap_visualization.py        # UMAP embedding visualization
ablation_optuna_search.py    # Hyperparameter search
static_guide.py              # Static embedding guide helpers
late_interaction.py          # ColBERT/MaxSim encoder, scoring and SCOPe metrics
train_late_interaction.py    # Late-interaction contrastive training
late_interaction_eval.py     # SCOPe / CATH / few-shot evaluation for late models
run_experiment_queue.sh      # Resumable, disconnect-proof training queue
run_late_bench.sh            # Pooled benchmarks over each arm's dense view
snapshot_checkpoints.sh      # Keeps weights-only checkpoint copies for the curve
pyproject.toml               # Dependencies and project config
tests/                       # Unit and integration tests
data/                        # Small metadata files (large data is generated)
```

## Citation

```bibtex
@article{ofer2025protsent,
  title={ProtSent: Protein Sentence Transformers},
  author={Ofer, Dan and Perets, Oriel and Linial, Michal and Rappoport, Nadav},
  journal={arXiv preprint arXiv:2605.06830},
  year={2025},
  url={https://arxiv.org/abs/2605.06830}
}
```

## License

MIT
