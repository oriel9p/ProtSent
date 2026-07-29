# Training runs — what exists, what it was trained on, where its results live

One row per model that has been trained or is training. The point of this file is
that no run silently overwrites another and no results table is ambiguous about
which checkpoint produced it.

| name | backbone | corpus | output | results | status |
|---|---|---|---|---|---|
| ProtSent-V1-35M | ESM-2 35M | **unfiltered** (as submitted) | `oriel9p/protsent-esm2-35M` (Hub) | `results/benchmarks/v3/protsent_old_*` | published, submitted paper |
| ProtSent-V2-35M | Synthyra FastPLM ESM-2 35M | **decontaminated** dc40 | `models/protsent_esm2_35m_v3/final` | `results/benchmarks/v3/protsent_v3_*` | done 2026-07-29, 10 h 53 m |
| ProtSent-V2-35M (near-trough) | same run, checkpoint 4000 | same | `models/protsent_esm2_35m_v3_snapshots/checkpoint-4000` | `results/benchmarks/v3/protsent_v3_ckpt4000_*` | control for the peak-LR final checkpoint |
| **ProtSent-V2-150M** | Synthyra FastPLM ESM-2 150M | **decontaminated** dc40 | `models/protsent_esm2_150m_v2` | pending | **training, started 2026-07-29 16:03** |

The internal `v3` in the 35M paths is an inherited `RUN_NAME`; the paper name for
that model is **ProtSent-V2-35M**. The 150M directory is named for the paper
name directly.

## The corpus

Both V2 runs read `/storage/users/ddofer/data/protsent-data-dc40`, which is every
pretraining source filtered at 40% identity / 80% coverage against the benchmark
test sequences:

| file | rows | flagged sequences surviving |
|---|---:|---:|
| `pfam_sorted.parquet` | 27,929,772 | 0 |
| `afdb_sorted.parquet` | 126,301,607 | 0 |
| `stringdb_train_15M.parquet` | 15,000,000 | 0 |

Verified by `verify_training_corpus.py`, which semi-joins each file against the
recorded removal lists rather than trusting the filtering job's own report. Both
training scripts refuse to start if any of the three files is missing, so neither
can silently fall back to the unfiltered corpus.

## ProtSent-V2-150M configuration

`train_esm2_150m.sh`. Sibling of `train_esm2_35m.sh`, same objective and data,
with the deltas forced by model size. Full reasoning is in the script header.

| setting | value | why |
|---|---|---|
| backbone | `Synthyra/ESM2-150M` | fast_esm, hidden 640, 30 layers |
| batch size | 1024 / device | measured: bs512 gains only 8% throughput for half the negatives |
| mini-batch | **64** | measured frontier; 128 OOMs at both bs1024 and bs512 |
| gradient checkpointing | **off** | slower here (36.6 vs 34.0 s/it) and CachedMNRL already bounds memory |
| attention | `flash_attention_2` | FA3 ships sm_80/sm_90a only, dies on these sm_103 B300s |
| torch.compile | off | measured no effect at 35M, and inputs are variable-length |
| gather across devices | off | each rank already carries 1024 in-batch negatives |
| sampler | proportional | favoured by the paper's own ablations |
| synthetic hard negatives | off | ablations: 20/23 tasks at +7.9% without, vs 16/23 at +6.7% with |
| warmup | 300 steps | ~5% of the 5,659-step epoch; the 35M run's flat 1000 was 20.6% |
| GPUs | 6 | 5,659 steps/epoch |
| save | every 250, keep 2 | ~2.4 h between checkpoints at this step time |

Measured memory frontier, 6 steps each on 6x B300 (267 GiB):

| config | result |
|---|---|
| mini 512 / bs 1024 | OOM at 262 GiB |
| mini 256 / bs 1024 | OOM at 262 GiB |
| mini 128 / bs 1024 | OOM at 262 GiB |
| mini 128 / bs 512 | OOM at 262 GiB — not a batch-size problem |
| mini 64 / bs 1024 | **OK, 245 GiB peak, 34.0 s/it** |
| mini 64 / bs 512 | OK, 15.7 s/it (195.7 seq/s vs 180.7) |
| mini 256 / bs 1024 + checkpointing | OK, 118 GiB peak, 36.6 s/it |

Expected wall clock **~53 hours**. To cut it, lower `MAX_PAIRS_PER_CLUSTER`
(8 -> 4 roughly quarters the AFDB and Pfam pair counts) or set `MAX_STEPS`. Do
not cut `BATCH_SIZE`; the throughput measurement above shows it buys almost
nothing.

Two traps recorded so they are not rediscovered:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks the dataloader
  workers here with `pidfd_getfd: Operation not permitted`.
- Headroom at mini-batch 64 is ~22 GiB, which is thin. If a long-sequence batch
  OOMs mid-run, resume with `RESUME=1 MINI_BATCH=48 bash train_esm2_150m.sh`.

## Not overwriting things

- Each run has its own `RUN_NAME`, so its output is `models/$RUN_NAME` and its
  checkpoints, logs and results are separate.
- `train_esm2_150m.sh` refuses to start if `models/$RUN_NAME/final` already
  exists, unless `ALLOW_OVERWRITE=1`.
- `--no_resume` is the default, so an aborted attempt's stale checkpoint is never
  picked up silently; `RESUME=1` opts in.
- The benchmark suite appends to its results CSV rather than overwriting, and
  `bench_arm_status.py` resolves completeness per task, so re-measuring an arm
  never destroys the earlier measurement.
