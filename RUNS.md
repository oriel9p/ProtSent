# Training runs — what exists, what it was trained on, where its results live

One row per model that has been trained or is training. The point of this file is
that no run silently overwrites another and no results table is ambiguous about
which checkpoint produced it.

| name | backbone | corpus | output | results | status |
|---|---|---|---|---|---|
| ProtSent-V1-35M | ESM-2 35M | **unfiltered** (as submitted) | `oriel9p/protsent-esm2-35M` (Hub) | `results/benchmarks/v3/protsent_old_*` | published, submitted paper |
| ProtSent-V2-35M | Synthyra FastPLM ESM-2 35M | **decontaminated** dc40 | `models/protsent_esm2_35m_v3/final` | `results/benchmarks/v3/protsent_v3_*` | done 2026-07-29, 10 h 53 m |
| ProtSent-V2-35M (near-trough) | same run, checkpoint 4000 | same | `models/protsent_esm2_35m_v3_snapshots/checkpoint-4000` | `results/benchmarks/v3/protsent_v3_ckpt4000_*` | control for the peak-LR final checkpoint |
| ProtSent-V1-150M | ESM-2 150M | **unfiltered** (as submitted) | `oriel9p/protsent-esm2-150M` (Hub) | `results/benchmarks/v2_150m/protsent_v1_150m_*` | published, submitted paper |
| **ProtSent-V2-150M** | Synthyra FastPLM ESM-2 150M | **decontaminated** dc40 | `models/protsent_esm2_150m_v2/final` | `results/benchmarks/v2_150m/protsent_v2_150m_*` | **done 2026-07-30 23:0x**, k=5, 3,890 steps |
| ProtSent-V2-150M (near-trough) | same run, checkpoint 3250 | same | `models/protsent_esm2_150m_v2_snapshots/checkpoint-3250` | `results/benchmarks/v2_150m/protsent_v2_150m_ckpt3250_*` | control for the peak-LR final checkpoint |

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

## Resuming a run after a machine restart

Resume is not a one-liner for this pipeline; two things break and both are fixed.

**1. The checkpoint tokenizer cannot be instantiated.** Checkpoints are saved with
`tokenizer_class: FastEsmTokenizer` but no `AutoTokenizer` entry in `auto_map`, so
loading dies with `Unrecognized processing class`. Rewrite that one field to
`EsmTokenizer` — the vocab is the standard 33-token ESM2 vocab in `vocab.txt`, and
token ids were verified identical between the two tokenizers on residue strings,
'J', and non-residue characters. Leave `config.json` alone so the model still loads
through FastPLM's `auto_map` and keeps flash_attention_2.

**2. The saved optimizer has 6 more parameters than the rebuilt model.** A fresh run
builds the model with `load_model_for_training(args.model)` from a MaskedLM backbone
and keeps its 6-tensor LM head (491 params at 150M). On resume the SentenceTransformer
trainer rebuilds from the checkpoint directory, which round-trips as a plain encoder
(485). Resume then fails with `loaded state dict contains a parameter group that
doesn't match the size of optimizer's group`, and it fails *after* the full dataset
rebuild, roughly eight minutes in.

`fix_resume_optimizer.py <checkpoint>` repairs it. It aligns the optimizer's entries
against the rebuilt model **per parameter group** — a global alignment can satisfy the
total while leaving the per-group counts wrong, which is exactly how the first attempt
still failed — and refuses to write unless every surviving moment tensor matches the
model position for position. On checkpoint-2500 it dropped indices 182, 183 and
487-490, all of which had **no Adam moments at all**, and verified the remaining 483
against all 485 parameters. The original is kept as `optimizer.pt.orig`.

Recipe:

    python fix_resume_optimizer.py models/<run>/checkpoint-<N>
    RESUME=1 MINI_BATCH=64 MAX_PAIRS_PER_CLUSTER=5 bash train_esm2_150m.sh

**Do not change the GPU count when resuming.** The checkpoint stores one RNG state per
rank, and the step count is derived from the world size, so moving 6 -> 7 GPUs would
re-derive 3,890 steps as 3,334 and desynchronise the schedule from `global_step`.

**Pass the same data knobs.** `MAX_PAIRS_PER_CLUSTER` determines the corpus size and
therefore the step count; resuming with a different value silently trains on a
different dataset. The k=5 run rebuilds AFDB to 8,612,331 pairs, which is the number
to look for in the log.

## ProtSent-V2-150M results

Sweep: `run_benchmarks_150m.sh`, 4 arms x {3-NN, linear} x 23 tasks, `--eval_split test`,
one code path for every arm. Raw CSVs under `results/benchmarks/v2_150m/`.

**SCOPe-40 retrieval** (eligible queries, n=1,693 of 2,207; retrieval has no probe, so
the kNN and linear rows are identical by construction):

| method | R@1 | R@10 | MAP |
|---|---:|---:|---:|
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (`-s 7.5`) | 0.6556 | 0.7401 | 0.4098 |
| ProtSent-V1-150M (submitted) | 0.6615 | 0.8943 | 0.6431 |
| **ProtSent-V2-150M** | **0.7431** | **0.9368** | **0.7042** |
| V2-150M near-trough (ckpt3250) | 0.7295 | 0.9344 | 0.6841 |

Paired bootstrap, 10,000 resamples (`results/benchmarks/scope40_bootstrap_ci_150m.json`).
Every one of these excludes zero:

| comparison | R@1 | R@10 | MAP |
|---|---|---|---|
| V2-150M - V1-150M | +0.0809 [+0.0602, +0.1022] | +0.0431 [+0.0301, +0.0561] | +0.0607 [+0.0477, +0.0735] |
| V2-150M - ESM-2 150M | +0.1896 [+0.1654, +0.2138] | +0.1672 [+0.1477, +0.1867] | +0.2806 [+0.2644, +0.2967] |
| V2-150M - MMseqs2 | +0.0868 [+0.0620, +0.1116] | +0.1973 [+0.1754, +0.2191] | +0.2950 [+0.2751, +0.3144] |

The near-trough checkpoint differs from the final by 0.002-0.021, so the final
checkpoint is not an artifact of the 3-cycle schedule ending at peak LR.

### Alignment baselines at 150M — the top-1 concession does not apply at this scale

`results/benchmarks/alignment_paired_ci_150m.json`, paired bootstrap, 10,000 resamples,
eligible queries. HMMER (phmmer) is the stronger alignment baseline: R@1 0.6970 vs
MMseqs2's 0.6556.

| comparison | R@1 | R@10 | MAP |
|---|---|---|---|
| ProtSent-V2-150M - HMMER | **+0.0455 [+0.0219, +0.0691]** | +0.1565 [+0.1364, +0.1766] | +0.2301 [+0.2111, +0.2492] |
| ProtSent-V2-150M - MMseqs2 | +0.0868 [+0.0620, +0.1116] | +0.1973 | +0.2950 |
| ProtSent-V1-150M - HMMER | -0.0354 [-0.0608, -0.0100] | +0.1134 | +0.1610 |
| HMMER - ESM-2 150M | +0.1441 [+0.1169, +0.1719] | +0.0106 [-0.0148, +0.0360] | +0.0504 [+0.0290, +0.0721] |

**This is scale-dependent and the difference matters.** At 35M, ProtSent-V2 only *ties*
HMMER at top-1 (-0.0124 [-0.0372, +0.0124]), which is why the standing guidance is never
to claim ProtSent beats alignment at top-1. At 150M it beats HMMER significantly on every
metric, and the published V1-150M still loses to HMMER at top-1. So the honest statement
is per-scale, not global.

Worth stating for credibility: HMMER beats vanilla ESM-2 150M at top-1 by +0.144 while
losing at R@30, i.e. alignment remains the better top-1 finder than an untuned pLM and
the embedding advantage is in ranking depth.

### Remote homology at 150M — the direction depends on the probe, so state both

Remote homology is the task the corpus was filtered against, so it is the load-bearing
number. **It moves in opposite directions under the two probes, and quoting only one is
misleading** — an earlier version of this section led with the kNN drop alone, which
overstated the case:

| method | kNN acc | kNN macro-F1 | linear acc | linear macro-F1 |
|---|---:|---:|---:|---:|
| ESM-2 150M | 0.5194 | 0.2764 | 0.7500 | 0.5162 |
| ProtSent-V1-150M | **0.7047** | **0.4297** | 0.7401 | 0.4775 |
| ProtSent-V2-150M | 0.6612 | 0.3885 | **0.7503** | 0.4941 |

Under **3-NN**, decontamination costs 4.4 points (V1 0.7047 -> V2 0.6612), where the 35M
gained 0.8. Both ProtSent models remain far above vanilla (0.5194). The straightforward
reading is that this is what the filtering is *for*: V1-150M trained on a corpus
containing sequences at >=40% identity to this test set, and removing them removed the
inflation. The larger model appears to have exploited that leakage more.

Under a **linear probe the ordering reverses**: V2 (accuracy 0.7497, macro-F1 0.4929) is
*better* than V1 (0.7411, 0.4740), and ties vanilla on accuracy. So decontamination did
not uniformly cost performance here — it cost kNN and helped the linear probe.

State the confound: V2-150M also differs from V1-150M in configuration and data budget
(k=5 pairs per cluster), so this is not a single-variable ablation.

**Verified independently** (`verify_remote_homology.py`,
`results/benchmarks/verify_remote_homology_150m.json`), because the linear macro-F1 gap
against vanilla looked suspicious. Findings:

- The sweep's numbers reproduce from fresh embeddings (0.7497 vs 0.7503 accuracy,
  0.4929 vs 0.4941 macro-F1). No reporting or row-selection error; same split, seed and
  strategy for every arm.
- The macro-F1 deficit against vanilla is **statistically real**, not noise: paired
  bootstrap over the test set gives V2 - vanilla macro-F1 -0.0262 [-0.0450, -0.0071],
  excluding zero, while accuracy is unresolved (-0.0008 [-0.0108, +0.0092]).
- It is nonetheless **mostly a rare-class effect**. The test set has 457 classes with
  median support 3 and 209 classes with <=2 examples. Restricting to classes with >=3
  test examples shrinks the gap from -0.0257 to **-0.0036**; at >=10 it is -0.0079. So
  roughly 86% of the deficit sits in classes too small to estimate reliably.
- Against vanilla the *published* V1 is worse than V2 on this metric
  (-0.0443 [-0.0658, -0.0235]), so decontamination improved it.

When quoting remote homology, give accuracy and macro-F1 together, name the probe, and
say that macro-F1 over 457 classes with median support 3 is dominated by rare classes.

Aggregate across the 23 tasks, V2-150M vs V1-150M: **12 win / 4 tie / 7 lose** under
3-NN (median +0.0055) and **7 / 6 / 10** under a linear probe (median -0.0045). The same
probe-dependence seen at 35M.

### Identity-vs-gain at 150M, with the headroom control

`scope_identity_correlation_150m_v2.json`, `scope_identity_partial_150m_v2.json`.

| max identity to pretraining | n | dR@10 | dMAP |
|---|---:|---:|---:|
| [0.2, 0.4) | 164 | +0.1768 | +0.2885 |
| [0.4, 0.7) | 315 | +0.1778 | +0.3028 |
| [0.7, 1.0] | 1,214 | +0.1631 | +0.2737 |

Raw Spearman between identity and per-query gain is negative (R@10 -0.062 p=0.011,
MAP -0.083 p=6.1e-4). **After controlling for baseline headroom it collapses to a null**
(R@10 -0.002 p=0.93, MAP -0.037 p=0.12) — unlike the 35M, where the partial correlation
stayed significantly negative. Say "no relationship at 150M, slightly negative at 35M".
Neither is positive, and memorization predicts positive; that is the whole claim.

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
