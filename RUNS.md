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
| **ProtSent-V2.5-35M** | continues ProtSent-V2-35M | **decontaminated** dc40 + DMS | `models/protsent_esm2_35m_v2p5/final` | `results/benchmarks/v3/protsent_v2p5_*` | started 2026-08-03 18:38, 14,924 steps |

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
| ProtSent-V2-150M - HMMER (default filters) | +0.0455 [+0.0219, +0.0691] | +0.1565 [+0.1364, +0.1766] | +0.2301 [+0.2111, +0.2492] |
| ProtSent-V2-150M - MMseqs2 | +0.0868 [+0.0620, +0.1116] | +0.1973 | +0.2950 |
| ProtSent-V1-150M - HMMER | -0.0354 [-0.0608, -0.0100] | +0.1134 | +0.1610 |
| HMMER - ESM-2 150M | +0.1441 [+0.1169, +0.1719] | +0.0106 [-0.0148, +0.0360] | +0.0504 [+0.0290, +0.0721] |

**Correction: this comparison used phmmer with default filters, which overstates the model.**
Against the filters-off phmmer (`hmmer_maxsens.json`, R@1 0.7525, R@10 0.8978, MAP 0.6067)
ProtSent-V2-150M is **behind at top-1** (0.7431) and ahead at depth (0.9368) and MAP
(0.7042). A maximally sensitive profile search leads at top-1 at both scales.

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

## ISM-C-300M — a third-party structure-informed model, benchmarked 2026-08-02

Reviewer jVGf asked us to position ProtSent against structure-informed protein LMs
(ESM-S, S-PLM, ISM, Magneton). ISM was the one we could load and run inside the discussion window; the other three
have no checkpoint we could obtain in loadable form. Do not claim they have no public
weights -- S-PLM and ESM-S distribute outside the HF Hub.

`jozhang97/ismc-300m-2024-12` ships as a bare 1.33 GB `.pth` with no config, tokenizer or
safetensors, and targets `esm.models.esmc.ESMC` — a package whose latest release declares
`requires_python <3.13` against our 3.14 venv. `convert_ismc_to_hf.py` loads it into
`Synthyra/ESMplusplus_small`, a name-for-name HF port of upstream ESM-C at exactly that
size, producing `/storage/models/ISM-C-300M`. All 308 tensors load with `strict=True` and
no key remapping. Three gates, all passed: strict load; weights differ from vanilla
(308/308); forward pass finite and at cosine 0.353–0.762 to vanilla.

`Synthyra/ESMplusplus_small` **is** vanilla ESM-C-300M, so it is the matched control —
same architecture, parameter count, tokenizer and code path. Run
`run_benchmarks_ism.sh` (2 arms x 2 probes, 23 tasks, `-e test`, GPU 2, 3 h 05 m).
Results in `results/benchmarks/ism/`, joined by `ism_comparison.py` into
`ISM_COMPARISON.md`. All four arm/probe directories: 23/23 clean, zero error rows.

### Structure distillation is a trade, not a free win

### The resolved result: SCOPe-40 retrieval, with intervals

Quote these. Paired bootstrap over the 1,693 eligible queries, 10,000 resamples
(`scope40_bootstrap_ci_ism.json`). Every interval excludes zero:

| comparison | dR@1 | dR@10 | dMAP |
|---|---|---|---|
| ISM-C - ESM-C | +0.060 [+0.034, +0.085] | +0.078 [+0.053, +0.103] | +0.053 [+0.038, +0.067] |
| ProtSent-V2 150M - ISM-C | +0.311 [+0.285, +0.338] | +0.281 [+0.258, +0.304] | +0.431 [+0.412, +0.449] |
| ProtSent-V2 150M - ESM-C | +0.371 [+0.343, +0.400] | +0.359 [+0.336, +0.383] | +0.484 [+0.466, +0.501] |

The first row independently reproduces the direction ISM's own paper claims, on a benchmark
they never ran. That is stronger evidence the weight conversion is faithful than the strict
load is: `strict=True` checks names and shapes, not semantics.

### TLDR: ProtSent-V2 150M head to head against the ESM-C arms

Our best model minus each ESM-C arm, 20 tasks, tie tolerance 0.005. Full per-task table in
`ISM_COMPARISON.md`. Both columns cross model family and scale, so this is a comparison of
levels, not a controlled experiment.

| probe | vs ESM-C 300M | vs ISM-C 300M |
|---|---|---|
| kNN | 12W / 2T / 6L, median +0.012 (p=0.24) | 13W / 2T / 5L, median +0.028 (p=0.10) |
| linear | 4W / 2T / 14L, median -0.020 (p=0.03) | 4W / 3T / 13L, median -0.014 (p=0.05) |

Same shape as everywhere else in this project: we lead under nearest-neighbour retrieval
and lose under a trained probe, and only the linear-probe losses are significant. The
retrieval margin is the outlier and it is enormous — SCOPe-40 eligible R@10 +0.357 against
ESM-C and +0.278 against ISM-C, both with intervals excluding zero (above).

Largest kNN gaps against ISM-C: SCOPe-40 +0.278, Optimal pH +0.137, Metal Ion Binding
+0.122, EC +0.098, GO +0.077. Largest kNN losses: Fluorescence -0.160, Solubility -0.159,
Cloning -0.144, beta-lactamase -0.073, Material Production -0.044.

### The trade-off: descriptive only, nothing resolved

Each model against **its own** backbone, the 20 tasks with a defined main metric, tie
tolerance 0.005. Twenty and not 23 deliberately — see the caveat below.

| comparison | probe | W | T | L | median | sign p |
|---|---|---:|---:|---:|---:|---:|
| ISM-C 300M vs ESM-C 300M | kNN | 7 | 1 | 12 | -0.0078 | 0.36 |
| ISM-C 300M vs ESM-C 300M | linear | 7 | 3 | 10 | -0.0062 | 0.63 |
| ProtSent-V2 150M vs ESM-2 150M | kNN | 10 | 3 | 7 | +0.0049 | 0.63 |
| ProtSent-V2 150M vs ESM-2 150M | linear | 4 | 4 | 12 | -0.0130 | 0.08 |
| ProtSent-V2 35M vs ESM-2 35M | kNN | 10 | 3 | 7 | +0.0041 | 0.63 |
| ProtSent-V2 35M vs ESM-2 35M | linear | 2 | 7 | 11 | -0.0107 | 0.02 |

These reproduce our posted numbers exactly (`FINAL_rebuttal.md:50`: V2-150M kNN 10/3/7
median +0.004, linear 4/4/12 median -0.014).

**No kNN record is resolved.** Fisher exact on the contrast that matters — ISM-C 7W/12L
against ProtSent-150M 10W/7L — gives p = 0.22. The only rows that resolve are our own
linear-probe losses, which we already conceded. `FINAL_rebuttal.md:72` promises "we draw no
inferential claim from the aggregate", so **do not write "ours is the milder trade"** or any
comparative adjective over these tallies. The supportable sentence is: under kNN ProtSent is
net positive where ISM-C is net negative; under a linear probe both are net negative.

ISM-C wins on structure- and solubility-flavoured tasks (Cloning +0.180, Fluorescence
+0.173, Solubility +0.091, Material Production +0.082, SCOPe-40 +0.080 eligible R@10,
Remote Homology +0.048 under kNN) and loses on function and fitness (Stability -0.114,
EC -0.113, Temperature Stability -0.086, Variant Effect -0.079, GO -0.079). Both probes
agree on the direction.

**Metric caveat, and a trap.** Antibiotic Resistance, Remote Homology (Fold) and
Temperature Stability have an empty AUC column — one-vs-rest AUC is undefined when the test
split contains a class absent from training, which is the reason `FINAL_rebuttal.md:52`
already gives reviewers. `ism_comparison.py` can fall back to Accuracy and score all 23,
giving ISM-C 8/1/14 (-0.0086) and 7/3/13 (-0.0125), ProtSent-150M 11/4/8 (+0.0037) and
4/5/14 (-0.0124).

**Do not use that 23-task version as the headline.** The fallback shifts ISM-C's linear
median by -0.0063 and ours by +0.0006 — same procedure, asymmetric effect — turning a real
gap (ISM-C's linear trade is half ours on the declared metrics) into "nearly identical". It
also contradicts what our public text already commits to. Report the 20, and give the three
separately, where the interesting number is visible instead of averaged away: remote
homology moves +0.142 under contrastive post-training against +0.048 under distillation.

**MMseqs2 was scored twice; use the later one.** `mmseqs_baseline.json` (2026-07-29) gives
eligible R@10 0.7348 / MAP 0.4041; `bootstrap_ci.py`'s own hit-table scoring (2026-07-31)
gives 0.7401 / 0.4098, and hit30 differs by 0.021, so the two are genuinely different
scorings rather than rounding. `ism_comparison.py` takes the later one, which is also the
row already published in `FINAL_rebuttal.md:34`. Do not quote `mmseqs_baseline.json` for
SCOPe-40 in anything reviewer-facing.

Three of the 23 rows (EC, GO, SCOPe-40) are probe-invariant by construction: multilabel
and retrieval tasks use a built-in evaluator and ignore `--probe_type`.

### SCOPe-40, eligible queries only

| method | R@1 | R@10 | MAP |
|---|---:|---:|---:|
| ESM-C 300M | 0.3709 | 0.5794 | 0.2212 |
| ISM-C 300M | 0.4300 | 0.6592 | 0.2733 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.4210 |
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (-s 7.5) | 0.6556 | 0.7348 | 0.4041 |
| ProtSent-V2 35M | 0.6852 | 0.9220 | 0.6459 |
| ProtSent-V2 150M | 0.7431 | 0.9368 | 0.7046 |
| HMMER (phmmer, filters off) | 0.7525 | 0.8978 | 0.6067 |

**Do not read the ProtSent rows as beating ISM in a controlled sense.** They differ in
both family and scale, and raw mean-pooled ESM-C is simply weak at retrieval — below even
ESM-2 35M. Nothing here separates "contrastive post-training beats structure
distillation" from "ESM-2 beats ESM-C at this task". The experiment that would is
ProtSent post-training on an ESM-C backbone, giving a 2x2 of {ESM-C, ISM-C} x {raw,
ProtSent}. `load_model_for_training` already builds a SentenceTransformer over
`/storage/models/ISM-C-300M` (Transformer + Pooling, dim 960, 332,997,184 trainable), so
that needs no code — only GPU time: 23.9M pairs at batch 1024 on one GPU is ~23,300 steps.

## CATH v4.3 midnight-zone (ProtTucker / EAT)

Full writeup, results and controls: `results/benchmarks/cath_eat/CATH_COMPARISON.md`.
Per-level table: `results/benchmarks/cath_eat/CATH_LEVELS.md`. Reproduce with
`run_benchmarks_cath.sh` (H-level, all 8 arms), `ProtBench/cath_levels.py` (full
C/A/T/H), `run_benchmarks_cath_controls.sh` (L2-normalization, ESM2 scaling, k-mer
floor controls). Benchmark and dataset live in the sibling `~/ProtBench` repo
(task `cath_eat`, dataset `GrimSqueaker/cath43-eat`), not this one.

| Model | H (test219→lookup69k) | vs. own frozen base |
|---|---:|---:|
| ESM2-35M (frozen) | 40.7 | — |
| ProtSent-V1-35M | 50.7 | +10.0 |
| ProtSent-V2-35M | 56.7 | +16.0 |
| ESM2-150M (frozen) | 43.3 | — |
| ProtSent-V1-150M | 58.0 | +14.7 |
| ProtSent-V2-150M | 62.7 | +19.4 |
| ESM-C-300M (frozen) | 18.7 | — |
| ISM-C-300M | 25.3 | +6.6 |
| MMseqs2 baseline | 34.7 | (paper: 35) |
| phmmer baseline | 38.0 | (not comparable to paper's profile-HMMER 77) |

MMseqs2 reproducing the paper's published 35 at 34.7 is the pipeline-fidelity check —
same splits, same labels, same 150-query H-level denominator, independent
implementation. Two controls rule out the obvious objections: L2-normalizing vanilla's
embeddings (giving it ProtSent's cosine-friendly geometry for free) closes only
1.3–2.7 of the 16–19 point gap, and ESM2 itself is flat with scale on this task
(35M→650M: 40.7→43.3→42.7), so the gain is not something more parameters would buy.

**Open, not measured:** decontamination (`protsent-data-dc40`) was run against the
`remote_homology`/fold-prediction and PPI test sets, not against CATH test219. CATH-specific
leakage into the AFDB/Pfam/STRING training corpus has not been checked.

## ProtSent-V2.5-35M — a continuation pass on V2-35M

`train_esm2_35m_v2p5.sh`. Started 2026-08-03 18:38 on one B300 (GPU 5), 14,924
steps, measured 2.0–2.6 s/it, so ~8–11 h. **Results not yet measured.**

Not a plain "more of the same" epoch. Four things change at once, so V2.5 minus
V2 is not attributable to any single one:

| setting | V2-35M | V2.5-35M | why |
|---|---|---|---|
| init | `/storage/models/ESM2-35M` | `models/protsent_v2p5_init` | V2's own final weights |
| GOR | off | **0.1** | anisotropy; see the whitening result in `probe_gap_analysis.py` |
| DMS CoSENT | absent | **1.0M rows**, ~6.5% of steps | auxiliary fitness target |
| k (pairs/cluster) | 8 | **5** | fresh draw, and a smaller budget |
| seeds (shuffle/global) | 40 / 41 | **13 / 7** | different sample of the same corpora |
| LR | 2e-4, 3 cosine cycles | **5e-5, one half-cosine** | ends annealed, not at peak |
| effective max_seq_length | **unbounded** (bug) | **512** | see the section below |
| batch / mini-batch | 1024 / 256 | 1024 / 256 | unchanged |
| GPUs | 7 | 1 | availability |

Corpus actually built (from the run log):

| dataset | rows | note |
|---|---:|---|
| `pfam_sorted` | 284,683 | whole, k=5 |
| `afdb_sorted` | 7,000,000 | **truncated** from 8,612,331 by the per-file cap |
| `stringdb_train_15M` | 7,000,000 | *sampled* under the cap, seed 13 |
| `dms_cosent` | 1,000,000 | prefix of 2,175,734; the file is pre-interleaved |
| total | 15,284,683 | ~44% of V2's 34.8M-pair epoch |

**How much of this pass is actually new: about 28% of rows.** Worth stating because
"one more epoch" implies more than it delivers here. V2 drew k=8 members per cluster
and V2.5 draws k=5, both uniform and independent, so for a pair V2.5 draws from a
cluster of size n, P(V2 also sampled both) = S8(S8-1) / (n(n-1)) — exactly 1 when
n <= 8, since V2 took the whole cluster. Weighted by each cluster's k=5 pair count:

| dataset | rows | new to V2 | new rows |
|---|---:|---:|---:|
| `afdb_sorted` | 7,000,000 | 46.2% | ~3,234,000 |
| `stringdb_train_15M` | 7,000,000 | 0% — V2 consumed all 15M | 0 |
| `dms_cosent` | 1,000,000 | 100% | 1,000,000 |
| `pfam_sorted` | 284,683 | not measured | <=284,683 |
| total | 15,284,683 | | ~4.2-4.5M (~28%) |

AFDB is 1,818,848 clusters over 126,301,607 sequences, extremely skewed: 97.3% of
sequences sit in the 27.2% of clusters with more than 8 members, the largest holding
362,703. Those giant clusters contribute almost no overlap, but clusters just above
8 still overlap heavily (n=9 gives 56/72 = 78%), which is why the total lands at
46.2% rather than near 100%. The scan reproduces both V2's logged k=8 count of
18,987,468 and the 126,301,607 corpus row count above, which is the check that it is
reading the file the same way the pipeline does.

**The AFDB truncation is uniform, not the failure the pipeline warns about.**
`--max_map_rows` splits evenly across the three `--files`, so the 21M budget is a
7.0M cap per corpus, and AFDB exceeds it at k=5. A clustered corpus is truncated
at the first N pairs of a group-sorted file, which in general keeps only the
lowest-sorted clusters — but here the covered prefix averages
7,000,000 / 1,478,201 = 4.7355 pairs per cluster against 8,612,331 / 1,818,848 =
4.7350 for the file as a whole. Identical to four decimals, so the 81.3% of
clusters that were visited are effectively a random 81.3%. Independently
measured by scanning `group_id` directly; the 8,612,331 total reproduces the
figure the 150M k=5 run logged, which is the check that the scan is right.

Two code fixes were required before any of this ran; both are in the loss path
and both changed measured memory by more than an order of magnitude.

- `gor_loss.py` sliced batches by first dimension, which matches nothing under
  `DataCollatorWithFlattening`, so GOR embedded the **whole** batch with grad.
  It now slices with sentence-transformers' own `_create_minibatch` and caps at
  128 samples. GOR must also wrap `MatryoshkaLoss` from the outside: inside, it
  desynchronises `ForwardDecorator`'s index-keyed cache and CachedMNRL's
  backward hook dies with `inconsistent tensor size, expected tensor [122880]
  and src [16384]`.
- `CoSENTLoss` has no gradient cache and DMS sequences are the longest corpus
  (median 448 residues), so it, not the contrastive loss, set peak memory. It is
  now wrapped in `SubsampledLoss` at `--mnrl_mini_batch_size` rows. Without that
  cap, DMS forces the *global* `per_device_train_batch_size` down — one batch
  size covers every dataset in a multi-task dict — and the contrastive batch
  cannot stay at 1024.

Measured throughput, one B300, batch 1024, mini-batch 256, GOR + DMS +
Matryoshka: **2.0–2.6 s/it, 110 GiB peak of 267**. V2's 8.08 s/it for the same
1024 rows is not a fair comparison — it was padding to 1,561 tokens.

## Cross-device gather deadlocks with a CoSENT target (reproduced 2026-08-03)

`train_v2.sh` says it keeps DMS out of the joint interleave to avoid a
"CoSENT/gather deadlock under DDP". That is real, and it now has a test:

    CUDA_VISIBLE_DEVICES=6,2 GATHER=1 BATCH_SIZE=256 MINI_BATCH=128 \
      MAX_MAP_ROWS=60000 DMS_MAX_ROWS=40000 MAX_STEPS=15 \
      bash train_esm2_35m_v2p5.sh

hangs. The log reaches `DDP (world_size=2): enabling gather_across_devices` and
then no step ever completes: both ranks spin at ~91% CPU with no GPU allocation,
and a 20-minute timeout kills it (exit 124) at step 0 of 15. `GATHER` is a new
toggle on that script and defaults to 0; anything multi-GPU with `--dms_file`
should be smoke-tested this way before being trusted.

**What this rules out.** On N GPUs there is no way to lower the per-device batch
and recover the lost in-batch negatives through a gather. The parallelism that
works is the one V2 already used: gather off, per-device batch 1024, so each rank
carries its own 1024 negatives and the effective batch is N x 1024. That trades
optimizer steps for GPUs — 15.28M rows gives 3,731 steps on 4 GPUs against 14,924
on one — which is in the same range as V2's own 4,850 and so is a known-good
regime, but it is not the same run.

## Embedding geometry across the 35M line (measured 2026-08-03)

`probe_gap_analysis.py --models NAME=PATH ...` (the flag is new; the built-in
`ARMS` are unchanged). SCOPe-40 spectrum plus remote-homology accuracy under three
readouts. Raw JSON in `results/benchmarks/probe_gap_v2_baseline.json` and
`probe_gap_v2p5_ckpt2000.json`.

| model | mean random-pair cosine | participation ratio | eff. rank | RH 3-NN raw | whitened | linear |
|---|---:|---:|---:|---:|---:|---:|
| ESM-2 35M | 0.848 | 7.9 | 32.5 | 0.5832 | 0.6800 | 0.6902 |
| ProtSent-V1-35M | 0.294 | 31.0 | 61.0 | 0.6572 | 0.6766 | 0.6905 |
| ProtSent-V2-35M | 0.152 | 52.5 | 97.4 | 0.6813 | 0.7010 | 0.7007 |

**V1 -> V2 is a large, previously unmeasured geometry change**: mean pairwise cosine
0.294 -> 0.152 and participation ratio 31 -> 52.5 of 480 dimensions. V2 is much closer
to isotropic than the published model, which is consistent with its better retrieval.

**The anisotropy channel is close to exhausted at V2, which bounds what GOR can buy.**
Whitening — an invertible linear map fitted without labels — is worth +0.097 remote-homology
3-NN accuracy to vanilla ESM-2, +0.019 to V1 and only +0.020 to V2. For V2 the whitened
k-NN (0.7010) already equals the linear probe (0.7007). Since a linear probe absorbs any
invertible linear map for free, **GOR cannot help a linear probe by de-correlating a fixed
representation**; if it helps at all it must be by changing what the encoder learns during
training. Keep those two mechanisms separate when reading V2.5.

## `--max_seq_length` was never applied during training (found 2026-08-03)

**Every training run in this file — V1, V2-35M and V2-150M — trained on untruncated
sequences, despite passing `--max_seq_length 512`.** Fixed in
`load_model_for_training`, which now sets the limit after the model is assembled.

The flag reached `models.Transformer(...)`, but the custom-code branches
(`fastplm_esm2`, `esmplusplus`, `amplify`, `dplm2`, `profluent_e1`) then swap that
module's tokenizer for the backbone's own, and FastPLM's declares
`model_max_length` = 1e24. sentence-transformers reads the truncation limit off the
tokenizer, so the requested 512 was discarded and every batch was padded to the
longest sequence in it. Measured directly: a 512-pair batch arrived as
`(512, 1561)`, and `model.tokenize(["A" * 3000])` returned 3,002 tokens.

Consequences worth stating plainly:

- The trained models are **not** wrong, but the recorded configuration was. AFDB and
  STRING have ~29% of sequences longer than 512 residues (medians 276 and 334), so
  those runs saw materially longer inputs than the paper's stated 512.
- **Evaluation is unaffected.** `protein_benchmark_suite.py:1246` assigns
  `model_obj.max_seq_length` explicitly, at `DEFAULT_EMBED_MAX_LENGTH = 1024`. So
  every benchmark number in this file was measured with truncation at 1024 — the
  bug is training-side only, and there was a train/eval length mismatch throughout.
- It is why memory looked fine before and did not for V2.5: CachedMNRL bounds peak
  memory by mini-batch and so absorbed the extra length silently (6 GiB per step),
  while GOR and CoSENT embed with grad and hit 150 GiB on the same batch.

ProtSent-V2.5-35M trains at a genuine 512. That is the documented configuration and
it is memory-safe, but it is *not* what V2-35M did, so the continuation is not a
pure "more of the same" pass. Training at 1024 instead would match the evaluation
cap; it was not attempted because attention cost is quadratic in length and the
512 configuration already peaks at 110 GiB of a 267 GiB card.

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
