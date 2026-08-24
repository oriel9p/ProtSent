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
| **ProtSent-V2.5-35M** | continues ProtSent-V2-35M | **decontaminated** dc40 + DMS | `models/protsent_esm2_35m_v2p5/final` | `results/benchmarks/v3/protsent_v2p5_*` | **done 2026-08-04 06:27**, 14,924 steps, GOR 0.1 |
| ProtSent-V2.5-35M (noGOR) | same config, `--gor_weight 0` | same | `models/protsent_esm2_35m_v2p5_nogor/final` | `results/benchmarks/v3/protsent_v2p5_nogor_*` | ablation, done 2026-08-04 12:36 |
| **ProtSent-V2.5-150M** | continues ProtSent-V2-150M | **decontaminated** dc40 + DMS | `models/protsent_esm2_150m_v2p5/final` | `results/benchmarks/v2_150m/protsent_v2p5_150m_*` | **done 2026-08-05 08:33**, k=8, 3,600 steps, GOR 1.0 |
| **ProtSent-ESMC-300M-V2** | Synthyra ESMplusplus-small (vanilla ESM-C-300M) | **decontaminated** dc40 + DMS | `/storage/users/ddofer/protsent_models/protsent_esmc_300m_v2/final` | `results/benchmarks/ism/protsent_esmc_300m_v2_*` | **done 2026-08-07 15:00**, k=10, 7,216 steps, GOR/Matryoshka off |

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
| **ProtSent-ESMC-300M-V2** | **0.7974** | **0.9539** | **0.7692** |

**Do not read the ProtSent-V2-150M / -35M rows as beating ISM in a controlled sense.**
They differ in both family and scale, and raw mean-pooled ESM-C is simply weak at
retrieval — below even ESM-2 35M. Nothing there separates "contrastive post-training
beats structure distillation" from "ESM-2 beats ESM-C at this task".

**ProtSent-ESMC-300M-V2 is that missing cell.** ProtSent post-training run directly on
the ESM-C backbone, giving the 2x2 of {ESM-C, ISM-C} x {raw, ProtSent} this section asked
for. `load_model_for_training` already built a SentenceTransformer over
`Synthyra/ESMplusplus_small` with no code changes needed — see
`train_esmc_300m_v2.sh` and the sampler-hazard section below for what it took to get a
multi-GPU run to survive. Point estimate is the best R@1 and MAP in the whole table,
ProtSent or baseline — above HMMER filters-off, the project's authoritative top baseline
(`results/benchmarks/hmmer_maxsens.json`). **This is a point estimate, not yet bootstrap
CI'd** (the "Quote these" intervals above are for the 150M arms only) — do not cite the
HMMER comparison as resolved until `bootstrap_ci.py` is run against this arm.

### ProtSent-ESMC-300M-V2 vs its own backbone and vs ISM-C, 23 tasks

Full per-task CSVs: `results/benchmarks/ism/protsent_esmc_300m_v2_{knn,linear}/`. Same
`-e test`, seed 42, 23-task set as the ESM-C/ISM-C arms above, so directly comparable —
`ism_comparison.py` has not been re-run for this arm, so these are hand-tallied counts,
not the sign-test / bootstrap machinery used above. Do not upgrade these to a
significance claim.

| probe | vs ESM-C 300M (own backbone) | vs ISM-C 300M |
|---|---|---|
| kNN | 16W / 0T / 7L | 14W / 1T / 8L |
| linear | 6W / 3T / 14L | 6W / 4T / 13L |

Same shape as every other arm in this project: strong under kNN, weak under a trained
linear probe. kNN wins concentrate in SCOPe-40 (see above), Remote Homology (0.681 vs
0.355 base / 0.403 ISM-C), Metal Ion Binding, Optimal pH, Subcellular Localisation.
Linear losses are largest exactly where kNN wins are largest elsewhere in the suite —
Stability (0.459 vs 0.755 base / 0.700 ISM-C), beta-lactamase (0.682 vs 0.832 / 0.808),
AAV Fitness (0.450 vs 0.606 / 0.581) — consistent with the alignment-leads-at-top-1
pattern noted for the other arms; this is not a new failure mode.

**Contamination caveat, not unique to this arm.** `train_esmc_300m_v2.sh` interleaved
`/storage/users/ddofer/data/dms_cosent.parquet` under CoSENT, and that file predates
`protsent-data-dc40` — it was never run through the MMseqs2 decontamination pass the
Pfam/AFDB/STRING sources got. Fluorescence, Stability, beta-lactamase-PEER and Variant
Effect (GB1) are DMS/ProteinGym-derived, so this arm's numbers on those four tasks are
unverified for train/test leakage; treat Stability and beta-lactamase above accordingly
until checked at sequence identity. **The same unfiltered `dms_cosent.parquet` is also
used by ProtSent-V2.5-35M and ProtSent-V2.5-150M above** (`train_esm2_35m_v2p5.sh`,
`train_esm2_150m_v2p5.sh` both point at the identical path) — their "decontaminated dc40
+ DMS" table label refers only to the Pfam/AFDB/STRING portion, not the DMS portion. This
was not previously called out in this file; the same four tasks on those two arms carry
the same unverified-leakage caveat and it should be checked project-wide rather than
assumed clean.

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

### V2.5 results, and the GOR ablation (2026-08-04)

Trained 2026-08-03 18:38 to 2026-08-04 06:27, 14,924 steps, one B300. The GOR-off
ablation (`protsent_v2p5_nogor`, `--gor_weight 0`, everything else identical)
finished 2026-08-04 12:36. Both arms benchmarked 23/23 under both probes.

SCOPe-40 retrieval, eligible queries (n=1,693 of 2,207):

| model | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1-35M | 0.5854 | 0.8511 | 0.9256 | 0.5509 |
| ProtSent-V2-35M | 0.6852 | 0.9220 | 0.9634 | 0.6459 |
| ProtSent-V2.5-35M noGOR | 0.6923 | 0.9250 | 0.9663 | 0.6528 |
| ProtSent-V2.5-35M | 0.6899 | 0.9244 | 0.9681 | 0.6521 |

Paired bootstrap, 10,000 resamples, V2.5 − V2: R@1 +0.0077 [−0.0041, +0.0201],
R@10 +0.0018 [−0.0053, +0.0089], R@30 +0.0047 [+0.0000, +0.0100], **MAP +0.0078
[+0.0032, +0.0125]**. MAP is the only metric that excludes zero.
`results/benchmarks/scope40_bootstrap_ci_v2p5.json`.

Win/tie/loss on the 20 tasks with a defined one-vs-rest AUC, ties at |δ| < 0.005:

| comparison | k-NN | linear |
|---|---|---|
| V2.5 vs ESM-2 35M | 9W/7T/4L, +0.0046 | 4W/4T/12L, −0.0103 |
| V2.5 vs V2 | 7W/8T/5L, +0.0010 | 7W/8T/5L, +0.0013 |

Indistinguishable from V2 on the suite; the k-NN median flips sign (+0.0010 to
−0.0005) depending on whether the tally is over 20 or 22 tasks, which is what a
noise-level statistic looks like. The linear-probe deficit against vanilla ESM-2
is unchanged. The per-task linear gains are concentrated on DMS-derived tasks
(Stability +0.0946, AAV Fitness +0.0732, Fluorescence +0.0149, Variant Effect
+0.0128, beta-lactamase +0.0110), consistent with re-adding the CoSENT target
rather than with GOR.

**GOR contributed nothing measurable, at a real cost.** Ablation vs the GOR arm:

| | GOR off | GOR 0.1 |
|---|---:|---:|
| SCOPe-40 eligible R@1 / MAP | 0.6923 / 0.6528 | 0.6899 / 0.6521 |
| k-NN vs V2 (20 tasks) | 8W/7T/4L, +0.0028 | 7W/8T/5L, +0.0010 |
| k-NN GOR vs noGOR | — | 2W/15T/2L, +0.0004 |
| mean random-pair cosine | 0.121 | 0.113 |
| step time (A/B, `time_gor_ab.sh`) | 2.385 s/it | 2.665 s/it (+11.7%) |

15 ties of 19 between the two arms. The whole SCOPe-40 gain reproduces with GOR
off. And most of the isotropy change was the extra training pass, not GOR: V2
0.152 → noGOR 0.121 → GOR 0.113, so GOR accounts for about a fifth of it.
`results/benchmarks/probe_gap_v2p5_nogor.json`.

**Why 35M was the wrong testbed.** The geometry section above shows whitening
buys V2-35M only +0.0197 and lands its k-NN exactly on its linear probe
(−0.0003). No geometry-only intervention has room left there. At 150M the same
measurement leaves +0.0154 on the table after whitening and the whitening gain is
+0.0555, so the channel is open at that scale. GOR was run at `gor_weight=0.1`
with `max_samples=128` and both terms at 1.0 — an arbitrary weight, a tenth of
the 1.0 the GOR paper uses. The null result is for that configuration at 35M.

## ProtSent-V2.5-150M — a continuation pass on V2-150M (2026-08-05)

`train_esm2_150m_v2p5.sh`. Started 2026-08-04 18:13 on 4 B300s, finished
2026-08-05 08:33: 3,600 steps, 14 h 18 m, no OOM.

Init is `models/protsent_150m_v2p5_init`, which is
`models/protsent_esm2_150m_v2/final` with FastPLM's `config.json` and
`tokenizer_config.json` restored from the `.fastplm` backups. Verified before
training: 515 tensors, 147.7M parameters, max absolute difference **0.0** against
the V2-150M weights.

| setting | V2-150M | V2.5-150M |
|---|---|---|
| loss | CachedMNRL | + GOR 1.0 + DMS CoSENT |
| `--mnrl_directions` | one-directional | symmetric |
| `--batch_sampler` | none | none (explicit) |
| batch / mini-batch | 1024 / 512 | 1024 / 64 |
| `--gor_max_samples` | — | 64 |
| k (pairs per cluster) | 5 | 8 |
| LR | 2e-4 | 5e-5, half-cosine |
| seeds (shuffle / global) | 40 / 41 | 17 / 11 |
| steps / GPUs | 3,890 / 1 | 3,600 / 4 |
| Matryoshka | off | off |

Corpus: Pfam 777,306 + AFDB 18.98M + STRING 15.0M + DMS 1.0M = **35.8M pairs
available**, of which 3,600 x 4,096 = **14.7M trained** (41%). AFDB exhausted at
k=8 before the `--max_map_rows` cap bound; STRING hit its own file size.

**Sizing.** `probe_150m_v2p5.sh`, 4 B300s: mini 256 OOMs at 260 GiB of 267 and
mini 128 fits; len 768 OOMs at mini 128; Matryoshka [128, 640] hung at step 6 of
10 twice and was killed at 20 min, so it was dropped; batch 2048 measures 501
samples/s against batch 1024 / mini 64 at 273, but needs a near-empty box and
OOMed once a co-tenant job took ~57 GiB per card. Two probe lessons worth
keeping: a 10-step average is useless here because step one costs ~4 minutes
(measure steps over a wall-clock window instead), and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks the DataLoader with
`pidfd_getfd: Operation not permitted` because expandable segments do not
support CUDA IPC.

### Results

SCOPe-40 retrieval, eligible queries (n=1,693 of 2,207). Alignment baselines are
the authoritative ones named at the top of this file.

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| ESM-2 150M | 0.5535 | 0.7702 | 0.8423 | 0.4236 |
| MMseqs2 | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| HMMER, filters off | 0.7525 | 0.8978 | 0.9232 | 0.6067 |
| ProtSent-V1-150M | 0.6615 | 0.8943 | 0.9439 | 0.6431 |
| ProtSent-V2-150M | 0.7431 | 0.9368 | 0.9681 | 0.7046 |
| **ProtSent-V2.5-150M** | **0.7513** | **0.9445** | **0.9722** | **0.7227** |

Paired bootstrap, 10,000 resamples, V2.5 − V2: R@1 +0.0095 [−0.0024, +0.0213]
unresolved; R@10 +0.0077 [+0.0012, +0.0142]; R@30 +0.0053 [+0.0006, +0.0106];
MAP +0.0189 [+0.0137, +0.0241]. Three of four exclude zero, against one of four
at 35M. `results/benchmarks/scope40_bootstrap_ci_v2p5_150m.json`.

Note the arm used: the authoritative V2-150M is `protsent_v2_150m_*`. The
`protsent_v2_150m_ckpt3250_*` arm is the near-trough control, and comparing
against it by mistake inflates every V2.5 − V2 delta (hit1 +0.0224 rather than
+0.0095, all four "significant").

Win/tie/loss over the 20 tasks with a defined one-vs-rest AUC, ties at
|δ| < 0.005, median δ:

| comparison | k-NN | linear |
|---|---|---|
| V2.5 vs ESM-2 150M | 12W/2T/5L, +0.010 | 3W/2T/14L, −0.016 |
| V2 vs ESM-2 150M | 9W/3T/7L, +0.004 | 3W/4T/12L, −0.014 |
| V2.5 vs V1-150M | 13W/3T/3L, +0.007 | 4W/6T/9L, −0.004 |
| V2.5 vs V2-150M | 8W/8T/3L, +0.001 | 3W/11T/5L, −0.003 |

Both V2.5 − V2 medians sit inside the tie band. The per-metric breakdown shows
the effect is entirely in the Spearman regression tasks, and that they move in
opposite directions by probe: AUC +0.000 k-NN / −0.001 linear, F1_Macro −0.009 /
−0.009, **Spearman +0.020 k-NN / −0.006 linear** (means over 8 / 2 / 9 tasks).
Eval protocol is identical across all four arms — `--eval_split test`,
`EvalMode=standard`, `BenchmarkSeed=42`, 23/23 clean per `bench_arm_status.py`.

Largest k-NN gains: Fluorescence +0.068, Thermostability +0.033, Variant Effect
(GB1) +0.025, Cloning +0.023, beta-lactamase +0.021. Largest linear losses: AAV
Fitness −0.086, Stability −0.023. Both of those decline monotonically across the
whole lineage at this scale (AAV 0.589 → 0.398 → 0.451 → 0.365; Stability 0.706 →
0.699 → 0.663 → 0.639), so V2.5 continues a trend rather than starting one.

**The AAV linear drop is not embedding collapse.** The obvious hypothesis — MNRL
teaches near-identical sequences to be positives, so a mutational scan gets
flattened — is measurable and false here. Over 3,000 AAV test variants:

| arm | mean pairwise cosine | effective dim | best-direction corr with fitness |
|---|---:|---:|---:|
| ESM-2 150M | 0.9990 | 7.73 | 0.554 |
| ProtSent-V2-150M | 0.9916 | 6.79 | 0.542 |
| ProtSent-V2.5-150M | 0.9829 | 10.39 | 0.576 |

V2.5 spreads the variants more, not less, and its best single linear direction
correlates better with fitness than vanilla ESM-2's. Since that correlation is
fitted in-sample on test while the probe is fitted on train, what degraded is
train → test transfer, not the information content. AAV FLIP splits are
constructed to be hard in exactly that way.

DMS training sequences have zero exact overlap with the AAV Fitness (0/50,430),
Stability, Variant Effect and Fluorescence test sets. The DMS parquet is still
not MMseqs2-filtered, so this is an exact-match check, not an identity check.

**Not attributable to GOR.** V2.5-150M changes six things at once (GOR, DMS,
symmetric directions, k, seeds, batch geometry) and there is no GOR-off ablation
at this scale. The 35M ablation found GOR contributed nothing measurable there.

## Loss-configuration audit (2026-08-04)

Three settings were wrong or unexamined in every run to date, found by reading
the CachedMNRL docs against what the pipeline passes. Fixed on
`fix/train-max-seq-length-and-gor`; V1, V2 and V2.5 all trained without them.

- **`batch_sampler` was silently `none`.** `_resolve_batch_sampler` mapped `auto`
  to `NO_DUPLICATES` only for `loss_mode in {mnrl, cached_mnrl, cached_gist,
  gist}`. Multi-task runs pass `loss_mode="multi"`, which fell through to `None`,
  so no multi-dataset run ever used the sampler the CachedMNRL docs pair with the
  loss. It now dispatches on the resolved primary loss. Measured cost at smoke
  scale is +0.6-1.0 s/it, but that smoke has 20k rows per corpus against a 256
  batch and so rejects constantly; the cost at 7M rows and batch 1024 is not
  measured.
- **`directions` was one-directional.** ST defaults to `("query_to_doc",)` for
  asymmetric retrieval. Every corpus here is symmetric — two cluster members, two
  interacting proteins — so the default is now
  `("query_to_doc", "doc_to_query")`. It costs nothing: the embeddings are
  already computed, it only adds terms to the softmax. Confirmed at smoke scale.
- **GOR knobs were unreachable.** `max_samples` was a wrapper default of 128,
  never plumbed to the CLI, chosen while GOR was still OOMing at 150 GiB and
  never revisited. `mean_weight`, `second_moment_weight` and `aggregation` were
  frozen at the ST defaults, so the EmbeddingGemma recipe (mean term off) was not
  expressible. All four are now flags.

`mini_batch_num_tokens` — token-count-packed mini-batches, which suits
variable-length protein data under flash-attention with input flattening — is
documented upstream but **is not in sentence-transformers 5.6.1**, which is the
latest release on PyPI as of 2026-08-04. Nothing to enable yet.

`--max_seq_length` now sets the limit in both directions, clamped to the
backbone's `max_position_embeddings` (1026 for ESM-2) rather than to the
tokenizer, which is the thing that lies. Tighten-only was wrong once checkpoints
started carrying a 512 tokenizer: resuming one at 1024 silently stayed at 512.

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

**Gather is not the cause. Fixed 2026-08-05; this heading is kept only because
the old name is what people will search for.** Diagnosed on ESM-C 300M
(`train_esmc_300m_v2.sh`) and reproduced five times: 2 and 3 ranks, gather on and
off, Matryoshka on and off, machine load 107 to 722. It always died at step 3,
11, 19, 19 or 20, and turning gather off changed nothing except which collective
mismatched first.

The real mechanism is dataset sharding. accelerate's `BatchSamplerShard`
(`_iter_with_no_split`) hands rank *i* batches *i, i+N, i+2N, ...*, so with a
multi-dataset sampler the ranks routinely run **different datasets on the same
step**. That is harmless while every dataset shares a loss, and fatal the moment
one does not: `CachedMultipleNegativesRankingLoss` calls `.backward()` inside its
own forward and `CoSENTLoss` does not, so on a mixed step the ranks emit
different numbers of DDP gradient allreduces. Their collective streams drift
apart and NCCL aborts on a type mismatch at one sequence number:

    Rank 0: SeqNum=14603, ALLREDUCE, NumelIn=9285184     <- still in training_step
    Rank 1: SeqNum=14603, ALLGATHER, NumelIn=1, NumelOut=3  <- already logging loss
    Rank 2: SeqNum=14603, ALLGATHER, NumelIn=1, NumelOut=3

`py-spy`-style SIGUSR1 stacks agree: rank 0 in `trainer.py:training_step`, the
others in `trainer.py:_maybe_log_save_evaluate`, whose `_nested_gather(tr_loss)`
is that 1-element allgather. A slow rank changes *when* a collective is issued,
never *which* one, so this is not contention -- which is also why the failure
step did not move while load varied sevenfold.

**Fix:** `_align_proportional_sampler_to_world_size` in `protein_pipeline.py`
emits batches in blocks of `world_size` from a single dataset, so every rank's
slice of a block is that same dataset. It is applied automatically when the
sampler is PROPORTIONAL and `dms_cosent` is in the interleave. Cost is the last
`world_size - 1` batches of each dataset, single digits out of ~21.6k.
`tests/test_sampler_alignment.py` asserts the hazard exists unpatched (4.4% of
steps at 2 ranks, 6.5% at 3, 8.6% at 4) and is zero patched. After the fix the
ESM-C 300M run cleared step 20 and kept going.

**A short smoke does not clear this class of bug.** Ten steps ran clean at 37.5
s/it before step 11 hung. The hazard is per-step and only fires once the sampler
first splits datasets across ranks, so "it stepped fine for 10 steps" is not
evidence of anything. Budget 30+ steps.

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

## Late interaction (ColBERT/MaxSim) — campaign started 2026-08-23

A second scoring mode, trained with `train_late_interaction.py` rather than `protein_pipeline.py`:
one vector per residue, scored by MaxSim, on the same dc40 pair corpus. Every arm exports both a
`late/` multi-vector model and a `dense_view/` mean-pooled equivalent, so the pooled benchmarks can
score it unchanged. Outputs under `models/late_interaction/$NAME`, results under
`results/late_interaction/`.

| name | backbone | steps × pairs/step | head | pool | status |
|---|---|---|---|---|---|
| `protsent_late` | ProtSent-V2-35M | 2,000 × 256 (2 GPU) | 64-D | capped 2M/file | done, pilot |
| `esm2_late` | Synthyra/ESM2-35M | 2,000 × 256 (2 GPU) | 64-D | capped 2M/file | done, pilot |
| `protsent_late_150m` / `esm2_late_150m` | V2-150M / Synthyra-150M | 5,000 × 128 | 64-D | capped | done |
| `protsent_late_proj128` | ProtSent-V2-35M | 4,000 × 128 | **128-D** | capped | done — head-size ablation |
| `protsent_late_swap` | ProtSent-V2-35M | 4,000 × 128 | 128-D | capped | done — symmetry ablation |
| `protsent_late_pool_control` | ProtSent-V2-35M | 4,000 × 128 | 128-D | **uncapped** | matched-pool control |
| **`protsent_late_long`** | ProtSent-V2-35M | 18,219 × 128 | 128-D | uncapped | phase 1, running |
| **`protsent_late_150m_long`** | ProtSent-V2-150M | 18,219 × 128 | 128-D | uncapped | phase 1, running |
| `esm2_late_long` | facebook/esm2_t12_35M | 18,219 × 128 | 128-D | uncapped | phase 1, running |
| `*_prop` | continues the above | proportional sampler | 128-D | uncapped | phase 2, queued |

**Two controls make the long runs interpretable.** `long` vs `pool_control` isolates *steps* (both
uncapped, 18,219 vs 4,000); `pool_control` vs `proj128` isolates *pool diversity* (both 4,000 steps,
uncapped vs capped at 2M pairs/file). Without both, "the long run is better" cannot be attributed.

**"One round-robin pass" is a pass over Pfam, not over the data.** The built pool is Pfam 777,306 +
AFDB 18,987,468 + STRING 6,000,000 = 25.76M pairs, and ROUND_ROBIN cycles the three sources until the
smallest is exhausted. 18,219 steps × 128 therefore consumes 100% of Pfam, 13% of STRING and **4.1%
of AFDB** — 9.1% of the pool. A real epoch is ~201k steps (~50 h/card). Phase 2 switches to
PROPORTIONAL for that reason, not for more steps: on SCOPe the pilot curve is flat from ~1,500 steps
(ProtSent +.0002 over the last 500; ESM-2 peaks at 1,500 and drifts down), so mixture, not step
count, is the variable still worth moving.

Checkpoints are deleted at exit (`--save_total_limit 1` plus cleanup), so `snapshot_checkpoints.sh`
keeps a weights-only copy of each one under `$RUN/snapshots/step-N/`. Two points (step0 and final)
cannot tell "still improving" from "peaked early and drifted", which is exactly what ESM-2 did.

**Attention backend is recorded per run** in `runtime.json`. Everything before 2026-08-24 ran on
sdpa, including runs whose logs claimed otherwise: transformers silently rejects `kernels` outside
`[0.15.2, 0.16)` and falls back. The phase-1 long arms are the first to actually train under flash
(1.48–1.97× faster, peak VRAM 11.9 → 9.0 GB). A paired A/B found no quality cost at the scale of
this campaign's effects (deltas ≤ .0045 against a CI width of .011) — which is not the same as "no
cost": a sub-.005 degradation is not excluded, and resolving that would take ~9 seeds/arm.

## bf16 weights froze the backbone (found 2026-08-24)

Every late-interaction run between 2026-08-24 17:05 and 21:00 trained a **frozen backbone**. The
flash-attention path loaded the model with `dtype=torch.bfloat16`, which put AdamW's *parameters*
in bf16 rather than keeping fp32 masters. bf16 has an 8-bit mantissa, so near the median backbone
weight (0.0606) representable values are ~2.4e-4 apart, while an AdamW step at the trainer's
1e-5 backbone LR moves each weight by ~1e-5 — **1/24th of one representable step**. The addition
rounds to a no-op.

Measured directly, one AdamW step at lr 1e-5, fraction of backbone *elements* that changed:

| params | elements moved |
|---|---:|
| fp32 | **93.4%** |
| bf16 | **2.4%** |

and it scales with the step size exactly as the mechanism predicts: lr 1e-4 → 21.2%, lr 1e-3 → 88.8%.
Regression test: `tests/test_late_interaction.py::test_configured_learning_rate_actually_moves_the_backbone`.

**Why it looked like a data problem.** The first symptom was `protsent_late_pool_control` scoring
.6244 SCOPe superfamily eligible MAP against `protsent_late_proj128`'s .7057, and those two arms
differed in both pair pool *and* backend. Mean cosine of each arm's dense view against the base
ProtSent-V2-35M, over 400 SCOPe-40 sequences, settles it:

| arm | pool | backend | cos vs base | SCOPe elig. MAP |
|---|---|---|---:|---:|
| `protsent_late_proj128` | capped 2M/2M | **fp32/sdpa** | **0.836** | **.7057** |
| `protsent_late_pool_control_bf16bug` | uncapped 19M/6M | bf16/flash | 0.9976 | .6244 |
| `protsent_late_capped_flash` | capped 2M/2M | bf16/flash | 0.9977 | .6214 |

The two bf16 arms trained on completely different pools and landed indistinguishable from each
other *and* from the untrained base. The pool never mattered; the backbone never moved. What did
change is the randomly-initialised 128-D head, which trained at lr 1e-4 (21% of elements able to
move) against a frozen backbone — worse than the fully-trained pair, which is the .08 gap.

Vanilla ESM-2 appeared to improve on the same broken config (.4551 → .6090) because a vanilla
backbone does not need to move: the head alone carries most of that gain. An arm improving is
therefore **not** evidence that its backbone trained.

**Fix.** Drop the bf16 weight pin and keep flash. The kernel needs half-precision *activations*,
which `bf16=True` autocast already supplies; transformers accepts a pinned flash backend on an
fp32 model — verified by loading it, not assumed. TF32 is enabled for the fp32 matmuls
(`torch.backends.cuda.matmul.allow_tf32`), which lowers mantissa precision *inside* a matmul
without touching stored weights, so updates stay exact.

**What this invalidates.** Every throughput number measured under the bf16 config, including the
1.48–1.97x flash speedups and the compile verdict (+3.6% Pfam / −13% STRING) — all were taken with
a model that was barely training. Re-measured under fp32. The earlier paired flash quality A/B is
also void: it compared sdpa-fp32 against flash-bf16 and attributed the difference to the kernel.
Runs archived as `*_bf16bug`, with their curve rows relabelled rather than deleted.

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
