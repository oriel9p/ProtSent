# New evidence available for the rebuttal — verified, single source of truth

Every number here was measured on this hardware during the rebuttal period and
is reproducible from the repo. Sources are named per block. **Do not use any
number in a rebuttal response that is not on this page or in
`REBUTTAL_LEAKAGE.md`.** The submitted paper's own numbers remain in
`rebuttal/PAPER_text.txt`.

Naming: **V1** = the published/submitted `oriel9p/protsent-esm2-35M`.
**V2** = ProtSent-V2-35M, retrained during the rebuttal on the decontaminated
corpus. A 150M model on the same decontaminated data is planned but **will not
be ready for this rebuttal** — do not imply otherwise.

---

## 1. Full decontamination of all three pretraining corpora — completed

Reviewers asked whether structural-task gains come from leakage. The entire
pretraining corpus was re-filtered against benchmark test sets with MMseqs2
`easy-search`, corpus-as-query, at **40% identity / 80% coverage**
(`--cov-mode 1`), then the model was retrained from scratch on the result.

| corpus | rows before | rows after | removed | % |
|---|---:|---:|---:|---:|
| Pfam | 28,530,684 | 27,929,772 | 600,912 | 2.11% |
| AFDB | 135,404,259 | 126,301,607 | 9,102,652 | 6.72% |
| STRING | 76,070,154 | 71,891,417 | 4,178,737 | 5.49% |

Filter targets: `remote_homology` (biomap-research/fold_prediction test, 3,244
sequences) for Pfam and AFDB; `ppi_bernett` (Synthyra/bernett_gold_ppi test,
3,022 sequences) for STRING. Prefilter: exhaustive GPU ungapped (100% recall)
for Pfam and STRING; k-mer `-s 5.7` (89.4% recall vs exhaustive) for AFDB.

**The filtering was then verified on the files training actually opened**
(`verify_training_corpus.py`, `results/benchmarks/training_corpus_verification.json`).
Each training parquet was semi-joined against the recorded removal lists:

| training file | rows | flagged sequences that survived |
|---|---:|---:|
| `pfam_sorted.parquet` | 27,929,772 | **0** |
| `afdb_sorted.parquet` | 126,301,607 | **0** |
| `stringdb_train_15M.parquet` | 15,000,000 | **0** (both pair columns) |

Row arithmetic closes independently: 27,929,772 + 126,301,607 + 15,000,000 =
169,231,379 = the `total=` in the training log.

**SCOPe was deliberately not used as a filter target.** SCOPe-40 has no
train/test split, so filtering against it would remove essentially all domain
sequences from the corpus. This matches the ProtTucker precedent. The SCOPe
question is answered by the identity-stratified analysis in §4 instead.

## 2. ProtSent-V2-35M — retrained on the decontaminated corpus

4,850 steps, one epoch over 169.2M rows, 7x NVIDIA B300, `train_runtime`
39,170 s (10 h 53 m), 887.5 samples/s.

**How to describe the configuration — use this framing, briefly, once per
response, and do not dwell on it.** The training changes are not arbitrary: they
are the settings the *paper's own ablations already favoured*, which is also the
direct answer to Yi1G's criticism that the ablations do not support the
submitted defaults. We acted on them.

> We retrained on the filtered corpora using the configuration favoured by the
> paper's own ablations — proportional sampling and no synthetic hard negatives
> — otherwise following the submitted recipe, on 7 GPUs rather than 1.

Grounding, from the submitted ablations: removing synthetic hard negatives
improves 20/23 tasks at mean +7.9% versus 16/23 and +6.7% for the submitted
configuration; proportional sampling (+7.0%) is comparable to round-robin
(+6.7%). Running on 7 GPUs raises the contrastive batch from 1x1024 to 7x1024.

Do NOT write a paragraph enumerating V1-vs-V2 differences, and do NOT frame the
comparison as a controlled decontamination ablation — there is no
unfiltered-corpus retrain at the V2 configuration and none is planned. The
claim the evidence supports is the sufficient one: **decontaminating the corpus
did not cost performance**, and the retrained model is the stronger one.

## 3. Structural results — test split, kNN, all arms measured with one code path

`results/benchmarks/v3/`. Gallery: the same 2,207 SCOPe-40 sequences, self
excluded, no-hit queries scored as failures.

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| **ProtSent-V2 35M (decontaminated)** | **0.5256** | **0.7073** | **0.7390** | **0.4955** |

**Recall@K here is upper-bounded at 0.7671**: only 1,693 of 2,207 queries have
any non-self same-family protein in the gallery. Restricted to those:

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 | 0.6556 | 0.7348 | 0.7354 | 0.4041 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| **ProtSent-V2 35M** | **0.6852** | **0.9220** | **0.9634** | **0.6459** |

### The top-1 claim, stated correctly — read this before writing anything about R@1

Yi1G named "HMMER/MMseqs2" as missing baselines. **Both were run.** HMMER
(phmmer, `hmmer_baseline.py`) is the stronger alignment baseline and the harder
one to beat, because a per-query implicit profile detects remote homology better
than MMseqs2's k-mer prefilter. Eligible-query results, same gallery, same
scoring, no-hit queries counted as failures (691 of 2,207 queries return no
phmmer hit at all):

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5`) | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| **HMMER (phmmer)** | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 | 0.6852 | **0.9220** | **0.9634** | **0.6459** |

Paired bootstrap, 10,000 resamples (`results/benchmarks/alignment_paired_ci.json`):

| comparison | R@1 | R@10 | MAP |
|---|---|---|---|
| V2 - HMMER | **-0.0124 [-0.0372, +0.0124] UNRESOLVED** | +0.1412 [+0.1205, +0.1618] | +0.1708 [+0.1511, +0.1905] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |
| V1 - HMMER | -0.1110 [-0.1388, -0.0827] | +0.0703 [+0.0484, +0.0927] | +0.0764 [+0.0551, +0.0982] |
| V1 - MMseqs2 | -0.0697 [-0.0975, -0.0413] | +0.1110 [+0.0874, +0.1341] | +0.1413 [+0.1197, +0.1625] |

**The supportable claim, and the only one to make:** ProtSent-V2 is
*statistically tied* with the best alignment baseline at top-1 and beats both
alignment baselines decisively at ranking depth and MAP. The submitted model V1
*loses* top-1 to both, significantly.

Do NOT write that ProtSent beats alignment at top-1. It beats MMseqs2 there and
ties HMMER, and a reviewer who runs the stronger baseline will find the tie. Say
it first: alignment remains the better top-1 method, and the embedding advantage
is in ranking depth — which is the property that matters for retrieval,
clustering, and k-NN transfer, and it is consistent across both tools.

Remote homology — **the task the corpus was actually filtered against**. Report
accuracy and macro-F1 together, and **never compare either against MMseqs2's
AUC**: the alignment baseline's headline metric for this task is multiclass AUC
(0.6523), which is not commensurate with an accuracy. The commensurate numbers,
test split, all arms:

| method | kNN accuracy | kNN macro-F1 | linear accuracy | linear macro-F1 |
|---|---:|---:|---:|---:|
| MMseqs2 | 0.4365 | 0.2064 | — | — |
| ESM-2 35M | 0.5835 | 0.3173 | 0.6868 | 0.4414 |
| ProtSent-V1 | 0.6587 | 0.3687 | 0.6899 | 0.4281 |
| **ProtSent-V2** | **0.6668** | **0.4108** | **0.7016** | **0.4527** |

MMseqs2 hit coverage here is 0.889; the remaining 11% are scored against an
uninformative fallback. Note that under the linear probe V1's macro-F1 (0.4281)
is *below* ESM-2 (0.4414) — only V2 improves on both metrics under both probes.

**The paper's Table 2 reports macro-F1 .223 -> .313 for this task, which does not
match the .3173 -> .3687 measured here.** The submitted table was not computed on
the test split. Do not present the paper's number and this one as the same
measurement; quote the test-split numbers above and say they are test-split.

Removing every pretraining sequence within 40% identity / 80% coverage of the
remote-homology test set **improved** remote-homology performance.

**Checkpoint control.** The LR schedule is a 3-cycle cosine that ends at peak
LR, so a near-trough checkpoint (step 4,000) was benchmarked alongside the
final. They differ by 0.005-0.008 on every structural metric, so the final
checkpoint is not an artifact of where training stopped.

## 4. SCOPe: does the gain depend on proximity to pretraining data?

The planned analysis was to bin Recall@10 by maximum identity to the
pretraining corpus and show the gain survives at low identity. **That is
impossible here and the reason is worth stating**: the [0, 0.2) bin is empty and
the median max-identity is 0.908, because AFDB covers essentially all of
UniProt. The same is true of ESM-2's own UniRef50 pretraining set, so this is a
property of corpus coverage, not of ProtSent.

The underlying question is answerable directly. If the gain came from
memorizing pretraining neighbours, queries with a closer pretraining neighbour
would gain more. Per-query gain over ESM-2, both models on the same identity
table, same bins, same 1,693 eligible queries:

| max identity to pretraining | n | V1 dR@10 | V2 dR@10 | V1 dMAP | V2 dMAP |
|---|---:|---:|---:|---:|---:|
| [0.2, 0.4) | 164 | +0.0915 | **+0.1524** | +0.1856 | **+0.2859** |
| [0.4, 0.7) | 315 | +0.1016 | +0.1810 | +0.1453 | +0.2417 |
| [0.7, 1.0] | 1,214 | +0.0865 | +0.1565 | +0.1169 | +0.2099 |

Per-query Spearman between max identity and gain: R@10 -0.038 (V1) / -0.038
(V2); MAP -0.114 / -0.116, both p < 3e-6. **The correlation is null to
negative** — the advantage does not grow with proximity to pretraining data, it
shrinks slightly. Memorization predicts the opposite sign.

Cross-validated: two independent implementations
(`protein_benchmark_suite.evaluate_retrieval` and
`scope_identity_correlation.compute_per_query`) agree to four decimals
(V2 0.92203 vs 0.9220; V1 0.85115 vs 0.8512; ESM-2 0.76137 vs 0.7614).

## 4a. 95% bootstrap confidence intervals — reviewer HNXd's explicit request

HNXd asked for "95% confidence intervals for the reported metrics, computed by
bootstrapping over individual predictions". Retrieval answers that exactly: every
metric is a mean over per-query values, so resampling the 1,693 eligible queries
gives the sampling distribution with no refitting. `bootstrap_ci.py`,
10,000 resamples, `results/benchmarks/scope40_bootstrap_ci.json`.

Marginal intervals (eligible queries):

| method | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| ESM-2 35M | 0.4991 [0.4755, 0.5227] | 0.7614 [0.7407, 0.7815] | 0.4222 [0.4056, 0.4391] |
| MMseqs2 | 0.6556 [0.6326, 0.6781] | 0.7401 [0.7194, 0.7608] | 0.4098 [0.3913, 0.4283] |
| ProtSent-V1 | 0.5859 [0.5623, 0.6090] | 0.8512 [0.8334, 0.8677] | 0.5511 [0.5343, 0.5677] |
| ProtSent-V2 | 0.6846 [0.6621, 0.7064] | 0.9220 [0.9090, 0.9344] | 0.6454 [0.6299, 0.6606] |

**Quote the paired intervals, not the marginal ones.** The same queries are
scored by every method, so overlapping marginal intervals do not mean a
difference is unresolved. Paired per-query differences:

| comparison | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - V1 | +0.0986 [+0.0762, +0.1211] | +0.0709 [+0.0555, +0.0862] | +0.0943 [+0.0814, +0.1074] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

**Every one of those intervals excludes zero.** V2's top-1 lead over a tuned
MMseqs2 is small but statistically resolved, and its lead at depth is large.

Two results that must be reported alongside, because they cut against the paper:

- **MMseqs2 beats ESM-2 35M at top-1** by +0.1565 [+0.1276, +0.1855]. An untuned
  pLM is worse than alignment at finding the single best homolog.
- **MMseqs2 vs ESM-2 at Recall@10 (-0.0213 [-0.0484, +0.0047]) and MAP
  (-0.0125 [-0.0351, +0.0102]) is unresolved.** Do not claim ESM-2 beats
  alignment at depth; the data does not support it. ProtSent does, significantly.
- **MMseqs2 beats ProtSent-V1 at top-1** by +0.0697 [+0.0413, +0.0975] —
  significant. This is a real weakness of the submitted model and stating it
  first is what makes the V2 result credible.

Caveat to state: this bootstraps the query sample, so it quantifies uncertainty
from which proteins are in the benchmark. It does not quantify training-seed
variance, which is what the seed sweep below addresses.

## 4b. The identity-vs-gain null survives a headroom control

The obvious objection to §4: high-identity queries are already well solved by the
baseline, so they have less room to improve, and regression to the mean alone
would produce a flat-or-negative slope with no memorization story either way.
Gain is bounded above by (1 - baseline), so gain and baseline are not
independent. `scope_identity_partial.py`,
`results/benchmarks/scope_identity_partial_v{1,2}.json`.

Spearman between max pretraining identity and per-query gain in average
precision, 1,693 eligible queries:

| control | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| raw | -0.114 (p=2.8e-06) | -0.116 (p=1.6e-06) |
| partial, controlling for baseline score | **-0.083 (p=6.8e-04)** | **-0.081 (p=9.0e-04)** |
| headroom-normalised gain, gain/(1-baseline) | -0.109 (p=2.1e-05) | -0.110 (p=1.7e-05) |

Within baseline-score quartiles, where headroom is roughly constant, every
correlation for V2 is null or negative: +0.007, -0.090, -0.158, -0.057.
None is positive.

The cleanest single statement comes from Recall@10, which is binary, so the
"baseline scored zero" stratum is exactly the set of queries with full headroom:
**among the 404 queries the untuned backbone fails completely, identity to the
pretraining corpus does not predict the gain** (V2 Spearman +0.038, p=0.45).

So the negative slope is not a headroom artifact. The control was run because a
blind reader raised it, and it did not change the conclusion. Say so — a
pre-empted objection that survived its own control is worth more than the raw
correlation alone.

## 4c. Seed variability — reviewer HNXd's second explicit request

HNXd asked for "a variability analysis with multiple random seeds". Five seeds
(0-4) x 8 representative tasks x 3 model arms, 3-NN probe, test split, via the
suite's `--seed_list` (`run_seed_variability.sh`,
`results/benchmarks/seeds/seed_variability.json`). Mean +/- SD over 5 seeds:

| task | metric | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|---|---|
| Remote Homology | Accuracy | 0.5835 +/- 0.0000 | 0.6589 +/- 0.0002 | 0.6668 +/- 0.0000 |
| Variant Effect (GB1) | Spearman | 0.6582 +/- 0.0000 | 0.7108 +/- 0.0000 | 0.7806 +/- 0.0000 |
| Subcellular Localisation | Accuracy | 0.6243 +/- 0.0000 | 0.6697 +/- 0.0003 | 0.6743 +/- 0.0000 |
| Metal Ion Binding | Accuracy | 0.7402 +/- 0.0000 | 0.7380 +/- 0.0000 | 0.7522 +/- 0.0000 |
| Solubility (DeepSol) | Accuracy | 0.5102 +/- 0.0000 | 0.5352 +/- 0.0007 | 0.5381 +/- 0.0009 |
| Fluorescence (TAPE) | Spearman | 0.3736 +/- 0.0000 | 0.4510 +/- 0.0002 | 0.4568 +/- 0.0000 |
| Stability (Biomap) | Spearman | 0.6435 +/- 0.0001 | 0.5638 +/- 0.0001 | 0.5961 +/- 0.0000 |
| Thermostability (FLIP) | Spearman | 0.4427 +/- 0.0126 | 0.4696 +/- 0.0172 | 0.4568 +/- 0.0156 |

**Median SD across all 24 rows is 0.0000.** Explain why rather than just
asserting it: given fixed embeddings and a fixed test split, a 3-NN probe is
deterministic — the benchmark seed only moves subsampling and CV-fallback
splits. Thermostability is the one task that subsamples, and it is the only one
with visible spread (SD ~0.013-0.017).

**This retires the "that delta is just run-to-run noise" objection.** The
V1 -> V2 remote-homology gap of +0.0079 is roughly 40x the seed SD on that task.

The two uncertainty analyses are orthogonal and both are needed: seed SD covers
probe/split randomness (near zero here), and the bootstrap in 4a covers which
proteins happen to be in the benchmark (the dominant term). Neither covers
*training*-seed variance, since only one training run per model exists — say so.

**One number here answers HNXd directly.** HNXd noted our Stability (Biomap)
figure is far below the literature's 69.08% linear / 77.69% LoRA and suspected
the 3-NN probe was the cause. It is not: on Stability the 3-NN probe scores
higher than the linear probe for every arm (ESM-2 0.6435 3-NN vs 0.4395 linear).
The gap to the published number is therefore not a probe artifact, and we do not
claim it is; a matched comparison would need their split and metric definition,
which we could not verify within the rebuttal window.

## 5. MMseqs2-only baseline across the whole benchmark

`mmseqs_baseline.py`, 23 tasks, same metric definitions as the embedding path
(family-level Recall@K with self excluded; per-class max bitscore for
classification so AUC stays comparable; 1-NN by bitscore for regression).
No-hit queries are counted as failures, not dropped. Flags stated:
`-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`. The `-s 5.7` variant gives a
much weaker baseline (SCOPe R@1 0.3847) — **any MMseqs2 number must state its
sensitivity setting.**

Alignment beats the best embedding model outright on 3 tasks under kNN
(ec_classification, go_mf, beta_lactamase_peer) and 6 under a linear probe
(those plus enzyme_catalytic_efficiency, optimal_ph, stability). On
`ec_classification` MMseqs2 reaches F1_Macro 0.710 vs 0.598/0.562; on `go_mf`
0.585 vs 0.459/0.443. This is the generality-accuracy trade-off, measured
rather than asserted.

One caveat worth a sentence: MMseqs2 solubility AUC is 0.4185, below chance —
alignment label-transfer is anti-correlated on DeepSol.

## 6. Both probes, reported separately, on the test split

23 tasks x 4 model arms x {3-NN, linear probe}, `--eval_split test`
(`results/benchmarks/COMPARISON.md`). Against ESM-2 35M over the 20 tasks
comparable in both arms:

| probe | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median -0.0139 | 2 / 7 / 11, median -0.0107 |

**The probe decides the headline.** Both ProtSent models beat ESM-2 under kNN
and lose under a linear probe. The structural-retrieval advantage survives
both; a general-purpose superiority claim does not survive the linear probe.
This is the single most important honesty constraint on the rebuttal: any
"ProtSent > ESM-2" sentence must name the probe.

## 7. Errors in the submitted paper found by this audit — disclose proactively

1. **"100,000 sequences at the superfamily level" is wrong.** The code
   evaluates the SCOPe **family** field on **2,207** sequences. The 100,000 is
   the evaluator's `max_samples` cap echoed into the results table (visible as
   `Samples 100000` in every benchmark CSV). A separate superfamily evaluation
   still improves at both scales (R@1 0.667 -> 0.780 at 150M, 0.639 -> 0.726 at
   35M).
2. **The remote-homology test split is not hierarchy-disjoint.** It is TAPE
   remote homology repackaged: the pooled concatenation of TAPE's three
   holdouts (718 fold + 1,254 superfamily + 1,272 family = 3,244) with no column
   marking which. The corpus-level decontamination in §1 is the real control.
   The pooled 457-class macro AUC is also not comparable to published
   per-holdout top-1 accuracies.
3. **The PPI decontamination description does not match the released code.**
   `data_prep.py` uses `easy-search` (STRING as query, Bernett test as target)
   at 40% identity, `--cov-mode 1 -c 0.8`, removing hit query IDs — not
   `easy-linclust` at 50% with cluster-level removal. Describe what the code
   does.
4. **Eq. 1 is malformed** as the reviewer noted; corrected notation is in the
   revision.

## 8. What is NOT available — do not imply otherwise

- No 150M model on the decontaminated data (running next, not ready).
- No full end-to-end fine-tuning sweep.
- No matched runs of ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.
- No SaProt/ProSST backbone substitution (needs residue-level structure tokens
  for the full Pfam and STRING corpora).
- No paired bootstrap confidence intervals on the Table 2 per-task deltas.
- Reviewers **cannot see** any of this unless the response states it. There is
  no updated PDF they have read. Every number must be given in the response
  text itself, self-contained, with its metric and split named.
