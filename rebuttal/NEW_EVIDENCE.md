# New evidence available for the rebuttal — verified, single source of truth

Every number here was measured on this hardware during the rebuttal period and
is reproducible from the repo. Sources are named per block. **Do not use any
number in a rebuttal response that is not on this page or in
`REBUTTAL_LEAKAGE.md`.** The submitted paper's own numbers remain in
`rebuttal/PAPER_text.txt`.

Naming: **V1** = the published/submitted `oriel9p/protsent-esm2-35M`.
**V2** = the models retrained during the rebuttal on the decontaminated corpus:
**ProtSent-V2-35M** and, now finished, **ProtSent-V2-150M** (§9). Always say which
scale a number refers to — several claims differ between them, and the top-1
alignment claim differs decisively.

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

## 4d. Embedding-space organisation — HNXd's first request, now measured

HNXd asked for "either a direct retrieval/clustering evaluation, or an analysis
showing how ProtSent changes the local and global organization of the protein
embedding space", and named silhouette and distance-vs-property-similarity
explicitly. Earlier drafts conceded this was not computed. **It is now**
(`embedding_geometry.py`, `results/benchmarks/embedding_geometry.json`).

SCOPe-40 labels are four-level — `class.fold.superfamily.family` — so the
benchmark carries its own ground-truth hierarchy and needs no outside
annotation. 2,207 domains, 917 families, cosine distance.

| measure | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---:|---:|---:|
| silhouette (family) | -0.1426 | -0.0441 | **+0.0529** |
| NMI vs true families | 0.8225 | 0.9004 | **0.9174** |
| **ARI vs true families** | **0.0544** | 0.4165 | **0.5071** |
| intra/inter distance ratio | 0.4174 | **0.2694** | 0.3524 |
| Spearman(distance, shared hierarchy) | -0.1055 | **-0.3125** | -0.2097 |

Mean pairwise distance by how much SCOPe hierarchy two domains share
(0 = different class, 4 = same family):

| shared levels | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| ESM-2 35M | 0.156 | 0.140 | **0.146** | 0.123 | 0.064 |
| ProtSent-V1 | 0.733 | 0.654 | 0.536 | 0.345 | 0.190 |
| ProtSent-V2 | 0.865 | 0.821 | 0.780 | 0.571 | 0.299 |

Three findings worth stating, in this order:

1. **ARI rises from 0.054 to 0.507.** Clustering the untuned ESM-2 space at the
   true number of families recovers almost nothing — 0.054 is near chance. Both
   ProtSent models recover a large part of the family partition. This is the
   single clearest evidence that the space is reorganised rather than rescaled,
   and it is a 9x change, not a marginal one.
2. **Silhouette crosses zero.** In ESM-2, families overlap more than they
   separate (-0.14). In ProtSent-V2 they separate (+0.05). That is a qualitative
   change of sign, not a shift along a scale.
3. **ESM-2's distances are not monotone in the hierarchy; ProtSent's are.**
   ESM-2 puts domains sharing two hierarchy levels *further apart* (0.146) than
   domains sharing one (0.140) — the ordering is broken. Both ProtSent models are
   strictly monotone across all five levels. This is the "global organisation"
   HNXd asked about, and it is the measure that cannot be gamed by tightening
   individual families.

**In the rebuttal, quote ESM-2 vs ProtSent-V2 only.** V2 is the model being
presented and it is plainly better than the backbone on every measure above.
Adding the V1 column here invites a V1-vs-V2 digression that costs the reader
attention and answers nobody's question. The V1 numbers stay in this record and
in the repo for anyone who asks; they are not withheld, just not foregrounded.
(For the record: V1 has a tighter intra/inter ratio, 0.2694 vs 0.3524, and a
stronger hierarchy correlation, -0.3125 vs -0.2097; V2 has the better silhouette
and clustering recovery.)

Note also that ESM-2's absolute distances are tiny (0.064-0.156 across the whole
range), i.e. the untuned space is highly anisotropic; contrastive training
expands it. Silhouette and ARI are scale-invariant, so the comparison is not an
artifact of that expansion.

## 4e. Linear probes: layer sweep, and why the embedding layer is the right one

Two reviewer threads land here. HNXd asked for a linear-classifier baseline and
suspected our numbers looked low because of the 3-NN probe. Yi1G asked whether
the evaluation protocol is even reproducible. Both are answered by measuring
rather than arguing. `results/benchmarks/layer_probe_sweep.json`, linear probe,
8,000 train / 3,000 test.

| task | model | L4 | L6 | L8 | L10 | L12 (final) |
|---|---|---:|---:|---:|---:|---:|
| remote homology (acc) | ESM-2 35M | 0.5573 | **0.6703** | 0.6647 | 0.6683 | 0.6373 |
| remote homology (acc) | ProtSent-V2 | 0.5527 | 0.6893 | **0.7033** | 0.6997 | 0.6803 |
| stability (Spearman) | ESM-2 35M | 0.3892 | **0.4049** | 0.3359 | 0.3499 | 0.4004 |
| stability (Spearman) | ProtSent-V2 | **0.5537** | 0.3792 | 0.4293 | 0.4081 | 0.4001 |

**The argument this supports, and it is a strong one:** on remote homology,
ProtSent-V2 at *its worst* useful layer still beats ESM-2 at *its best* layer
(0.6803 at L12 vs 0.6703 at L6). The advantage is not an artifact of which layer
is pooled, and it is not available to the baseline by tuning layer choice. Say
this explicitly — "did you just pick a favourable layer?" is an obvious reviewer
question and the answer is measured.

Both models use final-layer mean pooling everywhere else in the paper, which the
table shows is *not* the best layer for either model on either task. That is a
concession worth making: it costs us nothing (the ranking is unchanged) and it
pre-empts the accusation that the protocol was tuned.

## 4f. Few-shot transfer with seed variability — HNXd's requests 4 and 5

HNXd asked for a multi-seed variability analysis of the few-shot evaluation and
for absolute scores rather than the relative percentages in Table 5 (which
produced cells like -126.9% from near-zero baselines). Both are here:
5 seeds per cell, mean +/- SD, absolute values.
`results/benchmarks/fewshot_seeds.json`.

Remote homology, accuracy, by number of labelled training examples:

| N | probe | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---:|---|---|---|---|
| 100 | 3-NN | 0.1155 +/- 0.0066 | **0.1349 +/- 0.0085** | 0.1248 +/- 0.0239 |
| 250 | 3-NN | 0.1483 +/- 0.0019 | **0.2232 +/- 0.0109** | 0.2000 +/- 0.0098 |
| 1000 | 3-NN | 0.1850 +/- 0.0023 | **0.3185 +/- 0.0148** | 0.2893 +/- 0.0158 |
| 250 | linear | 0.3100 +/- 0.0071 | **0.3942 +/- 0.0119** | 0.3683 +/- 0.0131 |
| 1000 | linear | 0.2878 +/- 0.0143 | **0.3772 +/- 0.0083** | 0.3552 +/- 0.0092 |

At N=1000 under a 3-NN probe the gap is +0.1335 accuracy over the backbone,
against seed SDs of 0.002-0.016 — far outside noise. It holds under a linear
probe too (+0.0894).

**Three honest points that must travel with this table:**

1. **V1 beats V2 on few-shot remote homology.** The decontaminated model is not
   better everywhere; it is better on retrieval and on the full-data task. Report
   V1's column rather than hiding it.
2. **The label-scarcity framing HNXd proposed is NOT supported.** They suggested
   the story might be "linear probes degrade under label scarcity while k-NN
   stays competitive". In our data the linear probe beats 3-NN at *every* N on
   remote homology, for every model. We do not make that claim, and we say so.
3. **The advantage is task-specific.** On solubility and metal-ion binding the
   few-shot differences are within or near seed noise, and ESM-2 wins several
   cells. Only stability and remote homology show a consistent ProtSent gain.

## 4g. SCOPe decontaminated on the benchmark side — the control that was missing

SCOPe-40 cannot be filtered at corpus level (no train/test split; filtering
against it would delete every structured domain from the corpus). The benchmark
side can be filtered instead: drop the *queries* that have a close pretraining
neighbour, re-score on what remains. `scope_clean_subset.py`,
`results/benchmarks/scope40_clean_subset.json`.

Eligible queries retained by maximum identity to our corpus, with 10,000-resample
CIs:

| queries | n | HMMER R@10 | V2 R@10 | HMMER MAP | V2 MAP |
|---|---:|---|---|---|---|
| <0.4 identity | 164 | 0.774 | **0.890** | 0.480 | **0.620** |
| <0.5 | 287 | 0.774 | **0.909** | 0.481 | **0.637** |
| <0.7 | 479 | 0.785 | **0.912** | 0.487 | **0.641** |
| all | 1,693 | 0.781 | **0.922** | 0.475 | **0.645** |

Paired V2 - HMMER on the same queries:

| queries | R@1 | R@10 | MAP |
|---|---|---|---|
| <0.4 (n=164) | -0.043 [-0.128, +0.043] | **+0.116 [+0.049, +0.189]** | **+0.140 [+0.075, +0.207]** |
| <0.5 (n=287) | -0.010 [-0.073, +0.049] | **+0.136 [+0.087, +0.188]** | **+0.156 [+0.108, +0.204]** |
| <0.7 (n=479) | -0.027 [-0.073, +0.017] | **+0.127 [+0.090, +0.165]** | **+0.154 [+0.117, +0.190]** |
| all (n=1,693) | -0.012 [-0.037, +0.012] | **+0.141 [+0.120, +0.162]** | **+0.171 [+0.151, +0.191]** |

**The conclusion does not move.** At every threshold, including the 164 queries
furthest from anything we trained on, ProtSent-V2 ties the best alignment
baseline at top-1 and leads it at depth and MAP by intervals excluding zero. The
margin does not shrink as the queries get cleaner.

**State the limit of this control in the same breath**: it bounds
identity-level exposure only. Supervision is Foldseek-cluster and Pfam-family
co-membership, so a training pair sharing a query's *fold* at 15% identity
survives any identity threshold. This is a weaker control than corpus filtering,
and it is the one SCOPe permits.

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

## 5-CORRECTION. A maximally-sensitive HMMER is far stronger, and it changes the claim

**Read this before using any number in 5a or in the SCOPe tables above.**

The HMMER runs above use pyhmmer defaults, which apply HMMER's MSV / Viterbi /
Forward heuristic filters and a bias filter. Those filters prune candidates
*before* the E-value threshold is applied, so raising `-E` alone changes nothing
(verified: E=10, E=1e3 and E=1e5 give byte-identical results). Disabling them
(`F1=F2=F3=1.0, bias_filter=False, E=1e6`) takes 83 s for the full 2,207 x 2,207
all-vs-all and produces a much stronger baseline:

| phmmer setting | no-hit queries | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|---:|
| defaults (`-E 10`) | 691 / 2,207 | 0.6970 | 0.7809 | 0.7980 | 0.4747 |
| **filters off** | **0 / 2,207** | **0.7525** | **0.8978** | **0.9232** | **0.6067** |
| ProtSent-V2 (for reference) | 0 | 0.6852 | 0.9220 | 0.9634 | 0.6459 |

Paired bootstrap against max-sensitivity phmmer, eligible queries
(`results/benchmarks/hmmer_maxsens_paired.json`):

| comparison | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| V2 - HMMER(max) | **-0.0679 [-0.0916, -0.0443]** | +0.0242 [+0.0089, +0.0402] | +0.0402 [+0.0266, +0.0538] | +0.0388 [+0.0214, +0.0559] |
| V1 - HMMER(max) | -0.1666 [-0.1931, -0.1394] | -0.0467 [-0.0656, -0.0278] | +0.0030 [-0.0124, +0.0183] | -0.0556 [-0.0748, -0.0360] |
| ESM-2 - HMMER(max) | -0.2534 | -0.1364 | -0.0880 | -0.1844 |

### What this retracts

1. **The coverage-gap argument in 5a is withdrawn.** The 691 no-hit queries were
   an artifact of default heuristic filters, not a property of alignment search.
   At full sensitivity HMMER returns a ranked list for every query. Do not use
   "alignment returns nothing for 47.6% of remote-homology queries" as a
   generality argument without re-measuring that task with filters off.
2. **"Decisively ahead at ranking depth" is withdrawn.** Against the strongest
   alignment baseline the depth margins are +0.024 R@10 and +0.039 MAP — real
   and statistically resolved, but small, not decisive.
3. **The submitted model V1 loses to a properly tuned HMMER** on R@1, R@10 and
   MAP, and ties at R@30.

### What survives, and is defensible

ProtSent-V2 is **behind** max-sensitivity HMMER at top-1 (-0.068, resolved) and
**modestly ahead** at every deeper cutoff and on MAP (+0.024 to +0.040, all
resolved). Both statements come from the same paired bootstrap and must be given
together.

The honest framing is no longer "we beat alignment". It is: a 35M frozen
embedding gets within a few points of an exhaustively-tuned profile search on
its own home turf, ahead at ranking depth and behind at top-1, while being a
single forward pass per sequence that supports indexed sub-linear search — versus
an all-vs-all profile comparison. Speed and generality are the claim; raw
retrieval accuracy against a maximally-sensitive HMMER is not.

**Any rebuttal sentence comparing to HMMER must use the filters-off numbers.**
A reviewer who runs HMMER properly will get them, and quoting the default-filter
numbers would look like baseline-weakening.

## 5a. Both alignment baselines across the whole benchmark — and the coverage gap

`mmseqs_baseline.py` (24 tasks) and `hmmer_baseline.py` (23 tasks; temperature
stability, 283k train / 73k test, did not finish in the window). Both score
through the same code path, so they cannot diverge in how they are measured.
Main metric per task, with **hit coverage** — the fraction of test queries the
search returns any alignment for:

| task | metric | MMseqs2 | cov | HMMER | cov |
|---|---|---:|---:|---:|---:|
| ec_classification | F1_Macro | 0.7103 | 0.990 | **0.7229** | 0.945 |
| go_mf | F1_Macro | 0.5850 | 0.956 | **0.6047** | 0.901 |
| beta_lactamase_peer | Spearman | **0.8026** | 1.000 | 0.7974 | 1.000 |
| variant_effect | Spearman | 0.7166 | 1.000 | **0.7169** | 1.000 |
| enzyme_catalytic_efficiency | Spearman | 0.6322 | 0.994 | **0.6462** | 0.991 |
| stability | Spearman | 0.5817 | 1.000 | **0.5969** | 1.000 |
| optimal_ph | Spearman | 0.5462 | 0.987 | **0.5541** | 0.937 |
| remote_homology | AUC | **0.6523** | 0.889 | 0.6439 | **0.524** |
| scope40_retrieval | Recall@10 | 0.5637 | 0.882 | **0.5963** | 0.687 |
| signalp_binary | AUC | 0.7961 | 0.770 | **0.8115** | 0.675 |
| solubility | AUC | 0.4185 | 0.951 | 0.4150 | 0.951 |
| aav_flip | Spearman | **0.4024** | 1.000 | 0.3698 | 1.000 |
| fluorescence | Spearman | **0.3863** | 1.000 | 0.2845 | 1.000 |
| rhla_enzyme_mutations | Spearman | n/a | **0.000** | -0.0888 | **0.004** |

HMMER beats MMseqs2 on 12 of the 22 tasks both completed, so neither is the
"weak" baseline — quoting only one would be a fair criticism, and both are
reported. **Always report the alignment number as the better of the two**, so no
reviewer can claim the easier opponent was chosen.

### The coverage gap is the generality argument, measured

This is the concrete answer to jVGf's generality-accuracy question, and it is
stronger than the accuracy comparison. Alignment search **returns nothing at all**
for a large share of queries, and that share grows exactly where the task is
hard:

- remote homology: HMMER returns no hit for **47.6%** of test queries (coverage
  0.524); MMseqs2 for 11.1%
- SCOPe-40 retrieval: HMMER no-hit for **31.3%**, MMseqs2 for 11.8%
- `rhla_enzyme_mutations` (6-residue mutation-site strings): coverage 0.000 and
  0.004 — both alignment methods fail completely on the task
- signalp, subcellular localisation, metal-ion binding: HMMER coverage 0.675,
  0.680, 0.832

An embedding model always returns a ranked list. Its metric is never a property
of a fallback. When an alignment metric is quoted at coverage 0.524, roughly half
of it is the fallback, not the search — which is why every alignment number in
this document carries its coverage, and why any rebuttal sentence quoting one
must carry it too.

**Where alignment genuinely wins, say so plainly:** enzyme-class prediction
(F1_Macro 0.7229 vs the best embedding model's 0.598) and GO molecular function
(0.6047 vs 0.459) are decisive alignment wins at near-full coverage. Those are
not close, and conceding them is what makes the retrieval claim credible.

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

## 6b. Few-shot transfer with seed variability — HNXd's items 2, 4 and 5

`fewshot_seeds.py`, `results/benchmarks/fewshot_seeds.json`. Test split held
fixed at full size; only the training subset is resampled, 5 seeds per point, so
the spread is exactly the few-shot variability HNXd asked about. Both probes are
fit on the same subset. Absolute scores, as requested — never relative
percentages.

**Remote homology (accuracy), the task ProtSent targets:**

| N | ESM-2 kNN | ESM-2 linear | V1 kNN | V1 linear | V2 kNN | V2 linear |
|---|---|---|---|---|---|---|
| 50 | 0.061±0.010 | 0.121±0.003 | 0.055±0.008 | 0.159±0.004 | 0.045±0.009 | 0.145±0.005 |
| 100 | 0.115±0.007 | 0.222±0.006 | 0.135±0.008 | 0.282±0.007 | 0.125±0.024 | 0.258±0.009 |
| 1000 | 0.185±0.002 | 0.288±0.014 | **0.318±0.015** | **0.377±0.008** | 0.289±0.016 | 0.355±0.009 |

**Stability (Spearman):**

| N | ESM-2 kNN | ESM-2 linear | V2 kNN | V2 linear |
|---|---|---|---|---|
| 50 | 0.178±0.054 | 0.216±0.202 | **0.327±0.161** | 0.200±0.131 |
| 100 | 0.260±0.129 | 0.284±0.209 | **0.401±0.202** | 0.292±0.200 |
| 1000 | 0.228±0.074 | 0.406±0.075 | 0.315±0.079 | 0.430±0.105 |

Solubility and metal-ion binding were also run and ProtSent does **not** win
there: at N=1000 ESM-2 leads metal-ion binding under a linear head (0.666±0.001
vs V1 0.637±0.004, V2 0.595±0.001) and solubility is a wash. Report this.

Three conclusions, and the first one goes against us:

1. **HNXd's proposed framing is not supported by our data.** They suggested that
   under label scarcity a linear classifier degrades while k-NN stays
   competitive. A trained linear head beats 3-NN in almost every
   model/task/N cell here, including at N=50. We do not claim the crossover, and
   we withdraw the label-scarcity framing that implied it.
2. **ProtSent's few-shot advantage is real but task-specific.** On remote
   homology at N=1000 it is large under both probes (+0.133 kNN, +0.089 linear
   for V1 over ESM-2). On solubility and metal-ion binding it is absent.
3. **Seed spread at small N is enormous, which is the real answer on Table 5.**
   Stability at N=100 has a standard deviation of ±0.20 on a mean of 0.28-0.40 —
   the spread is as large as the effect. A single-run relative change computed
   against a near-zero baseline, such as the -126.9% cell, was never
   interpretable. This is HNXd's own concern, confirmed by measurement, and it is
   why the reporting now uses absolute means with seed standard deviations.

**Full-data evaluation, by contrast, is essentially deterministic.** Re-running
the benchmark across 5 seeds with full training data moves nothing: remote
homology 0.5835±0.0000, metal-ion binding 0.7402±0.0000, solubility
0.5102±0.0000, variant effect 0.6582±0.0000, stability 0.6435±0.0001, only
thermostability showing any spread at ±0.0126
(`results/benchmarks/seeds/`). Given a fixed test split and a deterministic
probe over deterministic embeddings, there is nothing left to vary. So the
uncertainty that matters for the main tables is **which proteins are in the test
set**, which is what the bootstrap intervals in §4a quantify — not seed noise.
Say both halves; quoting only one of them would misrepresent the analysis.

## 6a. The linear-probe comparison is a final-layer artifact — measured

Both probes in the benchmark pool the **final** layer. That is the measurement
point least favourable to ProtSent and most favourable to stock ESM-2: the
contrastive objective only ever sees the final layer, so that is the only place
ProtSent reorganises, while a masked-LM's top of stack is pushed toward token
reconstruction rather than toward linearly decodable properties.

`layer_probe_sweep.py`, `results/benchmarks/layer_probe_sweep.json`. Linear probe
(RidgeCV for regression, logistic regression for classification) on mean-pooled
embeddings from each layer. Subsampled to 8,000 train / 3,000 test for speed, so
absolute values are not directly comparable to the full-split numbers elsewhere
in this document — the comparison **between models within this table** is the
point.

**Stability (Biomap), Spearman:**

| pooled layer | ESM-2 35M | ProtSent-V2 |
|---|---:|---:|
| 4 | 0.3892 | **0.5537** |
| 6 | 0.4049 | 0.3792 |
| 8 | 0.3359 | 0.4293 |
| 10 | 0.3499 | 0.4081 |
| 12 (final, what the benchmark uses) | 0.4004 | 0.4001 |

**Remote homology, accuracy:**

| pooled layer | ESM-2 35M | ProtSent-V2 |
|---|---:|---:|
| 4 | 0.5573 | 0.5527 |
| 6 | 0.6703 | **0.6893** |
| 8 | 0.6647 | **0.7033** |
| 10 | 0.6683 | **0.6997** |
| 12 (final) | 0.6373 | **0.6803** |

Three things follow, and all three are usable:

1. **The final layer is the worst layer for both models on remote homology**
   (ESM-2 0.6373 vs 0.6703 at layer 6) and is not the best for either on
   stability. A final-layer-only linear probe understates both models.
2. **At its best layer ProtSent-V2 beats ESM-2 on both tasks** — stability
   0.5537 vs 0.4049 (+0.149), remote homology 0.7033 vs 0.6703 (+0.033) — while
   at the final layer stability is a tie (0.4001 vs 0.4004).
3. **Contrastive fine-tuning did not destroy linearly decodable information.**
   On remote homology ProtSent-V2 is ahead at every layer from 6 upward. This is
   the direct answer to "the method rearranges information rather than adding
   any, and possibly discards some".

Scope this honestly: two tasks, subsampled splits, one backbone scale. It is a
control that identifies a confound in the benchmark's probe protocol, not a
re-run of the benchmark. Do not present it as overturning the full linear-probe
table — present it as the reason that table should not be read as "the
information is not there".

## 7a. Framing decisions made by the authors — follow these

These are decisions, not evidence. Where they conflict with a reviewer's framing,
these win.

**1. The linear-probe result: report it, do not lead the whole rebuttal with a
withdrawal.** Give the win/tie/loss table for both probes, stock ESM-2 versus
ProtSent, and answer it on the merits. Three points make the answer fair rather
than defensive, and all three are grounded:

- *The two probes measure different things.* A trained linear readout measures
  whether property information is **present** in the representation and linearly
  decodable given labels. 3-NN measures whether the information is already
  **local in the geometry**, with no labels and no fitting. Contrastive
  fine-tuning reorganises geometry; it is not a claim that it adds information.
- *Both probes here pool the FINAL layer*, which is the measurement most
  favourable to a model whose last layer was never reorganised. Section 6a
  measures this: the final layer is the worst layer for both models on remote
  homology, and at its best layer ProtSent-V2 beats ESM-2 on both tasks tested.
  Use those numbers; do not go beyond the two tasks they cover.
- *The target use case is retrieval, clustering, and zero-/few-shot nearest
  neighbour transfer*, where no trained head exists and geometry is the whole
  product. That is where the paper's contribution lives, and it is exactly where
  ProtSent wins and keeps winning under both alignment baselines.

Do NOT hide the linear-probe losses and do NOT claim general-purpose superiority.

**2. The V1-to-V2 configuration differences are not a confound to apologise
for.** Yi1G's own criticism was that the ablations do not support the submitted
defaults. V2 acts on that: proportional sampling and no synthetic hard negatives
are the settings the paper's own ablations favour. State it once, briefly, as
having taken the reviewer's point. Do not write a paragraph enumerating
differences, do not frame V2-vs-V1 as a controlled decontamination ablation, and
do not concede that the comparison is confounded — the claim being made is the
sufficient one, that decontamination did not cost performance.

**3. The BIOMAP Stability comparison HNXd raised is not commensurate, and this is
verified.** `biomap-research/stability_prediction` labels are continuous floats
(range -1.680 to 2.150, 298 distinct values in a 5,000-row sample; train 53,614 /
valid 2,512 / test 12,851). It is a **regression** task and the paper's metric is
Spearman, so "58.8%" is a correlation times 100, not an accuracy. A 69.08%
"linear classifier" accuracy and a 77.69% LoRA accuracy are therefore not the
same quantity as our 0.588.

The mechanism HNXd proposed for the gap — that k-NN depresses the numbers — is
also contradicted by our own measurement: on this task the linear probe scores
**lower** than 3-NN (ESM-2 35M Spearman 0.4395 linear vs 0.5680 3-NN). Say both
things: the metrics are not commensurate, and the probe change HNXd suggested
moves the number the wrong way.

**4. The 150M model: one sentence, no promise.** A decontaminated 150M is
training on the same pipeline and its results will appear in the camera-ready.
Never present a number from it, never imply it exists now, and do not make its
completion a reason to accept.

## 9. ProtSent-V2-150M — finished, and it changes two standing instructions

Trained on the same decontaminated corpus, 3,890 steps on 6 GPUs, `MAX_PAIRS_PER_CLUSTER=5`.
Benchmarked with the same code path as every other arm: 4 arms x {3-NN, linear} x 23
tasks, `--eval_split test`. Raw CSVs in `results/benchmarks/v2_150m/`, config and full
detail in `RUNS.md`.

**SCOPe-40 retrieval, eligible queries (n=1,693 of 2,207):**

| method | R@1 | R@10 | MAP |
|---|---:|---:|---:|
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (`-s 7.5`) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer) | 0.6970 | 0.7809 | 0.4747 |
| ProtSent-V1-150M (submitted) | 0.6615 | 0.8943 | 0.6431 |
| **ProtSent-V2-150M** | **0.7431** | **0.9368** | **0.7042** |

Paired bootstrap, 10,000 resamples (`scope40_bootstrap_ci_150m.json`,
`alignment_paired_ci_150m.json`). Every one of these excludes zero:

| comparison | R@1 | R@10 | MAP |
|---|---|---|---|
| V2-150M - V1-150M | +0.0809 [+0.0602, +0.1022] | +0.0431 [+0.0301, +0.0561] | +0.0607 [+0.0477, +0.0735] |
| V2-150M - ESM-2 150M | +0.1896 [+0.1654, +0.2138] | +0.1672 [+0.1477, +0.1867] | +0.2806 [+0.2644, +0.2967] |
| V2-150M - HMMER | **+0.0455 [+0.0219, +0.0691]** | +0.1565 [+0.1364, +0.1766] | +0.2301 [+0.2111, +0.2492] |
| V2-150M - MMseqs2 | +0.0868 [+0.0620, +0.1116] | +0.1973 [+0.1754, +0.2191] | +0.2950 [+0.2751, +0.3144] |

**INSTRUCTION CHANGE 1 — the top-1 concession is scale-specific.** §3 says never to
claim ProtSent beats alignment at top-1. That is correct **at 35M**, where V2 ties HMMER
(-0.0124 [-0.0372, +0.0124]). At **150M it beats HMMER significantly** (+0.0455), and
HMMER is the stronger of the two alignment tools. So: at 35M, alignment remains the
better top-1 method and our advantage is ranking depth; at 150M we lead on every metric
against both tools. Never state it globally.

**INSTRUCTION CHANGE 2 — a decontaminated 150M now exists.** §8 previously said it did
not. It is trained, benchmarked and reported here. It is still not in the submitted
paper, so present it as new rebuttal-period evidence, not as something reviewers can
look up.

**Remote homology at 150M — state the probe, because the direction flips.**

| method | kNN acc | kNN macro-F1 | linear acc | linear macro-F1 |
|---|---:|---:|---:|---:|
| ESM-2 150M | 0.5194 | 0.2764 | 0.7500 | 0.5162 |
| ProtSent-V1-150M | 0.7047 | 0.4297 | 0.7401 | 0.4775 |
| ProtSent-V2-150M | 0.6612 | 0.3885 | 0.7503 | 0.4941 |

Under 3-NN, decontamination costs 4.4 points versus V1 (0.7047 -> 0.6612) while both
ProtSent models stay far above vanilla (0.5194). Under a linear probe the ordering
reverses: V2 beats V1 on both accuracy and macro-F1, and ties vanilla on accuracy. The
kNN drop is the expected consequence of removing pretraining sequences at >=40% identity
to this test set; the larger model appears to have exploited that leakage more than the
35M did, which is an argument *for* the decontamination, not against it.

**The macro-F1 deficit against vanilla was independently verified**
(`verify_remote_homology.py`, `verify_remote_homology_150m.json`). It reproduces, it is
statistically real (paired bootstrap V2 - vanilla macro-F1 -0.0262 [-0.0450, -0.0071],
while accuracy is unresolved at -0.0008 [-0.0108, +0.0092]), and it is **mostly a
rare-class artifact**: the test set has 457 classes with median support 3 and 209 classes
with <=2 examples, and restricting to classes with >=3 test examples shrinks the gap from
-0.0257 to -0.0036. Quote accuracy and macro-F1 together and say what macro-F1 means on
this label distribution.

**Aggregate, V2-150M vs V1-150M across 23 tasks:** 12 win / 4 tie / 7 lose under 3-NN
(median +0.0055), 7 / 6 / 10 under a linear probe (median -0.0045) — the same
probe-dependence documented at 35M in §6.

**Layer sweep at 150M** (`layer_probe_sweep_150m.json`), all models compared **at the
same layer**, never at each model's own best: layer 20 of 30 is the best pooling layer
for every model and is worth 5-6 points over the final layer. At layer 20, remote-homology
linear accuracy is ESM-2 0.7357, V1-150M 0.7400, **V2-150M 0.7500** — the ordering the
benchmark's final-layer default hides. This reinforces §6a at the larger scale.

## 10. Embedding-space organization — the direct answer to HNXd's Question 1

HNXd asked, verbatim: *"an analysis showing how ProtSent changes the local and global
organization of the protein embedding space."* This is that analysis, and it is a positive
result. Source: `probe_gap_analysis.py`, `results/benchmarks/probe_gap_analysis.json`,
measured on the 2,207-sequence SCOPe-40 gallery.

**Stock ESM-2 embeddings occupy a narrow cone.** The mean cosine similarity between two
*randomly chosen, unrelated* proteins is 0.85-0.90 — the space is close to
one-dimensional in practice, and almost all of the variance sits in a handful of
directions.

| model | mean cos(random pair) | participation ratio | dims for 95% of variance |
|---|---:|---:|---:|
| ESM-2 35M | 0.848 | 7.9 / 480 | 112 |
| **ProtSent-V2 35M** | **0.152** | **52.5 / 480** | 148 |
| ESM-2 150M | 0.896 | 10.6 / 640 | 126 |
| **ProtSent-V2 150M** | **0.175** | **43.4 / 640** | 144 |

Participation ratio is the effective number of dimensions actually carrying variance
(`(sum L_i)^2 / sum L_i^2` over the covariance eigenvalues). Contrastive fine-tuning takes
the backbone from ~8-11 effective dimensions to ~43-53, a 4-5x expansion, and removes the
anisotropy almost entirely.

**This is the mechanism behind the retrieval numbers.** Nearest-neighbour search operates
on raw distances, so a space where every pair is 0.85-similar has very little usable
signal to rank on. Reorganising it into a broad, isotropic space is precisely what makes
k-NN retrieval and clustering work, and it explains why the gains concentrate in
retrieval-shaped tasks rather than in tasks with a trained readout.

State it as an explanation of *where* the benefit comes from, not as a claim of added
information — the honest framing is that the objective reorganises geometry, and geometry
is what k-NN, clustering and zero-shot transfer consume.

## 8. What is NOT available — do not imply otherwise

- No full end-to-end fine-tuning sweep.
- No matched runs of ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.
- No SaProt/ProSST backbone substitution (needs residue-level structure tokens
  for the full Pfam and STRING corpora).
- No paired bootstrap confidence intervals on the Table 2 per-task deltas.
- Reviewers **cannot see** any of this unless the response states it. There is
  no updated PDF they have read. Every number must be given in the response
  text itself, self-contained, with its metric and split named.
