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
39,170 s (10 h 53 m), 887.5 samples/s. Differences from V1 beyond
decontamination, which **must be stated whenever V1 and V2 are compared**:
7x1024 effective batch (vs 1x1024), no synthetic hard negatives, proportional
multi-dataset sampling, one epoch.

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

**The top-1 story changed.** A tuned MMseqs2 beats the *submitted* model at R@1
(0.5029 vs 0.4490). It does not beat the decontaminated retrained model
(0.5029 vs 0.5256). Be precise about which model each claim refers to; do not
retro-claim a top-1 win for the submitted paper.

Remote homology — **the task the corpus was actually filtered against**:

| model | kNN accuracy | linear-probe accuracy |
|---|---:|---:|
| ESM-2 35M | 0.5835 | 0.6868 |
| ProtSent-V1 | 0.6587 | 0.6899 |
| **ProtSent-V2** | **0.6668** | **0.7016** |

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
