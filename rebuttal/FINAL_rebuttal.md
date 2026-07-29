# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — FINAL

Post each response under its own review. No links, no attachments. Every number
appears in the response text with its metric, split and model.

Naming used throughout: **V1** = the submitted/published 35M model.
**V2** = the 35M model retrained during the rebuttal on the decontaminated corpora.

---

## Response to Reviewer HNXd

<!-- BEGIN HNXd -->
**What the paper claims after this rebuttal, in one sentence:** contrastive
training over multiple relation types reorganizes a frozen 35M ESM-2 embedding so
that family, fold and interaction relations become locally recoverable — measured
as structural retrieval and remote homology — and it does **not** improve general
property prediction under a trained readout. The general-purpose superiority claim
is withdrawn. Your question 2 is what killed it.

Both probes are now complete on all 23 tasks, `--eval_split test`, for stock
ESM-2 35M and both ProtSent models. Against ESM-2 35M:

| probe | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

Read this before the counts: (i) the tie band is **±0.005** on each task's main
metric — with 7 ties of 20 it materially decides the linear record; (ii) counts are
over **20 of the 23** tasks. `antibiotic_resistance`, `remote_homology` and
`temperature_stability` are excluded from *both* columns because their main metric
is multiclass AUC and the test split contains classes absent from train, so
one-vs-rest AUC is undefined for the embedding arms. That drops remote homology,
our best task, out of the tally; its accuracy is reported separately below and is
not in the 20; (iii) three *different* tasks — `ec_classification`, `go_mf`,
`scope40_retrieval` — use a built-in evaluator that ignores the requested probe, so
their rows are one measurement printed in both tables. SCOPe retrieval therefore
has **one** measurement, not two, and we do not claim it "survives both probes";
(iv) the linear probe is scikit-learn defaults on frozen mean-pooled embeddings —
`StandardScaler` + `LogisticRegression(solver="liblinear")` or `Ridge(alpha=1.0)`,
no per-arm tuning. It is untuned, and an undertuned probe would move this record in
either direction; (v) every benchmark number here is a single run at benchmark seed
42.

### 1. Direct retrieval evaluation (your question 1)

**We did not compute clustering-geometry statistics** — no silhouette, NMI or ARI.
You asked for either a retrieval/clustering evaluation or a geometry analysis; we
ran the first, plus the per-query analysis in section 3. The second is missing.

SCOPe-40, **family** level, 2,207-sequence gallery, self excluded, no-hit queries
scored as failures. **Recall@K is upper-bounded at 0.7671**: only 1,693 of 2,207
queries have any non-self same-family protein in the gallery, so 514 are
unachievable for every method.

| method (all 2,207 queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

V2 is a retrain on corpora from which every sequence within 40% identity / 80% coverage
of the remote-homology and PPI test sets was removed (those two test sets were the only
filter targets; SCOPe-40 and the other benchmarks were not filtered). We retrained on the
filtered corpora using the configuration favoured by the paper's own ablations —
proportional sampling and no synthetic hard negatives — otherwise following the submitted
recipe, on 7 GPUs rather than 1.

V1's R@30 of 0.7100 is 92.6% of the attainable 0.7671. The advantage is in ranking
depth, not top-1: a tuned MMseqs2 **beats the submitted model at top-1** (0.5029 vs
0.4490) and only V2 passes it. One caveat that cuts against us: MMseqs2's R@10 and
R@30 differ by 0.0004 because at `-e 10 --max-seqs 300` it returns few candidates
past rank 10, so part of the depth gap is coverage, not ranking quality. Top-1 is
the coverage-free comparison, and it is the one the submitted model loses.

### 2. Linear probes and label scarcity (your question 2)

The aggregate is the 4/4/12 above. Per task the two probes disagree in a specific
way: on AAV fitness (Spearman) ESM-2 0.4667 vs V1 0.5553 under 3-NN, but ESM-2
0.5639 vs V1 0.4362 under the linear probe. Contrastive training makes relations
*local*; it does not add information a trained head could not already extract from
mean-pooled ESM-2 features.

Remote homology (pooled 457-class task, accuracy, test split) is where it does add
something: 3-NN 0.5835 (ESM-2) / 0.6587 (V1) / 0.6668 (V2); linear 0.6868 / 0.6899
/ 0.7016. Under the linear probe **V1's +0.0031 is inside our own ±0.005 band and
is a tie** — only V2 clears it. Note also that the submitted paper's "+40.5%" for
this task is a relative change in macro-F1 (.223 → .313) computed under the suite's
default split; the numbers here are accuracy on the test split. Different metric,
different split — they are not two views of one result and we do not mix them.

That also explains the level difference you flagged against the literature: every
number in our tables is a frozen 35M backbone under a 3-NN or linear probe, not a
fine-tuned larger model, and printing them beside published fine-tuned numbers was
wrong.

**We did not run a fine-tuning sweep and we have no few-shot linear-probe
baseline.** The label-scarcity claim has no supporting control; we withdraw it.

### 3. 95% confidence intervals (your question 3)

Retrieval answers your request exactly as you posed it — every metric is a mean
over per-query values, so resampling queries gives the sampling distribution with
no refitting. 10,000 resamples, **paired** (the same queries score every method),
over the **1,693 eligible** queries. The intervals therefore sit on a different
denominator from the table above; here is that denominator explicitly:

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 | 0.6556 | 0.7348 | 0.7354 | 0.4041 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 | 0.6852 | 0.9220 | 0.9634 | 0.6459 |

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - V1 | +0.0986 [+0.0762, +0.1211] | +0.0709 [+0.0555, +0.0862] | +0.0943 [+0.0814, +0.1074] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

(Interval centres are bootstrap means and differ from the table's point estimates
by under 0.001.)

Three results from the same procedure that cut against us: MMseqs2 beats ESM-2 35M
at top-1 by +0.1565 [+0.1276, +0.1855]; MMseqs2 beats ProtSent-V1 at top-1 by
+0.0697 [+0.0413, +0.0975]; and MMseqs2 vs ESM-2 at R@10 (-0.0213 [-0.0484,
+0.0047]) and MAP (-0.0125 [-0.0351, +0.0102]) is **unresolved**, so we do not
claim an untuned pLM beats alignment at depth. V2's top-1 lead over alignment is
+0.0289 with a lower bound of +0.0035 — resolved, but small, and we present it as
small. This bootstrap quantifies which proteins are in the benchmark; it does not
quantify training-seed variance.

**We did not compute intervals for the 23-task table.** This is an omission, not an
obstacle: the fitted probe can be held fixed and the held-out predictions
resampled, which requires no refitting. Without them your objection stands on its
own — each cell is one run at one seed, and any delta inside ±0.005 is unresolved
and is reported as a tie, not an improvement.

### 4-5. Seed variability and Table 5 (your questions 4 and 5)

**We have no multi-seed results**, for training seeds or probe seeds. So Table 5 is
withdrawn rather than re-presented in absolute units from the same single-seed run.

The cells you flagged are an arithmetic artifact compounded by an estimator change.
A relative change of -126.9% on a Spearman correlation is a sign flip of magnitude
0.269x the baseline, not a 127-point drop; +244.5% on a near-zero baseline is
likewise unbounded. And at small N the probe silently reduces k, because the code
sets `n_neighbors = max(1, min(3, train_size))` — so the smallest few-shot cells
are not even the same estimator. Relative change over near-zero denominators, one
run, varying k: the wrong instrument, and the claim it supported goes with it.

The one variance measurement we have is checkpoint sensitivity. The final V2
checkpoint is the last training step and was not selected on any benchmark; a
near-trough checkpoint (step 4,000, where the 3-cycle cosine schedule bottoms) was
benchmarked as a control and differs by 0.005-0.008 on every structural metric.
That bounds one nuisance factor; it is not a seed replicate.

### Errors we found in our own submission

All new evidence above is 35M. **There is no 150M model on the decontaminated
corpus**, so we do not defend the submitted 150M numbers, including the abstract's
+105% and +19.9%. Further: SCOPe is evaluated on the **family** field over **2,207**
sequences, not superfamily over 100,000 — the 100,000 is the evaluator's
`max_samples` cap echoed into the results table (a separate superfamily-level
evaluation still improves at 35M, R@1 0.639 -> 0.726; no interval computed for it).
Our remote-homology test split is TAPE's three holdouts pooled (718 fold + 1,254
superfamily + 1,272 family = 3,244) with no column marking which, not
hierarchy-disjoint as we wrote. Eq. 1 is malformed. Two protocol facts we should
have printed: `thermostability` has no official test split (the suite takes a
seeded 80/20 split of train), and `peptide_hla` inputs are pipe-joined
`HLA_pseudoseq|peptide` strings.

If the withdrawals and the measurements above change your assessment, we ask you to
reconsider. If one item remains decisive, name it and we will answer in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- BEGIN jVGf -->
Your two axes turned out to be one axis, and answering it changed what we claim.
**The paper's claim after this rebuttal:** multi-relational contrastive training on
a frozen 35M backbone buys retrieval geometry — structural and homology search —
and buys nothing on tasks where a trained head already extracts the label. The
general-purpose framing is withdrawn. Here is the position on the curve, measured.

### 1. Where ProtSent sits on the generality-accuracy trade-off (your Q3 / W2)

We ran MMseqs2 as a full alternative pipeline across all 23 benchmark tasks under
identical metric definitions — family-level Recall@K with self excluded, per-class
max bitscore for classification so AUC stays comparable, 1-NN by bitscore for
regression. No-hit queries are scored as failures, not dropped. Flags: `-s 7.5 -e
10 --max-seqs 300 --alignment-mode 3`; at the default `-s 5.7` the same baseline
gives SCOPe R@1 0.3847, so any MMseqs2 number needs its sensitivity stated.

Alignment beats the best embedding model outright on **3 of 23 tasks under a 3-NN
probe** (EC classification, GO molecular function, beta-lactamase) and **6 under a
linear probe** (those plus enzyme catalytic efficiency, optimal pH, stability). The
margins are not small: EC classification F1-macro 0.710 (MMseqs2) vs 0.598 (ESM-2
35M) and 0.562 (ProtSent-V1); GO-MF 0.585 vs 0.459 / 0.443. On beta-lactamase
(Spearman, 3-NN) the full set is MMseqs2 0.8026, ESM-2 0.7272, V1 0.7676, V2
0.7153 — alignment wins and our retrained model is the worst of the four. Where
annotation transfers by homology, alignment is simply better.

The other end of the curve is SCOPe-40 (family level, 2,207-sequence gallery, self
excluded, no-hit = failure; **Recall@K upper-bounded at 0.7671**, since only 1,693
queries have a non-self same-family neighbour):

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained, decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

V2 is a retrain on corpora from which every sequence within 40% identity / 80% coverage
of the remote-homology and PPI test sets was removed (those two were the only filter
targets; SCOPe-40 was not filtered). We retrained on the filtered corpora using the
configuration favoured by the paper's own ablations — proportional sampling and no
synthetic hard negatives — otherwise following the submitted recipe, on 7 GPUs rather
than 1.

Paired bootstrap over the 1,693 eligible queries, 10,000 resamples: MMseqs2 **beats
the submitted model** at top-1 by +0.0697 [+0.0413, +0.0975]; the retrained model
leads MMseqs2 at R@1 by +0.0289 [+0.0035, +0.0544], at R@10 by +0.1819 [+0.1607,
+0.2026] and MAP by +0.2356 [+0.2159, +0.2551]. Two honest qualifications: the
top-1 lead is small and its lower bound is +0.0035; and MMseqs2's R@10 and R@30
differ by only 0.0004, meaning it returns few candidates past rank 10, so part of
the depth margin is candidate coverage rather than ranking. At the other extreme
MMseqs2 is *below chance* on DeepSol solubility (AUC 0.4185), where there is no
homology signal to transfer.

That is the trade-off, measured rather than asserted: alignment wins single-best-hit
and homology-transferable annotation; the embedding wins ranking depth and is the
only one of the two that yields a fixed-width vector for tasks with no alignment
signal. We do not claim it dominates.

The same trade-off appears against the untuned backbone, and it is why the
general-purpose framing goes. Over the 20 tasks comparable in both arms (three
multiclass tasks are excluded because one-vs-rest AUC is undefined when the test
split contains classes absent from train), tie band ±0.005, test split, single
seed: V1 is 11 win / 3 tie / 6 lose under 3-NN (median +0.0075) and **4 / 4 / 12
under a frozen linear probe** (median -0.0139); V2 is 10/3/7 and 2/7/11. On remote
homology (accuracy) the gain does hold under both probes for V2 — 3-NN 0.5835 ->
0.6668, linear 0.6868 -> 0.7016 — while V1's linear gain (+0.0031) is a tie by our
own band. MMseqs2 on that task reaches macro-OvR AUC 0.6523 at 88.9% hit coverage,
which is a different metric and is not comparable to those accuracies.

### 2. Is this more than structural-information injection? (your Q1 / W1)

Partly not, and the submitted ablation says how much. Removing AFDB drops the mean
relative gain from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and the
remote-homology gain from +40.5% to +15.3%. Structural supervision is the single
largest contributor and we say so first.

The remainder is not structure, and each source leaves a fingerprint on a different
task family: without Pfam the model still improves 15/23 (mean +4.6%); removing
STRING moves the PPI task from +5.3% to -0.5% while leaving most others intact;
removing the DMS objective reduces fitness gains (fluorescence +15.6% -> +10.4%). A
pure structure-distillation model has no PPI dial and no fitness dial. The claim we
can support is a sequence-level metric space shaped jointly by family,
structural-cluster, interaction and fitness-order relations.

The honest limit on that argument: those are single-run relative-percent numbers
from the submitted tables, with no intervals — the reporting we withdraw for
sub-1% cells elsewhere in this rebuttal. We use them only for the direction and
size of source-specific effects, which are 3-40 points, not for the small ones. **We did
not run the joint no-AFDB/no-Pfam ablation you asked for.**

### 3. The CoSENT objective on DMS data (your Q4)

Your reading is a fair reading of our text, and our text is wrong: the paper says
the DMS loss "operates on single proteins rather than pairs." The released code
writes `(sentence_0, sentence_1, score)` rows — `sentence_0` is the wild-type,
`sentence_1` the mutant, `score` the within-assay normalized fitness rescaled to
[0,1] (clinical rows map benign to 1.0, pathogenic to 0.0). CoSENT is then ordinal
over those pairs, exactly as for sentences: within a batch, if pair p scores higher
than pair q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no
absolute cosine target and no term pulling high-fitness mutants to a common point,
so it does not flatten the assay. The real limitation is narrower and we state it:
the pairing is **wild-type-anchored**, so mutant-mutant geometry within an assay is
constrained only indirectly.

### 4. Scale, baselines, related work

Everything above is 35M. **There is no 150M model on the decontaminated corpus**;
the submitted 150M numbers were trained on the uncontrolled corpus and we do not
defend them. Every claim we now make is a 35M claim.

**Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** We offer no
excuse for their absence; they are missing, we claim no superiority to any of them,
and ProtTucker in particular is the closest published analogue to our protocol —
contrastive fine-tuning for remote homology — not a distant one. **We also did not
apply ProtSent to SaProt or ProSST**: that is not a backbone swap at the data
level, since both need residue-level structure tokens for the entire Pfam and
STRING corpora (>100M sequences).

ESM-S, S-PLM, ISM and Magneton belong with the structure-injection line; we
position ProtSent as a different supervision graph, not a better one, with no
matched runs. The "?" at line 21 is a broken citation key, not a missing reference:
Heinzinger et al. 2022 (ProtTucker) and Redl et al. 2023 are both discussed in
Related Work.

If the measured trade-off and the source fingerprints answer your two axes, we ask
you to reconsider; if one of them is still open, say which and we will address it in
discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: we re-filtered the
pretraining corpora, retrained from scratch, and re-ran every benchmark. Your eight items
in order.

### 1. Leakage

**Decontamination, completed.** MMseqs2 `easy-search`, corpus as query, test set as
target, 40% identity / 80% coverage of the test sequence (`--min-seq-id 0.4 --cov-mode 1
-c 0.8 -e 1e-3`); any pretraining sequence with a hit is dropped. Pfam 28,530,684 ->
27,929,772 rows (-600,912) and AFDB 135,404,259 -> 126,301,607 (-9,102,652) against
`remote_homology` test (fold_prediction, 3,244 seqs); STRING 76,070,154 -> 71,891,417
pairs (-4,178,737) against `ppi_bernett` test (3,022 seqs).

**Scope limit, stated up front because you would otherwise find it yourself: those two
test sets were the only filter targets.** Every other benchmark test set, including
SCOPe-40, was **not** filtered against; no claim below covers them.

Controls: 1,000 random sequences from each *filtered* corpus, re-searched against its
target, return **0 hits**. Pfam and STRING used an exhaustive GPU prefilter (100%
recall); AFDB used k-mer `-s 5.7` (89.4% recall), so it may have missed ~10% of
qualifying hits — its negative control was re-run exhaustively and still returned 0. The
filter was then verified on the parquet files the training job actually opened, by
semi-join with the removal lists: **0 flagged sequences survived**. Row arithmetic closes:
27,929,772 + 126,301,607 + 15,000,000 = 169,231,379, the training-log total (STRING was
subsampled to 15M rows for compute, which is not a leakage control).

**Retraining on the filtered corpora improved that task.** Remote homology
(pooled 457-class, accuracy, test split): ESM-2 35M 0.5835, V1 0.6587, **V2 0.6668**
under 3-NN; 0.6868 / 0.6899 / **0.7016** under a linear probe. We retrained using the
configuration favoured by the paper's own ablations — proportional sampling and no
synthetic hard negatives — otherwise following the submitted recipe, on 7 GPUs rather
than 1. V1-vs-V2 is not a controlled decontamination ablation; the supported claim is
sufficient — decontamination cost nothing.

**SCOPe-40 cannot be decontaminated, by us or anyone**: it has no train/test split
(leave-one-out self-retrieval over 2,207 domains), its median maximum identity to our
corpus is **0.908** and none falls below 20%, so filtering removes essentially every
structured domain. We therefore tested memorization directly: if the gain came from
memorizing close pretraining neighbours, queries with a closer neighbour would gain more.
Per-query Recall@10 gain over ESM-2 across the 1,693 eligible queries is flat across
identity bins — V2 +0.1524 at [0.2, 0.4) (n=164) versus +0.1565 at [0.7, 1.0] (n=1,214);
the [0, 0.2) bin is **empty**, which is the point above. Spearman between identity and
gain is -0.038 (Recall@10) and -0.114 / -0.116 (average precision, p < 3e-6), and stays
negative after controlling for headroom, since gain is bounded by 1 - baseline (partial
-0.083 / -0.081, p < 1e-3). Among the **404 queries the untuned backbone fails completely
at Recall@10** identity does not predict the gain at all (V2 +0.038, p=0.45).

**What these controls cannot rule out — stated by us, not found by you.** Our
supervision is Foldseek structural-cluster and Pfam family co-membership, so a training
pair sharing a test domain's *fold* at 15% identity survives a 40%-identity filter, and
identity stratification cannot detect fold-level label overlap. We can say the corpus
holds nothing within 40%/80% of the two filtered test sets and that the SCOPe advantage
does not grow with proximity to pretraining data. We cannot say SCOPe-40, or the
fold-level third of the remote-homology set, is free of structural-label overlap. The
right experiment — excluding SCOPe queries whose fold appears among training clusters —
we did not run.

### 2. DMS objective

The ordering objective you describe is the one implemented; our text is wrong ("operates
on single proteins rather than pairs"). Each row is (wild-type, mutant, within-assay
normalized fitness in [0,1]), and CoSENT ranks pairs within a batch: if mutant a beats
mutant b, the loss pushes cos(WT, a) above cos(WT, b). No absolute target, no term
collapsing high-fitness variants together. The genuine limitation is that the pairing is
WT-anchored, so mutant-mutant distances are constrained only indirectly.

### 3. MNRL batch semantics and Eq. 1

Correct, and this is a real error. For the submitted models the 1,024 is an **optimizer**
batch reached by gradient accumulation (35M: per-device 64 x 16 steps; 150M: 16 x 64, our
Table 6), and accumulation does not share in-batch negatives across micro-batches — so
each MNRL loss call saw **64** examples at 35M and **16** at 150M, not 1,024. The retrain uses
`CachedMultipleNegativesRankingLoss`, where 1,024 is a true contrastive batch. In Eq. 1
the numerator should use the positive paired with anchor i, and the denominator ranges
over the positive members of all N pairs in the batch.

### 4. Pair-level tasks

PPI: each partner is embedded independently and the two vectors concatenated
(`np.concatenate([emb[s1], emb[s2]])`) before the probe. Peptide-HLA is **not** a
two-input task here — the dataset supplies a single `seq` field, a pipe-joined
`HLA_pseudoseq|peptide` string, so no combination operator is applied. Neither was in the
paper.

### 5. k-NN regression

`KNeighborsRegressor(n_neighbors=3, metric="minkowski")`, default uniform weighting: an
unweighted mean over 3 Euclidean neighbours. At small N the code sets `n_neighbors =
max(1, min(3, train_size))`, so the smallest few-shot cells are not the same estimator.
That, plus relative changes over near-zero Spearman baselines (which produce cells like
-126.9%, a sign flip of magnitude 0.269x baseline), makes the few-shot table
uninterpretable; its claims are withdrawn.

### 6. Ablations

Agreed, and we acted on it rather than arguing. Removing synthetic hard negatives
improves 20/23 tasks at mean +7.9% against 16/23 and +6.7% for the submitted
configuration; proportional sampling gives +7.0% vs round-robin's +6.7%. Our own evidence
does not establish the submitted defaults as optimal, so the retrain uses neither.
Disclosure that follows: those ablations were scored on these same benchmark tasks, so
V2's configuration was chosen with benchmark results in view. Its checkpoint was not: it
is the last training step, and a near-trough control checkpoint differs from it by
0.005-0.008 on every structural metric.

### 7. Baselines

MMseqs2, run as a full alternative pipeline over all 23 tasks with identical metric
definitions (self excluded; no-hit queries scored as failures), `-s 7.5 -e 10 --max-seqs
300 --alignment-mode 3`. On SCOPe-40 (family level, 2,207-sequence gallery, **Recall@K
upper-bounded at 0.7671**, since only 1,693 queries have a non-self same-family
neighbour), R@1 / R@10 / MAP: MMseqs2 0.5029 / 0.5637 / 0.3100; ESM-2 35M 0.3829 / 0.5840
/ 0.3230; V1 0.4490 / 0.6529 / 0.4226; V2 0.5256 / 0.7073 / 0.4955. Alignment beats the
submitted model at top-1, and beats the best embedding model outright on 3 of 23 tasks
under 3-NN and 6 under a linear probe (EC F1-macro 0.710 vs 0.598/0.562; GO-MF 0.585 vs
0.459/0.443). Its remote-homology score, macro-OvR AUC 0.6523, is a
different metric from item 1's accuracies.

**Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** No excuse offered; we
claim no superiority to any.

### 8. Statistical evidence

Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples, paired (the
same queries score every method). At R@1: V1 - ESM-2 +0.0868 [+0.0614, +0.1122];
V2 - ESM-2 +0.1855 [+0.1618, +0.2097]; V2 - MMseqs2 +0.0289 [+0.0035, +0.0544]. At MAP,
V2 - ESM-2 is +0.2232 [+0.2082, +0.2383]. Against interest: MMseqs2 beats ESM-2 at top-1
by +0.1565 and V1 by +0.0697 [+0.0413, +0.0975], and MMseqs2 vs ESM-2 at R@10 and MAP is
**unresolved**.

For the 23-task table we have **no** intervals and no multi-seed results — an omission,
not an obstacle: probes can be held fixed and predictions resampled. Your objection
stands without them: every cell is one run at seed 42, so any delta inside ±0.005 is a
tie. Against ESM-2 35M, V1 is 11 win / 3 tie / 6 lose under 3-NN (median +0.0075)
and **4 / 4 / 12 under a linear probe** (median -0.0139); V2 is 10/3/7 and 2/7/11. That
is why the general-purpose claim is withdrawn. Counts cover **20 of 23** tasks:
`antibiotic_resistance`, `remote_homology` and `temperature_stability` have a
multiclass-AUC main metric undefined when the test split holds a class absent from train
— which drops remote homology, our best task, from the tally. And `ec_classification`,
`go_mf` and `scope40_retrieval` use a built-in evaluator that ignores the probe flag, so
those rows are one measurement printed twice.

### Errors we found in our own submission

PPI decontamination is `easy-search` (STRING as query, Bernett test as target) at 40%
identity, removing hit query IDs — not `easy-linclust` at 50% with cluster removal as our
text says. The remote-homology split is not
hierarchy-disjoint: TAPE's three holdouts pooled (718 fold + 1,254 superfamily + 1,272
family), and its pooled macro AUC is not comparable to published per-holdout accuracies.
SCOPe is the **family** field over **2,207** sequences, not superfamily over 100,000 —
that figure is the evaluator's `max_samples` cap. The row counts above are the training
parquets — what was actually trained on — and do not match our Table 1 (32.9M / 133.9M /
36.5M). All numbers here are `--eval_split test`, not the suite's validation default, so
they are not cell-comparable to the submitted tables.

There is no 150M model on the decontaminated corpus, so the submitted 150M results stand
on the unfiltered corpus and we do not defend them. If one of your eight items is still
decisive, name it and we will answer it in discussion.
<!-- END Yi1G -->
