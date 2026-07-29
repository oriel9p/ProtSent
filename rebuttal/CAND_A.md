# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — CANDIDATE A

Post each response under its own review. No links, no attachments. Every number
appears in the response text with its metric, split and model.

Naming used throughout: **V1** = the submitted/published 35M model.
**V2** = the 35M model retrained during the rebuttal on the decontaminated corpora.

---

## Response to Reviewer HNXd

<!-- BEGIN HNXd -->
**What the paper claims after this rebuttal:** contrastive training of a 35M
ESM-2 backbone — evaluated frozen, under 3-NN and linear probes — reorganizes
the embedding so that family, fold and interaction relations become locally
recoverable, measured as structural retrieval and remote homology, and it does
**not** improve general property prediction under a trained readout. The
general-purpose superiority claim is withdrawn; your question 2 killed it.

Both probes are now complete on all 23 tasks, `--eval_split test`, for stock
ESM-2 35M, the submitted model (V1) and a model retrained during the rebuttal on
decontaminated corpora (V2). Against ESM-2 35M:

| probe (20 comparable tasks, test split, vs ESM-2 35M) | V1 | V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

(i) The tie band is **±0.005** in absolute units on each task's main metric — a
convention, not a derived threshold, applied identically to Spearman, accuracy
and AUC. With 7 ties of 20 it materially decides V2's linear record, and no
setting of it turns that record into a win. (ii) Counts cover **20 of 23** tasks:
`antibiotic_resistance`, `remote_homology` and `temperature_stability` are
excluded from *both* columns because their main metric is multiclass AUC and the
test split contains classes absent from train, so one-vs-rest AUC is undefined
for the embedding arms. That drops remote homology, our best task, out of the
tally; its accuracy is below. (iii) `ec_classification`, `go_mf` and
`scope40_retrieval` **are** inside the 20 and use a built-in evaluator that
ignores the probe flag, so each contributes an identical value to both rows — the
two records are not independent, and SCOPe retrieval is **one** measurement, not
two. (iv) The linear probe is scikit-learn defaults (`StandardScaler` +
`LogisticRegression(solver="liblinear")` or `Ridge(alpha=1.0)`) on frozen
mean-pooled embeddings, untuned, single run at seed 42; an undertuned probe could
move this record either way.

### 1. Direct retrieval evaluation (your question 1)

**We did not compute clustering-geometry statistics** — no silhouette, NMI or
ARI. We ran the retrieval half of your request; the geometry half is missing.

SCOPe-40, **family** level, 2,207-sequence gallery, self excluded, no-hit
queries scored as failures. **Recall@K is upper-bounded at 0.7671**: only 1,693
of 2,207 queries have any non-self same-family protein in the gallery.

| method (all 2,207 queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

V2 removed every corpus sequence within 40% identity / 80% coverage of the
remote-homology and PPI test sets (the only two filter targets; SCOPe-40 and the
other benchmarks were **not** filtered), adopted the configuration the paper's
own ablations favour (proportional sampling, no synthetic hard negatives), and
fixed a batch bug: the submitted 1,024 was an optimizer batch via gradient
accumulation, so each loss call saw **64** examples, where V2 uses a true
1,024-example contrastive batch. **V2 - V1 is therefore not a decontamination
ablation**; no unfiltered-corpus retrain at the V2 recipe exists.

The advantage is ranking depth, not top-1: a tuned MMseqs2 **beats the submitted
model at top-1** (0.5029 vs 0.4490), and HMMER beats it by more. Against us:
MMseqs2's R@10 and R@30 differ by 0.0004, so part of the depth gap is candidate
coverage, not ranking — though HMMER does not have that problem and V2 still
leads it at depth (section 3).

### 2. Linear probes and label scarcity (your question 2)

Per task the probes disagree in a specific way: on AAV fitness (Spearman) ESM-2
0.4667 vs V1 0.5553 under 3-NN, but ESM-2 0.5639 vs V1 0.4362 under the linear
probe. Contrastive training makes relations *local*; it does not add information
a trained head could not already extract from mean-pooled ESM-2 features.

Remote homology (pooled 457-class, test split) is where it does add something.
Accuracy: 3-NN 0.5835 (ESM-2) / 0.6587 (V1) / 0.6668 (V2); linear 0.6868 /
0.6899 / 0.7016. Macro-F1, stated because it is where V1 loses: 3-NN 0.3173 /
0.3687 / 0.4108; linear **0.4414 / 0.4281 / 0.4527** — V1 is *below* the untuned
backbone. V1's +0.0031 linear accuracy is inside our ±0.005 band and is a tie;
only V2 improves on both metrics under both probes. The paper's "+40.5%" is a
relative macro-F1 change (.223 → .313) under the suite's default split, not these
test-split numbers; we do not mix them.

**On the level difference you flagged, we tested your hypothesis and it fails.**
BIOMAP `stability_prediction` labels are continuous floats and the task is
regression scored by Spearman, so our "58.8%" is a correlation ×100, not an
accuracy commensurate with 69.08% linear / 77.69% LoRA. Nor is the probe the
cause: on that task 3-NN scores *higher* than our linear probe for every arm
(ESM-2 35M Spearman 0.6435 3-NN vs 0.4395 linear). Printing our numbers beside
published fine-tuned ones was wrong; we withdraw the comparison.

**We did not run a fine-tuning sweep and we have no few-shot linear-probe
baseline.** The label-scarcity claim has no supporting control; we withdraw it.

### 3. 95% confidence intervals (your question 3)

Every retrieval metric is a mean over per-query values, so resampling queries
gives the sampling distribution with no refitting. 10,000 resamples, **paired**
(the same queries score every method), over the **1,693 eligible** queries — a
different denominator from the table above. HMMER (phmmer) is included as the
stronger alignment baseline; 691 of 2,207 queries return no phmmer hit and are
scored as failures. On that denominator HMMER is the best top-1 method (R@1
0.6970, versus V2 0.6846, MMseqs2 0.6556, V1 0.5859, ESM-2 35M 0.4991); the
alignment rows come from the head-to-head run and do not rescale exactly from
the all-query table.

| paired difference (1,693 eligible queries) | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - V1 | +0.0986 [+0.0762, +0.1211] | +0.0709 [+0.0555, +0.0862] | +0.0943 [+0.0814, +0.1074] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |
| V2 - HMMER | **-0.0124 [-0.0372, +0.0124]** | +0.1412 [+0.1205, +0.1618] | +0.1708 [+0.1511, +0.1905] |

**We do not claim to beat alignment at top-1.** V2 ties HMMER there and beats
only the weaker tool, MMseqs2; V1 loses to HMMER outright, -0.1110 [-0.1388,
-0.0827]. The embedding advantage is ranking depth, consistent across both tools.
Also against us: MMseqs2 beats ESM-2 35M at top-1 by +0.1565 [+0.1276, +0.1855]
and V1 by +0.0697 [+0.0413, +0.0975]. On local organisation, V2's per-query R@10
gain over ESM-2 is flat across max-identity-to-corpus bins (+0.1524 at [0.2,0.4),
n=164; +0.1810 at [0.4,0.7), n=315; +0.1565 at [0.7,1.0], n=1,214), so the
reorganisation is not concentrated near close pretraining neighbours.

**We did not compute intervals for the 23-task table.** Your objection stands
there: each cell is one run, and any delta inside ±0.005 is reported as a tie.

### 4-5. Seed variability and Table 5 (your questions 4 and 5)

**Five seeds (0-4) × 8 representative tasks × 3 arms, 3-NN, test split.** Median
SD across all 24 rows is **0.0000**. Remote-homology accuracy: 0.5835 ± 0.0000
(ESM-2), 0.6589 ± 0.0002 (V1; the seed-42 run above is 0.6587), 0.6668 ± 0.0000
(V2). The reason is mechanical, not impressive: with fixed embeddings and a fixed
test split a 3-NN probe is deterministic, so the seed only moves subsampling and
CV-fallback splits. `thermostability` is the one task that subsamples and the only
one with visible spread — Spearman 0.4427 ± 0.0126 (ESM-2), 0.4696 ± 0.0172 (V1),
0.4568 ± 0.0156 (V2). **These are probe seeds: one training run per model exists,
so training-seed variance is unmeasured**, and a different nuisance factor is
larger — a near-trough checkpoint (step 4,208, where the 3-cycle cosine bottoms)
differs from the final V2 checkpoint by 0.005-0.008 on every structural metric,
at or above our own tie band. We therefore do not present the +0.0079 V1→V2
remote-homology gap as a resolved improvement, only as evidence that
decontamination did not cost performance.

Table 5 is withdrawn on the estimator, not for lack of seeds. A relative change
of -126.9% on a Spearman correlation is a sign flip of magnitude 0.269× baseline,
not a 127-point drop, and +244.5% on a near-zero baseline is unbounded. At small
N the probe also silently reduces k, since the code sets `n_neighbors = max(1,
min(3, train_size))` — those cells are not even the same estimator, so we do not
re-present them in absolute units.

### Errors we found in our own submission

All new evidence above is 35M. **There is no 150M model on the decontaminated
corpus**, so we do not defend the submitted 150M numbers, including the
abstract's +105% and +19.9%. SCOPe is evaluated on the **family** field over
**2,207** sequences, not superfamily over 100,000 — that figure is the
evaluator's `max_samples` cap echoed into the results table. Our remote-homology
test split is TAPE's three holdouts pooled (718 + 1,254 + 1,272 = 3,244), not
hierarchy-disjoint as we wrote, so its pooled macro AUC is not comparable to
published per-holdout accuracies.

If the withdrawals and the measurements above change your assessment, we ask you
to reconsider; if one item remains decisive, name it and we will answer it in
discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- BEGIN jVGf -->
**The paper's claim after this rebuttal:** multi-relational contrastive training
of a 35M ESM-2 backbone, evaluated frozen, buys retrieval geometry — structural
and homology search — and buys nothing on tasks where a trained head already
extracts the label. The general-purpose framing is withdrawn.

### 1. Where ProtSent sits on the generality-accuracy trade-off (your Q3 / W2)

We ran MMseqs2 over all 23 tasks under identical metric definitions —
family-level Recall@K with self excluded, per-class max bitscore for
classification so AUC stays comparable, 1-NN by bitscore for regression. No-hit
queries are scored as failures, not dropped. Flags: `-s 7.5 -e 10 --max-seqs 300
--alignment-mode 3`; at the default `-s 5.7` the same baseline gives SCOPe R@1
0.3847, so any MMseqs2 number needs its sensitivity stated.

Alignment beats the best embedding model outright on **3 of 23 tasks under a
3-NN probe** (EC classification, GO molecular function, beta-lactamase) and **6
under a linear probe** (those plus enzyme catalytic efficiency, optimal pH,
stability). EC classification F1-macro 0.710 (MMseqs2) vs 0.598 (ESM-2 35M) and
0.562 (ProtSent-V1); GO-MF 0.585 vs 0.459 / 0.443. On beta-lactamase (Spearman,
3-NN) MMseqs2 reaches 0.8026 and beats every embedding arm, including our
retrained model. Where annotation transfers by homology, alignment is better.

The other end of the curve is SCOPe-40 (family level, 2,207-sequence gallery,
self excluded, no-hit = failure; **Recall@K upper-bounded at 0.7671**, since only
1,693 queries have a non-self same-family neighbour):

| method (all 2,207 queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained, decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

V2 removed every corpus sequence within 40% identity / 80% coverage of the
remote-homology and PPI test sets (the only two filter targets; SCOPe-40 was not
filtered), adopted the configuration the paper's own ablations favour
(proportional sampling, no synthetic hard negatives), and fixed a batch bug: the
submitted 1,024 was an optimizer batch via gradient accumulation, so each loss
call saw 64 examples, where V2 uses a true 1,024-example contrastive batch. **V2
- V1 is therefore not a decontamination ablation.**

**We also ran HMMER (phmmer), the stronger alignment baseline, and report it
against ourselves.** Eligible-query R@1 / R@10 / MAP (1,693 queries; 691 of 2,207
return no phmmer hit, scored as failures): HMMER 0.6970 / 0.7809 / 0.4747;
MMseqs2 0.6556 / 0.7401 / 0.4098; ProtSent-V2 0.6846 / 0.9220 / 0.6454. Paired
bootstrap, 10,000 resamples, V2 - HMMER: R@1 **-0.0124 [-0.0372, +0.0124],
unresolved**; R@10 +0.1412 [+0.1205, +0.1618]; MAP +0.1708 [+0.1511, +0.1905].
V1 - HMMER at R@1 is -0.1110 [-0.1388, -0.0827], an outright loss. Against
MMseqs2, V2 leads at R@1 by +0.0289 [+0.0035, +0.0544] and V1 *loses* by -0.0697
[-0.0975, -0.0413]. **We do not claim to beat alignment at single-best-hit**: V2
ties the stronger tool there and beats only the weaker one. Two qualifications:
MMseqs2's R@10 and R@30 differ by 0.0004, so part of its depth gap is candidate
coverage rather than ranking — HMMER does not have that problem (0.7809 → 0.7980
at R@30), so the depth result survives the caveat. At the other extreme MMseqs2
is *below chance* on DeepSol solubility (AUC 0.4185).

The trade-off: alignment wins single-best-hit and homology-transferable
annotation; the embedding wins ranking depth. For tasks with no alignment signal
an embedding is required — but there the stronger embedding is often stock ESM-2
under a linear probe, which is why the general-purpose claim goes. Over the 20
tasks comparable in both arms (three multiclass tasks excluded: one-vs-rest AUC
is undefined when the test split contains classes absent from train), tie band
±0.005, test split, single seed: V1 is 11 win / 3 tie / 6 lose under 3-NN
(median +0.0075) and **4 / 4 / 12 under a frozen linear probe** (median -0.0139);
V2 is 10/3/7 and 2/7/11. On remote homology (accuracy) the gain holds under both
probes for V2 — 3-NN 0.5835 → 0.6668, linear 0.6868 → 0.7016 — while V1's linear
gain (+0.0031) is a tie by our own band.

### 2. Is this more than structural-information injection? (your Q1 / W1)

Partly not, and the submitted ablation says how much. Removing AFDB drops the
mean relative gain from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and
the remote-homology gain from +40.5% to +15.3%. Structural supervision is the
single largest contributor.

The remainder is not structure, and each source leaves a fingerprint on a
different task family: without Pfam the model still improves 15/23 (mean +4.6%);
removing STRING moves the PPI task from +5.3% to -0.5% while leaving most others
intact; removing the DMS objective reduces fitness gains (fluorescence +15.6% →
+10.4%). A pure structure-distillation model has no PPI dial and no fitness dial.
This is also our empirical answer to whether heterogeneous relations can share one
space: they do not interfere — each source moves its own task family and leaves
the rest — but we have no theory for when that holds.

The limit: those are single-run relative-percent numbers on the submitted
(pre-decontamination) V1 model under the suite's default split — the convention
we withdraw for sub-1% cells elsewhere in this rebuttal. We use them only for the
direction and size of source-specific effects, which are 2-25 points. **We did
not run the joint no-AFDB/no-Pfam ablation you asked for.** One
decontaminated-model trace of the non-structural half: GB1 variant effect
(Spearman, 3-NN, test split, mean over 5 seeds) is 0.6582 ± 0.0000 (ESM-2 35M),
0.7108 ± 0.0000 (V1), 0.7806 ± 0.0000 (V2) — a fitness task no structure
distillation supervises.

### 3. The CoSENT objective on DMS data (your Q4)

Our text is wrong: the paper says the DMS loss "operates on single proteins
rather than pairs." The released code writes `(sentence_0, sentence_1, score)`
rows — `sentence_0` is the wild-type, `sentence_1` the mutant, `score` the
within-assay normalized fitness rescaled to [0,1] (clinical rows map benign to
1.0, pathogenic to 0.0). CoSENT is then ordinal over those pairs, exactly as for
sentences: within a batch, if pair p scores higher than pair q, the loss pushes
cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no absolute cosine target and
no term pulling high-fitness mutants to a common point, so it does not flatten
the assay. The real limitation is narrower and we state it: the pairing is
**wild-type-anchored**, so mutant-mutant geometry within an assay is constrained
only indirectly.

### 4. Scale, baselines, related work

Everything above is 35M. **There is no 150M model on the decontaminated corpus**;
the submitted 150M numbers were trained on the uncontrolled corpus and we do not
defend them.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.** They are missing, we
claim no superiority to any, and ProtTucker in particular is the closest
published analogue to our protocol — contrastive fine-tuning for remote homology
— not a distant one. ProTrek we will cite in Related Work as the trimodal
retrieval point on exactly the curve you asked about: sequence, structure and
text encoders trained for search, against a 35M sequence-only encoder with no
structure or text input at inference. We expect ProTrek to win retrieval accuracy
and we claim nothing against it. **We also did not apply ProtSent to SaProt or
ProSST**: the blocker is data, not code — both consume residue-level structure
tokens, and while our AFDB subset is Foldseek-clustered, we have no predicted
structures for the Pfam and STRING sequences, which are the majority of the
corpus.

On ESM-S, S-PLM, ISM and Magneton: those distil or align a *structure* signal
into a sequence encoder — one relation type, from a structure model or predicted
structure, usually at residue level. ProtSent supervises a heterogeneous relation
graph at the sequence level (Pfam family co-membership, Foldseek cluster
co-membership, STRING interaction, DMS fitness order) with no structure encoder
anywhere in the pipeline, which is why it has PPI and fitness dials that a
structure-distillation model does not. That is a difference in supervision
source, not a claim of superiority: we have no matched runs against any of the
four, and the ablation above shows the structural half is our largest single
contributor. The "?" at line 21 is a broken citation key, not a missing
reference: Heinzinger et al. 2022 (ProtTucker) and Redl et al. 2023 are both
discussed in Related Work.

If the measured trade-off and the source fingerprints answer your two axes, we
ask you to reconsider; if one is still open, say which and we will address it in
discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: we re-filtered the
corpora, retrained from scratch, and re-ran every benchmark.

### 1. Leakage

**Decontamination, completed.** MMseqs2 `easy-search`, corpus as query, test set as
target, 40% identity / 80% coverage of the test sequence (`--min-seq-id 0.4 --cov-mode 1
-c 0.8`); any pretraining sequence with a hit is dropped. Pfam -600,912 rows and AFDB
-9,102,652 against `remote_homology` test (3,244 seqs); STRING -4,178,737 pairs against
`ppi_bernett` test (3,022 seqs). **Those two test sets were the only filter targets** —
every other benchmark test set, SCOPe-40 included, was **not** filtered against. A
negative control (1,000 random sequences from each *filtered* corpus, re-searched, AFDB's
run exhaustively) returns **0 hits**, which bounds the residue loosely, not tightly. The
filter was then verified on the parquet files training actually opened, by semi-join with
the removal lists: **0 flagged sequences survived**.

**Decontamination did not cost accuracy on the task it targeted.** Remote homology
(pooled 457-class, test split): accuracy 0.5835 (ESM-2 35M) / 0.6587 (V1) / **0.6668**
(V2) under 3-NN and 0.6868 / 0.6899 / **0.7016** under a linear probe; linear macro-F1
0.4414 / **0.4281** / 0.4527, where V1 is *below* the untuned backbone. Only V2 improves
on both metrics under both probes. **V2 is not a decontamination ablation**: it changes
four things at once — filtered corpus, proportional sampling, no synthetic hard negatives,
and (item 3) a true 1,024-example contrastive batch where V1's loss call saw 64. No
retrain of the V1 recipe on filtered data exists, so we attribute nothing to
decontamination; the supported claim is only the sufficient one, that a decontaminated
corpus costs no performance. We report no post-decontamination `ppi_bernett` result — a
pair-input task outside the single-sequence sweep — so even that covers the
remote-homology target only.

**SCOPe-40 cannot be decontaminated, by us or anyone**: it has no train/test split
(leave-one-out self-retrieval over 2,207 domains), its median maximum identity to our
corpus is **0.908** and none falls below 20%, so filtering removes essentially every
structured domain. We tested memorization directly instead: if the gain came from
memorizing close neighbours, closer queries would gain more. Per-query
Recall@10 gain over ESM-2 across the 1,693 eligible queries is flat across identity bins
(V2 +0.1524 at [0.2, 0.4), n=164; +0.1810 at [0.4, 0.7), n=315; +0.1565 at [0.7, 1.0],
n=1,214; the [0, 0.2) bin is **empty**). Spearman between identity and gain is -0.038
(Recall@10) and -0.116 (average precision, p < 3e-6), still negative after a headroom
control (partial -0.081). And among the **404 queries where the untuned backbone fails
completely** — full headroom — identity does not predict the gain (+0.038, p=0.45).

**What these controls cannot rule out.** Our supervision is Foldseek structural-cluster
and Pfam family co-membership, so a training pair sharing a test domain's *fold* at 15%
identity survives a 40%-identity filter, and identity stratification cannot detect
fold-level label overlap. We cannot say SCOPe-40, or the fold-level third of the
remote-homology set, is free of it, and the right experiment — excluding SCOPe queries
whose fold appears among training clusters — we did not run. Our strongest surviving
result therefore sits on the one benchmark we could not filter.

### 2. DMS objective

The ordering objective you describe is what is implemented; our text is wrong ("operates
on single proteins rather than pairs"). Rows are (wild-type, mutant, within-assay
normalized fitness in [0,1]) and CoSENT ranks pairs within a batch: if mutant a beats
mutant b, the loss pushes cos(WT, a) above cos(WT, b). No absolute target, no term
collapsing high-fitness variants together. The real limitation is that the pairing is
WT-anchored, so mutant-mutant distances are constrained only indirectly.

### 3. MNRL batch semantics and Eq. 1

Correct — a real error. For the submitted models the 1,024 is an **optimizer** batch from
gradient accumulation (64 x 16 steps at 35M, 16 x 64 at 150M — our Table 6), and
accumulation does not share in-batch negatives across micro-batches, so each MNRL loss
call saw **64** examples at 35M and **16** at 150M — so the submitted models are
misdescribed and under-negatived. The retrain uses `CachedMultipleNegativesRankingLoss`,
where 1,024 is a true contrastive batch per device. In Eq. 1 the numerator should use the
positive paired with anchor i; the denominator ranges over the positive members of all N
pairs.

### 4. Pair-level tasks

PPI: each partner is embedded independently and the two vectors concatenated
(`np.concatenate([emb[s1], emb[s2]])`) before the probe. Peptide-HLA is **not** a
two-input task here: the dataset supplies a single `seq` field, a pipe-joined
`HLA_pseudoseq|peptide` string, so no operator applies. Neither was in the paper.

### 5. k-NN regression

`KNeighborsRegressor(n_neighbors=3, metric="minkowski")`, default **uniform** weighting.
At small N the code sets `n_neighbors = max(1, min(3, train_size))`, so the smallest
few-shot cells are not even the same estimator; with relative change over near-zero
baselines on top (-126.9% is a sign flip of magnitude 0.269x), that table is
uninterpretable and its claims are withdrawn.

### 6. Ablations

Removing synthetic hard negatives improves 20/23 tasks at mean +7.9% against 16/23 and
+6.7% for the submitted configuration; proportional sampling gives +7.0% vs round-robin's
+6.7%. Our evidence does not establish the submitted defaults as optimal, so the retrain
uses neither. Those ablations were scored on these same benchmarks, so V2's configuration
was chosen with benchmark results in view — a selection channel the corpus filter does not
touch. Its checkpoint was not: it is the last training step, and a near-trough control
checkpoint (step 4,208) differs by 0.005-0.008 on every structural metric, at or above
item 8's tie band, so we call no sub-0.01 structural delta resolved. On your limitations
point about one embedding space for heterogeneous relations, these same ablations are our
only empirical answer: each source moves its own task family and leaves the others largely
intact (removing STRING takes PPI +5.3% -> -0.5%, the DMS objective takes fluorescence
+15.6% -> +10.4%, AFDB takes remote homology +40.5% -> +15.3%). Single-run
relative-percent numbers from V1, and we have no account of when the property holds.

### 7. Baselines

**HMMER (phmmer, `-E 10`, top 300 hits per query, no-hit = failure) was run**, on the same
gallery and through the same shared scorer as MMseqs2, and it is the stronger alignment
baseline. On the 1,693 eligible SCOPe-40 queries (family level; **Recall@K upper-bounded
at 0.7671**, since only 1,693 of 2,207 have a non-self same-family neighbour), R@1 / R@10
/ MAP: HMMER 0.6970 / 0.7809 / 0.4747; MMseqs2 0.6556 / 0.7401 / 0.4098; ESM-2 35M 0.4991
/ 0.7614 / 0.4222; V1 0.5859 / 0.8512 / 0.5511; V2 0.6846 / 0.9220 / 0.6454. **Alignment
remains the better top-1 method** — item 8. MMseqs2 was also run over all 23 tasks under
the same metric definitions and beats the best embedding model outright on 3 tasks under
3-NN, 6 under a linear probe (EC F1-macro 0.710 vs 0.598 for ESM-2 35M, 0.562 for V1); on
remote homology it gets accuracy 0.4365, macro-F1 0.2064, at 0.889 hit coverage.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek** — we claim no superiority to
any, and ProtTucker is the closest published analogue to our protocol. Redl et al. 2023,
"Optimizing Protein Language Models with Sentence Transformers", is cited in Related Work
but **not run**: different supervision set, no matched run, no comparative claim.

### 8. Statistical evidence

Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples; the same queries
score every method. At R@1: V2 - ESM-2 +0.1855 [+0.1618, +0.2097]; V2 - MMseqs2 +0.0289
[+0.0035, +0.0544]; **V2 - HMMER -0.0124 [-0.0372, +0.0124], unresolved**; V1 - HMMER
-0.1110 [-0.1388, -0.0827], an outright loss. So V2 *ties* the best alignment baseline at
top-1 and beats both at depth: V2 - HMMER R@10 +0.1412 [+0.1205, +0.1618], MAP +0.1708
[+0.1511, +0.1905]. Against interest, MMseqs2 also beats ESM-2 at top-1 (+0.1565 [+0.1276,
+0.1855]) and V1 (+0.0697).

**Multi-seed:** 5 seeds (0-4) x 8 tasks x 3 arms, 3-NN, test split. Median SD across the
24 rows is **0.0000** — with fixed embeddings and a fixed test split a 3-NN probe is
deterministic, so the seed moves only subsampling and CV-fallback splits;
`thermostability` is the one task that subsamples and the only one with spread (SD
0.013-0.017). The probe therefore contributes almost no variance, and the residual
uncertainty in Table 2 is benchmark composition, which the bootstrap above measures.
**These are probe seeds; one training run per model exists, so training-seed variance is
unmeasured**, and the 23-task table still has **no** intervals: one run at seed 42 per
cell, so any delta inside ±0.005 is a tie. That band is a convention in absolute units,
smaller than item 6's checkpoint spread — both facts cut against us. Against ESM-2 35M,
over the 20 of 23 tasks whose main metric is defined for all arms (the excluded three are
multiclass-AUC tasks undefined when the test split holds a class absent from train, which
drops remote homology from the tally), V1 is 11 win / 3 tie / 6 lose under 3-NN and **4 /
4 / 12 under a linear probe**, medians +0.0075 and -0.0139; V2 is 10/3/7 and 2/7/11.
Hence the withdrawal of the general-purpose claim.

### Errors we found in our own submission

PPI decontamination is `easy-search` (STRING as query, Bernett test as target) at
40% identity, removing hit query IDs — not `easy-linclust` at 50% with cluster removal as
our text says. The remote-homology split is not hierarchy-disjoint: TAPE's three holdouts
pooled (718 + 1,254 + 1,272), so its pooled macro AUC is not comparable to published
per-holdout accuracies. All numbers here are `--eval_split test`, not the suite's
validation default, so they are not cell-comparable to the submitted tables. There is no
150M model on the decontaminated corpus; the submitted 150M results stand on the
unfiltered corpus and we do not defend them.

If your leakage item is now bounded to the residual we state — fold-level overlap on
SCOPe-40, untested — we ask you to reconsider. If another item is decisive, name it and we
will answer it in discussion.
<!-- END Yi1G -->
