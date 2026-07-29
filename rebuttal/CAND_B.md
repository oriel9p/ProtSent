# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — CANDIDATE B

Post each response under its own review. No links, no attachments. Every number
appears in the response text with its metric, split and model. The paste unit for
each reviewer is everything between its BEGIN and END markers; the character count
of that unit is stated in the comment directly above it.

Naming: **V1** = the submitted/published 35M model. **V2** = the 35M model
retrained during the rebuttal on the decontaminated corpora.

---

## Response to Reviewer HNXd

<!-- paste-unit character count: 9916 (everything between BEGIN and END) -->
<!-- BEGIN HNXd -->
**We ran the linear probe you asked for, it reversed our headline, and the
general-purpose superiority claim is withdrawn.** On frozen mean-pooled
embeddings under a trained linear readout, both ProtSent models lose to stock
ESM-2 35M across the task suite. What survives is the narrower claim you asked us
to measure directly: contrastive fine-tuning of a 35M ESM-2 backbone (evaluated
frozen under both probes) reorganises the geometry so family and fold relations
become locally recoverable — retrieval and remote homology — while adding nothing
a trained head could not already extract. Below: your five questions in order —
four carry new measurements, two carry withdrawals.

### 1. Linear probe, label scarcity, and the gap to the literature (Q2)

23 tasks, `--eval_split test`, stock ESM-2 35M vs both ProtSent models:

| probe (20 comparable tasks, vs ESM-2 35M) | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

Every reading rule cuts against us. The ±0.005 tie band is a convention in
absolute units, not a derived threshold, applied alike to accuracy, Spearman, F1
and AUC; it creates 7 of V2's 20 linear cells, and with a negative median,
splitting those ties would not favour us. Counts cover **20 of 23** tasks:
`antibiotic_resistance`, `remote_homology` and `temperature_stability` leave both
columns because their main metric is multiclass AUC and the test split holds
classes absent from train, so one-vs-rest AUC is undefined for the embedding arms
— which drops remote homology, our best task, from the tally.
`ec_classification`, `go_mf` and `scope40_retrieval` *are* inside the 20 but use a
built-in evaluator that ignores the probe flag, so each contributes an identical
value to both rows; only 17 of 20 actually differ by probe, and SCOPe retrieval is
**one** measurement, not two. The linear probe is scikit-learn defaults
(`StandardScaler` + `LogisticRegression(solver="liblinear")` or `Ridge(alpha=1.0)`),
untuned — not a clean measurement in either direction.

Remote homology (pooled 457-class, test split) is where contrastive training does
add something, and we give both metrics because one of them embarrasses V1.
Accuracy under 3-NN: 0.5835 (ESM-2) / 0.6589 (V1) / **0.6668** (V2); under the
linear probe 0.6868 / 0.6899 / **0.7016** — V1's +0.0031 is inside our own band
and is a tie. Macro-F1 under 3-NN: 0.3173 / 0.3687 / 0.4108; under the linear
probe 0.4414 / **0.4281** / 0.4527 — V1 is *below* the untuned backbone. Only V2
improves on both metrics under both probes. (The paper's "+40.5%" for this task is
a relative macro-F1 change of .223 → .313 on the suite's default split, a
different metric and a different split from the numbers here; we do not mix them.)

**On the level gap to the literature you flagged: we tested your hypothesis and
it is not the probe.** On Stability (BIOMAP) the 3-NN probe scores *higher* than
our linear probe for every arm — ESM-2 35M Spearman 0.6435 under 3-NN vs 0.4395
under the linear probe. So k-NN is not what puts us below 69.08% linear / 77.69%
LoRA. That task is also regression scored by Spearman in our suite, so our "58.8%"
is a correlation, not an accuracy, and printing it beside published fine-tuned
accuracies was wrong. Separately, on this task ProtSent *loses* to ESM-2 under
3-NN (0.6435 vs V1 0.5638, V2 0.5961).

**We did not run a fine-tuning sweep and we have no few-shot linear baseline.**
The label-scarcity claim therefore has no control and is withdrawn, not defended.

### 2. Retrieval and how the space reorganises (Q1)

**We did not compute clustering-geometry statistics** — no silhouette, NMI or
ARI. We ran retrieval plus a per-query organisation analysis.

SCOPe-40, **family** level, 2,207-sequence gallery, self excluded, no-hit queries
scored as failures. Only 1,693 queries have any non-self same-family protein in
the gallery, so **Recall@K is capped at 0.7671** and all numbers below use those
1,693 eligible queries (over all 2,207 every method scales by the same 0.7671).

| method (1,693 eligible queries) | R@1 | R@10 | MAP |
|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10 --max-seqs 300`) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer, `-E 10`, top 300) | **0.6970** | 0.7809 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.4222 |
| ProtSent-V1 35M (submitted) | 0.5859 | 0.8512 | 0.5511 |
| ProtSent-V2 35M (retrained) | 0.6846 | **0.9220** | **0.6454** |

**Alignment remains the better top-1 method.** V1 loses top-1 to both tools; V2
ties the stronger one (HMMER) and passes only the weaker (see intervals below).
The embedding advantage is ranking *depth*, and it holds against both tools. V2 is
a retrain on corpora filtered at 40% identity / 80% coverage against the
remote-homology and PPI test sets (the only filter targets; SCOPe-40 was not
filtered) that also changes the sampling scheme, drops synthetic hard negatives,
and uses a true 1,024-example contrastive batch where V1's loss call saw 64 — so
V2 - V1 is not a decontamination ablation.

Where the reorganisation happens: per-query R@10 gain over ESM-2 is flat in each
query's maximum identity to our pretraining corpus — V2 +0.1524 at [0.2,0.4)
(n=164), +0.1810 at [0.4,0.7) (n=315), +0.1565 at [0.7,1.0] (n=1,214) — and among
the **404 queries where ESM-2 fails outright** at R@10, identity does not predict
the gain (Spearman +0.038, p=0.45). Neighbourhoods reorganise regardless of
proximity to training data. This is also our leakage control, and it cannot detect
fold-level overlap, only identity-level.

### 3. Bootstrap confidence intervals (Q3)

Every retrieval metric is a mean over per-query values, so resampling queries
gives the sampling distribution with no refitting. 10,000 resamples, **paired**
(the same queries score every method), 1,693 eligible queries:

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - HMMER | **-0.0124 [-0.0372, +0.0124] unresolved** | +0.1412 [+0.1205, +0.1618] | +0.1708 [+0.1511, +0.1905] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

Against us: **V1 - HMMER at R@1 is -0.1110 [-0.1388, -0.0827]** and V1 - MMseqs2
is -0.0697 [-0.0975, -0.0413] — the submitted model loses top-1 to alignment
outright. MMseqs2 beats ESM-2 35M at top-1 by +0.1565 [+0.1276, +0.1855], and
MMseqs2 vs ESM-2 at R@10 (-0.0213 [-0.0484, +0.0047]) is unresolved, so we do not
claim an untuned pLM beats alignment at depth. MMseqs2 returns few candidates past
rank 10 at these flags, so part of its depth deficit is coverage; HMMER does not
have that problem and V2 still leads it at depth by +0.1412.

**We did not compute intervals for the 23-task table.** Your objection stands
there: each cell is one run, and any delta inside ±0.005 is reported as a tie.

### 4. Seed variability (Q4)

**Five seeds (0-4) x 8 representative tasks x 3 arms, 3-NN, test split. Median SD
across all 24 rows is 0.0000.** The reason is mechanical: with fixed embeddings and a fixed test split a 3-NN probe is deterministic, so the
benchmark seed only moves subsampling and CV-fallback splits. `thermostability` is
the one task that subsamples and the only one with visible spread — Spearman
0.4427 ± 0.0126 (ESM-2) / 0.4696 ± 0.0172 (V1) / 0.4568 ± 0.0156 (V2). Remote
homology accuracy is 0.5835 ± 0.0000 / 0.6589 ± 0.0002 / 0.6668 ± 0.0000; GB1
variant effect (Spearman) 0.6582 / 0.7108 / 0.7806, all ± 0.0000.

So probe/split randomness is **not** the noise source behind sub-1% deltas. The
binding uncertainty is elsewhere, and we state both parts: the bootstrap in
section 3 covers which proteins are in the benchmark, and **training**-side
variation is unmeasured, since only one training run per model exists. The one
bound we have there is checkpoint choice: the final V2 checkpoint is the last
training step (not benchmark-selected), and a near-trough control checkpoint
(step 4,208, where the 3-cycle cosine schedule bottoms) differs from it by
0.005-0.008 on every structural metric — at or above our own tie band. We
therefore do not treat any sub-0.01 structural delta as resolved, including the
V1→V2 remote-homology gap of +0.0079.

### 5. Table 5 absolute scores (Q5)

Withdrawn, for a reason worse than the one you raised. The relative cells are
unbounded over near-zero Spearman baselines (-126.9% is a sign
flip of magnitude 0.269x baseline, not a 127-point drop), *and* the estimator is
not constant across the table: the code sets `n_neighbors = max(1, min(3,
train_size))`, so the smallest few-shot cells are a 1-NN or 2-NN probe, not the
3-NN probe named in the caption. Absolute numbers from that run would inherit the
estimator change. The few-shot claims go with the table.

### Errors we found in our own submission

All evidence here is 35M. **There is no 150M model on the decontaminated
corpus**, so we do not defend the submitted 150M results, including the abstract's
+105% and +19.9%. SCOPe is evaluated on the **family** field over **2,207**
sequences, not superfamily over 100,000 — that figure is the evaluator's
`max_samples` cap echoed into the table. Our remote-homology test
split is TAPE's three holdouts pooled (718 fold + 1,254 superfamily + 1,272 family
= 3,244) with no column marking which — not hierarchy-disjoint as written, so its
pooled macro AUC is not comparable to published per-holdout accuracies.

Four of your five questions now carry measurements; Table 5 and the label-scarcity
claim are withdrawn. If that changes your assessment we ask you to reconsider; if
one item remains decisive, name it and we will answer it in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- paste-unit character count: 9396 (everything between BEGIN and END) -->
<!-- BEGIN jVGf -->
Both of your axes now have measurements, and the first one goes against us:
structural supervision is the single largest contributor, and under a trained
linear probe ProtSent loses to its own untuned backbone on most tasks, so the
general-purpose framing is withdrawn. What we can defend is a position on the
generality-accuracy curve you asked about — measured against two alignment
baselines rather than asserted — plus evidence that the non-structural sources do
distinct work.

### 1. Where ProtSent sits on the generality-accuracy trade-off (Q3 / W2)

We ran MMseqs2 as a full alternative pipeline over all 23 tasks under identical
metric definitions (family-level Recall@K with self excluded, per-class max
bitscore for classification so AUC stays comparable, 1-NN by bitscore for
regression; no-hit queries scored as failures, not dropped), at maximum
sensitivity `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3` rather than the much
weaker default. We also ran HMMER (phmmer, `-E 10`, top 300 hits per query,
identical scoring code) on SCOPe-40, because it is the harder alignment baseline
and it is the one that costs us a claim.

**Alignment beats the best embedding model outright on 3 of 23 tasks under a 3-NN
probe** (EC classification, GO molecular function, beta-lactamase fitness) **and 6
under a linear probe** (those plus enzyme catalytic efficiency, optimal pH,
stability). The margins are large: EC classification F1-macro 0.710 (MMseqs2) vs
0.598 (ESM-2 35M) and 0.562 (V1); GO-MF 0.585 vs 0.459 / 0.443. On beta-lactamase
MMseqs2 reaches Spearman 0.8026 and beats every embedding arm including our
retrained model. Where annotation transfers by homology, alignment is simply
better. At the other end, MMseqs2 is *below chance* on DeepSol solubility (AUC
0.4185): homology label-transfer is anti-correlated there.

SCOPe-40 retrieval, family level, 2,207-sequence gallery, self excluded, no-hit =
failure. Only 1,693 queries have a non-self same-family neighbour, so **Recall@K
is capped at 0.7671**; all rows use those 1,693 eligible queries.

| method (1,693 eligible queries) | R@1 | R@10 | MAP |
|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer) | **0.6970** | 0.7809 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.4222 |
| ProtSent-V1 35M (submitted) | 0.5859 | 0.8512 | 0.5511 |
| ProtSent-V2 35M (decontaminated retrain) | 0.6846 | **0.9220** | **0.6454** |

Paired bootstrap, 10,000 resamples, same queries score every method. **We do not
beat alignment at top-1.** V2 - HMMER at R@1 is -0.0124 [-0.0372, +0.0124],
unresolved; V1 - HMMER is -0.1110 [-0.1388, -0.0827], a clear loss. V2 leads
MMseqs2 at R@1 by only +0.0289 [+0.0035, +0.0544]. The embedding wins at depth
against both: V2 - HMMER +0.1412 [+0.1205, +0.1618] at R@10 and +0.1708 [+0.1511,
+0.1905] at MAP; V2 - MMseqs2 +0.1819 [+0.1607, +0.2026] and +0.2356 [+0.2159,
+0.2551]. MMseqs2 returns few candidates past rank 10 at these flags, so part of
its depth deficit is coverage — HMMER does not have that problem, which is why we
report it.

One caveat we raise ourselves, since this retrieval result is now the paper's
surviving positive claim and SCOPe-40 was never a decontamination target: per-query
R@10 gain over ESM-2 does not grow with a query's maximum identity to our
pretraining corpus (V2 +0.1524 at [0.2,0.4), +0.1810 at [0.4,0.7), +0.1565 at
[0.7,1.0]), and among the 404 queries where ESM-2 fails outright identity does not
predict the gain (Spearman +0.038, p=0.45). That rules out identity-level
memorization but not fold-level overlap — our supervision is Foldseek cluster and
Pfam family co-membership, so a training pair sharing a test domain's fold at 15%
identity survives a 40%-identity filter. The fold-exclusion control is the
experiment we did not run.

The trade-off, then: alignment wins single-best-hit and homology-transferable
annotation; the embedding wins ranking depth, and it is the only one of the two
that yields a fixed-width vector where there is no alignment signal at all. That
last property belongs to embeddings generally, not to ProtSent — under a linear
probe stock ESM-2 35M is the better embedding on 12 of 20 comparable tasks
(median -0.0139 for V1; V2 is 2 win / 7 tie / 11 lose, median -0.0107, tie band
±0.005). That record is why the general-purpose claim goes.

### 2. Is this more than structural-information injection? (Q1 / W1)

Partly not, and our own ablation says how much: removing AFDB drops the mean
relative gain from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and remote
homology from +40.5% to +15.3%. Structural supervision is the largest single
contributor.

The remainder is not structure, and each source leaves a fingerprint on a
different task family: without Pfam the model still improves 15/23 (mean +4.6%);
removing STRING moves PPI from +5.3% to -0.5% while leaving most other tasks
intact; removing the DMS objective reduces fitness gains (fluorescence +15.6% →
+10.4%). A pure structure-distillation model has no PPI dial and no fitness dial.
The limit on that argument: those are single-run relative-percent numbers from the
submitted (pre-decontamination) tables on the suite's default split — the same
reporting convention we withdraw for sub-1% cells elsewhere in this rebuttal. We
use them only for the direction and size of source-specific effects, which run
from 2 to 25 relative points, never for small ones. **We did not run the joint
no-AFDB/no-Pfam ablation you asked for**, so "does anything survive when both
structural sources are gone" is unanswered.

One decontaminated, absolute, non-structural number does exist: GB1 variant effect
(Spearman, 3-NN, test split, mean over 5 seeds) is 0.6582 (ESM-2 35M) / 0.7108
(V1) / 0.7806 (V2), SD 0.0000. Fitness ordering is a relation no structure
distillation model supervises.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton, ProTrek (W1 / Q3)

Those four inject structure into a sequence model by distilling a structure
encoder or structural tokens: the supervision is one relation (structural
similarity), the teacher is a structure model, and the target is a residue- or
sequence-level representation that mimics it. ProtSent's supervision is a
heterogeneous relation *graph* over sequences — Pfam family co-membership,
Foldseek cluster co-membership, STRING interaction, and DMS fitness order — with
no structure encoder anywhere in the pipeline, at training or inference. The
contribution claimed is that relation type is a design axis (section 2 shows each
type moving a different task family), not that this beats structure distillation:
we have **no matched runs against any of the four** and claim no superiority.
ProTrek is the trimodal (sequence/structure/text) retrieval-optimized point on the
same curve; we add it to Related Work as such, we expect it to win retrieval
accuracy against a 35M sequence-only model, and we did not run it.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.** ProtTucker is the
closest published analogue to our protocol — contrastive fine-tuning for remote
homology — and its absence is the one we would most want to fix. **We also did not
apply ProtSent to SaProt or ProSST**: not a backbone swap at the data level, since
both need residue-level structure tokens for the whole Pfam and STRING corpora,
which our AFDB/Foldseek pipeline does not supply for non-AFDB sequences.

### 4. The CoSENT objective on DMS data (Q4)

Your reading of our text is fair and our text is wrong: the paper says the DMS
loss "operates on single proteins rather than pairs." The released code writes
`(sentence_0, sentence_1, score)` rows — `sentence_0` is the wild-type,
`sentence_1` the mutant, `score` the within-assay normalized fitness rescaled to
[0,1] (clinical rows map benign to 1.0, pathogenic to 0.0). CoSENT is ordinal over
those pairs exactly as for sentences: within a batch, if pair p scores above pair
q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no absolute
cosine target and no term pulling high-fitness mutants to a common point, so it
does not flatten an assay. The real limitation is narrower and we state it: the
pairing is **wild-type-anchored**, so mutant-mutant geometry within an assay is
constrained only indirectly.

### 5. Scale and the minor note

Everything above is 35M. **There is no 150M model on the decontaminated corpus**;
the submitted 150M numbers were trained on the uncontrolled corpus and we do not
defend them. V2 is a retrain on corpora filtered at 40% identity / 80% coverage
against the remote-homology and PPI test sets (the only two filter targets;
SCOPe-40 was not filtered), and it also changes three other things at once —
proportional sampling, no synthetic hard negatives, and a true 1,024-example
contrastive batch where the submitted model's loss call saw 64 — so V2 - V1 is not
a controlled decontamination ablation and we attribute nothing to decontamination
beyond "it cost nothing on the task we filtered against."

The "?" at line 21 is a broken citation key, not a missing reference: Heinzinger
et al. 2022 (ProtTucker) and Redl et al. 2023 are both in Related Work.

If the measured trade-off, the source fingerprints and the positioning answer your
two axes, we ask you to reconsider. If the missing no-AFDB/no-Pfam ablation is the
decisive one, say so and we will address it in discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- paste-unit character count: 9925 (everything between BEGIN and END) -->
<!-- BEGIN Yi1G -->
Leakage was your most serious concern and we treated it as decisive: all three
pretraining corpora re-filtered at 40% identity / 80% coverage, retrained from
scratch, every benchmark re-run. Both baselines you named were run, and HMMER
retires our top-1 retrieval claim — we retract it in item 7. Your eight weaknesses
in order, then the single-space assumption.

### 1. Leakage

**Decontamination, completed.** MMseqs2 `easy-search`, corpus as query, test set as
target (`--min-seq-id 0.4 --cov-mode 1 -c 0.8 -e 1e-3`); any pretraining sequence
with a hit is dropped. Pfam 28,530,684 → 27,929,772 rows and AFDB 135,404,259 →
126,301,607 against `remote_homology` test (3,244 seqs); STRING 76,070,154 →
71,891,417 pairs against `ppi_bernett` test (3,022 seqs). **Those two test sets
were the only filter targets** — every other benchmark test set, SCOPe-40 included,
was not filtered against. Negative controls (1,000 random sequences from each
*filtered* corpus, re-searched against its target) return **0 hits**; AFDB's was
re-run exhaustively because its k-mer prefilter is only 89.4% recall, and still
returned 0. The filter was then verified on the parquet files training actually
opened, by semi-join with the removal lists: **0 flagged sequences survived**, and
their row counts sum to the 169,231,379 in the training log.

**What the retrain does and does not show.** Remote homology (pooled 457-class,
test split), ESM-2 35M / V1 / V2: 3-NN accuracy 0.5835 / 0.6589 / 0.6668, linear
accuracy 0.6868 / 0.6899 / 0.7016, linear macro-F1 0.4414 / **0.4281** / 0.4527 —
V1 is below the untuned backbone on macro-F1; only V2 improves on both metrics
under both probes. But V2 changes four things at once: filtered corpus,
proportional sampling, no synthetic hard negatives, and a true 1,024-example
contrastive batch where V1's loss call saw 64 (item 3). With no unfiltered-corpus
retrain at the V2 configuration we attribute nothing to decontamination, and
+0.0079 is within checkpoint-choice noise (item 6). The supported claim is only
that **decontamination did not cost accuracy on the task we filtered against** —
and not even that for PPI, since we do not re-report a post-decontamination
`ppi_bernett` number here. There is also no 150M model on the decontaminated
corpus, so the submitted 150M results are undefended.

**SCOPe-40 cannot be decontaminated, by us or anyone**: no train/test split
(leave-one-out over 2,207 domains), median maximum identity to our corpus
**0.908**, none below 20%, so filtering removes essentially every structured
domain. We tested memorization directly: memorization would make queries with a
closer pretraining neighbour gain more. Across the
1,693 eligible queries V2's per-query Recall@10 gain over ESM-2 is flat in max
identity to the corpus — +0.1524 at [0.2,0.4), +0.1810 at [0.4,0.7), +0.1565 at
[0.7,1.0] (n = 164 / 315 / 1,214; the [0,0.2) bin is **empty**) — the identity-gain
Spearman is -0.038 (R@10) and -0.116 (average precision, p < 3e-6), still negative
after controlling for headroom (partial -0.081, p < 1e-3), and among the **404
queries the untuned backbone fails completely** at Recall@10 identity does not
predict the gain at all (+0.038, p=0.45).

**What these controls cannot rule out.** Our supervision is Foldseek
structural-cluster and Pfam family co-membership, so a training pair sharing a test
domain's *fold* at 15% identity survives a 40%-identity filter, and identity
stratification cannot detect fold-level label overlap. We cannot say SCOPe-40, or
the fold-level third of the remote-homology set, is free of it — and SCOPe is where
our one surviving positive claim lives. The right experiment, excluding SCOPe
queries whose fold appears among training clusters, we did not run.

### 2. DMS objective

The ordering objective you describe is the one implemented; our text is wrong
("operates on single proteins rather than pairs"). Each row is (wild-type, mutant,
within-assay normalized fitness in [0,1]) and CoSENT ranks pairs within a batch: if
mutant a beats mutant b, the loss pushes cos(WT, a) above cos(WT, b). No absolute
target, nothing collapsing high-fitness variants together. The real limitation is
that the pairing is WT-anchored, so mutant-mutant distances are constrained only
indirectly.

### 3. MNRL batch semantics and Eq. 1

Correct — a real error. The submitted 1,024 is an **optimizer** batch reached by
gradient accumulation (35M: per-device 64 x 16 steps; 150M: 16 x 64, our Table 6),
which does not share in-batch negatives across micro-batches, so each MNRL call saw
**64** examples at 35M and **16** at 150M. The retrain uses
`CachedMultipleNegativesRankingLoss` with a true 1,024-example contrastive batch
per device (cross-device gather off). In Eq. 1 the numerator should use the
positive paired with anchor i, the denominator the positives of all N pairs.

### 4. Pair-level tasks

PPI: each partner is embedded independently and the two vectors concatenated
(`np.concatenate([emb[s1], emb[s2]])`) before the probe. Peptide-HLA is **not** a
two-input task here — the dataset supplies one `seq` field, a pipe-joined
`HLA_pseudoseq|peptide` string, so no operator applies. Neither was in the paper.

### 5. k-NN regression

Uniform: `KNeighborsRegressor(n_neighbors=3, metric="minkowski")`, an unweighted
mean over 3 Euclidean neighbours. At small N the code sets `n_neighbors = max(1,
min(3, train_size))`, so the smallest few-shot cells are a different estimator;
with relative changes over near-zero Spearman baselines (-126.9% is a sign flip of
magnitude 0.269x baseline), that table is uninterpretable and its claims are
withdrawn.

### 6. Ablations

Agreed, and we acted on it. Removing synthetic hard negatives improves 20/23
tasks at mean +7.9% against 16/23 and +6.7% for the submitted configuration;
proportional sampling gives +7.0% vs round-robin's +6.7%. The retrain uses neither
default — but those ablations were scored on these same benchmarks, so V2's
configuration was chosen with benchmark results in view, a selection channel the
corpus filter does not touch. The checkpoint was not: it is the last training step,
and a near-trough control (step 4,208) differs by 0.005-0.008 on every structural
metric, at or above our ±0.005 band, so we treat no sub-0.01 structural delta as
resolved.

### 7. Baselines

**HMMER (phmmer) is the stronger alignment baseline and it retires our top-1
claim.** Same gallery and scoring code, `-E 10`, top 300 hits/query; the 691 of
2,207 queries returning no hit are scored as failures. On the 1,693 eligible
SCOPe-40 queries (Recall@K capped at 0.7671; only those have a non-self same-family
neighbour), R@1 / R@10 / MAP: **HMMER 0.6970 / 0.7809 / 0.4747**; MMseqs2 (`-s 7.5
-e 10`) 0.6556 / 0.7401 / 0.4098; ESM-2 35M 0.4991 / 0.7614 / 0.4222; V1 0.5859 /
0.8512 / 0.5511; V2 0.6846 / 0.9220 / 0.6454. MMseqs2 was also run over all 23
tasks under identical metric definitions and beats the best embedding arm on 3
under 3-NN and 6 under a linear probe (EC F1-macro 0.710 vs ESM-2's 0.598).

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek** — we claim no
superiority to any; ProtTucker is the closest published analogue to our protocol.
Redl et al. 2023, "Optimizing Protein Language Models with Sentence Transformers",
is in Related Work but **was not run**, and we have no matched comparison.

### 8. Statistical evidence

Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples; the same
queries score every method. **V2 - HMMER at R@1 is -0.0124 [-0.0372, +0.0124],
unresolved; V1 - HMMER is -0.1110 [-0.1388, -0.0827]**; V2 - MMseqs2 is +0.0289
[+0.0035, +0.0544]. At depth V2 - HMMER is +0.1412 [+0.1205, +0.1618] (R@10) and
+0.1708 [+0.1511, +0.1905] (MAP).

Seeds, since Table 2's small deltas are your point: 5 seeds x 8 tasks x 3 arms,
3-NN, test split — median SD across all 24 rows is **0.0000** (a 3-NN probe on
fixed embeddings and a fixed split is deterministic; `thermostability` is the only
task that subsamples, SD 0.013-0.017). The probe is not the noise source, and
training-seed variance is unmeasured: one training run per model exists.

For the 23-task table we have **no** confidence intervals: every cell is one run,
so any delta inside ±0.005 is a tie. Against ESM-2 35M over the 20 comparable tasks
(three multiclass tasks, remote homology among them, are excluded from both arms:
their AUC metric is undefined when the test split holds a class absent from train),
V1 is 11 win / 3 tie / 6 lose under 3-NN (median +0.0075) and **4 / 4 / 12
under a linear probe** (median -0.0139); V2 is 10/3/7 and 2/7/11. That is why the
general-purpose claim is withdrawn.

### 9. Limitations: one space for heterogeneous relations

Removal ablations show the relation types do not collapse into one another: STRING
removal moves PPI from +5.3% to -0.5% while leaving most other tasks intact, DMS
removal moves fluorescence from +15.6% to +10.4%, AFDB removal moves remote
homology from +40.5% to +15.3%. We cannot show the absence of *interference*: the
linear-probe record above is consistent with the shared space costing something.

### Errors we found in our own submission

The paper's PPI decontamination description does not match the released code
(`easy-search` at 40% identity removing hit query IDs, not `easy-linclust` at 50%
with cluster removal). The remote-homology split is not hierarchy-disjoint — TAPE's
three holdouts pooled (718 + 1,254 + 1,272 = 3,244) — so its pooled macro AUC is
not comparable to published per-holdout accuracies. And SCOPe is the **family**
field over **2,207** sequences, not superfamily over 100,000: that is the
evaluator's `max_samples` cap.

If the completed decontamination, the corrected baselines and the retractions
resolve your first weakness, we ask you to reconsider. If the un-run fold-overlap
control on SCOPe remains decisive, say so and we will answer it in discussion.
<!-- END Yi1G -->
