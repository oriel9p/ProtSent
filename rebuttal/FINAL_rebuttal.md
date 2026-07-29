# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — POSTABLE FINAL

Post each response under its own review. No links, no attachments, no figures.
Every number appears in the response text with its metric, split and model.

Naming used throughout: **V1** = the submitted/published 35M model.
**V2** = the 35M model retrained during the rebuttal on the decontaminated corpora.

Paste unit = everything strictly between `<!-- BEGIN X -->` and `<!-- END X -->`,
stripped of leading and trailing whitespace. The character count stated above each
BEGIN marker is that body; the comment itself sits outside the paste unit.

---

## Response to Reviewer HNXd

<!-- character count of the pasted body below: 9610 (limit 10,000) -->
<!-- BEGIN HNXd -->
**All five questions now carry measurements**, absolute few-shot scores with seed
SDs included. Two of the answers go against us.

**What we defend, narrower than the paper:** contrastive fine-tuning of a 35M
ESM-2 backbone, evaluated frozen, makes family/fold relations *locally*
recoverable — SCOPe-40 retrieval depth and remote homology — and adds nothing a
trained linear head could not already extract from mean-pooled ESM-2. Withdrawn:
general-purpose superiority (your Q2 killed it), the label-scarcity claim, Table
5, all 150M results. Our strongest surviving result sits on SCOPe-40, the one
benchmark we could not decontaminate.

### 1. Retrieval and how the space reorganises (Q1)

**We computed no clustering statistics — no silhouette, NMI or ARI.** We ran
direct retrieval, a per-query organisation analysis, and a layer sweep.

SCOPe-40, **family** level, 2,207-domain gallery, leave-one-out, self excluded,
no-hit queries scored as failures. Only 1,693 of 2,207 queries have a non-self
same-family neighbour; every row is restricted to those, where the Recall ceiling
is 1.0 (over all 2,207 each method's Recall scales by 1,693/2,207 = 0.7671).

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10 --max-seqs 300`) | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| HMMER (phmmer, `-E 10`, top 300) | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M (submitted) | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 35M (retrained) | 0.6852 | **0.9220** | **0.9634** | **0.6459** |

**Alignment is the better top-1 method**, and V1 loses top-1 to both tools. Our
advantage is ranking *depth*, and part of that margin is list coverage rather
than ranking: 691 of 2,207 queries return no phmmer hit at `-E 10` and score 0 at
every K, and both tools flatten from R@10 to R@30 (+0.0171, +0.0165) where the
embeddings do not (+0.0726 ESM-2, +0.0414 V2). Until both are re-run with that
threshold removed, the depth margin is an upper bound.

Where the reorganisation happens: V2's per-query R@10 gain over ESM-2 is flat in
each query's maximum identity to our pretraining corpus (+0.1524 at [0.2,0.4),
n=164; +0.1565 at [0.7,1.0], n=1,214), and among the 404 queries where ESM-2
fails outright at R@10, identity does not predict the gain
(Spearman +0.038, p=0.45). This is identity-level only; it cannot see fold-level
overlap.

V2 is a retrain on corpora filtered at 40% identity / 80% coverage against the
remote-homology and PPI test sets — the only two filter targets, SCOPe-40 was not
one — with the settings our own ablations favour and a true 1,024-example
contrastive batch where V1's loss call saw 64. **V2 - V1 is therefore not a
decontamination ablation**; no unfiltered-corpus retrain at that recipe exists.

### 2. Linear probe, and a frozen linear baseline under label scarcity (Q2)

23 tasks, `--eval_split test`, frozen mean-pooled embeddings, vs stock ESM-2 35M:

| probe (20 comparable tasks) | V1 | V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

Reading rules, all against us. The ±0.005 tie band is in absolute units across
accuracy, Spearman, F1 and AUC; no setting of it turns the linear record into a
win. Three tasks are outside the 20 — `antibiotic_resistance`, `remote_homology`,
`temperature_stability` — dropped from *both* columns because one-vs-rest AUC is
undefined when the test split holds a class absent from train, which removes our
best task from the tally. Three inside the 20 use a built-in evaluator that
ignores the probe flag and contribute identically to both rows: EC (F1-macro
0.598 ESM-2 vs 0.562 V1) and GO-MF (0.459 vs 0.443), both losses, and SCOPe-40
retrieval, a win. So only 17 of 20 cells differ by probe. The probe is
scikit-learn defaults, untuned.

Remote homology (pooled 457-class, test split), where contrastive training does
add something: 3-NN accuracy 0.5835 (ESM-2) / 0.6587 (V1) / **0.6668** (V2);
linear accuracy 0.6868 / 0.6899 / **0.7016**; linear macro-F1 0.4414 / **0.4281**
/ 0.4527 — V1 is *below* the untuned backbone there, and V1's +0.0031 linear
accuracy is a tie by our own band. Only V2 improves on both metrics under both
probes.

One diagnostic, not a defence: both probes pool the **final** layer, and a
per-layer linear sweep (subsampled 8,000 train / 3,000 test, so not comparable to
the numbers above) shows that is the worst layer for remote homology in *both*
models — ESM-2 0.6373 there vs 0.6703 at layer 6; V2 0.6803 vs 0.7033 at layer 8,
with V2 ahead at every layer from 6 up. Two tasks, one scale; it does not
overturn the table.

**Your level-gap hypothesis: tested, and the probe is not the cause.** BIOMAP
`stability_prediction` labels are continuous floats and our suite scores the task
by Spearman, so our "58.8%" is a correlation ×100, not an accuracy commensurate
with 69.08% linear / 77.69% LoRA; we withdraw that comparison. And 3-NN scores
*higher* than our linear probe there for every arm (ESM-2 35M Spearman 0.6435 vs
0.4395), so the probe change you suggested moves the number the wrong way.

### 3. Bootstrap confidence intervals (Q3)

Every retrieval metric is a mean over per-query values, so resampling the 1,693
queries needs no refitting. 10,000 resamples, **paired**; this run reproduces
every cell of the table above to within 0.0012.

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - HMMER | **-0.0124 [-0.0372, +0.0124]** | +0.1412 [+0.1205, +0.1618] | +0.1708 [+0.1511, +0.1905] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

**We do not claim to beat alignment at top-1**: V2 ties HMMER and passes only the
weaker tool. V1 - HMMER is -0.1110 [-0.1388, -0.0827] and V1 - MMseqs2 -0.0697
[-0.0975, -0.0413], outright losses.

**We computed no intervals for the 23-task table; your objection stands there.**
Each cell is one run at seed 42, any delta inside ±0.005 is called a tie, and the
checkpoint spread below is at or above that band, so a large part of that table
is unresolved by our own standard.

### 4-5. Few-shot: absolute scores and seed variability (Q5, Q4)

**Table 5 is replaced, not defended.** Its relative cells were uninterpretable —
unbounded over near-zero Spearman baselines (-126.9% is a sign flip of magnitude
0.269× the baseline) — and the estimator was not constant across it, since the
code sets `n_neighbors = max(1, min(3, train_size))`. Re-run with absolute
scores, 5 training-subset draws per point, test split held at full size, both
probes fit on the same subset. **Remote homology, accuracy, mean ± SD, kNN /
linear:**

| N | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|---|
| 50 | 0.061±0.010 / 0.121±0.003 | 0.055±0.008 / 0.159±0.004 | 0.045±0.009 / 0.145±0.005 |
| 1000 | 0.185±0.002 / 0.288±0.014 | **0.318±0.015 / 0.377±0.008** | 0.289±0.016 / 0.355±0.009 |

Stability (Spearman) at N=50: ESM-2 0.178±0.054 kNN / 0.216±0.202 linear vs V2
0.327±0.161 / 0.200±0.131. Metal-ion binding at N=1000 under the linear head:
ESM-2 0.666±0.001 beats V1 0.637±0.004 and V2 0.595±0.001.

Three conclusions, the first against us. (i) **Your proposed framing is not
supported by our data**: a trained linear head beats 3-NN in almost every
model/task/N cell, including N=50, so "linear degrades while k-NN stays
competitive" is false here, and the label-scarcity claim is withdrawn rather than
reframed. (ii) The few-shot advantage is real but task-specific — large on remote
homology, absent on metal-ion binding. (iii) At small N the seed spread is as
large as the effect (Stability at N=100: ±0.20 on means of 0.28-0.40), your
concern confirmed. **We ran no fine-tuning sweep**, so how a fine-tuned ESM-2
compares is unmeasured.

Full-data evaluation is near-deterministic: 5 seeds × 8 tasks × 3 arms under 3-NN
gives a median SD of 0.0000 across 24 rows, since fixed embeddings and a fixed
test split make that probe deterministic; only `thermostability` subsamples (SD
0.013-0.017). Two caveats we volunteer: one *training* run per model exists, so
training-seed variance is unmeasured; and checkpoint 4,000 — the snapshot nearest
the last cosine trough at step 4,208 — differs from the final V2 checkpoint by
0.005-0.008 on every structural metric. We therefore call no sub-0.01 structural
delta resolved, including the +0.0079 V1→V2 remote-homology gap.

### Errors we found in our own submission

All evidence here is 35M. **There is no 150M model on the decontaminated
corpus**, so we do not defend the submitted 150M numbers, including the abstract's
+105% and +19.9%. SCOPe is evaluated on the **family** field over **2,207**
sequences, not superfamily over 100,000 — that figure is the evaluator's
`max_samples` cap. Our remote-homology test split is TAPE's three holdouts pooled
(718 + 1,254 + 1,272 = 3,244), not hierarchy-disjoint as written, so its pooled
macro AUC is not comparable to published per-holdout accuracies; and the paper's
"+40.5%" there is a relative macro-F1 change (.223 → .313) on the suite's default
split, not the test-split numbers above, which we do not mix. The camera-ready is
therefore a 35M retrieval-and-remote-homology paper, both probes on the test
split.

If that changes your assessment we ask you to reconsider; if one item remains
decisive, name it and we will answer it in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 9234 (limit 10,000) -->
<!-- BEGIN jVGf -->
Both of your axes now carry measurements, and the first goes against us:
structural supervision is the single largest contributor, and under a trained
linear probe ProtSent loses to its own untuned backbone on most tasks, so the
general-purpose framing is withdrawn. What we defend is a *measured* position on
the generality-accuracy curve, plus evidence that the non-structural sources do
distinct work. Everything below is 35M: **there is no 150M model on the
decontaminated corpus** and we do not defend the submitted 150M results.

### 1. Where ProtSent sits on the generality-accuracy trade-off (Q3 / W2)

We ran MMseqs2 as a full alternative pipeline over all 23 tasks under identical
metric definitions (family-level Recall@K with self excluded, per-class max
bitscore for classification so AUC stays comparable, 1-NN by bitscore for
regression; no-hit queries scored as failures, not dropped), at
`-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3` rather than the much weaker
default — at `-s 5.7` the same baseline gives SCOPe R@1 0.3847, so any MMseqs2
number needs its sensitivity stated. We also ran HMMER (phmmer, `-E 10`, top 300
hits per query, identical scoring code) on SCOPe-40, because it is the harder
alignment baseline and the one that costs us a claim.

**Alignment beats the best embedding model outright on 3 of 23 tasks under a 3-NN
probe** (EC classification, GO molecular function, beta-lactamase fitness) **and
6 under a linear probe** (those plus enzyme catalytic efficiency, optimal pH,
stability). The margins are large: EC classification F1-macro 0.710 (MMseqs2) vs
0.598 (ESM-2 35M) and 0.562 (V1); GO-MF 0.585 vs 0.459 / 0.443; beta-lactamase
Spearman 0.8026, above every embedding arm including our retrained model. Where
annotation transfers by homology, alignment is simply better. At the other end
MMseqs2 is *below chance* on DeepSol solubility (AUC 0.4185) — homology
label-transfer is anti-correlated there.

SCOPe-40, family level, 2,207-domain gallery, self excluded, no-hit = failure.
Only 1,693 queries have a non-self same-family neighbour; all rows use those
1,693, where the Recall ceiling is 1.0 (over all 2,207 every method scales by
1,693/2,207 = 0.7671).

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| HMMER (phmmer) | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M (submitted) | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 35M (decontaminated retrain) | 0.6852 | **0.9220** | **0.9634** | **0.6459** |

Paired bootstrap, 10,000 resamples, the same queries scoring every method (it
reproduces the table above to within 0.0012). **We do not beat alignment at
top-1**: V2 - HMMER at R@1 is -0.0124 [-0.0372, +0.0124], unresolved, and V1 -
HMMER is -0.1110 [-0.1388, -0.0827], a clear loss; V2 leads MMseqs2 by only
+0.0289 [+0.0035, +0.0544] while V1 loses to it by -0.0697 [-0.0975, -0.0413].
The embedding wins at depth against both: V2 - HMMER +0.1412 [+0.1205, +0.1618]
at R@10 and +0.1708 [+0.1511, +0.1905] at MAP; V2 - MMseqs2 +0.1819 [+0.1607,
+0.2026] and +0.2356 [+0.2159, +0.2551].

**Two limits on that depth result, both ours.** Part of it is list coverage
rather than ranking quality: 691 of 2,207 queries return no phmmer hit at `-E 10`
and are scored 0 at every K, and both tools flatten from R@10 to R@30 (+0.0171
HMMER, +0.0165 MMseqs2) where the embeddings do not (+0.0726 ESM-2, +0.0414 V2).
We did not re-run either tool with the reporting threshold removed, so the depth
margin is an upper bound. And SCOPe-40 was never a decontamination target:
per-query R@10 gain over ESM-2 does not grow with a query's maximum identity to
our corpus (V2 +0.1524 at [0.2,0.4), n=164; +0.1810 at [0.4,0.7), n=315; +0.1565
at [0.7,1.0], n=1,214) and among the 404 queries where ESM-2 fails outright
identity does not predict the gain (Spearman +0.038, p=0.45) — which rules out
identity-level memorization but not fold-level overlap, since our supervision is
Foldseek-cluster and Pfam-family co-membership and a training pair sharing a test
domain's fold at 15% identity survives a 40%-identity filter. The fold-exclusion
control is the experiment we did not run.

The trade-off, then: alignment wins single-best-hit and homology-transferable
annotation; the embedding wins ranking depth and is the only one of the two that
yields a fixed-width vector where there is no alignment signal at all. That last
property belongs to embeddings generally, not to ProtSent — under a linear probe
stock ESM-2 35M is the better embedding on 12 of 20 comparable tasks (V1 4 win /
4 tie / 12 lose, median -0.0139; V2 2 / 7 / 11, median -0.0107; tie band ±0.005,
test split, one seed each). That record is why the general-purpose claim goes.

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
**The limit on that argument, stated plainly:** those are single-run
relative-percent numbers from the submitted, pre-decontamination model on the
suite's default split — the same reporting convention we withdraw for sub-1%
cells elsewhere in this rebuttal. They support the *direction* of source-specific
effects at 2-25 relative points and nothing finer, and they cannot establish that
the sources fail to interfere; the linear-probe record above is consistent with
the shared space costing something. **We did not run the joint no-AFDB/no-Pfam
ablation you asked for**, so "does anything survive when both structural sources
are gone" is unanswered.

One decontaminated, absolute, non-structural number does exist: GB1 variant
effect (Spearman, 3-NN, test split, mean over 5 seeds) is 0.6582 (ESM-2 35M) /
0.7108 (V1) / 0.7806 (V2), SD 0.0000. Fitness ordering is a relation no structure
distillation model supervises.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton, ProTrek (W1)

Those four inject structure into a sequence model by distilling a structure
encoder or structural tokens: one relation type, a structure-model teacher, and a
residue- or sequence-level target that mimics it. ProtSent supervises a
heterogeneous relation *graph* over sequences — Pfam family co-membership,
Foldseek cluster co-membership, STRING interaction, DMS fitness order — with no
structure encoder anywhere, at training or inference. The claim is that relation
*type* is a design axis (section 2 shows each type moving a different task
family), not that this beats structure distillation: we have **no matched runs
against any of the four and claim no superiority to them**. ProTrek is the
trimodal (sequence/structure/text) retrieval-optimized point on the same curve;
we add it to Related Work as such, expect it to win retrieval accuracy against a
35M sequence-only encoder, and did not run it.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek**, nor Redl et al. 2023.
ProtTucker is the closest published analogue to our protocol — contrastive
fine-tuning for remote homology — and its absence is the one we would most want
to fix. **We also did not apply ProtSent to SaProt or ProSST**: the blocker is
data, not code, since both consume residue-level structure tokens and we have no
predicted structures for the Pfam and STRING sequences, which are the majority of
the corpus.

### 4. The CoSENT objective on DMS data (Q4)

Your reading is fair and our text is wrong: the paper says the DMS loss "operates
on single proteins rather than pairs." The released code writes
`(sentence_0, sentence_1, score)` rows — `sentence_0` is the wild-type,
`sentence_1` the mutant, `score` the within-assay normalized fitness rescaled to
[0,1] (clinical rows map benign to 1.0, pathogenic to 0.0). CoSENT is ordinal
over those pairs exactly as for sentences: within a batch, if pair p scores above
pair q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no
absolute cosine target and no term pulling high-fitness mutants toward a common
point, so it does not flatten an assay. The real limitation is narrower and we
state it: the pairing is **wild-type-anchored**, so mutant-mutant geometry within
an assay is constrained only indirectly.

The "?" at line 21 is a broken citation key, not a missing reference: Heinzinger
et al. 2022 (ProtTucker) and Redl et al. 2023 are both in Related Work.

If the measured trade-off, the source fingerprints and the positioning answer
your two axes, we ask you to reconsider. If the missing no-AFDB/no-Pfam ablation
or the missing ProtTucker run is the decisive one, say which and we will address
it in discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9969 (limit 10,000) -->
<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: all three
corpora re-filtered, retrained from scratch, benchmarks re-run. HMMER and MMseqs2
were both run. Every number defended below is **V2**'s, not the submitted
model's.

### 1. Leakage

**Decontamination, done.** MMseqs2 `easy-search`, corpus as query, test set
as target, 40% identity / 80% coverage; any corpus sequence with a hit is
dropped. Pfam 28,530,684 → 27,929,772 rows and AFDB 135,404,259 → 126,301,607
against `remote_homology` test (3,244 seqs); STRING 76,070,154 → 71,891,417 pairs
against `ppi_bernett` test (3,022 seqs). **Those two test sets were the only
filter targets** — every other benchmark test set, SCOPe-40 included, was not.
Verified on the parquet files training actually opened, by semi-join with the
removal lists: **0 flagged sequences survived**, row counts
summing to the 169,231,379 in the training log. Negative controls (1,000 random
sequences per *filtered* corpus, re-searched; AFDB's exhaustively, its k-mer
prefilter being only 89.4% recall) return **0 hits**, which by the rule of three
bounds the residual only at ~0.3%.

**What the retrain shows, and what it does not.** Remote homology (pooled
457-class, test split), ESM-2 35M / V1 / V2: 3-NN accuracy 0.5835 / 0.6587 /
0.6668; linear accuracy 0.6868 / 0.6899 / 0.7016; linear macro-F1 0.4414 /
**0.4281** / 0.4527 — V1 below the untuned backbone there, only V2 improving on
both metrics under both probes. But V2 also changes sampling, drops synthetic
hard negatives, and uses a true 1,024-example contrastive batch where V1's loss
call saw 64 (item 3). With no
unfiltered-corpus retrain at the V2 configuration **nothing is attributable to
decontamination in either direction**, and +0.0079 is inside item 6's checkpoint
spread anyway. Only the weak claim is supported: a decontaminated corpus still
trains a model at least as good as the submitted one on the filtered task.

**PPI: no post-decontamination number exists, and we are not withholding one.**
`ppi_bernett` is a pair-input task and is not in the 23-task sweep, so the filter
target that cost 4,178,737 STRING pairs has no downstream result (the
paper's +5.3% is a pre-decontamination V1 number). That half of weakness 1 is
unanswered.

**SCOPe-40 cannot be decontaminated at the corpus level**: no train/test split
(leave-one-out over 2,207 domains), median maximum identity to our corpus
**0.908**, none below 20%, so filtering against it removes essentially every
structured domain. The *evaluation* can still be restricted, and memorization
would make queries with a closer pretraining neighbour gain more.
V2's per-query R@10 gain over ESM-2 is instead flat in identity: +0.1524 at
[0.2,0.4) (n=164), +0.1565 at [0.7,1.0] (n=1,214), the [0,0.2) bin empty. Those
164 queries are themselves a decontaminated evaluation at the 40% threshold used
on every corpus (V2's MAP gain there is +0.2859 vs +0.2232 overall).
Identity-gain Spearman is -0.116 (average precision, p < 3e-6), negative after a
headroom control (partial -0.081), and among the **404 queries the untuned
backbone fails completely** identity does not predict the gain (+0.038, p=0.45).

**What this cannot rule out.** Supervision is Foldseek cluster and Pfam family
co-membership, so a training pair sharing a test domain's *fold* at 15% identity
survives a 40%-identity filter: identity stratification cannot see fold-level
*label* overlap, and a flat slope is what that leakage would also produce. We
cannot say SCOPe-40, or the fold third of the remote-homology set, is free of it,
and SCOPe carries our one surviving claim. The right experiment — excluding
queries whose fold is among training clusters — we did not run.

### 2. DMS objective

Implemented as you describe; our text ("operates on single proteins rather than
pairs") is wrong. Rows are (wild-type, mutant,
within-assay normalized fitness in [0,1]) and CoSENT ranks pairs within a batch:
if mutant a beats mutant b, the loss pushes cos(WT, a) above cos(WT, b) — no
absolute target, nothing collapsing high-fitness variants. The limitation: the
pairing is WT-anchored, so mutant-mutant distances are only indirect.

### 3. MNRL batch semantics and Eq. 1

Correct — a real error. The submitted 1,024 is an **optimizer** batch from
gradient accumulation (35M: 64 per device × 16 steps; 150M: 16 × 64, our Table
6), and accumulation does not share in-batch negatives, so each MNRL call saw
**64** examples at 35M and **16** at 150M — misdescribed, and the likeliest
explanation for the 150M results, which we no longer defend. The retrain uses a
true 1,024-example batch per device. In Eq. 1 the numerator should use the positive
paired with anchor i, the denominator the positives of all N pairs.

### 4-5. Pair-level tasks and k-NN regression

PPI: partners are embedded independently and the two vectors concatenated
before the probe. Peptide-HLA is **not** a
two-input task here — the dataset supplies one `seq` field, a pipe-joined
`HLA_pseudoseq|peptide` string — so no operator applies. Neither was in the
paper. k-NN regression is **uniform**: `KNeighborsRegressor(n_neighbors=3,
metric="minkowski")`, an unweighted mean over 3 neighbours; at small N the code
sets `n_neighbors = max(1, min(3, train_size))`, so the smallest few-shot cells
are a different estimator — which, with relative changes over near-zero baselines,
is why Table 5 is replaced by absolute means with seed SDs.

### 6. Ablations

Agreed, and we acted on it. Removing synthetic hard negatives improves 20/23
tasks at mean +7.9% against 16/23 and +6.7% for the submitted configuration;
proportional sampling gives +7.0% vs round-robin's +6.7%. V2 uses neither
default. **The consequence, stated not implied:** those
ablations were scored on these same benchmarks, so V2's configuration was chosen
with benchmark results in view — a selection channel the corpus filter does not
touch, and V2's numbers are therefore not a clean held-out measurement. The
checkpoint was not chosen that way: it is the last training step, and checkpoint
4,000 (the snapshot nearest the last cosine trough at step 4,208) differs from it
by 0.005-0.008 on every structural metric, at or above item 8's ±0.005 band, so
no sub-0.01 structural delta is resolved. They are also all we have on your
single-space question: each source moves its own task family (STRING removal
takes PPI +5.3% → -0.5%), which is not evidence against interference.

### 7. Baselines

**HMMER (phmmer, `-E 10`, top 300 hits/query, no-hit = failure) was run**, same
gallery, same scoring code as MMseqs2. SCOPe-40, **family** level, 2,207 domains;
the 1,693 eligible queries are those with a non-self same-family neighbour, so
the Recall ceiling is 1.0 (over all 2,207 every method scales by 0.7671):

| method (1,693 eligible) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| HMMER (phmmer) | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| MMseqs2 (`-s 7.5 -e 10`) | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 | 0.6852 | 0.9220 | 0.9634 | 0.6459 |

Paired bootstrap, 10,000 resamples, same queries throughout (it reproduces this
table to within 0.0012). At R@1: **V2 - HMMER -0.0124 [-0.0372,
+0.0124], unresolved; V1 - HMMER -0.1110 [-0.1388, -0.0827]**, an outright loss;
V2 - MMseqs2 +0.0289 [+0.0035, +0.0544]. At depth, V2 - HMMER +0.1412 [+0.1205,
+0.1618] (R@10) and +0.1708 [+0.1511, +0.1905] (MAP).

**Alignment remains the better top-1 method.** Against our own depth claim: 691
of 2,207 queries return no phmmer hit and score 0 at every K, and both tools
flatten almost identically from R@10 to R@30 (+0.0171, +0.0165) where the
embeddings do not (+0.0726 ESM-2, +0.0414 V2) — so part of the depth gap is
candidate coverage, not ranking, and until both are re-run with that
threshold removed the depth margin is an upper bound. MMseqs2, run over
all 23 tasks, beats the best embedding arm on 3 under 3-NN and 6 under a linear
probe.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, or Redl et al. 2023**
(in Related Work, no matched comparison). We claim no superiority to any;
ProtTucker is the closest published analogue to our protocol.

### 8. Statistical evidence

The SCOPe intervals are in item 7. On Table 2 you are right: no intervals, one
run at seed 42 per cell, ±0.005 called a tie — a
convention narrower than item 6's checkpoint spread. A 5-seed probe sweep gives
median SD 0.0000, which only shows a 3-NN probe is deterministic on fixed
embeddings; training-seed variance is unmeasured, one training run per model
existing. Relatedly, over the 20 tasks whose metric is defined for all arms
(three multiclass-AUC tasks excluded, remote homology among them — its accuracies
are in item 1), V1 beats ESM-2 35M 11/3/6 under 3-NN but **4 win / 4 tie / 12
lose under a linear probe**; V2 is 10/3/7 and 2/7/11. Hence the withdrawal.

### Errors in our own submission

Two bear on weakness 1. The PPI decontamination description does not
match the code (`easy-search` at 40% identity removing hit query
IDs, not `easy-linclust` at 50% with cluster removal), and the remote-homology
split is not hierarchy-disjoint — TAPE's three holdouts pooled (718 + 1,254 +
1,272 = 3,244) — so its pooled macro AUC is not comparable to published
per-holdout accuracies. Also: SCOPe is the family field over 2,207 sequences, not
superfamily over 100,000, and everything here is `--eval_split test`, so not
cell-comparable to the submitted tables.

If that bounds weakness 1 to the residual we state — untested fold-level overlap
on SCOPe-40, no PPI measurement — we ask you to reconsider; if one remains
decisive, say so and we will answer it.
<!-- END Yi1G -->
