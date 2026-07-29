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

<!-- character count of the pasted body below: 9982 (limit 10,000) -->
<!-- BEGIN HNXd -->
Retrained on decontaminated corpora, ProtSent-V2 (35M, frozen)
**ties HMMER at SCOPe-40 family Recall@1** — paired bootstrap -0.0124, 95% CI
[-0.0372, +0.0124] — and leads it at ranking depth (+0.1412 Recall@10, +0.1708
MAP, both intervals excluding zero). The clustering statistics you asked for now
exist: family ARI over the SCOPe-40 space goes 0.0544 (stock ESM-2 35M) to
0.5071 (V2). Two of your five questions answer against us: under a
trained linear probe V2 loses to stock ESM-2 35M on **11 of 20** comparable
tasks, and the label-scarcity claim is withdrawn.

Withdrawn: general-purpose superiority, label scarcity, Table 5, all 150M
results including the abstract's +105% and +19.9% — **there is no 150M model on
decontaminated data**. V1 = the submitted 35M, V2 = the retrain on filtered
corpora. Every number below is `--eval_split test`; the submitted tables use the suite's *default* split, so the two are not
comparable cell by cell and we never mix them.

### 1. Retrieval, clustering, and how the space reorganises (Q1)

Geometry: 2,207 SCOPe-40 domains, their 917 true families, cosine distance on
frozen embeddings.

| | ESM-2 35M | ProtSent-V2 |
|---|---|---|
| silhouette (family) | -0.1426 | **+0.0529** |
| NMI | 0.8225 | **0.9174** |
| ARI | 0.0544 | **0.5071** |
| Spearman(distance, shared hierarchy) | -0.1055 | **-0.2097** |

Silhouette crossing zero means families stop overlapping more than they separate.
ARI rising from 0.05 to 0.51 means clustering the space at the true family count
recovers half the partition instead of almost none. The last row is your
distance-versus-property example, with SCOPe hierarchy as the property: mean
distance falls monotonically with shared levels for ProtSent and does **not** for
ESM-2, which puts domains sharing two levels further apart (0.146) than domains
sharing one (0.140).

Retrieval: same gallery, self excluded, **no-hit scored as failure**; MAP is
average precision over the full ranking, so items an alignment never returns
contribute zero. Only 1,693 of 2,207 queries have a non-self same-family
neighbour and every row below is those; over all 2,207 each method scales by
0.7671 (R@1 0.3829 ESM-2 35M / 0.4490 V1 / 0.5256 V2). Erratum: Table 3 is a
**family**-level score over 2,207 domains, not superfamily over 100,000 — that
figure was the evaluator's `max_samples` cap.

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| HMMER (phmmer, `-E 10`) | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 35M | 0.6852 | **0.9220** | **0.9634** | **0.6459** |

**Alignment is the better top-1 method** and V1 loses top-1 to both tools. Our
advantage is ranking *depth*, and part of that margin is list coverage, not
ranking: 691 of all 2,207 queries return no phmmer hit at `-E 10` and score 0 at
every K, and both tools flatten from
R@10 to R@30 (+0.0171, +0.0165) where the embeddings do not (+0.0726 ESM-2,
+0.0414 V2). Until both are re-run with that threshold lifted, **the depth
margin is an upper bound**.

V2 was retrained on corpora filtered at 40% identity / 80% coverage against the
remote-homology and PPI test sets (SCOPe-40 was not a filter target), using the
configuration the paper's own ablations favour — proportional sampling, no
synthetic hard negatives. Because the recipe changed with the corpus, **V2 - V1
is not a decontamination ablation**; no unfiltered retrain at that recipe
exists. Its SCOPe R@10 gain over ESM-2 is flat in query identity to our corpus
(+0.1524 / +0.1810 / +0.1565 in bins [0.2,0.4), [0.4,0.7), [0.7,1.0]), which
bounds identity-level memorisation only; fold-level overlap is untested.

### 2. Linear probe on the frozen backbone (Q2)

23 tasks, test split, frozen mean-pooled embeddings, vs stock ESM-2 35M. The
probe is scikit-learn at defaults, untuned — logistic regression for
classification, RidgeCV for regression; 3-NN is `n_neighbors=3`, uniform.

| probe (20 comparable tasks) | V1 | V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

The ±0.005 tie band is absolute across accuracy,
Spearman, F1 and AUC; no setting of it makes the linear record a win. These 23
tasks are the single-sequence tasks with a paired alignment row (SCOPe-40
retrieval, EC and GO-MF in; `ppi_bernett` out as pair-input), so this is **not**
the paper's 23-task list, which contains PPI. Three tasks
(`antibiotic_resistance`, `remote_homology`, `temperature_stability`) fall
outside the 20, because one-vs-rest AUC is undefined when the test split holds a
class absent from train — which drops our best task from the tally.

Remote homology (457 classes, test split), the task the corpus was filtered
against, ESM-2 / V1 / V2: 3-NN accuracy 0.5835 / 0.6587 / **0.6668**; 3-NN
macro-F1 0.3173 / 0.3687 / **0.4108**; linear accuracy 0.6868 / 0.6899 /
**0.7016**; linear macro-F1 0.4414 / **0.4281** / 0.4527. V1 is *below* the
untuned backbone on linear macro-F1 and its +0.0031 linear accuracy is a tie by
our own band; only V2 improves on both metrics under both probes. Two errata: that
split is TAPE's three holdouts pooled, not hierarchy-disjoint as the paper says;
and the paper's "+40.5%" is a default-split relative macro-F1 change (.223 →
.313), not these numbers.

**Your level-gap hypothesis: tested, and the probe is not the cause.** Biomap
`stability_prediction` labels are continuous floats (-1.68 to 2.15) scored by
Spearman, so the percentage you set against the 69.08% linear / 77.69% LoRA
accuracies is a correlation ×100, not an accuracy; **we withdraw that
comparison**. And there 3-NN scores *higher* than our linear probe for every arm
(ESM-2 35M Spearman 0.6435 vs 0.4395, test split), so your proposed probe change
moves it the wrong way.

### 3. Bootstrap confidence intervals (Q3)

Retrieval metrics are per-query means, so resampling the 1,693 queries needs no
refitting. 10,000 **paired** resamples, reproducing the table above to 0.0012:

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - HMMER | **-0.0124 [-0.0372, +0.0124]** | +0.1412 [+0.1205, +0.1618] | +0.1708 [+0.1511, +0.1905] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

**We do not claim to beat alignment at top-1**: V2 ties HMMER, and its +0.0289
edge over MMseqs2 clears zero by 0.0035 across three uncorrected comparisons, so
we do not lean on that either. V1 - HMMER -0.1110 [-0.1388, -0.0827] and V1 -
MMseqs2 -0.0697 [-0.0975, -0.0413] are outright losses.

**No intervals exist for the 23-task table; your objection stands.** The cost is
bounded: it was the evidence for the general-purpose claim, which we withdraw.
Every surviving claim carries uncertainty — SCOPe-40 bootstrap, few-shot 5-draw
SD — bar the single-run linear remote-homology accuracies, bounded only by the
checkpoint spread below.


### 4-5. Few-shot: absolute scores and seed variability (Q5, Q4)

**Table 5 is replaced, not defended**: relative cells over near-zero Spearman
baselines were uninterpretable (its -126.9% cell is enzyme catalytic efficiency,
a sign flip of 0.269x the baseline) and the estimator was not constant, since
the code sets `n_neighbors = max(1, min(3, train_size))`. Re-run: absolute scores, 5
training-subset draws, full test split. **Remote homology, accuracy, mean ± SD, 3-NN / linear:**

| N | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|---|
| 50 | 0.061±0.010 / 0.121±0.003 | 0.055±0.008 / 0.159±0.004 | 0.045±0.009 / 0.145±0.005 |
| 250 | 0.148±0.002 / 0.310±0.007 | **0.223±0.011 / 0.394±0.012** | 0.200±0.010 / 0.368±0.013 |
| 1000 | 0.185±0.002 / 0.288±0.014 | **0.318±0.015 / 0.377±0.008** | 0.289±0.016 / 0.355±0.009 |

Your suspicion about the +244.5% cell was right: it is remote homology at N=100
under 3-NN, and re-run with a fixed estimator, 5 draws and the full test split
it is 0.1155 (ESM-2) → 0.1349 (V1) accuracy — **+0.0194 absolute, +16.8%
relative**, protocol rather than arithmetic. Metal-ion binding at N=1000 under the
linear head (accuracy) runs the other way: ESM-2 0.666±0.001 beats V1
0.637±0.004 and V2 0.595±0.001.

(i) **Your proposed framing is not supported**: a trained linear head beats 3-NN
in almost every model/task/N cell including N=50, so "linear degrades while k-NN
stays competitive" is false here and we withdraw that mechanism. What survives
is task-bound: under the linear head V1 leads ESM-2 by +0.084 at N=250 and
+0.089 at N=1000 on remote homology, by nothing on metal-ion binding. (ii)
At small N the seed spread is as large as the effect (Biomap stability at N=100:
±0.20 on means of 0.28-0.40), your concern confirmed. (iii) **We ran no
fine-tuning sweep**, and doubt it rescues us when the weaker of your two
baselines already beats us 11 of 20.

Full-data evaluation is near-deterministic: 5 seeds x 8 tasks x 3 arms under
3-NN gives median SD 0.0000 across 24 rows, since fixed embeddings and a fixed
test split make that probe deterministic; only `thermostability` subsamples (SD
0.013-0.017). Two caveats: one *training* run per model exists, so training-seed
variance is unmeasured, and checkpoint 4,000 differs from the final V2 by
0.005-0.008 on every structural metric. So no sub-0.01 structural delta is
resolved, including the V1→V2 remote-homology gap, which is **+0.0079** as
5-seed means (0.6589 → 0.6668; the 0.6587 quoted above is the single-run value).

What we defend is a 35M retrieval-and-remote-homology result on the test split.
That is the measured record and we ask you to reconsider on it; if one item
remains decisive, name it and we will answer it in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 9957 (limit 10,000) -->
<!-- BEGIN jVGf -->
**Structural supervision is the single largest contributor, as you suspected:**
removing AFDB drops the mean relative gain from +6.7% to +3.2%, improved tasks
from 16/23 to 13/23, and remote homology from +40.5% to +15.3% (submitted 35M
model V1, single run, suite default split, mean relative change over ESM-2 35M —
the paper's ablation table). Under a trained linear probe ProtSent also loses to
its own untuned backbone, stock ESM-2 35M, on 12 of 20 comparable tasks (V1) and
11 of 20 (V2), so the general-purpose framing is withdrawn. What survives is a
measured position on the generality-accuracy curve and evidence that the
non-structural sources each move a different task family.

**Naming.** V1 = the submitted 35M model; V2 = a 35M retrained during the
rebuttal on decontaminated corpora. **There is no 150M model on the
decontaminated corpus** and we do not defend the submitted 150M results. Test-
split numbers are `--eval_split test`, not cell-comparable to the submitted
tables, whose relative percentages are default-split single runs on V1; we do not
mix the two.

### 1. Where ProtSent sits on the generality-accuracy trade-off (Q3 / W2)

We ran MMseqs2 as a full alternative pipeline over all 23 tasks, scored through
the same code as the embedding path (per-class max bitscore for classification,
1-NN by bitscore for regression, no-hit queries counted as failures), at
`-s 7.5 -e 10 --max-seqs 300`, not the much weaker default — at `-s 5.7` it gives SCOPe-40 R@1 0.3847 against 0.5029 for
`-s 7.5` on the same basis (all 2,207 queries), so any MMseqs2 number needs its
sensitivity stated. We also ran HMMER (phmmer, `-E 10`, top 300 hits/query, same
scoring code), which beats MMseqs2 on 12 of the 22 tasks both completed, so we
quote the better of the two throughout.

**Alignment beats the best of all three embedding arms — V2 included — outright
on 3 of 23 tasks under a 3-NN probe** (EC classification, GO molecular function,
beta-lactamase fitness) **and 6 under a linear probe** (those plus enzyme
catalytic efficiency, optimal pH, stability). EC classification F1-macro 0.7229 (HMMER, hit coverage 0.945) against the best
embedding arm's 0.598 (ESM-2 35M); GO-MF 0.6047 (HMMER, coverage 0.901) vs 0.459 (ESM-2 35M);
beta-lactamase Spearman 0.8026 (MMseqs2). Where annotation transfers by homology,
alignment is better.

The other end of the curve is coverage. Alignment returns **nothing at all** for
a large share of queries, and that share grows where the task is hard: on remote
homology HMMER has no hit for 47.6% of test queries and MMseqs2 for 11.1%; on
SCOPe-40 retrieval 31.3% and 11.8%; on `rhla_enzyme_mutations` (6-residue
mutation-site strings) coverage is 0.004 and 0.000 — both fail completely. And
MMseqs2 is *below chance* on DeepSol solubility (AUC 0.4185; HMMER 0.4150). An
embedding always returns a ranked list; its metric is never a fallback's.

SCOPe-40, family level, 2,207-domain gallery, self excluded, no-hit = failure;
the 1,693 rows are the queries having a non-self same-family neighbour. MAP =
mean average precision over the returned ranking.

| method (1,693 eligible) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 `-s 7.5` | 0.6556 | 0.7401 | 0.7566 | 0.4098 |
| HMMER phmmer | **0.6970** | 0.7809 | 0.7980 | 0.4747 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M | 0.5854 | 0.8512 | 0.9256 | 0.5509 |
| ProtSent-V2 35M | 0.6852 | **0.9220** | **0.9634** | **0.6459** |

V2 is the decontaminated retrain, but it *also* drops synthetic hard negatives,
switches to proportional sampling and uses a true 1,024-example contrastive batch
where V1's loss call saw 64, so **V2 - V1 is not a decontamination ablation** —
no unfiltered retrain at the V2 recipe exists.

Paired bootstrap, 10,000 resamples, same queries throughout. **We do not beat
alignment at top-1**: V2 - HMMER at R@1 is -0.0124 [-0.0372, +0.0124],
unresolved; V1 - HMMER is -0.1110 [-0.1388, -0.0827], a clear loss; V2 leads
MMseqs2 by only +0.0289 [+0.0035, +0.0544] while V1 loses to it by -0.0697
[-0.0975, -0.0413]. At depth the embedding leads both: V2 - HMMER +0.1412
[+0.1205, +0.1618] at R@10 and +0.1708 [+0.1511, +0.1905] at MAP; V2 - MMseqs2
+0.1819 and +0.2356, both intervals excluding zero.

**Two limits on that depth result, both ours.** Part is candidate coverage
rather than ranking quality: 691 of all 2,207 queries return no phmmer hit at
`-E 10` and are scored 0 at every K, and alignment lists are truncated at 300
hits, so MAP carries the same caveat. We did not re-run either tool with the
reporting threshold lifted, so those margins are upper bounds. And SCOPe-40 was
never a decontamination target: V2's per-query R@10 gain over ESM-2 35M does not
grow with a query's maximum identity to our corpus (+0.1524 at [0.2,0.4), n=164;
+0.1810 at [0.4,0.7), n=315; +0.1565 at [0.7,1.0], n=1,214), and among the 404
queries where ESM-2 fails outright identity does not predict the gain (Spearman
+0.038, p=0.45). That rules out identity-level memorization but not fold-level
overlap: our supervision is Foldseek/Pfam co-membership, and a training pair
sharing a test domain's fold at 15% identity survives a 40%-identity filter. The
fold-exclusion control is the experiment we did not run.

The trade-off: alignment wins single-best-hit and homology-transferable
annotation; the embedding wins ranking depth and is the only one of the two that
yields a fixed-width vector where there is no alignment signal at all. That last
property belongs to embeddings generally, not to ProtSent — under a linear probe
stock ESM-2 35M is the better embedding on 12 of 20 comparable tasks (V1 4/4/12,
median -0.0139; V2 2/7/11, median -0.0107; tie band ±0.005, test split, one seed
each). That record is why the general-purpose claim goes.

### 2. Is this more than structural-information injection? (Q1 / W1)

Partly not, and the ablation above says how much. The remainder is not structure,
and each source leaves a fingerprint on a different task family (all V1, default
split, single run, relative change): without Pfam the model still improves 15/23
at mean +4.6%; removing STRING moves PPI from +5.3% to -0.5% while leaving most
other tasks intact; removing the DMS objective reduces fitness gains
(fluorescence +15.6% → +10.4%). A pure structure-distillation model has no PPI
dial and no fitness dial. **The limit:** those are single-run relative-percent
numbers on the pre-decontamination model, the convention we withdraw for sub-1%
cells elsewhere. They support the *direction* of
source-specific effects and nothing finer, and cannot show the sources fail to
interfere; the linear-probe record above is consistent with the shared space
costing something.

**We did not run the joint no-AFDB/no-Pfam ablation you asked for**, and the two
single ablations do not substitute for it. What we can put against it is the
non-structural half measured alone, decontaminated, in absolutes: GB1
variant effect (Spearman, 3-NN, **test split**, mean of 5 draws, SD 0.0000) is
0.6582 (ESM-2 35M) / 0.7108 (V1) / 0.7806 (V2) — a different split from the
submitted table's GB1 cell, which is why the two disagree in sign and why we do
not mix them. Fitness order and protein interaction are relations no structure
teacher supplies, so your ablation would settle how much of the *benchmark
aggregate* survives without structure, not whether the non-structural sources do
anything. If it is your decisive item, say so and we will report it in discussion.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton, ProTrek (W1)

ESM-S, S-PLM, ISM and Magneton inject structure into a sequence model by
distilling a structure encoder or structural tokens: one relation type, one
teacher. ProtSent supervises a heterogeneous relation *graph* over sequences —
Pfam family co-membership, Foldseek cluster co-membership, STRING interaction,
DMS fitness order — with no structure encoder anywhere, at training or inference.
The claim is that relation *type* is a design axis, each source moving a
different task family as measured in item 2 — not that this beats structure
distillation: we have **no matched runs against any of the four and claim no
superiority to them**. ProTrek is the trimodal (sequence/structure/text)
retrieval-optimized point on the same curve; we did not run it and expect it to
beat a 35M sequence-only encoder at retrieval.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek**, nor Redl et al. 2023.
ProtTucker is the closest published analogue to our protocol — contrastive
fine-tuning of frozen embeddings for remote homology — and the absence we would
most want to fix. **We also did not apply ProtSent to SaProt or ProSST**: the
blocker is data, not code — both consume residue-level structure tokens, and we
have no predicted structures for the Pfam and STRING sequences, the corpus
majority.

### 4. The CoSENT objective on DMS data (Q4)

Our text is wrong: the paper says the DMS loss "operates on single proteins
rather than pairs." The released code writes `(sentence_0, sentence_1, score)`
rows — `sentence_0` the wild-type, `sentence_1` the mutant, `score` the
within-assay normalized fitness rescaled to [0,1]. CoSENT is ordinal over those
pairs: within a batch, if
pair p scores above pair q, the loss pushes cos(WT_p, mut_p) above cos(WT_q,
mut_q). There is no absolute cosine target and no term pulling high-fitness
mutants toward a common point, so it does not flatten an assay. The real
limitation: the pairing is **wild-type-anchored**, so mutant-mutant geometry is
constrained only indirectly.

The "?" at line 21 is a broken citation key, not a missing reference: Heinzinger
et al. 2022 (ProtTucker) and Redl et al. 2023 are both in Related Work.

That is the measured trade-off, the source fingerprints and the positioning; we
ask you to reconsider on them. If the missing no-AFDB/no-Pfam ablation or the
missing ProtTucker run is decisive, say which.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9932 (limit 10,000) -->
<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: all three
corpora re-filtered at 40% identity / 80% coverage, retrained from scratch,
benchmarks re-run, **0 flagged sequences surviving**. HMMER was run too and
costs us a claim: ProtSent-V2 (35M) minus HMMER at SCOPe-40 family Recall@1 is
-0.0124, 95% CI [-0.0372, +0.0124] — a tie, not a win. V1 = the submitted 35M
model, V2 = the retrained one; every number below is `--eval_split test`, not
cell-comparable to the submitted tables.

### 1. Leakage

MMseqs2 `easy-search`, corpus as query, test set as target, 40% identity / 80%
coverage; any corpus sequence with a hit is dropped. Pfam 28,530,684 →
27,929,772 rows and AFDB 135,404,259 → 126,301,607 against `remote_homology`
test (3,244 seqs); STRING 76,070,154 → 71,891,417 pairs against `ppi_bernett`
test (3,022 seqs). These are training-parquet row counts, which do not
correspond one-to-one to Table 1's counts; we reconcile them in the camera-ready. **Those two were the only filter targets**; SCOPe-40 was not. Verified
by semi-join on the files training actually opened: **0 flagged sequences
survived**. STRING was then subsampled to a seeded 15,000,000-pair training
file (compute budget, not a leakage control), so the arithmetic closes:
27,929,772 + 126,301,607 + 15,000,000 = 169,231,379, the `total=` in the
training log. Re-searching the filtered corpora against the same test
sets returns **0 hits**.

**What the retrain shows, and what it does not.** Remote homology (457 classes,
test split), ESM-2 35M / V1 / V2: 3-NN accuracy 0.5835 / 0.6587 / 0.6668,
linear accuracy 0.6868 / 0.6899 / 0.7016, linear macro-F1 0.4414 / **0.4281** /
0.4527 — V1 below the untuned backbone there, only V2 improving on both metrics
under both probes. But the recipe changed with the corpus (item 3), and with no
unfiltered retrain at that configuration **nothing is attributable to
decontamination in either direction**; the V1→V2 3-NN gap
(+0.0079, 5-seed means) is inside item 6's checkpoint spread. Only the weak
claim holds: a decontaminated corpus still trains a model at least as good as
the submitted one.

**PPI: the filter you asked for was run** — 40% identity / 80% coverage,
stricter than the 50% you named, removing 4,178,737 STRING pairs, 0 surviving.
What does not exist is the downstream number: `ppi_bernett` is a pair-input
task, not in the 23-task sweep, so the paper's +5.3% AUC stays a pre-
decontamination V1 figure. That is the open half of weakness 1.

**SCOPe-40 cannot be decontaminated at corpus level**: no train/test split
(leave-one-out over 2,207 domains), median maximum identity to our corpus
**0.908**, none below 20%, so filtering against it removes essentially every
structured domain. Memorization predicts that queries with a closer pretraining
neighbour gain more. V2's per-query R@10 gain over ESM-2 35M is flat in
identity: +0.1524 at [0.2,0.4) (n=164), +0.1810 at [0.4,0.7) (n=315), +0.1565
at [0.7,1.0] (n=1,214); the [0,0.2) bin is empty. Those 164 queries are
themselves a decontaminated evaluation at the 40% threshold used on every
corpus, and V2 - ESM-2 35M MAP is *larger* there (+0.2859) than over all 1,693
(+0.2232 [+0.2082, +0.2383], paired bootstrap). Identity-to-gain Spearman is
-0.116, negative rather than positive, and among the **404 queries the untuned
backbone fails completely** identity does not predict the gain (+0.038,
p=0.45).

**What this cannot rule out, said before you have to.** Supervision is Foldseek
cluster and Pfam family co-membership, so a training pair sharing a test
domain's *fold* at 15% identity survives a 40%-identity filter. Identity
stratification cannot see fold-level *label* overlap, and a flat slope is what
that leakage would also produce, so this control has little power against your
actual objection. SCOPe carries our largest margin and we cannot say it is free
of that. The right experiment — re-scoring after excluding queries whose fold
is in a training cluster — we did not run; it is cheap, and if you name it
decisive we will report it in discussion.

### 2. DMS objective

Implemented as you describe; our text ("operates on single proteins rather than
pairs") is wrong. Rows are (wild-type, mutant, within-assay normalized fitness
in [0,1]) and CoSENT ranks pairs within a batch: if mutant a beats mutant b the
loss pushes cos(WT, a) above cos(WT, b) — no absolute target, nothing
collapsing high-fitness variants. The limitation: pairing is WT-anchored, so
mutant-mutant distances are only indirectly constrained.

### 3. MNRL batch semantics and Eq. 1

Correct — a real error. The submitted 1,024 is an **optimizer** batch from
gradient accumulation (35M: 64 per device × 16 steps; 150M: 16 × 64), and
accumulation does not share in-batch negatives, so each MNRL call saw **64**
examples at 35M and **16** at 150M — the likeliest explanation for the 150M
results, which we no longer defend. **No 150M model on the decontaminated
corpus exists.** V2 uses a true 1,024-example batch per device. In Eq. 1 the
numerator should use the positive paired with anchor i, the denominator all N
positives.

### 4-5. Pair-level tasks and k-NN regression

PPI: partners are embedded independently and the two vectors concatenated
before the probe. Peptide-HLA is **not** two-input here — the dataset supplies
one `seq` field, a pipe-joined `HLA_pseudoseq|peptide` string. k-NN regression
is **uniform**: `KNeighborsRegressor(n_neighbors=3)`, an unweighted mean over 3
neighbours; at small N the code sets `n_neighbors = max(1, min(3,
train_size))`, so the smallest few-shot cells use a different estimator —
which, with relative changes over near-zero baselines, is why Table 5 is now
absolute means with seed SDs.

### 6. Ablations

Removing synthetic hard negatives improves 20/23 tasks at mean +7.9% against
16/23 and +6.7% for the submitted configuration, and proportional sampling
+7.0% vs round-robin's +6.7% (relative gain over ESM-2 35M, V1, default split,
one run each). V2 uses neither default. **The consequence:** those ablations
were scored on these same benchmarks, so V2's configuration was chosen with
benchmark results in view — a selection channel the filter does not touch.
SCOPe-40 entered that aggregate as one task of 23, not as the criterion, and
selection over a few configurations does not manufacture a +0.1855 R@1 gap; but
V2's 23-task numbers are not a clean held-out measurement and we do not call
them one. Checkpoint 4,000 differs from the final one by 0.005-0.008 on every
structural metric, so no sub-0.01 delta is resolved. On your single-space
question the ablations only show each source moving its own task family (STRING
removal takes PPI +5.3% → -0.5%); the cost of sharing one space across four
relation types is item 8's linear-probe record, and that record is the price.

### 7. Baselines

**HMMER (phmmer, `-E 10`, top 300 hits/query, no-hit = failure) was run**, same
gallery and scoring code as MMseqs2. SCOPe-40, **family** level, 2,207 domains;
the 1,693 eligible queries have a non-self same-family neighbour; MAP is mean
average precision over the returned list.

R@1 / R@10 / MAP: HMMER **0.6970** / 0.7809 / 0.4747; MMseqs2 `-s 7.5` 0.6556 /
0.7401 / 0.4098; ESM-2 35M 0.4991 / 0.7614 / 0.4210; ProtSent-V1 0.5854 /
0.8512 / 0.5509; ProtSent-V2 0.6852 / **0.9220** / **0.6459**.

Paired bootstrap, 10,000 resamples, same queries throughout. **V2 - HMMER at
R@1 is -0.0124 [-0.0372, +0.0124], unresolved; V1 - HMMER -0.1110 [-0.1388,
-0.0827]**, an outright loss; V2 - MMseqs2 +0.0289 [+0.0035, +0.0544]. At depth
V2 - HMMER is +0.1412 [+0.1205, +0.1618] (R@10), +0.1708 [+0.1511, +0.1905]
(MAP).

**Alignment remains the better top-1 method.** Against our own depth claim: 691
of all 2,207 queries return no phmmer hit and score 0 at every K (not broken
out over the 1,693 scored), and alignment lists are truncated at 300 hits, so
part of the depth and MAP margins is coverage, not ranking; until both tools
are re-run without that threshold they are upper bounds.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, or Redl et al.
2023.** The two we ran take sequence only, so they bound what a sequence-only
encoder must beat, and HMMER is the one that costs us a claim; Foldseek and
ProTrek use structure at query time. We claim no superiority to any of the
five. ProtTucker is the real gap: contrastive fine-tuning of frozen embeddings
for remote homology is our protocol.

### 8. Statistical evidence

On Table 2 you are right (SCOPe intervals are in item 7): no intervals, one run
at seed 42 per cell, ±0.005 called a tie — narrower than item 6's checkpoint
spread. A 5-seed sweep gives median SD 0.0000, which only shows a 3-NN probe is
deterministic on fixed embeddings; training-seed variance is unmeasured, one
training run per model existing. Over the 20 tasks whose metric is defined for
all arms, V1 beats ESM-2 35M 11/3/6 under 3-NN but **4 win / 4 tie / 12 lose
under a linear probe**; V2 is 10/3/7 and 2/7/11. Hence the withdrawal, and the
envelope: this helps where the label *is* a homology or structure relation, not
where it is a property with no homology signal, and between the two we resolve
nothing below 0.01.

**Errors in our submission.** The PPI decontamination description does not
match the code (`easy-search` at 40% identity removing hit query IDs, not
`easy-linclust` at 50% with cluster removal), and the remote-homology split is
not hierarchy-disjoint but TAPE's three holdouts pooled — so our stated defence
against weakness 1 was itself wrong. SCOPe is the family field over 2,207
sequences, not superfamily over 100,000.

Weakness 1 now reduces to the residual we state — untested fold-level overlap
on SCOPe-40, no post-filter PPI number — and we ask you to reconsider on that.
If one remains decisive, say which: the fold-exclusion re-scoring and the
threshold-free alignment re-run are both runnable in discussion.
<!-- END Yi1G -->
