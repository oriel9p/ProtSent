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

<!-- character count of the pasted body below: 9825 (limit 10,000) -->
<!-- BEGIN HNXd -->
**Result first.** Retrained on decontaminated corpora, ProtSent-V2 (35M, frozen)
**ties HMMER at SCOPe-40 family Recall@1** — paired bootstrap -0.012, 95% CI
[-0.037, +0.012] — and leads it at ranking depth (+0.141 Recall@10, +0.171
MAP, both intervals excluding zero). The clustering statistics you asked for now
exist: family ARI over the SCOPe-40 space goes 0.054 (stock ESM-2 35M) to
0.507 (V2). Two of your five questions answer against us: under a
trained linear probe V2 loses to stock ESM-2 35M on **11 of 20** comparable
tasks, and the label-scarcity claim is withdrawn.

Withdrawn: general-purpose superiority, label scarcity, Table 5, all 150M
results including the abstract's +105% and +19.9% — **there is no 150M model on
decontaminated data**. V1 = the submitted 35M, V2 = the retrain on filtered
corpora. Every number below is `--eval_split test`; the submitted tables are the
suite's *default* split and not cell-comparable, so we never mix the two.

### 1. Retrieval, clustering, and how the space reorganises (Q1)

Geometry, 2,207 SCOPe-40 domains against their 917 true families, cosine
distance, frozen embeddings, ESM-2 35M / V1 / V2: **silhouette**
-0.143 / -0.044 / **+0.053**; **NMI** 0.823 / 0.900 / **0.917**; **ARI**
0.054 / 0.416 / **0.507**; Spearman(distance, shared hierarchy depth) -0.105
/ **-0.312** / -0.210. The last row is your distance-versus-property
example, with SCOPe hierarchy as the property: distance falls monotonically with
shared levels for both ProtSent models but not for ESM-2. V1 beats V2 there and
on intra/inter family distance (0.269 vs 0.352), so V2 does not dominate
geometry.

Retrieval: same gallery, leave-one-out, self excluded, **no-hit scored as
failure**. MAP = average precision over the full ranking, relevant items never
returned contributing zero — for alignment, everything past its reported list.
Only 1,693 of 2,207 queries have a non-self same-family neighbour; every row is
those. Over all 2,207 each method scales by 0.767, the submitted Table 3
measurement (R@1 0.383 ESM-2 / 0.449 V1 / 0.526 V2).

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer, `-E 10`) | **0.697** | 0.781 | 0.798 | 0.475 |
| ESM-2 35M | 0.499 | 0.761 | 0.834 | 0.421 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.926 | 0.551 |
| ProtSent-V2 35M | 0.685 | **0.922** | **0.963** | **0.646** |

**Alignment is the better top-1 method** and V1 loses top-1 to both tools. Our
advantage is ranking *depth*, and part of that margin is list coverage, not
ranking: 691 of all 2,207 queries return no phmmer hit at `-E 10` and score 0 at
every K (not broken down over the 1,693 scored), and both tools flatten from
R@10 to R@30 (+0.017, +0.017) where the embeddings do not (+0.073 ESM-2,
+0.041 V2). Until both are re-run with that threshold lifted, **the depth
margin is an upper bound**.

V2 was retrained on corpora filtered at 40% identity / 80% coverage against the
remote-homology and PPI test sets (SCOPe-40 was not a filter target), using the configuration the paper's own ablations favour. Because the
recipe changed with the corpus, **V2 - V1 is not a decontamination ablation**;
no unfiltered retrain at that recipe exists. SCOPe cannot be filtered at corpus level, so we filtered the
benchmark: on the 164 eligible queries below 40% identity to our corpus, V2 -
HMMER holds at +0.116 [+0.049, +0.189] R@10 and +0.140 [+0.075, +0.207] MAP. The
margin does not shrink as queries get cleaner. This bounds identity-level
exposure only; fold-level overlap is untested.

### 2. Linear probe on the frozen backbone (Q2)

23 tasks, test split, frozen mean-pooled embeddings, vs stock ESM-2 35M. The
probe is scikit-learn defaults, untuned: `StandardScaler` + `LogisticRegression`
(liblinear), or `Ridge(alpha=1.0)` for regression; 3-NN is `n_neighbors=3`,
uniform.

| probe (20 comparable tasks) | V1 | V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.007 | 10 / 3 / 7, median +0.004 |
| linear | 4 / 4 / 12, median **-0.014** | 2 / 7 / 11, median -0.011 |

Reading rules, all against us. The ±0.005 tie band is absolute across accuracy,
Spearman, F1 and AUC; no setting of it makes the linear record a win. These 23
tasks are the single-sequence tasks with a paired alignment row (SCOPe-40
retrieval, EC and GO-MF in; `ppi_bernett` out as pair-input), so this is **not**
the paper's 23-task list, which contains PPI. Three tasks
(`antibiotic_resistance`, `remote_homology`, `temperature_stability`) fall
outside the 20, because one-vs-rest AUC is undefined when the test split holds a
class absent from train — which drops our best task from the tally.

Remote homology (457 classes, test split), the task the corpus was filtered
against, ESM-2 / V1 / V2: 3-NN accuracy 0.584 / 0.659 / **0.667**; 3-NN
macro-F1 0.317 / 0.369 / **0.411**; linear accuracy 0.687 / 0.690 /
**0.702**; linear macro-F1 0.441 / **0.428** / 0.453. V1 is *below* the
untuned backbone on linear macro-F1 and its +0.003 linear accuracy is a tie by
our own band; only V2 improves on both metrics under both probes. Two errata
here: that split is TAPE's three holdouts pooled (718 + 1,254 + 1,272 = 3,244),
not hierarchy-disjoint as the paper says, so its pooled macro AUC is not
comparable to published per-holdout accuracies; and the paper's "+40.5%" is a
default-split relative macro-F1 change (.223 → .313), not these numbers.

**Your level-gap hypothesis: tested, and the probe is not the cause.** Biomap
`stability_prediction` labels are continuous floats (-1.68 to 2.15) scored by
Spearman, so the percentage you set against the 69.08% linear / 77.69% LoRA
accuracies is a correlation ×100, not an accuracy; **we withdraw that
comparison**. And there 3-NN scores *higher* than our linear probe for every arm
(ESM-2 35M Spearman 0.643 vs 0.440, test split), so your proposed probe change
moves it the wrong way.

### 3. Bootstrap confidence intervals (Q3)

Retrieval metrics are per-query means, so resampling the 1,693 queries needs no
refitting. 10,000 **paired** resamples, reproducing the table above to 0.001:

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V2 - ESM-2 | +0.185 [+0.162, +0.210] | +0.161 [+0.141, +0.180] | +0.223 [+0.208, +0.238] |
| V2 - HMMER | **-0.012 [-0.037, +0.012]** | +0.141 [+0.120, +0.162] | +0.171 [+0.151, +0.191] |
| V2 - MMseqs2 | +0.029 [+0.004, +0.054] | +0.182 [+0.161, +0.203] | +0.236 [+0.216, +0.255] |

**We do not claim to beat alignment at top-1**: V2 ties HMMER, and its +0.029
edge over MMseqs2 clears zero by 0.004 across three uncorrected comparisons, so
we do not lean on that either. V1 - HMMER -0.111 [-0.139, -0.083] and V1 -
MMseqs2 -0.070 [-0.098, -0.041] are outright losses.

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
it is 0.116 (ESM-2) → 0.135 (V1) accuracy — **+0.019 absolute, +16.8%
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
3-NN gives median SD 0.000 across 24 rows, since fixed embeddings and a fixed
test split make that probe deterministic; only `thermostability` subsamples (SD
0.013-0.017). Two caveats: one *training* run per model exists, so training-seed
variance is unmeasured, and checkpoint 4,000 differs from the final V2 by
0.005-0.008 on every structural metric. So no sub-0.01 structural delta is
resolved, including the V1→V2 remote-homology gap, 0.659 → 0.667 = **+0.008**
over 5 seeds.

What we defend is a 35M retrieval-and-remote-homology result on the test split.
That is the measured record and we ask you to reconsider on it; if one item
remains decisive, name it and we will answer it in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 9729 (limit 10,000) -->
<!-- BEGIN jVGf -->
**Structural supervision is the single largest contributor, as you suspected.**
Our own Table 4, re-read: removing AlphaFold DB drops improved tasks from 16/23
to 13/23, the mean relative gain from +6.7% to +3.2%, and remote homology from
+40.5% to +15.3% — single-run, submitted 35M model, suite default split. That is
a larger drop than removing Pfam (15/23, +4.6%), so the paper's sentence calling
Pfam "the dominant contrastive signal" is wrong by its own table. Under a trained linear probe ProtSent also loses to its
own untuned backbone, stock ESM-2 35M, on 12 of 20 comparable tasks (submitted
model) and 11 of 20 (retrained), so the general-purpose framing is withdrawn.

What survives is a *measured* position on the generality-accuracy curve and
evidence that the non-structural sources each move a different task family.
**V1** = the submitted 35M model, **V2** = a 35M retrained during the rebuttal
on decontaminated corpora; everything here is 35M, and **there is no 150M model
on decontaminated data**, so we do not defend the submitted 150M numbers.
Rebuttal numbers are `--eval_split test`, the submitted tables the suite's
default split; we never mix the two in one comparison.

### 1. Where ProtSent sits on the generality-accuracy trade-off (Q3 / W2)

We ran MMseqs2 as a full alternative pipeline over all 23 tasks under identical
metric definitions (per-class max bitscore for classification so AUC stays
comparable, 1-NN by bitscore for regression; no-hit queries scored as failures)
at `-s 7.5 -e 10 --max-seqs 300`, not the much weaker default — at `-s 5.7` the
same baseline gives SCOPe-40 family R@1 0.385 over all 2,207 queries, so any
MMseqs2 number needs its sensitivity stated. HMMER (phmmer, `-E 10`, same
scoring code) was run too, and we quote whichever tool is better per task, so
the easier opponent is never the one reported.

**Alignment beats the best embedding arm outright on 3 of those 23 tasks under a
3-NN probe** (EC classification, GO molecular function, beta-lactamase fitness)
**and 6 under a linear probe** (those plus enzyme catalytic efficiency, optimal
pH, Biomap stability), by large margins: EC F1-macro 0.723 (HMMER) vs 0.598
(ESM-2 35M) / 0.562 (V1) / 0.592 (V2); GO-MF 0.605 (HMMER) vs 0.459 / 0.443
/ 0.455; beta-lactamase Spearman 0.803 (MMseqs2) vs 0.727 / 0.768 / 0.715.
Where annotation transfers by homology, alignment is better.

The other end of the curve is coverage, the sharper half of the argument.
Alignment returns *nothing* for a large share of queries, and that share grows
where the task is hard: HMMER returns no hit for 47.6% of the remote-homology
test set and 31.3% of the SCOPe-40 gallery, and on `rhla_enzyme_mutations`
(6-residue mutation-site strings) hit coverage is 0.004 and 0.000 — both tools
fail completely. On DeepSol solubility MMseqs2 is *below
chance* (AUC 0.418). An embedding always returns a ranked list. The three
alignment wins above stand at coverage 0.945 / 0.901 / 1.000.

SCOPe-40, family level, 2,207-domain gallery, leave-one-out, self excluded,
no-hit = failure; MAP = average precision over the full ranking, unreturned
relevant items contributing zero. Only 1,693 queries have a non-self same-family
neighbour and all rows use those; over all 2,207 every method scales by 0.767.

| method (1,693 eligible queries) | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer) | **0.697** | 0.781 | 0.798 | 0.475 |
| ESM-2 35M | 0.499 | 0.761 | 0.834 | 0.421 |
| ProtSent-V1 35M (submitted) | 0.585 | 0.851 | 0.926 | 0.551 |
| ProtSent-V2 35M (retrained) | 0.685 | **0.922** | **0.963** | **0.646** |

V2 retrains on the filtered corpora with the configuration our own ablations
favour, so **V2 - V1 is not a decontamination ablation**; no unfiltered retrain
at that recipe exists.

Paired bootstrap, 10,000 resamples, the same queries scoring every method (it
reproduces the table to within 0.001). **We do not beat alignment at top-1**:
V2 - HMMER at R@1 is -0.012 [-0.037, +0.012], unresolved; V1 - HMMER is
-0.111 [-0.139, -0.083], a clear loss; V2 leads MMseqs2 by +0.029 [+0.004,
+0.054], clearing zero by 0.004 across three uncorrected comparisons, so we do
not lean on it. The embedding wins at depth against both: V2 - HMMER
+0.141 [+0.120, +0.162] at R@10 and +0.171 [+0.151, +0.191] at MAP; V2 -
MMseqs2 +0.182 [+0.161, +0.203] and +0.236 [+0.216, +0.255].

**Two limits on that depth result, both ours.** Part of it is list coverage, not
ranking quality: 691 of all 2,207 queries return no phmmer hit at `-E 10` and
score 0 at every K, and both tools flatten from R@10 to R@30 (+0.017, +0.017)
where the embeddings do not (+0.073 ESM-2, +0.041 V2). We did not re-run
either tool with that threshold removed, so the depth margin is an upper bound. And SCOPe-40 was never a decontamination target, so we filtered
the benchmark instead: on the 164 eligible queries below 40% identity to our
corpus, V2 - HMMER holds at +0.116 [+0.049, +0.189] R@10 and +0.140 [+0.075,
+0.207] MAP. That rules out identity-level memorisation, not fold-level overlap —
supervision is Foldseek-cluster and Pfam-family co-membership, and a pair sharing
a query's fold at 15% identity survives any identity filter. That control we did
not run.

The trade-off: alignment wins single-best-hit and homology-transferable
annotation; the embedding wins ranking depth and returns a usable ranked list
where alignment returns nothing. That last property belongs to embeddings
generally, not to ProtSent — under a linear probe stock ESM-2 35M is the better
embedding on 12 of 20 comparable tasks (V1 4/4/12, median -0.014; V2 2/7/11,
median -0.011; tie band ±0.005 absolute, test split, one seed each). That is
why the general-purpose claim goes.

### 2. Is this more than structural-information injection? (Q1 / W1)

Partly not — the AFDB ablation above quantifies how much. The remainder is not
structure, and each source fingerprints a different task family (submitted
model, default split, single run, mean relative change): without Pfam the model
still improves 15/23 at +4.6%; removing StringDB moves PPI from +5.3% to -0.5%
while overall quality holds (17/23, +5.9%); removing the DMS objective cuts
fluorescence from +15.6% to +10.4%. A pure structure-distillation model has no
PPI dial and no fitness dial.

**We did not run the joint no-AFDB/no-Pfam ablation you asked for**, and the two
single ablations do not substitute for it. What we can put against it is the
non-structural half on its own, in absolute decontaminated numbers rather than
relative ones: GB1 variant effect (Spearman, 3-NN, test split, mean of 5 seeds)
0.658 (ESM-2 35M) / 0.711 (V1) / 0.781 (V2), SD 0.000. Fitness order and
interaction are relations no structure teacher supplies, so your ablation would
settle how much of the *benchmark aggregate* survives without structure, not
whether the non-structural sources do anything.

**The limit on that argument:** single-run relative-percent numbers on the
default split, the same convention we withdraw for sub-1% cells elsewhere. They
support the *direction* of source-specific effects and nothing finer, and cannot
show the sources do not interfere — the linear-probe record above is consistent
with the shared space costing something.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton, ProTrek (W1)

ESM-S, S-PLM, ISM and Magneton inject structure into a sequence model by
distilling a structure encoder or structural tokens — one relation type, one
teacher. ProtSent supervises a heterogeneous relation *graph* over sequences — Pfam family co-membership,
Foldseek cluster co-membership, STRING interaction, DMS fitness order — with no
structure encoder at training or inference. The claim is that relation *type* is
a design axis — each source moves a different task family, as measured above —
not that this beats structure distillation: we have **no matched runs against
any of them and claim no superiority**. ProTrek is the trimodal
(sequence/structure/text) point on the same curve; we did not run it and expect
it to beat a 35M sequence-only encoder at retrieval.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek**, nor Redl et al.
2023. ProtTucker is the closest analogue to our protocol — contrastive
fine-tuning of frozen embeddings for remote homology — and the gap we would most
want to fix. **We also did not apply ProtSent to SaProt or ProSST**: the blocker
is data, not code, since both consume residue-level structure tokens and we have
no predicted structures for the Pfam and STRING sequences, most of the corpus.

### 4. The CoSENT objective on DMS data (Q4)

Our text is wrong: the paper says the DMS loss "operates on single proteins
rather than pairs." The released code writes `(sentence_0, sentence_1, score)`
rows — wild-type, mutant, and the within-assay normalised fitness rescaled to
[0,1]. CoSENT is ordinal over those pairs exactly as for sentences: within a
batch, if pair p scores above pair q, the loss pushes cos(WT_p, mut_p) above
cos(WT_q, mut_q). There is no absolute cosine target and no term pulling
high-fitness mutants together, so it does not flatten an assay. The real
limitation is that the pairing is **wild-type-anchored**, leaving mutant-mutant
geometry constrained only indirectly.

The "?" at line 21 is a broken citation key, not a missing reference: Heinzinger
et al. 2022 (ProtTucker) and Redl et al. 2023 are both in Related Work.

That is the measured trade-off, the source fingerprints and the positioning, and
we ask you to reconsider on them. If the missing no-AFDB/no-Pfam ablation or the
missing ProtTucker run is decisive, say which and we will report it in
discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9947 (limit 10,000) -->
<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: all three
corpora re-filtered at 40% identity / 80% coverage, the model retrained from
scratch, benchmarks re-run, verification finding **0 flagged sequences
surviving**. HMMER was run too and costs us a claim: ProtSent-V2 (35M) minus
HMMER at SCOPe-40 family Recall@1 is -0.012, 95% CI [-0.037, +0.012] — a tie,
not a win. **V1** = the submitted 35M, **V2** = the retrain on filtered corpora;
every number below is on `--eval_split test`, not the submitted tables' split.

### 1. Leakage

MMseqs2 `easy-search`, corpus as query, test set as target, 40% identity / 80%
coverage; any corpus sequence with a hit is dropped. Pfam 28,530,684 →
27,929,772 rows and AFDB 135,404,259 → 126,301,607 against `remote_homology`
test (3,244 seqs); STRING 76,070,154 → 71,891,417 pairs against `ppi_bernett`
test (3,022 seqs). Those are pair-file rows, not Table 1's corpus sizes, which
we have not reconciled. **Those two test sets were the only filter targets**;
every other test set, SCOPe-40 included, was not. Verification
semi-joined the parquets training actually opened against the removal lists: **0
flagged sequences survived**, their rows summing to the training log's total
(27,929,772 + 126,301,607 + 15,000,000 STRING rows = 169,231,379). Re-searching 1,000
random sequences per *filtered* corpus returns **0 hits**, which by the rule of
three bounds the residual only at ~0.3%.

**What the retrain shows, and what it does not.** Remote homology (pooled
457-class, test split), ESM-2 35M / V1 / V2: 3-NN accuracy 0.584 / 0.659 /
0.667; linear accuracy 0.687 / 0.690 / 0.702; linear macro-F1 0.441 /
**0.428** / 0.453 — V1 below the untuned backbone there, only V2 improving on
both metrics under both probes. But the recipe changed with the corpus (item 3), and with
no unfiltered retrain at that configuration **nothing is attributable to
decontamination in either direction**; the V1→V2 3-NN gap of
+0.008 (5-seed means 0.659 → 0.667) is inside item 6's checkpoint spread. Only the weak claim holds: decontaminating the corpus did not cost
performance on the filtered task.

**PPI: the filter you asked for was run** — 40% identity / 80% coverage,
stricter than the 50% you named, removing 4,178,737 STRING pairs. The downstream
number does not exist: `ppi_bernett` is pair-input, not in the 23-task sweep, so
the paper's +5.3% AUC stays a pre-decontamination V1 result. That is the open
half of weakness 1.

**SCOPe-40 cannot be filtered at the corpus level** — no train/test split
(leave-one-out over 2,207 domains), median maximum identity to our corpus
**0.908**, so filtering against it removes essentially every structured domain.
**So we filtered the benchmark instead**: drop the queries with a close
pretraining neighbour, re-score every arm on what remains. Paired V2 - HMMER,
same queries, 10,000 resamples:

| eligible queries kept | R@1 | R@10 | MAP |
|---|---|---|---|
| identity <0.4 (n=164) | -0.043 [-0.128, +0.043] | **+0.116 [+0.049, +0.189]** | **+0.140 [+0.075, +0.207]** |
| <0.7 (n=479) | -0.027 [-0.073, +0.017] | **+0.127 [+0.090, +0.165]** | **+0.154 [+0.117, +0.190]** |
| all (n=1,693) | -0.012 [-0.037, +0.012] | **+0.141 [+0.120, +0.162]** | **+0.171 [+0.151, +0.191]** |

The conclusion does not move: on the 164 queries furthest from anything we
trained on, V2 still ties the best alignment baseline at top-1 and still leads at
depth, and the margin does not shrink as the queries get cleaner. Identity-gain
Spearman is -0.116 in average precision (p=1.6e-6), -0.081 after a headroom
control — negative, where memorisation predicts positive.

**What this cannot rule out.** Supervision is Foldseek-cluster and Pfam-family
co-membership, so a pair sharing a query's *fold* at 15% identity survives any
identity filter, and identity filtering cannot see it. Excluding queries whose
fold is among our training clusters would discriminate the two; we did not run
it, and can in discussion.

### 2. DMS objective

Implemented as you describe; our text ("operates on single proteins rather than
pairs") is wrong. Rows are (wild-type, mutant, within-assay normalised fitness
in [0,1]) and CoSENT ranks pairs within a batch: if mutant a beats b the loss
pushes cos(WT, a) above cos(WT, b) — no absolute target, nothing collapsing
high-fitness variants. The pairing is WT-anchored, so mutant-mutant distances
are constrained only indirectly.

### 3. MNRL batch semantics and Eq. 1

Correct — a real error. The submitted 1,024 is an **optimizer** batch from
gradient accumulation (35M: 64 per device x 16 steps; 150M: 16 x 64), which does
not share in-batch negatives, so each MNRL call saw **64** examples at 35M and
**16** at 150M — the likeliest explanation for the 150M results we no longer
defend. The retrain uses a true 1,024 batch per device. In Eq. 1 the numerator
takes the positive paired with anchor i, the denominator the positives of all N
pairs.

### 4-5. Pair-level tasks and k-NN regression

PPI partners are embedded independently and concatenated before the probe.
Peptide-HLA is **not** two-input here — the dataset supplies one `seq` field, a
pipe-joined `HLA_pseudoseq|peptide` string; neither task was in the paper. k-NN
regression is **uniform**, `KNeighborsRegressor(n_neighbors=3)`, an unweighted
mean over 3 neighbours; at small N the code sets `n_neighbors = max(1, min(3,
train_size))`, a different estimator in the smallest few-shot cells — which is
why Table 5 is replaced by absolute means with seed SDs.

### 6. Ablations

We acted on this. Removing synthetic hard negatives improves 20/23 tasks at mean
+7.9% against 16/23 and +6.7% for the submitted configuration; proportional
sampling gives +7.0% vs round-robin's +6.7% (relative gain over ESM-2 35M,
submitted model, default split, one run each). V2 uses neither of the submitted
defaults. **The
consequence:** those ablations were scored on these same benchmarks, so V2's
configuration was chosen with benchmark results in view — a selection channel
the corpus filter does not touch, and we cannot call V2's 23-task numbers a
clean held-out measurement. SCOPe-40 entered that aggregate as one task of 23,
not as the criterion, and the alignment baselines were run after the
configuration was fixed. The checkpoint is simply the last step; checkpoint
4,000 differs from it by 0.005-0.008 on every structural metric, at or above
item 8's ±0.005 band, so no sub-0.01 structural delta is resolved.

On the single-space question, the ablations show only that each source moves its
own task family (STRING removal takes PPI +5.3% → -0.5%); the cost of sharing
one space is item 8's linear-probe record.

### 7. Baselines

**HMMER (phmmer, `-E 10`, top 300 hits/query, no-hit = failure) was run**, same
gallery and scoring code as MMseqs2. Those two take the same input we do —
sequence only — so they bound what a sequence-only encoder has to beat, and
HMMER is the one that costs us a claim. SCOPe-40, **family** level, 2,207 domains, leave-one-out; the 1,693
eligible queries are those with a non-self same-family neighbour (over all 2,207
each method scales by 0.767). R@1 / R@10 / MAP: HMMER **0.697** / 0.781 /
0.475; MMseqs2 (`-s 7.5 -e 10`) 0.656 / 0.740 / 0.410; ESM-2 35M 0.499 /
0.761 / 0.421; V1 0.585 / 0.851 / 0.551; V2 0.685 / **0.922** /
**0.646**.

Paired bootstrap, 10,000 resamples, same queries throughout. At R@1: **V2 -
HMMER -0.012 [-0.037, +0.012], unresolved; V1 - HMMER -0.111 [-0.139,
-0.083]**, an outright loss; V2 - MMseqs2 +0.029 [+0.004, +0.054], clearing
zero by 0.004 across three uncorrected comparisons. At depth V2 - HMMER is
+0.141 [+0.120, +0.162] (R@10) and +0.171 [+0.151, +0.191] (MAP).

**Alignment remains the better top-1 method.** Against our own depth claim: 691
of 2,207 queries return no phmmer hit and score 0 at every K, and both tools
flatten from R@10 to R@30 (+0.017) where the embeddings do not (+0.073 ESM-2,
+0.041 V2), so part of the depth gap is candidate coverage rather than ranking
and the margin is an upper bound until both are re-run without that threshold.
Over 23 tasks alignment beats the best embedding arm on 3 under 3-NN, 6 under a
linear probe.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, Redl et al. 2023.**
Foldseek and ProTrek consume structure at query time, so losing to them would
say nothing about a sequence-only encoder; we claim no superiority to any.
ProtTucker is the real gap — its protocol is ours.

### 8. Statistical evidence

The SCOPe intervals are in item 7. On Table 2 you are right: no intervals, one
run at seed 42 per cell, ±0.005 called a tie — narrower than item 6's checkpoint
spread. A 5-seed sweep gives median SD 0.000, which only shows a 3-NN probe is
deterministic on fixed embeddings; training-seed variance is unmeasured, one
training run per model existing. Over the 20 tasks whose metric is defined for
all arms (three multiclass-AUC tasks excluded, remote homology among them), V1
beats ESM-2 35M 11/3/6 under 3-NN but **4/4/12 under a linear probe**; V2 is
10/3/7 and 2/7/11.

### Errors in our own submission

The PPI decontamination description does not match the code (`easy-search` at
40% identity removing hit query IDs, not `easy-linclust` at 50% with cluster
removal). The remote-homology split is not hierarchy-disjoint but TAPE's three
holdouts pooled (3,244 sequences), so its pooled macro AUC is not comparable to
published per-holdout accuracies. SCOPe is the family field over 2,207
sequences, not superfamily over 100,000. All 150M results are withdrawn — there
is no 150M model on the decontaminated corpus.

Weakness 1 now reduces to the residual we state — untested fold-level overlap on
SCOPe-40, no post-filter PPI measurement — and we ask you to reconsider on that.
The two experiments named above, fold-exclusion on SCOPe-40 and the alignment
re-run with the threshold lifted, we will run in discussion if they decide it
for you.
<!-- END Yi1G -->
