# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — postable

Naming used throughout: **V1** = the submitted 35M model. **V2** = the 35M model
retrained during the rebuttal on the decontaminated corpora.

Paste unit = everything strictly between `<!-- BEGIN X -->` and `<!-- END X -->`.

---

## Response to Reviewer HNXd

<!-- character count of the pasted body below: 9931 (limit 10,000) -->
<!-- BEGIN HNXd -->
Clustering the frozen SCOPe-40 space at the true family count recovers the family
partition at adjusted Rand index 0.507 for ProtSent-V2 35M, against 0.054 for
stock ESM-2 35M. That is Q1, and our strongest result.

**The model you reviewed, on its own.** ProtSent-V1 35M reorganises the space
relative to its own backbone and nothing further: +0.087 Recall@1, +0.090
Recall@10, +0.129 MAP over ESM-2 35M on SCOPe-40, and a loss on all three to a
maximally-sensitive phmmer. Under a linear probe it loses 12 of 20 tasks to that
backbone. That is what the submission supports.

Naming, once. V2 is a 35M retrained on all three corpora re-filtered at 40%
identity / 80% coverage, with the configuration our ablations favour —
proportional sampling, no synthetic hard negatives — on 7 GPUs rather than 1. Not
a controlled decontamination ablation. Denominators: 23 tasks in the sweep, 20
scorable under both probes for every arm. Counts use a +/-0.005 tie band, against a
0.005 to 0.008 checkpoint spread in our own run, so we defend no delta below
0.010.

### Q1. Retrieval and clustering

2,207 SCOPe-40 domains, 917 families, frozen mean-pooled embeddings, cosine
distance, average-linkage agglomerative clustering cut at k = 917, the true family
count handed to both arms; deterministic. ARI leads because it is
chance-corrected and NMI is not, which is why NMI is high for both arms.

| SCOPe-40, family labels | ESM-2 35M | ProtSent-V2 35M |
|---|---|---|
| adjusted Rand index | 0.054 | 0.507 |
| normalised mutual information | 0.823 | 0.917 |
| silhouette | -0.143 | +0.053 |

Silhouette crosses zero: in ESM-2 the average domain sits closer to another
family than to its own, in V2 it does not. +0.053 is small.

Global organisation, as mean pairwise cosine distance by shared SCOPe hierarchy
levels (0 = different class, 4 = same family):

| shared levels | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| ESM-2 35M | 0.156 | 0.140 | 0.146 | 0.123 | 0.064 |
| ProtSent-V2 35M | 0.865 | 0.821 | 0.780 | 0.571 | 0.299 |

ESM-2's ordering breaks at two shared levels (0.146 above 0.140); V2's is
monotone. Spearman is -0.106 for ESM-2 35M and -0.210 for V2: improved, still
weak. ARI and silhouette are scale-invariant, so this is not the expansion.

Direct retrieval, SCOPe-40 family level, leave-one-out over the 2,207-domain
gallery, self excluded, no-hit as failure, over the 1,693 queries with a non-self
same-family neighbour; the other 514 are singletons, unachievable for anyone.

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| ESM-2 35M | 0.499 | 0.761 | 0.421 |
| HMMER phmmer, defaults (`-E 10`) | 0.697 | 0.781 | 0.475 |
| HMMER phmmer, heuristic filters off | 0.753 | 0.898 | 0.607 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.646 |

We do not beat alignment. The filters-off row is the strongest baseline we could
build; V2 is behind it at top-1, ahead at depth. It replaces Table 3, whose ESM-2
35M entries were 0.385 and 0.588.

### Q2. Linear classifier baselines

Under a final-layer linear head both ProtSent models lose to their own untuned
backbone, and the general-purpose claim goes with it. Per task against stock
ESM-2 35M, over the 20 tasks scorable for every arm:

| vs stock ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| 3-NN probe | 11 win / 3 tie / 6 lose | 10 / 3 / 7 |
| linear probe | 4 / 4 / 12 | 2 / 7 / 11 |

We report that split as a finding, not only a loss. A linear head asks whether
property information is present and decodable given labels; a 3-NN probe asks
whether it is already local in the geometry, with no labels and no fitting.
Compressing four relation types into one metric buys the second and costs the
first.

Part of the loss is our protocol. Both probes pool the final layer, and a layer
sweep (8,000 train / 3,000 test) shows that is the worst layer for both models on
remote-homology accuracy: ESM-2 35M 0.670 at layer 6 against 0.637 final, V2 0.703
at layer 8 against 0.680 final, V2 ahead at every layer from 6 up. A confound in
our probe protocol, on two tasks.

Three of the 23 are outside the 20 — remote homology, antibiotic resistance,
temperature stability — because their metric is one-vs-rest AUC and their test
split holds classes absent from train, so no arm scores. Remote homology has an
accuracy, test split, n=3,244, 457 classes:

| | 3-NN acc | 3-NN macro-F1 | linear acc | linear macro-F1 |
|---|---|---|---|---|
| ESM-2 35M | 0.584 | 0.317 | 0.687 | 0.441 |
| ProtSent-V1 | 0.659 | 0.369 | 0.690 | 0.428 |
| ProtSent-V2 | 0.667 | 0.411 | 0.702 | 0.453 |

V1 is below the untuned backbone on linear macro-F1; only V2 improves on both
metrics under both probes.

We did not fine-tune ESM-2 end to end. One fine-tune on remote homology is under a
day on our GPUs: an omission, not a cost argument.

Stability corrects in your favour. Those labels are continuous floats from -1.680
to 2.150 and our metric is Spearman, so 0.588 is a correlation, not an accuracy
comparable to 69.08% linear or 77.69% LoRA; we withdraw the comparison. Your
proposed mechanism fails too: on Stability 3-NN beats the linear probe for every
arm, ESM-2 35M 0.568 to 0.440.

### Q3. Bootstrap confidence intervals

We did not bootstrap the 23-task table. We withdraw its aggregate rather than
caveat it, so no surviving claim rests on it, and we defend no per-task delta
below 0.010, V1 to V2 on remote homology at +0.008 included. The work is open:
those metrics are functions of per-example predictions on disk.

Retrieval has intervals, since each metric is a mean over per-query values. 10,000
paired resamples, same 1,693 queries for every method:

| paired difference, SCOPe-40 | R@1 | R@10 | MAP |
|---|---|---|---|
| V2 - ESM-2 35M | +0.185 [+0.162, +0.210] | +0.161 [+0.141, +0.180] | +0.223 [+0.208, +0.238] |
| V2 - HMMER, filters off | -0.068 [-0.092, -0.044] | +0.024 [+0.009, +0.040] | +0.039 [+0.021, +0.056] |
| V1 - ESM-2 35M | +0.087 [+0.061, +0.112] | +0.090 [+0.069, +0.111] | +0.129 [+0.113, +0.145] |

V1 minus the filters-off phmmer is -0.167 [-0.193, -0.139] at R@1. Against phmmer
at defaults V2 is -0.012 [-0.037, +0.012] at R@1, +0.141 [+0.120, +0.162] at
R@10.

Two residuals these do not cover. One training run per model exists, so
training-seed variance is unmeasured. And V2's configuration was chosen from two
binary ablation results scored on the 23-task suite, so its numbers there are not
held out. SCOPe-40 was not an ablation criterion, and both alignment baselines ran
after the configuration was fixed.

### Q4 and Q5. Few-shot variability and absolute scores

Every +/- below is subset-draw spread: the training subset is redrawn 5 times per
cell, the test split stays full size, the probe is deterministic. It is not the
full-data seed sweep, whose median SD over 5 seeds, 8 tasks and 3 arms is 0.000 —
a deterministic probe, not evidence of stability.

Remote-homology accuracy, full test split, mean +/- SD over 5 draws (N=100 is in
the paragraph below):

| N | ESM-2 3-NN | ESM-2 linear | V1 3-NN | V1 linear | V2 3-NN | V2 linear |
|---|---|---|---|---|---|---|
| 50 | 0.061 +/- 0.010 | 0.121 +/- 0.003 | 0.055 +/- 0.008 | 0.159 +/- 0.004 | 0.045 +/- 0.009 | 0.145 +/- 0.005 |
| 1000 | 0.185 +/- 0.002 | 0.288 +/- 0.014 | 0.318 +/- 0.015 | 0.377 +/- 0.008 | 0.289 +/- 0.016 | 0.355 +/- 0.009 |

Table 5 is withdrawn as a table. Absolute scores replace it where we re-ran the
cells: remote homology above, and Biomap stability at N=100, ESM-2 35M 3-NN
Spearman 0.260 +/- 0.129 against V2 0.401 +/- 0.202 — spread as large as the
effect. The rest of Table 5 is not re-run and those percentages are withdrawn.

Your suspicion about the +244.5% cell was right. It is remote homology at N=100
under a 3-NN probe: 0.1155 for ESM-2 35M, 0.1349 for V1 and 0.1248 for V2, an
absolute gain of 0.019 for V1. A percentage against a baseline of 0.116 was never interpretable. Nor
was the probe constant across N — the few-shot code sets `n_neighbors = max(1,
min(3, train_size))`, so Table 5 cells below N=3 are not 3-NN. Every N above is
unaffected.

Your proposed framing, linear pipelines degrading under label scarcity while k-NN
stays competitive, is not supported: a linear head beats 3-NN in almost every
cell, including N=50. We tested it, it failed, and the label-scarcity claim goes
with it. Two results cut against us: V1 beats V2 in every few-shot
remote-homology cell, and at N=1000 under a linear head on metal-ion binding
ESM-2 35M leads at 0.666 +/- 0.001 against V2's 0.595 +/- 0.001.

### What the paper claims now, and three errors no reviewer raised

New abstract sentence, verbatim: "Contrastive fine-tuning on four relation types
reorganises a 35M pLM embedding space so that SCOPe family membership is
recoverable without labels, and yields a frozen embedding that leads a
maximally-sensitive phmmer at ranking depth on SCOPe-40 retrieval (+0.024
Recall@10, +0.039 MAP) while trailing it by 0.068 at Recall@1." Contributions:
family membership recoverable from geometry alone; the generality-accuracy
trade-off measured against both alignment tools; geometric reorganisation and
linear decodability coming apart. Removed: the two-scale bullet, every 150M
result, label scarcity, the Table 2 aggregate, Tables 3 and 5, Stability.

Three errors produced Table 3: the code scored the family field while the caption
said superfamily, the caption reported the evaluator's 100,000 sampling cap as the
gallery size, and the 514 impossible queries were scored as failures. Table 4
contradicts our own sentence calling Pfam the dominant contrastive signal:
removing AlphaFold DB costs more. And the remote-homology split is TAPE's three
holdouts pooled, so it is not hierarchy-disjoint.

Q1 and Q4 are answered in full, Q5 for the cells we re-ran, Q3 on retrieval and by
withdrawal on the 23-task table. Q2 has both probes plus a layer sweep and no
end-to-end fine-tune. If that resolves your concerns, we ask you to raise your
score.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 9917 (limit 10,000) -->
<!-- BEGIN jVGf -->
We measured the generality-accuracy trade-off, and alignment wins more of it than
the paper implies. A maximally-sensitive HMMER beats ProtSent-V2 35M at SCOPe-40
family Recall@1 by 0.068, and alignment beats our best embedding arm outright on 3
of 23 tasks under a 3-NN probe and 6 under a linear probe. What the embedding wins
is ranking depth: +0.024 Recall@10 and +0.039 MAP over that same HMMER, at one
forward pass per sequence.

**The model you reviewed, on its own.** ProtSent-V1 35M loses to a properly tuned
phmmer at Recall@1, Recall@10 and MAP, and loses 12 of 20 comparable tasks to its
own untuned backbone under a linear probe. It reorganises the space relative to
that backbone — +0.087 Recall@1, +0.090 Recall@10, +0.129 MAP on SCOPe-40, all
paired intervals excluding zero — and that is all.

Naming, once. V2 is a 35M retrained on corpora re-filtered at 40% identity / 80%
coverage, with the configuration our ablations favour — proportional sampling, no
synthetic hard negatives — on 7 GPUs rather than 1. Not a controlled
decontamination ablation. Denominators: 23 tasks in the sweep, 20 scorable under
both probes for every arm, 22 completed by HMMER.

You were right about structure, and our own Table 4 says so more plainly than our
text. Removing AlphaFold DB drops improved tasks from 16 of 23 to 13 of 23, the
mean relative gain from +6.7% to +3.2%, and remote homology from +40.5% to +15.3%.
Removing Pfam is milder: 15 of 23 and +4.6%. Our sentence calling Pfam "the
dominant contrastive signal" is contradicted by our own table.

### 1. Results without both AFDB and Pfam

**Not run.** It is the only experiment that shows whether the four sources
interfere. Neither claim we still make depends on it.

What we can put against it is what each non-structural source does alone. Removing
StringDB takes PPI prediction from +5.3% to -0.5% while overall quality holds at
17 of 23 tasks and +5.9%; dropping the DMS objective cuts fluorescence from +15.6%
to +10.4%. On GB1 variant effect, 3-NN Spearman on the test split over 5 seeds is
0.658 for ESM-2 35M, 0.711 for V1 and 0.781 for V2, SD 0.000 throughout. A pure
structure-distillation model has no interaction dial and no fitness dial.

Those ablation figures are single-run relative changes on the default split. They
support direction on large cells — 20 of 23 tasks improved against 16 of 23, which
is what fixed V2's configuration — and not sub-1% cells or percentages against
near-zero baselines, which is why Table 5's +244.5% cell goes.

### 2. ProtSent on SaProt or ProSST

Not run. The blocker is data, not code: both consume residue-level structure
tokens, and we have no predicted structures for the Pfam and STRING sequences,
most of the corpus.

### 3. Comparison to specialized methods

Both sequence-only alignment baselines ran over the whole benchmark, through the
same gallery and scoring code as the embeddings: MMseqs2 at `-s 7.5 -e 10
--max-seqs 300`, HMMER phmmer at `-E 10`, no-hit queries as failures. HMMER beats
MMseqs2 on 12 of the 22 tasks both finished, so neither is the weak baseline, and
we quote whichever is better per task.

Where alignment wins, it wins decisively:

| task | metric | best alignment | ESM-2 35M | V1 | V2 |
|---|---|---|---|---|---|
| EC classification | F1-macro | 0.723 (HMMER, cov 0.945) | 0.598 | 0.562 | 0.592 |
| GO molecular function | F1-macro | 0.605 (HMMER, cov 0.901) | 0.459 | 0.443 | 0.455 |
| beta-lactamase | Spearman | 0.803 (MMseqs2, cov 1.000) | 0.727 | 0.768 | 0.715 |

Where the embedding wins is ranking depth. SCOPe-40 family retrieval,
leave-one-out over a 2,207-domain gallery, self excluded, no-hit as failure, over
the 1,693 queries with a non-self same-family neighbour; the other 514 are
singletons, unachievable for any method.

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| ESM-2 35M | 0.499 | 0.761 | 0.421 |
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.410 |
| HMMER phmmer, defaults (`-E 10`) | 0.697 | 0.781 | 0.475 |
| HMMER phmmer, heuristic filters off | 0.753 | 0.898 | 0.607 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.646 |

Paired bootstrap, 10,000 resamples, same queries scoring every method. V2 minus
the filters-off phmmer is -0.068 [-0.092, -0.044] at R@1, +0.024 [+0.009, +0.040]
at R@10, +0.039 [+0.021, +0.056] at MAP. Against phmmer at defaults those become
-0.012 [-0.037, +0.012], +0.141 [+0.120, +0.162] and +0.171 [+0.151, +0.191]. V1
minus the filters-off baseline is -0.167 [-0.193, -0.139] at R@1. This replaces
Table 3, whose ESM-2 35M entries of 0.385 and 0.588 came from scoring the family
field under a superfamily caption.

Coverage is the other half of the trade-off, and it is settings-dependent. At
phmmer defaults HMMER returns no hit for 47.6% of remote-homology test queries and
31.3% of SCOPe-40 queries; MMseqs2 at `-s 7.5` for 11.1% and 11.8%.
Those no-hits come from the MSV/Viterbi/Forward prefilters, not the E-value —
raising `-E` alone changes nothing. With the filters off, phmmer returns a ranked
list for every SCOPe-40 query in 83 seconds for the full all-vs-all. We re-ran
only SCOPe-40 that way, so read those coverage figures as a property of the
default configuration, not of alignment search. What is not settings-dependent: an
embedding is one forward pass and an indexable metric.

SCOPe-40 cannot be filtered at corpus level, so we filtered the benchmark. On the
164 eligible queries below 40% identity to our corpus, HMMER at defaults reaches
R@10 0.774 against V2's 0.890, a paired +0.116 [+0.049, +0.189]; MAP is +0.140
[+0.075, +0.207]. The depth margin does not shrink as queries get cleaner. Top-1
is -0.043, unresolved at n=164. That subset was not re-scored against the
filters-off baseline, so it is a clean-query win over the weaker configuration.

**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.** We claim no
superiority to any, and do not quote their published numbers either:
our SCOPe-40 evaluation is leave-one-out over a 2,207-domain gallery at the family
level with singletons scored as failures, and no published table uses that
protocol, so an unmatched number would be the misleading comparison you are asking
us to avoid. ProtTucker is the gap that matters — closest protocol, public
checkpoint — and without it our R@10 of 0.922 has no reference frame beyond
alignment and the backbone. Foldseek and ProTrek consume structure at query time,
so losing to them would say little about a sequence-only encoder. Against Redl et
al. 2023, which shares our objective family, our difference is four relation types
rather than one, and with no matched run we claim no originality beyond that.

### 4. The CoSENT objective on DMS data

Our text is wrong: the paper says the DMS loss "operates on single proteins rather
than pairs". The released code writes `(sentence_0, sentence_1, score)` rows —
wild-type, mutant, within-assay normalised fitness rescaled to [0,1] — and CoSENT
is ordinal over exactly those pairs: within a batch, if pair p scores above pair
q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no absolute
cosine target and no term pulling high-fitness mutants together, so it does not
flatten an assay. Pairing is wild-type-anchored, which constrains mutant-mutant
geometry only indirectly.

### Positioning against ESM-S, S-PLM, ISM and Magneton

Those four inject structure by distilling a structure encoder or structural
tokens: one relation type, one teacher. ProtSent supervises a heterogeneous
relation graph over sequences — Pfam family, Foldseek cluster, STRING interaction,
DMS fitness order — with no structure encoder at training or inference. Relation
type is the design axis we claim, since each source moves a different task.
Not that it beats structure distillation: no matched runs exist, and we do not
quote their numbers against ours for the protocol reason above.

The measured result behind your weakness 1, rather than an argument from taxonomy:
under a final-layer linear probe V2 loses 11 of 20 comparable tasks to its own
untuned backbone (2 win, 7 tie) while recovering the SCOPe-40 family partition at
adjusted Rand index 0.507 against 0.054 for stock ESM-2 35M. Compressing four
relation types into one metric buys geometry and costs linear decodability. A
layer sweep shows part of that is our protocol: on remote homology V2 leads at
every pooled layer from 6 up, and the final layer the benchmark pools is the worst
for both models. That trade-off is a property of contrastive geometry rather than
of structure injection, and none of the four papers reports it.

One caveat on every V2 number: its configuration was chosen from two binary
ablation results scored on the 23-task suite, so those numbers are not held out.
SCOPe-40 was not an ablation criterion, and both alignment baselines ran after the
configuration was fixed.

Two errors no reviewer raised, disclosed because they touch claims you are
judging. RhlA at +77.2%, the paper's second-largest 35M gain, is measured on
6-residue mutation-site strings, not proteins. And the remote-homology test split
is TAPE's three holdouts pooled, so it is not hierarchy-disjoint and its macro AUC
is not comparable to published per-holdout accuracies.

The "?" at line 21 is a broken citation key; Heinzinger et al. 2022 and Redl et
al. 2023 are both in Related Work.

The paper now claims three things: family membership is recoverable from geometry
alone (ARI 0.054 to 0.507); a frozen embedding leads a maximally-sensitive phmmer
at ranking depth on SCOPe-40 and trails it by 0.068 at Recall@1; and geometric
reorganisation and linear decodability come apart. Removed: the two-scale bullet,
every 150M result, the Table 2 aggregate, Tables 3 and 5, label scarcity. That is
where ProtSent sits on the curve, measured rather than asserted, and we ask you to raise
your score on that basis. Open: the joint ablation and matched runs against the
four models above; neither is load-bearing for those claims.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9958 (limit 10,000) -->
<!-- BEGIN Yi1G -->
All three pretraining corpora were re-filtered at 40% identity / 80% coverage and
the model retrained from scratch, leaving an estimated residual of 0.7% of the
corpus that we quantify below. Running the baselines you named then cost us a
claim: against a maximally-sensitive HMMER, ProtSent-V2 35M is behind at SCOPe-40
family Recall@1 by -0.068 [-0.092, -0.044] and ahead at depth by +0.024 [+0.009,
+0.040] Recall@10.

**The model you reviewed, on its own.** ProtSent-V1 35M loses to that HMMER at
Recall@1, Recall@10 and MAP, and loses 12 of 20 comparable tasks to its own
untuned backbone under a linear probe. What it does is reorganise the space
relative to that backbone: +0.087 Recall@1, +0.090 Recall@10, +0.129 MAP on
SCOPe-40, paired intervals excluding zero.

V2 is the retrain on the filtered corpora with proportional sampling and no
synthetic hard negatives — the configuration your W6 points at — on 7 GPUs rather
than 1. It is not a controlled decontamination ablation. Denominators, once: 23
tasks in the sweep, 20 scorable under both probes for every arm, 22 completed by
HMMER. Counts use a +/-0.005 tie band; checkpoint-to-checkpoint spread inside our
own run is 0.005 to 0.008, so we defend no individual delta below 0.010.

**The claims left, as claims.** One: this supervision makes structural family
membership recoverable from the geometry without labels — SCOPe-40 adjusted Rand
index 0.054 for stock ESM-2 35M against 0.507 for V2 at the true family count.
Two: that space ranks homologs at depth competitively with a tuned profile search.
*Withdrawn:* the general-purpose claim, all 150M results and the two-scale
contribution bullet, Table 2's aggregate, Table 3, Table 5, the PPI gain, label
scarcity, the Stability comparison.

### W1. Leakage

**Filtered and verified.** MMseqs2 `easy-search`, corpus as query, test set as
target, 40% identity / 80% coverage of the test sequence; any corpus sequence with
a hit is dropped. Pfam 28,530,684 rows to 27,929,772 (-2.11%), AFDB 135,404,259 to
126,301,607 (-6.72%), STRING 76,070,154 to 71,891,417 (-5.49%). Each training
parquet was then semi-joined against the removal lists, both STRING columns: zero
flagged sequences survive. Training opened 27,929,772 + 126,301,607 + 15,000,000 =
169,231,379 rows; STRING was subsampled to 15M at seed 42 to fit a 12-hour compute
budget, a budget decision and not a leakage control. Targets: `remote_homology`
(3,244 test sequences) for Pfam and AFDB, `ppi_bernett` (3,022) for STRING. Those
two tasks are decontaminated; the other 21 are not. Three counts disagree with
Table 1, which reports Pfam 32.9M domains, AFDB 133.9M sequences and STRING 36.5M
pairs: Table 1 came from the dataset-construction logs and counts STRING pairs,
not rows.

**The residual, in full.** AFDB's filter used a k-mer prefilter (`-s 5.7`) at
89.4% recall against exhaustive search, and it flagged 7,414,137 unique AFDB
sequences. If that recall transfers, the 10.6% it missed is roughly 0.9M sequences
that have a 40%-identity hit to the fold test set and are still in the corpus:
about 0.7% of the 126.3M filtered AFDB rows. Pfam and STRING used the exhaustive
GPU prefilter at 100% recall, so the residual is AFDB-only. As a direct check,
1,000 random *filtered* AFDB sequences re-searched against the fold test set with
the exhaustive prefilter return zero hits.

**Fold-level exposure is not bounded by any identity filter**, and we say so
first. Supervision is Foldseek-cluster and Pfam-family co-membership, so a training
pair sharing a query's fold at 15% identity survives any threshold. The semi-join
that would settle it — gallery against our AFDB Foldseek cluster IDs — is not run.
What we can put against it is a memorization test, which points the other way. On
SCOPe-40, per-query Spearman between a query's maximum identity to the pretraining
corpus and V2's gain in average precision over ESM-2 is -0.116 (p=1.6e-06): the
advantage shrinks as queries get closer to training data, where memorization
predicts the opposite sign. It survives the obvious confound, that high-identity
queries have less headroom, three ways: partial Spearman controlling for baseline
score -0.081 (p=9.0e-04); every within-quartile correlation null or negative
(+0.007, -0.090, -0.158, -0.057); and among the 404 queries where the untuned
backbone scores zero at Recall@10, so headroom is maximal by construction,
identity does not predict the gain (+0.038, p=0.45). By identity bin, V2's
Recall@10 gain over ESM-2 is +0.152 (n=164, identity below 0.4), +0.181 (n=315)
and +0.157 (n=1,214, above 0.7). That is evidence against memorization, not a
fold-exclusion control, and we do not present it as one.

SCOPe-40 itself cannot be filtered at corpus level: it has no train/test split. We
filtered the benchmark instead. On the 164 eligible queries below 40% identity to
our corpus, paired V2 minus the default-settings HMMER over 10,000 resamples is
+0.116 [+0.049, +0.189] at Recall@10, against +0.141 [+0.120, +0.162] over all
1,693. Top-1 there is -0.043 [-0.128, +0.043], unresolved at n=164 rather than
tied. That subset was not re-scored at full sensitivity, so it is a clean-query
win over the weaker HMMER configuration.

**Remote homology, test split**, the task the corpus was filtered against: 3-NN
accuracy 0.584 for ESM-2 35M, 0.659 for V1, 0.667 for V2; linear macro-F1 0.441,
0.428, 0.453. Filtering did not cost performance on it. That is not a
decontamination measurement — no unfiltered retrain at V2's configuration exists —
and +0.008 from V1 to V2 is below what we defend.

**PPI: we withdraw the claim.** The filter ran at 40% identity, not the 50% you
named, removing 4,178,737 STRING pairs. `ppi_bernett` is not among the 23 tasks in
the rebuttal sweep, so no post-filter PPI number exists for any arm. The paper's
+5.3% PPI gain stays a pre-decontamination V1 number and we no longer claim it.

### W2 to W5. Objective, batching, evaluation protocols

**The DMS loss** works as you propose, and our text is wrong: the paper says it
"operates on single proteins rather than pairs". The code writes (wild-type,
mutant, within-assay normalised fitness in [0,1]) rows and CoSENT ranks those pairs
within a batch — if mutant a outscores b, the loss pushes cos(WT, a) above
cos(WT, b). Nothing pulls high-fitness variants onto the wild type. Pairing is
wild-type-anchored, so mutant-mutant geometry is set indirectly.

**MNRL: you are right, and it is a real error.** The submitted 1,024 is an
optimizer batch formed by gradient accumulation, which does not share in-batch
negatives, so each MNRL call saw 64 examples at 35M and 16 at 150M. The objective
the paper describes was never trained at either scale until V2 — one reason the
150M arm is withdrawn. Eq. 1's superscript marks the positive: numerator anchor
i's positive, denominator all N.

**Peptide-HLA is not a pair task in our code.** The dataset supplies one
pipe-joined `HLA_pseudoseq|peptide` field, so the model sees one mean-pooled
representation over both chains — a defect in the task, not a pair protocol. Our
tokenizer maps `|` to `X`; before that fix it raised a KeyError on the ESM-2 35M
3-NN arm only, and its rerun gives AUC 0.750 against the submitted 0.748. PPI
partners are embedded independently and concatenated, not encoded as a pair. k-NN
regression is uniform over 3 neighbours, except in the few-shot code, which sets
`n_neighbors = max(1, min(3, train_size))`.

### W6. Ablations do not support the defaults

We acted on this. Removing synthetic hard negatives improves 20 of 23 tasks at
mean +7.9%, against 16 of 23 and +6.7% for the submitted configuration;
proportional sampling gives +7.0% against round-robin's +6.7%. V2 uses neither
default. Acting on it has a price the corpus filter cannot pay: V2's configuration
was selected from two binary comparisons scored on the same 23-task suite it is now
evaluated on, so V2's numbers there are not held out. Selection consumed two bits
and no more, the SCOPe-40 results were not among the ablation criteria, and both
alignment tools ran after the configuration was fixed.

### W7. Baselines

HMMER phmmer and MMseqs2 both ran through the same gallery and scoring code as the
embeddings, over the 1,693 SCOPe-40 queries with a non-self same-family neighbour;
the other 514 are singleton families, unachievable for any method.

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.410 |
| HMMER phmmer, defaults (`-E 10`) | 0.697 | 0.781 | 0.475 |
| HMMER phmmer, heuristic filters off | 0.753 | 0.898 | 0.607 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.646 |

The filters-off row is the one we ask to be judged against. The default
MSV/Viterbi/Forward prefilters, not the E-value, leave 31.3% of SCOPe-40 and 47.6%
of remote-homology queries with no hit; raising `-E` alone changes nothing.
Turning them off takes 83 seconds for the whole all-vs-all and covers every query.
Every other alignment number in this response is at defaults, so read those
coverage figures as a property of that configuration, not of alignment search.
HMMER beats MMseqs2 on 12 of the 22 tasks both finished, and alignment beats our
best embedding arm outright on 3 of 23 tasks under a 3-NN probe and 6 under a
linear probe.

Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek. This is the weakness we did
not close: with no learned-embedding baseline our R@10 of 0.922 has no reference
frame beyond alignment and the backbone. We do not quote their published numbers
either, because none is protocol-comparable to a leave-one-out family-level
evaluation with singletons scored as failures. Against Redl et al. 2023, which
shares our objective family, our difference is four relation types, not one.

### W8. Statistical evidence

You are right about Table 2, so we withdraw its aggregate rather than caveat it.
It is single-run per cell with no intervals, and no delta below 0.010 is resolved
by it, the V1-to-V2 +0.008 included. Bootstrapping the per-task cells needs no
refitting; it is open.

Where intervals exist they are paired over the 1,693 queries, 10,000 resamples: V2
minus ESM-2 35M is +0.185 [+0.162, +0.210] at R@1, +0.161 [+0.141, +0.180] at R@10
and +0.223 [+0.208, +0.238] at MAP. Those resample queries, not training runs. Our
five-seed sweep varies only the probe, deterministic on fixed embeddings and a
fixed split (median SD 0.000 over 8 tasks and 3 arms). Model-level variance is
unmeasured and the surviving claim inherits that.

One exposure we name rather than wait to be asked: the ARI comparison clusters raw
mean-pooled embeddings with average linkage, and ESM-2's space is strongly
anisotropic. We did not cluster a centred or whitened ESM-2 as a third arm, and did
not try a second linkage, so we cannot rule out that part of the 0.054-to-0.507 gap
is preprocessing. Two things bound it: ARI and silhouette are scale-invariant, and
ESM-2's own NMI is 0.823, so the untuned space carries family information that ARI
at k = 917 does not credit.

On the single-space assumption in your Limitations. Retrieval reads geometry
directly; a linear probe refits per task and can recover what geometry has buried.
V2 loses 11 of 20 comparable tasks to its own untuned backbone under a linear probe
(2 win, 7 tie) while recovering the family partition at ARI 0.507 against 0.054.
Compressing four relation types into one metric buys the second and costs the
first. A layer sweep shows part of it is our protocol: on remote homology V2 leads
at every pooled layer from 6 upward, and the final layer the benchmark uses is the
worst layer for both models.

Two errors no reviewer raised, since they touch claims you are judging. The
remote-homology test split is not hierarchy-disjoint: it is TAPE's three holdouts
pooled, 718 fold + 1,254 superfamily + 1,272 family = 3,244, so its macro AUC is
not comparable to published per-holdout accuracies. And RhlA at +77.2%, the
paper's second-largest 35M gain, is measured on 6-residue mutation-site strings,
not proteins.

W1 reduces to a 0.7% identity residual in AFDB plus unbounded fold-level overlap,
against which the identity-vs-gain analysis is evidence and not proof. W2 to W6 are
answered. W7 has both alignment tools and no learned baseline. W8 is answered by
withdrawal on the suite and paired intervals on retrieval. We ask you to reconsider
on that basis.
<!-- END Yi1G -->
