# Follow-up response — additional results and remaining questions

*Post as a new official comment. Three parts; each is self-contained. Character counts are
noted so parts can be posted separately if the interface limits length.*

---

## Part 1 — To Reviewer HNXd, on whether the revised scope warrants re-review

*(~2,050 characters)*

We thank HNXd for engaging with the response in detail, and for raising the scope question
openly rather than leaving it implicit. We want to answer it directly, because we think it
is the right question.

**What changed is the claim's scope, not the contribution or the evidence base.** The
submission already contained the SCOPe-40 retrieval evaluation, the multi-source
contrastive objective, the 23-task suite, and the ablations. What the rebuttal changed is
which of those results we claim as the headline. We previously framed ProtSent as a
general-purpose embedding improvement; the linear-probe evidence does not support that, so
we withdrew it and now claim the retrieval and neighbourhood-structure result that the
submitted experiments already demonstrated. No experiment was added to create the narrower
claim — the narrower claim is what the original experiments supported.

**The new experiments are controls, not new contributions.** Decontamination, the retrained
models, the alignment baselines, the confidence intervals and the seed analyses all exist
to test whether the submitted result survives scrutiny. They constrain the claim; they do
not extend it.

**We agree the paper text must change to match.** The abstract and introduction currently
promise more than we can support, three descriptions of the evaluation were inaccurate, and
the ablation tables no longer describe the shipped configuration. We would make all of
those changes in a camera-ready and we would not object to the AC weighing whether that is
too much revision to accept now — that judgement is properly theirs.

We would only note that the direction of the change is toward a smaller, better-evidenced
claim, and that every correction in it was found and reported by us rather than by a
reviewer. We are grateful for the score increase and for the care that produced it.

---

## Part 2 — New results: decontaminated model at 150M, and embedding-space organization

*(~4,900 characters)*

Two analyses completed after our initial response.

### 2.1 The decontaminated retrain now exists at both scales

Our first response reported ProtSent-V2 at 35M. The 150M model has since finished training
on the same decontaminated corpus (40% identity / 80% coverage filtering against the
benchmark test sequences, verified to leave zero flagged sequences in the training files).

SCOPe-40 structural retrieval, test split, self excluded, no-hit queries counted as
failures, restricted to the 1,693 of 2,207 queries that have a non-self same-family protein
in the gallery:

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (-s 7.5) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer) | 0.6970 | 0.7809 | 0.4747 |
| ProtSent-V1 150M (submitted) | 0.6615 | 0.8943 | 0.6431 |
| **ProtSent-V2 150M (decontaminated)** | **0.7431** | **0.9368** | **0.7042** |

Paired bootstrap over queries, 10,000 resamples; all of these exclude zero:
V2 − V1 = +0.0809 R@1 [+0.0602, +0.1022]; V2 − HMMER = +0.0455 R@1 [+0.0219, +0.0691];
V2 − MMseqs2 = +0.0868 R@1 [+0.0620, +0.1116].

**We must correct one statement from our first response.** We wrote that alignment remains
the better top-1 method and that our advantage is confined to ranking depth. That is
accurate at 35M, where ProtSent-V2 is statistically tied with HMMER at top-1
(−0.0124 [−0.0372, +0.0124]). It is **not** accurate at 150M, where ProtSent-V2 exceeds
HMMER at top-1 significantly. The claim is scale-dependent and we should have said so.

On remote homology — the task the corpus was filtered against — the 150M behaves
differently from the 35M, and we report it as measured rather than selecting the
favourable probe:

| model | 3-NN accuracy | linear-probe accuracy |
|---|---|---|
| ESM-2 150M | 0.5194 | 0.7500 |
| ProtSent-V1 150M | 0.7047 | 0.7401 |
| ProtSent-V2 150M | 0.6612 | 0.7503 |

Under 3-NN, decontamination costs 4.4 points relative to V1 while both ProtSent models
remain far above the untuned backbone. Under a linear probe the ordering reverses. We read
the 3-NN drop as the expected consequence of removing pretraining sequences at ≥40%
identity to this test set — that is what the filtering is for, and it suggests the larger
model had been benefiting from that overlap more than the smaller one. We note two
confounds honestly: ProtSent-V2 differs from V1 in configuration as well as corpus, and the
150M run used a smaller per-cluster pair budget, so it saw 31% fewer training pairs than
the 35M.

### 2.2 Embedding-space organization (HNXd, Question 1)

HNXd asked for an analysis of how ProtSent changes the local and global organization of the
embedding space. Measured on the 2,207-sequence SCOPe-40 gallery:

| model | mean cosine, random pair | participation ratio | dims for 95% of variance |
|---|---|---|---|
| ESM-2 35M | 0.848 | 7.9 / 480 | 112 |
| ProtSent-V2 35M | 0.152 | 52.5 / 480 | 148 |
| ESM-2 150M | 0.896 | 10.6 / 640 | 126 |
| ProtSent-V2 150M | 0.175 | 43.4 / 640 | 144 |

Participation ratio is the effective number of variance-carrying dimensions,
(Σλ)² / Σλ², over the covariance eigenvalues.

In the untuned backbones two randomly chosen, unrelated proteins have cosine similarity
0.85–0.90: the representation occupies a narrow cone, with almost all variance in under
eleven effective directions. Contrastive fine-tuning expands this to 43–53 effective
dimensions and removes the anisotropy.

This is, we think, the mechanism behind the retrieval results. Nearest-neighbour search
consumes raw geometry, and a space in which every pair is 0.85-similar offers little to
rank on. It also explains why the gains concentrate in retrieval, clustering and
nearest-neighbour transfer rather than under a trained readout: a trained linear head can
compensate for a poorly conditioned space, and k-NN cannot. We would frame the contribution
in the revision as reorganising geometry rather than adding information.

---

## Part 3 — Answers to the remaining specification questions

*(~3,600 characters)*

These are questions we did not answer in enough detail the first time. All are statements
about what the released code does.

**k-NN regression weighting (Yi1G).** `KNeighborsRegressor(n_neighbors=3,
metric="minkowski")` with no `weights` argument, so scikit-learn's default applies:
**uniform averaging** over the 3 neighbours, Euclidean distance (minkowski, p=2). Not
distance-weighted. We will state this in the evaluation section.

**Pair-level tasks (Yi1G).** For PPI, each partner is embedded independently and the two
embeddings are **concatenated** before the probe is applied. Peptide-HLA is not a
two-input task in our implementation: that dataset supplies a single sequence field, so no
pair-combination operator is used, and describing it as pair-level was our error. Both
points will be explicit in the revision.

**MNRL batch semantics and Eq. 1 (Yi1G).** Each anchor is contrasted against the other
1,023 positive-side examples in its source batch. We use
`CachedMultipleNegativesRankingLoss` with a logical contrastive batch of 1,024; the
`mini_batch_size` parameter partitions only the forward/backward computation for memory and
does not reduce the negative set. Our use of "effective batch size" conflated the two, and
the revision reports the logical contrastive batch and the gradient-cache mini-batch
separately. On Eq. 1: the superscript `+` denotes the positive member of a pair; the
numerator should use the positive paired with anchor *i*, and the denominator ranges over
the positive members of all N pairs. The equation as printed is malformed and is corrected.

**CoSENT on DMS data (jVGf).** Each training row is a (wild type, mutant) pair carrying a
within-assay normalised fitness score. CoSENT is an ordinal objective over pairs: when pair
*p* has a higher fitness score than pair *q*, the loss encourages the WT–mutant cosine
similarity of *p* to exceed that of *q*. It assigns no absolute target similarity and does
not pull high-fitness mutants toward a single point. The limitation is narrower than the
review suggests but real: the configuration is WT-anchored, so mutant–mutant geometry is
not directly optimised. We state that design choice explicitly in the revision.

We should also record that the decontaminated retrains **do not include the DMS source**.
Our decontamination covered Pfam, AlphaFold DB and STRING, and the retrained models were
restricted to decontaminated data, so ProtSent-V2 is a three-source model where the
submitted ProtSent is four-source. Comparisons between them on fitness tasks are confounded
by this, and we will say so.

**Missing citation (jVGf).** The broken reference on line 21 is Heinzinger et al.; it is
fixed, along with the other reference issues noted.

**Positioning against structure-informed sequence models (jVGf).** We agree this is the
main missing piece of context and it is a writing gap, not an experimental one. The
revision discusses ESM-S, S-PLM, ISM and Magneton, and ProTrek among retrieval systems,
and distinguishes them from ProtSent on method rather than on performance: those approaches
inject structural information into a sequence encoder through structure-aware pretraining
or distillation objectives, whereas ProtSent aligns sequence-level representations across
several relation types — family, structural cluster, physical interaction — using
contrastive objectives over sequence pairs alone, with no structural input at training or
inference time. We claim no superiority to any of them; we have not run matched
comparisons, and we say so rather than implying a ranking. We think the honest positioning
is that these are complementary routes to related goals, and that ProtSent's distinguishing
property is that its supervision is relational and its inference is sequence-only.
