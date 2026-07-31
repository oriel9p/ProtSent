# Follow-up response — additional results

*Post as a new official comment. Two parts, each self-contained.*

*Before posting, check every claim against `rebuttal/FINAL_rebuttal.md` (what we already
said) — this document deliberately does NOT repeat answers already given there on MNRL
batch semantics, Eq. 1, pair-level tasks, k-NN regression weighting, or CoSENT. Those were
answered in the first response and repeating them risks contradicting it.*

---

## Part 1 — To Reviewer HNXd, on whether the revised scope warrants re-review

*(1,700 characters)*

We thank HNXd for engaging in detail, and for raising the scope question openly rather than
leaving it implicit. It is the right question and we want to answer it directly.

**What changed is the claim's scope, not the contribution or the evidence base.** The
submission already contained the SCOPe-40 retrieval evaluation, the multi-source
contrastive objective, the 23-task suite and the ablations. What changed is which of those
results we claim as the headline. We had framed ProtSent as a general-purpose embedding
improvement; the linear-probe evidence does not support that, so we withdrew it and now
claim the retrieval and neighbourhood-structure result the submitted experiments already
demonstrated. No experiment was added to create the narrower claim.

**The new experiments are controls, not new contributions.** Decontamination, the retrained
models, the alignment baselines, the confidence intervals and the seed analyses exist to
test whether the submitted result survives scrutiny. They constrain the claim rather than
extend it.

**We agree the paper text must change to match.** The abstract and introduction promise
more than we can support, several descriptions of the evaluation were inaccurate, and the
ablation tables no longer describe the configuration we would release. We would make those
changes in a camera-ready, and we do not object to the AC weighing whether that is too much
revision to accept now. That judgement is properly theirs.

We are grateful for the score increase and for the care that produced it.

---

## Part 2 — The decontaminated model at 150M, and embedding-space organization

*(4,200 characters)*

Two results completed after our initial response. One of them corrects something we said
there.

### 2.1 The decontaminated retrain now exists at both scales

Our first response reported ProtSent-V2 at 35M. The 150M has since finished training on the
same decontaminated corpus (40% identity / 80% coverage against the benchmark test
sequences, verified to leave zero flagged sequences in the training files).

SCOPe-40 structural retrieval, test split, self excluded, no-hit queries counted as
failures, restricted to the 1,693 of 2,207 queries with a non-self same-family protein in
the gallery:

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (-s 7.5) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer, filters off) | **0.7525** | 0.8978 | 0.6067 |
| ProtSent-V1 150M (submitted) | 0.6615 | 0.8943 | 0.6431 |
| ProtSent-V2 150M (decontaminated) | 0.7431 | **0.9368** | **0.7042** |

Paired bootstrap over queries, 10,000 resamples: ProtSent-V2 − ProtSent-V1 is +0.0809 R@1
[+0.0602, +0.1022] and +0.0607 MAP [+0.0477, +0.0735]; ProtSent-V2 − MMseqs2 is +0.0868 R@1
[+0.0620, +0.1116]. Both exclude zero.

**The conclusion we drew at 35M holds at 150M, and we want to be explicit that scale did
not change it.** A maximally sensitive profile search still leads at top-1 — 0.7525 against
our 0.7431 — while the embedding leads at ranking depth and MAP. Our advantage is where it
was: in how deep the correct family stays in the ranking, at one forward pass per sequence
with indexable sub-linear search, rather than an all-vs-all profile comparison.

On remote homology, the task the corpus was filtered against, the 150M behaves differently
from the 35M and we report both probes rather than the favourable one:

| model | 3-NN accuracy | linear-probe accuracy |
|---|---|---|
| ESM-2 150M | 0.5194 | 0.7500 |
| ProtSent-V1 150M | 0.7047 | 0.7401 |
| ProtSent-V2 150M | 0.6612 | 0.7503 |

Under 3-NN, decontamination costs 4.4 points relative to V1 while both ProtSent models stay
far above the untuned backbone; under a linear probe the ordering reverses. We read the
3-NN drop as the expected consequence of removing pretraining sequences at ≥40% identity to
this test set — which is what the filtering is for, and which suggests the larger model had
been benefiting from that overlap more than the smaller one. Two confounds, stated: V2
differs from V1 in configuration as well as corpus, and the 150M run used a smaller
per-cluster pair budget, so it saw 31% fewer training pairs than the 35M.

### 2.2 Embedding-space organization (HNXd, Question 1)

Measured on the 2,207-sequence SCOPe-40 gallery:

| model | mean cosine, random pair | participation ratio | dims for 95% of variance |
|---|---|---|---|
| ESM-2 35M | 0.848 | 7.9 / 480 | 112 |
| ProtSent-V2 35M | 0.152 | 52.5 / 480 | 148 |
| ESM-2 150M | 0.896 | 10.6 / 640 | 126 |
| ProtSent-V2 150M | 0.175 | 43.4 / 640 | 144 |

Participation ratio is the effective number of variance-carrying dimensions, (Σλ)² / Σλ²
over the covariance eigenvalues.

In the untuned backbones, two randomly chosen unrelated proteins have cosine similarity
0.85–0.90: the representation occupies a narrow cone with almost all variance in fewer than
eleven effective directions. Contrastive fine-tuning expands this to 43–53 effective
dimensions and largely removes the anisotropy.

We offer this as a description of what the objective does to the geometry, which is what
was asked, and not as a claim that it adds information the backbone lacked. The
reorganisation is what nearest-neighbour search consumes, and it is consistent with gains
concentrating in retrieval and clustering rather than under a trained readout.
