# DRAFT — late-interaction section for the ProtSent paper

**Status: draft for author review. Not proofed, not vetted. Numbers are current as of 2026-08-27
and regenerate with `python analyze_paired_effects.py` (no GPU).**

> **Corpus caveat, read first.** Our SCOPe-40 tables use `tattabio/scope40_test` (2,207 domains,
> all-vs-all). The paper's Table 3 states "the full validation set (100,000 proteins)". Absolute
> Recall@K is therefore **not comparable** between the two: a 45x smaller gallery makes retrieval
> easier, which is why our baseline Recall@1 is 0.698 against Table 3's 0.385. **Deltas within our
> tables are valid; the numbers must not be placed alongside Table 3 without either rerunning on the
> paper's corpus or captioning the difference.** Rerunning is the cleaner fix and is not yet done.

---

## The story, in one line

ProtSent's contribution is embedding *neighbourhood* quality. Mean-pooling compresses L residues into
one vector; late interaction reads the same frozen weights at residue level and recovers signal that
pooling discards — at no training cost.

Four beats:

1. **Pooling loses retrievable signal.** MaxSim on identical frozen weights beats cosine.
2. **It is not retrieval-specific.** Swap MaxSim for cosine inside ProtSent's own kNN probe: it wins
   16 of 18 model-task pairs.
3. **The gain is specific to ProtSent pretraining**, and is the largest effect we measure.
4. **Scale interacts with pretraining**, with a measurable mechanism — and the cost is manageable.

---

## Table A — MaxSim vs cosine on identical frozen weights (SCOPe-40, superfamily)

| Model | Scoring | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|---|
| ESM-2 35M | cosine | 0.698 | 0.845 | 0.902 | 0.398 |
| ESM-2 35M | **MaxSim** | 0.719 | 0.856 | 0.901 | **0.474** (+0.075) |
| ProtSent 35M | cosine | 0.857 | 0.956 | 0.971 | 0.654 |
| ProtSent 35M | **MaxSim** | 0.878 | 0.956 | 0.975 | **0.683** (+0.029) |
| ProtSent 150M | cosine | 0.895 | 0.953 | 0.972 | 0.670 |
| ProtSent 150M | **MaxSim** | 0.916 | 0.963 | 0.978 | **0.719** (+0.049) |

**Caption.** *Residue-level (MaxSim) versus mean-pooled (cosine) scoring of the **same frozen
weights** on SCOPe-40 superfamily retrieval. No model is retrained or fine-tuned; only the
similarity function differs. MAP deltas are paired bootstraps over the 2,020 queries with at least
one same-superfamily target (all p<0.05; ESM-2 35M +0.0754 [+0.0672,+0.0835], ProtSent 150M +0.0490
[+0.0444,+0.0533]). Corpus is `tattabio/scope40_test`, not the 100k set used in Table 3.*

## Table B — MaxSim as the metric inside the kNN probe (full train split, k=3)

| Task | Model | MaxSim | cosine kNN | linear |
|---|---|---|---|---|
| Remote Homology (F1_M) | ProtSent 150M | **0.401** | 0.387 | 0.435 |
| Remote Homology (F1_M) | ESM-2 150M | **0.299** | 0.275 | 0.374 |
| Metal Ion Binding (F1_M) | ProtSent 150M | **0.774** | 0.760 | 0.708 |
| Metal Ion Binding (F1_M) | ESM-2 150M | **0.765** | 0.735 | 0.720 |
| Subcellular Loc. (F1_M) | ProtSent 150M | **0.564** | 0.552 | 0.633 |
| Subcellular Loc. (F1_M) | ESM-2 150M | **0.577** | 0.536 | 0.620 |
| **Across 18 model-task pairs** | | **16/18 beat cosine kNN** | — | 8/18 |

**Caption.** *ProtSent's evaluation protocol with one substitution: the k=3 kNN probe scores
neighbours by MaxSim rather than by distance between mean-pooled vectors. Splits, k, metrics and
training set are unchanged, so this isolates the similarity function. MaxSim beats pooled cosine on
16 of 18 model-task pairs and a linear probe on 8 — it is a better **neighbourhood metric**, not a
replacement for supervised readout.*

## Table C — Pretraining dominates, and scale interacts with it (SCOPe-40 superfamily MAP, frozen MaxSim)

| Backbone | 35M | 150M | Δ scale |
|---|---|---|---|
| ESM-2 | 0.474 | 0.445 | **−0.029** |
| ProtSent | 0.683 | 0.719 | **+0.036** |
| **Δ pretraining** | **+0.210** | **+0.274** | |
| *Effective residue rank* | *12.1 → 15.5* | *10.5 → 22.5* | |

**Caption.** *Frozen residue-level retrieval, no projection or fine-tuning. ProtSent pretraining is
worth +0.210 [+0.197,+0.222] at 35M and +0.274 at 150M — the largest effect in this study, roughly
5x the scoring change in Table A. Scale is an **interaction, not a main effect**: it costs ESM-2
0.029 MAP and gains ProtSent 0.036. The italic row gives the mechanism — mean effective rank
(participation ratio of singular values) of each protein's residue matrix over 300 SCOPe domains.
MaxSim scores Σ_i max_j A_i·B_j and so requires residues to be individually distinguishable; ESM-2's
residues grow **more** collinear with scale while ProtSent's grow less so. The negative ESM-2 scaling
was predicted from rank before it was measured.*

---

## Prose (draft)

**Late interaction over ProtSent embeddings.** ProtSent is evaluated, throughout this work, on the
quality of embedding *neighbourhoods*. Yet the sequence embedding is a mean over L residue vectors,
and that pooling step is lossy by construction. We therefore ask whether the residue representations
underlying a ProtSent embedding retain retrieval signal that mean-pooling discards. We score protein
pairs with MaxSim, S(A,B) = Σ_i max_j A_i·B_j, the late-interaction operator from ColBERT, applied to
the **frozen** ProtSent weights with no additional training.

They do. On SCOPe-40 superfamily retrieval, MaxSim improves MAP over cosine on identical weights by
+0.049 at 150M and +0.029 at 35M (Table A), and the effect is not confined to retrieval: substituting
MaxSim for pooled distance inside our own k=3 kNN probe improves 16 of 18 model-task pairs (Table B).
MaxSim does not displace a supervised readout — a linear probe still wins where classes are linearly
separable — but as a *neighbourhood* metric it is consistently the better choice, which is precisely
the axis this work argues matters.

The gain is specific to ProtSent. At matched model size, ProtSent pretraining is worth +0.210 MAP at
35M and +0.274 at 150M over stock ESM-2 under identical MaxSim scoring (Table C) — roughly five times
the effect of the scoring change itself. More striking is that scale and pretraining **interact**:
scaling ESM-2 from 35M to 150M *degrades* frozen MaxSim retrieval by 0.029 MAP, while scaling
ProtSent improves it by 0.036. The mechanism is measurable. MaxSim requires residues within a protein
to be individually distinguishable; we quantify this as the effective rank of each protein's residue
matrix. ESM-2's residue representations become more collinear with scale (12.1 → 10.5) while
ProtSent's become less so (15.5 → 22.5). Contrastive training on pooled vectors, perhaps
counter-intuitively, *decorrelates* the residues underneath: to make a mean discriminative across many
proteins, the model cannot let its parts collapse onto one direction.

**Cost.** Late interaction stores L vectors per protein rather than one, and scores pairs rather than
looking up a single vector. Two-stage retrieval removes most of this: a pooled-cosine shortlist
reranked by MaxSim recovers 97.7% of exhaustive MaxSim MAP at superfamily with only 100 candidates
scored per query (0.703 vs 0.719), and 94.5% at 10 candidates. Only the shortlisted candidates need
residue embeddings, so the stored index stays pooled-size.

**What does not help.** Training a dedicated late-interaction projection on top of ProtSent adds
+0.004 MAP (n.s.) at 150M. The representation is already residue-level; the scoring function, not
further training, is what unlocks it.

---

## Reviewer objections to prepare for

| # | Objection | Status |
|---|---|---|
| 1 | SCOPe corpus differs from Table 3 | **Open.** Rerun on the 100k set, or caption it. |
| 2 | 18 model-task pairs is a small sweep | Partial — 3 tasks. 2 regression tasks queued. |
| 3 | Effective rank is 4 models | **Weak.** Needs more points to be a trend. |
| 4 | Storage cost of L vectors | Answered by two-stage rerank; latency not measured. |
| 5 | Why not compare to Foldseek / ProtTucker? | Same limitation the paper already states. |

## Deliberately excluded

Our late-interaction *training* runs (four init x size arms). They add nothing over frozen ProtSent
(+0.004, n.s.) and would shift the section from "ProtSent embeddings support this for free" to "we
trained models", which is both weaker and off-thesis. The proj_dim ablation (128-d beats 640-d) is
held back for the same reason — it is about our training, not about ProtSent.
