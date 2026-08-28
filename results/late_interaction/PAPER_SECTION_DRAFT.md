# Late interaction over ProtSent embeddings — draft section

**File:** `results/late_interaction/PAPER_SECTION_DRAFT.md`. Paper-ready prose is in Part 1;
Part 2 is working notes and is not for the manuscript. Numbers regenerate with
`python analyze_paired_effects.py` (no GPU). Last updated 2026-08-27.

> **Blocker before this goes in the paper.** Our SCOPe-40 tables use `tattabio/scope40_test`
> (2,207 domains, all-vs-all). Section 9 of the paper describes SCOPe-40 retrieval as "the full
> validation set (100,000 proteins)". A 45x smaller gallery makes retrieval easier — our baseline
> Recall@1 is 0.698 against Table 3's 0.385. **Deltas within our tables are valid; absolute recalls
> are not comparable to Table 3.** Either rerun on the paper's corpus or caption the difference.
> Rerunning is cleaner and is not yet done.

---

# Part 1 — for the manuscript

## 3.7 Late interaction (Method)

A ProtSent embedding is a mean over per-residue vectors, so pooling is lossy by construction. To test
whether the discarded detail is useful we score pairs with MaxSim, the late-interaction operator from
ColBERT [cite]: S(A,B) = sum_i max_j A_i · B_j, over L2-normalised residue vectors. Every result
below applies this to **frozen** weights — no fine-tuning, no projection, no added parameters — so
differences are attributable to the scoring geometry alone. Retrieval follows Section 4.3; probe
tasks follow Section 4.1 exactly (frozen embeddings, k=3, held-out test split) with MaxSim
substituted for pooled distance when selecting neighbours.

## 5.5 Late interaction recovers signal lost to pooling (Results)

**Table A.** SCOPe-40 superfamily retrieval, identical frozen weights, cosine vs MaxSim.

| Model | R@1 cos / MaxSim | MAP cos | MAP MaxSim | Δ |
|---|---|---|---|---|
| ESM-2 35M | 0.698 / 0.719 | 0.3981 | 0.4735 | +0.075 |
| ESM-2 150M | 0.728 / 0.791 | 0.3580 | 0.4449 | **+0.087** |
| ProtSent 35M | 0.857 / 0.878 | 0.6540 | 0.6831 | +0.029 |
| ProtSent 150M | 0.895 / 0.916 | 0.6702 | 0.7192 | +0.049 |

*Caption.* No model is retrained; only the similarity changes. All four gains are significant under a
paired bootstrap over the 2,020 queries with at least one same-superfamily target. The gain is
largest where the pooled baseline is weakest. Under MaxSim, ProtSent leads ESM-2 by +0.210 (35M) and
+0.274 (150M) — roughly five times the scoring effect. Corpus is `tattabio/scope40_test` (2,207
domains), **not** the set used in Table 3.

**Table B.** MaxSim substituted for pooled distance inside the k=3 probe of Section 4.1.

Absolute scores are for frozen ProtSent-150M; win counts are over all six models.

| Task | Metric | MaxSim | cosine kNN | linear | wins vs cos / lin |
|---|---|---|---|---|---|
| beta-lactamase | Spearman | **0.838** | 0.760 | 0.698 | 5/6, 5/6 |
| Metal Ion Binding | F1_macro | **0.774** | 0.760 | 0.708 | 6/6, 6/6 |
| Optimal pH | Spearman | **0.605** | 0.586 | 0.503 | 6/6, 6/6 |
| Subcellular Loc. | F1_macro | 0.564 | 0.552 | **0.633** | 6/6, 1/6 |
| Remote Homology | F1_macro | 0.401 | 0.387 | **0.435** | 4/6, 1/6 |
| **Total, 30 pairs** | | | | | **27/30, 19/30** |

*Caption.* Six models x five tasks; splits, k, training set and metrics are unchanged from
Section 4.1 and only neighbour selection differs. MaxSim beats pooled cosine on 27 of 30 model-task
pairs. Against a learned readout the result is task-dependent: it beats a linear probe on both
regression tasks and on metal ion binding, and loses on the two multiclass tasks, where the labels
are evidently linearly separable in the pooled space.

### Prose

Pooling L residue vectors into one is lossy, and the lost detail is retrievable. Scoring identical
frozen weights with MaxSim instead of cosine raises SCOPe-40 superfamily MAP by 0.049 at 150M and
0.029 at 35M, with no training (Table A); the gain reaches 0.087 on ESM-2 150M, whose pooled baseline
is weakest, and is smallest on ProtSent 35M, whose pooled baseline is already strong. The effect is
not confined to retrieval: substituting MaxSim inside the k=3 probe of Section 4.1 improves 27 of 30
model-task pairs (Table B). Against a linear probe the picture is task-dependent — MaxSim wins on
regression and binary tasks and loses on multiclass, where the labels are evidently linearly
separable in the pooled space. It replicates on an unrelated benchmark: few-shot remote homology
gives +0.047 accuracy for the same contrast against SCOPe's +0.049 MAP.

The advantage over stock ESM-2 is specific to ProtSent: +0.210 MAP at 35M and +0.274 at 150M under
identical scoring, roughly five times the scoring effect itself. Scaling ProtSent improves frozen
retrieval (+0.036 MAP) while scaling ESM-2 degrades it, though that decline is not a late-interaction
artefact — ESM-2 loses more under cosine (0.040) than under MaxSim (0.029), so MaxSim recovers part
of the loss rather than causing it. One property of the residue representations tracks MaxSim's
absolute level across these four models: effective rank of the residue matrix rises for ProtSent with
scale (15.5 to 22.5) and falls for ESM-2 (12.1 to 10.5), consistent with MaxSim requiring residues to
be individually distinguishable. We report this as a correlation over four models, not a cause.

Late interaction stores L vectors per protein rather than one. Two-stage retrieval removes most of
that cost: a pooled-cosine shortlist reranked by MaxSim recovers 97.7% of exhaustive MaxSim MAP at
superfamily while scoring 100 candidates per query, and 94.5% at ten. Only shortlisted candidates
need residue embeddings, so the stored index stays pooled-size. Finally, training a dedicated
late-interaction projection on ProtSent adds +0.004 MAP, not significant: the representation is
already residue-level, and the scoring function rather than further training is what makes it
accessible.

---

# Part 2 — working notes, not for the manuscript

## Evidence that does not support the section

Stated rather than omitted, so a reviewer finding these files does not assume they were buried.

| Benchmark | Status |
|---|---|
| 22-task ProtBench suite, k=3 | Separates nothing: eight model means span 0.579-0.604. These are probe tasks over pooled vectors, which late interaction does not alter. A do-no-harm check, which passes. |
| ProteinGym, full coverage | **Cut from the section, and the reason matters.** It never tested this section's claim: there is no cosine ProteinGym arm, so it says nothing about MaxSim versus pooling. What it does test is ProtSent versus ESM-2 on mutation effects -- see the note below, which is for the authors, not for this section. |
| CATH-EAT | **Excluded.** `test_h` has 150 queries and ±0.07 CIs, which cannot carry a 0.02-0.05 effect. In `r2_final/cath_eat.csv`; do not cite. |

### ProteinGym: for the authors, not for this section

Frozen MaxSim, matched size, ProtSent-V2 minus stock ESM-2:

| Variant | 35M | 150M |
|---|---|---|
| DMS substitutions (corrected avg) | -0.021 | -0.010 |
| DMS indels (Spearman) | -0.017 | -0.043 |
| *SCOPe-40 superfamily, same weights* | *+0.210* | *+0.274* |

ProtSent trails stock ESM-2 in all four comparisons on mutation effects, having led it by +0.21 to
+0.27 on structural retrieval. No single difference is significant -- the intervals overlap -- but the
direction holds across both variants and both scales. A plausible reading is that contrastive
training on family and structural positives builds exactly the invariance a point substitution must
break for ProteinGym to score it; the paper already reports stability and thermostability regressing
on that axis.

This is out of scope for a section about scoring geometry, which is why it is not in Part 1. It is in
scope for the paper's general-purpose-embedding claim, so it should be a deliberate decision by the
authors rather than a silent omission. Full numbers: `r2_final/proteingym_maxsim.csv`. ProtSent-V2
does not train on `dms_cosent.parquet`, so ProteinGym is genuinely held out for these models.

## Open items

| # | Item | Severity |
|---|---|---|
| 1 | SCOPe corpus differs from Table 3 (2,207 vs 100k) | **Blocker** |
| 2 | Effective rank rests on 4 models — correct prediction, not yet a trend | Medium |
| 3 | Table B covers 5 tasks x 6 models; EC excluded (multilabel) | Low |
| 3b | ProteinGym frozen arms OOM: an unprojected 150M arm needs a 26.6 GiB allocation for one 8192-query chunk at 1024 residues. Needs a smaller `_QUERY_CHUNK`. Not chased -- ProteinGym does not discriminate anyway | Low |
| 4 | Rerank latency measured in MAP recovered, not wall-clock | Low |

## Excluded by choice

The four late-interaction training arms and the proj_dim ablation (128-d beats 640-d). They add
+0.004 MAP (n.s.) over frozen ProtSent and would move the section from "ProtSent embeddings support
this for free" to "we trained models" — weaker, and off the paper's thesis. The campaign was worth
running to establish that training adds nothing; that conclusion belongs here, the runs do not.

## Data provenance

| Table | Source |
|---|---|
| A, C | `pilot_35m/scope/scope_hierarchy.csv`, `analyze_paired_effects.py` |
| B | `r2_final/benchmarks/maxsim_knn_bench.csv` |
| C italic row | `r2_final/residue_geometry.csv` (`campaign/token_spread.py`) |
| Replication | `r2_final/benchmarks/late_fewshot_knn.csv` |
| Rerank | `r2_final/scope/scope_two_stage_rerank.csv` |
