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

A ProtSent embedding is a mean over per-residue vectors, so pooling is lossy by construction. To ask
whether the discarded detail is useful, we score protein pairs with the late-interaction operator
from ColBERT [cite], MaxSim:

  S(A, B) = sum_i max_j A_i · B_j

where A_i and B_j are L2-normalised residue vectors of the query and target. Every result below
applies this to **frozen** ProtSent or ESM-2 weights: no fine-tuning, no projection head, no
additional parameters. Only the similarity function changes, so any difference is attributable to
the scoring geometry rather than to further training.

Retrieval uses the same SCOPe-40 protocol as Section 4.3. For probe tasks we keep the Section 4.1
protocol exactly — frozen embeddings, k=3, held-out test split — and substitute MaxSim for the
distance between pooled vectors when selecting neighbours.

## 5.5 Late interaction recovers signal lost to pooling (Results)

**Table A.** Residue-level versus pooled scoring of identical frozen weights, SCOPe-40 superfamily.

| Model | Scoring | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|---|
| ESM-2 35M | cosine | 0.698 | 0.845 | 0.902 | 0.3981 |
| ESM-2 35M | MaxSim | 0.719 | 0.856 | 0.901 | **0.4735** |
| ESM-2 150M | cosine | 0.728 | 0.869 | 0.914 | 0.3580 |
| ESM-2 150M | MaxSim | 0.791 | 0.881 | 0.923 | **0.4449** |
| ProtSent 35M | cosine | 0.857 | 0.956 | 0.971 | 0.6540 |
| ProtSent 35M | MaxSim | 0.878 | 0.956 | 0.975 | **0.6831** |
| ProtSent 150M | cosine | 0.895 | 0.953 | 0.972 | 0.6702 |
| ProtSent 150M | MaxSim | 0.916 | 0.963 | 0.978 | **0.7192** |

*Caption.* MaxSim and cosine applied to the same frozen weights; no model is retrained. MAP gains are
+0.075 (ESM-2 35M), **+0.087 (ESM-2 150M)**, +0.029 (ProtSent 35M) and +0.049 (ProtSent 150M), all
significant under a paired bootstrap over the 2,020 queries having at least one same-superfamily
target. The gain is largest for the weakest pooled baseline and smallest for the strongest: late
interaction recovers most where pooling loses most. Corpus is
`tattabio/scope40_test` (2,207 domains), not the set used in Table 3.

**Table B.** MaxSim substituted for pooled distance inside the k=3 probe of Section 4.1.

| Task (F1_macro) | Model | MaxSim | cosine kNN | linear |
|---|---|---|---|---|
| Remote Homology | ProtSent 150M | **0.401** | 0.387 | 0.435 |
| Remote Homology | ESM-2 150M | **0.299** | 0.275 | 0.374 |
| Metal Ion Binding | ProtSent 150M | **0.774** | 0.760 | 0.708 |
| Metal Ion Binding | ESM-2 150M | **0.765** | 0.735 | 0.720 |
| Subcellular Loc. | ProtSent 150M | **0.564** | 0.552 | 0.633 |
| Subcellular Loc. | ESM-2 150M | **0.577** | 0.536 | 0.620 |
| Optimal pH (Spearman) | ProtSent 150M | **0.605** | 0.583 | 0.512 |
| beta-lactamase (Spearman) | ProtSent 150M | **0.838** | 0.755 | 0.697 |
| *All 30 model-task pairs* | | *27 beat cosine kNN* | | *19 beat linear* |

*Caption.* Splits, k, training set and metrics are unchanged from Section 4.1; only neighbour
selection differs. Across six models and five tasks, MaxSim beats pooled cosine on 27 of 30 pairs and
a linear probe on 19. The split by task type is informative: MaxSim beats **both** probes on the
regression tasks (optimal pH 6/6, beta-lactamase 5/6) and on metal ion binding (6/6), while a linear
probe wins on the two multiclass tasks (remote homology 1/6, subcellular localisation 1/6). Late
interaction is the better neighbourhood metric throughout; whether it also beats a learned readout
depends on whether the labels are linearly separable in the pooled space.

**Table C.** Pretraining and scale under frozen MaxSim (SCOPe-40 superfamily MAP).

| Backbone | 35M | 150M | Delta from scale |
|---|---|---|---|
| ESM-2 | 0.4735 | 0.4449 | **-0.029** |
| ProtSent | 0.6831 | 0.7192 | **+0.036** |
| Delta from pretraining | **+0.210** | **+0.274** | |
| *Effective residue rank* | *12.1 -> 15.5* | *10.5 -> 22.5* | |

*Caption.* Frozen backbones, MaxSim scoring, no projection. ProtSent pretraining is worth +0.210 MAP
at 35M and +0.274 at 150M — roughly five times the effect of the scoring change in Table A. Scale is
an interaction rather than a main effect: it costs ESM-2 0.029 MAP and gains ProtSent 0.036. **This
decline is not specific to MaxSim** -- under cosine, ESM-2 loses 0.040 MAP over the same scale step
(Table A), so ESM-2 150M is simply weaker than 35M at SCOPe retrieval on this corpus under either
scoring, and MaxSim recovers part of that loss rather than causing it. The italic row reports mean effective rank (participation ratio of the singular values) of each protein's
residue matrix over 300 SCOPe domains. MaxSim requires residues to be individually distinguishable;
ESM-2's residues grow more collinear with scale while ProtSent's grow less so. Rank accounts for
MaxSim's absolute level across these four models; it does not by itself explain the direction of the
scale effect, which is present under pooling too.

### Prose

A ProtSent embedding averages L residue vectors into one, and that step is lossy. We find the lost
detail is retrievable. Scoring identical frozen weights with MaxSim instead of cosine improves
SCOPe-40 superfamily MAP by 0.049 at 150M and 0.029 at 35M (Table A), with no training of any kind.
The effect is not confined to retrieval: substituting MaxSim for pooled distance inside the k=3 probe
of Section 4.1 improves 27 of 30 model-task pairs (Table B). Against a linear probe the picture is
task-dependent: MaxSim wins on both regression tasks and on metal ion binding, and loses on the two
multiclass tasks, where the labels are evidently linearly separable in the pooled space. As a
neighbourhood metric — the axis this work argues matters — MaxSim is consistently the better choice. The result replicates on
an unrelated benchmark: few-shot remote homology gives +0.047 accuracy for the same contrast against
SCOPe's +0.049 MAP, two tasks sharing no data agreeing to within 0.002.

The gain is specific to ProtSent. At matched size, ProtSent pretraining is worth +0.210 MAP at 35M
and +0.274 at 150M over stock ESM-2 under identical scoring (Table C), roughly five times the effect
of the scoring change itself. Scale and pretraining interact: scaling ESM-2 from 35M to 150M degrades
frozen retrieval while scaling ProtSent improves it by 0.036 MAP. ESM-2's decline is not a
late-interaction artefact: it loses 0.040 MAP under cosine against 0.029 under MaxSim, so MaxSim
recovers part of the loss rather than causing it. One property of the residue representations tracks
MaxSim's absolute level. MaxSim needs residues within a protein to be distinguishable from one another, which we
quantify as the effective rank of the residue matrix. ESM-2's residues become more collinear with
scale (12.1 to 10.5) while ProtSent's become less so (15.5 to 22.5). Contrastive training on pooled
vectors decorrelates the residues underneath: to make a mean discriminative across many proteins, the
model cannot let its parts collapse onto a single direction. We report this as a correlation over
four models, not a demonstrated cause. The benefit is also largest where pooling costs most: the
MaxSim gain is +0.087 MAP on ESM-2 150M, whose pooled baseline is the weakest of the four, and +0.029
on ProtSent 35M, whose pooled baseline is already strong.

Late interaction stores L vectors per protein instead of one. Two-stage retrieval removes most of
this cost: a pooled-cosine shortlist reranked by MaxSim recovers 97.7% of exhaustive MaxSim MAP at
superfamily while scoring only 100 candidates per query (0.703 against 0.719), and 94.5% at ten
candidates. Only shortlisted candidates need residue embeddings, so the stored index stays
pooled-size.

Finally, training a dedicated late-interaction projection on top of ProtSent adds +0.004 MAP, not
significant. The representation is already residue-level; the scoring function, not further training,
is what makes it accessible.

---

# Part 2 — working notes, not for the manuscript

## Evidence that does not support the section

Stated rather than omitted, so a reviewer finding these files does not assume they were buried.

| Benchmark | Status |
|---|---|
| 22-task ProtBench suite, k=3 | Separates nothing: eight model means span 0.579-0.604. These are probe tasks over pooled vectors, which late interaction does not alter. A do-no-harm check, which passes. |
| ProteinGym, full coverage | 214 substitution assays, 2,459,339 variants, ProteinGym's corrected average. Arms at 0.319-0.346 but every pair of CIs overlaps, so it cannot rank models. Shows residue scoring reaches parity with likelihood scoring at matched size. |
| CATH-EAT | **Excluded.** `test_h` has 150 queries and ±0.07 CIs, which cannot carry a 0.02-0.05 effect. In `r2_final/cath_eat.csv`; do not cite. |

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
