# ProtSent — NeurIPS 2026 rebuttal DRAFT

OpenReview: submission 28056. Each response must be **under 10,000 characters**.
**No links or attachments** in OpenReview. Every `[[RESULT: ...]]` placeholder must
be filled or deleted before posting. Post each response under its own review.

Status: this draft predates the current decontamination/retraining runs. Numbers
marked `[[RESULT]]` are still pending.

---

## Response to Reviewer HNXd

Thank you for the constructive review and for stating which analyses would change your assessment. We agree that the paper should separate two questions more clearly: whether ProtSent improves embedding neighborhoods, and how well a trained downstream predictor can use those embeddings.

### 1. Direct retrieval and embedding-space organization

One clarification is important: the submission already includes a direct retrieval experiment in Table 3 — cosine nearest-neighbor retrieval on SCOPe-40 with Recall@1/10/30. The missing part was a broader analysis of the geometry and clustering, which we have now added.

We first reproduced the submitted retrieval comparison:

| Scale | Model | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|---|
| 150M | ESM-2 | 0.4237 | 0.5908 | 0.6457 | 0.3249 |
| 150M | ProtSent | 0.5066 | 0.6860 | 0.7245 | 0.4932 |
| 35M | ESM-2 | 0.3833 | 0.5841 | 0.6402 | 0.3235 |
| 35M | ProtSent | 0.4495 | 0.6529 | 0.7100 | 0.4225 |

The MAP gains show that the effect is not confined to the first neighbor. A basic MMseqs2 nearest-neighbor baseline on the same gallery obtains family-level R@1=0.3539 and MAP=0.1795.

In a separate residue-only mean-pooling audit on the same fixed gallery, we also measured the geometry directly. At 150M, ProtSent changes family silhouette from -0.148 to 0.039, NMI from 0.852 to 0.893, ARI from 0.165 to 0.313, and the intra/inter-family distance ratio from 0.701 to 0.418. The Spearman correlation between embedding distance and shared SCOPe hierarchy depth strengthens from -0.247 to -0.561; the 35M model shows the same pattern. Not every metric improves: class-balanced alignment worsens, so the supported claim is improved global separation and hierarchical organization — not that every family becomes tighter.

Our audit also found a description error: the submitted code evaluates the family field on 2,207 SCOPe sequences, whereas the text calls this superfamily retrieval and elsewhere states 100,000 sequences. We will correct both. Under the same residue-only audit, a separate superfamily analysis still shows substantial gains: R@1 increases from 0.667 to 0.780 at 150M and from 0.639 to 0.726 at 35M.

### 2. Linear/ridge probes and downstream fine-tuning

We agree that a conventional learned readout is needed to contextualize the 3-NN probe. The 3-NN results were intended to measure neighborhood quality, not to claim state-of-the-art task prediction. We are therefore running frozen logistic-regression and ridge probes on the same splits for vanilla ESM-2 and ProtSent.

[[RESULT: one compact sentence: number of completed tasks improved at 35M and 150M, then 3-5 representative absolute scores/deltas for remote homology, PPI, fluorescence, variant effect, stability.]]

[[RESULT: if representative LoRA/PEFT results finish, add one sentence. Otherwise omit. Do not imply full fine-tuning was completed.]]

A full end-to-end sweep over four encoders and 23 tasks is not feasible during rebuttal and would evaluate task adaptation rather than frozen representation quality. We therefore do not present it as completed. We will frame the available results accordingly: trained heads measure downstream task adaptation; 3-NN measures whether useful relationships are already local in the frozen embedding space. We will also narrow any label-scarcity claim if the linear-probe results do not support it.

### 3. Confidence intervals and small changes

We agree that several sub-1% differences in Table 2 should not be treated as established improvements.

[[RESULT: paired bootstrap summary: at each model scale, number of positive deltas whose 95% CI excludes zero, number negative, number unresolved. Add 3 representative intervals, including one large gain and one sub-1% change.]]

The revised reporting will use absolute metric-point deltas and mark unresolved differences as such, rather than bolding every positive point estimate.

### 4. Few-shot variability and Table 5

We agree on both points. Relative changes around near-zero baselines are misleading and explain cells such as -126.9%.

[[RESULT: 5- or 10-seed few-shot summary and absolute base/ProtSent values at N=100 and N=1000 for remote homology, one regression task, and one negative result. Include mean±SD.]]

[[RESULT: if few-shot linear-probe baselines finish, add the crossover/relative conclusion. Otherwise do not claim ProtSent is superior to a trained head under label scarcity.]]

Table 5 will report absolute scores and seed variability; relative changes will be secondary.

The new analyses directly address the criteria you identified for reconsidering the paper. If these results resolve your concerns, we would appreciate an updated assessment; if one point remains decisive, please indicate which one so we can address it during discussion.

---

## Response to Reviewer jVGf

Thank you for identifying the two questions that would change your assessment: whether ProtSent contributes more than structural-information injection, and where it lies on the generality–accuracy trade-off.

### 1. What is learned beyond structural supervision?

Structural supervision is important, and we should state that plainly. In the submitted ablation, removing AFDB reduces the mean relative gain from +6.7% to +3.2%, the number of improved tasks from 16/23 to 13/23, and the remote-homology gain from +40.5% to +15.3%. Thus, some structural-task gains are indeed attributable to AFDB/Foldseek supervision.

However, the ablations also show that the method is not only an AFDB structure-injection model:

- Without AFDB, ProtSent still improves 13/23 tasks.
- Without Pfam, it improves 15/23 tasks, with mean +4.6%.
- Removing STRING changes PPI from +5.3% to -0.5%, while most other tasks remain similar.
- Removing DMS primarily reduces fitness-related gains, including fluorescence.

These source-specific effects are the intended contribution: a sequence-level metric space jointly shaped by evolutionary family, structural-cluster, physical-interaction, and fitness-order relations. This differs from the specific contribution we claim: multi-relation, sequence-level metric learning rather than structure-focused representation enrichment alone. We will revise the related-work discussion to position ProtSent alongside ESM-S, S-PLM, ISM, Magneton, SaProt, and ProSST, without claiming superiority to them.

[[RESULT: if the joint no-AFDB/no-Pfam ablation finishes, insert completed-task count and representative structural, PPI, fitness results. Otherwise delete this line; the single-source ablations above already support the narrower claim.]]

Applying ProtSent to SaProt or ProSST is not a simple backbone substitution at the data level: their inputs require residue-level structure-derived tokens for the large Pfam and STRING training corpora. Preparing those inputs is outside the rebuttal window. We therefore do not present an unreliable comparison or promise that result as completed.

### 2. Generality–accuracy trade-off

We added a directly matched sequence-search reference on SCOPe. Self hits are removed; absent hits count as failures.

| Method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 nearest-neighbor search | 0.3539 | 0.3856 | 0.3856 | 0.1795 |
| ESM-2 150M | 0.4237 | 0.5908 | 0.6457 | 0.3249 |
| ProtSent 150M | 0.5066 | 0.6860 | 0.7245 | 0.4932 |

This is a basic MMseqs2 nearest-neighbor baseline, not an optimized profile-search system. We do not infer superiority to HMMER, Foldseek, ProtTucker, PLMSearch, DHR, or ProTrek from this table. Those systems use different inputs, indexes, training objectives, or benchmark protocols. A fair expected trade-off is that a specialized retrieval system may perform better on its target retrieval problem, whereas ProtSent provides one frozen sequence embedding that can also be used across classification, regression, PPI, and fitness tasks. We will state this as the paper's scope rather than leave it implicit.

[[RESULT: insert the frozen linear/ridge summary if available, because it further quantifies how broadly the representation transfers beyond nearest-neighbor retrieval.]]

### 3. CoSENT on DMS data

Each training row is a (wild type, mutant) pair with a within-assay normalized fitness score. CoSENT does not regress that score to an absolute cosine value and does not simply pull all high-fitness mutants to one point. It compares pairs within a batch: when pair p has a higher fitness score than pair q, the loss encourages the WT–mutant similarity of p to exceed that of q. Thus, the implemented objective is ordinal over WT–mutant similarities.

The limitation is narrower: the submitted configuration is WT-anchored and does not directly optimize mutant–mutant geometry. We will state this design choice explicitly.

We also correct the missing Heinzinger et al. citation and add the structure-informed and specialized-retrieval work you identified.

You indicated that clarifying these two axes could raise your score to accept. We believe the ablations, matched MMseqs2 reference, and clarified scope answer them without overstating what has not been run. If a specific remaining comparison is essential to your assessment, please identify it during discussion.

---

## Response to Reviewer Yi1G

Thank you for the detailed review. We address the eight concerns in the order raised.

### 1. AFDB/SCOPe and STRING/PPI overlap

**SCOPe.** We agree that noting possible overlap was insufficient. The original AFDB preparation filters by pLDDT and fragment status and assigns Foldseek clusters, but it does not decontaminate against SCOPe. We therefore searched every SCOPe query against the released AFDB training-source pool with MMseqs2 using 80% query coverage. This is a conservative source-overlap audit rather than proof of exact checkpoint exposure, because the released pool does not include the exact sampled-pair manifest. We then recomputed retrieval on the retained queries.

At the 50% threshold, 155/2,207 queries remain:

| Scale | Model | R@1 | R@30 | MAP |
|---|---|---|---|---|
| 150M | ESM-2 | 0.303 | 0.477 | 0.199 |
| 150M | ProtSent | 0.329 | 0.581 | 0.319 |
| 35M | ESM-2 | 0.265 | 0.503 | 0.204 |
| 35M | ProtSent | 0.297 | 0.594 | 0.287 |

As a stricter sensitivity analysis, excluding a query when either AFDB or STRING has a hit at >=50% leaves 92 queries, only 57 of which have a non-self family positive. At 150M, R@1 ties (0.250 vs 0.250), while R@30 increases from 0.413 to 0.500 and MAP from 0.182 to 0.248. At 35M, R@1 is 0.207 vs 0.239, R@30 is 0.424 vs 0.533, and MAP is 0.171 vs 0.256. Paired bootstrap intervals include zero for both R@1 deltas, but exclude zero for R@30 and MAP at both scales. Thus, the strict subset does not support a robust top-1 claim, but it retains evidence for better deeper retrieval. We will report this narrower conclusion and the retained sample counts.

This completed sensitivity filters clean queries while retaining the fixed full gallery.

[[RESULT: insert the 40% and clean-query/clean-gallery results when complete. If not complete, delete this line and explicitly label the table above as query-filtered/full-gallery.]]

We also found that the submitted code evaluates 2,207 proteins using the SCOPe family field, not 100,000 proteins at the superfamily level. We will correct the description. A separate superfamily evaluation still improves at both scales.

For remote homology, the downstream split is disjoint in its evaluation hierarchy, which prevents direct task-label leakage but does not by itself exclude sequence exposure through the large fine-tuning sources. We therefore do not use the split alone as a leakage defense and will state this residual limitation.

**PPI.** This was already decontaminated in the original data pipeline: Bernett test proteins were added to the STRING sequence pool, MMseqs2 easy-linclust was run at 50% identity and 80% target coverage, and every STRING protein in a Bernett-containing cluster was removed before the final STRING pairs were constructed. The requested 40% analysis is therefore an additional sensitivity check, not a missing train/test control.

[[RESULT: insert the <40% PPI subset size, class balance, vanilla AUC, and ProtSent AUC when complete. Delete if not complete.]]

### 2. DMS objective

The requested ordering objective is the one implemented. Each row is a WT–mutant pair with normalized fitness. CoSENT ranks pair similarities: if mutant a has higher fitness than mutant b, it encourages sim(WT,a) > sim(WT,b). It does not assign an absolute similarity target or collapse all high-fitness variants together. We agree that this was not explained clearly enough in the main text.

The remaining limitation is that the default data are WT-anchored; mutant–mutant distances are not directly optimized.

### 3. MNRL batch and Eq. 1

The released paper-reproduction path uses CachedMultipleNegativesRankingLoss with a logical batch size of 1024. The loss is evaluated against the full logical batch; mini_batch_size=256 only partitions the forward/backward computation to reduce memory. Thus, on an MNRL step, each anchor is contrasted against the other 1023 positive-side examples in that source batch. Round-robin sampling means a step contains examples from one source, not a mixture of all sources.

The phrase "effective batch size" was ambiguous and will be replaced by the logical contrastive batch and cached mini-batch separately.

Eq. 1 is also malformed. The numerator should use the positive paired with anchor i, while the denominator ranges over the positive members of all N pairs. The superscript + denotes the positive member of a pair. We will correct the notation.

### 4. Pair-level tasks

For PPI, each partner is embedded independently and the two embeddings are concatenated before applying the same probe. This is implemented but omitted from the paper.

Peptide–HLA is not a two-input task in our submitted benchmark implementation: the dataset supplies one seq field, so no partner-combination operator is used there. We will make both points explicit.

### 5. k-NN regression

Regression uses scikit-learn KNeighborsRegressor(n_neighbors=3, metric="minkowski") without a weights argument; therefore it uses uniform averaging and Euclidean distance. We will specify this.

### 6. Ablations

We agree that the ablations do not establish the submitted choices as uniformly optimal. Removing hard negatives improves 20/23 tasks with mean relative change +7.9%, compared with 16/23 and +6.7% for the full 35M configuration. The per-task results show a trade-off rather than a universally better choice. Likewise, proportional sampling (+7.0%) is effectively comparable to round-robin (+6.7%). We will revise the interpretation and will not claim that either hard negatives or round-robin sampling is validated as generally superior.

### 7. Baselines

We added the matched MMseqs2 SCOPe baseline above. We are also running frozen logistic-regression/ridge probes on vanilla ESM-2 and ProtSent using identical splits.

[[RESULT: insert the completed linear/ridge aggregate and representative tasks.]]

We agree that ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, and related sentence-transformer work must be discussed more clearly. In particular, Redl et al.'s "Optimizing Protein Language Models with Sentence Transformers" is the closest methodological antecedent and will be compared explicitly rather than mentioned only briefly. We cannot produce reliable matched runs for all of these systems in the rebuttal window because they require different backbones, structural inputs, or indexing pipelines. We will not claim superiority to them; we will position ProtSent as a general-purpose sequence embedding and report specialized comparisons only where dataset, split, label level, and metric are matched.

### 8. Statistical evidence

We agree that several small Table 2 differences are not interpretable without uncertainty.

[[RESULT: paired-bootstrap task summary and representative confidence intervals. Do not describe an improvement as established when its interval includes zero.]]

The final reporting will separate supported improvements, supported degradations, and unresolved differences, and will use absolute deltas rather than only relative percentages. The mixed ablations and the stability/thermostability degradations will also be described as evidence that combining heterogeneous relations in one space creates real task-dependent trade-offs.

These checks resolve several reproducibility ambiguities and materially narrow the SCOPe claim on the strict clean subset. If these responses and completed results resolve the concerns, we would appreciate an updated assessment. If one concern remains decisive, please identify it during discussion so we can respond directly.
