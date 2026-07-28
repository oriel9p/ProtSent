# ProtSent — NeurIPS 2026 rebuttal (edited)

OpenReview: submission 28056. Each response must be **under 10,000 characters**.
**No links or attachments** in OpenReview. Every `[[RESULT: ...]]` placeholder must
be filled or deleted before posting. Post each response under its own review.

Status: this draft predates the current decontamination/retraining runs. Numbers
marked `[[RESULT]]` are still pending.

---

## Response to Reviewer HNXd

You asked us to separate two questions: whether ProtSent improves embedding neighborhoods, and how well a trained predictor can use those embeddings. That separation is the right one, and the results below are organized around it.

### 1. Direct retrieval and embedding-space organization

The geometry analysis you asked for was genuinely missing, and we have now run it. Table 3 reports cosine nearest-neighbor retrieval on SCOPe-40 (Recall@1/10/30), but retrieval numbers alone do not show whether the space itself is better organized. It is.

First, the reproduced retrieval comparison:

| Scale | Model | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|---|
| 150M | ESM-2 | 0.4237 | 0.5908 | 0.6457 | 0.3249 |
| 150M | ProtSent | 0.5066 | 0.6860 | 0.7245 | 0.4932 |
| 35M | ESM-2 | 0.3833 | 0.5841 | 0.6402 | 0.3235 |
| 35M | ProtSent | 0.4495 | 0.6529 | 0.7100 | 0.4225 |

The MAP gains show the effect is not confined to the first neighbor. A basic MMseqs2 nearest-neighbor baseline on the same gallery obtains family-level R@1=0.3539 and MAP=0.1795.

We then measured the geometry directly, in a residue-only mean-pooling audit on the same fixed gallery. At 150M, ProtSent moves family silhouette from -0.148 to 0.039, NMI from 0.852 to 0.893, ARI from 0.165 to 0.313, and the intra/inter-family distance ratio from 0.701 to 0.418. The Spearman correlation between embedding distance and shared SCOPe hierarchy depth strengthens from -0.247 to -0.561. The 35M model shows the same pattern. Not every metric improves: class-balanced alignment worsens. The supported claim is therefore improved global separation and hierarchical organization, not that every family becomes tighter.

The audit also surfaced a description error in the submission. The code evaluates the family field on 2,207 SCOPe sequences, whereas the text calls this superfamily retrieval and elsewhere states 100,000 sequences. We correct both. Under the same residue-only audit, a separate superfamily analysis still shows substantial gains: R@1 rises from 0.667 to 0.780 at 150M and from 0.639 to 0.726 at 35M.

### 2. Linear/ridge probes and downstream fine-tuning

Frozen logistic-regression and ridge probes are running now on the same splits, for vanilla ESM-2 and ProtSent. A conventional learned readout is the right way to contextualize the 3-NN probe, which was intended to measure neighborhood quality rather than to claim state-of-the-art task prediction.

[[RESULT: one compact sentence: number of completed tasks improved at 35M and 150M, then 3-5 representative absolute scores/deltas for remote homology, PPI, fluorescence, variant effect, stability.]]

[[RESULT: if representative LoRA/PEFT results finish, add one sentence. Otherwise omit. Do not imply full fine-tuning was completed.]]

A full end-to-end sweep over four encoders and 23 tasks does not fit the rebuttal window, and it would measure task adaptation rather than frozen representation quality. We do not present it as completed. The revision frames the two probes distinctly: trained heads measure downstream task adaptation, while 3-NN measures whether useful relationships are already local in the frozen embedding space. If the linear probes do not support the label-scarcity claim, we narrow that claim.

### 3. Confidence intervals and small changes

Sub-1% differences in Table 2 are not established improvements, and we stop presenting them as such.

[[RESULT: paired bootstrap summary: at each model scale, number of positive deltas whose 95% CI excludes zero, number negative, number unresolved. Add 3 representative intervals, including one large gain and one sub-1% change.]]

The revised reporting uses absolute metric-point deltas and marks unresolved differences as unresolved, rather than bolding every positive point estimate.

### 4. Few-shot variability and Table 5

Relative changes around near-zero baselines are misleading, and they are what produce cells such as -126.9%.

[[RESULT: 5- or 10-seed few-shot summary and absolute base/ProtSent values at N=100 and N=1000 for remote homology, one regression task, and one negative result. Include mean±SD.]]

[[RESULT: if few-shot linear-probe baselines finish, add the crossover/relative conclusion. Otherwise do not claim ProtSent is superior to a trained head under label scarcity.]]

Table 5 reports absolute scores and seed variability, with relative change secondary.

These analyses target the criteria you named for reconsidering the paper. If they resolve your concerns, we would appreciate an updated assessment. If one point remains decisive, tell us which one and we will address it during discussion.

---

## Response to Reviewer jVGf

You named two questions that would change your assessment: whether ProtSent contributes more than structural-information injection, and where it sits on the generality-accuracy trade-off. We answer both directly.

### 1. What is learned beyond structural supervision?

Structural supervision accounts for part of the gain, not all of it. In the submitted ablation, removing AFDB reduces the mean relative gain from +6.7% to +3.2%, the number of improved tasks from 16/23 to 13/23, and the remote-homology gain from +40.5% to +15.3%. Some structural-task gains are indeed attributable to AFDB/Foldseek supervision.

The remaining ablations show the model is not only AFDB structure injection:

- Without AFDB, ProtSent still improves 13/23 tasks.
- Without Pfam, it improves 15/23 tasks, with mean +4.6%.
- Removing STRING changes PPI from +5.3% to -0.5%, while most other tasks remain similar.
- Removing DMS primarily reduces fitness-related gains, including fluorescence.

Each source leaves a distinct fingerprint on a distinct task family. Those source-specific effects are the contribution we claim: a sequence-level metric space jointly shaped by evolutionary family, structural-cluster, physical-interaction, and fitness-order relations, that is, multi-relation sequence-level metric learning rather than structure-focused representation enrichment alone. The related-work discussion will position ProtSent alongside ESM-S, S-PLM, ISM, Magneton, SaProt, and ProSST, without claiming superiority to them.

[[RESULT: if the joint no-AFDB/no-Pfam ablation finishes, insert completed-task count and representative structural, PPI, fitness results. Otherwise delete this line; the single-source ablations above already support the narrower claim.]]

Applying ProtSent to SaProt or ProSST is not a backbone substitution at the data level: their inputs require residue-level structure-derived tokens for the large Pfam and STRING training corpora. Preparing those inputs falls outside the rebuttal window, so we neither present an unreliable comparison nor promise that result as completed.

### 2. Generality-accuracy trade-off

We added a directly matched sequence-search reference on SCOPe. Self hits are removed; absent hits count as failures.

| Method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 nearest-neighbor search | 0.3539 | 0.3856 | 0.3856 | 0.1795 |
| ESM-2 150M | 0.4237 | 0.5908 | 0.6457 | 0.3249 |
| ProtSent 150M | 0.5066 | 0.6860 | 0.7245 | 0.4932 |

This is a basic MMseqs2 nearest-neighbor baseline, not an optimized profile-search system, and we infer nothing from it about HMMER, Foldseek, ProtTucker, PLMSearch, DHR, or ProTrek. Those systems use different inputs, indexes, training objectives, or benchmark protocols. The trade-off we expect and will state as the paper's scope: a specialized retrieval system may perform better on its target retrieval problem, while ProtSent provides one frozen sequence embedding that also serves classification, regression, PPI, and fitness tasks.

[[RESULT: insert the frozen linear/ridge summary if available, because it further quantifies how broadly the representation transfers beyond nearest-neighbor retrieval.]]

### 3. CoSENT on DMS data

The implemented objective is ordinal over WT-mutant similarities, not a regression onto absolute cosine values. Each training row is a (wild type, mutant) pair with a within-assay normalized fitness score. CoSENT compares pairs within a batch: when pair p has a higher fitness score than pair q, the loss encourages the WT-mutant similarity of p to exceed that of q. It assigns no absolute similarity target and does not pull all high-fitness mutants to one point.

The limitation is narrower than stated in the review: the submitted configuration is WT-anchored and does not directly optimize mutant-mutant geometry. We state that design choice explicitly.

We add the missing Heinzinger et al. citation, plus the structure-informed and specialized-retrieval work you identified.

You indicated that clarifying these two axes could raise your score to accept. The ablations, the matched MMseqs2 reference, and the clarified scope answer them without overstating what we have not run. If a specific remaining comparison is essential to your assessment, identify it during discussion and we will respond.

---

## Response to Reviewer Yi1G

Eight concerns, answered in the order raised.

### 1. AFDB/SCOPe and STRING/PPI overlap

**SCOPe.** You are right that noting possible overlap was insufficient, so we ran the audit. The original AFDB preparation filters by pLDDT and fragment status and assigns Foldseek clusters, but it does not decontaminate against SCOPe. We searched every SCOPe query against the released AFDB training-source pool with MMseqs2 at 80% query coverage, then recomputed retrieval on the retained queries. This is a source-overlap audit, conservative rather than proof of exact checkpoint exposure, because the released pool does not include the exact sampled-pair manifest.

At the 50% threshold, 155/2,207 queries remain:

| Scale | Model | R@1 | R@30 | MAP |
|---|---|---|---|---|
| 150M | ESM-2 | 0.303 | 0.477 | 0.199 |
| 150M | ProtSent | 0.329 | 0.581 | 0.319 |
| 35M | ESM-2 | 0.265 | 0.503 | 0.204 |
| 35M | ProtSent | 0.297 | 0.594 | 0.287 |

As a stricter sensitivity analysis, excluding a query when either AFDB or STRING has a hit at >=50% leaves 92 queries, only 57 of which have a non-self family positive. At 150M, R@1 ties (0.250 vs 0.250), while R@30 increases from 0.413 to 0.500 and MAP from 0.182 to 0.248. At 35M, R@1 is 0.207 vs 0.239, R@30 is 0.424 vs 0.533, and MAP is 0.171 vs 0.256. Paired bootstrap intervals include zero for both R@1 deltas but exclude zero for R@30 and MAP at both scales. The strict subset therefore does not support a robust top-1 claim, while it does retain evidence for better deeper retrieval. We report that narrower conclusion and the retained sample counts.

This completed sensitivity filters clean queries while retaining the fixed full gallery.

[[RESULT: insert the 40% and clean-query/clean-gallery results when complete. If not complete, delete this line and explicitly label the table above as query-filtered/full-gallery.]]

We also found that the submitted code evaluates 2,207 proteins using the SCOPe family field, not 100,000 proteins at the superfamily level. We correct the description. A separate superfamily evaluation still improves at both scales.

For remote homology, the downstream split is disjoint in its evaluation hierarchy, which prevents direct task-label leakage but does not by itself exclude sequence exposure through the large fine-tuning sources. We do not use the split alone as a leakage defense, and we state that residual limitation.

**PPI.** Already decontaminated in the original data pipeline: Bernett test proteins were added to the STRING sequence pool, MMseqs2 easy-linclust was run at 50% identity and 80% target coverage, and every STRING protein in a Bernett-containing cluster was removed before the final STRING pairs were constructed. The requested 40% analysis is an additional sensitivity check, not a missing train/test control.

[[RESULT: insert the <40% PPI subset size, class balance, vanilla AUC, and ProtSent AUC when complete. Delete if not complete.]]

### 2. DMS objective

The ordering objective you requested is the one implemented. Each row is a WT-mutant pair with normalized fitness. CoSENT ranks pair similarities: if mutant a has higher fitness than mutant b, it encourages sim(WT,a) > sim(WT,b). It assigns no absolute similarity target and does not collapse all high-fitness variants together. The main text did not explain this clearly enough, and the revision does.

The remaining limitation is that the default data are WT-anchored, so mutant-mutant distances are not directly optimized.

### 3. MNRL batch and Eq. 1

Each anchor is contrasted against the other 1023 positive-side examples in its source batch. The released paper-reproduction path uses CachedMultipleNegativesRankingLoss with a logical batch size of 1024, and the loss is evaluated against the full logical batch; mini_batch_size=256 only partitions the forward/backward computation to reduce memory. Round-robin sampling means a step contains examples from one source, not a mixture of all sources.

The phrase "effective batch size" was ambiguous. The revision reports the logical contrastive batch and the cached mini-batch separately.

Eq. 1 is malformed, as you note. The numerator should use the positive paired with anchor i, the denominator ranges over the positive members of all N pairs, and the superscript + denotes the positive member of a pair. We correct the notation.

### 4. Pair-level tasks

For PPI, each partner is embedded independently and the two embeddings are concatenated before the same probe is applied. This is implemented but omitted from the paper. Peptide-HLA is not a two-input task in our submitted benchmark implementation: the dataset supplies one seq field, so no partner-combination operator is used there. Both points become explicit.

### 5. k-NN regression

Regression uses scikit-learn KNeighborsRegressor(n_neighbors=3, metric="minkowski") with no weights argument, so it uses uniform averaging and Euclidean distance. We specify this.

### 6. Ablations

The ablations do not establish the submitted choices as uniformly optimal, and we will not claim they do. Removing hard negatives improves 20/23 tasks with mean relative change +7.9%, against 16/23 and +6.7% for the full 35M configuration. Proportional sampling (+7.0%) is effectively comparable to round-robin (+6.7%). The per-task results show a trade-off rather than a universally better choice. The revision drops any claim that hard negatives or round-robin sampling is validated as generally superior.

### 7. Baselines

The matched MMseqs2 SCOPe baseline is above. Frozen logistic-regression and ridge probes on vanilla ESM-2 and ProtSent, using identical splits, are running.

[[RESULT: insert the completed linear/ridge aggregate and representative tasks.]]

ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, and related sentence-transformer work need clearer discussion. Redl et al., "Optimizing Protein Language Models with Sentence Transformers," is the closest methodological antecedent, and the revision compares it explicitly rather than mentioning it in passing. Reliable matched runs for all these systems do not fit the rebuttal window, since they require different backbones, structural inputs, or indexing pipelines. We claim no superiority to them. We position ProtSent as a general-purpose sequence embedding and report specialized comparisons only where dataset, split, label level, and metric are matched.

### 8. Statistical evidence

Several small Table 2 differences are not interpretable without uncertainty, and the revision treats them that way.

[[RESULT: paired-bootstrap task summary and representative confidence intervals. Do not describe an improvement as established when its interval includes zero.]]

The final reporting separates supported improvements, supported degradations, and unresolved differences, and uses absolute deltas rather than only relative percentages. The mixed ablations and the stability/thermostability degradations are described as what they are: evidence that combining heterogeneous relations in one space creates real task-dependent trade-offs.

These checks resolve several reproducibility ambiguities and materially narrow the SCOPe claim on the strict clean subset. If they resolve your concerns, we would appreciate an updated assessment. If one concern remains decisive, identify it during discussion and we will respond directly.
