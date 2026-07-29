ProtSent: Protein Sentence Transformers
https://openreview.net/forum?id=yfGA2cCxJB#discussion


High level: Additional analysis/results needed:

https://github.com/ddofer/ProteinSentenceTransformers/tree/agent/protsent-rebuttal-experiments

---------------------
Before posting
Each response below is under the NeurIPS 2026 limit of 10,000 characters.
Do not include links or attachments in OpenReview. Reviewer only have access to rebuttal, and original paper, not revised paper and results.
Replace [[RESULT: ...]] lines with the completed result, or delete that line before posting.
Keep the responses separate: post each under the corresponding review. The final AC note is optional. Goal is paper acceptance, preferably while minimizing extra changes (beyond those done), but we can and will add changes + tests + experiments as needed if it is really important (e.g. the mmseqs2 baseline)
Internal validation notes — do not paste into OpenReview
data_prep.py::prep_afdb filters AFDB by pLDDT/fragment status and assigns Foldseek clusters, but contains no SCOPe decontamination.
The STRING pipeline explicitly decontaminates against Bernett at 50% identity and 80% target coverage before constructing the training pairs.
benchmark_tasks.py defines SCOPe retrieval using the family field; PPI has two sequence fields, whereas peptide–HLA has one seq field.
protein_benchmark_suite.py concatenates the two PPI embeddings. Its 3-NN regression constructor does not pass weights, so scikit-learn uses uniform weighting.
The released reproduction command uses cached MNRL with batch_size=1024; the loss constructor uses mini_batch_size=256.
Existing numerical results come from INTERIM_RESULTS.md and SCOPE_AFDB_STRING_50_RESULTS.md. The latter treats the released source pools as a proxy.


-------------------------------------------

https://chatgpt.com/share/6a65c3e2-68e8-83ed-be13-2f9f8cee3268
Additional detail included in watsap sent to oriel. 
TL;DR — tasks for the coauthor
Finalize the SCOPe full-set reproduction table
Record paper versus reproduced results for vanilla ESM-2 and ProtSent at 35M and 150M.
Report R@1, R@10, R@30 and MAP.
Use the existing cached embeddings/results if available; do not rerun unless necessary.
Correct the paper description:
SCOPe contains 2,207 sequences, not 100,000.
Submitted Table 3 uses family labels, not superfamily labels.
Run the primary SCOPe leakage analysis against AFDB
AFDB was not filtered against SCOPe in the original preparation code; prep_afdb builds structural clusters but performs no SCOPe decontamination.
Filter SCOPe sequences with an AFDB hit at:
≥50% identity and  ≥80% query coverage
Evaluate (optionally both both):
clean queries against the full gallery;
clean queries against a clean gallery — preferred stricter analysis.
Report:
retained queries/gallery sequences;
eligible queries with a non-self family match;
singleton count;
R@1/R@10/R@30/MAP;
ProtSent–vanilla absolute delta;
paired bootstrap 95% CI.(?)
Maybe also do same for remote homology benchmark - only if they asked for that? 
Finalize the PPI/STRING reproduction table
We already did filtering of string by 50% sim to PPI test benchmark! (the reviewer may have missed it, or just wants more filter - 40% is not that different). 
Record paper versus reproduced Bernett PPI results for all four models.
Include test-pair count, split, probe and AUC.
State clearly that STRING was already filtered against Bernett (PPI benchmark) at 50% identity and 80% coverage in the original pipeline.
A stricter 40% analysis is optional:
retain a pair only when both proteins are below 40% identity to STRING;
evaluate the existing fitted probe without retraining;
report remaining proteins, pairs, class balance and AUC.
Organize the existing MMseqs2 SCOPe baseline
MMseqs2 as baseline
Do not rerun unless artifacts are missing.
Put vanilla ESM-2, ProtSent and MMseqs2 in one table.
Report all-query and eligible-query R@1/R@10/R@30/MAP.
Confirm:
self hits removed;
ranking by E-value, then bitscore.
Describe it as a basic nearest-neighbor MMseqs2 baseline, not an optimized profile-search comparison.
Optional: run it on other benchmarks that are relevant: remote homology at the least. Include results in results table.
Run frozen linear/ridge probes on feasible benchmarks
Models:
vanilla ESM-2 35M;
ProtSent 35M;
vanilla ESM-2 150M;
ProtSent 150M.
Use one model per GPU where possible.
Reuse the existing benchmark code:
protein_benchmark_suite.py
benchmark_tasks.py
benchmark_utils.py
Existing code already implements:
standardized logistic regression for classification;
one-vs-rest logistic regression for multiclass;
standardized Ridge(α=1\alpha=1α=1) for regression.
Skip:
ProteinGym;
EC unless the submitted multiclass protocol can be restored cheaply. (Multilabel is slow; and benchmark was broken with wrong labels/classes in original paper/results)
Report per task:
vanilla score;
ProtSent score;
absolute delta;
whether the direction matches the submitted 3-NN result.
Summarize tasks improved/degraded/tied and median signed delta.
Optional: Validate the linear-probe environment with a small k-NN anchor rerun
Use cached embeddings.
Check:
remote homology;
fluorescence;
Investigate discrepancies >≈0.05 before trusting the linear results.
LoRA/PEFT only after the linear table is complete
Optional, not required.
Limit to 3–4 representative tasks:
remote homology;
Stability
PPI or solubility;
fluorescence;
Do not run LoRA across the full benchmark suite.
Make the easy paper/method corrections
Add/note “absolute” score differences (can be in appendix?)
SCOPe: 2,207 sequences; family retrieval; query/gallery convention; singleton handling.
Add MAP and (opt or appendix) retained-sample counts.
Clarify hyperparameters:
State PPI embedding combination: concatenate the two partner embeddings.
State whether k-NN regression uses uniform or distance weighting.
Clarify the actual MNRL negative pool versus optimizer/effective batch size.
State truncation, pooling and special-token handling exactly.
Explain CoSENT as ranking WT–mutant similarities by fitness.
Reframe geometry as improved global separation/hierarchy, not uniform tightening of every family.
 missing reference on line 21 ("?" between "Lin et al., 2023" and "Henzinger et al., 2022") 

Make https://huggingface.co/datasets/GrimSqueaker/protsent-data have correct pretraining dataset ? (for reproducibility + so Dan can more easily handle extra compute without worrying about differences). - See that it’s filtered?. 
Do not start
New ProtSent training.
Structural-model evaluations such as SaProt, S-PLM, ESM-S or ISM. (Out of scope)
More geometry metrics.
ProteinGym reruns.
Full-benchmark LoRA.

Up to you:
 Bootstrapping/CI of results? (use existing results). 
Few-shot evaluation - they asked about random seeds? 
Opt: Add a citation to a few of the papers reviewers mentioned maybe, to make them like us (saprot, “A trimodal protein language” ; if relevant and comparable). 
-----------
Reviews + Rebuttals
(Rebuttals are DRAFT and lack extra runs/results like the leakage concern. Also, no rebuttal needed for AC). 

Meta Review of Submission 28056 by Area Chair 
Metareview:
Summary
The paper presents ProtSent, a contrastive fine-tuning framework that adapts frozen sequence-only protein language models (ESM-2 35M and 150M) into general-purpose embedding models. Training combines MultipleNegativesRankingLoss over five paired data sources (Pfam families, Pfam-derived hard negatives, AlphaFold DB/Foldseek structural clusters, STRING-DB PPI pairs) with a CoSENT ranking loss over ProteinGym DMS/clinical variant data. The authors evaluate the resulting frozen embeddings with a k-NN probe across 23 downstream tasks plus SCOPe-40 structural retrieval and few-shot transfer, reporting improvements on 15 to 16 of 23 tasks, including large relative gains in remote homology detection and structural retrieval.
Main strengths
Reviewers agreed the problem is well-motivated and practically relevant given the growing use of pLMs in drug discovery pipelines. The evaluation is broad (23 tasks, multiple properties), the dataset construction is described in enough detail to support reproducibility of the data pipeline itself, and reviewer HNXd noted the authors' explicit discussion of leakage as a positive rather than an oversight to be uncovered later.
Main weaknesses
Across reviews, weaknesses cluster into a few recurring themes:
Potential train-test leakage (Yi1G). AFDB training sequences do not appear to be filtered against SCOPe test domains, which directly affects the headline SCOPe-40 retrieval numbers. PPI train-test overlap is addressed only at the sequence-identity level the authors chose, without a stricter threshold analysis (e.g., 40 to 50 percent identity).
Evidentiary mismatch between the claimed contribution and the evaluation (HNXd, Yi1G). The central claim is that contrastive fine-tuning "restructures the embedding space", but the evaluation proxies this only through k-NN classification accuracy rather than retrieval precision, clustering quality, or explicit distance-to-similarity correlation. Combined with this, several reported gains are small in absolute terms, and there are no confidence intervals, bootstrapped or otherwise, and no multi-seed variance analysis for the few-shot results, so it is hard to tell which improvements are real.
Baseline adequacy (all reviewers). No comparison to a linear probe or fully fine-tuned baseline on top of frozen ESM-2 (needed to support the label-scarcity claim), and no comparison to existing structure-informed sequence models (ESM-S, S-PLM, ISM, Magneton) or specialized retrieval systems (ProtTucker, Foldseek, PLMSearch, ProTrek, DHR). Reviewer jVGf specifically raises the concern that gains may simply reflect injecting structural information into a sequence-only model rather than a novel embedding-organization effect, which is currently unaddressed.
Under-specified methodology (Yi1G). The effective batch size and in-batch negative construction for MNRL is ambiguous, the pair-combination method for pair-level tasks (PPI, peptide-HLA) is not described, and the k-NN regression averaging scheme (uniform vs. distance-weighted) is unspecified. Reviewer Yi1G also flags that the ablations do not clearly support the default design choices (removing Pfam hard negatives improves aggregate performance; round-robin vs. proportional sampling are comparable).
Table 5 reporting (HNXd). Relative-improvement-only reporting makes it impossible to judge whether large percentage gains reflect meaningful absolute performance.
Prioritized weaknesses for the rebuttal
All the weaknesses listed above should be prioritized. If these concerns turn out to be well-founded and not resolvable, I would not expect a rebuttal to overcome them; if they are resolved or shown to be factually incorrect, this could become a considerably stronger paper.-----------------------------------------------------------------------------------
Official Review of Submission28056 by Reviewer HNXd
Summary:
The paper proposes ProtSent, a contrastive fine-tuning approach that aims to improve the sequence-level embeddings of protein language models. The authors train ESM2-35M and ESM2-150M with MultipleNegativesRankingLoss across five protein-pair datasets, so that proteins with related functional or structural properties are closer in embedding space. They evaluate the resulting frozen embeddings with a k-NN probe across 23 downstream tasks and report improvements on 15 to 16 tasks, including large gains in remote homology detection.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:
Protein language models are becoming increasingly useful in the drug discovery pipeline, making this work both interesting and valuable.
The paper is well written.
The evaluation covers a relatively broad range of tasks and properties.
I appreciate that the authors acknowledge and discuss leakage controls.
Weaknesses:
In my opinion, the paper currently sits between two narratives. Contrastive learning is used to better organize proteins in the embedding space, so that proteins with similar properties are closer together (From the paper: "Contrastive fine-tuning restructures the embedding space to better capture protein function and structure"). This is useful for "retrieval, clustering, and few-shot" transfer, as mentioned in the first paragraph. However, the authors do not directly evaluate retrieval or clustering (retrieval precision@k, silhouette score, or correlation between embedding distance and property similarity). Instead, they use downstream property prediction with k-NN. This is a reasonable proxy, but the paper mainly discusses the performance of the k-NN model rather than using k-NN as a probe of the embedding space and the local/global arrangement of proteins. The issue is that the reported numbers are much lower than those in the literature for the same models, likely due to the use of k-NN. For instance, Stability in BIOMAP reports 58.8% for the base model, while the original paper reports 69.08% with a linear classifier and 77.69% with LoRA. Later in the discussion, the authors mention: "We do not compare to specialized retrieval systems [...] our goal is general-purpose embeddings rather than retrieval-optimized models". I would find it better to either provide a retrieval and clustering evaluation or to provide an extensive analysis of how the embedding space changes after contrastive training.
The claim of improved performance under label scarcity needs a stronger baseline. What would a simple linear classifier on top of the base model achieve? What about fine-tuning the baseline? Comparing only to k-NN does not reflect how practitioners typically use ESM2. I think a stronger framing would be that, when labeled data is scarce, standard linear classifiers and fine-tuning pipelines degrade substantially, while k-NN remains competitive because of contrastive alignment. However, this claim needs to be supported by the relevant baselines.
My main concern is the lack of statistical tests, confidence intervals, or standard deviations. Some improvements are below 1%, and some of these benchmarks are known to be noisy. I appreciate that the authors cover a broad range of tasks and properties, but I would like to see 95% confidence intervals for the reported metrics, computed by bootstrapping over individual predictions.
In addition to confidence intervals, I would also like to see a variability analysis with multiple random seeds for the few-shot evaluation, especially since the authors acknowledge that the results are noisy.
For Table 5, reporting relative improvements without absolute numbers makes the results hard to interpret. A +244% improvement from a very low baseline may still correspond to a small absolute gain, so I would like the author to report the absolute scores as well.
Overall, I like the approach. Improving protein sequence-level embeddings is valuable, and the authors have put significant effort into curating and processing the datasets. However, the paper currently sits between two narratives, and the evaluation would be much stronger with confidence intervals, variability analysis, and statistical tests. I would consider increasing my score if the authors provide these additional analyses.
Quality: 2: not good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Can the authors provide either a direct retrieval/clustering evaluation, or an analysis showing how ProtSent changes the local and global organization of the protein embedding space?
Can the authors compare ProtSent to a linear classifier on top of the (frozen and fine-tuned) base model, especially in the label-scarce setting?
Can the authors compute 95% confidence intervals for the reported metrics by bootstrapping over individual predictions?
Can the authors provide a variability analysis with multiple random seeds for the few-shot evaluation?
Can the authors report absolute scores in Table 5 in addition to relative improvements?
Limitations:
The authors acknowledge the main limitations.
Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 4



Rebuttal 1:
Response to Reviewer HNXd
Thank you for the constructive review and for stating which analyses would change your assessment. We agree that the paper should separate two questions more clearly: whether ProtSent improves embedding neighborhoods, and how well a trained downstream predictor can use those embeddings.
1. Direct retrieval and embedding-space organization
One clarification is important: the submission already includes a direct retrieval experiment in Table 3—cosine nearest-neighbor retrieval on SCOPe-40 with Recall@1/10/30. The missing part was a broader analysis of the geometry and clustering, which we have now added.
We first reproduced the submitted retrieval comparison:
Scale
Model
R@1
R@10
R@30
MAP
150M
ESM-2
0.4237
0.5908
0.6457
0.3249
150M
ProtSent
0.5066
0.6860
0.7245
0.4932
35M
ESM-2
0.3833
0.5841
0.6402
0.3235
35M
ProtSent
0.4495
0.6529
0.7100
0.4225
The MAP gains show that the effect is not confined to the first neighbor. A basic MMseqs2 nearest-neighbor baseline on the same gallery obtains family-level R@1=0.3539 and MAP=0.1795.
In a separate residue-only mean-pooling audit on the same fixed gallery, we also measured the geometry directly. At 150M, ProtSent changes family silhouette from -0.148 to 0.039, NMI from 0.852 to 0.893, ARI from 0.165 to 0.313, and the intra/inter-family distance ratio from 0.701 to 0.418. The Spearman correlation between embedding distance and shared SCOPe hierarchy depth strengthens from -0.247 to -0.561; the 35M model shows the same pattern. Not every metric improves: class-balanced alignment worsens, so the supported claim is improved global separation and hierarchical organization—not that every family becomes tighter.
Our audit also found a description error: the submitted code evaluates the family field on 2,207 SCOPe sequences, whereas the text calls this superfamily retrieval and elsewhere states 100,000 sequences. We will correct both. Under the same residue-only audit, a separate superfamily analysis still shows substantial gains: R@1 increases from 0.667 to 0.780 at 150M and from 0.639 to 0.726 at 35M.
2. Linear/ridge probes and downstream fine-tuning
We agree that a conventional learned readout is needed to contextualize the 3-NN probe. The 3-NN results were intended to measure neighborhood quality, not to claim state-of-the-art task prediction. We are therefore running frozen logistic-regression and ridge probes on the same splits for vanilla ESM-2 and ProtSent.
[[RESULT: Insert one compact sentence: number of completed tasks improved at 35M and 150M, followed by 3–5 representative absolute scores/deltas for remote homology, PPI, fluorescence, variant effect, and stability.]]
[[RESULT: If representative LoRA/PEFT results finish, add one sentence. Otherwise omit. Do not imply that full fine-tuning was completed.]]
A full end-to-end sweep over four encoders and 23 tasks is not feasible during rebuttal and would evaluate task adaptation rather than frozen representation quality. We therefore do not present it as completed. We will frame the available results accordingly: trained heads measure downstream task adaptation; 3-NN measures whether useful relationships are already local in the frozen embedding space. We will also narrow any label-scarcity claim if the linear-probe results do not support it.
3. Confidence intervals and small changes
We agree that several sub-1% differences in Table 2 should not be treated as established improvements.
[[RESULT: Insert paired bootstrap summary: at each model scale, number of positive deltas whose 95% CI excludes zero, number negative, and number unresolved. Add 3 representative intervals, including one large gain and one sub-1% change.]]
The revised reporting will use absolute metric-point deltas and mark unresolved differences as such, rather than bolding every positive point estimate.
4. Few-shot variability and Table 5
We agree on both points. Relative changes around near-zero baselines are misleading and explain cells such as -126.9%.
[[RESULT: Insert 5- or 10-seed few-shot summary and absolute base/ProtSent values at N=100 and N=1000 for remote homology, one regression task, and one negative result. Include mean±SD.]]
[[RESULT: If few-shot linear-probe baselines finish, add the crossover/relative conclusion here. Otherwise do not claim that ProtSent is superior to a trained head under label scarcity.]]
Table 5 will report absolute scores and seed variability; relative changes will be secondary.
The new analyses directly address the criteria you identified for reconsidering the paper. If these results resolve your concerns, we would appreciate an updated assessment; if one point remains decisive, please indicate which one so we can address it during discussion.


-------------------------------------------


Official Review of Submission28056 by Reviewer jVGf
Summary:
The authors propose a method for incorporating structural and functional information into sequence-only protein language models via contrastive learning. The datasets used cover substructural similarity (Pfam domains), structural similarity (AlphaFold DB FoldSeek clusters), protein-protein interactions (STRING-DB), and functional effects of point mutations (ProteinGym). These datasets are used in a round-robin fashion, sampling a batch from each dataset for contrastive tuning of a base model. The authors test their method on two sequence-only models from the same family but different scales (ESM-2 35M and 150M). The effects of the method are assessed using a KNN probe on the resulting embedding space with labels from a variety of downstream tasks, mirroring the retrieval-based setting the authors are interested in. The results are mixed, with the contrastive tuning improving performance for some tasks and hurting for others, but with a net positive improvement when considered across all tasks.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
The goal of tuning PLMs to have an embedding space that captures a range of meaningful types of similarities for protein pairs is interesting and could be broadly useful
Models are tested on a wide range of downstream tasks capturing different aspects of protein biology
Dataset construction is described in depth, which is great for both understanding exact steps as well as reproducibility
Weaknesses
Since the results only use sequence-only models, I wonder if a lot of the performance improvements here are driven by injecting structural or substructural information into the sequence-only model. While this isn't inherently a bad thing, there's a fair amount of previous work in this space (see ESM-S [1], S-PLM [2], ISM [3], Magneton [4]), and I feel this work needs to be positioned in this context.
Somewhat similar to the point above, there's a lack of comparison to other approaches. The limitations section states that "We do not compare to specialized retrieval systems...our goal is general-purpose embeddings rather than retrieval-optimized models, but such comparisons would help quantify the generality–accuracy trade-off." I wholly agree and feel that understanding this generality-accuracy trade-off is a very important missing piece for a reader to decide whether to use this method over a more specialized method.
References:
Zhang et al., "Structure-Informed Protein Language Model", 2024, https://doi.org/10.48550/arXiv.2402.05856
Wang et al., "S-PLM: Structure-Aware Protein Language Model via Contrastive Learning Between Sequence and Structure", 2025, Advanced Science
Ouyang-Zhang et al., "Distilling Structural Representations into Protein Sequence Models", 2025, ICLR
Calef et al., "Greater than the Sum of Its Parts: Building Substructure into Protein Encoding Models", 2026, ICLR
Quality: 2: not good
Clarity: 3: good
Significance: 3: good
Originality: 2: not good
Questions:
Related to weakness 1, answering the following questions could help understand how much of the effects of ProtSent come from just injecting structural information into sequence-only models:
How do the results hold up in absence of both AFDB and Pfam?
How do the results change if the ProtSent framework is applied to a sequence-structure model such as SaProt or ProSST? I believe SaProt uses a very similar architecture/codebase to ESM-2, so may be easy to drop into existing code
Related to weakness 2:
I understand this may be hard to perform in the short rebuttal period, but how does ProtSent compare to more specialized methods? If ProtSent slightly loses out, but is able to perform a much broader range of tasks, then this could still be a favorable comparison for ProtSent and help strengthen the narrative. ProTrek [1] is also a good model to cite and possibly compare to.
How exactly does the CoSENT loss for DMS data work? It seems like the ranked labels example in the original CoSENT paper still refers to pairs of sentences, which are ranked based on their pairwise similarity. In this case, I don't understand how mutants within a DMS assay are paired and what value is used for their similarity.
Overall
I think there are interesting ideas here, but it's hard to understand how useful they are for others due to the lack of contextualization with other work. If the authors are able to help provide insight along two axes:
Either why ProtSent goes beyond adding structure information to sequence models or why ProtSent is superior to existing methods
Where this method sits in the "generality-accuracy" trade-off they've described then I would be happy to increase my score to an accept, as this could contribute valuable insight for the community.
Minor notes
There appears to be a missing reference on line 21 ("?" between "Lin et al., 2023" and "Henzinger et al., 2022")
References
[1] Su, J., He, Y., You, S. et al. A trimodal protein language model enables advanced protein searches. Nat Biotechnol (2025). https://doi.org/10.1038/s41587-025-02836-0
Limitations:
I believe the authors have identified limitations (e.g. lack of comparison to other methods) but further steps need to be taken to adequately address them.
Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 4


Rebuttal 2:
Response to Reviewer jVGf
Thank you for identifying the two questions that would change your assessment: whether ProtSent contributes more than structural-information injection, and where it lies on the generality–accuracy trade-off.
1. What is learned beyond structural supervision?
Structural supervision is important, and we should state that plainly. In the submitted ablation, removing AFDB reduces the mean relative gain from +6.7% to +3.2%, the number of improved tasks from 16/23 to 13/23, and the remote-homology gain from +40.5% to +15.3%. Thus, some structural-task gains are indeed attributable to AFDB/Foldseek supervision.
However, the ablations also show that the method is not only an AFDB structure-injection model:
Without AFDB, ProtSent still improves 13/23 tasks.
Without Pfam, it improves 15/23 tasks, with mean +4.6%.
Removing STRING changes PPI from +5.3% to -0.5%, while most other tasks remain similar.
Removing DMS primarily reduces fitness-related gains, including fluorescence.
These source-specific effects are the intended contribution: a sequence-level metric space jointly shaped by evolutionary family, structural-cluster, physical-interaction, and fitness-order relations. This differs from the specific contribution we claim: multi-relation, sequence-level metric learning rather than structure-focused representation enrichment alone. We will revise the related-work discussion to position ProtSent alongside ESM-S, S-PLM, ISM, Magneton, SaProt, and ProSST, without claiming superiority to them.
[[RESULT: If the joint no-AFDB/no-Pfam ablation finishes, insert its completed-task count and representative structural, PPI, and fitness results here. Otherwise delete this line; the single-source ablations above already support the narrower claim.]]
Applying ProtSent to SaProt or ProSST is not a simple backbone substitution at the data level: their inputs require residue-level structure-derived tokens for the large Pfam and STRING training corpora. Preparing those inputs is outside the rebuttal window. We therefore do not present an unreliable comparison or promise that result as completed.
2. Generality–accuracy trade-off
We added a directly matched sequence-search reference on SCOPe. Self hits are removed; absent hits count as failures.
Method
R@1
R@10
R@30
MAP
MMseqs2 nearest-neighbor search
0.3539
0.3856
0.3856
0.1795
ESM-2 150M
0.4237
0.5908
0.6457
0.3249
ProtSent 150M
0.5066
0.6860
0.7245
0.4932
This is a basic MMseqs2 nearest-neighbor baseline, not an optimized profile-search system. We do not infer superiority to HMMER, Foldseek, ProtTucker, PLMSearch, DHR, or ProTrek from this table. Those systems use different inputs, indexes, training objectives, or benchmark protocols. A fair expected trade-off is that a specialized retrieval system may perform better on its target retrieval problem, whereas ProtSent provides one frozen sequence embedding that can also be used across classification, regression, PPI, and fitness tasks. We will state this as the paper’s scope rather than leave it implicit.
[[RESULT: Insert the frozen linear/ridge summary here if available, because it further quantifies how broadly the representation transfers beyond nearest-neighbor retrieval.]]
3. CoSENT on DMS data
Each training row is a (wild type, mutant) pair with a within-assay normalized fitness score. CoSENT does not regress that score to an absolute cosine value and does not simply pull all high-fitness mutants to one point. It compares pairs within a batch: when pair (p) has a higher fitness score than pair (q), the loss encourages the WT–mutant similarity of (p) to exceed that of (q). Thus, the implemented objective is ordinal over WT–mutant similarities.
The limitation is narrower: the submitted configuration is WT-anchored and does not directly optimize mutant–mutant geometry. We will state this design choice explicitly.
We also correct the missing Heinzinger et al. citation and add the structure-informed and specialized-retrieval work you identified.
You indicated that clarifying these two axes could raise your score to accept. We believe the ablations, matched MMseqs2 reference, and clarified scope answer them without overstating what has not been run. If a specific remaining comparison is essential to your assessment, please identify it during discussion.

-------------------------------------------

Official Review of Submission28056 by Reviewer Yi1G

Summary:
This paper introduces ProtSent, a contrastive fine-tuning framework for adapting pretrained pLMs into general purpose embedding models. The authors use ESM-2 35M and 150M as backbones, obtain sequence embeddings by mean pooling residue representations, and fine-tune the models using multiple datasets, including Pfam family pairs, Pfam-derived hard negatives, AlphaFold DB / Foldseek structural clusters, STRING-DB PPI pairs, and ProteinGym DMS / clinical variant data. The main training objective is MultipleNegativesRankingLoss for pair-based data sources, while DMS data are incorporated through a CoSENT ranking loss over continuous fitness scores. The fine-tuned models are evaluated as frozen embedding encoders using KNN probes across 23 downstream tasks, SCOPe-40 structural retrieval, and few-shot KNN evaluation. The results demonstrate that contrastive fine-tuning can improve the neighborhood structure of protein embeddings, especially for remote homology, structural retrieval, PPI prediction, and some fitness-related tasks.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:
The paper focused on an important problem: converting pretrained pLMs into better sequence-level embedding models for retrieval, clustering, and nearest-neighbor transfer. The high-level idea of applying contrastive fine-tuning to ESM-2 is reasonable, and the use of multiple biological supervision sources, including Pfam, AFDB clusters, STRING, and DMS data, is potentially useful. The evaluation is also fairly broad, covering classification, regression, structural retrieval, few-shot transfer, and ablations over training data sources.
Weaknesses:
Despite the authors’ efforts, I think this study has several issues that should be addressed to improve its overall quality.
Potential train-test leakage is the most serious concern. For structure retrieval, the authors acknowledge that AFDB training sequences were not filtered against SCOPe test domains, which weakens the reported SCOPe and remote homology results. For PPI evaluation, although the authors remove test PPI sequences from training, a stricter analysis is still needed, for example ensuring that test and training sequences share less than 50% or even 40% sequence identity.
The biological assumption behind the DMS objective is not fully justified. A more reasonable goal is to preserve fitness-induced ordering of WT-mutant distances, rather than simply pulling all high-fitness variants close to the wild type.
The MNRL implementation are under-specified. It is unclear whether the effective batch size of 1024 actually contributes to the in-batch negative set, or whether negatives are only computed within each smaller forward micro-batch. In addition, the superscript “+” in Eq. (1) is not clearly defined.
The evaluation protocol for pair-level tasks is not reproducible. PPI and peptide-HLA binding require pair inputs, but the paper does not explain how two protein embeddings are combined for KNN classification.
The regression evaluation is also missing important details. The paper does not specify whether KNN regression uses uniform averaging or distance-weighted averaging.
The ablation results do not fully support the default design. Removing Pfam hard negatives improves the aggregate performance, and proportional sampling performs comparably to or slightly better than round-robin sampling.
The baseline comparison is insufficient. The paper mainly compares against ESM-2 mean pooling, but should include stronger protein embedding and retrieval baselines, such as ProtTucker, HMMER/MMseqs2, Foldseek, PLMSearch, DHR, and the prior work “Optimizing Protein Language Models with Sentence Transformers.”
The statistical evidence is weak. Some improvements reported in Table 2 are very small.
Quality: 2: not good
Clarity: 2: not good
Significance: 2: not good
Originality: 2: not good
Questions:
Please see the weaknesses mentioned above.
Limitations:
No. The authors mention several limitations, including possible AFDB/SCOPe overlap, lack of specialized retrieval baselines, and single-run results. However, the discussion remains insufficient given that these issues directly affect the paper’s main claims.
In particular, the paper should more explicitly address potential train-test leakage, the ambiguity between effective batch size and the actual number of MNRL in-batch negatives, the under-specified evaluation protocols for pair-level and regression tasks, and the biological assumption of mapping heterogeneous protein relationships into a single embedding space. A clearer discussion of these limitations is necessary to properly interpret when ProtSent is expected to be reliable.
Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 4

Rebuttal 3:
Response to Reviewer Yi1G
Thank you for the detailed review. We address the eight concerns in the order raised.
1. AFDB/SCOPe and STRING/PPI overlap
SCOPe. We agree that noting possible overlap was insufficient. The original AFDB preparation filters by pLDDT and fragment status and assigns Foldseek clusters, but it does not decontaminate against SCOPe. We therefore searched every SCOPe query against the released AFDB training-source pool with MMseqs2 using 80% query coverage. This is a conservative source-overlap audit rather than proof of exact checkpoint exposure, because the released pool does not include the exact sampled-pair manifest. We then recomputed retrieval on the retained queries.
At the 50% threshold, 155/2,207 queries remain:
Scale
Model
R@1
R@30
MAP
150M
ESM-2
0.303
0.477
0.199
150M
ProtSent
0.329
0.581
0.319
35M
ESM-2
0.265
0.503
0.204
35M
ProtSent
0.297
0.594
0.287
As a stricter sensitivity analysis, excluding a query when either AFDB or STRING has a hit at ≥50% leaves 92 queries, only 57 of which have a non-self family positive. At 150M, R@1 ties (0.250 vs 0.250), while R@30 increases from 0.413 to 0.500 and MAP from 0.182 to 0.248. At 35M, R@1 is 0.207 vs 0.239, R@30 is 0.424 vs 0.533, and MAP is 0.171 vs 0.256. Paired bootstrap intervals include zero for both R@1 deltas, but exclude zero for R@30 and MAP at both scales. Thus, the strict subset does not support a robust top-1 claim, but it retains evidence for better deeper retrieval. We will report this narrower conclusion and the retained sample counts.
This completed sensitivity filters clean queries while retaining the fixed full gallery.[[RESULT: Insert the 40% and clean-query/clean-gallery results when complete. If they are not complete, delete this line and explicitly label the table above as query-filtered/full-gallery.]]
We also found that the submitted code evaluates 2,207 proteins using the SCOPe family field, not 100,000 proteins at the superfamily level. We will correct the description. A separate superfamily evaluation still improves at both scales.
For remote homology, the downstream split is disjoint in its evaluation hierarchy, which prevents direct task-label leakage but does not by itself exclude sequence exposure through the large fine-tuning sources. We therefore do not use the split alone as a leakage defense and will state this residual limitation.
PPI. This was already decontaminated in the original data pipeline: Bernett test proteins were added to the STRING sequence pool, MMseqs2 easy-linclust was run at 50% identity and 80% target coverage, and every STRING protein in a Bernett-containing cluster was removed before the final STRING pairs were constructed. The requested 40% analysis is therefore an additional sensitivity check, not a missing train/test control.
[[RESULT: Insert the <40% PPI subset size, class balance, vanilla AUC, and ProtSent AUC when complete. Delete if not complete.]]
2. DMS objective
The requested ordering objective is the one implemented. Each row is a WT–mutant pair with normalized fitness. CoSENT ranks pair similarities: if mutant (a) has higher fitness than mutant (b), it encourages (\mathrm{sim}(WT,a)>\mathrm{sim}(WT,b)). It does not assign an absolute similarity target or collapse all high-fitness variants together. We agree that this was not explained clearly enough in the main text.
The remaining limitation is that the default data are WT-anchored; mutant–mutant distances are not directly optimized.
3. MNRL batch and Eq. 1
The released paper-reproduction path uses CachedMultipleNegativesRankingLoss with a logical batch size of 1024. The loss is evaluated against the full logical batch; mini_batch_size=256 only partitions the forward/backward computation to reduce memory. Thus, on an MNRL step, each anchor is contrasted against the other 1023 positive-side examples in that source batch. Round-robin sampling means a step contains examples from one source, not a mixture of all sources.
The phrase “effective batch size” was ambiguous and will be replaced by the logical contrastive batch and cached mini-batch separately.
Eq. 1 is also malformed. The numerator should use the positive paired with anchor (i), while the denominator ranges over the positive members of all (N) pairs. The superscript (+) denotes the positive member of a pair. We will correct the notation.
4. Pair-level tasks
For PPI, each partner is embedded independently and the two embeddings are concatenated before applying the same probe. This is implemented but omitted from the paper.
Peptide–HLA is not a two-input task in our submitted benchmark implementation: the dataset supplies one seq field, so no partner-combination operator is used there. We will make both points explicit.
5. k-NN regression
Regression uses scikit-learn KNeighborsRegressor(n_neighbors=3, metric="minkowski") without a weights argument; therefore it uses uniform averaging and Euclidean distance. We will specify this.
6. Ablations
We agree that the ablations do not establish the submitted choices as uniformly optimal. Removing hard negatives improves 20/23 tasks with mean relative change +7.9%, compared with 16/23 and +6.7% for the full 35M configuration. The per-task results show a trade-off rather than a universally better choice. Likewise, proportional sampling (+7.0%) is effectively comparable to round-robin (+6.7%). We will revise the interpretation and will not claim that either hard negatives or round-robin sampling is validated as generally superior.
7. Baselines
We added the matched MMseqs2 SCOPe baseline above. We are also running frozen logistic-regression/ridge probes on vanilla ESM-2 and ProtSent using identical splits.
[[RESULT: Insert the completed linear/ridge aggregate and representative tasks.]]
We agree that ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, and related sentence-transformer work must be discussed more clearly. In particular, Redl et al.'s “Optimizing Protein Language Models with Sentence Transformers” is the closest methodological antecedent and will be compared explicitly rather than mentioned only briefly. We cannot produce reliable matched runs for all of these systems in the rebuttal window because they require different backbones, structural inputs, or indexing pipelines. We will not claim superiority to them; we will position ProtSent as a general-purpose sequence embedding and report specialized comparisons only where dataset, split, label level, and metric are matched.
8. Statistical evidence
We agree that several small Table 2 differences are not interpretable without uncertainty.
[[RESULT: Insert paired-bootstrap task summary and representative confidence intervals. Do not describe an improvement as established when its interval includes zero.]]
The final reporting will separate supported improvements, supported degradations, and unresolved differences, and will use absolute deltas rather than only relative percentages. The mixed ablations and the stability/thermostability degradations will also be described as evidence that combining heterogeneous relations in one space creates real task-dependent trade-offs.
These checks resolve several reproducibility ambiguities and materially narrow the SCOPe claim on the strict clean subset. If these responses and completed results resolve the concerns, we would appreciate an updated assessment. If one concern remains decisive, please identify it during discussion so we can respond directly.

