# The three official reviews, verbatim (NeurIPS 2026 submission 28056)

All three scored **2: Reject**, all with Confidence 4.

HNXd and jVGf both state explicitly that they would raise their score if specific
things are provided. Those are the highest-value targets in the rebuttal.


---

## Official Review of Submission28056 by Reviewer HNXd
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

---

## Official Review of Submission28056 by Reviewer jVGf
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

---

## Official Review of Submission28056 by Reviewer Yi1G

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