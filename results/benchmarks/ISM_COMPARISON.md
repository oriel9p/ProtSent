# ISM-C-300M on the ProtSent benchmark suite

ISM-C-300M is a structure-distilled ESM-C-300M. `Synthyra/ESMplusplus_small`
is vanilla ESM-C-300M: same architecture, parameter count, tokenizer and
code path, so that pairing isolates the distillation and nothing else.

The ProtSent and ESM-2 rows differ from the ESM-C rows in BOTH family and
scale. They are context, not a controlled comparison, and no claim of the
form "contrastive training beats structure distillation" follows from them
alone -- raw mean-pooled ESM-C is simply weak at retrieval, below even
ESM-2 35M. Separating the two needs ProtSent post-training on ESM-C.

Three of the 20 rows below (EC, GO, SCOPe-40) are probe-invariant by
construction: multilabel and retrieval tasks use a built-in evaluator and
ignore the --probe_type flag, so their knn and linear numbers are identical.

## SCOPe-40 structural retrieval

Test split, self excluded, restricted to the 1,693 of 2,207 queries with a
non-self same-family protein in the gallery.

| method | R@1 | R@10 | MAP |
|---|---|---|---|
| MMseqs2 (-s 7.5) | 0.6556 | 0.7348 | 0.4041 |
| HMMER (phmmer, filters off) | 0.7525 | 0.8978 | 0.6067 |
| ESM-C 300M | 0.3709 | 0.5794 | 0.2212 |
| ISM-C 300M | 0.4300 | 0.6592 | 0.2733 |
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| ProtSent-V1 150M | 0.6615 | 0.8943 | 0.6431 |
| ProtSent-V2 150M | 0.7431 | 0.9368 | 0.7046 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.4210 |
| ProtSent-V2 35M | 0.6852 | 0.9220 | 0.6459 |

## Structure distillation on the 23-task suite (knn probe)

ISM-C beats vanilla ESM-C on **8** tasks, ties on **1**, loses on **14** of 23 (tie tolerance 0.005). Median delta -0.0086.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| Stability (Biomap) | Spearman | 0.6935 | 0.5794 | -0.1141 |
| EC Classification | F1_Macro | 0.6404 | 0.5274 | -0.1129 |
| Temperature Stability | Accuracy | 0.9257 | 0.8394 | -0.0863 |
| Variant Effect (GB1) | Spearman | 0.7989 | 0.7199 | -0.0790 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | -0.0787 |
| Optimal pH | Spearman | 0.5113 | 0.4525 | -0.0587 |
| Metal Ion Binding | AUC | 0.7369 | 0.6947 | -0.0422 |
| Antibiotic Resistance | Accuracy | 0.9628 | 0.9420 | -0.0208 |
| Binary Subcellular Localization | AUC | 0.8813 | 0.8660 | -0.0153 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.9781 | 0.9673 | -0.0108 |
| Subcellular Localisation | AUC | 0.7862 | 0.7756 | -0.0106 |
| AAV Fitness (FLIP) | Spearman | 0.4921 | 0.4835 | -0.0086 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.9749 | 0.9679 | -0.0069 |
| Thermostability (FLIP) | Spearman | 0.4497 | 0.4447 | -0.0050 |
| Enzyme Catalytic Efficiency | Spearman | 0.6957 | 0.6908 | -0.0049 |
| beta-lactamase-PEER | Spearman | 0.8239 | 0.8351 | +0.0112 |
| Peptide-HLA Binding | AUC | 0.7220 | 0.7457 | +0.0237 |
| Remote Homology (Fold) | Accuracy | 0.3545 | 0.4026 | +0.0481 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.5794 | 0.6592 | +0.0797 |
| Material Production | AUC | 0.7266 | 0.8087 | +0.0821 |
| Solubility (DeepSol) | AUC | 0.5944 | 0.6852 | +0.0908 |
| Fluorescence (TAPE) | Spearman | 0.3706 | 0.5436 | +0.1731 |
| Cloning Classification | Spearman | 0.3272 | 0.5071 | +0.1799 |

## Structure distillation on the 23-task suite (linear probe)

ISM-C beats vanilla ESM-C on **7** tasks, ties on **3**, loses on **13** of 23 (tie tolerance 0.005). Median delta -0.0125.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| EC Classification | F1_Macro | 0.6404 | 0.5274 | -0.1129 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | -0.0787 |
| Stability (Biomap) | Spearman | 0.7550 | 0.7003 | -0.0548 |
| Remote Homology (Fold) | Accuracy | 0.6995 | 0.6705 | -0.0290 |
| AAV Fitness (FLIP) | Spearman | 0.6058 | 0.5814 | -0.0244 |
| beta-lactamase-PEER | Spearman | 0.8324 | 0.8082 | -0.0242 |
| Temperature Stability | Accuracy | 0.9562 | 0.9333 | -0.0228 |
| Enzyme Catalytic Efficiency | Spearman | 0.6073 | 0.5845 | -0.0228 |
| Metal Ion Binding | AUC | 0.7882 | 0.7662 | -0.0220 |
| Optimal pH | Spearman | 0.5238 | 0.5079 | -0.0159 |
| Antibiotic Resistance | Accuracy | 0.9814 | 0.9658 | -0.0156 |
| Binary Subcellular Localization | AUC | 0.9519 | 0.9394 | -0.0125 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.9915 | 0.9837 | -0.0078 |
| Subcellular Localisation | AUC | 0.9231 | 0.9185 | -0.0046 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.9942 | 0.9957 | +0.0015 |
| Thermostability (FLIP) | Spearman | 0.5498 | 0.5517 | +0.0020 |
| Variant Effect (GB1) | Spearman | 0.8165 | 0.8412 | +0.0247 |
| Peptide-HLA Binding | AUC | 0.8659 | 0.8946 | +0.0288 |
| Material Production | AUC | 0.8454 | 0.8771 | +0.0316 |
| Fluorescence (TAPE) | Spearman | 0.6155 | 0.6585 | +0.0430 |
| Cloning Classification | Spearman | 0.5129 | 0.5736 | +0.0607 |
| Solubility (DeepSol) | AUC | 0.7538 | 0.8222 | +0.0684 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.5794 | 0.6592 | +0.0797 |

## Every arm side by side (knn probe)

| task | metric | ESM-C 300M | ISM-C 300M | ESM-2 150M | ProtSent-V1 150M | ProtSent-V2 150M | ESM-2 35M | ProtSent-V2 35M | MMseqs2 |
|---|---|---|---|---|---|---|---|---|---|
| AAV Fitness (FLIP) | Spearman | 0.4921 | 0.4835 | 0.4345 | 0.4663 | 0.5031 | 0.4667 | 0.5154 | 0.4024 |
| Antibiotic Resistance | Accuracy | 0.9628 | 0.9420 | 0.9680 | 0.9628 | 0.9658 | 0.9673 | 0.9650 | 0.9544 |
| Binary Subcellular Localization | AUC | 0.8813 | 0.8660 | 0.8964 | 0.9010 | 0.9032 | 0.8810 | 0.8884 | 0.6834 |
| Cloning Classification | Spearman | 0.3272 | 0.5071 | 0.3754 | 0.3789 | 0.3627 | 0.3906 | 0.3791 | 0.1707 |
| EC Classification | F1_Macro | 0.6404 | 0.5274 | 0.6466 | 0.6040 | 0.6253 | 0.5984 | 0.5924 | 0.7103 |
| Enzyme Catalytic Efficiency | Spearman | 0.6957 | 0.6908 | 0.7060 | 0.6777 | 0.6909 | 0.6915 | 0.6687 | 0.6322 |
| Fluorescence (TAPE) | Spearman | 0.3706 | 0.5436 | 0.3802 | 0.4619 | 0.3839 | 0.3736 | 0.4568 | 0.3863 |
| Material Production | AUC | 0.7266 | 0.8087 | 0.7582 | 0.7666 | 0.7644 | 0.7684 | 0.7653 | 0.5796 |
| Metal Ion Binding | AUC | 0.7369 | 0.6947 | 0.7973 | 0.8147 | 0.8166 | 0.7957 | 0.8158 | 0.7239 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | 0.5116 | 0.4961 | 0.5148 | 0.4590 | 0.4547 | 0.5850 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.9781 | 0.9673 | 0.9777 | 0.9614 | 0.9713 | 0.9579 | 0.9601 | 0.9010 |
| Optimal pH | Spearman | 0.5113 | 0.4525 | 0.5622 | 0.5836 | 0.5892 | 0.5821 | 0.5756 | 0.5462 |
| Peptide-HLA Binding | AUC | 0.7220 | 0.7457 | 0.7565 | 0.7790 | 0.7999 | 0.7496 | 0.8022 | 0.6374 |
| Remote Homology (Fold) | Accuracy | 0.3545 | 0.4026 | 0.5194 | 0.7047 | 0.6612 | 0.5835 | 0.6668 | 0.6523 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.5794 | 0.6592 | 0.7702 | 0.8943 | 0.9368 | 0.7614 | 0.9220 | 0.7348 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.9749 | 0.9679 | 0.9804 | 0.9740 | 0.9827 | 0.9780 | 0.9840 | 0.7961 |
| Solubility (DeepSol) | AUC | 0.5944 | 0.6852 | 0.5465 | 0.5485 | 0.5263 | 0.5321 | 0.5426 | 0.4185 |
| Stability (Biomap) | Spearman | 0.6935 | 0.5794 | 0.6495 | 0.6097 | 0.6239 | 0.6435 | 0.5961 | 0.5817 |
| Subcellular Localisation | AUC | 0.7862 | 0.7756 | 0.8188 | 0.8265 | 0.8358 | 0.8015 | 0.8294 | 0.6828 |
| Temperature Stability | Accuracy | 0.9257 | 0.8394 | 0.8911 | 0.8480 | 0.8619 | 0.8613 | 0.8514 | 0.6853 |
| Thermostability (FLIP) | Spearman | 0.4497 | 0.4447 | 0.4377 | 0.4606 | 0.4623 | 0.4449 | 0.4367 | 0.4799 |
| Variant Effect (GB1) | Spearman | 0.7989 | 0.7199 | 0.6607 | 0.7983 | 0.7700 | 0.6582 | 0.7806 | 0.7166 |
| beta-lactamase-PEER | Spearman | 0.8239 | 0.8351 | 0.7728 | 0.7423 | 0.7619 | 0.7272 | 0.7153 | 0.8026 |

## Every arm side by side (linear probe)

| task | metric | ESM-C 300M | ISM-C 300M | ESM-2 150M | ProtSent-V1 150M | ProtSent-V2 150M | ESM-2 35M | ProtSent-V2 35M | MMseqs2 |
|---|---|---|---|---|---|---|---|---|---|
| AAV Fitness (FLIP) | Spearman | 0.6058 | 0.5814 | 0.5887 | 0.3976 | 0.4509 | 0.5639 | 0.2471 | 0.4024 |
| Antibiotic Resistance | Accuracy | 0.9814 | 0.9658 | 0.9814 | 0.9747 | 0.9725 | 0.9755 | 0.9665 | 0.9544 |
| Binary Subcellular Localization | AUC | 0.9519 | 0.9394 | 0.9499 | 0.9351 | 0.9234 | 0.9572 | 0.9093 | 0.6834 |
| Cloning Classification | Spearman | 0.5129 | 0.5736 | 0.5071 | 0.4783 | 0.4345 | 0.4780 | 0.4257 | 0.1707 |
| EC Classification | F1_Macro | 0.6404 | 0.5274 | 0.6466 | 0.6040 | 0.6253 | 0.5984 | 0.5924 | 0.7103 |
| Enzyme Catalytic Efficiency | Spearman | 0.6073 | 0.5845 | 0.5466 | 0.5478 | 0.5432 | 0.5456 | 0.5002 | 0.6322 |
| Fluorescence (TAPE) | Spearman | 0.6155 | 0.6585 | 0.5799 | 0.6079 | 0.5908 | 0.5913 | 0.5883 | 0.3863 |
| Material Production | AUC | 0.8454 | 0.8771 | 0.8528 | 0.8400 | 0.8344 | 0.8399 | 0.8225 | 0.5796 |
| Metal Ion Binding | AUC | 0.7882 | 0.7662 | 0.7920 | 0.7891 | 0.7690 | 0.7903 | 0.7466 | 0.7239 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | 0.5116 | 0.4961 | 0.5148 | 0.4590 | 0.4547 | 0.5850 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.9915 | 0.9837 | 0.9802 | 0.9762 | 0.9714 | 0.9748 | 0.9594 | 0.9010 |
| Optimal pH | Spearman | 0.5238 | 0.5079 | 0.5315 | 0.5170 | 0.5034 | 0.4864 | 0.4845 | 0.5462 |
| Peptide-HLA Binding | AUC | 0.8659 | 0.8946 | 0.8630 | 0.8374 | 0.8712 | 0.8628 | 0.8579 | 0.6374 |
| Remote Homology (Fold) | Accuracy | 0.6995 | 0.6705 | 0.7500 | 0.7401 | 0.7503 | 0.6868 | 0.7016 | 0.6523 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.5794 | 0.6592 | 0.7702 | 0.8943 | 0.9368 | 0.7614 | 0.9220 | 0.7348 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.9942 | 0.9957 | 0.9942 | 0.9941 | 0.9961 | 0.9942 | 0.9957 | 0.7961 |
| Solubility (DeepSol) | AUC | 0.7538 | 0.8222 | 0.7211 | 0.7085 | 0.7076 | 0.6963 | 0.6976 | 0.4185 |
| Stability (Biomap) | Spearman | 0.7550 | 0.7003 | 0.7060 | 0.6987 | 0.6625 | 0.4395 | 0.3878 | 0.5817 |
| Subcellular Localisation | AUC | 0.9231 | 0.9185 | 0.9203 | 0.9156 | 0.9079 | 0.9116 | 0.8921 | 0.6828 |
| Temperature Stability | Accuracy | 0.9562 | 0.9333 | 0.9264 | 0.8908 | 0.9097 | 0.8973 | 0.8963 | 0.6853 |
| Thermostability (FLIP) | Spearman | 0.5498 | 0.5517 | 0.5605 | 0.5581 | 0.5606 | 0.5413 | 0.5564 | 0.4799 |
| Variant Effect (GB1) | Spearman | 0.8165 | 0.8412 | 0.8047 | 0.8448 | 0.8294 | 0.8163 | 0.8126 | 0.7166 |
| beta-lactamase-PEER | Spearman | 0.8324 | 0.8082 | 0.7788 | 0.6807 | 0.6454 | 0.6639 | 0.6086 | 0.8026 |
