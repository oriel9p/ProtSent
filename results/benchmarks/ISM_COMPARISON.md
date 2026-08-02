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

ISM-C beats vanilla ESM-C on **7** tasks, ties on **1**, loses on **12** of 20 (tie tolerance 0.005). Median delta -0.0069.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| Stability (Biomap) | Spearman | 0.6935 | 0.5794 | -0.1141 |
| EC Classification | F1_Macro | 0.6404 | 0.5274 | -0.1129 |
| Variant Effect (GB1) | Spearman | 0.7989 | 0.7199 | -0.0790 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | -0.0787 |
| Optimal pH | Spearman | 0.5113 | 0.4525 | -0.0587 |
| Metal Ion Binding | AUC | 0.7369 | 0.6947 | -0.0422 |
| Binary Subcellular Localization | AUC | 0.8813 | 0.8660 | -0.0153 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.9781 | 0.9673 | -0.0108 |
| Subcellular Localisation | AUC | 0.7862 | 0.7756 | -0.0106 |
| AAV Fitness (FLIP) | Spearman | 0.4921 | 0.4835 | -0.0086 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.9749 | 0.9679 | -0.0069 |
| Thermostability (FLIP) | Spearman | 0.4497 | 0.4447 | -0.0050 |
| Enzyme Catalytic Efficiency | Spearman | 0.6957 | 0.6908 | -0.0049 |
| beta-lactamase-PEER | Spearman | 0.8239 | 0.8351 | +0.0112 |
| Peptide-HLA Binding | AUC | 0.7220 | 0.7457 | +0.0237 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.5794 | 0.6592 | +0.0797 |
| Material Production | AUC | 0.7266 | 0.8087 | +0.0821 |
| Solubility (DeepSol) | AUC | 0.5944 | 0.6852 | +0.0908 |
| Fluorescence (TAPE) | Spearman | 0.3706 | 0.5436 | +0.1731 |
| Cloning Classification | Spearman | 0.3272 | 0.5071 | +0.1799 |

## Structure distillation on the 23-task suite (linear probe)

ISM-C beats vanilla ESM-C on **7** tasks, ties on **3**, loses on **10** of 20 (tie tolerance 0.005). Median delta -0.0046.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| EC Classification | F1_Macro | 0.6404 | 0.5274 | -0.1129 |
| Molecular Function (GO) | F1_Macro | 0.5164 | 0.4378 | -0.0787 |
| Stability (Biomap) | Spearman | 0.7550 | 0.7003 | -0.0548 |
| AAV Fitness (FLIP) | Spearman | 0.6058 | 0.5814 | -0.0244 |
| beta-lactamase-PEER | Spearman | 0.8324 | 0.8082 | -0.0242 |
| Enzyme Catalytic Efficiency | Spearman | 0.6073 | 0.5845 | -0.0228 |
| Metal Ion Binding | AUC | 0.7882 | 0.7662 | -0.0220 |
| Optimal pH | Spearman | 0.5238 | 0.5079 | -0.0159 |
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
