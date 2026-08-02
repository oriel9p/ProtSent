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
| MMseqs2 (-s 7.5) | 0.656 | 0.740 | 0.410 |
| HMMER (phmmer, filters off) | 0.753 | 0.898 | 0.607 |
| ESM-C 300M | 0.371 | 0.579 | 0.221 |
| ISM-C 300M | 0.430 | 0.659 | 0.273 |
| ESM-2 150M | 0.553 | 0.770 | 0.424 |
| ProtSent-V1 150M | 0.662 | 0.894 | 0.643 |
| ProtSent-V2 150M | 0.743 | 0.937 | 0.705 |
| ESM-2 35M | 0.499 | 0.761 | 0.421 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.646 |

## Structure distillation on the 23-task suite (knn probe)

ISM-C beats vanilla ESM-C on **8** tasks, ties on **1**, loses on **14** of 23 (tie tolerance 0.005). Median delta -0.009.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| Stability (Biomap) | Spearman | 0.693 | 0.579 | -0.114 |
| EC Classification | F1_Macro | 0.640 | 0.527 | -0.113 |
| Temperature Stability | Accuracy | 0.926 | 0.839 | -0.086 |
| Variant Effect (GB1) | Spearman | 0.799 | 0.720 | -0.079 |
| Molecular Function (GO) | F1_Macro | 0.516 | 0.438 | -0.079 |
| Optimal pH | Spearman | 0.511 | 0.453 | -0.059 |
| Metal Ion Binding | AUC | 0.737 | 0.695 | -0.042 |
| Antibiotic Resistance | Accuracy | 0.963 | 0.942 | -0.021 |
| Binary Subcellular Localization | AUC | 0.881 | 0.866 | -0.015 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.978 | 0.967 | -0.011 |
| Subcellular Localisation | AUC | 0.786 | 0.776 | -0.011 |
| AAV Fitness (FLIP) | Spearman | 0.492 | 0.483 | -0.009 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.975 | 0.968 | -0.007 |
| Thermostability (FLIP) | Spearman | 0.450 | 0.445 | -0.005 |
| Enzyme Catalytic Efficiency | Spearman | 0.696 | 0.691 | -0.005 |
| beta-lactamase-PEER | Spearman | 0.824 | 0.835 | +0.011 |
| Peptide-HLA Binding | AUC | 0.722 | 0.746 | +0.024 |
| Remote Homology (Fold) | Accuracy | 0.354 | 0.403 | +0.048 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.579 | 0.659 | +0.080 |
| Material Production | AUC | 0.727 | 0.809 | +0.082 |
| Solubility (DeepSol) | AUC | 0.594 | 0.685 | +0.091 |
| Fluorescence (TAPE) | Spearman | 0.371 | 0.544 | +0.173 |
| Cloning Classification | Spearman | 0.327 | 0.507 | +0.180 |

## Structure distillation on the 23-task suite (linear probe)

ISM-C beats vanilla ESM-C on **7** tasks, ties on **3**, loses on **13** of 23 (tie tolerance 0.005). Median delta -0.013.

| task | metric | ESM-C 300M | ISM-C 300M | delta |
|---|---|---|---|---|
| EC Classification | F1_Macro | 0.640 | 0.527 | -0.113 |
| Molecular Function (GO) | F1_Macro | 0.516 | 0.438 | -0.079 |
| Stability (Biomap) | Spearman | 0.755 | 0.700 | -0.055 |
| Remote Homology (Fold) | Accuracy | 0.699 | 0.670 | -0.029 |
| AAV Fitness (FLIP) | Spearman | 0.606 | 0.581 | -0.024 |
| beta-lactamase-PEER | Spearman | 0.832 | 0.808 | -0.024 |
| Temperature Stability | Accuracy | 0.956 | 0.933 | -0.023 |
| Enzyme Catalytic Efficiency | Spearman | 0.607 | 0.584 | -0.023 |
| Metal Ion Binding | AUC | 0.788 | 0.766 | -0.022 |
| Optimal pH | Spearman | 0.524 | 0.508 | -0.016 |
| Antibiotic Resistance | Accuracy | 0.981 | 0.966 | -0.016 |
| Binary Subcellular Localization | AUC | 0.952 | 0.939 | -0.013 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.991 | 0.984 | -0.008 |
| Subcellular Localisation | AUC | 0.923 | 0.918 | -0.005 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.994 | 0.996 | +0.001 |
| Thermostability (FLIP) | Spearman | 0.550 | 0.552 | +0.002 |
| Variant Effect (GB1) | Spearman | 0.817 | 0.841 | +0.025 |
| Peptide-HLA Binding | AUC | 0.866 | 0.895 | +0.029 |
| Material Production | AUC | 0.845 | 0.877 | +0.032 |
| Fluorescence (TAPE) | Spearman | 0.615 | 0.659 | +0.043 |
| Cloning Classification | Spearman | 0.513 | 0.574 | +0.061 |
| Solubility (DeepSol) | AUC | 0.754 | 0.822 | +0.068 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.579 | 0.659 | +0.080 |

## Every arm side by side (knn probe)

| task | metric | ESM-C 300M | ISM-C 300M | ESM-2 150M | ProtSent-V1 150M | ProtSent-V2 150M | ESM-2 35M | ProtSent-V2 35M | MMseqs2 |
|---|---|---|---|---|---|---|---|---|---|
| AAV Fitness (FLIP) | Spearman | 0.492 | 0.483 | 0.435 | 0.466 | 0.503 | 0.467 | 0.515 | 0.402 |
| Antibiotic Resistance | Accuracy | 0.963 | 0.942 | 0.968 | 0.963 | 0.966 | 0.967 | 0.965 | 0.954 |
| Binary Subcellular Localization | AUC | 0.881 | 0.866 | 0.896 | 0.901 | 0.903 | 0.881 | 0.888 | 0.683 |
| Cloning Classification | Spearman | 0.327 | 0.507 | 0.375 | 0.379 | 0.363 | 0.391 | 0.379 | 0.171 |
| EC Classification | F1_Macro | 0.640 | 0.527 | 0.647 | 0.604 | 0.625 | 0.598 | 0.592 | 0.710 |
| Enzyme Catalytic Efficiency | Spearman | 0.696 | 0.691 | 0.706 | 0.678 | 0.691 | 0.692 | 0.669 | 0.632 |
| Fluorescence (TAPE) | Spearman | 0.371 | 0.544 | 0.380 | 0.462 | 0.384 | 0.374 | 0.457 | 0.386 |
| Material Production | AUC | 0.727 | 0.809 | 0.758 | 0.767 | 0.764 | 0.768 | 0.765 | 0.580 |
| Metal Ion Binding | AUC | 0.737 | 0.695 | 0.797 | 0.815 | 0.817 | 0.796 | 0.816 | 0.724 |
| Molecular Function (GO) | F1_Macro | 0.516 | 0.438 | 0.512 | 0.496 | 0.515 | 0.459 | 0.455 | 0.585 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.978 | 0.967 | 0.978 | 0.961 | 0.971 | 0.958 | 0.960 | 0.901 |
| Optimal pH | Spearman | 0.511 | 0.453 | 0.562 | 0.584 | 0.589 | 0.582 | 0.576 | 0.546 |
| Peptide-HLA Binding | AUC | 0.722 | 0.746 | 0.757 | 0.779 | 0.800 | 0.750 | 0.802 | 0.637 |
| Remote Homology (Fold) | Accuracy | 0.354 | 0.403 | 0.519 | 0.705 | 0.661 | 0.584 | 0.667 | 0.652 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.579 | 0.659 | 0.770 | 0.894 | 0.937 | 0.761 | 0.922 | 0.740 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.975 | 0.968 | 0.980 | 0.974 | 0.983 | 0.978 | 0.984 | 0.796 |
| Solubility (DeepSol) | AUC | 0.594 | 0.685 | 0.547 | 0.548 | 0.526 | 0.532 | 0.543 | 0.418 |
| Stability (Biomap) | Spearman | 0.693 | 0.579 | 0.650 | 0.610 | 0.624 | 0.643 | 0.596 | 0.582 |
| Subcellular Localisation | AUC | 0.786 | 0.776 | 0.819 | 0.826 | 0.836 | 0.801 | 0.829 | 0.683 |
| Temperature Stability | Accuracy | 0.926 | 0.839 | 0.891 | 0.848 | 0.862 | 0.861 | 0.851 | 0.685 |
| Thermostability (FLIP) | Spearman | 0.450 | 0.445 | 0.438 | 0.461 | 0.462 | 0.445 | 0.437 | 0.480 |
| Variant Effect (GB1) | Spearman | 0.799 | 0.720 | 0.661 | 0.798 | 0.770 | 0.658 | 0.781 | 0.717 |
| beta-lactamase-PEER | Spearman | 0.824 | 0.835 | 0.773 | 0.742 | 0.762 | 0.727 | 0.715 | 0.803 |

## Every arm side by side (linear probe)

| task | metric | ESM-C 300M | ISM-C 300M | ESM-2 150M | ProtSent-V1 150M | ProtSent-V2 150M | ESM-2 35M | ProtSent-V2 35M | MMseqs2 |
|---|---|---|---|---|---|---|---|---|---|
| AAV Fitness (FLIP) | Spearman | 0.606 | 0.581 | 0.589 | 0.398 | 0.451 | 0.564 | 0.247 | 0.402 |
| Antibiotic Resistance | Accuracy | 0.981 | 0.966 | 0.981 | 0.975 | 0.972 | 0.975 | 0.967 | 0.954 |
| Binary Subcellular Localization | AUC | 0.952 | 0.939 | 0.950 | 0.935 | 0.923 | 0.957 | 0.909 | 0.683 |
| Cloning Classification | Spearman | 0.513 | 0.574 | 0.507 | 0.478 | 0.435 | 0.478 | 0.426 | 0.171 |
| EC Classification | F1_Macro | 0.640 | 0.527 | 0.647 | 0.604 | 0.625 | 0.598 | 0.592 | 0.710 |
| Enzyme Catalytic Efficiency | Spearman | 0.607 | 0.584 | 0.547 | 0.548 | 0.543 | 0.546 | 0.500 | 0.632 |
| Fluorescence (TAPE) | Spearman | 0.615 | 0.659 | 0.580 | 0.608 | 0.591 | 0.591 | 0.588 | 0.386 |
| Material Production | AUC | 0.845 | 0.877 | 0.853 | 0.840 | 0.834 | 0.840 | 0.823 | 0.580 |
| Metal Ion Binding | AUC | 0.788 | 0.766 | 0.792 | 0.789 | 0.769 | 0.790 | 0.747 | 0.724 |
| Molecular Function (GO) | F1_Macro | 0.516 | 0.438 | 0.512 | 0.496 | 0.515 | 0.459 | 0.455 | 0.585 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.991 | 0.984 | 0.980 | 0.976 | 0.971 | 0.975 | 0.959 | 0.901 |
| Optimal pH | Spearman | 0.524 | 0.508 | 0.531 | 0.517 | 0.503 | 0.486 | 0.484 | 0.546 |
| Peptide-HLA Binding | AUC | 0.866 | 0.895 | 0.863 | 0.837 | 0.871 | 0.863 | 0.858 | 0.637 |
| Remote Homology (Fold) | Accuracy | 0.699 | 0.670 | 0.750 | 0.740 | 0.750 | 0.687 | 0.702 | 0.652 |
| SCOPe-40 Structural Retrieval | eligible_Recall@10 | 0.579 | 0.659 | 0.770 | 0.894 | 0.937 | 0.761 | 0.922 | 0.740 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.994 | 0.996 | 0.994 | 0.994 | 0.996 | 0.994 | 0.996 | 0.796 |
| Solubility (DeepSol) | AUC | 0.754 | 0.822 | 0.721 | 0.708 | 0.708 | 0.696 | 0.698 | 0.418 |
| Stability (Biomap) | Spearman | 0.755 | 0.700 | 0.706 | 0.699 | 0.663 | 0.440 | 0.388 | 0.582 |
| Subcellular Localisation | AUC | 0.923 | 0.918 | 0.920 | 0.916 | 0.908 | 0.912 | 0.892 | 0.683 |
| Temperature Stability | Accuracy | 0.956 | 0.933 | 0.926 | 0.891 | 0.910 | 0.897 | 0.896 | 0.685 |
| Thermostability (FLIP) | Spearman | 0.550 | 0.552 | 0.561 | 0.558 | 0.561 | 0.541 | 0.556 | 0.480 |
| Variant Effect (GB1) | Spearman | 0.817 | 0.841 | 0.805 | 0.845 | 0.829 | 0.816 | 0.813 | 0.717 |
| beta-lactamase-PEER | Spearman | 0.832 | 0.808 | 0.779 | 0.681 | 0.645 | 0.664 | 0.609 | 0.803 |
