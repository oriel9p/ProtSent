---
license: mit
language:
- en
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- protein
- esm2
- biology
pipeline_tag: sentence-similarity
base_model: GrimSqueaker/ProtSent-V2-150M
---

# ProtSent-V2.5 ESM-2 150M

Continued contrastive training of [ProtSent-V2 150M](https://huggingface.co/GrimSqueaker/ProtSent-V2-150M)
on a fresh draw of the corpus, adding a DMS/ProteinGym CoSENT target and a Global
Orthogonal Regularization term.

Mean-pooled ESM-2 150M embeddings, dimension 640, max sequence length 512.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("GrimSqueaker/ProtSent-V2.5-150M")
emb = model.encode(["MKTLLLTLVVVTIVCLDLGYT", "MKTLLLTLVVVTIVCLDLGYN", "AGWYRSPQEGLKPVDTFKDIV"])
print(cos_sim(emb[0], emb[1:]))
```

## Training data

| source | pairs available | selection |
|---|---:|---|
| Pfam | 777,306 | k=8 per cluster |
| AlphaFold DB (Foldseek clusters) | 18,980,000 | k=8 per cluster |
| STRING-DB v12 PPI | 15,000,000 | full filtered table |
| DMS / ProteinGym (CoSENT) | 1,000,000 | prefix of interleaved assays |
| **total** | **35.8M** | |

Trained on 14.7M pairs (3,600 steps x 4,096 effective batch), 41% of the pool.

Pfam, AFDB and STRING are decontaminated with MMseqs2 at 40% identity / 80%
coverage against the remote-homology and PPI test splits. The DMS parquet is not
MMseqs2-filtered; it has zero exact-sequence overlap with the AAV Fitness
(0/50,430), Stability, Variant Effect and Fluorescence test sets.

## Run parameters

| parameter | value |
|---|---|
| init | ProtSent-V2-150M (verified identical, 515 tensors, max abs diff 0.0) |
| primary loss | `CachedMultipleNegativesRankingLoss`, scale 20 |
| `--mnrl_directions` | `query_to_doc doc_to_query` (symmetric) |
| `--gor_weight` | 1.0 |
| `--gor_max_samples` | 64 |
| auxiliary loss | `CoSENTLoss` on DMS, batch capped at 64, scale 20 |
| Matryoshka | off |
| batch size (per device) | 1024 |
| `--mnrl_mini_batch_size` | 64 |
| gather across devices | off |
| `--batch_sampler` | none |
| `--max_seq_length` | 512 |
| `--max_pairs_per_cluster` | 8 |
| learning rate | 5e-5, `cosine_with_min_lr`, 0.5 cycles, 200 warmup |
| seeds (shuffle / global) | 17 / 11 |
| precision / attention | bf16, flash-attention-2 |
| hardware | 4x NVIDIA B300, 14 h 18 m |
| steps | 3,600 |

Training code: [github.com/oriel9p/ProtSent](https://github.com/oriel9p/ProtSent),
`train_esm2_150m_v2p5.sh`.

## Results

SCOPe-40 structural retrieval, test split, self excluded, restricted to the 1,693
of 2,207 queries with a non-self same-family protein in the gallery.

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| ESM-2 150M | 0.554 | 0.770 | 0.842 | 0.424 |
| MMseqs2 (`-s 7.5`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer, max sensitivity) | 0.753 | 0.898 | 0.923 | 0.607 |
| ProtSent-V1 150M | 0.662 | 0.894 | 0.944 | 0.643 |
| ProtSent-V2 150M | 0.743 | 0.937 | 0.968 | 0.705 |
| **ProtSent-V2.5 150M** | **0.751** | **0.945** | **0.972** | **0.723** |

Paired bootstrap over queries, 10,000 resamples, V2.5 − V2:

| metric | delta | 95% CI |
|---|---:|---|
| R@1 | +0.0095 | [−0.0024, +0.0213] |
| R@10 | +0.0077 | [+0.0012, +0.0142] |
| R@30 | +0.0053 | [+0.0006, +0.0106] |
| MAP | +0.0189 | [+0.0137, +0.0241] |

R@10, R@30 and MAP exclude zero; R@1 does not. HMMER still leads at R@1.

### 23-task downstream suite

Win/tie/loss over the 20 tasks with a defined one-vs-rest AUC, ties at
|delta| < 0.005, with the median delta. All arms scored with `--eval_split test`,
`EvalMode=standard`, seed 42.

| comparison | k-NN probe | linear probe |
|---|---|---|
| V2.5 vs ESM-2 150M | 12W/2T/5L, +0.010 | 3W/2T/14L, −0.016 |
| V2 vs ESM-2 150M | 9W/3T/7L, +0.004 | 3W/4T/12L, −0.014 |
| V2.5 vs V1 150M | 13W/3T/3L, +0.007 | 4W/6T/9L, −0.004 |
| V2.5 vs V2 150M | 8W/8T/3L, +0.001 | 3W/11T/5L, −0.003 |

V2.5 improves on the untuned backbone under a k-NN probe by more than V2 did.
Under a linear probe, contrastive post-training costs about 0.015 against the
backbone at this scale, and V2.5 does not change that.

Per-task, both V2.5 − V2 medians are inside the tie band. The movement is
concentrated in the Spearman regression tasks and reverses by probe: mean delta
+0.020 under k-NN and −0.006 under linear, while AUC tasks are flat under both
(+0.000 and −0.001).

| task | metric | k-NN delta | linear delta |
|---|---|---:|---:|
| Fluorescence (TAPE) | Spearman | +0.068 | +0.015 |
| Thermostability (FLIP) | Spearman | +0.033 | −0.000 |
| Variant Effect (GB1) | Spearman | +0.025 | −0.003 |
| Cloning Classification | Spearman | +0.023 | +0.006 |
| beta-lactamase-PEER | Spearman | +0.021 | +0.049 |
| Molecular Function (GO) | F1_Macro | −0.014 | −0.014 |
| Metal Ion Binding | AUC | −0.011 | −0.008 |
| Stability (Biomap) | Spearman | +0.014 | −0.023 |
| AAV Fitness (FLIP) | Spearman | +0.003 | −0.086 |

AAV Fitness and Stability decline monotonically across the whole 150M lineage
under a linear probe (AAV 0.589 → 0.398 → 0.451 → 0.365 for ESM-2 → V1 → V2 →
V2.5; Stability 0.706 → 0.699 → 0.663 → 0.639).

The AAV drop is not embedding collapse. Over 3,000 AAV test variants, V2.5 has a
lower mean pairwise cosine than V2 (0.983 vs 0.992), higher effective
dimensionality (10.4 vs 6.8) and a higher best-single-direction correlation with
fitness (0.576 vs 0.542, against 0.554 for vanilla ESM-2). The variants are
spread further apart and carry more linear signal in-sample; what degrades is
transfer from the train split to the deliberately hard FLIP test split.

V2.5 changes six settings at once relative to V2 (GOR, DMS target, symmetric
directions, k, seeds, batch geometry) and there is no GOR-off ablation at this
scale, so no individual change is isolated here. The equivalent ablation at 35M
found GOR contributed nothing measurable.
