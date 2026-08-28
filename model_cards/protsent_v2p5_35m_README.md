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
base_model: GrimSqueaker/ProtSent-V2-35M
---

# ProtSent-V2.5 ESM-2 35M

[ProtSent-V2 35M](https://huggingface.co/GrimSqueaker/ProtSent-V2-35M) plus one more
contrastive pass on a fresh draw of the corpus, with a DMS/ProteinGym CoSENT target and a
Global Orthogonal Regularization term added.

Mean-pooled ESM-2 35M embeddings, dimension 480, Matryoshka heads at 64/128/256.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("GrimSqueaker/ProtSent-V2.5-35M")
emb = model.encode(["MKTLLLTLVVVTIVCLDLGYT", "MKTLLLTLVVVTIVCLDLGYN", "AGWYRSPQEGLKPVDTFKDIV"])
print(cos_sim(emb[0], emb[1:]))
```

## Training

| | V2 | V2.5 |
|---|---|---|
| init | ESM-2 35M | ProtSent-V2 35M |
| loss | CachedMNRL + Matryoshka | + GOR (weight 0.1) + DMS CoSENT |
| pairs | 34.8M | 15.3M — Pfam 285k, AFDB 7M, STRING 7M, DMS 1M |
| cluster sample | k=10, seed 42 | k=5, seed 13 (fresh draw, ~28% rows new) |
| batch / mini-batch | 1024 / 256 | 1024 / 256, CoSENT capped at 256 |
| LR | 2e-4, 3 cosine cycles | 5e-5, half-cosine to zero, 200 warmup |
| max sequence length | 512 | 512 |
| steps / hardware | 4,850 on 7xB300 | 14,924 on 1xB300, 11 h 49 m |

Corpora are decontaminated with MMseqs2 at 40% identity / 80% coverage against the
remote-homology and PPI test splits. The DMS parquet is **not** decontaminated — four suite
tasks are DMS-derived (Stability, Fluorescence, beta-lactamase, Variant Effect), and only
exact-match overlap has been checked (zero on all three tested).

## Results

**SCOPe-40 structural retrieval**, test split, self excluded, restricted to the 1,693 of
2,207 queries that have a non-self same-family protein in the gallery.

| model | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent-V1 35M | 0.5854 | 0.8511 | 0.9256 | 0.5509 |
| ProtSent-V2 35M | 0.6852 | 0.9220 | 0.9634 | 0.6459 |
| **ProtSent-V2.5 35M** | **0.6899** | **0.9244** | **0.9681** | **0.6521** |

Paired bootstrap over queries, 10,000 resamples, V2.5 − V2: R@1 +0.0077 [−0.0041, +0.0201],
R@10 +0.0018 [−0.0053, +0.0089], R@30 +0.0047 [+0.0000, +0.0100], **MAP +0.0078 [+0.0032,
+0.0125]**. MAP is the only metric excluding zero, so the supportable claim is ranking
depth, not top-1. Profile alignment (HMMER) still leads at R@1.

**23-task downstream suite.** Win/tie/loss over the 20 tasks with a defined one-vs-rest AUC
(ties = |delta| < 0.005), with the median delta:

| comparison | k-NN probe | linear probe |
|---|---|---|
| V2 vs ESM-2 35M | 10W/3T/7L, +0.0041 | 2W/7T/11L, −0.0107 |
| V2.5 vs ESM-2 35M | 9W/7T/4L, +0.0046 | 4W/4T/12L, −0.0103 |
| V2.5 vs V2 | 7W/8T/5L, +0.0010 | 7W/8T/5L, +0.0013 |

V2.5 is indistinguishable from V2 on this aggregate. A sign test resolves almost none of
these records, so no inferential claim is drawn from the tallies. The linear-probe deficit
against vanilla ESM-2 is unchanged.

Largest per-task moves, V2.5 − V2, linear probe: Stability +0.0946, AAV Fitness +0.0732,
Fluorescence +0.0149, Variant Effect +0.0128, beta-lactamase +0.0110, Optimal pH −0.0247,
Binary Subcellular Localization −0.0128. The gains are the DMS-derived tasks, matching the
re-added CoSENT target.

An ablation trained with `--gor_weight 0` and everything else identical matches this model
within noise (2W/15T/2L on the k-NN suite; SCOPe-40 eligible R@1 0.6923 / MAP 0.6528). GOR
cost +11.7% per step and is not the source of the improvement over V2.

Training code and full run log:
[github.com/oriel9p/ProtSent](https://github.com/oriel9p/ProtSent)
(`train_esm2_35m_v2p5.sh`, `RUNS.md`).
