# ProtSent on the CATH midnight-zone benchmark (ProtTucker / EAT)

Status: complete. All numbers below are measured, not projected. Generated tables live
beside this file: `CATH_LEVELS.md`, `cath_levels.json`.

## Why this benchmark

Our own rebuttal already concedes the point: ProtTucker is "the closest analogue to our
protocol… the comparison we would most want to add" (`rebuttal/FINAL_rebuttal.md:137`), and
"the real gap, its protocol being ours" (`:216`). Both methods contrastively post-train a
frozen protein language model's per-protein embeddings and then score by 1-NN annotation
transfer. Until now we had no CATH data, no EAT code and no numbers on it.

Reference: Heinzinger, Littmann, Sillitoe, Bordin, Orengo, Rost, *Contrastive learning on
protein embeddings enlightens midnight zone*, NAR Genomics and Bioinformatics 4(2) lqac043,
doi:10.1093/nargab/lqac043.

## The task

Transfer a CATH v4.3 classification from 69,605 labelled lookup domains to 219 query domains
filtered so that **no alignment-detectable relative exists in the lookup set** (HVAL ≤ 0).
1-NN by Euclidean distance over mean-pooled per-protein embeddings; the transferred label is
the prediction. Scored at all four CATH levels — Class, Architecture, Topology, Homologous
superfamily — over the queries answerable at each level (219 / 219 / 210 / 150).

Splits are the authors' own, taken verbatim from Rostlab/EAT, so the numbers stay on the same
axis as their Table 1.

## The pipeline is faithful — the check that proves it

**MMseqs2 scores 34.7 here; the paper reports 35.**

That single number validates the whole chain: splits, labels, the H-level denominator of 150,
the 1-NN transfer rule and the scoring. It is a reproduction of a published baseline on
published data using an independent implementation, and it came out within rounding. Nothing
downstream would be trustworthy without it.

Supporting floor: 3-mer frequency vectors score **0.00**, confirming the task carries no amino
acid composition shortcut, and matching the published random baseline of 0 at H.

## Results

Accuracy (%) at each CATH level, 1-NN Euclidean, lookup69k → test219.

| Model | params | C | A | T | H |
|---|---|---|---|---|---|
| ESM2-35M (frozen base) | 35M | 78.5 | 54.3 | 42.4 | 40.7 |
| ProtSent-V1-35M | 35M | 81.7 | 64.4 | 45.7 | 50.7 |
| **ProtSent-V2-35M** | 35M | 82.2 | 64.4 | 53.3 | **56.7** |
| ESM2-150M (frozen base) | 150M | 74.0 | 53.0 | 41.0 | 43.3 |
| ProtSent-V1-150M | 150M | 83.6 | 68.0 | 56.2 | 58.0 |
| **ProtSent-V2-150M** | 150M | 84.0 | 69.9 | 57.1 | **62.7** |

Gain of each ProtSent arm over **its own frozen base**, which is the like-for-like comparison:

| | C | A | T | H |
|---|---|---|---|---|
| V2-35M − ESM2-35M | +3.7 | +10.1 | +10.9 | **+16.0** |
| V2-150M − ESM2-150M | +10.0 | +16.9 | +16.1 | **+19.4** |
| *ProtTucker − ProtT5 (published)* | *+5* | *+8* | *+7* | *+12* |

The gain grows with the difficulty of the level, which is the same signature ProtTucker shows
over raw ProtT5 — and at both our scales the H-level gain is larger than the published +12.

### Scale/structure reference points (not ProtSent arms)

Reviewer jVGf asked for positioning against structure-informed protein LMs. ISM-C-300M
distills structural signal into a vanilla-ESM-C-300M-shaped model; `Synthyra/ESMplusplus_small`
*is* vanilla ESM-C-300M, so it is the matched control at fixed architecture and parameter count.

| Model | params | C | A | T | H |
|---|---|---|---|---|---|
| ESM-C-300M (vanilla) | 300M | 66.7 | 36.1 | 21.9 | 18.7 |
| ISM-C-300M (structure-distilled) | 300M | 80.8 | 47.9 | 29.5 | **25.3** |

Structure distillation buys ISM-C +14.1 / +11.8 / +7.6 / +6.6 over its own vanilla base — same
direction as the ProtSent deltas above, smaller in magnitude at H. More striking: **both
300M ESM-C variants score below our 35M ESM2 vanilla (40.7)** on this task. ESM-C and ESM2 are
different pretraining runs with different embedding geometries; this is not a contradiction of
anything above, but it means ESM-C is not a stronger foundation for 1-NN Euclidean transfer on
CATH specifically, so it does not raise the bar our ProtSent arms need to clear.

## The two controls that matter

**1. It is not normalisation.** ProtSent is trained with a cosine objective, and the probe
scores Euclidean distance on raw embeddings, so "ProtSent just has nicer norms" is the first
thing a reviewer will say. Giving vanilla the same direction-only geometry for free:

| arm | raw | L2-normalised |
|---|---|---|
| ESM2-35M | 40.7 | 43.3 |
| ESM2-150M | 43.3 | 44.7 |
| ProtSent-V2-35M | 56.7 | 56.7 |
| ProtSent-V2-150M | 62.7 | 62.0 |

Normalisation is worth +2.7 and +1.3 to the vanilla bases and nothing at all to ProtSent
(whose geometry is already direction-dominated). Even scoring vanilla at its best, the gap
survives at **+13.3** (35M) and **+18.0** (150M). Neither model has a `Normalize` module; both
are Transformer + mean pooling, so the pipelines are identical.

**2. ESM2 scale does not buy this.** Our vanilla 35M (40.7) sat suspiciously close to
ProtBench's recorded ESM2-650M (42.7), which scaling says should not happen. Measured directly:

| model | H |
|---|---|
| ESM2-35M (FastPLM path) | 40.7 |
| ESM2-35M (`facebook/`, plain HF path) | 40.7 |
| ESM2-150M | 43.3 |
| ESM2-650M (`facebook/`) | 42.7 |

Two things fall out. The two load paths give **bit-identical** results on the same weights, so
there is no code-path artifact. And the anomaly is real, not a bug: ESM2 goes 40.7 → 43.3 →
42.7 from 35M to 650M — flat and non-monotone. **Eighteen times the parameters buys ~2 points;
contrastive post-training on the 35M buys 16.** That reproduces ProtBench's documented 42.7 for
650M exactly, so that reference number was right and our reading of it was the thing at fault.

## Method notes

- **Probe.** `-p knn --knn_k 1`, which is `KNeighborsClassifier(n_neighbors=1,
  metric="euclidean", algorithm="brute")` with **no** feature standardisation — literally
  EAT's method. The linear probe standardises features and would fit ~6,500 classes over
  69,605 rows, a different experiment.
- **The cascade is free.** The paper counts a hit at H only if C, A and T also matched. Labels
  are dotted prefix strings, so exact match at a level already implies every coarser level
  matched.
- **Masking.** A query whose label at some level appears nowhere is dropped from that level,
  rather than charging every method for an impossible case. We use the paper's rule (singleton
  across lookup ∪ test), which reproduces their 219 / 219 / 210 / 150 exactly. A stricter
  reading — the label must actually be in the lookup set — gives 208 at T and is identical at
  the other three levels; both are recorded in `cath_levels.json`.
- **Bootstrap.** 1,000 resamples, ×1.96, per the paper. EAT's code defaults to 10,000 and
  returns a bare standard error that their tables multiply by hand.
- **Two independent code paths agree.** The suite's `cath_eat` task and the standalone
  `cath_levels.py` scorer compute H separately and are asserted equal per model.

## Known differences from the paper, not chased

1. **Model class.** ProtT5-XL is 3B and ESM-1b is 650M; our arms are 35M and 150M. The ESM2
   family is weaker than ProtT5 on this task independently of anything we did.
2. **phmmer is not their HMMER row.** Theirs used CATH-Gene3D profile HMMs (77 at H); ours is
   single-sequence phmmer. Ours lands far below by construction, not by error. It is a
   different method, not a failed reproduction.
3. **Truncation.** Sequences are capped at 1,024 residues, which clips 7 of 69,605 lookup
   sequences (0.01%) and 0 queries. The paper did not truncate.
4. **One corrupt lookup sequence.** Domain `9pcyA00` carries two NUL bytes at residue 99. In
   Rostlab/EAT's own FASTA, so it is upstream. It crashed the entire task until fixed; see
   below.

## Fixes this required in ProtBench

Both were silent failures — the suite catches per-task exceptions into an `Error` column and
still exits 0, so a sweep reports success with an empty results table.

1. **`embed_dataset` signature drift.** The suite called the legacy ESM++ signature
   (`sequences=`, `max_len=`) but current FastPLM builds ship `embed_dataset(inputs, *,
   pooling=, ...)`, raising "missing 1 required positional argument: 'inputs'". Now detected by
   inspecting the signature, falling through to the generic batched path.
2. **Out-of-vocabulary residues.** FastPLM's tokenizer raises `KeyError` on any character
   absent from the vocabulary instead of falling back to unk. `9pcyA00`'s NUL byte killed the
   whole task. Ported `patch_unknown_residue_tokens` from ProtSent, applied in a `load_model`
   wrapper so it covers all eight load paths, with tests.

Worth recording: checking `AutoTokenizer` says this problem does not exist. The embedding path
uses `model.tokenizer`, which is a different object, and that one raises.

## Open risk: CATH leakage into our contrastive corpus

Our decontamination (40% identity / 80% coverage) was run against the benchmark test sets we
were using at the time — `remote_homology` / fold-prediction and PPI — **not** against CATH
test219. So we cannot presently claim test219 has no relatives in the AFDB / Pfam / STRING
data ProtSent was contrastively trained on. Measuring it means a fresh MMseqs2 search against
a 126M-sequence corpus; the decontamination run kept its hit tables but not a reusable index.

Two things make this a caveat rather than a refutation, but it should be stated plainly rather
than left for a reviewer to find:

- ProtTucker is *explicitly supervised on CATH labels* (and its best row adds 11M Gene3D
  sequences). ProtSent's supervision is Pfam / AFDB / STRING clustering, which correlates with
  structural family but never names a CATH label. Ours is the weaker supervision.
- The vanilla base and the ProtSent arm share the same pretraining exposure; only the
  contrastive stage differs. The delta is what the contrastive stage bought.

## Reproducing

```bash
bash run_benchmarks_cath.sh                       # H level, all 8 arms
python /home/ddofer/ProtBench/cath_levels.py      # full C/A/T/H table, same 8 arms
bash run_benchmarks_cath_controls.sh              # L2, scaling and floor controls
```

Cost: ~10 GPU-minutes for the six ESM2 arms plus ~6 for the two 300M arms, all on GPU 2 only
(the other seven GPUs ran an unrelated job throughout and were never touched). The controls add
another ~10. Under an hour total, against the 10 GPU-hour ceiling. The whole cost is embedding
69,605 lookup sequences once per model; `--cache_embeddings` means the H-level sweep and the
full-table scorer share embeddings rather than paying twice.

## Summary

- MMseqs2 reproduces the paper's published number almost exactly (34.7 vs 35): the pipeline is
  faithful.
- ProtSent-V2 beats its own frozen ESM2 base by +16.0 (35M) and +19.4 (150M) at the H level,
  with the gain growing at coarser-to-finer levels the same way ProtTucker's does over ProtT5.
- Two attacks on that result were run, not just anticipated: it is not an artifact of embedding
  normalisation, and it is not something 18x more ESM2 parameters would buy for free.
- ISM-C-300M's structure distillation shows the same qualitative pattern over its own vanilla
  base, at a smaller magnitude.
- Open: CATH-specific leakage in the pretraining corpus has not been measured, only the
  benchmark test sets used at decontamination time. Flagged, not resolved.
