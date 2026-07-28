# ProtSent — pretraining/benchmark leakage: methods, controls, and baselines

Working document for the reviewer question on leakage between the **structural
tasks** (SCOPe-40 retrieval, remote homology) and the ProtSent **pretraining
corpora** (AFDB, Pfam, STRING).

Status: decontamination complete and audited. ProtSent-v2-35M retraining on the
filtered corpus in progress. Sections marked *PENDING* are not yet filled.

---

## 1. What was actually filtered

All three pretraining corpora were filtered against benchmark **test** sequences
with MMseqs2 `easy-search`, at **40% sequence identity / 80% coverage of the test
sequence** (`--min-seq-id 0.4 --cov-mode 1 -c 0.8 --alignment-mode 3 -e 1e-3`).

Orientation is deliberate: **pretraining corpus = query, benchmark test set =
target**. Each pretraining sequence only needs *any* hit to be dropped, so the
default `--max-seqs 300` prefilter cap is harmless. The reverse orientation would
silently truncate at 300 hits per test sequence and under-remove.
`--cov-mode 1` is coverage of the *target* (the test sequence), so a long AFDB
protein that merely *contains* a test-length domain is still caught.

| corpus | filtered against | rows before | rows after | removed | % |
|---|---|---:|---:|---:|---:|
| AFDB | `biomap-research/fold_prediction[test]` (3,244 seqs) | 135,404,259 | 126,301,607 | 9,102,652 | **6.72%** |
| Pfam | `biomap-research/fold_prediction[test]` (3,244 seqs) | 28,530,684 | 27,929,772 | 600,912 | **2.11%** |
| STRING | `Synthyra/bernett_gold_ppi[test]` (3,022 seqs) | 76,070,154 pairs | 71,891,417 pairs | 4,178,737 | **5.49%** |

Additional detail: AFDB 117,549,800 unique sequences searched → 7,414,137 leaked;
clusters 819,790 → 817,282. Pfam 600,899 leaked unique sequences; families
29,395 → 29,368. STRING 14,567,625 unique sequences searched → 319,282 leaked; a
pair is dropped if **either** partner leaked.

Removing members can strand new singleton clusters (which then produce zero
pairs), so the singleton drop and the cluster-contiguity sort were re-applied
after filtering.

Artifacts: `/storage/users/ddofer/data/protsent-data-dc40/` — filtered parquets,
`decontam_report.json`, dataset card, and a `decontam/` audit subfolder holding
every hit with per-hit `fident`/`qcov`/`tcov` plus the leaked-sequence lists.

### Controls

| control | result |
|---|---|
| Negative — 1,000 random seqs from *filtered* AFDB vs fold test | **0 hits** |
| Negative — same, re-run with GPU prefilter (stricter; AFDB's production run used k-mer `-s 5.7` at 89.4% recall) | **0 hits** |
| Negative — 1,000 random seqs from *filtered* Pfam vs fold test | **0 hits** |
| Negative — 1,000 random seqs from *filtered* STRING vs bernett test | **0 hits** |
| Positive — fold test vs itself | **3,244 / 3,244** self-hit at `fident = 1.000` |
| Positive — bernett test vs itself | **3,022 / 3,022** self-hit at `fident = 1.000` |

The positive controls establish that the flags do what is claimed; the negative
controls establish that the leakage is actually gone from the shipped corpus.

---

## 2. SCOPe-40: why it is NOT used as a filter target

`tattabio/scope40_test` has **no train/test split**. The benchmark
(`benchmark_tasks.py:416-425`) uses `train_split="train"`, `test_split="train"`:
it is leave-one-out **self-retrieval** over the whole set, scored as family-level
Recall@10 with self-matches excluded.

Filtering AFDB/Pfam at 40% identity against *all* of SCOPe would therefore
remove essentially every structured domain from the pretraining corpus. That is
not a decontamination, it is corpus destruction, and the resulting model would
not be informative about anything.

The protocol used instead has three parts:

**(a) Identity stratification.** For every SCOPe-40 sequence, compute the maximum
sequence identity to any pretraining sequence, then report Recall@10 binned by
that identity (<20% / 20-40% / 40-70% / >70%). If ProtSent's advantage over
ESM2-35M holds in the low-identity bins, the gain is not memorisation.
*PENDING — see §5.*

**(b) Both corpora reported.** SCOPe-40 is evaluated with the model trained on
the fold_prediction-filtered corpus **and** on the unfiltered one, so the effect
of decontamination on this task is directly visible. *PENDING — §5.*

**(c) Baseline parity.** Every PLM baseline compared against (ESM2-35M and
friends) is pretrained on UniRef50, which contains all of SCOPe. The
contamination is common to every model in the table, so the **delta** is the
quantity being measured, and it is measured fairly.

> Open item: confirm the precedent citation (ProtTucker / TM-Vec / DGEB) for
> what the field accepts here before quoting it in the response letter. Not yet
> verified against the primary sources.

---

## 3. MMseqs2-only baseline

Reviewer-relevant question: *how much of the structural performance is just
sequence similarity?* These numbers answer it by scoring the same tasks with
alignment instead of embeddings, under the **same metric definitions**.

Implementation: `mmseqs_baseline.py`. For retrieval it reproduces
`evaluate_retrieval()` (`protein_benchmark_suite.py:1863-1907`) exactly —
family-level Recall@K, self excluded — with cosine-NN rank replaced by MMseqs2
bitscore rank. For classification, per-class score = max bitscore over that
class's training sequences, giving a dense score vector so AUC stays comparable
rather than degenerating to hard 1-NN accuracy. For regression, 1-NN by bitscore.

Search flags: `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`.

**Queries with no hit are counted as failures, not dropped.** "Found no
homolog" is a real failure mode of sequence search and is exactly the gap an
embedding model should close; hiding it would flatter the baseline.

### Structural tasks

| task | metric | MMseqs2 | secondary |
|---|---|---:|---|
| SCOPe-40 retrieval | Recall@10 | **0.5637** | R@1 0.5029, R@30 0.5641 |
| Remote homology (fold) | AUC | **0.6523** | Acc 0.4365, F1-macro 0.2064, 457 classes |

Remote homology hit coverage: **0.8893** — 359 of 3,244 test sequences find no
homolog at any sensitivity, and score 0.

### Other benchmarks

*PENDING* — sweep running across the binary / multiclass / regression /
multilabel tasks, plus a low-sensitivity (`-s 5.7`) variant to document the
speed-vs-accuracy tradeoff.

Excluded by construction: `ppi_bernett` (pair-input, not single-sequence), all
`proteingym_*` (mutant-vs-WT scoring), `chezod_disorder` (local data dir).

> **Comparability caveat.** `mmseqs_baseline.py` evaluates on each task's
> declared **test** split. `protein_benchmark_suite.py` defaults to
> `--eval_split validation`, which falls back to 4-fold CV on train when a task
> has no validation split. Those protocols are **not** comparable — the model
> side must be run with `--eval_split test` for the table to be honest.

---

## 4. ProtSent-v2-35M training configuration

Retrained on the filtered corpus. Every value below was measured on this
hardware (8× NVIDIA B300, sm_103), not assumed.

| setting | value | justification |
|---|---|---|
| model | Synthyra FastPLM ESM2-35M | |
| attention | `flash_attention_2` | 10.48 s/it vs sdpa 16.79 s/it in the real loop — sdpa is 60% slower |
| loss | `cached_mnrl`, mini-batch 256 | plain MNRL OOMs at ~260 GiB even at bs 256 under bf16 autocast |
| batch size | 1024 / device | CachedMNRL bounds memory by mini-batch, so this is free |
| gather across devices | **off** | 1024 in-batch negatives per rank already matches the paper; avoids allgather |
| dataset sampler | `proportional` | round-robin truncates to the smallest corpus |
| synthetic hard negatives | **off** | as specified |
| torch.compile | off | measured 8.87 vs 8.89 s/it — no effect |
| gradient checkpointing | off | not needed at 35M; it also forced `dataloader_num_workers=0` |
| Matryoshka dims | 64 / 128 / 256 (+ native 480) | |
| steps | 4,244 (1 epoch, proportional) | pfam 759 + afdb 18,542 + string 14,648 batches |
| throughput | ~8.4 s/it → **~10 h** | |

**FlashAttention-3 is not usable on this hardware.** The pinned
`kernels-community/flash-attn3` build contains `sm_80, sm_90a` only and fails on
sm_103 with `CUDA error: no kernel image is available for execution on the
device`. FA3 is Hopper-only; Blackwell needs FA4, which FastPLM's
`AttentionBackend` enum does not contain. FA2 is used instead.

### Data budget caveat — state this if asked

To fit a ~12 h budget, STRING was subsampled to **15M of 71.9M** filtered pairs
(seed 42), and `--max_pairs_per_cluster = 8` (which samples 8 sequences per
cluster and emits all C(8,2) = 28 pairs). Final corpus: AFDB 18,987,468 +
Pfam 777,306 + STRING 15,000,000 ≈ **34.8M pairs**.

AFDB and Pfam clusters are **all** visited — a substantial improvement over the
earlier round-robin run, which exhausted its pair budget within the first ~2% of
the group-sorted corpus and therefore only ever saw the lowest-sorted clusters.
But the paper cannot claim "trained on the entire filtered corpus."

### Fixes made along the way

- `FastPLMESM2Wrapper` requested `output_hidden_states=True` and ran the
  `lm_head` on every forward, neither of which is used for embedding
  (`model_utils.py:539`).
- `save_total_limit` was hardcoded to 1 (`protein_pipeline.py:2302`).
- `--multi_dataset_sampler` left at `round_robin` truncates to the smallest
  dataset.

---

## 5. Pending

- [ ] SCOPe-40 identity stratification vs the pretraining corpus (§2a, §2b)
- [ ] MMseqs2 baseline across the remaining benchmarks + sensitivity variant (§3)
- [ ] ProtSent-v2-35M benchmark results: kNN probe and linear probe, vs ESM2-35M
- [ ] Verify the ProtTucker / TM-Vec / DGEB precedent citation (§2)
