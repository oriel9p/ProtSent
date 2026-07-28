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

**Precedent (verified).** ProtTucker (Heinzinger *et al.*, *NAR Genom. Bioinform.*
4(2):lqac043, 2022, doi:10.1093/nargab/lqac043) is the closest published analogue
and takes exactly posture (c):

- Data: CATH v4.3 sequence-unique set, CATH-S100 (123k domains). `test300`
  (300 proteins) and `val200` (200) were **randomly split off** from CATH-S100,
  constrained so that every homologous superfamily appears at most once *within*
  the held-out sets and every held-out protein carries an SSG annotation.
- Redundancy reduction: proteins sharing **>20% PIDE** with any val/test protein
  were removed **from the training set**, using MMseqs2 iterative profile search
  (`--num-iterations 3 -s 7.5 --cov-mode 0`). Result: `train66k`, lookup set
  `lookup69k`, query set `test219` (test300 minus queries with no same-H protein
  left in the lookup set).
- **The holdout is at the sequence-identity level, not the H level.** Training
  and lookup sets deliberately still contain the *same* homologous superfamilies
  as the queries — they must, since the task is transferring an H-level label
  from lookup to query by embedding kNN. So "no H-level leakage" is not what
  ProtTucker claims; it claims no >20%-PIDE sequence leakage.
- **They applied no decontamination to the underlying pLM's pretraining corpus.**
  ProtTucker is a 2-layer FNN (1024→256→tanh→128) on frozen ProtT5-XL-U50
  embeddings; ProtT5 was pretrained on BFD + UniRef50, which contains CATH in
  full. No statement addressing this overlap appears in the paper. *(Established
  by repeated search of the methods text; the journal/bioRxiv hosts are blocked
  by this cluster's firewall, so confirm by eye before quoting an absence in the
  response letter.)*
- Eval: embedding-based annotation transfer (EAT), Euclidean 1-NN from lookup to
  query, accuracy reported per CATH level (C/A/T/H); queries whose top hit came
  from a different level counted wrong; sequence-search baselines (MMseqs2,
  HMMER) scored as incorrect when no hit at E<10, and the headline claim is
  performance in the "midnight zone" (<20% PIDE).

The field's accepted standard is therefore identity-based decontamination of the
*supervised* split plus low-identity stratification — not removal of the
benchmark from the self-supervised pretraining corpus. §1 (40% PIDE filtering of
the pretraining corpora) is *stricter* than this precedent; §2a is the same
midnight-zone stratification.

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

**`hit coverage`** (reported alongside every task) is the fraction of test
queries for which MMseqs2 returned *any* alignment at E<10. The remainder are
scored against a fallback carrying no information from the search — lowest-rank
class for classification, the training mean for regression, the empty label set
for multilabel. It is the column that separates "search ran and was right or
wrong" from "search found nothing and we scored a default", so a headline metric
should always be read next to it: at coverage 1.0 the metric is a real measure of
alignment; at coverage 0.0 it is a property of the fallback and means nothing.

### SCOPe-40: head-to-head, all measured with the same code

Every row below was produced on this machine by `protein_benchmark_suite.py`
(embeddings) or `mmseqs_baseline.py` (alignment), on the same 2,207-sequence
gallery, `--eval_split test`, self-matches excluded, no-hit queries scored as
failures. Raw values: `results/benchmarks/scope40_table.json`.

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10`) | **0.5029** | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent 35M (published) | 0.4490 | **0.6529** | **0.7100** | **0.4226** |

Eligible queries only (n = 1,693 of 2,207):

| method | R@1 | R@10 | R@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 | **0.6556** | 0.7348 | 0.7354 | 0.4041 |
| ESM-2 35M | 0.4991 | 0.7614 | 0.8340 | 0.4210 |
| ProtSent 35M | 0.5854 | **0.8511** | **0.9256** | **0.5509** |

**The evaluation path reproduces the submitted paper.** Measured here versus
Table 3 as submitted: ESM-2 35M R@1 0.3829 vs 0.3833, R@10 0.5840 vs 0.5841,
R@30 0.6398 vs 0.6402, MAP 0.3230 vs 0.3235; ProtSent 35M R@1 0.4490 vs 0.4495,
R@10 0.6529 vs 0.6529, R@30 0.7100 vs 0.7100, MAP 0.4226 vs 0.4225. The MAP
agreement to four decimals confirms the paper's MAP convention is the one now
implemented in `evaluate_retrieval()`: average precision over the full ranking,
averaged over all queries, unretrieved relevant items contributing zero.

**Recall@K on this benchmark is upper-bounded at 0.7671.** Only 1,693 of 2,207
queries (76.71%) have any non-self same-family protein in the gallery; the
remaining 514 are singleton families and are unachievable for any method. State
this in every caption carrying a SCOPe recall — ProtSent 35M's R@30 of 0.7100 is
**92.6% of the attainable maximum**, not 71% of 100%.

**A tuned MMseqs2 is the strongest top-1 method in the table.** At R@1 it beats
ESM-2 35M by 12.0 points and ProtSent 35M by 5.4. ProtSent leads at every deeper
cutoff (R@10 +8.9, R@30 +14.6 over MMseqs2) and on MAP (+11.3). The defensible
claim is therefore about **ranking depth, not top-1**.

The `-s 5.7` variant reproduces a much weaker alignment baseline (R@1 0.3847,
R@10 0.4259). Any MMseqs2 comparison must state its sensitivity: at default
settings the baseline looks far worse than it is, and publishing that number
while a stronger one is reproducible from this repo would be indefensible.

### Remote homology

| metric | MMseqs2 |
|---|---:|
| AUC (457 classes, macro OvR) | 0.6523 |
| Accuracy | 0.4365 |
| F1-macro | 0.2064 |
| hit coverage | 0.8893 |

359 of 3,244 test sequences find no homolog at any sensitivity and score 0.

### Full sweep — all 24 evaluable tasks

All rows on the **test** split (`--eval_split test`), sorted by task type.
Raw JSON: `results/benchmarks/mmseqs_baseline.json`.

| task | type | metric | MMseqs2 | secondary | hit cov. | n train/test |
|---|---|---|---:|---|---:|---|
| profet_np_sp_cleaved | binary | AUC | **0.9010** | Acc 0.9228; F1m 0.9127 | 0.9555 | 2,727/337 |
| signalp_binary | binary | AUC | 0.7961 | Acc 0.9345; F1m 0.8728 | 0.7695 | 16,606/4,152 |
| metal_ion_binding | binary | AUC | 0.7239 | Acc 0.7755; F1m 0.7755 | 0.9542 | 6,000/1,332 |
| binary_subcellular_localization | binary | AUC | 0.6834 | Acc 0.7176; F1m 0.7167 | 0.8954 | 5,184/1,749 |
| peptide_hla | binary | AUC | 0.6374 | Acc 0.7761; F1m 0.7761 | 1.0000 | 57,357/8,406 |
| material_production | binary | AUC | 0.5796 | Acc 0.6577; F1m 0.5964 | 0.9643 | 23,339/4,791 |
| solubility | binary | AUC | **0.4185** | Acc 0.4183; F1m 0.4116 | 0.9505 | 62,478/2,001 |
| antibiotic_resistance | multiclass | AUC | **0.9544** | Acc 0.9821; F1m 0.9199 | 0.9948 | 2,072/1,344 |
| temperature_stability | multiclass | AUC | 0.6853 | Acc 0.8552; F1m 0.8552 | 0.9717 | 283,057/73,205 |
| subcellular_loc | multiclass | AUC | 0.6828 | Acc 0.5304; F1m 0.3988 | 0.8974 | 6,622/1,842 |
| **remote_homology** | multiclass | AUC | **0.6523** | Acc 0.4365; F1m 0.2064 (457 cls) | 0.8893 | 12,312/3,244 |
| ec_classification | multilabel | F1_Macro | 0.7103 | F1_Micro 0.8777 | 0.9900 | 13,090/1,604 |
| go_mf | multilabel | F1_Macro | 0.5850 | F1_Micro 0.6406 | 0.9555 | 22,081/3,350 |
| beta_lactamase_peer | regression | Spearman | **0.8026** | MSE 0.0380 | 1.0000 | 4,158/520 |
| variant_effect | regression | Spearman | 0.7166 | MSE 1.0326 | 1.0000 | 6,289/1,745 |
| enzyme_catalytic_efficiency | regression | Spearman | 0.6322 | MSE 17.79 | 0.9941 | 13,470/1,684 |
| stability | regression | Spearman | 0.5817 | MSE 0.3503 | 1.0000 | 53,614/12,851 |
| optimal_ph | regression | Spearman | 0.5462 | MSE 1.0000 | 0.9868 | 7,124/1,971 |
| thermostability | regression | Spearman | 0.4799 | MSE 0.0448 | 0.9933 | 5,377/1,345 |
| aav_flip | regression | Spearman | 0.4024 | MSE 11.65 | 1.0000 | 22,246/50,432 |
| fluorescence | regression | Spearman | 0.3863 | MSE 2.0808 | 1.0000 | 21,446/27,217 |
| cloning_clf | regression | Spearman | 0.1707 | MSE 0.3969 | 0.9666 | 23,375/4,791 |
| rhla_enzyme_mutations | regression | Spearman | *n/a* | MSE 0.2045 | **0.0000** | 942/511 |
| **scope40_retrieval** | retrieval | Recall@10 | **0.5637** | R@1 0.5029; R@30 0.5641 | n/a | 2,207/2,207 |

Reading of these numbers:

- **`antibiotic_resistance` (0.954) and `beta_lactamase_peer` (0.803) are close to
  saturated by alignment alone.** These are the tasks where an embedding model has
  to justify its existence; a PLM that merely matches these adds nothing.
- **`solubility` at AUC 0.4185 is *below chance*** — the nearest homolog's
  solubility label is anti-correlated. Solubility is not conserved by homology, so
  a sequence-similarity prior actively misleads. Good evidence that some tasks
  cannot be solved by retrieval.
- **`rhla_enzyme_mutations` has 0% hit coverage** and no Spearman. Its `protein`
  column holds 6-residue mutation-site strings, not proteins; MMseqs2 reports
  `No k-mer could be extracted`. Structurally incompatible with alignment search
  — not a bug, and not a task where this baseline means anything.
- `peptide_hla` inputs are pipe-joined `HLA_pseudoseq|peptide` strings (~44 chars).
  MMseqs2 treats `|` as an unknown residue. It is the *same* string the model side
  embeds, so the comparison is fair, but it is not a biologically meaningful
  alignment.

Excluded by construction: `ppi_bernett` (pair input, not single-sequence), all
`proteingym_*` (mutant-vs-WT scoring), `chezod_disorder` (local data dir),
`cafa5` (size).

### Sensitivity variant

Documenting the speed/accuracy trade, since a cheaper search was considered:

| task | metric | `-s 7.5` | `-s 5.7` | Δ |
|---|---|---:|---:|---:|
| scope40_retrieval | Recall@10 | 0.5637 | 0.4259 | **−0.1378** |
| scope40_retrieval | Recall@1 | 0.5029 | 0.3847 | −0.1182 |
| remote_homology | AUC | 0.6523 | 0.6262 | −0.0261 |
| remote_homology | hit coverage | 0.8893 | 0.6233 | **−0.2660** |

MMseqs2 search time 4.09 s → 3.40 s (scope40) and 5.34 s → 3.62 s
(remote_homology). `-s 5.7` saves ~1.7 s on a 5 s search and costs 13.8 points of
Recall@10 — plainly the wrong trade at this scale. `-s 7.5` is used throughout.

### Split protocol — read before comparing any number here to a model number

Everything above is the **test** split. The benchmark suite defaults to
`--eval_split validation`, which falls back to 4-fold CV on *train* when a task
declares no validation split, so the default is **not** comparable to this table.
`run_benchmarks_v3.sh` therefore passes `-e test` for both models.

Two specifics worth stating in the paper:

- These 6 tasks have **no validation split** and would hit CV-on-train under the
  suite default: `metal_ion_binding`, `material_production`, `subcellular_loc`,
  `antibiotic_resistance`, `cloning_clf`, `thermostability`.
- **`thermostability` has no real test split either.** Under `-e test` the suite
  takes a seeded 80/20 split of train (`eval_strategy=test_random_split`). The
  MMseqs2 row uses that same seeded split, so the pairing is self-consistent, but
  it is not an official held-out set and should not be presented as one.
- `remote_homology`'s test split (3,244) is TAPE remote homology repackaged: the
  *pooled* concatenation of TAPE's three holdouts (718 fold + 1,254 superfamily +
  1,272 family), with no column marking which. Published work reports per-holdout
  top-1 accuracy on those three separately, so **our pooled 457-class macro AUC is
  not comparable to a published TAPE number** — say so rather than let a reviewer
  attempt the comparison.

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
- [ ] SCOPe-40 for ProtSent-v2-35M (retrained on the filtered corpus), reported
      both unfiltered and identity-stratified, alongside the published
      ProtSent 35M row above so the ± decontamination effect is directly visible

---

## 6. Corrections the draft rebuttal needs (grounded, from this repo)

Each item below is a factual error or omission in `rebuttal/DRAFT_rebuttal.md`,
with the source that settles it. These are not stylistic preferences.

1. **The MMseqs2 baseline in the draft (R@1 0.3539, MAP 0.1795) is a
   default-sensitivity run.** §3 above reproduces R@1 0.5029 / MAP 0.3100 at
   `-s 7.5`, and `-s 5.7` gives R@1 0.3847 — bracketing the draft's number.
   Publishing the weaker figure while the stronger one is reproducible from
   `results/benchmarks/mmseqs_baseline.json` in the released repo is a
   self-inflicted integrity problem. Publish 0.5029 with flags stated.

2. **R@10 = R@30 = 0.3856 exactly** in the draft's table. Exact equality to four
   decimals indicates a truncated hit list, not a plateau. The measured run gives
   0.5637 vs 0.5641 — near-equal but distinct.

3. **The remote-homology test split is not hierarchy-disjoint.** The draft claims
   it is. It is TAPE remote homology repackaged: the pooled concatenation of
   TAPE's three holdouts (718 fold + 1,254 superfamily + 1,272 family = 3,244),
   with no column marking which. Two thirds is not fold-disjoint. Rely on
   corpus-level decontamination instead, and note that the pooled 457-class macro
   AUC is not comparable to published per-holdout top-1 accuracies.

4. **The PPI decontamination description contradicts the released code.** The
   draft describes `easy-linclust` at 50% identity with cluster-level removal.
   `data_prep.py` uses `easy-search` (STRING as query, Bernett test as target) at
   `--decontam_min_seq_id` (default 0.4), `--cov-mode 1 -c 0.8`, removing hit
   query IDs, not clusters — and its own docstring states `easy-search` was
   chosen deliberately because linclust loses sensitivity below ~50% identity.
   Describe what the code does. The completed 40% pass is the stronger answer
   anyway: 4,178,737 STRING pairs (5.49%) and 319,282 unique sequences removed,
   with 0-hit negative and 3,022/3,022 positive controls.

5. **"100,000 sequences" has a mechanical explanation.** It is the evaluator's
   `max_samples` cap echoed into the results table (visible in every benchmark
   CSV as `Samples 100000`), applied to a 2,207-row dataset. Saying so converts
   an apparent 45x error in reported N into a logging artifact.

6. **Task count is 23 in the draft and 24 here.** `mmseqs_baseline.json` has 24
   rows. Pick one and state the exclusions (`ppi_bernett` pair-input,
   `proteingym_*`, `chezod_disorder`, `cafa5`, `rhla_enzyme_mutations`).

7. **Use the decontamination that is already finished.** §1 above — all three
   corpora filtered at 40%/80% with negative controls at 0 hits and positive
   controls at 3,244/3,244 and 3,022/3,022 — is complete, auditable, and stricter
   than the ProtTucker precedent verified in §2. The draft concedes the leakage
   point instead of citing this work.

8. **Keep one story about R@1 across all three responses.** The measured table in
   §3 shows a tuned alignment baseline beating ProtSent 35M at top-1. Asserting a
   top-1 win to two reviewers while conceding it to a third is inconsistent, and
   the data does not support the win. The consistent, defensible claim is that
   the effect is in ranking depth (R@10/R@30/MAP), which survives both the
   alignment comparison and the decontamination subset.

9. **The n=92 strict-subset analysis has a ceiling of 57/92 = 0.620**, which the
   draft does not state, so its R@30 of 0.500 reads against an implied 1.0.
   The identity-stratified analysis (§2a) retains all 2,207 queries and has the
   statistical power the 92-query subset lacks.
- [x] MMseqs2 baseline across all 24 evaluable benchmarks + sensitivity variant (§3)
- [ ] ProtSent-v2-35M benchmark results: kNN probe and linear probe, vs ESM2-35M,
      both with `-e test`
- [ ] **Additional metrics to match the comparison papers.** The table in §3
      reports each task's declared `main_metric` plus whatever the evaluator
      emits. Papers we are compared against may report different quantities
      (e.g. per-holdout top-1 accuracy for TAPE remote homology; Foldseek/TM-Vec
      style "sensitivity up to the first false positive" for SCOPe rather than
      Recall@K; alignment/embedding-geometry diagnostics such as
      alignment-vs-uniformity, embedding anisotropy, or TM-score correlation).
      Decide which are actually needed for the response letter before adding
      them — each one is a separate evaluator, and the ones above are not
      currently computed by either `mmseqs_baseline.py` or the benchmark suite.
- [x] Verify the ProtTucker precedent citation (§2) — done, see §2
- [ ] Optional: run ProtTucker as a second baseline. **Blocked, not recommended.**
      The `ProtTucker_ProtT5.pt` checkpoint is served only from `rostlab.org` and
      `zenodo.org`, both unreachable from this cluster (NETWORK_WHITELIST.md), and
      is not mirrored on HF. Even with the weights, embedding both task sets with
      ProtT5-XL (1.2B-param encoder, 3.07M residues) is ~4-8 h of CPU. No
      published SCOPe-40 or fold-prediction number is protocol-comparable to ours
      (see below), so there is nothing to cite in its place either.
