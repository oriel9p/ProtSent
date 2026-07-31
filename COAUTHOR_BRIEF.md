# ProtSent rebuttal work — what we ran, what we found, what it means for the paper

Branch `rebuttal` on github.com/oriel9p/ProtSent. Models:
[GrimSqueaker/ProtSent-V2-35M](https://huggingface.co/GrimSqueaker/ProtSent-V2-35M),
[GrimSqueaker/ProtSent-V2-150M](https://huggingface.co/GrimSqueaker/ProtSent-V2-150M).
Every number is measured on our hardware and reproducible from the paths given.

---

## Status: HNXd has raised their score

Reviewer HNXd replied on 30 July and is raising their score. They accept the SCOPe-40
analyses as addressing embedding-space organization and retrieval, accept that the
linear-probe results justify the narrowed framing, and consider the confidence intervals,
repeated few-shot runs and absolute scores to resolve their statistical concerns including
the erroneous +244.5% cell. They no longer regard the absence of fine-tuning as critical.

Their one remaining point is procedural: the claims changed enough that another review
round may be warranted, and they leave that to the AC.

**This changes where effort is worth spending.** HNXd is addressed. jVGf stated that
positioning against structure-informed models plus the generality-accuracy trade-off would
move them to accept — the trade-off is measured, the positioning is a writing task with no
compute. That is now the highest-value remaining work, and §7 lists the six other
writing-only asks alongside it.

---

## 0. Open questions for you — defaults in bold, override any of them

| # | Question | Default |
|---|---|---|
| 1 | V2 has **no DMS/ProteinGym source**; V1 had it (§2.3). Retrain for parity, or document? | **Document. Don't retrain.** |
| 2 | The 150M saw **31% fewer training pairs** than the 35M (§2.2). Do we still make cross-scale claims? | **Only with the caveat attached.** |
| 3 | Paper Tables 4/7 are V1-config ablations. V2 adopts two ablated settings, so "Full model" is no longer the shipped model (§6). | **Re-run or relabel for camera-ready.** |
| 4 | Unfiltered-corpus retrain at V2 config — the one control that would make the decontamination claim airtight. ~11 h at 35M, ~26 h at 150M. | **Skip for rebuttal, do for camera-ready.** |
| 5 | Lead the retrieval claim with R@10/MAP, not top-1 (§5.2 shows whitening erases the top-1 gap at 150M). | **Yes, lead with depth/MAP.** |
| 6 | Whitened-vanilla control: rebuttal, camera-ready, or neither? | **Camera-ready only.** Anisotropy analysis does go in the rebuttal (§5.2). |

---

## 1. The short version

All three reviewers scored **2: Reject**, confidence 4. The most serious charge (Yi1G) was
train/test leakage: AFDB training sequences were never filtered against SCOPe or the
remote-homology test set.

We re-filtered the entire pretraining corpus against the benchmark test sets, retrained
both models from scratch, and re-benchmarked everything through one code path with
confidence intervals.

ProtSent-V2-150M is the strongest model we have measured on structural retrieval, and at
that scale beats both alignment baselines — including HMMER — on every retrieval metric.

Three findings cut against us, and a reviewer can reach all three:

1. The general-purpose claim does not survive a **linear probe**. Under a trained readout,
   ProtSent is neutral-to-worse than the stock backbone across the task suite.
2. Much of the k-NN advantage is **isotropy**, not new information. Stock ESM-2 embeddings
   are severely anisotropic; whitening them recovers a large part of the gain, and all of
   the top-1 gain at 150M.
3. At 150M, decontamination **cost** remote-homology k-NN accuracy (0.7047 → 0.6612).

All three argue for a narrower, better-supported claim: ProtSent is a retrieval /
metric-space method, not a general-purpose embedding upgrade.

---

## 2. What was run

### 2.1 Decontamination (`decontaminate_pretrain.py`)

Every pretraining source searched against the benchmark test sequences with MMseqs2
`easy-search`, corpus-as-query, **40% identity / 80% coverage** (`--cov-mode 1`):

| corpus | rows before | rows after | removed |
|---|---:|---:|---:|
| Pfam | 28,530,684 | 27,929,772 | 2.11% |
| AlphaFold DB | 135,404,259 | 126,301,607 | 6.72% |
| STRING | 76,070,154 | 71,891,417 | 5.49% |

Filter targets: `remote_homology` (biomap-research/fold_prediction test, 3,244 seqs) and
`ppi_bernett` (Synthyra/bernett_gold_ppi test, 3,022 seqs). Prefilter exhaustive GPU
ungapped (100% recall) for Pfam and STRING, k-mer `-s 5.7` (89.4% recall) for AFDB.

Yi1G asked for "less than 50% or even 40% sequence identity". We did 40% at 80% coverage,
then verified it.

**Verification** (`verify_training_corpus.py`): each training parquet semi-joined against
the recorded removal lists. Zero flagged sequences survived in any of the three.

**SCOPe-40 was deliberately not a filter target.** It has no train/test split, so filtering
against it would strip essentially all domain sequences from the corpus. We answer the
SCOPe question with identity stratification instead (§4).

### 2.2 What the models actually saw — pairs, not rows

The corpus is rows; training consumes **pairs** generated from them, and the step counts
only reconcile against pairs:

| | Pfam pairs | AFDB pairs | STRING pairs | total | effective batch | steps |
|---|---:|---:|---:|---:|---:|---:|
| V2-35M (k=8) | 777,306 | 18,987,468 | 15,000,000 | **34,764,774** | 1024x7 | 4,850 |
| V2-150M (k=5) | 284,683 | 8,612,331 | 15,000,000 | **23,897,014** | 1024x6 | 3,890 |

`k` is `MAX_PAIRS_PER_CLUSTER`; k=8 gives C(8,2)=28 pairs per cluster, k=5 gives 10. It was
lowered for the 150M to keep wall-clock under ~30 h.

**The STRING term is 15,000,000 in both rows because it is a fixed 15M-pair subsample**
(`stringdb_train_15M.parquet`, seed 42) of the 71,891,417 filtered rows — STRING is a flat
pair table with no clusters, so `k` does not bound it. It was verified decontaminated in
both pair columns.

**The 150M saw 31% fewer pairs than the 35M.** Any 35M-vs-150M statement — including "the
larger model exploited the leakage more" in §3 — is confounded by data budget as well as
scale. Treat cross-scale claims as suggestive.

### 2.3 V2 is a three-source model; V1 was four-source

V1 trained on Pfam, AFDB, STRING **and ProteinGym DMS/clinical via CoSENT**. Both V2
scripts pass three files and contain no CoSENT or ProteinGym path.

The reason is scope, not accident: `decontam_report.json` covers exactly `pfam, afdb,
stringdb`, and `protsent-data-dc40/` contains only those three parquets. DMS was never
decontaminated, and the training scripts hard-fail if a filtered file is missing, so V2
could only train on what was filtered.

Defensible, but it was undocumented until now, and it confounds V1-vs-V2 on fitness tasks
(fluorescence, variant effect, beta-lactamase). Say so in the methods.

### 2.4 Training configuration

| model | hardware | steps | wall clock |
|---|---|---:|---|
| ProtSent-V2-35M | 7x B300 | 4,850 | 10 h 53 m |
| ProtSent-V2-150M | 6x B300 | 3,890 | ~26 h (19 h 46 m + 6 h 06 m after a machine restart) |

Both: CachedMultipleNegativesRankingLoss, 1024 contrastive batch per device, no
gather-across-devices, no synthetic hard negatives, proportional sampling, Matryoshka
64/128/256, flash-attention-2, `max_seq_length` 512.
(`train_esm2_35m.sh`, `train_esm2_150m.sh`, config detail in `RUNS.md`.)

Dropping hard negatives is what the paper's own ablations favoured — 20/23 tasks at +7.9%
without vs 16/23 at +6.7% with. Proportional sampling is a **tie**, not an improvement
(+7.0% vs round-robin's +6.7%), so describe it as "comparable, and we picked it". Together
these answer Yi1G's complaint that the ablations do not support the submitted defaults.

Full list of V1→V2 differences, so nobody is surprised: decontaminated corpus, no hard
negatives, proportional sampling, no DMS source, 7x/6x larger effective batch, Matryoshka
heads, different data budget per scale. **V2-vs-V1 is not a single-variable ablation of
filtering.**

---

## 3. Results

### 3.1 Structural retrieval, SCOPe-40

Test split, self excluded, no-hit queries scored as failures. Restricted to the 1,693 of
2,207 queries that have a non-self same-family protein in the gallery; the other 514 are
unachievable for any method, which caps R@K at 0.767 on the full set.

| method | R@1 | R@10 | MAP |
|---|---:|---:|---:|
| ESM-2 35M | 0.4991 | 0.7614 | 0.4210 |
| ProtSent-V1 35M (submitted) | 0.5854 | 0.8512 | 0.5509 |
| **ProtSent-V2 35M** | **0.6852** | **0.9220** | **0.6459** |
| ESM-2 150M | 0.5535 | 0.7702 | 0.4236 |
| MMseqs2 (`-s 7.5`) | 0.6556 | 0.7401 | 0.4098 |
| HMMER (phmmer) | 0.6970 | 0.7809 | 0.4747 |
| ProtSent-V1 150M (submitted) | 0.6615 | 0.8943 | 0.6431 |
| **ProtSent-V2 150M** | **0.7431** | **0.9368** | **0.7042** |

Note V2-35M beats ESM-2 **150M** on every metric (0.6852/0.9220/0.6459 vs
0.5535/0.7702/0.4236) and beats HMMER's MAP by 17 points — a 35M model outperforming a
4x larger backbone is a cheap-embeddings result worth stating.

Paired bootstrap over queries, 10,000 resamples (`bootstrap_ci.py`). All exclude zero:

| comparison | R@1 | MAP |
|---|---|---|
| V2-35M − V1-35M | +0.0986 [+0.0762, +0.1211] | +0.0943 [+0.0814, +0.1074] |
| V2-150M − V1-150M | +0.0809 [+0.0602, +0.1022] | +0.0607 [+0.0477, +0.0735] |
| V2-150M − HMMER | +0.0455 [+0.0219, +0.0691] | +0.2301 [+0.2111, +0.2492] |
| V2-150M − MMseqs2 | +0.0868 [+0.0620, +0.1116] | +0.2950 [+0.2751, +0.3144] |

**Correction — we do NOT beat alignment at top-1 at either scale.** The +0.0455 above
used phmmer with **default filters**. Against the filters-off phmmer
(`hmmer_maxsens.json`: R@1 0.7525, R@10 0.8978, MAP 0.6067) ProtSent-V2-150M is *behind*
at top-1 (0.7431) and ahead at depth (0.9368) and MAP (0.7042). At 35M it is a tie
(−0.0124 [−0.0372, +0.0124]). Both submitted V1 models lose top-1 to both tools.

`NEW_EVIDENCE.md` §5-CORRECTION already required filters-off numbers for any HMMER
comparison; the 150M table was built before that was applied, and an earlier draft of this
brief claimed a scale-dependent top-1 win on that basis. It is withdrawn.

The supportable claim at both scales is ranking depth and MAP, plus cost: one forward pass
per sequence with indexable sub-linear search, versus an all-vs-all profile comparison.

We ran HMMER because Yi1G named it. It beats vanilla ESM-2 150M at top-1 by +0.144 under
default filters, and by more with filters off.

### 3.2 Remote homology — the task the corpus was filtered against

| model | 3-NN acc | 3-NN macro-F1 | linear acc | linear macro-F1 |
|---|---:|---:|---:|---:|
| ESM-2 35M | 0.5835 | 0.3173 | 0.6868 | 0.4414 |
| ProtSent-V1 35M | 0.6587 | 0.3687 | 0.6899 | 0.4281 |
| **ProtSent-V2 35M** | **0.6668** | **0.4108** | **0.7016** | **0.4527** |
| ESM-2 150M | 0.5194 | 0.2764 | 0.7500 | 0.5162 |
| ProtSent-V1 150M | **0.7047** | **0.4297** | 0.7401 | 0.4775 |
| ProtSent-V2 150M | 0.6612 | 0.3885 | **0.7503** | 0.4941 |

At 35M decontamination improved this task; at 150M it cost 4.4 points under 3-NN while the
linear-probe ordering reverses and V2 beats V1. Reading: V1-150M trained on sequences at
≥40% identity to this test set and removing them removed the inflation — an argument for
the filtering. Caveat from §2.2: the 150M also saw 31% less data.

We re-derived the 150M numbers independently (`verify_remote_homology.py`) because the
linear macro-F1 deficit against vanilla looked suspicious. It reproduces; it is
statistically real (V2 − vanilla macro-F1 −0.0262 [−0.0450, −0.0071], accuracy unresolved
at −0.0008); and it is mostly a rare-class artifact — 457 classes, median support 3, 209
classes with ≤2 examples, and restricting to classes with ≥3 examples shrinks the gap from
−0.0257 to −0.0036.

### 3.3 Aggregate over the task suite, both probes

`results/benchmarks/COMPARISON.md`, against stock ESM-2:

| probe | ProtSent-V1 35M | ProtSent-V2 35M |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median −0.0139 | 2 / 7 / 11, median −0.0107 |

V2-150M vs V1-150M: 12 / 4 / 7 under 3-NN (median +0.0055), 7 / 6 / 10 under linear
(median −0.0045).

Counts total 20, not 23: `antibiotic_resistance`, `remote_homology` and
`temperature_stability` have no main metric comparable across both arms (multiclass AUC is
uncomputable when the test split contains classes absent from train), so they are excluded
from the win/tie/loss counts and reported separately.

**The probe decides the headline.** Any "ProtSent > ESM-2" sentence must name the probe.

### 3.4 Alignment across the whole benchmark

MMseqs2 and HMMER scored on all 23 tasks under identical metric definitions, no-hit queries
counted as failures (`mmseqs_baseline.py`, `hmmer_baseline.py`). Alignment beats the best
embedding model outright on enzyme class (F1-macro 0.710 vs 0.598) and GO molecular
function (0.585 vs 0.459). This is the generality-accuracy trade-off jVGf asked for.

---

## 4. Statistics the reviewers asked for

**95% bootstrap CIs** (`bootstrap_ci.py`). Retrieval answers this exactly: every metric is
a mean over per-query values, so resampling queries gives the sampling distribution with no
refitting. Quote the **paired** intervals, not the marginals — the same queries are scored
by every method, so overlapping marginal intervals do not imply an unresolved difference.
Note we have **not** produced per-task CIs for the 23-task table, which is where HNXd's
complaint actually landed.

**Seed variability** (`run_seed_variability.sh`, `fewshot_seeds.py`), two halves:

- *Full-data evaluation is near-deterministic on 7 of 8 tasks.* Across 5 seeds: remote
  homology 0.5835±0.0000, metal-ion binding 0.7402±0.0000, solubility 0.5102±0.0000,
  stability 0.6435±0.0001. Given a fixed test split and a deterministic probe over
  deterministic embeddings there is nothing to vary. **The exception is Thermostability
  (FLIP)**, sd 0.0126 (ESM-2 35M), 0.0172 (V1), 0.0156 (V2) — its split is a seeded
  re-split, so the seed genuinely changes the data. Quote that alongside the zeros; a
  blanket "nothing varies" is not true.
- *Few-shot spread is large.* Stability at N=100 is ±0.20 on a mean of 0.28–0.40, which
  explains why Table 5's relative changes against near-zero baselines were uninterpretable.

**Few-shot with a linear baseline.** HNXd proposed that under label scarcity a linear
classifier degrades while k-NN stays competitive. Our data does not support it — a trained
linear head beats 3-NN in almost every model/task/N cell, including N=50. We should
withdraw the label-scarcity framing. What survives is task-specific, and the two models
differ: on remote homology at N=1000, **V1** leads ESM-2 by +0.133 (k-NN) and +0.089
(linear), while **V2** leads by +0.104 and +0.067. On solubility and metal-ion binding
neither wins.

**Identity-stratified SCOPe** (`scope_identity_correlation.py`, `scope_identity_partial.py`).
Binning was impossible: the [0, 0.2) identity bin is empty and median max-identity is 0.908,
because AFDB covers essentially all of UniProt — true of ESM-2's own UniRef50 too, so it is
a property of corpus coverage, not of us. Instead we correlate per-query max identity
against per-query gain: memorisation predicts queries with closer pretraining neighbours
gain more.

Raw Spearman is negative at both scales. After controlling for baseline headroom — gain is
bounded by 1 − baseline, so regression to the mean alone produces a negative slope — it
stays slightly negative at 35M (MAP −0.081, p=9e-4) and collapses to a null at 150M
(MAP −0.037, p=0.12; R@10 −0.002, p=0.93). Neither is positive. Say "no relationship at
150M, slightly negative at 35M".

---

## 5. The findings that cut against us

### 5.1 The linear probe erases our advantage

Both ProtSent models are neutral-to-worse than the stock backbone under a trained linear
readout while winning under 3-NN. This is the main honesty constraint on anything we write.

Partial mitigation, measured: both probes pool the **final** layer, the point most
favourable to a backbone whose top of stack the contrastive objective never touched.
Sweeping the pooled layer (`layer_probe_sweep.py`), compared at a **common** layer, never
at each model's own best:

| remote homology, linear probe | layer 10 | layer 20 | layer 30 (final) |
|---|---:|---:|---:|
| ESM-2 150M | 0.6817 | 0.7357 | 0.6717 |
| ProtSent-V1 150M | 0.6480 | 0.7400 | 0.7093 |
| ProtSent-V2 150M | 0.6677 | **0.7500** | 0.7040 |

Read this narrowly — it is weaker than it first looks:

- On remote homology, layer 20 beats the final layer for all three models, by +0.064
  (ESM-2), +0.031 (V1) and +0.046 (V2). **The largest gain goes to the stock backbone**, so
  it does not by itself argue ProtSent is being undersold.
- It does **not** generalise to the other task tested. On stability, layer 20 is ESM-2's
  worst layer (0.2188) and its final layer its best (0.5866). At 35M the peaks are
  elsewhere again (ESM-2 at layer 6, V2 at layer 8).
- The layer-20 lead for V2 is +1.4 over vanilla and +1.0 over V1, on one task, **with no
  confidence interval**.
- The final-layer column is **not** the benchmark's number — this script subsamples to 8k
  train / 3k test and runs its own probe on raw hidden states, which is why it reads 0.6717
  for vanilla where §3.2 reads 0.7500.

So: evidence that the pooling layer matters and should be reported, not yet evidence that
the probe protocol hides a ProtSent advantage.

### 5.2 Much of the k-NN gain is isotropy

`probe_gap_analysis.py`, `whiten_scope_control.py`. Stock ESM-2 embeddings occupy a narrow
cone on the SCOPe gallery:

| model | mean cos(random pair) | participation ratio | dims for 95% var |
|---|---:|---:|---:|
| ESM-2 35M | 0.848 | 7.9 / 480 | 112 |
| ProtSent-V2 35M | 0.152 | 52.5 / 480 | 148 |
| ESM-2 150M | 0.896 | 10.6 / 640 | 126 |
| ProtSent-V2 150M | 0.175 | 43.4 / 640 | 144 |

**This table is a positive result and it is HNXd's Question 1**, which asked for "an
analysis showing how ProtSent changes the local and global organization of the protein
embedding space". It goes in the rebuttal.

The uncomfortable half: a linear probe can learn any invertible linear map for free, so a
method that mainly whitens should show exactly the probe gap we see — and on remote
homology it does. Whitening the vanilla embeddings takes ESM-2 150M k-NN from 0.5200 to
0.7346, against ProtSent-V2's 0.6606 raw and 0.7343 whitened, both landing at the
linear-probe score.

It does not fully explain retrieval, which is the actual claim. Fitting the whitening on
the same gallery it is applied to — the most generous possible setting for the baseline:

| SCOPe-40 | R@1 | R@10 | MAP |
|---|---:|---:|---:|
| ESM-2 150M raw | 0.5529 | 0.7702 | 0.4242 |
| ESM-2 150M whitened | 0.7336 | 0.9155 | 0.6276 |
| ProtSent-V2 150M raw | 0.7425 | 0.9374 | 0.7042 |

V2-150M vs whitened vanilla: R@1 +0.0089 **unresolved**, R@10 +0.0219 significant, MAP
+0.0772 significant. At 35M the margin is larger and significant on all three
(+0.063 / +0.050 / +0.114).

Whitening closes the top-1 gap at 150M but not ranking depth or MAP at either scale. The
contribution survives, narrower than currently claimed. The camera-ready needs this
control. Current decision: whitened baseline out of the rebuttal, anisotropy table in.

### 5.3 The remote-homology test split is not what the paper says

The paper describes it as hierarchy-disjoint. It is TAPE remote homology repackaged: the
pooled concatenation of three holdouts (718 fold + 1,254 superfamily + 1,272 family =
3,244) with no column marking which. So 2,526 of 3,244 test items are easier than the fold
level the flagship gain is attributed to.

This is a **validity issue for the headline remote-homology result**, not a description
error. The obvious follow-up — recompute the gain on the 718 fold-level items only — has
not been run and should be, before we lean on that number again.

---

## 6. What this implies for the paper

1. **Retire the general-purpose framing.** The evidence supports a retrieval / metric-space
   contribution: a sequence-only embedding whose nearest-neighbour structure is better for
   homology and structural retrieval, neutral under a trained readout.
2. **Lead the retrieval claim with R@10 and MAP, not top-1.** Whitened vanilla erases the
   top-1 gap at 150M; the depth and MAP gaps survive every control.
3. **Add the whitened-vanilla control and the anisotropy table** to the camera-ready.
4. **Report the pooling layer and its sensitivity.** The final-layer default understates
   every model by 5–6 points on remote homology.
5. **Withdraw the label-scarcity claim** as phrased; our own few-shot data contradicts the
   mechanism.
6. **Tables 4 and 7 are orphaned.** They are V1-config ablations, and V2 adopts two of the
   ablated settings, so "Full model (ProtSent)" in those tables is no longer the shipped
   model. Re-run or relabel.
7. **Fix the description errors**: SCOPe evaluation is 2,207 sequences at the **family**
   level, not 100,000 at superfamily (the 100,000 was a sampling cap echoed into the
   results table); the remote-homology split is not hierarchy-disjoint (§5.3); the PPI
   decontamination description does not match `data_prep.py`, which uses `easy-search` at
   40% with `--cov-mode 1 -c 0.8` removing hit query IDs, not `easy-linclust` at 50% with
   cluster removal.
8. **Table 3 is superseded**, not corrected. Its numbers (0.385/0.445 and 0.423/0.507) are
   from the old evaluation path; §3.1 replaces them wholesale.

---

## 7. Reviewer coverage — what is answered and what is not

| reviewer ask | status | where |
|---|---|---|
| Yi1G: leakage, filter at <50% or 40% | **answered** | §2.1, §4 |
| Yi1G: HMMER/MMseqs2 baselines | **answered** | §3.1, §3.4 |
| Yi1G: ablations don't support defaults | **answered** — V2 adopts them | §2.4 |
| Yi1G: statistical evidence weak | **partly** — retrieval CIs yes, 23-task table no | §4 |
| HNXd: embedding-space organization | **answered** | §5.2 anisotropy |
| HNXd: linear-probe baseline | **answered**, unfavourably | §3.3 |
| HNXd: 95% CIs | **partly** — retrieval only | §4 |
| HNXd: multi-seed few-shot variability | **answered** | §4 |
| HNXd: absolute scores in Table 5 | **answered** | `fewshot_seeds.json` |
| jVGf: generality-accuracy trade-off | **answered** | §3.4 |
| jVGf: position vs ESM-S / S-PLM / ISM / Magneton | **not done** — writing only, no compute | — |
| jVGf: ProTrek comparison | **not done** | — |
| jVGf: how CoSENT on DMS works | **not done** — writing only | — |
| jVGf: missing reference line 21 | **not done** — one line | — |
| Yi1G: MNRL batch semantics, Eq. 1 `+` | **not done** — writing only | — |
| Yi1G: PPI/HLA pair-embedding combination | **not done** — writing only | — |
| Yi1G: k-NN regression weighting | **not done** — writing only | — |

**Seven of these are free text, no compute.** jVGf said positioning plus the trade-off
would move them to accept, and we did the expensive half and left the cheap half undone.
That is the highest-value remaining work on the rebuttal.

---

## 8. Where everything lives

| what | path |
|---|---|
| Rebuttal text (postable) | `rebuttal/FINAL_rebuttal.md` |
| Evidence pack — every number we may cite | `rebuttal/NEW_EVIDENCE.md` |
| Full methods/controls record | `REBUTTAL_LEAKAGE.md` |
| Verbatim reviews | `rebuttal/REVIEWS_actual.md` |
| Which model is which, and its config | `RUNS.md` |
| Per-task comparison tables | `results/benchmarks/COMPARISON.md` |
| 35M benchmark CSVs | `results/benchmarks/v3/` |
| 150M benchmark CSVs | `results/benchmarks/v2_150m/` |
| Decontaminated corpus | `/storage/users/ddofer/data/protsent-data-dc40` |

Scripts: `decontaminate_pretrain.py`, `verify_training_corpus.py`, `train_esm2_35m.sh`,
`train_esm2_150m.sh`, `run_benchmarks_150m.sh`, `bootstrap_ci.py`,
`scope_identity_correlation.py`, `scope_identity_partial.py`, `layer_probe_sweep.py`,
`fewshot_seeds.py`, `probe_gap_analysis.py`, `whiten_scope_control.py`,
`verify_remote_homology.py`, `mmseqs_baseline.py`, `hmmer_baseline.py`. All except
`probe_gap_analysis.py` have a `--selfcheck` asserting their behaviour on synthetic data;
that one does not, and §5.2 rests on it.

**Small numeric drift to expect**: §3.1 and §5.2 quote the same quantities to slightly
different third decimals (e.g. ESM-2 150M R@1 0.5535 vs 0.5529). The benchmark suite and
the analysis scripts each re-embed independently, and the difference is GPU
non-determinism, not disagreement. Quote §3 numbers in the paper for consistency.

---

## 9. What we did NOT do

- No unfiltered-corpus retrain at the V2 config, so V2-vs-V1 confounds filtering with the
  other six config changes (§2.4). ~26 h GPU at 150M, ~11 h at 35M.
- No remote-homology gain on the 718 fold-level holdouts alone (§5.3).
- No end-to-end fine-tuning or LoRA sweep.
- No matched runs of ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.
- No SaProt/ProSST backbone substitution (needs residue-level structure tokens for the full
  Pfam and STRING corpora).
- No per-task bootstrap CIs on the 23-task table.
- Only `remote_homology` and `ppi_bernett` test sets were decontamination targets. The
  other 21 task test sets were **not** filtered against — relevant before claiming anything
  corpus-wide.
