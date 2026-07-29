# Hostile fact-check: `rebuttal/DRAFT_rebuttal_edited.md`

Every factual claim and number in the draft was traced to `rebuttal/NEW_EVIDENCE.md`,
`REBUTTAL_LEAKAGE.md`, `results/benchmarks/COMPARISON.md`, `rebuttal/PAPER_text.txt`,
or the repo's code and result files. Ranked by damage if a reviewer catches it.

**Headline:** 4 blocks of numbers in the draft exist in no source and in no artifact
this repo has ever contained. 6 claims contradict a source. The draft's strongest
available evidence — full corpus decontamination and the retrained V2 model — is
absent from all three responses.

Character counts (all currently under the 10,000 limit, but every `[[RESULT]]` still
has to fit): HNXd 4,661 · jVGf 4,350 · Yi1G 7,388.

---

## TIER 1 — UNSUPPORTED. No source anywhere. Highest damage.

### F1. The entire embedding-geometry paragraph (HNXd §1, draft line 31)

> "At 150M, ProtSent moves family silhouette from -0.148 to 0.039, NMI from 0.852 to
> 0.893, ARI from 0.165 to 0.313, and the intra/inter-family distance ratio from 0.701
> to 0.418. The Spearman correlation between embedding distance and shared SCOPe
> hierarchy depth strengthens from -0.247 to -0.561. The 35M model shows the same
> pattern. Not every metric improves: class-balanced alignment worsens."

**UNSUPPORTED — 11 numbers, zero sources.** Not in `NEW_EVIDENCE.md`. Not in
`REBUTTAL_LEAKAGE.md`. Not in `COMPARISON.md`.

Repo evidence that this was never computed:

- `grep -rIn -iE "silhouette|\bNMI\b|adjusted_rand|anisotropy|uniformity" --include="*.py" .`
  → **zero hits**. No evaluator exists.
- `grep -rI "silhouette" .` (excluding `.git`) → **zero hits** in any results file.
- `git log --all --diff-filter=A --name-only | grep -iE "geom|silhou"` → **no such file
  has ever been added on any branch.**
- There is no "shared SCOPe hierarchy depth" table anywhere; `benchmark_tasks.py:416-425`
  loads `tattabio/scope40_test` with `label_col="family"` and no hierarchy column.

The draft asserts these as completed work — *"we have now run it"* (line 18),
*"we measured the geometry directly, in a residue-only mean-pooling audit"* (line 31).
This is the single most dangerous item in the document: the reviewer explicitly asked
for geometry analysis, so this paragraph is the one they will read most carefully, and
it is the response's opening claim.

### F2. The n=155 and n=92 SCOPe leakage-subset analysis (Yi1G §1, lines 120-133)

**UNSUPPORTED — 14 table numbers plus a bootstrap conclusion, zero sources.**

- "At the 50% threshold, 155/2,207 queries remain" + the 4x3 table (0.303/0.477/0.199,
  0.329/0.581/0.319, 0.265/0.503/0.204, 0.297/0.594/0.287): no source.
- "excluding a query when either AFDB or STRING has a hit at >=50% leaves 92 queries,
  only 57 of which have a non-self family positive" + 0.250/0.250, 0.413→0.500,
  0.182→0.248, 0.207/0.239, 0.424/0.533, 0.171/0.256: no source.
- "Paired bootstrap intervals include zero for both R@1 deltas but exclude zero for R@30
  and MAP at both scales": no bootstrap output exists anywhere in the repo.

`grep -rI "155/2,207"` and `grep -rI "n=92"` return nothing outside the drafts and
`REBUTTAL_LEAKAGE.md` §6 item 9, which refers to the analysis only to criticise the
draft's presentation of it. `ls results/benchmarks/` contains no subset file; the only
SCOPe analyses on disk are `scope40_table.json` and
`scope_identity_correlation{,_v1,_v2}.json`, none of which is query-filtered by
AFDB/STRING hits.

Note also: this analysis reports 150M numbers (see F3) — a model with no artifact in
this repo at all.

### F3. Every 150M number in both tables (HNXd §1 lines 24-25, jVGf §2 lines 95-96)

**UNSUPPORTED as measured results.** There is no 150M model, no 150M checkpoint, no
150M CSV, and no 150M JSON anywhere: `find . -iname "*150*"` returns only unrelated
site-packages files; `ls models/` has no 150M entry; every CSV under
`results/benchmarks/v3/` is 35M. `NEW_EVIDENCE.md` §8 lists "No 150M model on the
decontaminated data" — but the 150M *baseline* results are equally absent from both
approved sources, which cover 35M arms only.

Worse, the Recall values do not round-trip to the submitted paper either:

| | paper Table 3 (`PAPER_text.txt:334-340`) | draft | draft rounded |
|---|---|---|---|
| ESM-2 150M | 0.423 / 0.589 / 0.644 | 0.4237 / 0.5908 / 0.6457 | 0.424 / **0.591** / **0.646** |
| ProtSent 150M | 0.507 / 0.685 / 0.724 | 0.5066 / 0.6860 / 0.7245 | 0.507 / **0.686** / 0.724 |

And **MAP 0.3249 / 0.4932 has no source at all** — paper Table 3 has no MAP column, and
`REBUTTAL_LEAKAGE.md` §3 notes the MAP convention was only reproduced during the rebuttal,
on 35M.

### F4. The MMseqs2 baseline row: 0.3539 / 0.3856 / 0.3856 / 0.1795

**UNSUPPORTED and, separately, WRONG to use — see W1.** No run in this repo produces
those values. `grep -rI "0.3539"` hits only `REBUTTAL_LEAKAGE.md:623`, where it appears
as a criticism of the draft. `results/benchmarks/mmseqs_baseline.json` is the only
MMseqs2 artifact and it does not contain them.

`R@10 = R@30 = 0.3856` to four decimals is a truncated hit list, not a plateau
(`REBUTTAL_LEAKAGE.md` §6 item 2). The measured run gives 0.5637 vs 0.5641 — near-equal
but distinct. A reviewer who notices the exact equality will conclude the baseline was
mis-run, which taints the whole "generality-accuracy trade-off" answer to jVGf.

---

## TIER 2 — WRONG. Contradicted by a source.

### W1. Publishing the weak MMseqs2 baseline (HNXd line 29, jVGf lines 92-98)

**WRONG usage.** Draft: R@1 0.3539, MAP 0.1795. Repo, `results/benchmarks/mmseqs_baseline.json`
via `REBUTTAL_LEAKAGE.md` §3 / `NEW_EVIDENCE.md` §3, flags `-s 7.5 -e 10 --max-seqs 300
--alignment-mode 3`, same 2,207 gallery, self excluded, no-hit = failure:

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 `-s 7.5` | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| MMseqs2 `-s 5.7` | 0.3847 | 0.4259 | — | — |

The draft's 0.3539 is below even the `-s 5.7` variant. `REBUTTAL_LEAKAGE.md` §6 item 1
is unambiguous: *"Publishing the weaker figure while the stronger one is reproducible
from `results/benchmarks/mmseqs_baseline.json` in the released repo is a self-inflicted
integrity problem."* The repo is public (paper footnote 3). A reviewer can run it.

The substantive consequence: with the correct baseline the *submitted* 35M model
**loses** at top-1 (0.4490 vs 0.5029) and leads only at depth (R@10 +8.9, R@30 +14.6,
MAP +11.3). Only V2 wins top-1 (0.5256). The draft's table is arranged so the reader
concludes ProtSent beats alignment at R@1. It does not — not for the model the paper
released.

### W2. "the downstream split is disjoint in its evaluation hierarchy" (Yi1G line 139)

**WRONG.** `NEW_EVIDENCE.md` §7 item 2 and `REBUTTAL_LEAKAGE.md` §6 item 3 / §3
"Split protocol": `remote_homology`'s test split is TAPE remote homology repackaged —
the **pooled** concatenation of three holdouts (718 fold + 1,254 superfamily + 1,272
family = 3,244) with **no column marking which**. Two thirds is not fold-disjoint.

The draft is repeating the submitted paper's own claim (`PAPER_text.txt:575-577`,
"Training and test sets are split by superfamily so that no superfamily appears in
both"), which this rebuttal's own audit falsified. Re-asserting a falsified claim to
the reviewer who raised leakage is the worst possible venue for it. Also missing: the
pooled 457-class macro AUC is not comparable to published per-holdout top-1 accuracies.

### W3. The PPI decontamination description (Yi1G §1 "PPI", line 141)

**WRONG.** Draft: *"Bernett test proteins were added to the STRING sequence pool,
MMseqs2 easy-linclust was run at 50% identity and 80% target coverage, and every STRING
protein in a Bernett-containing cluster was removed."*

What `data_prep.py` actually does (verified by reading it):

- `data_prep.py:1291-1300` calls `_mmseqs_leaked_query_ids(decontam_source_fasta,
  bernett_test_fasta, ..., min_seq_id=decontam_min_seq_id, cov=0.8, cov_mode=1)`.
- `data_prep.py:331-341`: `mmseqs easy-search`, `--cov-mode 1 -c 0.8 --alignment-mode 3
  -e 1e-3 -s 7.5`. STRING is the **query**, Bernett test is the **target**.
- `data_prep.py:357-362`: removal is of **hit query IDs**, not clusters.
- `--decontam_min_seq_id` default is **0.4**, not 0.5 (`data_prep.py:1087, 2858`).
- The function's own docstring (`data_prep.py:321-323`) says linclust was rejected
  deliberately: *"easy-search (not easy-linclust) is used deliberately: linclust's k-mer
  prefilter loses sensitivity below ~50% identity."*

The draft is again reproducing the paper's appendix 12.4 text
(`PAPER_text.txt:704-708`), which the code contradicts. `NEW_EVIDENCE.md` §7 item 3:
*"Describe what the code does."*

Second problem in the same paragraph: *"The requested 40% analysis is an additional
sensitivity check, not a missing train/test control."* The 40% STRING pass is **already
complete** — 4,178,737 pairs / 319,282 unique sequences removed, negative control 0 hits,
positive control 3,022/3,022 at `fident=1.000` (`REBUTTAL_LEAKAGE.md` §1). Calling
finished, controlled work a pending "sensitivity check" throws away the answer.

### W4. MNRL batch semantics (Yi1G §3, line 153) — most checkable by the reviewer

> "Each anchor is contrasted against the other 1023 positive-side examples in its source
> batch. The released paper-reproduction path uses CachedMultipleNegativesRankingLoss
> with a logical batch size of 1024 ... mini_batch_size=256 only partitions the
> forward/backward computation to reduce memory."

**WRONG / STALE — this describes V2, not the submitted model.**

- The submitted model's config, from the paper the reviewer is holding
  (`PAPER_text.txt:558-563`, Table 6): **per-device batch 64, gradient accumulation 16,
  effective batch 1024.**
- Gradient accumulation does **not** share in-batch negatives across micro-batches. Each
  of the 16 micro-steps computes its own MNRL loss over 64 examples. The contrastive
  batch for the submitted model is **64, not 1024** — and no loss variant changes that,
  because CachedMNRL's caching operates inside one loss call, not across accumulation
  steps.
- `batch_size=1024` with `mnrl_mini_batch_size=256` is the **V2** run
  (`train_esm2_35m.sh:41-42`; `REBUTTAL_LEAKAGE.md` §4 "batch size 1024 / device",
  "loss `cached_mnrl`, mini-batch 256"). `train_esm2_35m.sh` is the *decontaminated* run
  script — its header says so on line 2.
- Neither `NEW_EVIDENCE.md` nor `REBUTTAL_LEAKAGE.md` documents V1's loss configuration
  at all, so the claim is unsupported even before it is contradicted.

The reviewer asked this question *because* "effective batch size" was ambiguous. Answering
with a different model's config, when the paper's own Table 6 lists 64x16, hands them the
finding. If the intent is to say "V2 uses a true 1024 logical batch", say that and name V2.

### W5. "We add the missing Heinzinger et al. citation" (jVGf line 108)

**WRONG (verify against the review text).** Heinzinger et al. 2022 is already cited in
the submitted paper — Related Work, `PAPER_text.txt:83-85`: *"For proteins, Heinzinger
et al. [2022] proposed ProtTucker, which fine-tunes ProtT5 with triplet loss on CATH
superfamily labels (S30 subset, 3,186 training proteins)"* — and in the References
(`PAPER_text.txt:465-467`). Redl et al. 2023 is likewise already cited
(`PAPER_text.txt:85-87`).

Promising to add a citation that is in the paper reads as not having reread your own
related-work section. If jVGf named a *different* Heinzinger paper (ProstT5, bilingual
LM, etc.), the sentence must name which one.

### W6. "Redl et al. ... is the closest methodological antecedent" (Yi1G line 177)

**Inconsistent with your own record.** `REBUTTAL_LEAKAGE.md` §2 spends a full verified
section establishing ProtTucker (Heinzinger et al.) as *"the closest published analogue"*
and reconstructs its protocol precisely to justify your decontamination posture. The same
Yi1G paragraph then lists ProtTucker among systems you did not compare against. Pick one
antecedent and keep it consistent across §7 and the leakage answer in §1.

---

## TIER 3 — STALE. True of an earlier state; false now, or superseded.

### S1. "Frozen logistic-regression and ridge probes are running now" (HNXd line 37, Yi1G line 173)

**STALE — they finished, and the result is not the one the placeholders assume.**
`results/benchmarks/COMPARISON.md` has all four arms on 23 tasks x {3-NN, linear},
`--eval_split test`. Against ESM-2 35M over the 20 comparable tasks:

| probe | ProtSent-V1 | ProtSent-V2 |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median **+0.0075** | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / **12** lose, median **-0.0139** | 2 / 7 / 11, median -0.0107 |

The HNXd placeholder (line 39) asks for *"number of completed tasks improved at 35M and
150M"* plus representative gains. There is no 150M linear probe, and at 35M the honest
answer is a **loss**. Filling that placeholder as drafted produces a false claim.
`NEW_EVIDENCE.md` §6: *"any 'ProtSent > ESM-2' sentence must name the probe."*

Also affects the placeholder's task list: **PPI has no linear-probe number for any arm.**
`ppi_bernett` is not in `run_benchmarks_v3.sh`'s `TASKS` and has no row in
`COMPARISON.md`.

### S2. The whole draft never mentions decontamination or V2 — the largest strategic error

Yi1G item 1 opens by **conceding** — *"the original AFDB preparation ... does not
decontaminate against SCOPe"* — and offers a 155-query sensitivity table (which does not
exist, F2). Meanwhile the completed, audited, controlled work goes unmentioned in all
three responses:

- All three corpora filtered at 40% identity / 80% coverage: Pfam -600,912 (2.11%),
  AFDB -9,102,652 (6.72%), STRING -4,178,737 (5.49%) (`NEW_EVIDENCE.md` §1).
- Verified on the exact parquets the trainer opened: **0** surviving flagged sequences in
  all three files; row arithmetic closes to the trainer's logged `total=169,231,379`
  (`verify_training_corpus.py`, `results/benchmarks/training_corpus_verification.json`).
- Controls: negative 0 hits on all three filtered corpora; positive 3,244/3,244 and
  3,022/3,022 self-hit at `fident = 1.000`.
- Retrained V2 **improved** the task the corpus was filtered against: remote homology
  kNN 0.6587 → 0.6668, linear 0.6899 → 0.7016 (ESM-2 0.5835 / 0.6868).
- V2 SCOPe: R@1 0.4490 → **0.5256**, R@10 0.6529 → **0.7073**, MAP 0.4226 → **0.4955**.
- Memorization test: per-query Spearman between max identity to pretraining and gain is
  **negative** (MAP -0.114 V1 / -0.116 V2, p < 3e-6); largest gains in the *lowest*
  identity bin.

`REBUTTAL_LEAKAGE.md` §6 item 7: *"The draft concedes the leakage point instead of
citing this work."* Every V1/V2 comparison must also carry the confound caveat
(`NEW_EVIDENCE.md` §2): 7x1024 batch vs 1x1024, no hard negatives, proportional sampling,
one epoch — so the defensible claim is *removing the overlap did not cost performance*,
not *decontamination caused the gain*.

### S3. "the released pool does not include the exact sampled-pair manifest" (Yi1G line 120-121)

**STALE.** That limitation was true before `verify_training_corpus.py` existed. It now
semi-joins the exact training parquets against the recorded removal lists and returns 0
survivors, including both STRING pair columns — the load-bearing check, since
`stringdb_train_15M.parquet` was subsampled after filtering
(`REBUTTAL_LEAKAGE.md` §5). Stating the weaker limitation now understates the evidence.

### S4. Two placeholders promise results that do not exist and cannot be filled

- HNXd §3 line 49 and Yi1G §8 line 183: paired-bootstrap CIs on Table 2 deltas.
  `NEW_EVIDENCE.md` §8: *"No paired bootstrap confidence intervals on the Table 2
  per-task deltas."* → **delete, do not fill.**
- HNXd §4 line 57: 5- or 10-seed few-shot summary with mean±SD. No multi-seed few-shot
  run exists in the repo or in either source. → **delete, do not fill.**
- HNXd §4 line 59, jVGf line 100, Yi1G lines 135/143/175: same class — check each against
  §8 before filling.

---

## TIER 4 — Internal inconsistency and theory-of-mind failures

### I1. Yi1G §7: "The matched MMseqs2 SCOPe baseline is above." (line 173)

**Dangling reference.** The MMseqs2 table appears only in the **jVGf** response. Each
response is posted under its own review; Yi1G sees nothing "above". This is exactly the
failure mode the brief prohibits — every number must appear in the response text that
carries it, with metric, split, and model named.

### I2. Three different R@1 stories across three responses

- To HNXd and jVGf: ProtSent 0.5066 next to MMseqs2 0.3539, arranged to read as a
  decisive top-1 win.
- To Yi1G: *"The strict subset therefore does not support a robust top-1 claim."*
- Ground truth (`NEW_EVIDENCE.md` §3): tuned MMseqs2 0.5029 **beats** the submitted 35M
  model's 0.4490 at R@1; only V2 (0.5256) beats it.

`REBUTTAL_LEAKAGE.md` §6 item 8: *"Asserting a top-1 win to two reviewers while conceding
it to a third is inconsistent, and the data does not support the win."* Reviewers read
each other's threads. The one consistent, defensible line is **ranking depth**
(R@10 / R@30 / MAP) for V1, plus top-1 **for V2 only, named as V2**.

### I3. Ablation posture vs. the model that would carry the rebuttal

Yi1G §6 says hard negatives and round-robin are not validated as superior — correct per
paper Table 4 (no-hard-neg 20/23, +7.9%; proportional 16/23, +7.0%; full 16/23, +6.7%).
But V2 is trained with **proportional sampling and no hard negatives**
(`REBUTTAL_LEAKAGE.md` §4) — i.e. the ablation winners. Not an error, but if V2 is added
these must be told as one story ("the ablations pointed here, and V2 adopts it"), or §6
reads as an unexplained about-face.

### I4. Recall ceiling omitted from all three SCOPe tables

Only 1,693 of 2,207 queries have any non-self same-family neighbour, so **Recall@K is
upper-bounded at 0.7671** (`REBUTTAL_LEAKAGE.md` §3: *"State this in every caption
carrying a SCOPe recall"*). None of the draft's three SCOPe tables says so. ProtSent 35M's
R@30 of 0.7100 is 92.6% of attainable, not 71% of 100% — omitting this understates your
own result and invites a reviewer to compute a wrong headroom.

---

## TIER 5 — Precision items

### M1. The 35M row is labelled "reproduced" but uses the paper's numbers

Draft (line 26-27): 0.3833 / 0.5841 / 0.6402 / 0.3235 and 0.4495 / 0.6529 / 0.7100 / 0.4225.
Actual reproduction, `results/benchmarks/v3/{esm2_35m,protsent_old}_knn/*.csv`:
0.38287 / 0.58405 / 0.63978 / 0.32295 and 0.44903 / 0.65292 / 0.71001 / 0.4226.
The paper's Table 3 as extracted reads 0.385 / 0.588 / 0.641 and 0.445 / 0.651 / 0.710,
which matches neither. Pick one provenance and label it; do not call paper values
"reproduced".

### M2. n=92 ceiling stated as a count but not as a bound (Yi1G line 131)

The draft does say "only 57 of which have a non-self family positive" — better than
`REBUTTAL_LEAKAGE.md` §6 item 9 implies — but then quotes R@30 0.500 and MAP 0.248 without
stating that the ceiling is **57/92 = 0.620**, so 0.500 reads against an implied 1.0.
(Moot if F2 is deleted, which it should be.)

### M3. "MMseqs2 at 80% query coverage" (Yi1G line 120)

Every MMseqs2 invocation in this repo uses `--cov-mode 1` = coverage of the **target**,
chosen deliberately (`data_prep.py:317-320`, `REBUTTAL_LEAKAGE.md` §1: *"a long AFDB
protein that merely contains a test-length domain is still caught"*). Either the audit
used a different mode than everything else you describe, or the sentence is wrong.

### M4. k-NN regression description is right but incomplete (Yi1G §5)

**VERIFIED** for the main path: `protein_benchmark_suite.py:1556`
`KNeighborsRegressor(n_neighbors=3, metric=_KNN_METRIC)`, `_KNN_METRIC = "minkowski"`
(line 1524), no `weights` → uniform, default `p=2` → Euclidean. Two omissions worth one
clause each, since HNXd is asking about few-shot:

- `--knn_metric` can override the global (lines 2506, 2681-2683).
- The small-sample path uses `n_neighbors = max(1, min(3, train_size))` (line 1578), so
  **k < 3 at very small N** — directly relevant to the N=50 column of Table 5 that
  produced the -126.9% cell.

### M5. Task count 23 vs 24

`REBUTTAL_LEAKAGE.md` §6 item 6: `mmseqs_baseline.json` has 24 rows; the draft says 23
throughout; `COMPARISON.md` counts 20 comparable + 3 `n/a` + `rhla` excluded. Not wrong,
but state the exclusions (`ppi_bernett` pair-input, `proteingym_*`, `chezod_disorder`,
`cafa5`, `rhla_enzyme_mutations`) if any number is quoted per-task.

### M6. CoSENT answer is correct — sharpen where the reviewer's misreading came from

The draft's description (jVGf §3, Yi1G §2) matches the code: `data_prep.py` §12.5,
2,175,734 WT-mutant pairs, per-assay z-score clipped to [-3,3] rescaled to [0,1],
*"CoSENT preserves the score-induced ordering of pairs rather than training to absolute
targets"*, mutant-mutant intra-assay pairing disabled. **VERIFIED.**

But the reviewer's claim came from somewhere concrete: `PAPER_text.txt:174-175` says
*"This auxiliary loss operates on single proteins rather than pairs"*, which contradicts
your own appendix 12.5. Naming that specific sentence as the error is far stronger than
"the main text did not explain this clearly enough."

---

## VERIFIED — checked and correct, cite as-is

| Claim (draft line) | Source |
|---|---|
| Code evaluates SCOPe **family** field on **2,207** sequences (33, 137) | `benchmark_tasks.py:416-425` `label_col="family"`; `NEW_EVIDENCE.md` §7.1 |
| "100,000" is the evaluator's `max_samples` cap | every CSV: `Samples,100000` on a 2,207-row dataset. **Draft omits this mechanical explanation — add it; it converts a 45x error into a logging artifact** |
| Superfamily R@1 0.667→0.780 (150M), 0.639→0.726 (35M) (33, 137) | `NEW_EVIDENCE.md` §7.1. *Caveat: no repo artifact backs the 150M half — see F3* |
| -AFDB: +6.7%→+3.2%, 16/23→13/23, remote homology +40.5%→+15.3% (73) | paper Tables 4 and 7 |
| -Pfam: 15/23, +4.6% (78) | paper Table 4 |
| -STRING: PPI +5.3%→-0.5% (79) | paper Table 7 |
| -HardNeg: 20/23, +7.9% vs full 16/23, +6.7% (169) | paper Table 4 |
| Proportional +7.0% vs round-robin +6.7% (169) | paper Table 4 |
| -DMS reduces fitness gains incl. fluorescence (+15.6%→+10.4%) (80) | paper Table 7, §5.3 |
| PPI: partners embedded independently, embeddings concatenated (161) | `protein_benchmark_suite.py:1438-1440`; `benchmark_tasks.py:164-172` `{seq1: SeqA, seq2: SeqB}` |
| Peptide-HLA is single-input, one `seq` field (161) | `benchmark_tasks.py:182-189`; `REBUTTAL_LEAKAGE.md` §3 (pipe-joined `HLA_pseudoseq\|peptide`) |
| Round-robin ⇒ one source per step (153) | paper §3.5 |
| Eq. 1 is malformed (157) | `NEW_EVIDENCE.md` §7.4 |
| "Frozen logistic-regression and ridge probes" is the right name | `protein_benchmark_suite.py:1499,1537` `LogisticRegression(liblinear)` / `Ridge(alpha=1.0)` |
| AFDB prep filters pLDDT/fragment + Foldseek clusters, no SCOPe filter (120) | paper appendix 12.3 + "Leakage controls" |
| SaProt/ProSST need residue-level structure tokens, out of window (86) | `NEW_EVIDENCE.md` §8 |
| No matched ProtTucker/Foldseek/PLMSearch/DHR/ProTrek runs (98, 177) | `NEW_EVIDENCE.md` §8; `REBUTTAL_LEAKAGE.md` §5a (ProtTucker weights unreachable per `NETWORK_WHITELIST.md`) |

---

## Minimum fixes before this can be posted

1. **Delete F1, F2, F3, F4.** Four unsourced blocks. Nothing else matters until they are gone.
2. **Rebuild every SCOPe table from `NEW_EVIDENCE.md` §3**, 35M only, MMseqs2 at
   `-s 7.5` (0.5029 / 0.5637 / 0.5641 / 0.3100), with the 0.7671 ceiling stated.
3. **Add decontamination + V2 to Yi1G item 1 and to jVGf** — it is the answer to the
   question actually asked, with the V1/V2 confound caveat attached.
4. **Fix W2 (hierarchy-disjoint) and W3 (PPI linclust/50%)** — both currently repeat
   paper text your own audit falsified.
5. **Rewrite W4 (MNRL)** to describe the submitted model's 64 x 16 accumulation, or to
   name V2 explicitly as the 1024-logical-batch run.
6. **Delete the placeholders that cannot be filled** (bootstrap CIs, multi-seed few-shot,
   150M linear probe, PPI linear probe) rather than leaving them as promises.
7. **One R@1 story in all three responses**: depth for V1, top-1 for V2 only.
8. **Remove "the baseline is above"** from Yi1G §7 and inline the numbers.
