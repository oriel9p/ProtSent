# Prose cuts — FINAL_rebuttal.md

Measurement basis: characters between `<!-- BEGIN X -->` / `<!-- END X -->`, stripped.
By that measure: **HNXd 9,874 · jVGf 7,821 · Yi1G 9,943**. Your ~10,009 / ~9,946 figures
are ~66 higher, so something in your count includes the `## Response to Reviewer X`
heading. Treat every margin below as 70 chars tighter than it looks.

After all cuts: **HNXd 9,044 · jVGf 7,566 · Yi1G 9,109**.

Quotes are verbatim including line wraps; each was verified to occur exactly once.

---

## Yi1G — 9,943, needs to lose ~110 minimum. Cuts free 834.

### Y3 · -224 · running -224
Config paragraph duplicated from §6, which already says "the retrain uses neither".

> We retrained using the
> configuration favoured by the paper's own ablations — proportional sampling and no
> synthetic hard negatives — otherwise following the submitted recipe, on 7 GPUs rather
> than 1. V1-vs-V2 is not a controlled decontamination ablation; the supported claim is
> sufficient — decontamination cost nothing.

->

> V1-vs-V2 is not a controlled ablation; the supported
> claim is that decontamination cost nothing.

Then append to §6's last sentence: `so the retrain uses neither, and runs on 7 GPUs
rather than 1.` (+35, already netted in the -224? No — add 35 back if you want the GPU
count kept; it is the only surviving V1/V2 config disclosure and it should be kept.)

### Y1 · -160 · running -384
First sentence restates the two paragraphs immediately above it. Zero information loss.

> We can say the corpus
> holds nothing within 40%/80% of the two filtered test sets and that the SCOPe advantage
> does not grow with proximity to pretraining data. We cannot say SCOPe-40, or the
> fold-level third of the remote-homology set, is free of structural-label overlap.

->

> We cannot say SCOPe-40, or the
> fold-level third of the remote-homology set, is free of structural-label overlap.

### Y2 · -145 · running -529
"an omission, not an obstacle" is an excuse for work not done — it buys nothing and
invites "then why didn't you". Also factually stale (see BLOCKER 2).

> For the 23-task table we have **no** intervals and no multi-seed results — an omission,
> not an obstacle: probes can be held fixed and predictions resampled. Your objection
> stands without them: every cell is one run at seed 42, so any delta inside ±0.005 is a
> tie.

->

> For the 23-task table we have **no** intervals: every cell is one run at seed 42, so
> any delta inside ±0.005 is a tie.

### Y4 · -62 · running -591
Self-congratulation. The disclosure is the content; the framing is posture.

> **Scope limit, stated up front because you would otherwise find it yourself: those two
> test sets were the only filter targets.**

->

> **Scope limit: those two test sets were the only filter targets.**

### Y5 · -48 · running -639
"rather than arguing" is a claim about our own virtue. The 20/23 and +7.9% right after
it prove the point.

> Agreed, and we acted on it rather than arguing. Removing synthetic hard negatives

->

> Removing synthetic hard negatives

### Y6 · -33 · running -672
Same posture problem.

> **What these controls cannot rule out — stated by us, not found by you.** Our

->

> **What these controls cannot rule out.** Our

### Y7 · -31 · running -703
"full alternative pipeline" is padding; "over all 23 tasks" already says it.

> MMseqs2, run as a full alternative pipeline over all 23 tasks with identical metric
> definitions

->

> MMseqs2, run over all 23 tasks with identical metric definitions

### Y8 · -27 · running -730
Meta-commentary on the response's own structure. The `### 1.`–`### 8.` headings do it.

> retrained from scratch, and re-ran every benchmark. Your eight items
> in order.

->

> retrained from scratch, and re-ran every benchmark.

### Y9 · -25 · running -755
"No excuse offered" is throat-clearing. Also HMMER must leave this list (BLOCKER 1).

> **Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** No excuse offered; we
> claim no superiority to any.

->

> **Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek** — we claim no superiority to
> any.

### Y10 · -25 · running -780
Announcing a disclosure instead of making it.

> Disclosure that follows: those ablations were scored on these same benchmark tasks, so
> V2's configuration was chosen with benchmark results in view.

->

> Those ablations were scored on these same benchmark tasks, so V2's configuration was
> chosen with benchmark results in view.

### Y11 · -13 · running -793

> filter was then verified on the parquet files the training job actually opened, by
> semi-join with the removal lists: **0 flagged sequences survived**.

->

> filter was verified on the parquet files training actually opened, by semi-join with
> the removal lists: **0 flagged sequences survived**.

### Y13 · -11 · running -804

> Correct, and this is a real error. For the submitted models

->

> Correct — a real error. For the submitted models

### Y12 · -11 · running -815

> That, plus relative changes over near-zero Spearman baselines (which produce cells like
> -126.9%, a sign flip of magnitude 0.269x baseline), makes the few-shot table
> uninterpretable; its claims are withdrawn.

->

> That, plus relative changes over near-zero Spearman baselines (cells like -126.9% are a
> sign flip of magnitude 0.269x baseline), makes the few-shot table uninterpretable; its
> claims are withdrawn.

### Y15 · -10 · running -825

> We therefore tested memorization directly: if -> We tested memorization directly: if

### Y14 · -9 · running -834
"Paired ... paired" in one sentence.

> Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples, paired (the
> same queries score every method).

->

> Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples; the same
> queries score every method.

**Yi1G: 9,943 -> 9,109. 891 chars of headroom, ~550 of which the HMMER fix will spend.**

---

## HNXd — 9,874, no margin. Cuts free 830.

### H1 · -166 · running -166
Same "omission, not an obstacle" excuse as Y2, longer. It describes an experiment we
did not run, in a paragraph conceding we did not run it.

> **We did not compute intervals for the 23-task table.** This is an omission, not an
> obstacle: the fitted probe can be held fixed and the held-out predictions
> resampled, which requires no refitting. Without them your objection stands on its
> own — each cell is one run at one seed, and any delta inside ±0.005 is unresolved
> and is reported as a tie, not an improvement.

->

> **We did not compute intervals for the 23-task table.** Your objection stands —
> each cell is one run at one seed, and any delta inside ±0.005 is unresolved and is
> reported as a tie, not an improvement.

### H2 · -95 · running -261
Restates the reviewer's own question back at them. Also "the per-query analysis in
section 3" is a dangling cross-reference — HNXd's section 3 is confidence intervals;
the per-query identity analysis is only in the Yi1G response, which HNXd never sees.

> **We did not compute clustering-geometry statistics** — no silhouette, NMI or ARI.
> You asked for either a retrieval/clustering evaluation or a geometry analysis; we
> ran the first, plus the per-query analysis in section 3. The second is missing.

->

> **We did not compute clustering-geometry statistics** — no silhouette, NMI or ARI.
> We ran the retrieval evaluation; the geometry analysis is missing.

### H3 · -86 · running -347
Two 40-word sentences fused into one 35-word sentence, same facts.

> V2 is a retrain on corpora from which every sequence within 40% identity / 80% coverage
> of the remote-homology and PPI test sets was removed (those two test sets were the only
> filter targets; SCOPe-40 and the other benchmarks were not filtered). We retrained on the
> filtered corpora using the configuration favoured by the paper's own ablations —
> proportional sampling and no synthetic hard negatives — otherwise following the submitted
> recipe, on 7 GPUs rather than 1.

->

> V2 is a retrain on corpora with every sequence within 40% identity / 80% coverage of the
> remote-homology and PPI test sets removed (the only two filter targets; SCOPe-40 and the
> other benchmarks were not filtered), using the configuration the paper's own ablations
> favour — proportional sampling, no synthetic hard negatives — otherwise the submitted
> recipe, on 7 GPUs rather than 1.

### H4 · -84 · running -431
"answers your request exactly as you posed it" is supplication; the mechanism is the
argument. "here is that denominator explicitly" narrates the table below it.

> Retrieval answers your request exactly as you posed it — every metric is a mean
> over per-query values, so resampling queries gives the sampling distribution with
> no refitting. 10,000 resamples, **paired** (the same queries score every method),
> over the **1,693 eligible** queries. The intervals therefore sit on a different
> denominator from the table above; here is that denominator explicitly:

->

> Every retrieval metric is a mean over per-query values, so resampling queries gives
> the sampling distribution with no refitting. 10,000 resamples, **paired** (the same
> queries score every method), over the **1,693 eligible** queries. The intervals
> therefore sit on a different denominator from the table above:

### H5 · -75 · running -506
"they are not two views of one result" restates "different metric, different split".

> Note also that the submitted paper's "+40.5%" for
> this task is a relative change in macro-F1 (.223 → .313) computed under the suite's
> default split; the numbers here are accuracy on the test split. Different metric,
> different split — they are not two views of one result and we do not mix them.

->

> The submitted paper's "+40.5%" for this task is a
> relative change in macro-F1 (.223 → .313) under the suite's default split; these are
> accuracies on the test split. Different metric, different split; we do not mix them.

### H6 · -64 · running -570
Pointer to the response's own structure.

> That drops remote homology,
> our best task, out of the tally; its accuracy is reported separately below and is
> not in the 20; (iii) three *different* tasks

->

> That drops remote homology,
> our best task, out of the tally; (iii) three *different* tasks

### H7 · -60 · running -630
Delete outright. The preceding sentence already says the checkpoint control differs by
0.005-0.008; this adds no fact. Becomes actively wrong once the seed data goes in.

> That bounds one nuisance factor; it is not a seed replicate.

-> (delete)

### H8 · -37 · running -667
Concession with no number attached — it gives ground and buys nothing.

> Two protocol facts we should
> have printed: `thermostability`

->

> Also: `thermostability`

### H9 · -35 · running -702
Restates the table two paragraphs up.

> The aggregate is the 4/4/12 above. Per task the two probes disagree in a specific
> way: on AAV fitness

->

> Per task the two probes disagree in a specific way: on AAV fitness

### H10 · -29 · running -731
Instruction to the reader about how to read the response.

> Read this before the counts: (i) the tie band -> (i) The tie band

### H11 · -28 · running -759
"and we present it as small" — we just did, in the same sentence.

> +0.0289 with a lower bound of +0.0035 — resolved, but small, and we present it as
> small.

->

> +0.0289 with a lower bound of +0.0035 — resolved, but small.

(This sentence also needs the word "alignment" -> "MMseqs2". See BLOCKER 1.)

### H12 · -24 · running -783

> Three results from the same procedure that cut against us: MMseqs2

->

> Three results that cut against us: MMseqs2

### H13 · -21 · running -804

> One caveat that cuts against us: MMseqs2's R@10 -> Against us: MMseqs2's R@10

### H14 · -17 · running -821

> **What the paper claims after this rebuttal, in one sentence:** contrastive

->

> **What the paper claims after this rebuttal:** contrastive

### H15 · -9 · running -830

> so
> their rows are one measurement printed in both tables. SCOPe retrieval therefore
> has **one** measurement, not two, and we do not claim it "survives both probes";

->

> so
> their rows are one measurement printed in both tables — SCOPe retrieval has **one**
> measurement, not two, and we do not claim it "survives both probes";

**HNXd: 9,874 -> 9,044. 956 chars of headroom, ~800 of which the seed fix will spend.**

---

## jVGf — 7,821, 2,179 of margin. Cuts free 255. Do these anyway; they are all slop.

| # | saved | running | text -> replacement |
|---|---|---|---|
| J2 | -48 | -48 | `Your reading is a fair reading of our text, and our text is wrong: the paper says` -> `Our text is wrong: the paper says` |
| J3 | -45 | -93 | `The\ngeneral-purpose framing is withdrawn. Here is the position on the curve, measured.` -> `The\ngeneral-purpose framing is withdrawn.` (narrates the response's own structure) |
| J1 | -45 | -138 | `**Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** We offer no\nexcuse for their absence; they are missing, we claim no superiority to any of them,\nand ProtTucker in particular` -> `**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek.** They are missing, we\nclaim no superiority to any of them, and ProtTucker in particular` (HMMER removed — BLOCKER 1) |
| J4 | -39 | -177 | `That is the trade-off, measured rather than asserted: alignment wins single-best-hit` -> `The trade-off: alignment wins single-best-hit` |
| J5 | -27 | -204 | `The\nmargins are not small: EC classification` -> `EC classification` (0.710 vs 0.598 says it) |
| J7 | -24 | -228 | `The honest limit on that argument: those` -> `The limit: those` |
| J6 | -20 | -248 | `Structural supervision is the single\nlargest contributor and we say so first.` -> `Structural supervision is the single largest contributor.` |
| J8 | -7 | -255 | `Two honest qualifications: the` -> `Two qualifications: the` |

**jVGf: 7,821 -> 7,566.**

---

## Two blockers that are not prose. Fix before posting.

These are why the cuts above are load-bearing: both fixes cost characters, and both
responses currently have none.

### BLOCKER 1 — HMMER was run. All three responses say it was not.

`NEW_EVIDENCE.md` §3 has phmmer measured on the same gallery with the same scoring, plus
paired bootstrap. Yi1G's weakness 7 names "HMMER/MMseqs2" by name, so "Not run: ... HMMER
..." is both false and throws away the direct answer to the item. Two consequences:

1. Strike `HMMER` from the "Not run" list in all three responses (folded into Y9 / J1
   above; HNXd does not carry the list).
2. HNXd currently reads `V2's top-1 lead over alignment is +0.0289` — false, and a direct
   violation of the honesty constraint. V2 ties HMMER at R@1 (-0.0124 [-0.0372, +0.0124]).
   Minimum fix, zero cost: `alignment` -> `MMseqs2`.
3. Yi1G §7 should gain the HMMER block. 547 chars, fits inside the 891 freed:

> HMMER (phmmer), same gallery and scoring (691 of 2,207 queries return no hit, scored as
> failures), 1,693 eligible queries, R@1 / R@10 / MAP: 0.6970 / 0.7809 / 0.4747 vs MMseqs2
> 0.6556 / 0.7401 / 0.4098 and V2 0.6852 / 0.9220 / 0.6459. HMMER is the stronger
> alignment baseline. Paired bootstrap V2 - HMMER: R@1 **-0.0124 [-0.0372, +0.0124],
> unresolved**; R@10 +0.1412 [+0.1205, +0.1618]; MAP +0.1708 [+0.1511, +0.1905]. V1 -
> HMMER at R@1 is -0.1110 [-0.1388, -0.0827]. Alignment is still the better top-1 method;
> V2 ties it there and wins on depth.

Yi1G lands at 9,109 + 547 = **9,656**.

### BLOCKER 2 — the 5-seed sweep exists. HNXd says it does not.

HNXd §4-5 opens `**We have no multi-seed results**, for training seeds or probe seeds.`
`NEW_EVIDENCE.md` §4c has 5 seeds x 8 tasks x 3 arms. HNXd asked for exactly this
(question 4) and said they would raise their score for it. Conceding work that was done
is the single most expensive line in the file. Replace that sentence **and** the
checkpoint paragraph (H7 already deletes its tail) with, 802 chars:

> **Multi-seed: 5 seeds (0-4) x 8 representative tasks x 3 arms, 3-NN, test split.**
> Median SD across all 24 rows is **0.0000**. Remote-homology accuracy: 0.5835 +/- 0.0000
> (ESM-2 35M), 0.6589 +/- 0.0002 (V1), 0.6668 +/- 0.0000 (V2). The reason is mechanical,
> not impressive: with fixed embeddings and a fixed test split a 3-NN probe is
> deterministic, so the benchmark seed only moves subsampling and CV-fallback splits.
> `thermostability` is the one task that subsamples and the only one with visible spread
> (SD 0.013-0.017 across arms). The V1 -> V2 remote-homology gap of +0.0079 is ~40x the
> seed SD on that task. **These are probe seeds: one training run per model exists, so we
> have no training-seed replicates**, and Table 5 is withdrawn rather than re-presented in
> absolute units from a single run.

That replaces 161 chars of existing text, so net +641. HNXd lands at 9,044 + 641 =
**9,685**. Yi1G's `and no multi-seed results` (Y2, already cut) must not come back.

Also available and unused: `NEW_EVIDENCE.md` §4c answers HNXd's Stability-vs-literature
complaint directly — on Stability the 3-NN probe beats the linear probe for every arm
(ESM-2 0.6435 3-NN vs 0.4395 linear), so the gap to the published 69.08% is not a probe
artifact. ~180 chars, and it converts a concession into a measured rebuttal. Room exists.
