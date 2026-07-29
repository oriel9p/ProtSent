# Line-level prose review — FINAL_rebuttal.md

Scope: line-level only, no restructuring. **No number changed. No honest admission weakened
or removed.** Source file was not modified; every edit below is verified to apply cleanly
(exact-match, unique) and the character counts are measured, not estimated.

Edited paste-unit bodies (all edits applied, for diffing/pasting):
- `/tmp/claude-2003/-home-ddofer-ProtSent/84b62ed8-3317-44d7-8aee-1187117be114/scratchpad/HNXd_edited.txt`
- `/tmp/claude-2003/-home-ddofer-ProtSent/84b62ed8-3317-44d7-8aee-1187117be114/scratchpad/jVGf_edited.txt`
- `/tmp/claude-2003/-home-ddofer-ProtSent/84b62ed8-3317-44d7-8aee-1187117be114/scratchpad/Yi1G_edited.txt`
- Apply script: `.../scratchpad/edits.py`

## Character budget (measured on the strict BEGIN/END body)

| response | before | after | delta | slack under 10,000 |
|---|---|---|---|---|
| HNXd | 9,610 | 9,801 | +191 | 199 |
| jVGf | 9,234 | 9,351 | +117 | 649 |
| Yi1G | 9,757 | 9,889 | +132 | 111 |

All three remain under the limit. The net is positive because rule 1 (openings) needs
numbers moved forward; the cuts in rules 2/4 pay for most of it.

**DEFECT (pre-existing, hard rule): the Yi1G header comment says `character count of the
pasted body below: 9969` — the true body is 9,757.** A stale count on the response with the
least slack is how a >10,000 paste happens. After these edits the three header comments must
read 9801 / 9351 / 9889.

---

# Response 1 — Reviewer HNXd (+191, ends at 9,801)

## 1. First three sentences

**Before**

> **All five questions now carry measurements**, absolute few-shot scores with seed
> SDs included. Two of the answers go against us.

Verdict: fails. Sentence 1 is process, not result — it describes the response's own
completeness, which is the least interesting thing it has. A hostile skimmer reaches the end
of the opening with zero numbers. The single most valuable thing this response owns is the
retrained model's measured standing against the strongest alignment baseline, and it is
buried until section 3.

**After**

> **Retrained on decontaminated corpora, ProtSent-V2 (35M) ties HMMER at SCOPe-40
> family Recall@1** — paired bootstrap -0.0124, 95% CI [-0.0372, +0.0124] — and
> leads it at ranking depth (+0.1412 Recall@10, +0.1708 MAP, both intervals
> excluding zero). Two of your five questions produce answers against us: under a
> trained linear probe V2 loses to stock ESM-2 35M on 11 of 20 comparable tasks,
> and the label-scarcity claim is withdrawn. All five carry measurements below,
> few-shot with seed SDs.

Result first, cost second, process third. The concession in sentence 2 now carries its own
number (11 of 20), so it reads as a measurement rather than an apology. Every figure names
metric, split-equivalent (SCOPe-40 leave-one-out gallery, defined in the table), and model.

## 2. Cuts

**a. Section table-of-contents (meta-commentary about own structure)**

> Before: `**We computed no clustering statistics — no silhouette, NMI or ARI.** We ran direct retrieval, a per-query organisation analysis, and a layer sweep.`
> After: `**We computed no clustering statistics — no silhouette, NMI or ARI.** We ran direct retrieval instead.`

The list is a menu of the next three paragraphs. The paragraphs are right there. (-46)

**b. Paragraph label in prose**

> Before: `Reading rules, all against us. The ±0.005 tie band is in absolute units...`
> After: `The ±0.005 tie band is in absolute units...`

"Reading rules" announces a section; the losses in the table above already announce
themselves. (-30)

**c. Pre-emptive self-defence**

> Before: `One diagnostic, not a defence: both probes pool the **final** layer, and a per-layer linear sweep...`
> After: `Both probes pool the **final** layer, and a per-layer linear sweep...`

The closing clause of the same paragraph ("Two tasks, one scale; it does not overturn the
table") already disclaims it. Saying it twice is what a defensive writer does. (-31)

**d. List announcement**

> Before: `Three conclusions, the first against us. (i) **Your proposed framing is not supported by our data**:`
> After: `(i) **Your proposed framing is not supported by our data**:`

(i)/(ii)/(iii) are visible. "the first against us" is redundant with (i)'s own text, which
says it in stronger words. (-41)

**e. Self-congratulation**

> Before: `Two caveats we volunteer: one *training* run per model exists...`
> After: `Two caveats: one *training* run per model exists...`

Volunteering it is the act; narrating that you volunteered it is asking for credit. (-13)

**f. Restating the reviewer's question**

> Before: `**Your level-gap hypothesis: tested, and the probe is not the cause.**`
> After: `**Tested: the probe is not the cause of the level gap.**`

Answer-first ordering; the reviewer knows what they asked. (-14)

**g. Dead words**

> Before: `10,000 resamples, **paired**; this run reproduces every cell of the table above to within 0.0012.`
> After: `10,000 resamples, **paired**, reproducing every cell of the table above to within 0.0012.`
> Before: `...a median SD of 0.0000 across 24 rows, since fixed embeddings and a fixed test split make that probe deterministic; only...`
> After: `...a median SD of 0.0000 across 24 rows (fixed embeddings, fixed test split); only...`

(-12, -71)

**Deliberately kept:** "Our strongest surviving result sits on SCOPe-40, the one benchmark we
could not decontaminate", "This is identity-level only; it cannot see fold-level overlap",
"**V2 - V1 is therefore not a decontamination ablation**", "Until both are re-run with that
threshold removed, the depth margin is an upper bound". These are load-bearing admissions and
the hard rules require them.

## 3. Concession audit

| concession | paired with | action |
|---|---|---|
| "Withdrawn: general-purpose superiority (your Q2 killed it)" | **was bare** | **fixed:** now `(your Q2 killed it: 2 win / 7 tie / 11 lose under a linear probe)` (+64) |
| no clustering statistics computed | what was run instead + the retrieval table | ok |
| V1 loses top-1 to both tools | -0.1110 [-0.1388, -0.0827] vs HMMER, -0.0697 vs MMseqs2 | ok |
| depth margin is an upper bound | 691/2,207 no-hit queries; +0.0171/+0.0165 vs +0.0726/+0.0414 | ok |
| no intervals for the 23-task table | seed 42, ±0.005 band, checkpoint spread 0.005-0.008 | ok |
| linear probe record | 2/7/11, median -0.0107 | ok |
| "**We ran no fine-tuning sweep**" | nothing measurable exists | acceptable — scope statement, not an excuse |
| "V2 - V1 is not a decontamination ablation" | names the missing run | ok |
| 150M results withdrawn | abstract's +105% and +19.9% named | ok |

Only one bare concession existed and it is now numbered.

## 4. Defensive / supplicating passages

> Before: `If that changes your assessment we ask you to reconsider; if one item remains decisive, name it and we will answer it in discussion.`
> After: `That is the measured record, and we ask you to reconsider on it. If one item remains decisive, name it and we will answer it in discussion.`

"If that changes your assessment" pre-concedes that it probably will not, and makes the ask
conditional on the reviewer's mood rather than on the evidence. The ask survives; the flinch
does not.

## 5. Tense

> Before: `The camera-ready is therefore a 35M retrieval-and-remote-homology paper, both probes on the test split.`
> After: `What we defend is therefore a 35M retrieval-and-remote-homology result, both probes on the test split.`

Doubles as a hard-rule fix: the reviewers have no revised PDF, so a sentence about what the
camera-ready *will be* points at a document they cannot check. Reframed as the claim being
defended now. Everything else in HNXd is already past/present ("we computed", "we ran",
"is replaced"); "we will answer it in discussion" is a legitimate future offer, not a claim.

---

# Response 2 — Reviewer jVGf (+117, ends at 9,351)

## 1. First three sentences

**Before**

> Both of your axes now carry measurements, and the first goes against us:
> structural supervision is the single largest contributor, and under a trained
> linear probe ProtSent loses to its own untuned backbone on most tasks, so the
> general-purpose framing is withdrawn. What we defend is a *measured* position on
> the generality-accuracy curve, plus evidence that the non-structural sources do
> distinct work.

Verdict: closest of the three, but still fails on one count — it makes two strong concessions
with **no number attached to either**, which is exactly the trade that costs credibility. "the
single largest contributor" and "on most tasks" are the qualitative versions of numbers the
authors already have. Also "Both of your axes now carry measurements" is coverage-reporting.

**After**

> **Structural supervision is the single largest contributor, as you suspected:**
> removing AFDB drops our mean relative gain from +6.7% to +3.2% and remote
> homology from +40.5% to +15.3% (submitted 35M model, single run, suite default
> split). Under a trained linear probe ProtSent also loses to its own untuned
> backbone, stock ESM-2 35M, on 12 of 20 comparable tasks (V1) and 11 of 20 (V2),
> so the general-purpose framing is withdrawn. What survives is a measured position
> on the generality-accuracy curve and evidence that the non-structural sources —
> Pfam, STRING, DMS — each move a different task family.

Same two concessions, now both priced. This also fixes a hard-rule defect: in the original,
the AFDB ablation numbers first appear in section 2 and are only attributed to "the submitted,
pre-decontamination model" a paragraph later — a skimmer reads them as V2 numbers. The model,
the run count and the split now travel with the number at first use. The 150M disclaimer
stays where it was, as sentence 4.

## 2. Cuts

**a. Agreement throat-clearing**

> Before: `Your reading is fair and our text is wrong: the paper says the DMS loss "operates on single proteins rather than pairs."`
> After: `Our text is wrong: the paper says the DMS loss "operates on single proteins rather than pairs."`

Conceding the error is the compliment; grading the reviewer's reading is filler. (-24)

**b. Self-praise about honesty**

> Before: `**The limit on that argument, stated plainly:** those are single-run relative-percent numbers...`
> After: `**The limit on that argument:** those are single-run relative-percent numbers...`

(-16)

**c. Empty intensifier**

> Before: `Where annotation transfers by homology, alignment is simply better.`
> After: `Where annotation transfers by homology, alignment is better.`

"simply" adds nothing to a claim already carried by 0.710 vs 0.598 vs 0.562. (-7)

**d. Cross-reference to an unseen revision**

> Before: `ProTrek is the trimodal (sequence/structure/text) retrieval-optimized point on the same curve; we add it to Related Work as such, expect it to win retrieval accuracy against a 35M sequence-only encoder, and did not run it.`
> After: `ProTrek is the trimodal (sequence/structure/text) retrieval-optimized point on the same curve; we did not run it and expect it to beat a 35M sequence-only encoder at retrieval.`

"we add it to Related Work" is a promise about a document the reviewers cannot see, and it
buries the admission ("did not run it") in last place. Admission moved forward. (-46)

**e. Ambiguous internal reference (hard-rule risk)**

> Before: `The claim is that relation *type* is a design axis (section 2 shows each type moving a different task family), not that this beats structure distillation:`
> After: `The claim is that relation *type* is a design axis — each source moves a different task family, measured above — not that this beats structure distillation:`

"section 2" reads as a section of the *paper* on first pass. (+10, worth it)

**f. Counting defect**

> Before: heading lists five systems; body opens `Those four inject structure into a sequence model...`
> After: `ESM-S, S-PLM, ISM and Magneton inject structure into a sequence model by distilling a structure encoder...`

Under a 90-second skim "those four" against five names reads as an error in the authors'
arithmetic — the worst possible impression in a rebuttal built on counting. (+29)

## 3. Concession audit

| concession | paired with | verdict |
|---|---|---|
| structure is the largest contributor | +6.7%→+3.2%, 16/23→13/23, +40.5%→+15.3% | ok (now in sentence 1) |
| general-purpose framing withdrawn | 4/4/12 median -0.0139 (V1), 2/7/11 median -0.0107 (V2) | ok |
| "**We do not beat alignment at top-1**" | -0.0124 [-0.0372, +0.0124]; V1 -0.1110 | ok |
| depth margin is an upper bound | 691/2,207; +0.0171/+0.0165 vs +0.0726/+0.0414 | ok |
| ablation numbers are single-run relative percents | bounded to "2-25 relative points and nothing finer" | ok — a bound is a number |
| "**We did not run the joint no-AFDB/no-Pfam ablation**" | names the unanswered question | acceptable, nothing measurable exists |
| "**no matched runs against any of the four**" | claims no superiority | ok |
| "**Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek**" | SaProt/ProSST blocker named as data, not code | ok |

No bare concessions.

## 4. Defensive / supplicating passages

> Before: `If the measured trade-off, the source fingerprints and the positioning answer your two axes, we ask you to reconsider. If the missing no-AFDB/no-Pfam ablation or the missing ProtTucker run is the decisive one, say which and we will address it in discussion.`
> After: `That is the measured trade-off, the source fingerprints and the positioning, and we ask you to reconsider on them. If the missing no-AFDB/no-Pfam ablation or the missing ProtTucker run is decisive, say which and we will address it in discussion.`

Asserts that the axes were answered instead of asking whether they were. (-5)

## 5. Tense

Clean after edit (d). "we ran MMseqs2", "We also ran HMMER", "removing AFDB drops",
"we did not run it" — all work done. No promised experiments anywhere.

---

# Response 3 — Reviewer Yi1G (+132, ends at 9,889 — 111 characters of slack)

## 1. First three sentences

**Before**

> Your leakage objection was correct and we treated it as decisive: all three
> corpora re-filtered, retrained from scratch, benchmarks re-run. HMMER and MMseqs2
> were both run. Every number defended below is **V2**'s, not the submitted
> model's.

Verdict: strong opening move (concession + action in one clause) but it spends three
sentences on *activity* and reaches sentence four with no result. For the leakage reviewer the
two most valuable facts in the whole response are the verification outcome (0 flagged
sequences survived) and the fact that the re-run baselines cost the authors the top-1 claim —
the second of which is the single most credibility-buying sentence available here.

**After**

> Your leakage objection was correct and we treated it as decisive: all three
> corpora re-filtered at 40% identity / 80% coverage, model retrained from scratch,
> benchmarks re-run, and verification found **0 flagged sequences surviving**.
> HMMER was run too, and it costs us a claim: ProtSent-V2 (35M) minus HMMER at
> SCOPe-40 family Recall@1 is -0.0124, 95% CI [-0.0372, +0.0124] — a tie, not a
> win. Every number defended below is **V2**'s, not the submitted model's.

Same three sentences, now carrying the filter threshold, the verification result, and the
result that hurts. A reviewer who reads only this knows the authors did the work and reported
the outcome that went against them.

## 2. Cuts

**a. Defensive pre-emption**

> Before: `**PPI: no post-decontamination number exists, and we are not withholding one.**`
> After: `**PPI: no post-decontamination number exists.**`

"we are not withholding one" plants the accusation it denies. The two sentences that follow
(pair-input task, not in the 23-task sweep, 4,178,737 pairs removed) prove it without the
protest. (-33)

**b. "We agree that" pattern**

> Before: `Agreed, and we acted on it. Removing synthetic hard negatives improves 20/23 tasks at mean +7.9%...`
> After: `Removing synthetic hard negatives improves 20/23 tasks at mean +7.9%...`

The numbers *are* the agreement; the ablations exist, so saying "agreed" first delays them by
a sentence. (-27)

**c. Self-praise about honesty**

> Before: `**The consequence, stated not implied:** those ablations were scored on these same benchmarks...`
> After: `**The consequence:** those ablations were scored on these same benchmarks...`

The admission that follows is genuinely damaging and genuinely volunteered — which is why it
does not need a badge. (-20)

**d. Header now redundant with the opening**

> Before: `**Decontamination, done.** MMseqs2 \`easy-search\`, corpus as query, test set as target...`
> After: `MMseqs2 \`easy-search\`, corpus as query, test set as target...`

(-26)

**e. Filler connective**

> Before: `existing. Relatedly, over the 20 tasks whose metric is defined for all arms...`
> After: `exists. Over the 20 tasks whose metric is defined for all arms...`

(-11)

**f. Clunky participle at a load-bearing admission**

> Before: `training-seed variance is unmeasured, one training run per model existing.`
> After: `training-seed variance is unmeasured; one training run per model exists.`

The trailing "existing" makes the sentence trail off exactly where it should land. (-2)

**g. Section table-of-contents**

> Before: `Two bear on weakness 1. The PPI decontamination description does not match the code...`
> After: `The PPI decontamination description does not match the code...`

(-25)

**h. Wordiness**

> Before: `**Those two test sets were the only filter targets** — every other benchmark test set, SCOPe-40 included, was not.`
> After: `**Those two test sets were the only filter targets**; SCOPe-40 and every other benchmark test set were not.`
> Before: `Verified on the parquet files training actually opened, by semi-join with the removal lists: **0 flagged sequences survived**, row counts summing to the 169,231,379 in the training log.`
> After: `Verified by semi-join with the removal lists on the parquet files training actually opened: **0 flagged sequences survived**, row counts summing to the 169,231,379 in the training log.`
> Before: `(35M: 64 per device × 16 steps; 150M: 16 × 64, our Table 6)`
> After: `(35M: 64 per device × 16 steps; 150M: 16 × 64)`

(-6, -6, -13)

**Deliberately kept:** "**nothing is attributable to decontamination in either direction**",
"Only the weak claim is supported", "That half of weakness 1 is unanswered", "**What this
cannot rule out**", "V2's numbers are therefore not a clean held-out measurement", "The right
experiment ... we did not run". These are the response.

## 3. Concession audit

| concession | paired with | verdict |
|---|---|---|
| leakage objection was correct | 40%/80% refilter, retrain, 0 flagged survivors, 0 negative-control hits, ~0.3% residual bound | ok |
| residual not bounded below ~0.3% | rule of three, 1,000 sequences/corpus | ok |
| V1 below the untuned backbone on macro-F1 | 0.4414 / 0.4281 / 0.4527 | ok |
| nothing attributable to decontamination | +0.0079 inside the 0.005-0.008 checkpoint spread | ok |
| no PPI post-decontamination number | 4,178,737 STRING pairs removed, task absent from the 23-task sweep | ok |
| SCOPe-40 not decontaminable | median max identity 0.908, none below 20% | ok |
| fold-level overlap untested | flat identity slope is consistent with it; names the missing experiment | ok — a bare admission that cannot be numbered, and it is correctly framed as the limit of the evidence |
| MNRL batch misdescribed | 1,024 optimizer vs 64 (35M) / 16 (150M) actual | ok |
| Eq. 1 wrong | states the correction | ok |
| ablations chose the config on the eval benchmarks | 20/23 at +7.9% vs 16/23 at +6.7%; +7.0% vs +6.7% | **fixed:** those percentages had no stated baseline or model. Now `(relative gain over ESM-2 35M, suite default split, one run each)` (+60) |
| no intervals on Table 2 | seed 42, ±0.005, checkpoint spread | ok |
| PPI decontamination description ≠ code | names both procedures | ok |
| remote-homology split not hierarchy-disjoint | 718 + 1,254 + 1,272 = 3,244 | ok |

## 4. Defensive / supplicating passages

> Before: `If that bounds weakness 1 to the residual we state — untested fold-level overlap on SCOPe-40, no PPI measurement — we ask you to reconsider; if one remains decisive, say so and we will answer it.`
> After: `Weakness 1 now reduces to the residual we state — untested fold-level overlap on SCOPe-40, no PPI measurement — and we ask you to reconsider on that. If one remains decisive, say so and we will answer it.`

The original makes the *authors'* conclusion contingent on the reviewer granting it. The
reduction is a fact about the evidence; state it, then ask. (+20)

Also see cut (a): "we are not withholding one" was the only outright defensive sentence in the
file.

## 5. Tense

Clean. "was run", "were re-filtered", "differs", "uses a true 1,024-example batch per device",
"V2 uses neither default" — all completed work. No experiment is promised anywhere in Yi1G;
the only future verbs are "we will answer it in discussion", which is an offer.

---

# Hard-rule compliance check (all three, post-edit)

- **Under 10,000 characters:** 9,801 / 9,351 / 9,889. Yes. **Update the three header comments
  to these values** — Yi1G's is currently wrong by 212 characters in the dangerous direction.
- **No links, no attachments, no figure references, no "see the revised paper":** two
  violations found and fixed — HNXd's "The camera-ready is therefore..." and jVGf's "we add it
  to Related Work as such". jVGf's "(section 2 ...)" removed as ambiguous with the paper's
  own section 2. References to the *submitted* paper's Table 5 / Table 6 are legitimate (the
  reviewers hold that PDF); "our Table 6" dropped only to buy characters.
- **Every number interpretable in place:** two defects found and fixed — jVGf's AFDB ablation
  percentages (model/run/split now attached at first use) and Yi1G item 6's +7.9%/+6.7%/+7.0%
  (baseline, split and run count now attached). All others already name metric, split and arm.
- **V1/V2 discipline:** intact. No decontaminated 150M implied anywhere; both "there is no
  150M model on the decontaminated corpus" statements untouched.
- **Forbidden claims:** none present before or after. The top-1 language is "ties HMMER",
  "**Alignment is the better top-1 method**", "**We do not claim to beat alignment at
  top-1**" — correct in all three responses. The ESM-2 comparison is stated as 2/7/11 linear
  and 10/3/7 3-NN, never as general superiority. The V1-vs-V2 comparison is explicitly denied
  the status of a controlled ablation in both HNXd and Yi1G.
- **Numbers changed:** none. Verified by numeric-token diff of each body before vs after: no
  result token is removed anywhere (the sole deletion is the "6" of the "our Table 6"
  cross-reference); every added token is a restatement of a figure already present elsewhere
  in the same response (the HMMER intervals moved into the HNXd and Yi1G openings, the AFDB
  ablation percentages and the 12/20 and 11/20 linear-probe counts moved into the jVGf
  opening, the 2/7/11 record into HNXd's withdrawal line).
