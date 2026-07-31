# Cold read of COAUTHOR_BRIEF.md — coauthor review

Read order: `REVIEWS_actual.md` → `PAPER_text.txt` → `COAUTHOR_BRIEF.md`. Nothing else.
Everything below is from those three files only, so if I say "unexplained" it means
unexplained *in the brief*, not necessarily unexplained in the repo. That distinction is
the brief's problem to solve, not mine — the brief is supposed to be the thing I read
instead of the repo.

Short version: the science is better than the document. I believe these results. I cannot
yet edit the paper from this.

---

## 1. Do I understand what was run and what was found?

The arc, yes: you re-filtered the corpus at 40%/80% against two benchmark test sets,
retrained both scales, re-benchmarked through one path with paired bootstraps, and found
that the retrieval story got stronger while the general-purpose story got weaker. That
landed on the first read.

What I had to re-read, or still cannot answer:

**a) I cannot reconstruct the V2 training corpus.** The decontamination table gives three
corpora ending at 27,929,772 / 126,301,607 / 71,891,417. The verification line is
`27,929,772 + 126,301,607 + 15,000,000 = 169,231,379`. Where does **15,000,000** come from?
It is not the STRING post-filter row count (71.9M), it is a round number, and it appears
nowhere else in the document. And where is DMS? **Is CoSENT/ProteinGym still in V2 at all?**
The paper's method section is built on five sources; the brief's config paragraph tells me
hard negatives are out, and the arithmetic implies three terms. I genuinely do not know
whether V2 is a 3-source or 4-source or 5-source model. I cannot write the methods section
without that.

This matters more than it sounds, because the brief presents that sum as proof:

> **Row arithmetic closes independently** — 27,929,772 + 126,301,607 + 15,000,000 =
> 169,231,379, exactly the total the trainer logged.

An identity containing one unexplained round-number term is not an independent check. It is
almost certainly an innocent cap, but as written it looks like the plug that makes the sum
work. Label it or drop the claim.

**b) The row counts do not match the paper's Table 1 and nobody says why.**

| corpus | paper Table 1 | brief "rows before" |
|---|---:|---:|
| Pfam | 32,943,498 | 28,530,684 |
| AFDB | 133,856,004 | 135,404,259 |
| STRING | 36,502,692 pairs | 76,070,154 |

Three different directions of mismatch. STRING looks like rows-vs-pairs (≈2×, but not
exactly). Pfam and AFDB I can't explain at all. One sentence fixes this; without it a
reviewer who does the same subtraction I just did concludes the corpus changed silently.

**c) Steps don't reconcile with batch size.** 169.2M rows / 4,850 steps ≈ 34,900 rows per
step. Config says 1024 per device, 7 devices, no gather → 7,168. Off by ~5×. Either "rows"
≠ training examples, or the batch figure is per-something-else. Also: the 150M run used
`MAX_PAIRS_PER_CLUSTER=5` and 3,890 steps on 6 devices ≈ 23.9M examples. **So 35M and 150M
were trained on materially different amounts of data**, which is never stated and which
undercuts every 35M-vs-150M comparison in §3 and §5.2 (including "the larger model exploited
the leakage more than the small one did" — maybe, or it saw a different corpus).

**d) I can't tell what the rebuttal says.** I was told my collaborators wrote a rebuttal.
This document never summarises it. §5.2 tells me one thing is out of it; §6.6 tells me three
fixes are "already in the rebuttal"; nothing else. I don't know its claims, its length
limit, what got cut, or **whether it has already been posted**. If it's posted, two-thirds
of §6 is unactionable by me and should be labelled as settled. There is no date anywhere in
the document.

**e) Which paper tables survive?** Table 3's absolute SCOPe numbers (0.385/0.445,
0.423/0.507) bear no relation to §3's (0.4991/0.5854, 0.5535/0.6615). §6.6 calls this a
"description error" about the eval set, but it is a **wholesale replacement of a results
table**, and the brief never says so. Same question unanswered for Table 2, and Table 4/7
are worse — see §7 below.

**f) Small unexplained drift between the brief's own tables.** §3 vs §5.2 for the same
quantities: SCOPe ESM-2 150M R@1 0.5535 vs 0.5529, MAP 0.4236 vs 0.4242; V2-150M 0.7431 vs
0.7425, 0.7046 vs 0.7048; remote homology V2-150M 0.6612 vs "0.6606 raw", ESM-2 150M 0.5194
vs "0.5200". Third-decimal, so presumably a different gallery or a seed in the whitening
script. In a paper where three reviewers are attacking sub-1% deltas, unexplained
third-decimal drift between two tables in our own internal doc is exactly the thing that
gets caught. One footnote.

**g) The 23-task aggregate is over 20 tasks at 35M.** 11+3+6 = 20, 10+3+7 = 20, 4+4+12 = 20,
2+7+11 = 20; the 150M rows sum to 23. Header says "Aggregate over 23 tasks". Which three
dropped, and why?

**h) Matryoshka 64/128/256 is new and unevaluated.** It's a fourth config change, it's not
in the "two config changes" sentence, and there are no truncated-dim results. At what
dimension were the benchmarks run?

---

## 2. Does it connect to the actual reviews?

Partly, and unevenly enough that I think it has misallocated the effort.

It names Yi1G for leakage and for HMMER, HNXd for the three statistics asks, jVGf for the
generality-accuracy trade-off. Those attributions are correct and the leakage/statistics
work is genuinely responsive.

But there is no coverage map, and when I built one, roughly **half the distinct reviewer
asks are untouched, and most of the untouched ones are free** — text, not compute:

Unaddressed, cheap:
- jVGf: position against ESM-S / S-PLM / ISM / Magneton, cite ProTrek. **This is a related-
  work paragraph and jVGf said explicitly they'd move to accept if we address positioning
  plus the trade-off.** We did the expensive half (trade-off) and skipped the free half.
- jVGf: "How exactly does the CoSENT loss for DMS data work? ... I don't understand how
  mutants within a DMS assay are paired." A confused reviewer asking us to explain our own
  loss. Free.
- jVGf: missing reference at line 21. Free.
- jVGf: results in absence of **both** AFDB and Pfam. Our ablations are single-factor only.
  Not run, and not even listed in §8's "what we did NOT do".
- Yi1G: MNRL under-specified — effective batch 1024 vs actual in-batch negatives, and the
  undefined `+` superscript in Eq. 1. Free, and V2's CachedMNRL / no-gather config is
  *directly* about this and could be turned into a real answer.
- Yi1G: how are two protein embeddings combined for KNN on PPI and peptide-HLA? Free.
- Yi1G: KNN regression, uniform or distance-weighted? Free.
- Yi1G: the DMS biological assumption (preserve fitness-induced ordering of WT-mutant
  distances rather than pulling all high-fitness variants to WT). CoSENT arguably already
  does ordering — that's a free rebuttal point we're not making.
- HNXd: absolute scores in Table 5. Free.
- HNXd: linear classifier on the **fine-tuned** base model, not just frozen. Not run, not in §8.

Partial:
- HNXd Q1 (retrieval or clustering eval, or embedding-space analysis): SCOPe covers
  retrieval. No clustering metric. The anisotropy analysis in §5.2 *is* the requested
  embedding-space analysis and is never connected to it — see §5 below.
- HNXd Q3 (bootstrap CIs on the reported metrics): retrieval only; §8 admits the 23-task
  table has none. HNXd's complaint was specifically about the sub-1% deltas in Table 2. We
  did CIs everywhere except the table they asked about.
- Yi1G baselines: HMMER and MMseqs2 done. ProtTucker, Foldseek, PLMSearch, DHR and Redl et
  al. not.

If the brief had one table mapping ~19 asks → evidence/plan/declined, this section would be
its most useful page. As it is, I had to build that map myself from the raw reviews, which
is precisely the work the brief exists to save.

---

## 3. The three "uncomfortable" findings — actionable, or hedged?

**5.2 (isotropy): well done.** Stated in its strongest form in §1 ("all of the top-1 gain at
150M"), given a clean control, the control fitted in the setting *most favourable to the
baseline*, and the residual claim stated honestly. This is how to report a result that hurts
you. I can act on it.

One correction, and it is not small. The reason given for withholding it from the rebuttal:

> Current decision: this stays **out of the rebuttal** (no reviewer asked, and it opens a new
> front)

**"No reviewer asked" is wrong.** HNXd, question 1, verbatim: *"an analysis showing how
ProtSent changes the local and global organization of the protein embedding space."* The
anisotropy table — mean random-pair cosine 0.848 → 0.152, participation ratio 7.9 → 52.5 —
is a direct, quantitative answer to that exact question, from the reviewer who said they'd
raise their score. We are sitting on the answer to a score-raising question and declining to
post it on the grounds that nobody asked. I think that's the single biggest strategic error
in the document, and it flows from the brief's own failure to build the coverage map in §2.

**5.1 (linear probe): the mitigation is longer than the finding and doesn't cover it.**
The finding is over 23 tasks. The mitigation — the layer sweep — is on **one task** (remote
homology), and the lead at layer 20 is ESM-2 0.7357 vs V2 0.7500, i.e. **+0.014, with no
CI**. The brief says "at that layer ProtSent-V2 leads". That thin, uncertain, single-task
lead is currently load-bearing for the most important honesty constraint in the paper. Two
things I'd want before I rely on it: the layer sweep on the aggregate, and a CI on that
+0.014. As written, §5.1 reads like it resolves the finding. It doesn't; it shows the
instrument is miscalibrated, which is a different and weaker claim (a true and useful one —
just say that).

**5.3: an entire numbered section that is a cross-reference.** Three lines, one of which is
"Covered in §3." Fold it into §1 and delete the heading.

**And a fourth finding that should have been in the list.** §6.6, presented as "description
errors":

> the remote-homology test split is *not* hierarchy-disjoint (it is TAPE's three holdouts
> pooled, 718 fold + 1,254 superfamily + 1,272 family = 3,244)

Paper, Appendix 9: *"Training and test sets are split by superfamily so that no superfamily
appears in both."* That is not a description error, it is a **validity problem with the
flagship +105% result**. 2,526 of 3,244 test items are superfamily- or family-level
holdouts, i.e. easier than what the paper claims to measure. The obvious next question from
any reviewer — what does the gain look like restricted to the 718 fold-level holdouts? — is
neither answered nor listed as not-done. Same for the SCOPe correction: 2,207 sequences at
**family** level, not 100,000 at **superfamily**, so the paper advertised a harder retrieval
task than was run. Both belong in §1's uncomfortable list, not as sub-bullets of item 6 of §6.

And this one gets seven words with no elaboration at all:

> the PPI decontamination description does not match what `data_prep.py` actually does

The paper claims a specific 50%-identity Bernett filter. If the code did something else,
that is a published methods statement that isn't true, and it's the exact control Yi1G
challenged. I need to know what it actually does before I touch that paragraph.

---

## 4. Spin, overclaiming, burying

Much less than I expected. But:

> **The headline is good.** ProtSent-V2-150M is the strongest model we have ever measured on
> structural retrieval

"Strongest we have ever measured" ranges over our own four checkpoints and two alignment
tools. §8 concedes we measured no learned retrieval baseline — no ProtTucker, PLMSearch,
DHR, ProTrek. Superlative over a field we chose. Say "strongest of our models, and ahead of
both alignment baselines we ran."

> at that scale it now beats *both* alignment baselines — including HMMER — on every
> retrieval metric.

Fine at 150M, and §3 self-corrects hard ("we must not state it globally"). But §1 is the
paragraph that gets pasted into the rebuttal. Fix it at source: scope it to 150M in the
sentence itself.

> **The top-1 claim is scale-dependent** … §6.2 **Lead with the 150M retrieval result.** It
> beats both alignment baselines, including HMMER at top-1

Internal contradiction the document doesn't flag: **§5.2 shows top-1 is the one metric
whitened vanilla erases** (V2-150M vs whitened ESM-2, R@1 +0.0089, unresolved). We are
recommending we lead with our least robust number. The robust numbers are R@10 and MAP
(+0.0219 and +0.0772 vs whitened vanilla, both significant; +0.2301 MAP vs HMMER). Lead with
ranking quality, mention top-1.

> That claim is strong, defensible, and survives every control we ran.

Three adjectives, and the last is false as stated — R@1 vs whitened vanilla at 150M did not
survive; the brief says so twenty lines earlier.

> **Verified, not assumed**

Slogan. The next sentence does the work.

> We answered it the expensive way.

Self-congratulation.

> **SCOPe-40 was deliberately not a filter target.** … This matches the ProtTucker precedent.

The reasoning (no split, would strip the corpus) is sound and I'd defend it. "Matches the
ProtTucker precedent" is an unsupported appeal carrying a lot of weight in a paragraph where
we decline the thing Yi1G specifically asked for. Cite it or cut it.

> **Fix three description errors the audit found**

Covered above. Two of the three are not description errors.

> None of these sink the paper.

Asserted before the evidence, in the summary. Probably true; delete it and let §6 make the case.

---

## 5. Underclaimed — where you've been harder on yourselves than needed

Five places, and the first is significant.

**a) The anisotropy result is filed as a threat when it is also a contribution.** Random
protein pairs at cosine 0.85–0.90 collapsing to 0.15; participation ratio 7.9/480 → 52.5/480.
That is a crisp, quantitative characterisation of what contrastive fine-tuning does to a
pLM embedding space, it is more informative than the UMAP figure currently in the paper, and
it is the literal answer to HNXd's Q1. It should appear as a *result* with the whitening
control as its honesty companion — not solely as the thing we're afraid a reviewer will find.

**b) The whitened-control result is stronger than its framing.** "Whitening closes the top-1
gap" is the sentence, but the actual finding is that whitening — fitted on the very gallery
it's applied to, maximally generous to the baseline — still loses R@10 and MAP at 150M and
loses all three at 35M by +0.063/+0.050/+0.114. That's a real, controlled, non-trivial
contribution and the brief undersells it.

**c) The 35M efficiency story is never told.** V2-35M: R@1 0.6852, MAP 0.6459. ESM-2 150M:
0.5535 / 0.4236. HMMER: 0.6970 / 0.4747. **A 35M model beats the 150M backbone at top-1 by
13 points and beats HMMER's MAP by 17.** For a paper whose entire premise is cheap
general-purpose embeddings, that comparison is a headline and it isn't made anywhere.

**d) Seed determinism is reported apologetically; it's a complete answer.** "0.5835±0.0000"
across 5 seeds, plus the reasoning that the uncertainty that matters is *which proteins are
in the test set*, which is what the bootstrap estimates. That is a correct and elegant answer
to HNXd's variability request. Say it as a win, not a caveat.

**e) We met Yi1G's numeric request exactly and never say so.** Yi1G: *"ensuring that test and
training sequences share less than 50% or even 40% sequence identity."* We ran 40% at 80%
coverage, on both named test sets, with independent post-hoc verification of the training
parquets. That's the stricter end of what was asked. It should be a sentence with Yi1G's
number quoted back at them.

Minor: the rare-class analysis (457 classes, median support 3, gap −0.0257 → −0.0036 when
restricted to ≥3 examples) is careful work that largely exonerates the result, written as if
it were damning.

---

## 6. Decisions I'm being asked to make

As I understand them — and note that **none of these is phrased as a question**. §6 is
titled "What this implies for the paper" and is written in the imperative. I had to infer
which are open.

1. Retire the general-purpose framing; reposition as retrieval / metric-space. (§6.1)
2. Lead the paper with the 150M retrieval result. (§6.2)
3. Add the whitened-vanilla control to the camera-ready. (§6.3)
4. Fix or justify the final-layer probe protocol. (§6.4)
5. Withdraw the label-scarcity claim. (§6.5)
6. Ship the three §6.6 corrections. (§6.6 — but see §3 above, two of these are bigger than labelled)
7. Whether the whitening control stays out of **the rebuttal**. (§5.2 — the only item
   actually flagged as an open judgement call, and it's buried mid-document behind "Current
   decision", which reads as closed)
8. Implicitly: whether to spend the ~17 h GPU on the unfiltered-corpus V2-config retrain. (§8
   — stated as a gap, never as a question, and with no deadline I can't judge feasibility)

I can decide 1, 5, 6 right now: **yes, yes, yes** — and 7: **no, put it in the rebuttal**, per
§3 above. I can decide 2 only in modified form: lead with R@10/MAP, not top-1. I cannot
decide 3, 4 or 8 without knowing the deadline, the rebuttal's remaining word budget, and
whether the rebuttal is already posted.

That's the core failure of the document as a decision request: **the one thing it explicitly
asks me about is the thing it buried deepest, and the things it states as decided are the
ones I'd want to argue with.** Invert that. Put an "Open questions for you" section at the
top with the 3–4 real forks, each with a default so I can reply "agree" and move on.

---

## 7. What's missing that I need before editing the paper

Ordered by how much it blocks me.

1. **The rebuttal text, or a one-paragraph summary of its claims — plus the deadline and
   whether it's posted.** Blocks everything time-sensitive.
2. **A reviewer-ask → evidence → plan table.** ~19 asks; ~half unaddressed; ~7 of those are
   free. Without this we will spend the remaining time on compute and lose the two reviewers
   who offered to raise their scores.
3. **A table-by-table disposition for the paper.** Table 2: do V2 per-task numbers exist at
   both scales in the same shape? Table 3: replaced wholesale (say so). Table 5: needs
   absolute numbers per HNXd and is partly repudiated by §4's few-shot finding. **Tables 4
   and 7 are the unmentioned landmine** — they are ablations of the V1 config, and V2 adopted
   two of the ablated settings, so the "Full model (ProtSent)" row is no longer ProtSent.
   Do we re-run them at V2, relabel them, or drop them? The brief cites those ablations as
   justification for the V2 config without noticing it has orphaned the table they live in.
4. **Is DMS in V2?** Plus the corpus-count reconciliation and the 15M term (§1a–b).
5. **Remote homology restricted to the 718 fold-level holdouts.** Given §6.6, this is the
   first thing a reviewer computes. If the gain survives there, it's a strong result; if it
   doesn't, we need to know before we write the abstract.
6. **What `data_prep.py` actually does for PPI decontamination**, in full.
7. **Numbers behind "a trained linear head beats 3-NN in almost every model/task/N cell."**
   One sentence, no table, and it retires an entire paper section. I need to see it.
8. **Layer sweep on the aggregate, and a CI on the +0.014 at layer 20**, before §5.1's
   mitigation can carry weight.
9. Matryoshka: eval dimension used, and truncated-dim results if they exist.
10. Ownership and time budget. Who writes which section, by when. Nothing in the document
    tells me what my job is.
11. Whether the V1 HF checkpoints are being retracted, superseded, or kept alongside.

---

## 8. Flabby prose — worst offenders, quoted

> We answered it the expensive way.

Delete. The table below it says it.

> **Verified, not assumed** (`verify_training_corpus.py`)

Delete the slogan, keep the sentence after it.

> This is the finding I would most want a reviewer not to discover first

Twelve words of theatre. → "A skeptical reader can run this in an afternoon."

> It is the stronger baseline (R@1 0.6970 vs MMseqs2's 0.6556) and it beats vanilla ESM-2
> 150M at top-1 by +0.144 — worth conceding, because it is true and it makes the rest
> credible.

Cut everything after the em-dash. Rhetoric about our own rhetoric.

> A whitened-baseline control is, in my view, something the camera-ready should contain.

→ "The camera-ready needs a whitened-baseline control."

> Current decision: this stays **out of the rebuttal** (no reviewer asked, and it opens a new
> front) and goes to you and into the paper revision. I think that is defensible; I also
> think it is the cheapest experiment a skeptical reader could run against us.

Two "I think"s in adjacent clauses, and the second half argues against the first half without
resolving it. This is the document's most important open question and it's written as a
shrug. → "Open question: in or out of the rebuttal? It answers HNXd Q1, and it's the cheapest
experiment a skeptic can run against us. I lean in."

> ### 5.3 The 150M remote-homology k-NN drop
> Covered in §3.

An entire numbered section whose first two words are a cross-reference. Fold into §1.

> None of these sink the paper. All three argue for a **narrower, better-supported claim**

"better-supported" is reassurance. And "none of these sink the paper" precedes the evidence.

> That claim is strong, defensible, and survives every control we ran.

Three adjectives, one fact, and the fact is wrong (R@1 vs whitened vanilla, unresolved).

> It costs us the broad version of the claim and buys the narrow one credibility that nothing
> else can.

"that nothing else can" — unfalsifiable flourish.

> The two config changes are not arbitrary

Then the paragraph above lists five (CachedMNRL, no gather-across-devices, no hard negatives,
proportional sampling, Matryoshka), on different hardware, at a different corpus size.

> Every analysis script has a `--selfcheck` that runs its assertions on synthetic data.

Reassurance without content. Either say what they assert or cut.

**Repetition.** The linear-probe finding is stated four times in materially the same words:
§1 item 1, §3 "**The probe decides the headline.**", §5.1 "the single most important honesty
constraint on anything we write", §6.1. Once in §1, once with numbers in §5.1. Cut the other
two.

**Nesting.** "(the other 514 are unachievable for any method — worth stating, since it means
R@K is capped at 0.767 on the full set)" — a parenthetical containing an aside containing the
actually-important number. Promote the 0.767 cap to its own sentence; it's a fact a reviewer
will want.

**What to keep.** This is the best-written passage in the document and the model for the rest:

> Retrieval answers this exactly, because every metric is a mean over per-query values —
> resampling queries gives the sampling distribution with no refitting. Report the **paired**
> intervals, not the marginals

Also good: the §4 identity-stratification paragraph, which explains why the planned analysis
was impossible, what replaced it, why the naive statistic is biased, and what to say out
loud. That's exactly the register the whole document should be in.

---

## 9. Ratings

**(a) How well I now understand the situation — 6/10.**
The strategic shape is clear and I could brief someone else on it. But I cannot reconstruct
the training corpus, don't know whether DMS is in V2, don't know what the rebuttal says or
whether it's posted, can't map the evidence to half the reviewer asks, and don't know which
paper tables survive. Those are not details; they're the things I'd need in the first hour of
editing.

**(b) How much I trust this reporting — 8/10.**
High, and earned. The tells that move me: running the whitening control at all, and fitting
it in the setting most favourable to the baseline; re-deriving the 150M remote-homology
numbers *because the result looked suspicious*, then reporting that the answer is a rare-class
artifact rather than quietly using it; a §8 that lists the one control that would make the
central claim airtight and admits it wasn't run; conceding HMMER beats vanilla ESM-2; stating
"our data does not support" a framing a reviewer handed us. That is a group reporting against
its own interest repeatedly.

Docked two points for: the §1 headline overreaching what §3 immediately walks back; "no
reviewer asked" being demonstrably false about a score-raising question; the unexplained
15,000,000 presented as verification; and classifying a flagship-claim validity problem as a
"description error."

**(c) Usefulness for deciding what to change in the paper — 5/10.**
It convinced me of the direction — narrow to retrieval/metric-space — and that's the decision
that matters most, so it isn't a failure. But it gives me no table-level plan, no reviewer
coverage map, no deadline, no ownership, no statement of what's already committed in the
rebuttal, and it leaves the ablation tables orphaned without noticing. It's an excellent lab
notebook and a mediocre work order.

**The single highest-value edit to this document:** replace §6 with (i) a reviewer-ask →
evidence → plan table, and (ii) an "Open questions for you, with defaults" block at the top.
The science is done. What's missing is the part that turns it into edits.
