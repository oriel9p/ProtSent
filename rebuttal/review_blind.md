# Blind review of the rebuttal experiment list

*Reviewer context: I have not read the paper, the reviews, or the code. I am judging
seven headlines and their opening paragraphs, as I would while triaging a rebuttal.*

---

## 1. What I think the contribution is

A general-purpose protein sequence **embedding** model — not a new architecture, a
**fine-tune of an existing pretrained backbone** (they keep comparing to "the untuned
backbone", so the backbone is someone else's). The novelty appears to be the training
*signal*: three heterogeneous supervision sources fused into one contrastive/metric
objective — Pfam family membership (28.5M), AlphaFold-DB structures (135.4M), and STRING
protein–protein pairs (76.1M) — roughly a "sentence embedding" recipe transplanted to
proteins, where the "sentences that mean the same thing" are proteins that share a
family, a fold, or an interaction partner.

The claimed payoff is retrieval: embeddings you can nearest-neighbour search for remote
homologs and structural analogs faster than, and now allegedly better than, alignment.
There is also a benchmark contribution — 23 downstream tasks evaluated under two probe
types.

That reading is inferred, not stated. Nothing in these seven headlines actually says what
the method *is*. That is itself a small mark against the packet: a rebuttal document
whose first line assumes I remember the paper is a rebuttal I will skim badly.

## 2. Most and least persuasive

**Most persuasive: #4 (gain does not track proximity to pretraining data).**

It is the only experiment that tests a *mechanism* rather than moving a number. #1 and #2
answer "did you cheat?" with "we removed the data and rechecked", which a reviewer must
take partly on trust. #4 answers the same question with a *prediction*: if the model is
memorising, gain should rise with identity to the training corpus. It does not — it is
flat-to-negative, and the largest gains sit in the 20–40% identity bin. That is a
falsifiable claim the authors could have lost, and it is the argument that survives even
if I distrust the filtering pipeline. It is also the only one that is per-query rather
than per-benchmark, so it is far harder to launder.

**Runner-up on credibility, not on persuasion: #5 and #6.** Reporting that MMseqs2 beats
you on 6 of 23 tasks, and that your model loses to its own untuned backbone 11–2 under a
linear probe, is self-harming in a way that buys real trust. I read the rest of the
packet more charitably because those two exist. But they persuade me the *authors* are
honest, not that the *method* is good.

**Least persuasive: #2 (performance went up after decontamination).**

The headline is framed as vindication and is the weakest item in the set. Three problems:

1. **A gain from removing data is not evidence of innocence; it is evidence something
   else changed.** Deleting 2.1% / 6.7% / 5.5% of the corpus and retraining from scratch
   should, on a clean benchmark, produce a wash — noise. Instead both probes improve by
   ~0.008–0.012. That is a *different training run*: new seed, new data order, new
   checkpoint. With no seed variance reported, +0.012 is indistinguishable from run-to-run
   jitter, and the paper is asking me to interpret jitter as a positive result.
2. **The buried number kills the section.** "The untuned backbone scores 0.5835 and
   0.6868." The decontaminated model's linear probe is 0.7016. So all of that
   contrastive training on 169M rows buys **+0.015 over the frozen backbone** on the
   flagship task. The kNN gap is large (0.5835 → 0.6668) but kNN is exactly the probe
   that rewards reshaping the metric space without adding information. That sentence is
   placed last and unremarked, which reads as hoping I do not do the subtraction.
3. **The decontamination threshold does not bind on the task it defends.** Remote homology
   detection is, by construction, the sub-30%-identity regime. Filtering training
   sequences at **40% identity / 80% coverage** removes close relatives that were never
   the mechanism of concern. Of course accuracy did not drop — the filter and the task
   barely overlap. To defend remote homology I want the filter run at the fold/superfamily
   level (structural or HMM-profile based), not a 40% ID cut.

## 3. "This paper matters to the field": **4 / 10**

The rebuttal, read cold, mostly narrows the contribution.

What is left standing after #5 and #6 is: *a fine-tune that improves nearest-neighbour
retrieval geometry over its backbone, on retrieval-shaped tasks, and is roughly neutral
or worse everywhere else.* #6 says it outright — 2 wins, 7 ties, 11 losses against the
untuned backbone under a linear probe. The linear probe is the standard instrument for
"is the information present in the representation". Losing on it while winning on kNN is
the textbook signature of a method that **rearranges** information rather than adding any,
and possibly discards some. That is a real and publishable finding for search
applications, but it is not a general-purpose protein embedding model, which is how the
framing reads.

#5 compounds it. If a tuned MMseqs2 beats the best embedding model on enzyme class
(0.710 vs 0.598) and GO-MF (0.585 vs 0.459) — margins that are not close — then for a
meaningful slice of the benchmark the neural approach is not the right tool, and the
honest headline is "embeddings are a speed/coverage tradeoff, not an accuracy win."

Points *for*: the multi-source (sequence + structure + interaction) supervision recipe is
a genuinely interesting idea, and #3's structural retrieval result — beating tuned
alignment at every cutoff, including MAP 0.4955 vs 0.3100 — is the one place where
something real seems to be happening. The MAP margin is large enough to survive my
skepticism in a way the Recall@1 margin (0.5256 vs 0.5029) is not.

4, not lower, because the negative results are useful to the community and the benchmark
plus the alignment baseline is a service. Not higher, because the headline claim has
visibly shrunk under the authors' own scrutiny.

## 4. "Did the hard thing rather than the easy thing": **9 / 10**

This is the clear strength of the packet and I want to say so plainly.

Retraining a model from scratch on a re-filtered 169M-row corpus, 11 GPU-hours × 7, inside
a rebuttal window, is the expensive answer to a contamination complaint. The easy answers
— "our splits are standard", "we report identity histograms", "a full retrain is
infeasible in the rebuttal period" — were all available and all usually accepted. They
took the costly one and then verified the filter against the actual files the training job
opened, semi-joined, with counts (0 / 27,929,772 etc.). That is the check a person does
when they want to know, not when they want to publish.

Running the alignment baseline across all 23 tasks — with no-hit queries scored as
failures rather than dropped, which is the choice that hurts *nobody but the honest* —
and then reporting that it wins several, is the hard thing. Reporting both probe types
when one of them says you lose 11–2 is the hard thing. Self-auditing your own submission
and publishing three description errors nobody caught is the hard thing.

Docked one point only because #7 is partly cleanup of a submission that should not have
shipped with a 100,000-vs-2,207 discrepancy in a results table, and because the *framing*
of #2 and #3 pulls in the opposite direction from all that rigour — spinning a
within-noise delta as a win undercuts the credibility the rest of the work earned.

## 5. The obvious objection they do not pre-empt

**One run, no seeds, no variance, no confidence intervals — and nearly every headline in
the packet is a delta small enough to be a seed.**

Concretely, what I would write in my review:

- #2 rests on +0.0081 (kNN) and +0.0117 (linear). #3's flagship Recall@1 claim is 0.5256
  vs 0.5029 — **a 0.023 margin on a 2,207-item gallery**, i.e. about 50 queries. That is
  well inside binomial noise for n=2,207, and no CI is offered. "Wins at every cutoff"
  is doing a lot of work for a margin that a different seed could erase. I would want
  ±CI on Recall@1 and a paired test (bootstrap over queries) versus MMseqs2.
- Because the decontaminated model is a *fresh run*, "decontaminated vs submitted" confounds
  the filter with the retrain. The missing control is trivial and damning by absence:
  **retrain from scratch on the unfiltered corpus with a new seed.** Without it, #2's
  entire claim is unfalsifiable.
- **Checkpoint selection is unaddressed.** Which checkpoint produced the reported numbers,
  and was it chosen using the benchmark? If checkpoints were scored on the same test sets
  the paper reports, the decontamination is moot — the test set re-entered through model
  selection.

Two more specific objections in the same family:

- **#4's binning is confounded.** Larger gains in the 20–40% identity bin (+0.286) than
  above 70% (+0.210) is exactly what a ceiling effect predicts: high-identity queries are
  already near-solved by the baseline and have no headroom. The comparison needs to
  control for baseline score per query (e.g. gain vs identity within matched-baseline
  strata, or a partial correlation), otherwise regression to the mean produces this
  result with no memorisation story either way. Separately, "max identity to the
  pretraining corpus" computed with an MMseqs2 prefilter *systematically misses remote
  homologs* — the low-identity bin is low-identity only in the sense that the search tool
  failed to find anything, which is precisely the population where undetected homology
  lives.
- **Decontamination covers one benchmark; the paper reports 23.** #1 says "the benchmark
  test sequences" (plural), #2 says remote homology is "the benchmark whose test set the
  corpus was decontaminated against". Which is it? If the other 22 task test sets were not
  filtered, then #5 and #6 — the all-task comparisons — are still reported on a possibly
  contaminated model, and the fix does not cover the claims.

## 6. Most confusingly written

**#6**, with **#2** a close second.

In #6 I had to re-read the last sentence three times:

> "Against the untuned backbone across 20 comparable tasks, the model wins 10 / ties 3 /
> loses 7 under nearest-neighbour, and wins 2 / ties 7 / loses 11 under a linear probe."

- *Who* wins is never stated in the clause — "the model wins 10" only resolves to "the
  proposed model, against the backbone" if you carry the subject from the previous
  sentence across a probe-type switch. On first pass I read the linear-probe half as the
  backbone's record.
- **20 tasks here, 23 in #5.** Three tasks vanish behind the word "comparable" with no
  explanation. Dropped tasks in the comparison that makes you look worst is the single
  thing a skeptical reviewer will circle first, and it is unaddressed.
- **"Ties" is undefined.** Ties within what margin? With 7 ties out of 20 under the linear
  probe, the tie threshold determines whether the record is 2–11 or something far worse.
  Given the packet reports no variance anywhere else, I suspect "tie" is an eyeballed
  band, and it is deciding the headline.
- The headline says the probes "disagree", which is a soft word for what the numbers say,
  which is that under the standard probe the method loses to the thing it fine-tuned.

#2 is the runner-up: the framing ("went up, not down") points the reader away from the
one number in the paragraph that matters. I had to re-read to notice that 0.7016 vs the
backbone's 0.6868 is a +0.015 linear-probe gain, and to notice that the reason it "went
up" may simply be that the filter does not bind on this task. Two re-reads, both to
recover something the prose was arranged not to emphasise.

## 7. Would this move a reject vote?

**Partially.**

It moves me from *reject* to *borderline / weak reject*. It does not get to accept.

What it earns: I no longer believe the results are a contamination artifact — #1 and #4
together largely retire that objection, and the effort is real. I also now trust the
authors' reporting, which is worth more than it sounds, because #5, #6 and #7 are all
volunteered against interest.

What it does not fix, and what is missing:

1. **Seeds and intervals.** At minimum: ≥3 seeds on the retrained model, plus the
   unfiltered-retrain control, plus bootstrap CIs on the #3 retrieval margins. Every
   headline delta in this packet is currently unfalsifiable. This is the blocking item.
2. **An explanation of the linear-probe result, not just its disclosure.** If the
   fine-tune degrades linearly-decodable information on 11 of 20 tasks, say so as a
   finding, characterise it (is it anisotropy? dimensional collapse? measure the
   embedding spectrum), and rescope the paper to retrieval. A paper that owns "this is a
   retrieval-geometry method, and here is why it costs you elsewhere" is a better paper
   than one that reports the loss in a table and moves on.
3. **Decontamination at the level the claim lives at.** 40% ID / 80% coverage does not
   defend a remote-homology claim. Redo it at the fold/superfamily level, or drop the
   claim that the benchmark is decontaminated for remote homology.
4. **A revised manuscript, not a diff list.** #7 tells me three descriptions in the
   submission were wrong, including a 45× overstatement of the evaluation gallery
   (100,000 → 2,207) and a "hierarchy-disjoint" split that is a pooled concatenation. That
   last one is not a description error, it is a *different experiment* — pooled holdout
   levels change what the remote-homology number means, and #2 and #3 both sit on top of
   it. I need to see the corrected split defined and the numbers recomputed under it
   before I can score the central result at all.
5. **A one-paragraph statement of what the paper now claims.** After this rebuttal the
   contribution is materially narrower than the submission's. I would like the authors,
   not me, to write the new claim.

Item 4 is the one that keeps me at weak reject rather than borderline accept. Everything
else in this packet is answerable with compute the authors have clearly shown they will
spend; the split-definition error means the flagship number may not yet mean what either
of us thinks it means.
