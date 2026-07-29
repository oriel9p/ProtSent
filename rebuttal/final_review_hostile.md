# Reviewer Yi1G — post-rebuttal assessment (submission 28056)

**Score: unchanged, 2 (Reject). Confidence 4.**

The rebuttal is the most honest one I have read this cycle. It concedes more than I
asked for, it names its own confounds before I could, and the retrain was real work.
None of that is a reason to accept. The concessions remove the paper's claims; the
work that replaces them is uncontrolled on exactly the axis I raised, and the model
that produces it is not the model under review.

Below: the five objections that survive, in the form I would post them, each with the
authors' best available answer stated as strongly as I can state it, and my verdict.

---

## 1. KILL SHOT — The one claim that survives is the one benchmark you chose not to control, and the control you offer has no power against the leakage you concede exists

**As I would post it:**

> The rebuttal's own framing is that the camera-ready is "a 35M retrieval-and-remote-homology
> paper," and that "our strongest surviving result sits on SCOPe-40, the one benchmark we could
> not decontaminate." That sentence is the whole review.
>
> The reported facts about SCOPe-40 are: median maximum identity of the 2,207 gallery domains to
> the training corpus is **0.908**, with **none below 20%**; the corpus was never filtered against
> it; and the supervision signal is **Foldseek-cluster and Pfam-family co-membership** — i.e. a
> partition of protein space that is a direct proxy for the SCOPe family labels being scored. The
> model was trained to co-locate members of a structural clustering, and is then evaluated on
> whether it co-locates members of a structural clustering, over a gallery it has essentially all
> seen.
>
> The offered control is that the per-query R@10 gain over ESM-2 is flat in sequence identity
> (+0.1524 at [0.2,0.4), n=164; +0.1565 at [0.7,1.0], n=1,214), with Spearman +0.038, p=0.45 among
> the 404 queries the backbone fails outright. The authors then state, correctly, that "a training
> pair sharing a test domain's fold at 15% identity survives a 40%-identity filter" and that "a flat
> slope is what that leakage would also produce."
>
> That is an admission that the control has **zero discriminating power**. A test whose positive and
> negative outcomes are both predicted by the hypothesis under test is not evidence. The rebuttal
> presents it in a reassuring register — three paragraphs, four statistics, a partial correlation —
> and then concedes in the fourth paragraph that none of it addresses the mechanism. The V2 − ESM-2
> margins on SCOPe-40 (R@1 +0.1855 [+0.1618, +0.2097], R@10 +0.1607, MAP +0.2232) are the largest
> effects anywhere in this submission, and contamination of the label-proxy kind is a confound with
> no upper bound on its magnitude. I cannot distinguish "restructures the embedding space" from
> "memorized the clustering it was trained on and is being scored against a near-copy of it."
>
> The authors name the required experiment themselves — "excluding queries whose fold is among
> training clusters" — and say they did not run it. Mapping 2,207 SCOPe domains onto the Foldseek
> clusters and Pfam families present in the corpus and re-scoring the held-out remainder is hours,
> not weeks, and it is smaller than the full from-scratch retrain that was completed. Its absence,
> in a rebuttal whose surviving contribution rests entirely on this benchmark, is the decisive fact.

**Authors' best available answer:**

Three real arguments exist.

(a) *SCOPe cannot be decontaminated at the corpus level.* True, and not evasion — at median 0.908
identity, filtering the corpus against SCOPe-40 removes essentially every structured domain, which
would leave nothing to train on. This is a genuine structural property of the benchmark.

(b) *The [0.2,0.4) identity bin is already a decontaminated evaluation.* This is the strongest single
argument in the entire rebuttal and it deserves to be said out loud: those 164 queries sit below the
same 40% threshold applied to every filtered corpus, and V2's MAP gain there is **+0.2859, larger
than the +0.2232 overall**. On the subset that passes the project's own decontamination criterion,
the effect is bigger, not smaller. Add the negative identity–gain Spearman (−0.116, p < 3e-6;
partial −0.081 after a headroom control): the *more* contaminated queries gain *less*.

(c) *ESM-2 is contaminated too.* SCOPe domains are in UniRef50. Both arms of the comparison have seen
the gallery during pretraining; what is measured is a delta on top of shared exposure.

**Does it hold? No.**

(a) explains why the experiment is hard; it does not license reporting the result without the
control, and it does not excuse skipping the *evaluation-side* exclusion, which is cheap and which
the authors themselves identify.

(b) is real evidence and I credit it — but it controls the wrong variable. Nobody alleged verbatim
sequence memorization. The allegation is label leakage through a supervision signal that is a proxy
for the test partition, and identity is orthogonal to that: a training domain at 15% identity in the
same Foldseek cluster sits in the [0.2,0.4) bin and leaks the label perfectly. The 164-query result
is therefore consistent with the objection rather than a rebuttal to it. n=164 is also thin for a
bin whose purpose is to carry the entire leakage defense.

(c) is fair as far as it goes, and it is the reason I do not think the *entire* effect is leakage.
But contrastive fine-tuning on a label proxy is precisely a mechanism that converts shared pretraining
exposure into benchmark-specific advantage, so "both arms saw it" does not neutralize a delta produced
by training on the labels.

**Net:** the paper's only surviving positive claim is uncontrolled, the control offered is
non-discriminating by the authors' own statement, and the discriminating control was not run.

---

## 2. The paper being defended is not the paper under review, and every number in the abstract is now withdrawn or replaced

**As I would post it:**

> Let me enumerate what the rebuttal removes from the submission.
>
> All 150M results, "including the abstract's +105% and +19.9%" — withdrawn, because there is no 150M
> model on the decontaminated corpus and because the 150M MNRL call saw **16** in-batch negatives, not
> the 1,024 the paper describes. That deletes Contribution 3 ("improvements are consistent across two
> model scales") outright.
>
> The 35M abstract numbers do not survive either. "+15.5% Recall@1 on SCOPe-40" was superfamily-level
> over a 100,000-sequence gallery; the rebuttal states this was the evaluator's `max_samples` cap and
> the metric is actually **family** level over **2,207** domains, and reports entirely different
> figures (ESM-2 0.4991, V1 0.5854). Table 3 is not corrected, it is replaced. "+40.5% on remote
> homology" is a relative macro-F1 change on the suite's *default* split, which the rebuttal declines
> to mix with the test-split numbers it now defends. "Improves 16 of 23 tasks" is superseded by an
> 11/3/6-of-20 test-split tally, and the framing it supported is withdrawn.
>
> So: **not one number in the abstract survives the rebuttal.** Add the method corrections — the MNRL
> objective was misdescribed by a factor of 16 (35M saw 64 negatives, not 1,024); the DMS loss "operates
> on single proteins rather than pairs" is wrong and it is pairwise; the STRING decontamination text
> describes `easy-linclust` at 50% with cluster removal while the code does `easy-search` at 40% with
> query-ID removal; and the leakage-control paragraph's claim that "the remote homology fold-level
> evaluation uses a disjoint fold partition" is false, the split being TAPE's three holdouts pooled
> (718 + 1,254 + 1,272 = 3,244).
>
> That last one matters beyond bookkeeping: the paper's *stated defense against my primary weakness*
> was factually incorrect. Every result now defended comes from a different model (V2), trained on a
> different corpus, with a different configuration, evaluated on a different split, against a benchmark
> defined differently from the one in Table 3. That is not a revision I can recommend conditionally.
> It is a different submission, and it has not been reviewed.

**Authors' best available answer:**

The errors were self-reported, not caught by reviewers — that is the behavior the process wants, and
punishing it teaches the wrong lesson. NeurIPS rebuttals routinely add experiments; V2 is the paper's
own method applied to a corrected corpus with the configuration its own published ablations favor
(Table 4 already reported that hard negatives hurt and proportional sampling was fine), so it is not
an arbitrary new model. And the direction of every correction is *against* the authors: they are
withdrawing claims, not smuggling in stronger ones.

**Does it hold? It holds as a defense of the authors' conduct and not at all as a defense of the paper.**

I want to be explicit that I read this rebuttal as unusually scrupulous and I would say so publicly.
But conditional acceptance requires that the reviewed artifact, with stated fixes, becomes the
camera-ready. Here the reviewed artifact's headline results are wholly withdrawn, its central framing
is withdrawn, one of its two tables of headline numbers is replaced by a differently-defined benchmark,
its objective is misdescribed, and its leakage-control paragraph is false. The replacement material is
substantial enough to require full review, which the discussion period cannot provide.

---

## 3. The improvement is not attributable to decontamination, and V2's configuration was selected on the evaluation benchmarks — so the rebuttal contains no clean held-out measurement at all

**As I would post it:**

> This is the direct answer to my weakness 1, and the answer is "we cannot tell."
>
> V2 differs from V1 in **four** ways simultaneously: filtered corpora, dropped synthetic hard
> negatives, proportional instead of round-robin sampling, and a true 1,024-example contrastive batch
> where V1's loss call saw 64. No unfiltered-corpus retrain at the V2 configuration exists. The
> rebuttal states the consequence itself: "nothing is attributable to decontamination in either
> direction." A full from-scratch retrain was run and the causal question I asked is no better answered
> than before it.
>
> Worse, the rebuttal volunteers the second half: "those ablations were scored on these same benchmarks,
> so V2's configuration was chosen with benchmark results in view — a selection channel the corpus
> filter does not touch, and V2's numbers are therefore not a clean held-out measurement." Every V2
> number in this rebuttal — the SCOPe table, the remote-homology accuracies, the few-shot means, the
> linear-probe tally — comes from a configuration selected by scoring candidates on the test benchmarks.
> A corpus filter does not repair test-set selection; it is a strictly separate leakage channel, and it
> is the one that cannot be fixed by any experiment run during discussion, because the selection already
> occurred.
>
> Concretely, the single number the decontamination was supposed to deliver — remote homology,
> V1 → V2, 3-NN accuracy 0.6587 → 0.6668 — is **+0.0079**, which the authors correctly place inside
> their own checkpoint spread (checkpoint 4,000 vs final differs by 0.005–0.008 on every structural
> metric). The retrain's headline deliverable is a delta the authors themselves decline to call
> resolved.

**Authors' best available answer:**

(a) The confound is disclosed in the response text, in both the HNXd and Yi1G threads, in the same
paragraph as the number.

(b) The weak claim they actually make is exactly supported: "a decontaminated corpus still trains a
model at least as good as the submitted one on the filtered task." Nothing stronger is asserted.

(c) **Selection over roughly six configurations cannot manufacture the large effects.** This is a good
argument and I accept it. A search over ~6 candidates on a fixed benchmark suite inflates small deltas;
it does not produce +0.1855 R@1 on SCOPe-40 or +0.0833 3-NN accuracy on remote homology
(ESM-2 0.5835 → V2 0.6668). Selection bias bounds the interpretation of sub-0.02 cells, which is
exactly where the authors have already withdrawn.

(d) V1 vs stock ESM-2 remains a comparison free of *this* particular confound — V1's configuration
was the pre-registered default, not a selected one.

**Does it hold? Partially, and I will say which part.**

(c) is correct and I withdraw any suggestion that selection explains the large structural effects. What
remains is narrower but still fatal to the rebuttal's own framing: the response opens with "Your leakage
objection was correct and we treated it as decisive," and the measured outcome of treating it as decisive
is that leakage remains unmeasured. My weakness 1 does not move from "possibly leaked" to "not leaked";
it moves from "possibly leaked" to "still unmeasured after a full retrain." And the honest disclosure in
(a) is a reason to trust these authors, not a reason to accept this paper.

---

## 4. Two of twenty-three-plus test sets were decontamination targets; one of them produced no downstream number at all; so the all-task results remain uncontrolled

**As I would post it:**

> The filter targets were `remote_homology` test (3,244 sequences) and `ppi_bernett` test (3,022
> sequences). "Those two test sets were the only filter targets — every other benchmark test set,
> SCOPe-40 included, was not."
>
> Then: "`ppi_bernett` is a pair-input task and is not in the 23-task sweep, so the filter target that
> cost 4,178,737 STRING pairs has no downstream result."
>
> So half the decontamination effort — a four-million-pair deletion aimed at the leakage path I flagged
> most specifically — yields **zero** measurable numbers, and the paper's PPI claim (+5.3% AUC) remains
> a pre-decontamination V1 figure. The other half yields the +0.0079 that item 3 shows is unresolved.
> Every remaining cell of the 23-task suite — EC, GO-MF, solubility, metal-ion binding, GB1, fluorescence,
> beta-lactamase, all of it — was trained against unfiltered corpora. The 3-NN tallies (V1 11/3/6, V2
> 10/3/7 of 20) that are the last quantitative support for any breadth claim are therefore uncontrolled
> in the same way the submission was.

**Authors' best available answer:**

(a) Filtering a 169M-row corpus against 23 benchmark test sets at 40%/80% is combinatorially expensive
and, as the SCOPe case demonstrates, would strip the corpus for structurally-defined benchmarks.

(b) **The asymmetry favors them where it matters.** Residual contamination inflates ProtSent, so the
*negative* results are conservative: a contaminated V2 still loses to stock ESM-2 on 11 of 20 tasks
under a linear probe. Contamination cannot explain a loss.

(c) The two targets chosen were the largest claimed gain (remote homology, +40.5%) and the most direct
leakage path (STRING → Bernett PPI), i.e. the highest-value pair.

(d) Decontamination quality on what *was* filtered is verified properly: semi-join on the parquet files
training opened (0 flagged survivors), row counts summing to the 169,231,379 in the training log, and
negative controls with exhaustive re-search for AFDB given its 89.4% prefilter recall (0 hits, rule-of-three
bound ~0.3%). That is better verification than most papers do at all.

**Does it hold? It holds for the negative results and for nothing else.**

(b) is legitimate and I accept it: the linear-probe losses are strengthened, not weakened, by residual
contamination. (d) is genuinely good practice on the subset it covers. But every claim the paper still
makes is a *positive* result on an *unfiltered* target — SCOPe-40 above all — and (b) provides no cover
there. And (c) is undercut by its own outcome: the higher-value of the two targets produced no number,
which means the decontamination programme delivered one interpretable measurement in total, and item 3
shows it is not interpretable.

---

## 5. The depth win over alignment — the only comparative claim still made — is a candidate-list artifact the authors identify and did not fix

**As I would post it:**

> After the R@1 concession (V2 − HMMER = −0.0124 [−0.0372, +0.0124], a tie; V1 − HMMER = −0.1110
> [−0.1388, −0.0827], a loss), the sole surviving comparative claim is ranking depth: V2 − HMMER +0.1412
> at R@10 and +0.1708 MAP.
>
> The rebuttal then supplies the reason to discount it. **691 of 2,207 queries return no phmmer hit at
> `-E 10` and are scored 0 at every K.** Both alignment tools flatten from R@10 to R@30 (+0.0171 HMMER,
> +0.0165 MMseqs2) where the embeddings do not (+0.0726 ESM-2, +0.0414 V2). That flattening pattern is
> the exact signature of a truncated candidate list, not of degraded ranking: a tool that has stopped
> emitting hits cannot gain recall at larger K regardless of how well it ranks. Roughly 31% of the
> gallery scoring a structural zero because of a reporting threshold is not a measurement of HMMER's
> ranking quality.
>
> The fix is `-E` set large (or `--max`) on a **2,207-sequence** gallery. That is minutes of CPU. It was
> not run, and instead the margin is labelled "an upper bound." An upper bound of unknown tightness on
> the only comparative claim in the paper is not a result. Note also the other direction: run over all
> 23 tasks, plain MMseqs2 beats the *best* embedding arm on 3 tasks under 3-NN and 6 under a linear
> probe — EC F1-macro 0.710 vs 0.598 (ESM-2 35M) and 0.562 (V1); GO-MF 0.585 vs 0.459/0.443;
> beta-lactamase Spearman 0.8026 above every embedding arm including V2.

**Authors' best available answer:**

(a) They flagged it themselves, unprompted, in all three threads, and stated the margin is an upper bound.

(b) The settings are already well beyond defaults — `-s 7.5 -e 10 --max-seqs 300` for MMseqs2 (they show
`-s 5.7` gives SCOPe R@1 0.3847, so they did not sandbag the baseline), phmmer `-E 10` top 300.

(c) **Scoring no-hit as failure is the user-facing semantics.** A search tool that returns nothing has
failed the user; silently dropping those queries is the standard way retrieval baselines get flattered.
This is a defensible methodological position and I would defend it myself in another context.

(d) The 691 figure is over all 2,207 domains; the metrics are computed on the 1,693 eligible queries, so
the number of no-hit queries actually inside the scored set is smaller than 691 and the artifact is
correspondingly smaller than it looks.

**Does it hold? Not as claimed, though it is close to holding.**

(c) is right, and it would fully support a claim worded as *"where alignment returns nothing, the
embedding still returns a usable ranked list"* — a genuinely useful and defensible property. It does not
support *"decisively ahead at ranking depth,"* which is a claim about ranking, evaluated on lists that
partly do not exist. (d) is a fair point that I would want quantified — the rebuttal does not state how
many of the 1,693 eligible queries are no-hit, which it should, and that omission is itself a small
reporting defect. The decisive issue is (a): naming a confound is not controlling it, and this particular
control costs less compute than any other experiment in the rebuttal.

---

## Is what remains a NeurIPS paper?

**No.** Stated plainly, because the authors asked to be told which item is decisive.

What is left after the rebuttal's withdrawals: a 35M ESM-2 contrastively fine-tuned on a heterogeneous
relation graph, which (i) improves 3-NN remote-homology accuracy from 0.5835 to 0.6668 on a
corpus-decontaminated test split — the best-controlled result in the submission and a real one; (ii)
improves SCOPe-40 family-level retrieval depth substantially on a benchmark that was never controlled;
(iii) ties the strongest alignment baseline at top-1 and beats it at depth by a margin its authors call an
upper bound; and (iv) is **worse than its own untuned backbone** on 11 of 20 tasks under a trained linear
probe, median −0.0107.

Result (i) is one number. Results (ii) and (iii) are each explicitly qualified into unresolvability by
their own authors. Result (iv) is the direct negation of the title, the abstract, the Sentence-BERT
framing, and Contribution 2. There is no 150M model, no clean attribution for the retrain, no
held-out configuration selection, no fold-exclusion control, no PPI measurement, no ProtTucker comparison,
no joint no-AFDB/no-Pfam ablation, and no training-seed variance. That is not a NeurIPS paper. It is a
solid workshop submission, or the raw material for a considerably better paper than this one.

I want to name that better paper, because I think it exists and the authors are close to it. The most
interesting findings in this rebuttal are all negatives: that contrastive fine-tuning improves k-NN
geometry while *destroying* linearly-decodable information (2 win / 7 tie / 11 lose, and the layer sweep
showing the final layer is the worst layer for remote homology in *both* models); that synthetic hard
negatives make things worse (20/23 tasks improve without them vs 16/23 with); that alignment still wins
top-1 and wins outright on homology-transferable annotation while being *below chance* on DeepSol
solubility (AUC 0.4185); and that "linear degrades under label scarcity while k-NN stays competitive" is
false in their own data at every N including 50. A paper about *when contrastive fine-tuning of a pLM
helps and what it costs*, with the fold-exclusion control run and the alignment baselines run at loose
thresholds, would be more useful to this community than the paper submitted, and I would review it
favorably. But it is a different paper.

**I would raise my score to a 4 (borderline reject, arguable) if, during discussion, the authors report
the fold-exclusion SCOPe evaluation — queries whose Foldseek cluster or Pfam family appears in the
training corpus removed — and the alignment baselines re-run with the reporting threshold lifted.** Those
two experiments are cheap and they decide item 1 and item 5. Nothing else on my list can be fixed in the
discussion period, and even both of those would not get me to accept, because items 2 and 3 are
structural. I am telling the authors this so they do not spend the period on the wrong thing.

---

## Defects against the stated rebuttal rules (report, not review)

Four found. All minor relative to the substance above, but the third is the kind of thing a hostile
reviewer will use.

1. **Yi1G response — an uninterpretable number.** "V2's MAP gain there is +0.2859 vs +0.2232 overall."
   `+0.2232` appears exactly once in the Yi1G body, with no baseline named. It is the V2 − ESM-2 MAP
   difference from the paired bootstrap, but that row is only in the HNXd response; the Yi1G table gives
   0.6459 vs 0.4210, whose difference is 0.2249, not 0.2232. A reviewer reading only their own thread
   cannot resolve where +0.2232 came from or what it is a gain over. Violates the "every number with its
   metric, split and model" rule. Fix: "V2 − ESM-2 MAP on those 164 queries is +0.2859, against
   +0.2232 over all 1,693 (paired bootstrap)."

2. **jVGf response — implies the V1/V2 comparison is a decontamination ablation.** The jVGf table labels
   the row "ProtSent-V2 35M (**decontaminated retrain**)" and the jVGf body never mentions that V2 also
   changed the sampler, dropped hard negatives, and fixed the batch semantics from 64 to 1,024 — the
   strings "1,024", "hard negative" and "sampling" do not occur anywhere in that response, nor does the
   disclaimer that appears in both other threads. A jVGf reader will read V1 → V2 as a controlled
   decontamination ablation, which the hard rules forbid implying. Reviewers read each other's responses;
   the inconsistency across threads is worse than the omission. Fix: one sentence in jVGf §1, e.g. "V2 also
   changes sampling, drops synthetic hard negatives and uses a true 1,024-example batch where V1's loss
   call saw 64, so V2 − V1 is not a decontamination ablation."

3. **Yi1G response — stated character count is wrong.** The marker says 9,969; the actual body between
   BEGIN/END, stripped, is **9,757**. Under the 10,000 limit, so not a violation, but a stale count on the
   response that sits closest to the ceiling is exactly the metadata you do not want wrong at paste time.
   (HNXd 9,610 and jVGf 9,234 are both stated correctly and both under the limit.)

4. **Missing figure that item 5 needs.** The rebuttal reports 691 no-hit phmmer queries out of 2,207 but
   never states how many of the **1,693 scored** queries are no-hit — which is the number that actually
   sizes the coverage artifact in the metrics being defended. As written it invites the worst-case reading
   (31%), and the true figure is probably smaller and more favorable. Add it.

No links, no attachments, no figure references, and no "see the revised paper" in any of the three bodies.
No response implies a decontaminated 150M model; all three deny one exists. No response claims a top-1 win
over alignment, general superiority to ESM-2, or a controlled decontamination ablation — except for defect
2 above, which implies the last of those by omission.
