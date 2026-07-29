# Slop review — `rebuttal/DRAFT_rebuttal_edited.md`

Read against `rebuttal/NEW_EVIDENCE.md` (source of truth) and `REBUTTAL_LEAKAGE.md`.

---

## PART 0 — Sourcing failures (higher severity than slop; fix these first)

The draft's own header says it "predates the current decontamination/retraining
runs." That is not a caveat, it is a disqualification: most of the draft's tables
cite numbers that appear in **neither** `NEW_EVIDENCE.md` nor `REBUTTAL_LEAKAGE.md`.
Several hedges flagged in Part 1 exist only to cushion these stale numbers — delete
the number and the hedge dies with it.

| Draft text | Problem | Source of truth |
|---|---|---|
| Every `150M` row (`ESM-2 0.4237 / ProtSent 0.5066 …`), lines 24-25, 95-96, 126-127 | No 150M ProtSent result is in `NEW_EVIDENCE.md`. §2/§8: only V1-35M and V2-35M exist; "No 150M model on the decontaminated data." | `NEW_EVIDENCE.md` §3, §8 |
| `MMseqs2 … R@1=0.3539, MAP=0.1795` (lines 29, 94) | Default-sensitivity run. Tuned `-s 7.5 -e 10` gives **R@1 0.5029 / MAP 0.3100** — which *beats* submitted V1's 0.4490. `REBUTTAL_LEAKAGE.md` §6.1 calls publishing the weaker figure "a self-inflicted integrity problem." | `NEW_EVIDENCE.md` §3, §5 |
| `R@10 = R@30 = 0.3856` (line 94) | Exact 4-decimal equality = truncated hit list, not a plateau. Measured: 0.5637 / 0.5641. | `REBUTTAL_LEAKAGE.md` §6.2 |
| Whole geometry paragraph: silhouette `-0.148 → 0.039`, NMI `0.852 → 0.893`, ARI `0.165 → 0.313`, ratio `0.701 → 0.418`, Spearman `-0.247 → -0.561` (line 31) | Not in either verified file. Nothing licenses these. | — |
| `155/2,207 queries` table and the `n=92` strict subset (lines 122-131) | Not in either verified file; and §6.9 of `REBUTTAL_LEAKAGE.md` says the 92-query subset has a **ceiling of 57/92 = 0.620**, unstated in the draft, so its `R@30 = 0.500` reads against an implied 1.0. Superseded by the identity-stratified analysis over all 1,693 eligible queries. | `NEW_EVIDENCE.md` §4 |
| "Paired bootstrap intervals include zero for both R@1 deltas but exclude zero for R@30 and MAP" (line 131); the CI placeholders at lines 49 and 183 | `NEW_EVIDENCE.md` §8: "**No paired bootstrap confidence intervals** on the Table 2 per-task deltas." Promising a reviewer a CI table that does not exist is the one unrecoverable error here. Answer HNXd/Yi1G with the win/tie/lose counts and medians in §6, which are real. | `NEW_EVIDENCE.md` §6, §8 |
| "the downstream split is disjoint in its evaluation hierarchy" (line 139) | False. TAPE remote homology repackaged: 718 fold + 1,254 superfamily + 1,272 family = 3,244 pooled, no column marking which. Two thirds is not fold-disjoint. | `NEW_EVIDENCE.md` §7.2 |
| PPI: "easy-linclust … at 50% identity … every STRING protein in a Bernett-containing cluster was removed" (line 141) | Contradicts released code. `data_prep.py` uses `easy-search`, STRING as query, **40%** identity, `--cov-mode 1 -c 0.8`, removing hit **query IDs**, not clusters. | `NEW_EVIDENCE.md` §7.3 |

Two whole answers are being written defensively when finished, stronger evidence
exists and is unused: the completed 40%/80% decontamination of all three corpora
with 0 surviving flagged sequences (§1), and **ProtSent-V2 retrained on it** (§2-3).
`REBUTTAL_LEAKAGE.md` §6.7: "The draft concedes the leakage point instead of citing
this work."

The single sentence the draft never writes, and the one that answers Yi1G #1:
**removing every pretraining sequence within 40% identity / 80% coverage of the
remote-homology test set improved remote-homology kNN accuracy, 0.6587 → 0.6668.**

---

## PART 1 — The slop, quoted

### A. Throat-clearing, restating the question, response meta-structure

**1. Line 14 (opening of HNXd)**
> "You asked us to separate two questions: whether ProtSent improves embedding neighborhoods, and how well a trained predictor can use those embeddings. That separation is the right one, and the results below are organized around it."

Restates the reviewer's question, praises the reviewer for asking it, then announces
its own table of contents. 39 words, zero information.
**Replace with:** *"Both probes are now run on all 23 tasks, test split, and they disagree. Under 3-NN, ProtSent-V1 beats ESM-2 35M on 11 of 20 comparable tasks (median +0.0075). Under a linear probe it loses 12 of 20 (median -0.0139)."*

**2. Line 69 (opening of jVGf)**
> "You named two questions that would change your assessment: whether ProtSent contributes more than structural-information injection, and where it sits on the generality-accuracy trade-off. We answer both directly."

Same pattern. "We answer both directly" is a promise to answer instead of an answer.
**Replace with:** *"Removing AFDB drops the mean relative gain from +6.7% to +3.2% and remote homology from +40.5% to +15.3%. Structure supervision is a large share of the gain, not all of it."*

**3. Line 20**
> "First, the reproduced retrieval comparison:"

Table scaffolding. Delete; label the table's columns with model, metric, and split
instead (reviewers cannot see the paper's tables).

**4. Line 116**
> "Eight concerns, answered in the order raised."

The eight numbered headers already say this. Delete.

**5. Line 18**
> "The geometry analysis you asked for was genuinely missing, and we have now run it."

"genuinely" is an intensifier defending a concession. Concession with no number
attached = free ground.
**Replace with:** the metric itself, or delete the sentence and open on the number.

**6. Line 120**
> "You are right that noting possible overlap was insufficient, so we ran the audit."

Supplication, and it concedes a point the finished decontamination already wins.
**Replace with:** *"We re-filtered all three pretraining corpora against the benchmark test sets at 40% identity / 80% coverage and retrained from scratch: Pfam -600,912 rows (2.11%), AFDB -9,102,652 (6.72%), STRING -4,178,737 (5.49%). Verified on the files training actually opened: 0 flagged sequences survived in any of the three."*

**7. Line 157**
> "Eq. 1 is malformed, as you note."

"as you note" is a curtsy. **Replace with:** *"Eq. 1 is malformed."* Then the correction.

### B. Promises where the work is already finished

**8. Line 37** — the worst instance in the draft.
> "Frozen logistic-regression and ridge probes are running now on the same splits, for vanilla ESM-2 and ProtSent."

They are done: 23 tasks x 4 arms x {3-NN, linear}, `--eval_split test`
(`NEW_EVIDENCE.md` §6). Saying "running now" about finished work reads as stalling,
and it hides the result — which is that ProtSent **loses** the linear probe. Reporting
the loss yourself is worth more than any positive number in the draft.
**Replace with:** *"Both probes are complete on the test split. Against ESM-2 35M over the 20 tasks comparable in both arms: 3-NN, V1 wins 11 / ties 3 / loses 6, median +0.0075; linear probe, V1 wins 4 / ties 4 / loses 12, median -0.0139. The probe decides the headline. The structural-retrieval advantage survives both; a general-purpose superiority claim does not survive the linear probe, and we have removed it."*

**9. Line 173** — same promise, repeated to Yi1G.
> "Frozen logistic-regression and ridge probes on vanilla ESM-2 and ProtSent, using identical splits, are running."
Same fix.

**10. Lines 51, 61, 155, 185** — a family of "the revision does X" sentences that
describe editing work rather than reporting a result:
> "The revised reporting uses absolute metric-point deltas and marks unresolved differences as unresolved, rather than bolding every positive point estimate."
> "Table 5 reports absolute scores and seed variability, with relative change secondary."
> "The final reporting separates supported improvements, supported degradations, and unresolved differences, and uses absolute deltas rather than only relative percentages."

Reviewers have not seen the revision and cannot check any of it. Each of these costs
~20 words and buys nothing.
**Replace with:** one sentence carrying a number, e.g. *"Median delta vs ESM-2 35M is +0.0075 under 3-NN and -0.0139 under the linear probe; individual sub-1% task deltas are within that noise and we no longer bold them."*

**11. Line 43**
> "If the linear probes do not support the label-scarcity claim, we narrow that claim."

Conditional promise about a completed experiment. They don't support it. Say so.

### C. Four-word promise stubs (a recurring tic — delete all)

> "We correct both." (L33) · "We specify this." (L165) · "Both points become explicit." (L161) · "We state that design choice explicitly." (L106) · "The main text did not explain this clearly enough, and the revision does." (L147) · "This is implemented but omitted from the paper." (L161)

Each ends a paragraph by telling the reviewer you will write down what you just wrote
down. Delete every one; the correction is already on the page. ~60 characters each,
six instances, and they cumulatively make the response sound like a to-do list.

### D. Hedges that weaken claims the evidence supports

**12. Line 98**
> "This is a basic MMseqs2 nearest-neighbor baseline, not an optimized profile-search system, and we infer nothing from it about HMMER, Foldseek, ProtTucker, PLMSearch, DHR, or ProTrek."

27 words pre-excusing a deliberately weak baseline. With the tuned run it is not
"basic" and needs no excuse.
**Replace with:** *"MMseqs2 `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`, same gallery, self excluded, no-hit queries scored as failures."* Flags stated, apology deleted.

**13. Line 98**
> "The trade-off we expect and will state as the paper's scope: a specialized retrieval system may perform better on its target retrieval problem, while ProtSent provides one frozen sequence embedding that also serves classification, regression, PPI, and fitness tasks."

39 words. "we expect and will state" = promise. "may perform better" = hedge that the
measurement contradicts — alignment *does* win, and you measured where.
**Replace with:** *"Measured, not asserted: MMseqs2 beats the best embedding model outright on 3 of 23 tasks under kNN (ec_classification, go_mf, beta_lactamase_peer) and 6 under a linear probe. On ec_classification it reaches F1_Macro 0.710 vs 0.598/0.562. It also falls below chance on DeepSol solubility (AUC 0.4185). That is the trade-off."*

**14. Line 169**
> "Proportional sampling (+7.0%) is effectively comparable to round-robin (+6.7%)."

"effectively comparable" hedges a fact that helps you: V2 was trained with
proportional sampling. **Replace with:** *"Proportional sampling gives +7.0% vs round-robin's +6.7%; V2 uses proportional sampling."*

**15. Line 131**
> "The strict subset therefore does not support a robust top-1 claim…"

"robust" used as a vague booster. Also the subset itself should go (Part 0).
**Replace with the powered analysis:** *"Across all 1,693 eligible queries binned by maximum identity to the pretraining corpus, V2's Recall@10 gain over ESM-2 is +0.1524 in the [0.2, 0.4) bin (n=164) and +0.1565 in [0.7, 1.0] (n=1,214). Per-query Spearman between identity and gain is -0.038 for R@10 and -0.116 for MAP (p < 3e-6). The advantage does not grow with proximity to pretraining data. Memorization predicts the opposite sign."*

### E. Dead ink

**16. Line 18**
> "…but retrieval numbers alone do not show whether the space itself is better organized. It is."

"It is." is a rhetorical flourish standing in for a measurement. Cut both clauses.

**17. Line 31**
> "The 35M model shows the same pattern."

Asserts a result with no number, so it is unverifiable to the reviewer and unusable.
Give the numbers or cut.

**18. Line 133**
> "This completed sensitivity filters clean queries while retaining the fixed full gallery."

Barely parses and repeats the table caption. Delete.

**19. Line 139**
> "We do not use the split alone as a leakage defense, and we state that residual limitation."

Meta-commentary on your own rhetorical posture. Delete — and fix the false claim in
the preceding sentence (Part 0).

**20. Line 169**
> "The per-task results show a trade-off rather than a universally better choice. The revision drops any claim that hard negatives or round-robin sampling is validated as generally superior."

The same sentence twice. Keep one, compressed.

**21. Line 43**
> "The revision frames the two probes distinctly: trained heads measure downstream task adaptation, while 3-NN measures whether useful relationships are already local in the frozen embedding space."

28 words defining terms the reviewer used first. Delete.

**22. Line 155**
> "The phrase 'effective batch size' was ambiguous."

Delete; keep only the correction that follows.

### F. Closing supplication — three near-identical paragraphs

**23. Lines 63, 110, 187**
> "These analyses target the criteria you named for reconsidering the paper. If they resolve your concerns, we would appreciate an updated assessment. If one point remains decisive, tell us which one and we will address it during discussion."
> "…If a specific remaining comparison is essential to your assessment, identify it during discussion and we will respond."
> "…If they resolve your concerns, we would appreciate an updated assessment. If one concern remains decisive, identify it during discussion and we will respond directly."

Reviewers of the same paper read each other's threads. Three variants of one template
reads as generated boilerplate. Keep **one** clause, once per response, and make it
different in each: *"If one concern remains decisive, name it and we will answer during discussion."*

**24. Line 110** — additionally, delete outright:
> "You indicated that clarifying these two axes could raise your score to accept."

Quoting a reviewer's conditional score back at them reads as leverage, not argument.
And in the same sentence:
> "…answer them without overstating what we have not run."

Self-congratulation on your own honesty. Honesty is demonstrated by the "we did not
run it" list, not announced.

### G. Sentences over ~35 words

**25. Line 82 (~34 words, and it restates its own first half)**
> "Those source-specific effects are the contribution we claim: a sequence-level metric space jointly shaped by evolutionary family, structural-cluster, physical-interaction, and fitness-order relations, that is, multi-relation sequence-level metric learning rather than structure-focused representation enrichment alone."

The "that is, …" clause repeats the colon clause in jargon.
**Replace with:** *"The claim is a sequence-level metric space shaped jointly by family, structural-cluster, interaction, and fitness-order relations. Each source moves a different task family."*

**26. Line 141 (~39 words, and factually wrong — see Part 0)**
> "Bernett test proteins were added to the STRING sequence pool, MMseqs2 easy-linclust was run at 50% identity and 80% target coverage, and every STRING protein in a Bernett-containing cluster was removed before the final STRING pairs were constructed."

**Replace with:** *"`data_prep.py` runs MMseqs2 `easy-search` with STRING as query and the Bernett test set as target at 40% identity, `--cov-mode 1 -c 0.8`, and removes hit query IDs. Our paper described this as `easy-linclust` at 50% with cluster-level removal; the code is what we just described, and we correct the text."*

**27. Line 86 (second sentence, 30 words)**
> "Preparing those inputs falls outside the rebuttal window, so we neither present an unreliable comparison nor promise that result as completed."

**Replace with:** *"We did not run it."*

**28. Line 177 (last sentence, 26 words, pure policy)**
> "We position ProtSent as a general-purpose sequence embedding and report specialized comparisons only where dataset, split, label level, and metric are matched."

**Replace with the concrete blocker:** *"ProtTucker's checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster, and is not mirrored on HF. We did not run it and claim no superiority to it."*

**29. Line 43 (27 words with a tacked-on excuse)**
> "A full end-to-end sweep over four encoders and 23 tasks does not fit the rebuttal window, and it would measure task adaptation rather than frozen representation quality."

The second clause converts a clean admission into an excuse and weakens it.
**Replace with:** *"We did not run a full fine-tuning sweep."*

### H. Concessions to KEEP (do not let a copy-editor cut these)

These are the credibility currency — each is a concession welded to a number:

- L31: "Not every metric improves: class-balanced alignment worsens." — keep, add the number.
- L33 / L137: the 2,207-vs-100,000 and family-vs-superfamily correction. Strengthen with the mechanism from `NEW_EVIDENCE.md` §7.1: the 100,000 is the evaluator's `max_samples` cap echoed into the results table, which converts an apparent 45x error into a logging artifact.
- L73: "removing AFDB reduces the mean relative gain from +6.7% to +3.2% … remote-homology gain from +40.5% to +15.3%."
- L80: "Removing STRING changes PPI from +5.3% to -0.5%."
- L106 / L149: the WT-anchored limitation.
- L169: "Removing hard negatives improves 20/23 tasks (+7.9%) against 16/23 (+6.7%)."
- **Add** the one missing from the draft entirely: MMseqs2 `-s 7.5` beats submitted V1 at R@1 (0.5029 vs 0.4490) and only V2 passes it (0.5256). `REBUTTAL_LEAKAGE.md` §6.8 — keep one R@1 story across all three responses; the defensible claim is that the effect is in ranking depth (R@10/R@30/MAP), which survives both the alignment comparison and decontamination.

---

## PART 2 — Style rules for the final rebuttal

1. **Open on a number, never on the reviewer's question.**
   Before: "You asked us to separate two questions: … That separation is the right one."
   After: "Both probes are now complete on the test split, and they disagree: 3-NN 11/3/6, linear 4/4/12."

2. **No sentence whose only content is that the paper will be edited.**
   Before: "Table 5 reports absolute scores and seed variability, with relative change secondary."
   After: delete, or attach the actual seed SD.

3. **Finished work is reported in past tense with numbers; never "is running".**
   Before: "Frozen logistic-regression and ridge probes are running now on the same splits."
   After: "Linear probe, 20 comparable tasks vs ESM-2 35M: V1 4 win / 4 tie / 12 lose, median -0.0139."

4. **Every concession carries a number or gets deleted.**
   Before: "The geometry analysis you asked for was genuinely missing, and we have now run it."
   After: "Not every metric improves: class-balanced alignment worsens." (with the value)

5. **Every "ProtSent > ESM-2" sentence names the probe, the metric, the split, and which model (V1 or V2).**
   Before: "ProtSent still improves 13/23 tasks."
   After: "ProtSent-V1 35M improves 13/23 tasks under the 3-NN probe on the test split."

6. **Every MMseqs2 number states its sensitivity flag.**
   Before: "A basic MMseqs2 nearest-neighbor baseline obtains family-level R@1=0.3539."
   After: "MMseqs2 `-s 7.5 -e 10`: R@1 0.5029, MAP 0.3100 (at `-s 5.7`, R@1 0.3847)."

7. **No number that is not in `NEW_EVIDENCE.md` or `REBUTTAL_LEAKAGE.md`. No 150M ProtSent row. No bootstrap CI.**
   Before: "Paired bootstrap intervals include zero for both R@1 deltas."
   After: use §6 win/tie/lose counts and medians, which exist.

8. **Delete "as you note", "you are right that", "we agree", "genuinely", "crucially", "robust", "comprehensive", "effectively comparable".**
   Before: "Eq. 1 is malformed, as you note."
   After: "Eq. 1 is malformed."

9. **State a limit once. Never explain that you are stating it.**
   Before: "We do not use the split alone as a leakage defense, and we state that residual limitation."
   After: delete the second clause.

10. **"We did not run X." — four words, no rationale clause.**
    Before: "Preparing those inputs falls outside the rebuttal window, so we neither present an unreliable comparison nor promise that result as completed."
    After: "We did not run it."

11. **Split anything over ~35 words. One clause, one fact.**
    Before: the 39-word generality-accuracy sentence at L98.
    After: two sentences, the second carrying the 3-tasks-under-kNN / 6-under-linear counts.

12. **No sentence that restates the previous sentence in different words.**
    Before: "The per-task results show a trade-off rather than a universally better choice. The revision drops any claim that hard negatives … is validated as generally superior."
    After: keep the first.

13. **Reviewers see no revision and no repo. Every table row carries model, metric, split, and n inline; no "see Table 3", no "the revision shows".**
    Before: "Table 3 reports cosine nearest-neighbor retrieval on SCOPe-40."
    After: "SCOPe-40, 2,207-sequence gallery, self excluded, no-hit queries counted as failures; Recall@K upper-bounded at 0.7671 because only 1,693 queries have a non-self same-family match."

14. **Every V1-vs-V2 comparison names the confounds.** Required by `NEW_EVIDENCE.md` §2 whenever both appear: 7x1024 effective batch vs 1x1024, no synthetic hard negatives, proportional sampling, one epoch — not decontamination alone.

15. **One closing clause, once, different per reviewer. No score-bargaining.**
    Before: "You indicated that clarifying these two axes could raise your score to accept. … we would appreciate an updated assessment. If a specific remaining comparison is essential …"
    After: "If one concern remains decisive, name it and we will answer during discussion."

---

**Budget note.** Draft is 16,864 characters across three responses, all placeholders
still empty. The cuts above free roughly 2,500-3,000 characters — enough to carry
the §1 decontamination table, the §3 retrieval table, the §4 identity-stratified
table, and the §6 both-probes table in full text, which is what the responses
actually need.
