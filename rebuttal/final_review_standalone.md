# Standalone reviewer read of the ProtSent rebuttal

**Position:** reviewer of submission 28056. I have read the submitted paper and the three
rebuttal responses. I have **no** access to the code, the results files, the retrained
checkpoints, or any revised manuscript. Everything below is what I can and cannot
reconstruct from the response text plus the submitted PDF.

Overall: this is an unusually candid rebuttal and the honesty is real — several claims are
withdrawn outright and the forbidden overclaims are, to their credit, **not** made. But the
number-hygiene is poor in a specific and repeated way: the rebuttal moves the entire
evaluation onto a **new split, a new task set, a new SCOPe level and a new metric family**,
then reports numbers from the old system and the new system in adjacent sentences without
telling me which is which. I cannot check most of the arithmetic that matters, and one of
the verification claims does not add up.

---

## 0. Hard-rule compliance check

| Rule | Verdict |
|---|---|
| Each response < 10,000 chars (BEGIN/END body, stripped) | **PASS.** Measured: HNXd 9,610; jVGf 9,234; Yi1G **9,757**. |
| Stated character counts accurate | **FAIL (minor).** Yi1G's header says "9969" but the body is 9,757 — off by 212. HNXd and jVGf match exactly. |
| No links / attachments / figure references | **PASS.** No URLs, no figures. |
| No "see the revised paper" | **BORDERLINE FAIL.** Two forward claims about an artifact I cannot see: HNXd "*The camera-ready is therefore a 35M retrieval-and-remote-homology paper*" and jVGf "*we add it to Related Work as such*" (ProTrek). Neither is verifiable; both ask for credit for a revision I have not been shown. |
| "ProtSent beats alignment at top-1" must not appear | **PASS, emphatically.** All three responses state the opposite ("Alignment is the better top-1 method"; "We do not claim to beat alignment at top-1"), report V2−HMMER R@1 = −0.0124 [−0.0372, +0.0124] as unresolved, and report V1's outright losses to both tools. The supportable framing (tie at top-1, ahead at depth) is exactly what is written. |
| "ProtSent generally superior to ESM-2" must not appear | **PASS.** The linear-probe records (V1 4/4/12, V2 2/7/11) are reported prominently, in every response, as the reason the general-purpose claim is withdrawn. |
| "V1 vs V2 is a controlled decontamination ablation" must not appear | **PASS in HNXd and Yi1G** ("V2 − V1 is therefore not a decontamination ablation"; "nothing is attributable to decontamination in either direction"). **FAIL in jVGf.** jVGf labels the row "ProtSent-V2 35M (decontaminated retrain)", presents "One decontaminated, absolute, non-structural number does exist: GB1 … 0.7108 (V1) / 0.7806 (V2)", and **never** tells that reviewer that V2 also changed batch size, sampling and hard negatives. A jVGf-only reader will read the V1→V2 jump as a decontamination effect. The caveat that exists twice elsewhere is missing exactly where it is needed. |
| Every number carries metric, split, model | **FAIL, repeatedly.** Itemised below. |
| No 150M-on-decontaminated implied | **PASS.** Stated explicitly in HNXd and jVGf. |

---

## 1. Numbers I cannot interpret from the sentence that uses them

### 1.1 The BIOMAP triple — three numbers, no model, no source, no split
> "*BIOMAP `stability_prediction` labels are continuous floats and our suite scores the task by Spearman, so our "58.8%" is a correlation ×100, not an accuracy commensurate with 69.08% linear / 77.69% LoRA*"

To follow this I would have to already know: (a) where "58.8%" appears — it is **not in the submitted paper**; Table 2 gives Stability (Biomap) Spearman .568 → .547 for 35M; (b) whose numbers 69.08% and 77.69% are — no model, no paper, no dataset version is named; presumably a published BIOMAP leaderboard the reviewer cited, but the response does not say so; (c) which model 58.8% belongs to (V1? the 150M?). The correction is admirable and I believe it, but I am being asked to verify a metric mismatch using three numbers I cannot locate.

### 1.2 "58.8%" vs 0.6435 vs .568 — the same task, three values, no reconciliation
> "*3-NN scores higher than our linear probe there for every arm (ESM-2 35M Spearman 0.6435 vs 0.4395)*"

Stock ESM-2 35M on Biomap stability is **.568** in the paper's Table 2 (3-NN, Spearman) and **0.6435** here (3-NN, Spearman). Same model, same task, same probe, same metric, two different numbers. The only offered explanation, buried in Yi1G, is "everything here is `--eval_split test`". That should be attached to the number, not to a closing errata paragraph.

### 1.3 "+0.0079" has no referent anywhere in the rebuttal
> HNXd: "*including the +0.0079 V1→V2 remote-homology gap*" — Yi1G: "*+0.0079 is inside item 6's checkpoint spread anyway*"

From the numbers given: 3-NN accuracy 0.6587→0.6668 = **+0.0081**; linear accuracy 0.6899→0.7016 = **+0.0117**; linear macro-F1 0.4281→0.4527 = **+0.0246**. **None is +0.0079.** A number used twice, in two responses, as the load-bearing "this delta is not resolved" quantity, and I cannot derive it from any pair of numbers in the text.

### 1.4 Metal-ion binding at N=1000 — no metric
> "*Metal-ion binding at N=1000 under the linear head: ESM-2 0.666±0.001 beats V1 0.637±0.004 and V2 0.595±0.001.*"

Metal ion binding is AUC in the paper. The sentence names the probe and the N but not the metric. 0.666 could be AUC or accuracy; the two would license different conclusions about how bad V2 is.

### 1.5 MMseqs2 at low sensitivity — no query set
> "*at `-s 5.7` the same baseline gives SCOPe R@1 0.3847*"

Every other SCOPe number in the rebuttal is over the **1,693 eligible** queries. Is 0.3847? Or over all 2,207? The two differ by a factor of 0.7671 and the response has just spent a paragraph insisting that distinction matters. Worse: 0.3847 is within noise of the paper's Table 3 baseline ESM-2 35M R@1 = 0.385, inviting exactly the confusion the paragraph is trying to prevent.

### 1.6 "691 of 2,207" against a table of 1,693
> "*691 of 2,207 queries return no phmmer hit at `-E 10` and score 0 at every K*"

The table denominator is 1,693. How many of the 691 no-hit queries are *among the 1,693 eligible ones*? That is the only version of the number that bears on the table, and it is not given. As written, a coverage statistic on one denominator is offered as an explanation of a margin computed on another.

### 1.7 MAP is never defined, and is not obviously commensurate across methods
MAP appears in every retrieval table and in the headline claim ("+0.1708 MAP vs HMMER"). Average precision over **what candidate list**? HMMER and MMseqs2 are capped at **top 300** with a reporting threshold; the embeddings rank the full 2,206-item gallery. MAP over a truncated, thresholded list versus MAP over a complete ranking is not the same quantity. The response flags the truncation issue for Recall depth ("the depth margin is an upper bound") but not for MAP, which is the larger of the two claimed margins.

### 1.8 "both metrics under both probes" = four cells, three reported
> "*only V2 improving on both metrics under both probes*" (Yi1G item 1; same sentence in HNXd)

Reported: 3-NN accuracy, linear accuracy, linear macro-F1. The **3-NN macro-F1** cell is missing. The claim is over four cells; I can check three.

### 1.9 Ablation percentages with no base and no arm
> "*removing AFDB drops the mean relative gain from +6.7% to +3.2%*", "*without Pfam the model still improves 15/23 (mean +4.6%)*", "*fluorescence +15.6% → +10.4%*", "*STRING removal takes PPI +5.3% → −0.5%*"

These are checkable against the paper's Tables 4 and 7, so they are not opaque — but every one is a **mean of relative percentages across metrics with different bases** (AUC near 0.95 and Spearman near 0.24 in the same average), single-run, on the paper's default split, from V1. The rebuttal condemns exactly this convention when discarding Table 5 ("*Its relative cells were uninterpretable — unbounded over near-zero Spearman baselines*") and then leans on it two responses later to establish "*Structural supervision is the largest single contributor*". The self-limitation paragraph in jVGf acknowledges the split and the single run, but not that the statistic itself is the one just withdrawn.

### 1.10 Layer sweep: "two tasks", one named
> "*a per-layer linear sweep (subsampled 8,000 train / 3,000 test …) shows that is the worst layer for remote homology in both models — ESM-2 0.6373 there vs 0.6703 at layer 6; V2 0.6803 vs 0.7033 at layer 8 … Two tasks, one scale*"

Which two tasks? Only remote homology is named. What metric are 0.6373/0.6703 — linear accuracy, presumably, but not stated. And ESM-2's full-data linear accuracy on this task was given four paragraphs earlier as 0.6868, above **both** sweep numbers, which is confusing until you notice the subsample disclaimer.

### 1.11 "checkpoint 4,000 … the last cosine trough at step 4,208"
Undefined: what a "cosine trough" is (learning-rate schedule? training cosine similarity?), how many total steps V2 trained for, and — critically — **in which direction** checkpoint 4,000 differs by "0.005–0.008 on every structural metric". Better or worse? Which metrics constitute "every structural metric"? The whole "we resolve no sub-0.01 delta" concession rests on a number I cannot situate.

---

## 2. Comparisons between numbers that are not commensurate

1. **Everything in the rebuttal is on `--eval_split test`; everything in the paper is not.** Yi1G says so in its last paragraph: "*everything here is `--eval_split test`, so not cell-comparable to the submitted tables*". The consequence is not drawn: **no number in this rebuttal can be compared to any number in the submission**, yet all three responses mix them freely (test-split absolutes in one sentence, +40.5% / +5.3% / +6.7% default-split relatives in the next).
2. **GB1 is the sharpest instance.** jVGf: "*GB1 variant effect (Spearman, 3-NN, test split, mean over 5 seeds) is 0.6582 (ESM-2 35M) / 0.7108 (V1) / 0.7806 (V2)*". The paper's Table 2 for the *same* model, task, probe and metric: baseline .656 → ProtSent **.651**, a **−0.8% decrease**. The rebuttal's V1 is 0.7108, a large increase. Same V1 checkpoint, opposite sign of effect, and the response presents 0.7806 as its one clean non-structural win without ever noting that its own V1 number contradicts the submitted table.
3. **SCOPe changed level and gallery.** Paper: superfamily, "full validation set (100,000 proteins)". Rebuttal: **family**, 2,207 domains, 1,693 eligible. The errata paragraph mentions this, but never performs the reconciliation, which I had to do myself: 0.4991 × 0.7671 = 0.383 ≈ the paper's 0.385, and 0.5854 × 0.7671 = 0.449 ≈ the paper's 0.445. That arithmetic should be in the response. Left as-is, a reviewer sees ESM-2 R@1 of 0.385 in the paper and 0.4991 in the rebuttal and reasonably suspects the baseline was moved.
4. **Nobody mentions that the surviving claim is now at the level closest to the training objective.** Retrieval moved from **superfamily** to **family**. The model's largest training source is **Pfam family co-membership**, and its second is Foldseek cluster co-membership. Evaluating family-level retrieval is the most favourable choice available, and the change is presented as a correction of a reporting error rather than as a choice needing defence.
5. **Remote homology is scored by macro-F1 in one place and macro-AUC in another.** HNXd: "*the paper's '+40.5%' there is a relative macro-F1 change (.223 → .313)*". Yi1G: "*its pooled macro AUC is not comparable to published per-holdout accuracies*". HNXd also says the task was **excluded** from the 20-task tally because "*one-vs-rest AUC is undefined*" — while the paper reports it as F1-macro. Three metric identities for one task inside one rebuttal.
6. **"3 of 23 under 3-NN and 6 under a linear probe"** (jVGf, alignment beating embeddings) uses denominator 23, while the probe win/tie/lose record uses denominator 20. Two denominators, same comparison family, no bridge.
7. **Three bootstrap comparisons, no multiplicity adjustment.** V2 − MMseqs2 at R@1 is +0.0289 **[+0.0035, +0.0544]** — an interval that clears zero by 0.0035 across three simultaneous method comparisons on three metrics. "*V2 … passes only the weaker tool*" is the one place a positive claim rests on a marginal interval, and it is uncorrected.

---

## 3. "23 tasks" means two different sets in the same rebuttal

- The paper's 23 (Section 4.2) = 8 binary (**including PPI**) + 5 multiclass + 10 regression. It contains **no** GO-MF and **no** SCOPe-40 retrieval; SCOPe-40 is a separate evaluation (Section 4.3) and GO-MF/CAFA5 appear only as two extra *multilabel* rows in appendix Table 7.
- The rebuttal's 23 **includes** GO-MF and SCOPe-40 retrieval — HNXd: "*Three inside the 20 use a built-in evaluator that ignores the probe flag … EC …, GO-MF …, and SCOPe-40 retrieval, a win*" — and **excludes** PPI — Yi1G: "*`ppi_bernett` … is not in the 23-task sweep*".
- Meanwhile jVGf's ablation argument quotes "16/23", "+6.7%", "PPI +5.3% → −0.5%" from the paper's Table 4/7, i.e. from the **other** 23 — the one that contains PPI.

So the headline counts (16/23, 20/23, 13/23, "3 of 23", "6 under a linear probe", 20 comparable tasks) are drawn from at least two different task sets, and no response says which is which. This is the single defect that most degrades my ability to weigh the evidence, because every summary statistic in the rebuttal is a count over an unspecified set.

---

## 4. Verification claims I cannot check, or that do not check out

### 4.1 The decontamination row counts do not sum to the stated total
> "*Verified on the parquet files training actually opened, by semi-join with the removal lists: **0 flagged sequences survived**, row counts summing to the 169,231,379 in the training log.*"

Using the response's own post-filter numbers: Pfam 27,929,772 + AFDB 126,301,607 = **154,231,379**; adding STRING 71,891,417 gives **226,122,796**. Neither equals **169,231,379**. (The Pfam+AFDB sum differs from the stated total by exactly 15,000,000, which looks like a cap or a sampled subset — but nothing in the response says so.) This is the sentence that is supposed to close the leakage objection, and its arithmetic does not close.

Separately, the paper states "Max training pairs 70M" and "a single epoch over 70M generated pairs". A training log of 169M rows is inconsistent with the submitted description, unexplained.

### 4.2 The pre-filter corpus sizes do not match the submitted paper
| Corpus | Paper (Table 1 / Appendix 12) | Rebuttal "before" | Rebuttal "after" |
|---|---|---|---|
| Pfam | 32,943,498 | 28,530,684 | 27,929,772 |
| AFDB | 133,856,004 | 135,404,259 | 126,301,607 |
| STRING | 36,502,692 pairs | 76,070,154 pairs | 71,891,417 pairs |

STRING is off by a factor of ~2 (both orientations of each pair?), Pfam by 4.4M, AFDB by 1.5M. Not one of these three discrepancies is mentioned. I am asked to accept a decontamination audit whose inputs do not match the corpus the paper describes.

### 4.3 The STRING decontamination erratum makes V1's status unclear
> Yi1G errata: "*The PPI decontamination description does not match the code (`easy-search` at 40% identity removing hit query IDs, not `easy-linclust` at 50% with cluster removal)*"

If V1's code already ran `easy-search` at 40% identity against the Bernett test set, then the V2 STRING filter (`easy-search`, 40%/80%, same target) is the **same filter run twice**, yet it removes a further 4,178,737 pairs from a corpus that is itself twice the size the paper reports. I cannot reconstruct what was actually filtered when.

### 4.4 Claims that rest entirely on the authors' report of unseen code and data
- "*The released code writes `(sentence_0, sentence_1, score)` rows*" (jVGf §4) and "*Rows are (wild-type, mutant, within-assay normalized fitness in [0,1])*" (Yi1G §2). The submitted Appendix 12.5 partly corroborates this (2,175,734 pairs over 3,576 wild-type targets, "*optional mutant–mutant intra-assay pairing is disabled*"), so I am inclined to believe it — but the argument that "*it does not flatten an assay*" is a claim about loss behaviour in code I cannot read.
- "*Peptide-HLA is not a two-input task here — the dataset supplies one `seq` field, a pipe-joined `HLA_pseudoseq|peptide` string*".
- "*Negative controls (1,000 random sequences per filtered corpus, re-searched; AFDB's exhaustively, its k-mer prefilter being only 89.4% recall) return 0 hits*" — where 89.4% comes from is unstated. The rule-of-three bound (~0.3%) is correctly and honestly stated.
- "*beta-lactamase Spearman 0.8026, above every embedding arm including our retrained model*" — V2's beta-lactamase number is **not given**. I must take on faith that 0.8026 beats it.
- "*EC classification F1-macro 0.710 (MMseqs2) vs 0.598 (ESM-2 35M) and 0.562 (V1)*" and "*GO-MF 0.585 vs 0.459 / 0.443*" — **V2's values are absent from both**, in a rebuttal whose stated position is that only V2's numbers are defended.

### 4.5 The one number that decides the leakage question is the one I must trust hardest
> "*median maximum identity to our corpus **0.908**, none below 20%*"

Half the SCOPe-40 evaluation set has a ≥90%-identical sequence in the training corpus. The entire defence of the surviving claim is the flat identity-gain stratification — and the authors themselves say that stratification cannot see fold-level label overlap, which is precisely the leakage their Foldseek/Pfam supervision would create. I accept the honesty; I cannot accept it as a control.

---

## 5. Selective presentation between responses (each response is read standalone)

Three reviewers each get a different subset of the same evidence:

1. **The middle identity bin.** jVGf gets all three bins (+0.1524 / **+0.1810** / +0.1565). HNXd gets only the two endpoints (+0.1524, +0.1565) under the word "**flat**". The omitted middle bin is the highest of the three, i.e. the stratification is not monotone and "flat" is doing work only the endpoints support.
2. **The negative Spearman.** Only Yi1G is told "*Identity-gain Spearman is −0.116 (average precision, p < 3e-6)*" — a *significant* relationship, on a different gain metric (average precision) from the R@10 bins shown to the other two. HNXd and jVGf are shown only the non-significant +0.038 restricted to 404 queries. What "*a headroom control (partial −0.081)*" is, is not explained to anyone.
3. **The benchmark-selection channel.** Only Yi1G is told "*those ablations were scored on these same benchmarks, so V2's configuration was chosen with benchmark results in view … V2's numbers are therefore not a clean held-out measurement*". HNXd and jVGf lean on V2's numbers throughout without it. This is the most important caveat in the whole rebuttal and two of three reviewers never see it.
4. **V1 beats V2 in every few-shot cell shown** (HNXd's table: N=1000 remote homology V1 0.318/0.377 vs V2 0.289/0.355; N=50 kNN V1 0.055 vs V2 0.045; metal-ion N=1000 V1 0.637 vs V2 0.595 — V1 bolded). No response comments on this, and jVGf, which is invited to read V1→V2 as improvement, is not shown it at all.

---

## 6. Terms, flags and identifiers that mean nothing without the repository

`--eval_split test` · `easy-search` / `easy-linclust` semantics as *this* pipeline uses them ·
`ppi_bernett` · `remote_homology` · `antibiotic_resistance` · `temperature_stability` ·
`thermostability` · BIOMAP `stability_prediction` · "the suite's default split" ·
"our suite" · "a built-in evaluator that ignores the probe flag" · "the evaluator's
`max_samples` cap" · "the training log" · "checkpoint 4,000" / "the last cosine trough at
step 4,208" · "our own ablations favour" (which settings, exactly? Yi1G implies no-hard-negatives
+ proportional sampling; HNXd and jVGf never say) · "structural metric" (the set is never
enumerated) · "headroom control" · "the 23-task sweep".

Also unglossed: **"the probe is scikit-learn defaults, untuned"** — which estimator? `LogisticRegression`
(default `C=1`, `max_iter=100`, which frequently fails to converge on 457 classes and 480-d inputs)?
`RidgeClassifier`? `LinearRegression` for the Spearman tasks? The linear-probe record is the evidence
for the single biggest withdrawal in the rebuttal, and I do not know what model produced it.

Minor: **"section 2 shows each type moving a different task family"** (jVGf) — "section 2" of the
*response*, but the paper's Section 2 is Related Work; ambiguous. Yi1G's "item 3 / item 6 / item 7 /
item 8" cross-references are followable but assume the reader keeps the numbering in view.
jVGf's "*Those four inject structure…*" follows a heading naming **five** systems (ESM-S, S-PLM, ISM,
Magneton, ProTrek); "four" excludes ProTrek, derivable only from the next sentence.
None of ESM-S, S-PLM, ISM or Magneton is described beyond "distilling a structure encoder", so their
characterisation is unverifiable from the paper.

Formatting: in HNXd's remote-homology line, "*linear macro-F1 0.4414 / **0.4281** / 0.4527*", the bold
marks V1's **worst** value, whereas bold in every table above marks the **best**. Two conventions,
no key.

---

## 7. A silent reversal of the paper's own ablation conclusion

The paper (Section 5.3): "*Removing Pfam family pairs causes the largest degradation … This confirms that Pfam provides the dominant contrastive signal.*"
The rebuttal (jVGf §2): "*Structural supervision is the largest single contributor.*"

Both are read off the **same** Table 4 (w/o Pfam: 15/23, +4.6%; w/o AFDB: 13/23, +3.2%). By mean delta the rebuttal is right and the paper was wrong — but the reversal is presented as a new measurement rather than as a correction of a stated claim in the submission, and I only caught it by re-reading Section 5.3.

---

## 8. What I now believe was actually run

Stated as beliefs, with confidence, from the rebuttal text alone.

**High confidence (specified well enough to act on):**

1. **A SCOPe-40 retrieval benchmark was re-run** at the **family** level over a **2,207-domain** gallery, leave-one-out, self excluded, no-hit scored as failure, restricted to the **1,693** queries having a non-self same-family neighbour. Five arms: MMseqs2 (`-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`), HMMER phmmer (`-E 10`, top 300), stock ESM-2 35M, ProtSent-V1 35M, ProtSent-V2 35M. Numbers as tabulated (HMMER R@1 0.6970 best; V2 R@1 0.6852, R@10 0.9220, R@30 0.9634, MAP 0.6459 best at depth).
2. **A paired bootstrap** (10,000 resamples over the 1,693 queries, no refitting) on R@1 / R@10 / MAP. Result: **V2 ties HMMER at R@1** (−0.0124, CI spans zero), edges MMseqs2 (+0.0289, CI barely clears zero), and is decisively ahead at depth (V2−HMMER R@10 +0.1412, MAP +0.1708; V2−MMseqs2 +0.1819 / +0.2356). **V1 loses to both at R@1** (−0.1110 vs HMMER, −0.0697 vs MMseqs2). This is the most solid thing in the rebuttal and it is reported against the authors' own interest.
3. **A corpus decontamination pass** — MMseqs2 `easy-search`, corpus as query, test set as target, 40% identity / 80% coverage, drop any corpus sequence with a hit — was applied to Pfam and AFDB against the `remote_homology` test set (3,244 seqs) and to STRING against `ppi_bernett` test (3,022 seqs). **Only those two test sets were filter targets; SCOPe-40 was not.** A 35M model (**V2**) was retrained from scratch on the filtered corpora.
4. **V2 differs from V1 in at least four ways at once** — decontaminated corpora, no synthetic hard negatives, proportional instead of round-robin sampling, and a true 1,024-example contrastive batch instead of the 64 that MNRL actually saw under gradient accumulation. **No unfiltered retrain at the V2 recipe exists**, so V1→V2 attributes nothing to decontamination.
5. **A real, conceded bug in the submission:** the reported 1,024 "effective batch" was an optimizer batch from gradient accumulation; MNRL's in-batch negatives were **64** at 35M and **16** at 150M. The authors withdraw all 150M results (including the abstract's +105% and +19.9%) on this basis. Eq. 1 as printed is wrong.
6. **A probe comparison over 20 tasks** (3 dropped for undefined one-vs-rest AUC), test split, frozen mean-pooled embeddings, single seed 42, ±0.005 tie band, vs stock ESM-2 35M: under 3-NN V1 11/3/6 and V2 10/3/7; **under a linear probe V1 4/4/12 and V2 2/7/11**. The general-purpose claim is withdrawn on this evidence. I believe this because it is against them.
7. **Remote homology (pooled TAPE holdouts, 718+1,254+1,272 = 3,244, 457 classes, test split):** 3-NN accuracy 0.5835 / 0.6587 / 0.6668 and linear accuracy 0.6868 / 0.6899 / 0.7016 and linear macro-F1 0.4414 / 0.4281 / 0.4527 for ESM-2 / V1 / V2. **V1 is below the stock backbone on linear macro-F1.**
8. **Table 5 (few-shot) is withdrawn and replaced** by absolute means ± SD over 5 training-subset draws, full-size test split, both probes on the same subset. Two conclusions against the authors: **a linear head beats 3-NN nearly everywhere including N=50**, so the label-scarcity claim is dead; and at small N the seed SD is the size of the effect.
9. **MMseqs2 was run as a full label-transfer pipeline** across the task suite and **beats the best embedding arm outright** on 3 tasks under 3-NN (EC, GO-MF, beta-lactamase) and 6 under a linear probe; and is **below chance** on DeepSol solubility (AUC 0.4185).
10. **Not run, and admitted:** ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, Redl et al. 2023; the joint no-AFDB/no-Pfam ablation; the fold-exclusion control on SCOPe; any fine-tuning sweep; any second training seed; any post-decontamination PPI number; any 150M model on decontaminated data.

**Believed but not reconstructible — I would have to take the authors' word:**

11. That the filtered parquets contain **zero** flagged sequences (§4.1: the row counts given do not sum to the stated training-log total).
12. That the corpora being filtered are the corpora the paper describes (§4.2: three size mismatches).
13. That V2's beta-lactamase, EC and GO-MF numbers are worse than MMseqs2's — the values are never shown.
14. That the DMS/CoSENT loss is WT-anchored pairwise-ordinal as described (partly corroborated by the submitted Appendix 12.5).
15. That the V2 checkpoint's ±0.005–0.008 checkpoint-to-checkpoint spread is symmetric/unbiased rather than favourable.

**Cannot reconstruct at all:**

16. **Which 23 tasks** any given count refers to (§3).
17. **What "+0.0079" is** (§1.3).
18. **What the linear probe estimator is** (§6) — and therefore how much weight the central withdrawal deserves.
19. **How the rebuttal's V1 numbers relate to the submitted V1 numbers** on any task — GB1 .651 vs 0.7108 and stability .568 vs 0.6435 are the same checkpoint under the same probe and metric, and the only offered explanation is a split change stated once, in a different response, in a closing paragraph.
20. **What MAP is computed over** for the alignment tools (§1.7), which is the metric carrying the largest claimed margin.

---

## 9. Bottom line for the score

The rebuttal does the hard thing: it withdraws the abstract's headline numbers, withdraws
general-purpose superiority, withdraws Table 5, withdraws all 150M results, concedes the
batch-semantics bug, concedes that V2's configuration was tuned on the evaluation
benchmarks, and reports a top-1 tie with HMMER instead of a win. On the two questions I
actually raised — leakage and baselines — the answers are real work, and the SCOPe-40
bootstrap is a genuine, well-specified result.

What it does not do is let me **check** any of it. Three responses, three different subsets
of the evidence; two different task sets both called "23"; every rebuttal number on a split
that is not the paper's, with paper numbers quoted alongside; a verification total whose
addends do not sum to it; corpus sizes that do not match Table 1; a repeatedly-cited
+0.0079 with no derivable source; and the surviving headline claim now measured at the
retrieval level nearest the training objective, on the one benchmark that was never
decontaminated, against a corpus with median 0.908 maximum identity to the query set.

Concretely, to move me the authors need, in text, in the discussion phase:
(a) one table, one split, one task list, with V2 present in **every** row where V1 or ESM-2 appears;
(b) the corpus row-count arithmetic that actually sums;
(c) the source of "+0.0079" and of "58.8% / 69.08% / 77.69%";
(d) the linear-probe estimator and its hyperparameters;
(e) the fold-exclusion control on SCOPe-40, or an explicit statement that the surviving
claim is conditional on fold-level overlap being benign.
