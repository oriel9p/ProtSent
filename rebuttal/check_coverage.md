# Coverage check of `FINAL_rebuttal.md` against the three verbatim reviews

Character budget as measured (content between the BEGIN/END markers, stripped):

| response | chars | headroom to 10,000 |
|---|---:|---:|
| HNXd | 9,874 | 126 |
| jVGf | 7,821 | 2,179 |
| Yi1G | 9,943 | 57 |

All three are under the limit already. jVGf is the only response with real room.

---

## TOP-LINE: three defects that outrank everything else in this file

**D1. The HNXd response says "We have no multi-seed results" — that is false, and
it is one of HNXd's five explicit score-raising requests.**
`NEW_EVIDENCE.md` §4c has a completed 5-seed sweep (seeds 0-4 x 8 tasks x 3 arms,
3-NN, test split), median SD 0.0000 across 24 rows, thermostability the only task
with visible spread (SD 0.013-0.017), and the V1->V2 remote-homology gap of +0.0079
at ~40x the seed SD on that task. The artifact is committed
(`results/benchmarks/seeds/seed_variability.json` at commit 84c061a; note the
working-tree copies are currently deleted — `git checkout HEAD~1 --
results/benchmarks/seeds/` restores them). The rebuttal currently answers HNXd's
question 4 with a withdrawal when it can answer it with a table. This is the single
highest-value fix in the document.

**D2. Both the jVGf and Yi1G responses print "Not run: ProtTucker, Foldseek,
HMMER, PLMSearch, DHR, ProTrek." HMMER *was* run.** `NEW_EVIDENCE.md` §3 reports
phmmer on SCOPe-40 (eligible-query R@1 0.6970, R@10 0.7809, R@30 0.7980, MAP
0.4747; 691 of 2,207 queries return no hit) plus a paired bootstrap against it, and
`results/benchmarks/hmmer_baseline.json` holds 21 further task rows. Yi1G named
"HMMER/MMseqs2" by name in weakness 7. Telling a Confidence-4 reviewer that the
baseline he asked for was not run, when it was run and is in the repo, is the worst
available outcome for that item.

**D3. The HNXd response violates the top-1 honesty constraint in one sentence.**
Section 3: "V2's top-1 lead **over alignment** is +0.0289 with a lower bound of
+0.0035 — resolved". Against the stronger alignment baseline V2 - HMMER at R@1 is
-0.0124 [-0.0372, +0.0124], i.e. **unresolved**, and V1 - HMMER is -0.1110
[-0.1388, -0.0827], a clear loss. Section 1 has the same problem in weaker form
("a tuned MMseqs2 beats the submitted model at top-1 and only V2 passes it"). The
supportable sentence is: alignment remains the better or equal top-1 method — V2
ties HMMER and beats MMseqs2 — and the embedding advantage is at ranking depth.
jVGf's summary sentence ("alignment wins single-best-hit ... the embedding wins
ranking depth") is already correct; HNXd's and Yi1G §8's framings are not.

---

## Reviewer HNXd — coverage matrix

Score-raising items are rows H1-H5.

| # | point (quoted/tight paraphrase) | addressed? | where | convincing? |
|---|---|---|---|---|
| H1a | "provide a retrieval and clustering evaluation" | **yes** (retrieval) | §1: SCOPe-40 family, 2,207 gallery, R@1/10/30/MAP for 4 arms, ceiling 0.7671 stated | Yes. Strongest part of the response. Weakened only by D3's framing. |
| H1b | "or ... an extensive analysis of how the embedding space changes after contrastive training" (silhouette, NMI, distance-vs-property correlation) | **partial**, explicitly disclaimed | §1 first line: "We did not compute clustering-geometry statistics" | Honest but thin. The per-query identity-stratified gain analysis (`NEW_EVIDENCE` §4/§4b) *is* a local-organisation analysis and it sits in the **Yi1G** response instead. HNXd's preamble points to "the per-query analysis in section 3" — but §3 is the CI section. Broken cross-reference to evidence that is not in this response. |
| H2a | "What would a simple linear classifier on top of the base model achieve?" | **yes** | preamble table + §2: 20 tasks, V1 4/4/12 median -0.0139, V2 2/7/11 median -0.0107, ±0.005 tie band, sklearn defaults named | Yes, and the concession is the credibility anchor of the whole rebuttal. |
| H2b | "What about fine-tuning the baseline?" | **no**, disclaimed | §2 end: "We did not run a fine-tuning sweep" | Unavoidable, correctly stated. |
| H2c | linear/fine-tune baselines **in the label-scarce setting** | **no**, claim withdrawn | §2 end: "no few-shot linear-probe baseline ... we withdraw it" | Honest. But this is 1 of 3 sub-parts of a score-raising question answered with a withdrawal; nothing in the response offers a partial substitute. |
| H2d | reported numbers "much lower than the literature ... likely due to the use of k-NN"; Stability 58.8% vs 69.08% linear / 77.69% LoRA | **partial** | §2 last paragraph: generic "frozen 35M under a 3-NN or linear probe, not a fine-tuned larger model" | **Weak, and the direct refutation is sitting unused.** `NEW_EVIDENCE` §4c: on Stability the 3-NN probe *beats* the linear probe for every arm (ESM-2 0.6435 3-NN vs 0.4395 linear), so the gap to the literature is **not** a k-NN artifact — which is exactly the causal claim HNXd made. Costs ~2 sentences. |
| H3a | "95% confidence intervals ... by bootstrapping over individual predictions" | **yes** for retrieval | §3: paired bootstrap, 10,000 resamples, 1,693 eligible queries, both marginal-denominator table and 4 paired-difference rows | Yes — and it answers the request in its literal form (resampling predictions, no refitting). |
| H3b | CIs for "the reported metrics" i.e. the 23-task table | **no**, disclaimed | §3 end: "We did not compute intervals for the 23-task table ... an omission, not an obstacle" | Honest, and the ±0.005 tie band partly substitutes. With D1 fixed, this row gets materially stronger (seed SD ≈ 0 means the *probe* is not the noise source). |
| H4 | "variability analysis with multiple random seeds for the few-shot evaluation" | **NO — wrongly disclaimed** | §4-5: "**We have no multi-seed results**, for training seeds or probe seeds" | **This is D1.** The available sweep is on the standard test split, not few-shot, so the honest version is "multi-seed variability, measured, on the standard evaluation; the few-shot table is withdrawn for the estimator reasons below" — still a direct, table-backed answer to a score-raising ask. As written it is a false negative. |
| H5 | "report the absolute scores [for Table 5] as well" | **partial** — table withdrawn instead | §4-5: withdrawal + explanation (relative change over near-zero Spearman; `n_neighbors = max(1, min(3, train_size))` silently changes the estimator) | Defensible: the diagnosis is better than the numbers would have been. But no absolute number from that table appears anywhere, so the literal request is unmet. Keep the withdrawal, and pair it with the seed table so §4-5 is not two withdrawals in a row. |
| H6 | "the paper mainly discusses the performance of the k-NN model rather than using k-NN as a probe of the embedding space" | **partial** | preamble reframes the claim to "family, fold and interaction relations become locally recoverable" | The reframing is right. But no measurement in this response is *about the space* rather than about probe accuracy — see H1b. |
| H7 | "improvements below 1% ... benchmarks known to be noisy" | **yes** | preamble (v) single seed, ±0.005 tie band, §3 | Yes. Strengthened by D1's fix. |
| H8 | "the paper currently sits between two narratives" | **yes** | preamble: one narrative kept (retrieval geometry), general-purpose claim withdrawn | Yes. This is the correct response to the review's central complaint. |

### Space HNXd spends on things HNXd did not ask about (~1,200-1,500 chars recoverable)

| item | chars (approx) | who asked |
|---|---:|---|
| "Errors we found in our own submission" — Eq. 1 malformed, `peptide_hla` pipe-joined strings, `thermostability` has no official test split, 150M / abstract +105% and +19.9% | ~600 of 1,106 | Eq. 1 and `peptide_hla` are **Yi1G's** items 3 and 4 (already answered there). The thermostability split is nobody's. Keep the SCOPe family/2,207 correction and the remote-homology-split correction: both bear on HNXd's own retrieval and level-of-numbers complaints. |
| Preamble caveat (iii), the built-in evaluator that ignores the probe flag | ~350 | Nobody. It exists to stop an overclaim; compress to one clause ("three tasks use a built-in evaluator, so their row is a single measurement — SCOPe is one measurement, not two"). |
| §1 MMseqs2 row + the R@10/R@30 coverage caveat paragraph | ~450 | **Yi1G** (baselines). The row is useful context; the caveat paragraph is duplicated verbatim in jVGf and Yi1G. |
| §3's three "cut against us" MMseqs2 comparisons | ~400 | **Yi1G** (baselines, statistics). HNXd never mentions baselines. |
| §4-5 checkpoint-control paragraph | ~350 | Nobody directly. It is a *substitute* for the seed data — once the real seed table is in, this shrinks to one clause. |

That is enough to fund the 8-row seed table (~800 chars), the Stability probe answer
(~250), and the D3 rewrite, without touching the linear-probe or CI sections.

---

## Reviewer jVGf — coverage matrix

Score-raising items are J1 and J2.

| # | point | addressed? | where | convincing? |
|---|---|---|---|---|
| J1 | "why ProtSent goes beyond adding structure information to sequence models" | **yes** | §2: AFDB removal drops mean gain +6.7%->+3.2%, 16/23->13/23, remote homology +40.5%->+15.3%; then source fingerprints — no-Pfam still 15/23 (+4.6%), no-STRING moves PPI +5.3%->-0.5%, no-DMS fluorescence +15.6%->+10.4% | Good argument, correctly ordered (concedes structure is the largest contributor *first*). Weakness: it rests entirely on single-run relative percentages from the submitted tables — the exact reporting style withdrawn elsewhere in the same document. The self-caveat is present and helps. |
| J1-alt | "or why ProtSent is superior to existing methods" | **no**, by design | §4 | Correctly declined — no matched runs exist. |
| J2 | "where this method sits in the generality-accuracy trade-off" | **yes** | §1: MMseqs2 over all 23 tasks, identical metric definitions; alignment wins outright on 3/23 under 3-NN and 6/23 under linear; EC F1-macro 0.710 vs 0.598/0.562; GO-MF 0.585 vs 0.459/0.443; beta-lactamase 0.8026 vs 0.7272/0.7676/**0.7153** (own model worst); MMseqs2 below chance on DeepSol 0.4185 | The best-executed section in the rebuttal. **Fixable gap:** the curve has only one alignment point on it. HMMER is the stronger point (SCOPe eligible R@1 0.6970 vs MMseqs2 0.6556) and is measured. Adding it makes the trade-off claim harder to attack, and jVGf's headroom is 2,179 chars. |
| J3 | "How do the results hold up in absence of both AFDB and Pfam?" | **no**, disclaimed | §2: "We did not run the joint no-AFDB/no-Pfam ablation you asked for" | Honest. This is a directly requested, cheap-looking ablation and its absence is the weakest point of the jVGf response. Nothing can be done about it now; keep the disclosure adjacent to the single-source ablation numbers so it does not read as evasion. |
| J4 | apply ProtSent to SaProt / ProSST | **no**, with a reason | §4: needs residue-level structure tokens for the full Pfam and STRING corpora (>100M sequences) | Reasonable and specific. Note jVGf pre-emptively suggested "may be easy to drop into existing code" — the answer correctly locates the cost at the *data* layer, not the model layer. Good. |
| J5 | compare to specialized methods; "ProTrek is also a good model to cite and possibly compare to" | **partial** | §4: "Not run: ... ProTrek", no excuse offered | The comparison is legitimately out of scope. But jVGf asked to **cite** ProTrek as the fallback, and the response never commits to citing it. One clause fixes this ("ProTrek is added to Related Work as the trimodal retrieval point on this curve"). Cheap, and it converts a flat refusal into a partial yes. |
| J6 | "How exactly does the CoSENT loss for DMS data work?" | **yes** | §3: `(sentence_0, sentence_1, score)` = (WT, mutant, within-assay normalized fitness in [0,1]; clinical benign 1.0 / pathogenic 0.0); ordinal over pairs; no absolute cosine target; WT-anchored limitation stated | Fully convincing, and it concedes the paper's text is wrong. Exactly the right shape of answer. |
| J7 | W1: position against ESM-S, S-PLM, ISM, Magneton | **partial / weak** | §4, one sentence: "belong with the structure-injection line; we position ProtSent as a different supervision graph, not a better one, with no matched runs" | **Weakest addressed item in this response.** jVGf's W1 is not a request for a run — it is a request for *positioning*, which costs only prose, and this response has 2,179 spare characters. One sentence for four cited papers reads as a brush-off of a reviewer who supplied the references. Two or three sentences distinguishing supervision *source* (structure distillation from a structure model vs. a heterogeneous relation graph over family/cluster/interaction/fitness) would land it. |
| J8 | minor: missing reference on line 21 | **yes** | §4 end: broken citation key, Heinzinger et al. 2022 (ProtTucker) and Redl et al. 2023 both in Related Work | Correct and complete. |

---

## Reviewer Yi1G — coverage matrix

| # | weakness | addressed? | where | convincing? |
|---|---|---|---|---|
| Y1a | leakage: AFDB training sequences not filtered against SCOPe test domains | **yes**, indirectly | §1: SCOPe cannot be decontaminated (no train/test split, median max identity 0.908, no query below 20%); memorization tested directly — V2 dR@10 +0.1524 at identity [0.2,0.4) (n=164) vs +0.1565 at [0.7,1.0] (n=1,214); Spearman -0.038 (R@10), -0.114/-0.116 (AP, p<3e-6); partial -0.083/-0.081 after headroom control; among the 404 queries the backbone fails completely, identity does not predict gain (+0.038, p=0.45) | Strong — the headroom control pre-empts the obvious rebuttal to the null. The limits paragraph ("fold-level overlap at 15% identity survives a 40% filter ... the right experiment we did not run") is the most credible passage in the document. |
| Y1b | PPI: "ensuring that test and training sequences share less than 50% or even 40% sequence identity" | **yes** on the filter | §1: STRING 76,070,154 -> 71,891,417 pairs at 40% id / 80% cov against `ppi_bernett` test (3,022 seqs); 0 hits on the negative control; 0 flagged rows survived in the training parquet | The literal request (a 40% filter) is met and verified. **Gap:** no post-decontamination **PPI benchmark result** appears anywhere in the rebuttal — the only "decontamination cost nothing" evidence is remote homology. Yi1G raised PPI as an equal partner to structure retrieval. There is no citable `ppi_bernett` number in `NEW_EVIDENCE.md`, so either it gets measured or the response should say plainly that the PPI *result* was not re-reported, rather than leaving the reader to notice. |
| Y2 | DMS objective should preserve fitness-induced ordering, not pull all high-fitness variants to WT | **yes** | §2: ordering objective is what is implemented; text is wrong; WT-anchored limitation stated | Fully convincing. Yi1G described the correct objective and is told he described the implementation. |
| Y3 | MNRL under-specified: does effective batch 1,024 contribute in-batch negatives?; Eq. 1 superscript | **yes** | §3: 1,024 is an optimizer batch via gradient accumulation, so each MNRL call saw **64** examples at 35M and **16** at 150M; retrain uses `CachedMultipleNegativesRankingLoss` with a true 1,024 batch; Eq. 1 numerator/denominator corrected | Excellent. Concedes a substantive methodological error and shows the retrain fixed it. Note this quietly makes the 150M results worse (16 negatives), which is consistent with not defending them. |
| Y4 | pair-level eval not reproducible (how are two embeddings combined?) | **yes** | §4: PPI concatenates the two embeddings; peptide-HLA is a single pipe-joined `HLA_pseudoseq\|peptide` string so no operator applies; neither was in the paper | Complete and specific. |
| Y5 | k-NN regression: uniform or distance-weighted? | **yes** | §5: `KNeighborsRegressor(n_neighbors=3, metric="minkowski")`, uniform; plus the `max(1, min(3, train_size))` estimator change at small N | Complete, and it converts a methods question into a reason to withdraw the few-shot table. |
| Y6 | ablations do not support the default design | **yes** | §6: no hard negatives 20/23 at +7.9% vs 16/23 at +6.7%; proportional +7.0% vs round-robin +6.7%; retrain uses neither default; discloses that ablations were scored on the same benchmarks and that V2's config was therefore chosen with benchmark results in view | Very strong. The volunteered disclosure about config selection is worth more than the ablation numbers. |
| Y7a | missing baselines: **HMMER**/MMseqs2 | **MMseqs2 yes; HMMER wrongly declared not run** | §7 | **This is D2.** MMseqs2 coverage is thorough. Declaring HMMER "not run" when `hmmer_baseline.json` exists is a direct, checkable misstatement on an item this reviewer named. |
| Y7b | missing baselines: ProtTucker, Foldseek, PLMSearch, DHR | **no**, disclaimed | §7: "No excuse offered; we claim no superiority to any" | The right way to decline. |
| Y7c | missing baseline: the prior work "Optimizing Protein Language Models with Sentence Transformers" | **NO — not mentioned at all** | — | **Unaddressed.** It is Redl et al. 2023, already in the paper's bibliography (`PAPER_text.txt` line 506) and named in the jVGf response as "discussed in Related Work". Yi1G asked for it as a *baseline*, not a citation. One sentence — cited, not run, and here is why it is not a matched comparison — closes an otherwise silently dropped item. Silence on a named item invites "the authors ignored weakness 7". |
| Y8 | statistical evidence weak; Table 2 improvements very small | **partial** | §8: SCOPe paired bootstrap; explicit "**no** intervals and no multi-seed results" for the 23-task table; win/tie/lose counts with ±0.005 band; general-purpose claim withdrawn | The "no multi-seed results" clause is **D1 again** — false, and here it is answering a reviewer whose complaint is precisely statistical weakness. The seed sweep (median SD 0.0000; thermostability the only mover) is the direct answer to "some improvements in Table 2 are very small": it shows the probe contributes ~no variance, so the residual uncertainty is benchmark composition, which *is* bootstrapped. Fixing D1 upgrades this row from partial to strong. |
| Y9 | Limitations: "the biological assumption of mapping heterogeneous protein relationships into a single embedding space" needs clearer discussion | **no** | — | Unaddressed. Yi1G lists it as one of four things the paper "should more explicitly address". The material exists — the jVGf response's source-fingerprint argument (each relation type moves a different task family; removing STRING moves PPI +5.3%->-0.5% and leaves others intact) is an *empirical* answer to whether one space can hold heterogeneous relations. One or two sentences. |
| Y10 | Limitations: single-run results | **partial** | §8 | Same as Y8/D1. |

### Space Yi1G spends on things Yi1G did not ask about

Little — the response maps 1:1 onto the eight weaknesses, which is correct. The two
compressible items:

| item | chars (approx) | note |
|---|---:|---|
| §8's re-explanation of the 20-of-23 exclusion and the built-in-evaluator plumbing | ~400 of 1,378 | Duplicated verbatim from the HNXd response. Yi1G asked about statistical strength, not probe plumbing; the win/tie/lose counts and the tie band are what he needs. Compress to one clause. |
| §1's row-arithmetic and prefilter-recall detail | ~350 of 3,693 | The verification claim ("0 flagged sequences survived") is load-bearing; the arithmetic that closes it to the training-log total is belt-and-braces. Trim if space is needed for Y7a/Y7c/Y9. |

§1 at 3,693 chars is 37% of the response, which is defensible — Yi1G called leakage
"the most serious concern". Do not cut the "what these controls cannot rule out"
paragraph; it is the reason the rest is believable.

---

## Everything flagged, in priority order

1. **D1 — HNXd §4-5 and Yi1G §8 falsely claim no multi-seed results.** Insert the
   `NEW_EVIDENCE` §4c table (8 tasks x 3 arms, mean ± SD over 5 seeds) with the
   caveat that it is probe-seed variability on the standard split, not few-shot, and
   not training-seed variance. Restore the deleted artifacts first
   (`git checkout HEAD~1 -- results/benchmarks/seeds/`).
2. **D2 — "Not run: ... HMMER" in jVGf §4 and Yi1G §7.** HMMER was run. Replace with
   the phmmer result and move it to the front of Yi1G §7, since he named it.
3. **D3 — HNXd §3 "V2's top-1 lead over alignment".** V2 - HMMER at R@1 is
   -0.0124 [-0.0372, +0.0124], unresolved. Restate as: ties the stronger alignment
   baseline at top-1, beats MMseqs2 there, wins decisively at depth. Yi1G §8's
   MMseqs2-only R@1 line needs the same treatment.
4. **Yi1G Y7c — Redl et al. 2023 never mentioned.** A named baseline dropped in
   silence.
5. **HNXd H2d — the Stability/literature gap.** The measured refutation (3-NN 0.6435
   beats linear 0.4395 on Stability for ESM-2, so the gap is not a probe artifact)
   is in `NEW_EVIDENCE` §4c and unused, while the response instead offers a generic
   explanation of the level difference.
6. **HNXd H1b — broken cross-reference.** The preamble cites "the per-query analysis
   in section 3"; §3 is the CI section. The per-query *organisation* analysis
   (identity-stratified gain, headroom control) is in the Yi1G response only, and it
   is the closest thing available to HNXd's embedding-space-organisation branch.
   Either point at it explicitly with two of its numbers, or drop the cross-reference.
7. **jVGf J7 — four cited papers answered in one sentence**, with 2,179 characters
   spare. Positioning costs prose, not compute.
8. **Yi1G Y9 — heterogeneous relations in one space** is listed in his Limitations
   and gets no reply; the jVGf fingerprint argument answers it empirically.
9. **jVGf J5 — commit to citing ProTrek.** He asked for a citation as the fallback
   to a comparison; the response gives neither.
10. **Yi1G Y1b — no post-decontamination PPI result appears anywhere.** Either
    measure `ppi_bernett` for V2 or say explicitly that the PPI *result* was not
    re-reported, so the reader is not left to infer it.
11. **Numeric inconsistency inside `NEW_EVIDENCE.md` itself.** §3's eligible-query
    table gives MMseqs2 R@10 0.7348 / R@30 0.7354 / MAP 0.4041 (what the rebuttal
    quotes), while the HMMER comparison table in the same section gives MMseqs2 R@10
    0.7401 / R@30 0.7566 / MAP 0.4098. Harmless today because only one set is used —
    but the moment HMMER is added (D2), both sets are in play and the pair must be
    reconciled before posting.
12. **HNXd's "Errors we found" carries ~600 chars of Yi1G's items** (Eq. 1,
    `peptide_hla`) plus the thermostability split nobody asked about. That is the
    budget for fix 1 and fix 5, with margin.
