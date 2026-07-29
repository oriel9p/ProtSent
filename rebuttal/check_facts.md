# Hostile fact-check of `rebuttal/FINAL_rebuttal.md`

Checked against `rebuttal/NEW_EVIDENCE.md`, `REBUTTAL_LEAKAGE.md`, `rebuttal/REVIEWS_actual.md`,
`rebuttal/PAPER_text.txt`, and the on-disk artifacts (`results/benchmarks/*.json`,
`comparison.json`, `protein_benchmark_suite.py`, `benchmark_tasks.py`, git history).

Character counts (body between the BEGIN/END markers, which is what gets posted):
**HNXd 9,874 · jVGf 7,821 · Yi1G 9,943.** All three are under 10,000. Yi1G has 57
characters of margin, HNXd has 126. Any edit below must be length-neutral or shorter.

Ranked by damage if a reviewer catches it.

---

## TIER 1 — response-destroying

### 1. The rebuttal tells two reviewers that HMMER was not run. HMMER was run, it is in the repo, and its result kills the top-1 claim.

**Quoted, jVGf §4:** "**Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** We offer no excuse for their absence; they are missing…"
**Quoted, Yi1G §7:** "**Not run: ProtTucker, Foldseek, HMMER, PLMSearch, DHR, ProTrek.** No excuse offered; we claim no superiority to any."

Classification: **WRONG**, and this is the single worst item on the page.

- `NEW_EVIDENCE.md` §3, "The top-1 claim, stated correctly — read this before writing anything about R@1": *"Yi1G named 'HMMER/MMseqs2' as missing baselines. **Both were run.** HMMER (phmmer, `hmmer_baseline.py`) is the stronger alignment baseline…"*
- Committed artifacts, commit `b97ab13` ("baselines: add HMMER, which retires the top-1 claim"): `hmmer_baseline.py`, `results/benchmarks/hmmer_scope40.json`, `results/benchmarks/alignment_paired_ci.json`.
- Measured, `hmmer_scope40.json`, eligible queries (n=1,693), same gallery, same scoring, 691 of 2,207 queries return no phmmer hit and are counted as failures:
  HMMER R@1 **0.6970**, R@10 0.7809, R@30 0.7980, MAP 0.4747 — vs V2 0.6852 / 0.9220 / 0.9634 / 0.6459.
- `alignment_paired_ci.json`: `ProtSent-V2 - HMMER / hit1` = **-0.0124 [-0.0372, +0.0124], `excludes_zero: false`**. `ProtSent-V1 - HMMER / hit1` = -0.1110 [-0.1388, -0.0827].

Yi1G is the reviewer who *asked for HMMER by name* (REVIEWS_actual.md line 104: "should include stronger… baselines, such as ProtTucker, HMMER/MMseqs2, Foldseek, PLMSearch, DHR…"). Telling that reviewer it was not run, when the released repo contains `hmmer_baseline.py` and `hmmer_scope40.json`, is not a slip a reviewer forgives — it reads as suppressing the one baseline that beats you at top-1.

**Fix:** report HMMER in both responses. The supportable sentence is already written for you in NEW_EVIDENCE §3: V2 is *statistically tied* with the best alignment baseline at top-1 (V2 − HMMER −0.0124 [−0.0372, +0.0124]) and ahead at depth against both tools (vs HMMER R@10 +0.1412 [+0.1205, +0.1618], MAP +0.1708 [+0.1511, +0.1905]). Remove HMMER from both "Not run" lists.

---

### 2. HNXd §3 claims a resolved top-1 win over "alignment". Against the best alignment baseline it is a tie.

**Quoted, HNXd §3:** "V2's top-1 lead over **alignment** is +0.0289 with a lower bound of +0.0035 — **resolved**, but small, and we present it as small."
**Also HNXd §1:** "a tuned MMseqs2 **beats the submitted model at top-1** (0.5029 vs 0.4490) and only V2 passes it."

Classification: **WRONG as generalised** (the +0.0289 [+0.0035, +0.0544] number itself is VERIFIED for V2 − *MMseqs2*, `scope40_bootstrap_ci.json`).

This is the exact sentence the honesty constraint forbids. NEW_EVIDENCE §3: *"**Do NOT write that ProtSent beats alignment at top-1.** It beats MMseqs2 there and ties HMMER, and a reviewer who runs the stronger baseline will find the tie."* HMMER's eligible R@1 is 0.6970 vs V2's 0.6852 — the point estimate is *against* V2.

**Fix:** name the tool. "V2's top-1 lead over MMseqs2 is +0.0289 [+0.0035, +0.0544]; against the stronger phmmer baseline the same comparison is a tie, −0.0124 [−0.0372, +0.0124]." Length-neutral if you drop "and we present it as small".

---

### 3. Cross-response contradiction on top-1: jVGf concedes it, HNXd claims it.

- **jVGf §1:** "That is the trade-off, measured rather than asserted: **alignment wins single-best-hit** and homology-transferable annotation; the embedding wins ranking depth."
- **HNXd §3:** "**V2's top-1 lead over alignment is +0.0289** … resolved."

Same measurement, opposite conclusions, in two responses posted publicly under the same submission. `REBUTTAL_LEAKAGE.md` §6 item 8 names this exact failure mode: *"Keep one story about R@1 across all three responses… Asserting different things to different reviewers is what must be avoided."*

With HMMER in the picture jVGf's version is the correct one and HNXd's must change. Yi1G §7 ("Alignment beats the submitted model at top-1") is silent on V2 and is safe either way.

---

### 4. The rebuttal concedes "no multi-seed results" to the reviewer who offered to raise his score for exactly that, after the sweep finished.

**Quoted, HNXd §4-5:** "**We have no multi-seed results**, for training seeds or probe seeds. So Table 5 is withdrawn…"
**Quoted, Yi1G §8:** "For the 23-task table we have **no** intervals and no multi-seed results."

Classification: **STALE / WRONG.**

`NEW_EVIDENCE.md` §4c, titled *"Seed variability — reviewer HNXd's second explicit request"*: five seeds (0–4) × 8 tasks × 3 arms, 3-NN probe, test split. Committed in `84c061a` as `results/benchmarks/seeds/seed_variability.json` plus three per-arm CSVs. Median SD across all 24 rows is 0.0000; Thermostability is the only task with visible spread (SD ~0.013–0.017); the V1→V2 remote-homology gap of +0.0079 is ~40× the seed SD on that task.

HNXd's review says "I would consider increasing my score if the authors provide these additional analyses" and lists the seed variability analysis as Q4. Writing "we have no multi-seed results" when they exist is throwing away the highest-value item on the page.

Two caveats to keep honest if you use it: (a) the sweep is a *probe/benchmark*-seed sweep, not training seeds — HNXd asked about the few-shot table specifically, so say "probe-seed" explicitly and keep "only one training run per model exists"; (b) **the working tree has deleted those four files** (`git status` shows 4 `D` entries under `results/benchmarks/seeds/`). They are still in `HEAD`. Restore them before the repo is re-released, or a reviewer who checks out the tip finds an empty `seeds/` directory next to a claim that cites it.

---

## TIER 2 — a reviewer with a calculator finds these

### 5. The eligible-query table and the paired-bootstrap table in the same response do not subtract to each other, and the parenthetical that waves it away is false.

**Quoted, HNXd §3:** "(Interval centres are bootstrap means and differ from the table's point estimates by **under 0.001**.)"

Classification: **WRONG.** The two tables come from two different code paths, which `REBUTTAL_LEAKAGE.md` §0a.7 explicitly warns against mixing: *"Both are correct for their own code path; **do not mix them in one table**."*

The eligible table uses `scope40_table.json` (MMseqs2 R@10 **0.7348**, R@30 **0.7354**, MAP **0.4041**). The paired deltas below it use `scope40_bootstrap_ci.json`, whose MMseqs2 marginals are R@10 **0.7401**, R@30 **0.7566**, MAP **0.4098**. Discrepancies: R@10 0.0053, MAP 0.0057, **R@30 0.0212** — all above the claimed 0.001.

What a reviewer sees, subtracting the printed table:

| stated paired delta | recomputed from the printed table | gap |
|---|---|---|
| V2 − MMseqs2 R@10 **+0.1819** | 0.9220 − 0.7348 = **+0.1872** | 0.0053 |
| V2 − MMseqs2 MAP **+0.2356** | 0.6459 − 0.4041 = **+0.2418** | 0.0062 |
| MMseqs2 − ESM-2 R@10 **−0.0213** | 0.7348 − 0.7614 = **−0.0266** | 0.0053 |
| MMseqs2 − ESM-2 MAP **−0.0125** | 0.4041 − 0.4210 = **−0.0169** | 0.0044 |
| V1 − ESM-2 MAP **+0.1289** | 0.5509 − 0.4210 = **+0.1299** | 0.0010 |

R@1 is clean throughout (MMseqs2 0.6556 in both paths; 0.6556 − 0.4991 = 0.1565 exactly as stated).

**jVGf §1 has the identical clash** — the same eligible-context numbers and the same paired deltas. Yi1G is safe: §7 prints the all-query table and §8 prints eligible-query paired deltas with the denominator named, so nothing invites the subtraction.

**Fix:** either use the bootstrap marginals in the eligible tables (0.6556 / 0.7401 / 0.7566 / 0.4098 for MMseqs2, 0.4222 for ESM-2 MAP, 0.5859 / 0.5511 for V1, 0.6846 / 0.6454 for V2), or delete the parenthetical and say the two tables come from two scoring implementations that agree on Recall@10 to four decimals and differ in the third decimal on AP.

### 6. HNXd is told the probe explains the gap to the literature. Your own evidence says it does not.

**Quoted, HNXd §2:** "That also explains the level difference you flagged against the literature: every number in our tables is a frozen 35M backbone under a 3-NN or linear probe, not a fine-tuned larger model…"

Classification: **contradicts NEW_EVIDENCE §4c**, which was written specifically to answer this: *"HNXd noted our Stability (Biomap) figure is far below the literature's 69.08% linear / 77.69% LoRA and suspected the 3-NN probe was the cause. **It is not**: on Stability the 3-NN probe scores higher than the linear probe for every arm (ESM-2 0.6435 3-NN vs 0.4395 linear). The gap to the published number is therefore **not a probe artifact, and we do not claim it is**."*

Confirmed in `comparison.json`: `stability` esm2_35m knn 0.64346, linear 0.43951.

As written, the rebuttal offers the probe as (part of) the explanation, which is the thing the evidence file forbids. It also passes up the strongest available answer to the reviewer's most concrete complaint. The frozen-vs-fine-tuned half of the sentence is fine; the "under a 3-NN or linear probe" half needs the measured rebuttal attached, or removal.

### 7. Selective disclosure across responses: only Yi1G is told V2's configuration was chosen with benchmark results in view, and only Yi1G is told V1-vs-V2 is not a controlled ablation.

- **Yi1G §6:** "Disclosure that follows: those ablations were scored on these same benchmark tasks, so **V2's configuration was chosen with benchmark results in view**."
- **Yi1G §1:** "**V1-vs-V2 is not a controlled decontamination ablation**; the supported claim is sufficient — decontamination cost nothing."
- **HNXd §1 and jVGf §1** both carry only the flattering half: "using the configuration favoured by the paper's own ablations — proportional sampling and no synthetic hard negatives".

Not a numeric error, but all three responses are visible to all three reviewers and to the AC. The asymmetry reads as telling the reviewer who found the problem, and not telling the ones who didn't. `REBUTTAL_LEAKAGE.md` §0a.2 item 9 makes this a durable constraint, not a Yi1G-specific concession. One clause in HNXd and jVGf ("the V1-vs-V2 comparison is not a controlled decontamination ablation") closes it.

### 8. jVGf leans on relative-percent ablation numbers that HNXd and Yi1G are told are the wrong instrument.

- **HNXd §2:** "the submitted paper's '+40.5%' for this task is a relative change in macro-F1 (.223 → .313) computed under the suite's default split… **Different metric, different split — they are not two views of one result and we do not mix them.**"
- **HNXd §4-5:** "Relative change over near-zero denominators, one run, varying k: **the wrong instrument**, and the claim it supported goes with it."
- **Yi1G §5:** "…makes the few-shot table uninterpretable; **its claims are withdrawn**."
- **jVGf §2:** "Removing AFDB drops the mean relative gain from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and the remote-homology gain **from +40.5% to +15.3%**."

Every one of those ablation figures is **VERIFIED** against `PAPER_text.txt` (Table 4 line 363–369: full 16 / +6.7; w/o Pfam 15 / +4.6; w/o hard negatives 20 / +7.9; w/o AFDB 13 / +3.2; proportional 16 / +7.0. Lines 347–351: AFDB remote homology +40.5→+15.3, EC −11.0; STRING PPI +5.3→−0.5. Line 390: DMS fluorescence +15.6→+10.4). jVGf does caveat them ("single-run relative-percent numbers from the submitted tables, with no intervals — the reporting we withdraw for sub-1% cells elsewhere in this rebuttal"). But an AC reading all three will see the +40.5% disowned in one response and load-bearing in another. The caveat sentence should name the asymmetry directly: *these are big-effect directions, not the sub-1% cells and not the near-zero-denominator cells we withdrew.*

### 9. Undisclosed against interest: V1's linear-probe remote-homology macro-F1 is *below* ESM-2.

**Quoted, HNXd §2:** "Under the linear probe **V1's +0.0031 is inside our own ±0.005 band and is a tie** — only V2 clears it."

The accuracy numbers are **VERIFIED** (0.6868 / 0.6899 / 0.7016, NEW_EVIDENCE §3; 0.6899 − 0.6868 = 0.0031). But NEW_EVIDENCE §3 flags the other metric: *"under the linear probe V1's macro-F1 (**0.4281**) is below ESM-2 (**0.4414**) — only V2 improves on both metrics under both probes."*

Presenting V1's linear remote homology as "a tie" while a second metric on the same task shows a regression is exactly the kind of thing that turns a rebuttal built on disclosure into one built on selection. It also costs you nothing: the sentence "only V2 improves on both accuracy and macro-F1 under both probes" is *stronger* for V2 than what is currently written.

### 10. Yi1G is left with MMseqs2's remote homology as an incomparable AUC, when the commensurate numbers exist and favour you.

**Quoted, Yi1G §7:** "Its remote-homology score, macro-OvR AUC 0.6523, is a different metric from item 1's accuracies."
**Quoted, jVGf §1:** "MMseqs2 on that task reaches macro-OvR AUC 0.6523 at 88.9% hit coverage, which is a different metric and is not comparable to those accuracies."

Both statements are **VERIFIED** and correct. But commit `b97ab13` fixed precisely this mismatch and produced the commensurate row (NEW_EVIDENCE §3): MMseqs2 kNN accuracy **0.4365**, macro-F1 **0.2064**, vs V2 **0.6668** / **0.4108**. Declining to state them leaves 0.6523 sitting next to 0.6668 looking near-parity, when the like-for-like comparison is 0.4365 vs 0.6668. Self-inflicted.

---

## TIER 3 — smaller, still findable

### 11. "flat across identity bins" is argued from the two endpoint bins; the middle bin is the largest.

**Quoted, Yi1G §1:** "Per-query Recall@10 gain over ESM-2 across the 1,693 eligible queries **is flat across identity bins** — V2 +0.1524 at [0.2, 0.4) (n=164) versus +0.1565 at [0.7, 1.0] (n=1,214)."

Both cited values are **VERIFIED** (NEW_EVIDENCE §4, REBUTTAL_LEAKAGE §5). The unquoted middle bin is **[0.4, 0.7), n=315, +0.1810** — the largest of the three. "Flat" is defensible (it is not monotone in identity, which is the point) but quoting only the endpoints when the omitted bin is the maximum is the kind of selection a reviewer with the JSON will name. The Spearman (−0.038) and the 404-query zero-baseline result (+0.038, p=0.45) already carry the argument; either add the middle bin or lead with the correlation.

Note in passing: `REBUTTAL_LEAKAGE.md` §2(a) asserts *"The gain is largest for the queries whose pretraining neighbour is most distant"* — that is true for AP but **false for Recall@10** for both V1 (+0.0915 / +0.1016 / +0.0865) and V2. The rebuttal correctly says "flat" instead. Do not let that sentence migrate in from the working doc.

### 12. ">100M sequences" for Pfam + STRING.

**Quoted, jVGf §4:** "both need residue-level structure tokens for the entire Pfam and STRING corpora (**>100M sequences**)."

Classification: **UNSUPPORTED / arithmetically wrong.** No such figure is in NEW_EVIDENCE or REBUTTAL_LEAKAGE (§8 of NEW_EVIDENCE says only "the full Pfam and STRING corpora", no number). From REBUTTAL_LEAKAGE §1: Pfam 28,530,684 rows; STRING **14,567,625 unique sequences** (76,070,154 *pairs*). Pfam + STRING ≈ **43M sequences**. You only clear 100M by counting STRING pair-rows as sequences, or by silently including AFDB (135M), which the sentence does not name. Say "tens of millions of sequences across Pfam and STRING, plus 135M in AFDB" or drop the parenthetical — it is decoration on an argument that stands without it.

### 13. "which are 3-40 points".

**Quoted, jVGf §2:** "We use them only for the direction and size of source-specific effects, **which are 3-40 points**, not for the small ones."

Classification: **UNSUPPORTED.** The source-specific effects actually cited in that paragraph span: Pfam mean delta 6.7 → 4.6 = **2.1 points**; DMS fluorescence 15.6 → 10.4 = 5.2; STRING PPI 5.3 → −0.5 = 5.8; AFDB remote homology 40.5 → 15.3 = **25.2 points**. Nothing in the paragraph is a 40-point effect (40.5% is a *level*, not a source-specific *delta*), and the smallest is 2.1, not 3. Write "2 to 25 points" or "single-digit to 25-point".

### 14. Yi1G's "Not run" list silently drops one baseline Yi1G named.

Yi1G asked for "ProtTucker, HMMER/MMseqs2, Foldseek, PLMSearch, DHR, **and the prior work 'Optimizing Protein Language Models with Sentence Transformers.'**" (REVIEWS_actual.md line 104). The response's list covers six items and omits the last. Since the whole rhetorical move is "we enumerate what is missing without excuse", a missing item in the enumeration undercuts it. One clause fixes it — but note both responses are at the character ceiling, so it has to come out of something else.

### 15. The checkpoint trough is at step 4,208, not 4,000.

**Quoted, HNXd §4-5:** "a near-trough checkpoint (**step 4,000, where the 3-cycle cosine schedule bottoms**)".

Classification: **WRONG in detail.** `REBUTTAL_LEAKAGE.md` §4: the schedule "troughs at steps 1,642 / 2,925 / **4,208**"; §0.6 describes checkpoint-4000 as "LR 5.5e-5; **nearest saved checkpoint** to the last cosine trough at step 4,208". So 4,000 is *near* the trough, not at it, and its LR is 5.5e-5 rather than the 1e-5 floor. The "differs by 0.005-0.008 on every structural metric" half is **VERIFIED** (§0a.2 item 8). Change "where … bottoms" to "the nearest saved snapshot to the last cosine trough".

### 16. `-s 5.7` R@1 0.3847 is cited but has no results file on disk.

**Quoted, jVGf §1:** "at the default `-s 5.7` the same baseline gives SCOPe R@1 **0.3847**, so any MMseqs2 number needs its sensitivity stated."

The value matches REBUTTAL_LEAKAGE §3, but §0.2 flags the artifact as **MISSING**: *"`mmseqs_baseline.py` was re-run at reduced sensitivity but the output was not persisted; the numbers survive only in §3 of this document… If it is ever quoted again, re-run and persist it."* Every other number in the rebuttal is reproducible from the released repo; this one is not. Either re-run and persist it (cheap — MMseqs2 search on SCOPe is ~3.4 s) or drop the figure and keep the qualitative point.

### 17. Four correct per-task numbers whose only source is outside the two permitted files.

All **VERIFIED against `results/benchmarks/comparison.json`**, none present in NEW_EVIDENCE.md or REBUTTAL_LEAKAGE.md, so they violate the letter of the "no number outside these two files" rule even though every digit is right:

| response | quoted | comparison.json |
|---|---|---|
| HNXd §2 | AAV 3-NN ESM-2 0.4667 / V1 0.5553 | 0.46665 / 0.55532 |
| HNXd §2 | AAV linear ESM-2 0.5639 / V1 0.4362 | 0.56395 / 0.43618 |
| jVGf §1 | beta-lactamase 3-NN MMseqs2 0.8026, ESM-2 0.7272, V1 0.7676, V2 0.7153 | 0.80260 / 0.72717 / 0.76763 / 0.71525 |
| jVGf §1, Yi1G §7 | EC 0.710 vs 0.598 / 0.562; GO-MF 0.585 vs 0.459 / 0.443 | 0.71025 / 0.59837 / 0.56166; 0.58495 / 0.45903 / 0.44342 |

No action needed on the numbers; add them to NEW_EVIDENCE.md so the sourcing rule stays true.

---

## Everything else checked and clean

Verified against the named source; no action needed.

**Retrieval / SCOPe.** All-query table (MMseqs2 0.5029 / 0.5637 / 0.5641 / 0.3100; ESM-2 0.3829 / 0.5840 / 0.6398 / 0.3230; V1 0.4490 / 0.6529 / 0.7100 / 0.4226; V2 0.5256 / 0.7073 / 0.7390 / 0.4955) — NEW_EVIDENCE §3, `scope40_table.json`, `mmseqs_baseline.json`. Eligible-query rows for ESM-2/V1/V2 — same sources (see finding 5 for the MMseqs2 row). Recall@K ceiling 0.7671 = 1,693/2,207 ✓. "V1's R@30 of 0.7100 is 92.6% of the attainable 0.7671" — 0.7100/0.7671 = 92.56% ✓. "MMseqs2's R@10 and R@30 differ by 0.0004" — 0.5641 − 0.5637 ✓. MMseqs2 flags `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3` ✓. Solubility AUC 0.4185 below chance ✓.

**Paired bootstrap.** Every interval quoted matches `scope40_bootstrap_ci.json` → `paired` to four decimals: V1−ESM-2 R@1 +0.0868 [+0.0614, +0.1122], R@10 +0.0898 [+0.0685, +0.1105], MAP +0.1289 [+0.1129, +0.1447]; V2−ESM-2 +0.1855 [+0.1618, +0.2097], +0.1607 [+0.1412, +0.1802], +0.2232 [+0.2082, +0.2383]; V2−V1 +0.0986 [+0.0762, +0.1211], +0.0709 [+0.0555, +0.0862], +0.0943 [+0.0814, +0.1074]; V2−MMseqs2 +0.0289 [+0.0035, +0.0544], +0.1819 [+0.1607, +0.2026], +0.2356 [+0.2159, +0.2551]; MMseqs2−ESM-2 R@1 +0.1565 [+0.1276, +0.1855]; MMseqs2 vs ESM-2 R@10 −0.0213 [−0.0484, +0.0047] and MAP −0.0125 [−0.0351, +0.0102] both unresolved; MMseqs2−V1 R@1 +0.0697 [+0.0413, +0.0975]. 10,000 resamples, 1,693 eligible, paired ✓. (Unused and available: MMseqs2 − ESM-2 at R@30 is −0.0786 [−0.1034, −0.0538], resolved *in ESM-2's favour* — the "unresolved at depth" concession is more conservative than the data requires.)

**Probe aggregates.** 3-NN V1 11/3/6 median +0.0075, V2 10/3/7 median +0.0041; linear V1 4/4/12 median −0.0139, V2 2/7/11 median −0.0107; tie band ±0.005; 20 of 23 tasks; exclusions `antibiotic_resistance` / `remote_homology` / `temperature_stability` (multiclass AUC undefined when test holds a class absent from train); the *different* three `ec_classification` / `go_mf` / `scope40_retrieval` ignore the probe flag (`comparison.json` `probe_ignored: true`) — all ✓ NEW_EVIDENCE §6, REBUTTAL_LEAKAGE §0. Seed 42 on every `v3/` row ✓. All three responses state these consistently.

**Remote homology.** 3-NN 0.5835 / 0.6587 / 0.6668; linear 0.6868 / 0.6899 / 0.7016 ✓ NEW_EVIDENCE §3. Pooled 457-class, TAPE's three holdouts 718 + 1,254 + 1,272 = 3,244 ✓. Paper Table 2 macro-F1 .223 → .313 = +40.5% ✓ (`PAPER_text.txt` line 306). MMseqs2 macro-OvR AUC 0.6523 at 88.9% hit coverage ✓.

**Leakage / decontamination.** Pfam 28,530,684 → 27,929,772 (−600,912); AFDB 135,404,259 → 126,301,607 (−9,102,652); STRING 76,070,154 → 71,891,417 (−4,178,737) ✓. Sum 27,929,772 + 126,301,607 + 15,000,000 = 169,231,379 ✓. Flags `--min-seq-id 0.4 --cov-mode 1 -c 0.8 -e 1e-3`, corpus-as-query ✓. Targets `biomap-research/fold_prediction[test]` (3,244) and `Synthyra/bernett_gold_ppi[test]` (3,022), **and only those two** — stated correctly in all three responses ✓. Negative controls 0 hits; Pfam/STRING exhaustive GPU prefilter 100% recall, AFDB k-mer `-s 5.7` 89.4% recall with its negative control re-run exhaustively ✓. Semi-join verification, 0 survivors ✓. STRING 15M subsample described as compute, not a control ✓.

**SCOPe memorization controls.** Median max identity 0.908, no sequence below 20%, [0, 0.2) bin empty ✓. Bin gains V2 +0.1524 (n=164) / +0.1565 (n=1,214) ✓. Spearman R@10 −0.038; AP −0.114 / −0.116, p < 3e-6 ✓. Partial (baseline-controlled) −0.083 / −0.081, both p < 1e-3 ✓. 404 zero-baseline queries, V2 +0.038, p=0.45 ✓. The "what these controls cannot rule out" paragraph (fold-level overlap surviving a 40%-identity filter) is well-founded and is the strongest paragraph in the rebuttal.

**Paper facts.** Abstract +105% (150M remote homology) and +19.9% (150M SCOPe R@1) ✓ lines 25–26, 55–56. Table 1 32.9M / 133.9M / 36.5M ✓ lines 140–143. Table 6 per-device 64 × accum 16 (35M) and 16 × 64 (150M), effective 1024 ✓ lines 560–562 — so "each MNRL loss call saw 64 examples at 35M and 16 at 150M" is right. Table 5 −126.9% (Enzyme Cat. Eff., Spearman) ✓ line 382 and +244.5% (Remote Homology, N=100) ✓ line 372. Line-21 "?" between "Lin et al., 2023" and "Henzinger et al., 2022" ✓ REVIEWS_actual.md line 78. "100,000" is the evaluator's `max_samples` cap over a 2,207-row family-level dataset ✓; superfamily-level V1 R@1 0.639 → 0.726 at 35M ✓ NEW_EVIDENCE §7.

**Code claims, checked in the repo, all correct.**
`protein_benchmark_suite.py:1578` `n_neighbors = max(1, min(3, train_size))` ✓;
`:1556` `KNeighborsRegressor(n_neighbors=3, metric=_KNN_METRIC)` with `:1524 _KNN_METRIC = "minkowski"`, default uniform weights ✓;
`:1499/1537/1541` `make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))` / `Ridge(alpha=1.0)`, no per-arm tuning ✓;
`:1440` `np.concatenate([emb_dict[s1], emb_dict[s2]])` for PPI ✓;
`benchmark_tasks.py:182-190` `peptide_hla` has a single `seq` field, so no combination operator ✓;
`thermostability` has no official test split (seeded 80/20 of train) ✓ REBUTTAL_LEAKAGE §3.

**V2 config.** `cached_mnrl`, 1,024 per device a true contrastive batch, gather-across-devices off ✓ REBUTTAL_LEAKAGE §4. All three responses say only "on 7 GPUs rather than 1" and correctly avoid NEW_EVIDENCE §2's overreach that the contrastive batch became 7×1024 — with allgather off it is 1,024 per rank. Ablation grounding (20/23 at +7.9% vs 16/23 at +6.7%; proportional +7.0% vs round-robin +6.7%) ✓ paper Table 4.

**"There is no 150M model on the decontaminated corpus"** — TRUE as of now. `models/protsent_esm2_150m_v2/` contains only `debug_traces`; commit `332be07` adds the training script, not a model. All three responses state this and none implies otherwise ✓.

**MMseqs2 all-task wins.** 3 tasks under 3-NN (`ec_classification`, `go_mf`, `beta_lactamase_peer`) and 6 under linear (those plus `enzyme_catalytic_efficiency`, `optimal_ph`, `stability`) ✓ NEW_EVIDENCE §5, confirmed in `comparison.json` (linear `stability`: MMseqs2 0.5817 > best embedding 0.5110; kNN `stability`: ESM-2 0.6435 > 0.5817, so it correctly appears only in the linear list).

**Format.** No links, no attachments, no figure references, no "see the revision" anywhere. Every number carries its metric, split and model. V1/V2 naming is used consistently in all three responses.

---

## Minimum edit list, in priority order

1. Remove HMMER from both "Not run" lists (jVGf §4, Yi1G §7) and report it: V2 − HMMER at R@1 −0.0124 [−0.0372, +0.0124], tied; R@10 +0.1412 [+0.1205, +0.1618] and MAP +0.1708 [+0.1511, +0.1905], ahead.
2. HNXd §3: change "V2's top-1 lead over alignment" to name MMseqs2, and add the HMMER tie.
3. HNXd §4-5 and Yi1G §8: replace "we have no multi-seed results" with the 5-seed probe-seed sweep (median SD 0.0000 over 24 rows; Thermostability the only spread at ~0.013–0.017; V1→V2 remote-homology +0.0079 ≈ 40× that task's seed SD), while keeping "only one training run per model exists".
4. HNXd §3 and jVGf §1: fix the eligible-table / paired-delta mismatch — one code path per response — and delete or correct the "under 0.001" parenthetical.
5. HNXd §2: stop offering the probe as the explanation for the literature gap; state that on Stability the 3-NN probe *beats* the linear probe for every arm (ESM-2 0.6435 vs 0.4395), so the gap is fine-tuning and scale, not the probe.
6. HNXd and jVGf: add the "V1-vs-V2 is not a controlled decontamination ablation" clause that currently appears only in Yi1G.
7. HNXd §2: disclose V1's linear macro-F1 0.4281 vs ESM-2 0.4414, and claim the stronger V2 statement (improves on both metrics under both probes).
8. Small: ">100M sequences" → ~43M across Pfam and STRING; "3-40 points" → "2 to 25 points"; "step 4,000, where the schedule bottoms" → "the nearest saved snapshot to the trough at step 4,208"; add or drop the unpersisted `-s 5.7` figure.
9. Housekeeping, not rebuttal text: `git checkout` the four deleted `results/benchmarks/seeds/` files, and add the per-task comparison.json values quoted in the responses to NEW_EVIDENCE.md.

Budget note: HNXd has 126 characters of slack and Yi1G has 57. Items 1–3 add text to both. The cheapest space in HNXd is the second half of §4-5 (the −126.9% / +244.5% arithmetic explanation duplicates Yi1G §5 almost verbatim); in Yi1G, the §1 identity-bin sentence can lose the parenthetical n-counts.
