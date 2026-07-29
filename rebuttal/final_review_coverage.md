# Coverage audit — FINAL_rebuttal.md vs the three verbatim reviews

Scope: every distinct concern, question and request in HNXd, jVGf and Yi1G, one row each.
CRITICAL = the reviewer states this is what would raise the score.

Paste-unit lengths (text strictly between BEGIN/END, stripped): **HNXd 9,610 · jVGf 9,234 ·
Yi1G 9,757** — all under 10,000. (The `Yi1G` comment claims 9,969; the real body is 9,757.
The stated count is wrong by 212 and will mislead whoever edits it.)
Free budget: HNXd 390 · jVGf 766 · Yi1G 243. Every replacement below is priced.

---

## 1. Reviewer HNXd

| # | Concern / request | Crit | Treatment | Where / note |
|---|---|---|---|---|
| H1 | Paper sits between two narratives (retrieval/clustering/few-shot framing vs k-NN property prediction) | | **ANSWERED BY EXPLANATION** | Opening picks one: "35M retrieval-and-remote-homology paper". Strongest structural move in the whole rebuttal. |
| H2 | Direct retrieval evaluation, precision@k | CRIT | **ANSWERED WITH DATA** | SCOPe-40 R@1/10/30/MAP, 1,693 eligible queries, vs MMseqs2/HMMER/ESM-2/V1/V2. |
| H3 | Clustering evaluation (silhouette) | CRIT | **DEFLECTED WEAKLY** | "We computed no clustering statistics" — bald, no reason. See **R1**. |
| H4 | Correlation between embedding distance and property similarity | CRIT | **IGNORED** | Not mentioned in the HNXd response at all. See **R2**. |
| H5 | Or: extensive analysis of how the space reorganises, local and global | CRIT | **ANSWERED WITH DATA** | Identity-stratified per-query R@10 gain, Spearman +0.038 p=0.45 on the 404 ESM-2-failure queries, R@10→R@30 flattening, layer sweep. |
| H6 | k-NN discussed as a model, not used as a probe of the space | | **ANSWERED WITH DATA** | Linear probe added on all 20 comparable tasks + per-layer sweep. |
| H7 | Reported numbers far below the literature for the same models | | **ANSWERED WITH DATA** | Metric mismatch identified (Spearman×100), plus 3-NN > linear on that task. |
| H8 | Specifically: BIOMAP Stability 58.8% vs 69.08% linear / 77.69% LoRA | | **CONCEDED** (correctly) + **DATA** | "our 58.8% is a correlation ×100 … we withdraw that comparison"; ESM-2 35M Spearman 0.6435 (3-NN) vs 0.4395 (linear). |
| H9 | The "we do not compare to specialized retrieval systems" limitation text | | **ANSWERED WITH DATA** | HMMER + MMseqs2 run (in jVGf/Yi1G; HNXd sees the table too). |
| H10 | Linear classifier on the frozen base model, esp. label-scarce | CRIT | **ANSWERED WITH DATA** | 20-task linear record: V1 4/4/12, V2 2/7/11, medians -0.0139 / -0.0107; few-shot table has a linear column. |
| H11 | Fine-tuning the base model as a baseline | CRIT | **DEFLECTED WEAKLY** | "We ran no fine-tuning sweep, so … is unmeasured." One clause, no argument. See **R3**. |
| H12 | "Comparing only to k-NN does not reflect how practitioners use ESM2" | | **ANSWERED WITH DATA** | Same linear record; conceded as the reason the general claim goes. |
| H13 | His proposed reframing (linear degrades under scarcity, k-NN holds) | CRIT | **ANSWERED WITH DATA** + **CONCEDED** | Explicitly tested and refuted with the reviewer's own hypothesis: "a trained linear head beats 3-NN in almost every model/task/N cell". Best paragraph in the document. |
| H14 | No statistical tests / CIs / SDs | CRIT | **ANSWERED WITH DATA** (retrieval) + **CONCEDED** (23-task table) | Paired bootstrap 10,000 resamples on SCOPe; "your objection stands there" on Table 2. |
| H15 | Improvements below 1%, benchmarks noisy | CRIT | **CONCEDED** (correctly, with evidence) | ±0.005 tie band, checkpoint-4,000 spread 0.005–0.008, "no sub-0.01 structural delta resolved". |
| H16 | 95% CIs by bootstrapping over individual predictions, for the reported metrics | CRIT | **ANSWERED WITH DATA** (SCOPe only) + **CONCEDED** (23 tasks) | The concession is on his single highest-value ask and is currently unbounded. See **R4**. |
| H17 | Multi-seed variability for the few-shot evaluation | CRIT | **ANSWERED WITH DATA** | 5 subset draws, ±SD everywhere; disclosed as subset-draw not training-seed variance; training-seed variance conceded unmeasured. |
| H18 | Table 5: absolute scores in addition to relative | CRIT | **ANSWERED WITH DATA — partial** | Only N=50 and N=1000 shown for remote homology; N=100 and N=250 exist in `results/benchmarks/fewshot_seeds.json` and are omitted. See **R5**. |
| H19 | "+244% from a low baseline may be a small absolute gain" | CRIT | **IGNORED (the specific cell)** | -126.9% is explained; +244.5% — his actual example — is never located or given an absolute. See **R5**. |

## 2. Reviewer jVGf

| # | Concern / request | Crit | Treatment | Where / note |
|---|---|---|---|---|
| J1 | Gains may just be structure injection; must be positioned vs ESM-S, S-PLM, ISM, Magneton | CRIT | **ANSWERED WITH DATA** + **DEFLECTED WELL** | AFDB ablation quantifies it (+6.7%→+3.2%, 16/23→13/23, RH +40.5%→+15.3%); relation-graph-vs-distillation distinction; "no matched runs … claim no superiority". |
| J2 | Q1: results with **both** AFDB and Pfam removed | CRIT | **DEFLECTED WEAKLY** | "We did not run the joint no-AFDB/no-Pfam ablation you asked for" — bare. This is his literal question 1. See **R6**. |
| J3 | Q2: apply ProtSent to SaProt / ProSST | CRIT | **DEFLECTED WELL** | Blocker is data (no predicted structures for Pfam/STRING, the corpus majority), not code. Residual risk: an AFDB-only SaProt run *is* feasible and a sharp reviewer may say so. |
| J4 | W2/Q3: where it sits on the generality–accuracy trade-off | CRIT | **ANSWERED WITH DATA** | MMseqs2 as a full pipeline over 23 tasks; alignment wins 3 (3-NN) / 6 (linear); MMseqs2 below chance on DeepSol (AUC 0.4185); SCOPe table + paired CIs. Strongest section in the document. |
| J5 | ProTrek: cite and possibly compare | | **DEFLECTED WELL** | Cited, positioned as the trimodal retrieval-optimised point, "expect it to win … did not run it." |
| J6 | Q4: how CoSENT works on DMS — what is paired, what similarity value | | **ANSWERED BY EXPLANATION** + **CONCEDED** | `(WT, mutant, normalized fitness∈[0,1])`, ordinal within batch, no absolute target; paper text admitted wrong; WT-anchored limitation volunteered. |
| J7 | Minor: missing reference, line 21 ("?") | | **ANSWERED BY EXPLANATION** | Broken citation key; Heinzinger 2022 + Redl 2023 in Related Work. |
| J8 | Score-raise axis A: why ProtSent goes beyond structure injection **or** why superior to existing methods | CRIT | **ANSWERED WITH DATA** (beyond-structure) + **DEFLECTED WELL** (superiority disclaimed) | Source fingerprints + GB1 0.6582/0.7108/0.7806. He offered "either"; they answer the first and decline the second. Correct choice. |
| J9 | Score-raise axis B: generality–accuracy position | CRIT | **ANSWERED WITH DATA** | As J4. |
| J10 | Not-run baselines named by him (ProtTucker etc.) | | **DEFLECTED WELL** | Explicit list, "no matched comparison", ProtTucker named as the gap. |

## 3. Reviewer Yi1G

| # | Concern / request | Treatment | Where / note |
|---|---|---|---|
| Y1 | AFDB training seqs not filtered against SCOPe test domains | **ANSWERED WITH DATA** + **CONCEDED** residual | Cannot filter (median max identity 0.908); identity stratification, 164-query decontaminated subset (MAP gain +0.2859 vs +0.2232 overall); fold-level overlap conceded untested. |
| Y2 | PPI: stricter <50%/40% identity analysis | **ANSWERED WITH DATA** (filter, at 40%/80% — stricter than asked) + **CONCEDED** (no post-filter PPI benchmark) | "That half of weakness 1 is unanswered." Compliance is quieter than the concession. See **OC5**. |
| Y3 | DMS assumption: preserve fitness-induced *ordering* of WT–mutant distances | **ANSWERED BY EXPLANATION** + **CONCEDED** text error | Best-case answer: what he asks for is literally what the code does. |
| Y4 | Effective batch 1024 vs actual in-batch negatives | **ANSWERED BY EXPLANATION** + **CONCEDED** | 64 at 35M, 16 at 150M; accumulation does not share negatives; named as the likeliest cause of the 150M results. |
| Y5 | Superscript "+" in Eq. 1 undefined | **ANSWERED BY EXPLANATION** | One sentence, adequate. Could be crisper ("+ marks the positive paired with anchor i"), not a defect. |
| Y6 | How two protein embeddings are combined for pair tasks (PPI, peptide-HLA) | **ANSWERED BY EXPLANATION** | Independent embedding + concatenation; peptide-HLA is a single pipe-joined `seq` field, so no operator applies. |
| Y7 | k-NN regression: uniform or distance-weighted | **ANSWERED BY EXPLANATION** | `KNeighborsRegressor(n_neighbors=3, metric="minkowski")`, unweighted; small-N estimator change disclosed. |
| Y8 | Ablations contradict the default config | **CONCEDED** + **DATA** + acted on | Hard negatives 20/23 at +7.9% vs 16/23 at +6.7%; proportional +7.0% vs +6.7%; V2 uses neither default. Volunteered selection-channel caveat — see **OC4**. |
| Y9 | Baselines: ProtTucker, HMMER, MMseqs2, Foldseek, PLMSearch, DHR, Redl et al. | **ANSWERED WITH DATA** (2 of 7) + **DEFLECTED WEAKLY** (5 of 7) | The not-run list is a bare list with no argument for why the two run are the right two. See **R7**. |
| Y10 | Statistical evidence weak, Table 2 deltas tiny | **ANSWERED WITH DATA** (SCOPe) + **CONCEDED** (Table 2) | As H14/H16. |
| Y11 | Biological assumption: heterogeneous relations in one embedding space | **DEFLECTED WEAKLY** | "not evidence against interference" — but the direct evidence of interference is in his own item 8 and is never connected. See **R9**. |
| Y12 | Limitations discussion insufficient; state when ProtSent is expected to be reliable | **DEFLECTED WEAKLY** | The envelope exists across the response but is never stated as one. See **R8**. |
| Y13 | Clarity 2 / reproducibility | **ANSWERED BY EXPLANATION** | Errors section + protocol specs (items 3–5) fix most of what he called unreproducible. |

---

## 4. Replacement text for every DEFLECTED WEAKLY / IGNORED row

### R1 — HNXd §1, clustering (H3). Replace the first sentence of §1.
> **We computed no clustering statistics — no silhouette, NMI or ARI — because on this
> gallery they are the weaker form of the same measurement.** Silhouette and ARI score
> family structure globally, assuming comparable within-family scatter and family sizes;
> on SCOPe-40 only 1,693 of 2,207 domains have a non-self same-family neighbour at all, so
> a global index is dominated by the 514 families a neighbourhood metric cannot score.
> Recall@K and MAP measure the same neighbourhood, per query, on exactly the queries where
> it is defined — and unlike any silhouette score they can be computed for HMMER and
> MMseqs2, which is what makes the comparison below possible.

Cost +590 (replaces 92). Fund from R5's ledger.

### R2 — HNXd §1, distance-vs-property correlation (H4). Append to §1.
> On your third example — correlation between embedding distance and property similarity —
> the one place we measure it is fitness, where the training objective targets it directly:
> GB1 variant effect (Spearman between predicted and measured fitness, 3-NN on frozen
> embeddings, test split, mean of 5 draws) is 0.6582 (ESM-2 35M) / 0.7108 (V1) / 0.7806
> (V2), SD 0.0000. That is probe-mediated, not a raw distance-vs-property curve; we did not
> compute the raw curve.

Cost +400.

### R3 — HNXd §4-5, fine-tuning baseline (H11). Replace "We ran no fine-tuning sweep, so how a fine-tuned ESM-2 compares is unmeasured."
> **We ran no fine-tuning sweep.** We do not think it would rescue us: fine-tuning is the
> stronger of the two baselines you named, and the weaker one already beats us — a linear
> head on stock ESM-2 35M wins 11 of 20 comparable tasks against V2 (median -0.0107, test
> split, tie band ±0.005). A stronger baseline cannot turn that into a win. What fine-tuning
> would change is the axis of comparison — one trained model per task versus one frozen
> index reused across tasks and used for retrieval, where no head exists at all — and that
> is the comparison the paper should have been making.

Cost +490 (replaces 80).

### R4 — HNXd §3, bound the CI concession (H16). Append after "your objection stands there."
> What that costs is bounded, and we would rather state the bound than the excuse: the
> 23-task table was the evidence for the general-purpose claim, which we withdraw above.
> Every claim we still make carries a number with its own uncertainty — SCOPe-40 by paired
> bootstrap, few-shot by 5-draw SD — with one exception we name: the single-run linear
> remote-homology accuracies (0.6868 ESM-2 35M / 0.6899 V1 / 0.7016 V2, test split), where
> only the 0.005–0.008 checkpoint spread bounds us.

Cost +430.

### R5 — HNXd §4-5, Table 5 completeness and the +244% cell (H18, H19). Add two rows and one sentence.
Rows to insert into the few-shot table (all remote homology, accuracy, mean ± SD, 3-NN / linear):

| N | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|---|
| 100 | 0.115±0.007 / 0.222±0.006 | 0.135±0.008 / 0.282±0.008 | 0.125±0.024 / 0.258±0.009 |
| 250 | 0.148±0.002 / 0.310±0.007 | **0.223±0.011 / 0.394±0.012** | 0.200±0.010 / 0.368±0.013 |

> Your example cell was right to be suspicious. The paper's +244.5% is remote homology at
> N=100 under 3-NN; re-run with a fixed estimator, 5 draws and the full test split it is
> 0.1155 → 0.1349 accuracy for V1, **+0.0194 absolute, +16.8% relative**. The distance
> between +244.5% and +16.8% is protocol — default split, one draw, and the small-N
> `n_neighbors` change — not arithmetic.

Cost ≈ +560 (rows) +390 (sentence). **HNXd cut ledger** to pay for R1–R5 (≈ +2,470 against
390 free): delete the layer-sweep paragraph (-470, it is labelled "not a defence" and
survives in the paper), the two duplicate "(over all 2,207 … 0.7671)" parentheticals
(-180), the `-s 7.5 -e 10 --max-seqs 300` flag strings after first use (-60), and the
V1-vs-ESM-2 columns of the 3-NN/linear table, keeping V2 only (-120). Remaining shortfall
≈1,250: drop R2 (400) and shorten R1 to its first sentence plus the 1,693/2,207 clause
(-300) if you must. **Do not drop R5** — it is the row an explicit score-raise depends on.

### R6 — jVGf §2, the joint ablation (J2). Replace "We did not run the joint … is unanswered."
> **We did not run the joint no-AFDB/no-Pfam ablation you asked for**, and the two single
> ablations do not substitute for it. What we can put against it is the non-structural half
> measured on its own, on a decontaminated model and absolute numbers rather than the
> relative table above: GB1 variant effect (Spearman, 3-NN, test split, 5 draws) 0.6582
> (ESM-2 35M) → 0.7806 (V2), SD 0.0000, and PPI moving +5.3% → -0.5% when STRING alone is
> removed. Fitness order and interaction are relations no structure teacher supplies, so
> what your ablation would settle is how much of the *benchmark aggregate* survives without
> structure, not whether the non-structural sources do anything. If that is your decisive
> item, say so and we will report it in discussion.

Cost +700 (replaces 190). jVGf has 766 free; also apply **R10** below, so cut the
"+0.1810 at [0.4,0.7), n=315;" bin (-30) and the `--alignment-mode 3` flag (-20).

### R7 — Yi1G §7, the not-run baselines (Y9). Replace "Not run: … closest published analogue to our protocol."
> **Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, or Redl et al. 2023.** The two
> we did run are the two that take the same input we do: HMMER and MMseqs2 see sequence
> only, so they bound what a sequence-only encoder has to beat, and HMMER is the one that
> costs us a claim. Foldseek and ProTrek consume structure (and, for ProTrek, text) at query
> time; losing to them would say nothing about ProtSent, and we claim no superiority to any
> of the five. ProtTucker is the real gap and we will not dress it otherwise: contrastive
> fine-tuning of frozen embeddings for remote homology is our protocol, and it is the run we
> would most want back.

Cost +640 (replaces 200). **Yi1G cut ledger** for R7–R9 (≈ +1,180 against 243 free): drop
"Identity-gain Spearman is -0.116 (average precision, p < 3e-6), negative after a headroom
control (partial -0.081), and" (-130 — it also introduces an unflagged metric switch, D4);
shorten the negative-control parenthetical to "(1,000 random sequences per filtered corpus,
AFDB's exhaustively)" (-65); delete the R@30 column from the item-7 table (-70); delete
"row counts summing to the 169,231,379 in the training log" (-60); delete the repeated
0.7671 scaling clause (-90); compress item 1's second SCOPe recap (-120); drop
"(the paper's +5.3% is a pre-decontamination V1 number)" from the PPI paragraph and fold it
into R8 (-55). Total ≈ -590 + 243 free = 833; shorten R7 by dropping its ProTrek clause and
R8 by one clause to land it.

### R8 — Yi1G, reliability envelope (Y12). Add as a new short paragraph before the closing.
> **When to expect this to work, and when not.** It helps where the label *is* a homology or
> structure relation: SCOPe-40 ranking depth (V2 - HMMER, R@10 +0.1412 [+0.1205, +0.1618];
> MAP +0.1708 [+0.1511, +0.1905]) and remote homology on the decontaminated test split,
> where V2 is above ESM-2 35M under both probes. It does not help where the label is a
> property with no homology signal: under a linear probe stock ESM-2 35M is the better
> embedding on 11 of 20 comparable tasks. Between those two regimes we resolve nothing below
> 0.01, and we say so rather than reporting the sign.

Cost +540.

### R9 — Yi1G §6, single-space interference (Y11). Replace "They are also all we have … evidence against interference."
> On your single-space question the ablations only show that each source moves its own task
> family (removing STRING takes PPI +5.3% → -0.5%, submitted V1, default split). The direct
> evidence of the cost is in item 8: sharing one space across four relation types leaves V2
> losing 11 of 20 comparable test-split tasks to its own untuned backbone under a linear
> probe, winning 2. Heterogeneous relations in one embedding space are not free, and that
> record is the price.

Cost +380 (replaces 210).

### R10 — jVGf, missing V1↔V2 caveat (defect D6, not a review row). Append to the SCOPe table caption in §1.
> (V2 also changes sampling, drops synthetic hard negatives and uses a true 1,024-example
> contrastive batch where V1's loss call saw 64, so V2 - V1 is not a decontamination
> ablation; no unfiltered retrain at the V2 recipe exists.)

Cost +230. Present in HNXd and Yi1G, absent in jVGf, which shows a V1/V2 table — without it
jVGf can read the V1→V2 jump as a controlled decontamination effect, which the hard rules
forbid implying.

---

## 5. Over-concessions — things conceded that did not need conceding

**OC1 (severe, HNXd opening).** "…and adds nothing a trained linear head could not already
extract from mean-pooled ESM-2." This is contradicted by the same response two sections
later: on remote homology V2 beats ESM-2 35M *under the linear probe* (accuracy 0.6868 →
0.7016, macro-F1 0.4414 → 0.4527, test split), and SCOPe-40 retrieval has no head at all —
V2 - ESM-2 is R@10 +0.1607 [+0.1412, +0.1802], MAP +0.2232 [+0.2082, +0.2383]. The
sentence gives away more than the data do, and a careful reviewer will catch the
contradiction and read it as sloppiness rather than candour.
**Replacement:** "…and, on the 20-task probe suite, adds nothing a trained linear head could
not already extract from mean-pooled ESM-2 (median -0.0107 for V2). The two exceptions are
the two things we still claim: retrieval, where there is no head, and remote homology, where
V2 leads under both probes."

**OC2 (severe, HNXd §4-5).** "the label-scarcity claim is withdrawn rather than reframed."
What the data kill is the *mechanism* — "linear degrades, k-NN holds" is false. The claim
that ProtSent embeddings help when labels are scarce is not killed; on remote homology under
the reviewer's own preferred linear head, V1 is 0.3100 → 0.3942 at N=250 and 0.2878 → 0.3772
at N=1000, V2 is 0.3683 and 0.3552, SDs ≈0.01. Withdrawing the whole claim throws away a
seed-stable result on the surviving task family.
**Replacement:** "(i) Your proposed framing is not supported: a trained linear head beats
3-NN in almost every model/task/N cell, including N=50, so 'linear degrades while k-NN stays
competitive' is false here and we withdraw that mechanism. What survives is narrower and
task-bound: under the linear head on remote homology, ProtSent-V1 is 0.3942±0.012 at N=250
and 0.3772±0.008 at N=1000 against 0.3100±0.007 and 0.2878±0.014 for ESM-2 35M (V2: 0.3683,
0.3552). On metal-ion binding at N=1000 the same head puts ESM-2 35M first, 0.666±0.001 vs
0.595±0.001 for V2. Label scarcity helps us on homology tasks and nowhere else we measured."

**OC3 (severe, HNXd opening).** "Our strongest surviving result sits on SCOPe-40, the one
benchmark we could not decontaminate." True of SCOPe, but it hands the AC a rejection
sentence while omitting that `remote_homology` **was** a filter target at 40%/80% and V2 wins
there under both probes. As written the reader concludes every surviving claim is
contaminated.
**Replacement:** "Our largest margin sits on SCOPe-40, which we could not decontaminate at
the corpus level; our decontaminated result is remote homology, where the corpus was filtered
at 40% identity / 80% coverage against the test set and V2 still leads ESM-2 35M under both
probes (3-NN 0.5835 → 0.6668, linear 0.6868 → 0.7016). Section 1 gives the identity
stratification that bounds what SCOPe's contamination can be."

**OC4 (moderate, Yi1G §6).** "V2's numbers are therefore not a clean held-out measurement" —
blanket, and it taints the SCOPe result that carries the paper. The selection channel is real
but narrow: the configuration was chosen on the 23-task aggregate relative gain, in which
SCOPe-40 is one task of 23, not the criterion.
**Replacement:** "The consequence, stated not implied: the configuration was chosen on the
23-task aggregate relative gain, which was scored on these same benchmarks — a selection
channel the corpus filter does not touch. SCOPe-40 enters that aggregate as one task of 23
rather than as the criterion, and the alignment baselines it is now scored against were run
after the configuration was fixed; but we cannot call V2's 23-task numbers a clean held-out
measurement, and we do not."

**OC5 (mild, Yi1G §1).** "That half of weakness 1 is unanswered" is louder than the
compliance that precedes it. He asked for a <50%-or-40% identity analysis; they filtered at
40% identity / 80% coverage and verified 0 survivors. Lead with that.
**Replacement:** "PPI: the filter you asked for was run — 40% identity / 80% coverage,
stricter than the 50% you named, removing 4,178,737 STRING pairs, 0 flagged sequences
surviving. What does not exist is the downstream number: `ppi_bernett` is a pair-input task
and is not in the 23-task sweep, so the paper's +5.3% remains a pre-decontamination V1 result
and we have not re-measured it. That is the open half of weakness 1."

**Correctly conceded, do not reverse:** the general-purpose superiority claim (jVGf opening,
HNXd §2 — the linear record is 2/7/11 and the hard evidence supports the withdrawal); all
150M results (no decontaminated 150M exists, and the 16-example MNRL batch explains them);
the BIOMAP 58.8% comparison; the Eq. 1 / batch-semantics / DMS-text errors; the top-1
retrieval claim against HMMER (V2 - HMMER R@1 -0.0124 [-0.0372, +0.0124] is a tie, and the
rebuttal says so in all three responses).

---

## 6. The Area Chair's verdict — "no manuscript left on the table"

**Still true of the rhetoric; false on the evidence.** Read the opening paragraphs alone and
the AC is right: "adds nothing a trained linear head could not already extract" (OC1) +
"our strongest surviving result sits on the one benchmark we could not decontaminate" (OC3) +
"the label-scarcity claim is withdrawn" (OC2) + "V2's numbers are not a clean held-out
measurement" (OC4) is a self-rejection, and all four are in the first screenful of the two
responses an AC will read first.

Read the numbers and there is a paper:

1. SCOPe-40 family retrieval, 1,693 eligible queries, paired bootstrap 10,000 resamples:
   V2 ties the best alignment tool at top-1 (V2 - HMMER R@1 -0.0124 [-0.0372, +0.0124]) and
   is decisively ahead at depth (R@10 +0.1412 [+0.1205, +0.1618]; MAP +0.1708 [+0.1511,
   +0.1905]), with the coverage caveat stated.
2. Remote homology, decontaminated corpus, test split: V2 above ESM-2 35M under **both**
   probes (3-NN 0.5835 → 0.6668; linear 0.6868 → 0.7016; macro-F1 0.4414 → 0.4527).
3. Few-shot remote homology under a trained linear head, 5 draws: +0.084 at N=250 and +0.089
   at N=1000 for V1 over ESM-2 35M, SDs ≈0.01.
4. GB1 variant effect, 3-NN, test split: 0.6582 → 0.7806, SD 0.0000 — a non-structural,
   decontaminated, absolute result.
5. A verified decontamination pipeline (0 flagged survivors, residual bounded ~0.3%) and a
   corrected specification of the training objective and evaluation protocol.

That is "contrastive fine-tuning of a small frozen PLM buys homology-centric retrieval depth
and few-shot transfer, at a measured cost on non-homology tasks" — a NeurIPS-plausible
narrow paper.

**Reversals required, in priority order:** OC1 (scope the linear-head sentence to the probe
suite and name the two exceptions), OC3 (name remote homology as the decontaminated result),
OC2 (withdraw the mechanism, keep the task-bound few-shot result), OC4 (narrow the selection
channel to the 23-task aggregate). Nothing else needs reversing, and the four rewrites cost
about 700 characters net across two responses — well inside the ledgers above.

---

## 7. Hard-rule audit

| Rule | Status |
|---|---|
| Each response < 10,000 chars | **PASS** — 9,610 / 9,234 / 9,757. Note the `Yi1G` header comment says 9,969; wrong by 212. |
| No links | **PASS** |
| No attachments / figure references / "see the revised paper" | **PASS**. Table 2/5/6 are referenced, but only as tables of the *submitted* paper the reviewers hold — HNXd himself asks about Table 5. Nothing points at a revision. |
| Every number carries metric, split, model | **MOSTLY PASS**, 5 defects below |
| No "beats alignment at top-1" | **PASS** — stated as a tie against HMMER and a win over MMseqs2 in all three responses |
| No "generally superior to ESM-2" | **PASS** — the withdrawal is explicit in all three |
| No implied V1→V2 decontamination ablation | **FAIL in jVGf only** (D6) — see **R10** |
| No 150M-on-decontaminated-data implication | **PASS** — negated explicitly in HNXd, jVGf and Yi1G |

**Numeric-attribution defects (all cheap to fix):**

- **D2 — Yi1G §6.** "+7.9% against 16/23 and +6.7%", "+7.0% vs round-robin's +6.7%" carry no
  model, split or metric, while the same numbers in jVGf are qualified as single-run relative
  percent from the submitted pre-decontamination model on the suite's default split. Add
  "(submitted V1, suite default split, single run, mean relative change)" once at first use.
- **D3 — HNXd §2 layer sweep.** 0.6373 / 0.6703 / 0.6803 / 0.7033 never name the metric. Add
  "linear-probe accuracy" once. (If the layer paragraph is cut per the R5 ledger, moot.)
- **D4 — Yi1G §1.** "Identity-gain Spearman is -0.116 (average precision, p < 3e-6)" switches
  the metric from R@10 to average precision mid-paragraph and does not say which model pair
  the gain is. Either qualify as "V2 - ESM-2 35M per-query average-precision gain" or cut it
  (it is in the R7 ledger).
- **D5 — jVGf §1.** "beta-lactamase Spearman 0.8026, above every embedding arm" — the arm
  producing 0.8026 (MMseqs2) is only inferable from context. Add "(MMseqs2)".
- **D7 — jVGf §3.** "section 2 shows each type moving a different task family" reads as the
  *paper's* section 2. Change to "item 2 above".
- **D8 (nit) — HNXd.** "your Q2", "(Q5, Q4)": the reviewer did not number his questions.
  Low risk given their order, but quoting three words ("a linear classifier on top of the
  base model") is safer than an index the reviewer has to reconstruct.
