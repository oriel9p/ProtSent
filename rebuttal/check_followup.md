# Proofread — `FOLLOWUP_openreview.md` + `ADDITIONS_to_rebuttal_docx.md`

Verified against the repo on 2026-07-31. Findings ranked by damage.
Verdict: **do not post Part 3 as written.** Two items in it are wrong or retract a
concession we already made in public. Part 2's numbers are almost all correct; the
problem there is what is omitted, not what is stated.

---

## Damage rank 1 — Part 3 contradicts our first response on the MNRL negative set, unflagged, on the one point where the reviewer was right

**Follow-up says (Part 3):**

> "**MNRL batch semantics and Eq. 1 (Yi1G).** Each anchor is contrasted against the other
> 1,023 positive-side examples in its source batch. We use `CachedMultipleNegativesRankingLoss`
> with a logical contrastive batch of 1,024; the `mini_batch_size` parameter partitions only
> the forward/backward computation for memory and does not reduce the negative set. Our use of
> "effective batch size" conflated the two [...]"

**Our first response, `rebuttal/FINAL_rebuttal.md:196`, already posted:**

> "The reviewer is right, and this is a real error. The 1,024 reported is an optimizer batch
> formed by gradient accumulation (64 per device over 16 steps at 35M; 16 over 64 at 150M),
> and accumulation does not share in-batch negatives. **Each MNRL call therefore saw 64
> examples at 35M and 16 at 150M** — the likeliest explanation for the 150M results we no
> longer defend. The retrain uses a true 1,024-example batch per device."

**Sources.** `rebuttal/PAPER_text.txt:560-562` — Table 6: "Per-device batch size 64 | 16",
"Gradient accumulation 16 | 64", "Effective batch size 1024 | 1024". `COAUTHOR_BRIEF.md:138`
— the 1,024 contrastive batch and CachedMNRL are the **V2** config: "Both:
CachedMultipleNegativesRankingLoss, 1024 contrastive batch per device". `COAUTHOR_BRIEF.md:149`
lists "7x/6x larger effective batch" as a V1→V2 **difference**.

**Why this is the worst item.** Yi1G's exact question (`rebuttal/REVIEWS_actual.md`, Yi1G
weaknesses) was: "It is unclear whether the effective batch size of 1024 actually contributes
to the in-batch negative set, or whether negatives are only computed within each smaller
forward micro-batch." We answered "you are right, it does not — negatives were 64/16." The
follow-up now answers "it does, 1,023, and our wording merely conflated two things." The
reviewer will read the two side by side. Worse, the paper's own Table 6 says *gradient
accumulation*, not `mini_batch_size`, so the follow-up's mechanism does not even match the
submitted config — it describes `train_esm2_35m.sh` / `train_esm2_150m.sh` (V2).

**Correction.** Split the answer in two, explicitly:

> The submitted models used per-device batch 64 (35M) / 16 (150M) with gradient accumulation
> to an optimizer batch of 1,024; accumulation does not share negatives, so each MNRL call
> saw 64 and 16 in-batch negatives respectively, as we stated in our first response. The
> decontaminated retrains use `CachedMultipleNegativesRankingLoss` with a true 1,024-example
> contrastive batch per device, where `mini_batch_size` partitions only the forward/backward
> pass and does not reduce the negative set. The paper's "effective batch size 1024" referred
> to the optimizer batch and should not have been read as the negative-set size.

Keep the Eq. 1 sentence as is — it matches `FINAL_rebuttal.md:196` and
`PAPER_text.txt:195-200`.

---

## Damage rank 2 — Part 1 claims every correction was self-found; our own first response credits the reviewers four times

**Follow-up says (Part 1):**

> "We would only note that the direction of the change is toward a smaller, better-evidenced
> claim, and that **every correction in it was found and reported by us rather than by a
> reviewer.**"

**Contradicted by `rebuttal/FINAL_rebuttal.md`, already posted:**

- `:196` "The reviewer is right, and this is a real error." (MNRL batch)
- `:158` "The leakage objection was correct and we treated it as decisive"
- `:141` "The reviewer is right that the paper's description is unclear, and in fact it is
  wrong" (CoSENT/DMS)
- `:220` "The reviewer is right about Table 2" (no intervals, one seed)

Also `rebuttal/REVIEWS_actual.md`: Yi1G raised the pair-level protocol and the k-NN
weighting gap; HNXd raised the +244.5% cell. Every one of those is a correction found by a
reviewer.

**Why it is damaging.** It is false, it is boastful, and it is the single easiest sentence in
the document for a hostile AC to disprove — the disproof is in our own first comment, one
scroll up. It also poisons the sentence it sits in, which is otherwise our best argument.

**Correction.** Delete the clause. The surviving sentence still works:

> "We would only note that the direction of the change is toward a smaller, better-evidenced
> claim. Several of the corrections were ones the reviewers identified; the remainder we
> found while checking their objections, and we reported those too."

---

## Damage rank 3 — the 150M top-1 "correction" is true only against the weaker phmmer setting, and our public repo says so in bold

**Follow-up says (Part 2.1):**

> "It is **not** accurate at 150M, where ProtSent-V2 exceeds HMMER at top-1 significantly."

with the table row "| HMMER (phmmer) | 0.6970 | 0.7809 | 0.4747 |" — **no flags stated**.

**`rebuttal/NEW_EVIDENCE.md:510` (on the public `origin/rebuttal` branch):**

| | no-hit | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|---|
| phmmer **filters off** | 0 / 2,207 | **0.7525** | 0.8978 | 0.9232 | 0.6067 |

`results/benchmarks/hmmer_maxsens.json`: `eligible.hit1 = 0.7525103366804489`.
ProtSent-V2-150M R@1 = **0.7431** (`results/benchmarks/v2_150m/protsent_v2_150m_linear/...csv`,
`eligible_Recall@1 = 0.74306`). **We are 0.009 behind max-sensitivity phmmer at top-1 at
150M, not ahead of it.**

**`rebuttal/NEW_EVIDENCE.md:549`, our own standing instruction:**

> "**Any rebuttal sentence comparing to HMMER must use the filters-off numbers.**"

and `:537` "ProtSent-V2 is **behind** max-sensitivity HMMER at top-1 (-0.068, resolved)".

**Aggravating factor.** In the same first response we criticised unstated alignment settings:
`FINAL_rebuttal.md:111` "any MMseqs2 number needs its sensitivity stated". The follow-up
states MMseqs2's `-s 7.5` in the table and omits phmmer's `-E 10`. That asymmetry is exactly
the thing we told the reviewers to watch for.

**Correction.** Either (a) drop the "exceeds HMMER at top-1" upgrade entirely and keep the
correction limited to "the 35M tie does not generalise; see the per-scale table", or (b) state
the flags and the max-sensitivity number:

> "ProtSent-V2-150M exceeds phmmer at `-E 10` at top-1 (+0.0455 [+0.0219, +0.0691]). Against
> phmmer with heuristic filters disabled, which is the stronger setting and the one we
> recommend for any comparison, phmmer reaches R@1 0.7525 against our 0.7431; we have not run
> the paired bootstrap for that pairing at 150M and do not claim a top-1 win there."

Option (b) costs the headline. Option (a) is safer and still corrects the record. Do not post
the sentence as written — `hmmer_maxsens.json` is already on `origin/rebuttal`.

---

## Damage rank 4 — Part 2.2's mechanism paragraph invites the whitening rebuttal, which our repo has already run and lost

**Follow-up says (Part 2.2):**

> "This is, we think, the mechanism behind the retrieval results. [...] It also explains why
> the gains concentrate in retrieval, clustering and nearest-neighbour transfer rather than
> under a trained readout: **a trained linear head can compensate for a poorly conditioned
> space, and k-NN cannot.**"

**What is in the repo, public:**

- `results/benchmarks/probe_gap_analysis.json` — remote-homology 3-NN: ESM-2-150M raw
  **0.5200**, ESM-2-150M **whitened 0.7346**, ProtSent-V2-150M raw **0.6606**, ProtSent-V2-150M
  whitened 0.7343. A label-free, untrained linear map takes the backbone *above* ProtSent-V2.
- `whiten_scope_control.py:6-9` (docstring): "on remote homology a whitened vanilla k-NN
  recovers essentially all of ProtSent's k-NN advantage. **That is a deflating result for the
  method.**"
- `results/benchmarks/whiten_scope_control.json` — SCOPe R@1 at 150M: ESM-2 whitened 0.7336
  vs ProtSent-V2 0.7425; paired `ProtSent-V2-150M [raw] - ESM-2-150M [whitened]` hit1
  **+0.0089 [-0.0106, +0.0289], `excludes_zero: false`.**
- `rebuttal/ADDITIONS_to_rebuttal_docx.md:98-101`, our own words: "simply whitening them
  recovers a large fraction of ProtSent's k-NN advantage — on remote homology it closes nearly
  all of it, and **on SCOPe it closes the top-1 gap at 150M**."

So the claim "k-NN cannot compensate" is refuted by our own file, and the top-1 upgrade in
rank 3 above is *entirely* absorbed by whitening the baseline. Note also
`probe_gap_analysis.json`: merely **standardizing** ESM-2 lifts participation ratio from 7.9
to 31.3 (35M) and 10.6 to 32.3 (150M) — a reader looking at our anisotropy table will ask why
the standardized column is missing.

**Correction.** Delete "and k-NN cannot", and pre-empt rather than invite:

> "A trained linear head is invariant to any invertible linear map of the space, so it is
> insensitive to this conditioning; k-NN is not. We ran the obvious control: PCA-whitening the
> untuned backbone, fitted on the evaluation gallery itself, recovers most of the k-NN
> advantage on remote homology and closes the SCOPe top-1 gap at 150M. What survives against
> that maximally generous baseline is ranking depth and MAP (+0.027 R@10 and +0.027 MAP at
> 150M, both excluding zero). We will report this control in the revision."

Volunteering it costs ~0.15 of the SCOPe margin and buys the paper. Concealing it costs the
paper if anyone opens `whiten_scope_control.json` — and it is on the branch we handed them.

---

## Damage rank 5 — the whole internal apparatus is on the public `origin/rebuttal` branch, including the file the task calls an "internal planning doc"

`ADDITIONS_to_rebuttal_docx.md` is described in-file as documentation, not reply text — but
`git cat-file -e origin/rebuttal:rebuttal/ADDITIONS_to_rebuttal_docx.md` succeeds. So does
every one of these:

```
rebuttal/ADDITIONS_to_rebuttal_docx.md   rebuttal/NEW_EVIDENCE.md
rebuttal/check_coauthor_{blind,facts,prose}.md
rebuttal/final_review_{hostile,coverage,prose,standalone}.md
rebuttal/DRAFT_rebuttal.md               rebuttal/BLIND_headlines.md
results/benchmarks/hmmer_maxsens.json    results/benchmarks/whiten_scope_control.json
results/benchmarks/probe_gap_analysis.json  COAUTHOR_BRIEF.md  RUNS.md
```

`ADDITIONS_to_rebuttal_docx.md:35` tells the reviewers the branch is an artifact:
"Code, results, documentation | github.com/oriel9p/ProtSent, branch `rebuttal`".

Quotable, verbatim, from files an AC can reach in two clicks:

- `ADDITIONS...md:21-25` — "**Implication for effort allocation.** HNXd is addressed. The
  remaining leverage is jVGf [...] **That is the cheapest remaining score movement and it needs
  no compute.**"
- `ADDITIONS...md:102-104` — "A reviewer can run this in an afternoon. **We judged it out of
  scope for the rebuttal** [...] it belongs in the camera-ready, where it strengthens the
  narrow claim more than it costs the broad one." (i.e. we knew about the whitening control
  and chose not to report it)
- `rebuttal/final_review_hostile.md:16` — a section header reading "**KILL SHOT** — The one
  claim that survives is the one benchmark you chose not to control, and the control you offer
  has no power against the leakage you concede exists"

**Correction.** Before posting anything: `git rm` the internal-review and planning files from
the `rebuttal` branch and force-push, or move the reviewer-facing artifacts to a clean branch
and repoint `ADDITIONS...md:35`. Keep `NEW_EVIDENCE.md`, `RUNS.md`, `COAUTHOR_BRIEF.md` and
the JSONs — those are defensible and being open about them is the point — but the strategy
memos and the simulated hostile reviews are not evidence, they are ammunition. This is
independent of the follow-up text and is the highest-value five minutes available.

---

## Damage rank 6 — "the ordering reverses" rests on 0.0003, and the metric where we lose is omitted

**Follow-up says (Part 2.1):**

> "Under 3-NN, decontamination costs 4.4 points relative to V1 [...] **Under a linear probe
> the ordering reverses.** [...] we report it as measured rather than selecting the favourable
> probe"

The table gives linear-probe accuracy ESM-2 150M **0.7500**, V1 **0.7401**, V2 **0.7503**.

**Verified:** all six cells match `results/benchmarks/v2_150m/*/bench_*.csv` exactly (kNN
0.51942 / 0.70469 / 0.66122; linear 0.75 / 0.74014 / 0.75031). The numbers are right. Three
problems with the sentence around them:

1. V2 beats ESM-2 by **0.0003** — one sequence in 3,244. `results/benchmarks/verify_remote_homology_150m.json`,
   also public, re-measures from fresh embeddings and gets **ESM-2 0.7506 > V2 0.7497 > V1
   0.7411** — the *opposite* ordering — and its paired bootstrap calls the V2−vanilla accuracy
   difference **unresolved**: `-0.00083 [-0.0108, +0.0092], excludes_zero: false`.
2. Macro-F1 is omitted. It runs the other way and is significant: ESM-2 0.5162 (CSV) / 0.5186
   (verify JSON) vs V2 0.4941 / 0.4929, with `V2 - ESM-2 macro_f1 = -0.0262 [-0.0450, -0.0071],
   excludes_zero: true`. `RUNS.md:217` is explicit: "When quoting remote homology, give accuracy
   and macro-F1 together, name the probe".
3. Claiming "we report it as measured rather than selecting the favourable probe" in the same
   paragraph that reports the favourable metric and drops the unfavourable one is the exact
   sentence a hostile reader wants.

**Correction.** Add the macro-F1 column (it is in the same CSVs), and downgrade "the ordering
reverses" to what `RUNS.md:193-194` actually supports:

> "Under a linear probe the drop disappears: V2 (0.7503 accuracy) is above V1 (0.7401) and
> level with the untuned backbone (0.7500) — an independent re-measurement puts all three
> within 0.011 and the V2-vs-backbone difference is unresolved. Macro-F1 over 457 classes with
> median support 3 still favours the backbone (0.516 vs 0.494, resolved), and we report both."

---

## Damage rank 7 — Part 3 re-answers three questions we already answered, and loses detail doing it

**Follow-up says:** "These are questions we did not answer in enough detail the first time."

Not true for three of the five. `FINAL_rebuttal.md:200` already gave, in one paragraph:

> "For PPI the two partners are embedded independently and concatenated before the probe.
> Peptide-HLA is not two-input in our implementation: the dataset supplies one sequence field
> holding **a pipe-joined HLA pseudo-sequence and peptide**, so no combination operator is
> used. The k-NN regressor uses uniform weighting over 3 neighbours with Euclidean distance.
> **In the few-shot code the neighbour count is the smaller of 3 and the training size**, so
> the estimator differs in the smallest cells — one reason Table 5 is re-run."

`FINAL_rebuttal.md:141` and `:192` already gave the full CoSENT/DMS answer. The follow-up
version of each is *shorter* and drops two load-bearing details (the pipe-joined field; the
few-shot neighbour count). Re-posting a thinner version of a settled answer reads as padding,
and invites "you said this already, and with more detail."

**Correction.** Cut the k-NN, PPI/peptide-HLA and CoSENT paragraphs down to one sentence that
points back — "Our reply to Yi1G items 4-5 and to jVGf item 4 answered these; we restate only
the corrections" — and keep only the MNRL paragraph (fixed per rank 1) and the positioning
paragraph. Part 3 then does one job instead of five.

---

## Damage rank 8 — Part 3 states specifics about ESM-S, S-PLM, ISM, Magneton and ProTrek

**Follow-up says (Part 3), the sentence to flag:**

> "[...] **those approaches inject structural information into a sequence encoder through
> structure-aware pretraining or distillation objectives**, whereas ProtSent aligns
> sequence-level representations across several relation types [...]"

and, earlier in the same sentence, "The revision discusses ESM-S, S-PLM, ISM and Magneton,
**and ProTrek among retrieval systems**" — which asserts what ProTrek is.

We have run none of these and cite none of them in our own bibliography yet. This is a
four-paper methodological generalisation asserted as fact to a reviewer who supplied the
citations and knows them better than we do (`rebuttal/REVIEWS_actual.md`, jVGf references
[1]-[4]). If any of the four does not fit "structure-aware pretraining or distillation" —
Magneton's title is "Building Substructure into Protein Encoding Models", which is not
obviously either — the sentence is wrong in front of the person who named it.

Note `FINAL_rebuttal.md:135` already committed a similar specific ("inject structure into a
sequence model by distilling a structure encoder or structural tokens: one relation type, one
teacher") plus a forecast ("**we would expect** [ProTrek] to beat a 35M sequence-only encoder
at retrieval"). The follow-up softens the forecast, which is good, but repeats the specifics.

**Correction.** Attribute rather than assert:

> "The revision positions ProtSent against the structure-informed sequence models the reviewer
> cites, and against retrieval systems including ProTrek, on method rather than performance:
> whatever route each takes to structural information, ProtSent's supervision is relational —
> family, structural cluster, physical interaction — over sequence pairs alone, with no
> structural input at training or inference time. We have read them as the reviewer grouped
> them and have not run matched comparisons, so we claim no superiority to any of them and
> will characterise each from its own text in the revision."

---

## Damage rank 9 — the citation answer is wrong and reverses our first response

**Follow-up says (Part 3):**

> "**Missing citation (jVGf).** The broken reference on line 21 is Heinzinger et al.; it is
> fixed, along with the other reference issues noted."

**jVGf actually wrote** (`rebuttal/REVIEWS_actual.md`, minor notes): "There appears to be a
missing reference on line 21 ('?' **between** 'Lin et al., 2023' and 'Henzinger et al.,
2022')". The `?` sits *between* two rendered citations, so it is a third key — not Heinzinger,
who renders fine immediately after it.

**Our first response, `FINAL_rebuttal.md:143`:** "the '?' at line 21 is **a broken citation key
rather than a missing reference**; Heinzinger et al. 2022 and Redl et al. 2023 both appear in
Related Work."

So the follow-up (a) re-labels it "Missing citation" after we told the reviewer it is not one,
and (b) identifies it as the one citation we said was already present. Also "the other
reference issues noted" — jVGf noted exactly one. Small, but it is a wrong statement about a
trivially checkable fact, which is disproportionately expensive.

**Correction.** Cut the item entirely (it was answered) or: "The broken citation key at line
21 is fixed."

---

## Damage rank 10 — "a writing gap, not an experimental one" answers a reviewer who asked for two experiments

**Follow-up says (Part 3):** "We agree this is the main missing piece of context and it is a
writing gap, not an experimental one."

jVGf's questions under that same weakness (`REVIEWS_actual.md`): "How do the results hold up
in absence of both AFDB and Pfam?" and "How do the results change if the ProtSent framework is
applied to a sequence-structure model such as SaProt or ProSST?" Both are experiments. Our own
first response acknowledged the first is outstanding — `FINAL_rebuttal.md:152`: "If the missing
no-AFDB/no-Pfam ablation is the decisive item, please say so and we will report it during the
discussion period."

Telling jVGf the gap is "writing, not experimental" invites "no — I asked for an ablation, and
you offered to run it two weeks ago."

**Correction.** "The positioning itself is a writing task. The two experiments you asked for —
the no-AFDB/no-Pfam ablation and applying ProtSent to SaProt/ProSST — remain outstanding; the
first we can report during discussion if it is decisive, the second is blocked on structure
tokens for the Pfam and STRING corpora."

---

## Damage rank 11 — Part 1 hands the AC a rejection rationale

> "We would make all of those changes in a camera-ready and **we would not object to the AC
> weighing whether that is too much revision to accept now** — that judgement is properly
> theirs."

Honest, and HNXd raised the point, so it must be answered. But as phrased it volunteers
consent. Combine with the preceding sentence, which lists the abstract, the introduction,
three evaluation descriptions and two ablation tables as needing change, and Part 1 reads as a
withdrawal notice.

**Correction.** Keep the deference, remove the invitation: "That judgement is properly the
AC's. We would note only that the changes are subtractive — narrowing claims and correcting
descriptions — and that the experiments supporting the narrowed claim are all in the
submission."

---

## Damage rank 12 — tone tells

Three phrases signal the thing they deny; a reviewer who just raised their score will notice
all three.

| line | phrase | why | fix |
|---|---|---|---|
| Part 2.1 | "we report it as measured rather than selecting the favourable probe" | said in the paragraph that omits macro-F1 (rank 6) | delete; just report both |
| Part 2.1 | "We note two confounds **honestly**" | protesting honesty reads as advertising it | "Two confounds:" |
| Part 1 | "for raising the scope question **openly rather than leaving it implicit**" | slightly patronising to a reviewer who was already direct | "for raising the scope question directly" |

Nothing reads as grovelling. Part 1's "We are grateful for the score increase and for the care
that produced it" is fine and should stay.

---

## Item A — every number in Part 2, verified

### 2.1 SCOPe-40 table (all values = "eligible" metrics, n=1,693 of 2,207)

| doc cell | value | source | verdict |
|---|---|---|---|
| ESM-2 150M | 0.5535 / 0.7702 / 0.4236 | `v2_150m/esm2_150m_linear/bench_Synthyra_ESM2-150M.csv` → 0.55346 / 0.77023 / 0.42359 | ✅ |
| MMseqs2 (-s 7.5) | 0.6556 / 0.7401 / 0.4098 | `scope40_bootstrap_ci_150m.json` marginal MMseqs2 → 0.65564 / 0.74011 / 0.40978 | ✅ |
| HMMER (phmmer) | 0.6970 / 0.7809 / 0.4747 | `hmmer_scope40.json` eligible → 0.69699 / 0.78086 / 0.47466 | ✅ value; ⚠️ flags unstated, see rank 3 |
| ProtSent-V1 150M | 0.6615 / 0.8943 / 0.6431 | `v2_150m/protsent_v1_150m_linear/bench_oriel9p_protsent-esm2-150M.csv` → 0.66155 / 0.89427 / 0.64313 | ✅ |
| ProtSent-V2 150M | 0.7431 / 0.9368 / 0.7042 | `v2_150m/protsent_v2_150m_linear/bench_models_protsent_esm2_150m_v2_final.csv` → 0.74306 / 0.9368 / **0.70458** | ⚠️ MAP |

**MAP nit.** 0.7042 is the bootstrap mean (`scope40_bootstrap_ci_150m.json` → 0.704176); the
CSV point estimate is 0.70458 → **0.7046**, and `alignment_paired_ci_150m.json` gives a third
value, 0.70481. Every other cell in the table is the CSV point estimate. `RUNS.md:136` has the
same 0.7042. Harmless in isolation, but it is the only cell that does not reconcile against the
CSV, so use **0.7046** and be consistent, or footnote that the row is bootstrap means.

Header text: "1,693 of 2,207" ✅ (`n_eligible: 1693`, `n_queries: 2207`, both JSONs and the CSV
columns). "self excluded, no-hit queries counted as failures" ✅ `hmmer_scope40.json` flags
block.

### 2.1 bootstrap intervals

| doc | source | verdict |
|---|---|---|
| "10,000 resamples" | `n_boot: 10000` in both CI JSONs | ✅ |
| V2 − V1 = +0.0809 [+0.0602, +0.1022] | `scope40_bootstrap_ci_150m.json` → 0.080921 [0.060248, 0.102185] | ✅ |
| V2 − HMMER = +0.0455 [+0.0219, +0.0691] | `alignment_paired_ci_150m.json`, `HMMER - ProtSent-V2-150M` hit1 = −0.04549 [−0.06911, −0.02186], sign-flipped | ✅ |
| V2 − MMseqs2 = +0.0868 [+0.0620, +0.1116] | same file / `scope40_bootstrap_ci_150m.json` → −0.086828 [−0.111636, −0.062020] flipped | ✅ |
| "all of these exclude zero" | `excludes_zero: true` on all three | ✅ |
| 35M tie: −0.0124 [−0.0372, +0.0124] | `alignment_paired_ci.json`, `ProtSent-V2 - HMMER / hit1` → −0.012404 [−0.037212, +0.012404], `excludes_zero: false` | ✅ |

### 2.1 remote-homology table

| doc cell | source value | verdict |
|---|---|---|
| ESM-2 150M 3-NN 0.5194 | `esm2_150m_knn/...csv` Accuracy 0.51942 | ✅ |
| V1 3-NN 0.7047 | `protsent_v1_150m_knn/...csv` 0.70469 | ✅ |
| V2 3-NN 0.6612 | `protsent_v2_150m_knn/...csv` 0.66122 | ✅ |
| ESM-2 linear 0.7500 | `esm2_150m_linear/...csv` 0.75 | ✅ |
| V1 linear 0.7401 | `protsent_v1_150m_linear/...csv` 0.74014 | ✅ |
| V2 linear 0.7503 | `protsent_v2_150m_linear/...csv` 0.75031 | ✅ |
| "costs 4.4 points" | 0.70469 − 0.66122 = 0.04347 | ✅ |
| "31% fewer training pairs" | `COAUTHOR_BRIEF.md:101` — 23,897,014 (150M, k=5) vs 34,764,774 (35M, k=8) = −31.3% | ✅ |
| "40% identity / 80% coverage [...] zero flagged sequences" | `RUNS.md:22-33` — 0 surviving in all three parquets | ✅ |
| "V2 differs from V1 in configuration as well as corpus" | `COAUTHOR_BRIEF.md:148-150` lists seven differences | ✅ (understated — see note) |

Note: "differs in configuration as well as corpus" is true but thin. The actual list is
decontaminated corpus, no hard negatives, proportional sampling, **no DMS source**, 7x/6x
larger batch, Matryoshka heads, different data budget. Part 3 discloses the DMS one separately,
so the follow-up is not hiding it — but a reader who only reads Part 2 gets "configuration".

### 2.2 anisotropy table — all twelve cells ✅

Source: `results/benchmarks/probe_gap_analysis.json`, keys `<model>/spectrum_scope40/`.

| model | mean cos | source | PR | source | dims95 | source |
|---|---|---|---|---|---|---|
| ESM-2 35M | 0.848 | 0.8477185 | 7.9 | 7.8828716 | 112 | 112 |
| ProtSent-V2 35M | 0.152 | 0.1521174 | 52.5 | 52.5416857 | 148 | 148 |
| ESM-2 150M | 0.896 | 0.8959996 | 10.6 | 10.6044584 | 126 | 126 |
| ProtSent-V2 150M | 0.175 | 0.1745536 | 43.4 | 43.3596844 | 144 | 144 |

Denominators 480 / 640 = ESM-2 35M / 150M hidden sizes ✅ (`RUNS.md:44` "hidden 640").
"(Σλ)² / Σλ², over the covariance eigenvalues" ✅ matches `probe_gap_analysis.py:99`
(`lam.sum() ** 2 / (lam**2).sum()`, computed on the centered covariance — the doc's wording is
consistent). "2,207-sequence SCOPe-40 gallery" ✅ (`n_domains: 2207` in `embedding_geometry.json`,
same gallery). "0.85–0.90" ✅. "under eleven effective directions" ✅ (7.9 and 10.6). "43–53" ✅.

The one gap: the same JSON carries `standardized_participation_ratio` = **31.3** (35M) and
**32.3** (150M) for the untuned backbones. See rank 4.

---

## Item B — the four code-behaviour claims in Part 3

| claim | code | verdict |
|---|---|---|
| k-NN regression uniform weighting, n_neighbors=3 | `protein_benchmark_suite.py:1556` `KNeighborsRegressor(n_neighbors=3, metric=_KNN_METRIC)` — no `weights` arg, so sklearn default `'uniform'` | ✅ correct |
| minkowski / Euclidean | `protein_benchmark_suite.py:1524` `_KNN_METRIC: str = "minkowski"`, sklearn default `p=2` | ✅ correct, ⚠️ see below |
| PPI embeddings CONCATENATED | `protein_benchmark_suite.py:1438-1440` `np.concatenate([emb_dict[s1], emb_dict[s2]])` for `is_pair` inputs; `benchmark_tasks.py:164-172` `ppi_bernett` has `input_map={"seq1": "SeqA", "seq2": "SeqB"}` | ✅ correct |
| peptide_hla is single-sequence in our implementation | `benchmark_tasks.py:182-190` `peptide_hla` has `input_map={"seq": "seq"}` — one field, so `extract_sequences` takes the single-column branch at `:733` and no pair path runs | ✅ correct, ⚠️ see below |
| `CachedMNRL` `mini_batch_size` does not reduce the negative set | true of `CachedMultipleNegativesRankingLoss`, and true of the V2 runs (`train_esm2_150m.sh:79` `BATCH_SIZE=1024`, `:147` `--mnrl_mini_batch_size 64`) | ⚠️ **true statement, wrong model** — see rank 1 |

Two accuracy caveats worth fixing before posting:

1. **Quote the code as it reads.** The follow-up writes `KNeighborsRegressor(n_neighbors=3,
   metric="minkowski")`. A reviewer opening `protein_benchmark_suite.py:1556` sees
   `metric=_KNN_METRIC`, a module global (`:1524`) overridable by `--knn_metric`
   (`:2506`, `:2681`). Say "the module default `_KNN_METRIC = "minkowski"`, unchanged in every
   reported run" — otherwise it looks like we paraphrased the code in our own favour.
2. **The few-shot estimator is not the same estimator.** `:1578` `n_neighbors = max(1, min(3,
   train_size))`. `FINAL_rebuttal.md:200` disclosed this; the follow-up drops it. Restore it or
   the omission looks deliberate, since Table 5 is the table HNXd challenged.

3. **peptide_hla — keep the pipe detail.** The claim is right, but "that dataset supplies a
   single sequence field" without saying *what is in the field* is a materially incomplete
   answer to someone asking how two proteins are combined. The field is a pipe-joined
   HLA-pseudo-sequence + peptide (`protein_benchmark_suite.py:414`; `FINAL_rebuttal.md:200`).
   Also note `build_comparison.py:400-402`: `'|'` is out-of-vocab for FastPLM and
   `_convert_token_to_id` **raises**, so some arms scored nothing on this task rather than
   scoring badly. If a reviewer finds that after we called peptide-HLA "a single-sequence task",
   it looks like we buried an evaluation failure on a task the paper reports a +3.6% gain for
   (`PAPER_text.txt:266`, `:313`).

Also correct, though unstated: "in its source batch" matches the samplers — the paper's
round-robin (`PAPER_text.txt:209-212`, "the sampler draws a batch from exactly one dataset")
and V2's proportional (`RUNS.md:51`) both yield single-source batches.

---

## Item C — is Part 3's CoSENT/DMS paragraph supported by `protein_pipeline.py`?

Yes, all four assertions, and the DMS-absence disclosure is supported too.

| claim | source | verdict |
|---|---|---|
| "Each training row is a (wild type, mutant) pair" | `data_prep.py:2073-2090` docstring: "Writes dms_cosent.parquet with paired sentence_0, sentence_1, score rows [...] higher score means the mutant should embed closer to target_seq" | ✅ |
| "carrying a within-assay normalised fitness score" | `data_prep.py:2412-2418` per-assay Z-score over `DMS_id`; docstring `:2086` "within-assay normalized DMS_score scaled to [0,1]" | ✅ |
| "CoSENT is an ordinal objective over pairs [...] no absolute target similarity" | `protein_pipeline.py:2929`/`:2935` `losses.CoSENTLoss(model, scale=mnrl_scale)`; behaviour is the library's, and it matches `FINAL_rebuttal.md:141` | ✅ (library behaviour, not our code — fine to assert, we cite Su 2022 in the paper) |
| "the configuration is WT-anchored, so mutant–mutant geometry is not directly optimised" | `data_prep.py:2075` `intra_pairs: bool = False` is the default and the mutant–mutant path (`--dms_intra_pairs`, `:2941`) appears in no training script | ✅ |
| "the decontaminated retrains **do not include the DMS source**" | `COAUTHOR_BRIEF.md:118-128`: "Both V2 scripts pass three files and contain no CoSENT or ProteinGym path"; `RUNS.md:22-31` lists three parquets | ✅ |
| "ProtSent-V2 is a three-source model where the submitted ProtSent is four-source" | `COAUTHOR_BRIEF.md:120` | ✅ |

One optional strengthening: because normalisation is per-assay z-score but CoSENT ranks all
pairs *within a batch*, batches mix assays and the ordering constraint is applied across
assays. That is defensible (z-scoring is what makes them comparable) but it is the obvious
next question from a reviewer who is already suspicious of the DMS objective. One clause
pre-empts it.

Attribution ✅: the CoSENT question is jVGf's ("How exactly does the CoSENT loss for DMS data
work?", `REVIEWS_actual.md` jVGf questions), matching `FINAL_rebuttal.md:139` "(Q4)". Yi1G
raised the related biological-assumption point separately, which `FINAL_rebuttal.md:190`
already handled — no need to re-open it.

---

## Item D — contradictions with `FINAL_rebuttal.md`

**Flagged and handled correctly:** the top-1/alignment correction. `FINAL_rebuttal.md:39`
"Alignment remains the better method at top-1" and `:214` "Alignment remains better at top-1"
→ follow-up Part 2.1 explicitly corrects it. Well done, and the 35M number it cites is right.

**Unflagged contradictions found — three:**

1. **MNRL negative set.** `FINAL_rebuttal.md:196` (64/16 negatives, "the reviewer is right")
   vs Part 3 (1,023 negatives, "our use of 'effective batch size' conflated the two"). **Rank
   1.** This is the one that must be fixed before posting.
2. **"every correction [...] found and reported by us rather than by a reviewer"** vs
   `FINAL_rebuttal.md:196`, `:158`, `:141`, `:220`. **Rank 2.**
3. **Citation at line 21.** `FINAL_rebuttal.md:143` "a broken citation key **rather than a
   missing reference**; Heinzinger et al. 2022 [...] appear[s] in Related Work" vs Part 3
   "**Missing citation** [...] The broken reference on line 21 **is** Heinzinger et al."
   **Rank 9.**

**Softenings, not contradictions** (fine, but know they are visible): Part 3 drops the
few-shot `min(3, train_size)` caveat and the pipe-joined peptide-HLA detail, both of which
`FINAL_rebuttal.md:200` gave; and it replaces `FINAL_rebuttal.md:135`'s "we would expect
[ProTrek] to beat a 35M sequence-only encoder at retrieval" with the neutral "we claim no
superiority to any of them", which is an improvement.

**Not a contradiction but worth knowing:** `FINAL_rebuttal.md:115` and `:121` and `:214` lean
on "691 of the 2,207 queries return no phmmer hit at all" as part of the depth argument.
`NEW_EVIDENCE.md:524-526` withdraws exactly that: "**The coverage-gap argument in 5a is
withdrawn.** The 691 no-hit queries were an artifact of default heuristic filters [...] At full
sensitivity HMMER returns a ranked list for every query. Do not use." That retraction is in the
public repo and the posted first response is not. The follow-up is the natural place to correct
it, and correcting it voluntarily would substantially offset rank 3 — it is the same underlying
fact, and disclosing it ourselves is much cheaper than having jVGf find `hmmer_maxsens.json`.

---

## Item E — sentences asserting specifics about ESM-S / S-PLM / ISM / Magneton / ProTrek

One sentence, flagged in full at rank 8:

> "the revision discusses ESM-S, S-PLM, ISM and Magneton, and ProTrek among retrieval systems,
> and distinguishes them from ProtSent on method rather than on performance: **those approaches
> inject structural information into a sequence encoder through structure-aware pretraining or
> distillation objectives**, whereas ProtSent [...]"

Two assertions: the mechanism attributed to four papers, and ProTrek's categorisation.
Everything after "whereas" is about ProtSent and is safe. "We claim no superiority to any of
them; we have not run matched comparisons, and we say so rather than implying a ranking" is
safe and should stay.

No other sentence in the follow-up names them.

---

## Item F — tone

Covered at ranks 2 (boastful), 6 (self-praise adjacent to an omission), 7 (re-arguing settled
points), 11 (over-concession) and 12 (phrasing tells). Summary of what to cut:

- Boastful: "every correction in it was found and reported by us rather than by a reviewer".
- Self-congratulatory: "we report it as measured rather than selecting the favourable probe";
  "We note two confounds honestly".
- Re-arguing settled points: the k-NN weighting, PPI/peptide-HLA and CoSENT paragraphs
  (answered at `FINAL_rebuttal.md:141`, `:192`, `:200`); the line-21 citation (answered at
  `:143`); arguably all of Part 2.2, since per `ADDITIONS_to_rebuttal_docx.md:11-14` HNXd has
  already accepted that the SCOPe-40 analyses address embedding-space organization.
- Over-conceding: "we would not object to the AC weighing whether that is too much revision".

Nothing grovels. Part 1 is otherwise the strongest section in the document — it answers the
scope question head-on and it is the answer HNXd actually needs.

---

## Item G — what a hostile reader quotes

Ordered by how bad it is when quoted. Everything below is reachable from
`github.com/oriel9p/ProtSent` branch `rebuttal`, which we told the reviewers about.

1. `results/benchmarks/hmmer_maxsens.json` → phmmer R@1 **0.7525** vs our 0.7431, against Part
   2.1's "exceeds HMMER at top-1 significantly" — plus `NEW_EVIDENCE.md:549` "**Any rebuttal
   sentence comparing to HMMER must use the filters-off numbers**", in our own bold.
2. `results/benchmarks/probe_gap_analysis.json` → whitened ESM-2-150M 3-NN **0.7346** beats
   ProtSent-V2-150M **0.6606**, against Part 2.2's "a trained linear head can compensate [...]
   and k-NN cannot" and Part 2.1's "both ProtSent models remain far above the untuned backbone".
3. `whiten_scope_control.py:6-9` → "**That is a deflating result for the method.**"
4. `ADDITIONS_to_rebuttal_docx.md:102` → "**We judged it out of scope for the rebuttal**" about
   the control in (2)/(3).
5. `ADDITIONS_to_rebuttal_docx.md:21-25` → "**the cheapest remaining score movement**".
6. `rebuttal/final_review_hostile.md:16` → "**KILL SHOT**".
7. `results/benchmarks/whiten_scope_control.json` → `ProtSent-V2-150M [raw] - ESM-2-150M
   [whitened]` hit1 **+0.0089 [-0.0106, +0.0289], excludes_zero: false** — the 150M top-1
   claim, erased by whitening the baseline.
8. `results/benchmarks/verify_remote_homology_150m.json` → ESM-2 0.7506 > V2 0.7497, the
   opposite of Part 2.1's "the ordering reverses", plus the resolved macro-F1 deficit.
9. `FINAL_rebuttal.md:196` vs Part 3 → the MNRL retraction, quotable as either "they changed
   their story" or "they walked back a concession".
10. `build_comparison.py:400` → `KeyError: '|'` on peptide_hla, against Part 3's clean
    "single sequence field" answer.
11. `NEW_EVIDENCE.md:524` → the withdrawn 691-no-hit coverage argument, which is still live in
    the posted first response at `FINAL_rebuttal.md:115`, `:121`, `:214`.

Items 4-6 are free to fix (rank 5: clean the branch). Items 1-3, 7-8 and 11 are best fixed by
volunteering them — every one of them is survivable if we report it and fatal if they find it.

---

## Character counts

Body text only: the `## Part N` headings and the `*(~N characters)*` italic notes excluded;
newlines and spaces counted, as OpenReview counts them.

| part | lines | chars (incl. `###` subheads) | chars (subheads also removed) | doc claims | delta |
|---|---|---|---|---|---|
| Part 1 | 12-38 | **1,864** | 1,864 | ~2,050 | −186 |
| Part 2 | 46-120 | **3,958** | 3,841 | ~4,900 | −942 |
| Part 3 | 128-180 | **3,786** | 3,786 | ~3,600 | +186 |
| total | | **9,608** | 9,491 | ~10,550 | −942 |

All three notes are wrong; Part 2's is out by 24%. If the counts are there to decide whether a
part fits a length limit, fix them — Part 3 is the one that is *longer* than advertised, and
it is the part that needs to shrink anyway (rank 7).

---

## Minimum change set before posting

1. Rewrite the MNRL paragraph to separate submitted-model from retrain (rank 1). **Blocking.**
2. Delete "every correction [...] rather than by a reviewer" (rank 2). **Blocking.**
3. Either drop the "exceeds HMMER at top-1" upgrade or state the flags and the filters-off
   number (rank 3). **Blocking.**
4. Delete "and k-NN cannot"; volunteer the whitening control (rank 4). **Strongly recommended.**
5. `git rm` the strategy memos and simulated hostile reviews from `origin/rebuttal` (rank 5).
   **Do this regardless of what gets posted.**
6. Add the macro-F1 column; soften "the ordering reverses" (rank 6).
7. Cut the three already-answered Part 3 paragraphs to a back-reference; fix the citation item
   (ranks 7, 9).
8. Attribute rather than assert the structure-model characterisation (rank 8).
9. Fix the three character-count notes.

`ADDITIONS_to_rebuttal_docx.md` itself is accurate throughout — every claim in sections A-E
reconciles against `RUNS.md`, `COAUTHOR_BRIEF.md` and the JSONs, including the 31%/23.9M-vs-34.8M
pair budget, the three-source V2, the k=5/k=8 difference, and the whitening note at D. Its only
problem is that it is published (rank 5).
