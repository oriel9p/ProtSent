# PI review of the ProtSent rebuttal — design notes

Read: `rebuttal/PAPER_text.txt`, `rebuttal/NEW_EVIDENCE.md`, `rebuttal/DRAFT_rebuttal_edited.md`,
`REBUTTAL_LEAKAGE.md`, `results/benchmarks/COMPARISON.md`.

Bottom line first: the draft is written as a **defense of the submitted paper**. That paper
cannot be defended — its headline claim dies at the linear probe and its biggest numbers
(150M, +105%) come from an uncontrolled corpus. The rebuttal has to be written as a
**re-scoping of the paper around a claim that got stronger during the rebuttal period**.
That claim exists, it is measured, and it is not in the submitted PDF.

---

## 1. The defensible one-sentence contribution

> Contrastive fine-tuning of a 35M-parameter ESM-2 on multi-relational protein pairs
> (family / structural cluster / interaction / fitness) produces a sequence-only embedding
> whose nearest-neighbour structure is substantially better for homology and structural
> retrieval — after removing every pretraining sequence within 40% identity / 80% coverage
> of the test sets, it beats a tuned MMseqs2 at every cutoff on SCOPe-40 family retrieval
> (R@1 0.5256 vs 0.5029, R@10 0.7073 vs 0.5637, MAP 0.4955 vs 0.3100) and beats stock
> ESM-2 on remote homology under **both** a 3-NN and a linear probe (0.5835 → 0.6668 kNN,
> 0.6868 → 0.7016 linear) — while general property-prediction performance under a trained
> linear readout is unchanged to slightly worse.

Two things about that sentence:

- **The last clause is not optional.** It is the price of the first clause. A skeptical AC
  who reads the linear-probe table (V2: 2 win / 7 tie / 11 lose vs ESM-2, median −0.0107)
  and finds you did not say it will assume you hid it. Say it in your own words first.
- **It is a retrieval paper now, not a general-purpose embedding paper.** The abstract's
  "general-purpose embedding models" framing and the "improves 16 of 23 tasks" headline
  do not survive §6 of NEW_EVIDENCE. Retire them.

Secondary contribution, real and worth one sentence: a **negative result about what
"benchmark leakage" means for pLMs** — SCOPe-40 has median max-identity 0.89 to any
comprehensive corpus and *no* sequence below 20%, so it cannot be decontaminated by anyone,
and the correct measurement is the identity-stratified model-vs-model delta. You have that
delta and its sign is the opposite of what memorization predicts. Most papers assert this;
you measured it.

---

## 2. The narrative that could move them

**Yes, the re-filter-and-retrain is the engine — but it is not the headline. Do not lead
with it.**

Reviewers do not raise scores for diligence. They raise scores when a claim they can
believe replaces a claim they could not. "We took your accusation seriously and did 240M
rows of work" is a *character* argument; three reviewers who voted 2 will read it as
"they worked hard and the paper is still what it was."

The narrative that flips is one step further:

> **We ran the experiment that could have killed the paper. It didn't kill it — it
> re-scoped it, and at the smaller scope the result is stronger than what we submitted.**

Concretely, in this order:
1. **The claim changed.** The general-purpose claim is withdrawn (linear probe). The
   retrieval/neighbourhood claim survives both probes and is now the paper.
2. **The claim is now bigger than the submitted one, on its own turf.** The submitted model
   *loses* to a tuned MMseqs2 at R@1 (0.4490 vs 0.5029). The decontaminated retrain wins at
   every cutoff. A 35M sequence-only model beating tuned alignment search on SCOPe-40 is a
   result the paper did not have and cannot be dismissed as a probe artifact.
3. **Why you can believe it.** Corpus filtered at 40%/80%, verified at 0 surviving flagged
   sequences in the exact training parquets, row arithmetic closing to the training log;
   the gain is *larger* in the low-identity bins and the identity-vs-gain Spearman is
   negative; a real MMseqs2 baseline with flags stated; both probes reported.
4. **What we got wrong.** Four disclosed errors, unprompted.

The decontamination lives at step 3. That is where it does its work: it converts step 2
from "a claim" into "a controlled claim". Promoted to step 1 it reads as a plea.

**Narratives to reject:**
- *"Look at all the experiments we ran."* Volume gets discounted. Three reviewers, eight
  items, everything runs together.
- *"The reviewers misread the CoSENT objective / MNRL batch."* You are right on both, and
  both are one-paragraph items. Leading with a correction of the reviewer is the single
  fastest way to keep a 2.
- *"Our claims were right."* Contradicted by your own linear table. An AC who later sees
  that table will treat the rebuttal as evidence of bad faith, which is worse than a
  reject.

---

## 3. First three sentences of each response

Page 1 is 80% of the paper; paragraph 1 is 80% of the rebuttal. Each opening must be
readable standalone **by the AC**, and all three must tell the identical story. Draft them
verbatim, then write the rest.

### HNXd — lead with the number that hurts you, because he asked for it

> You were right that the 3-NN probe was carrying the result. We have now run frozen
> logistic-regression / ridge probes on all 23 tasks on the **test** split for stock ESM-2
> 35M and ProtSent: over the 20 tasks comparable in both arms, ProtSent-V1 (the submitted
> model) is 11 win / 3 tie / 6 lose under 3-NN (median delta +0.0075) and **4 win / 4 tie /
> 12 lose under a linear probe (median delta −0.0139)**. We therefore withdraw the
> general-purpose superiority claim; what survives both probes is the retrieval and
> homology result — SCOPe-40 family Recall@10 0.5840 (ESM-2) → 0.6529 (V1) → 0.7073 (V2,
> decontaminated retrain), and remote-homology accuracy 0.5835 → 0.6587 → 0.6668 under
> 3-NN and 0.6868 → 0.6899 → 0.7016 under a linear probe.

Handing a reviewer the negative result he predicted, unprompted, in sentence two, is the
highest-leverage move available in this entire rebuttal. He cannot then accuse you of
spinning anything else.

### jVGf — lead with the trade-off curve, with ProtSent's position on it

> You asked where ProtSent sits on the generality–accuracy trade-off; we measured it rather
> than argued it. Scoring MMseqs2 (`-s 7.5 -e 10`, no-hit queries counted as failures) under
> the *same* metric definitions across the benchmark, **alignment beats the best embedding
> model outright on 3 tasks under a 3-NN probe and 6 under a linear probe** (EC F1-macro
> 0.710 vs 0.598/0.562; GO-MF 0.585 vs 0.459/0.443) — and ProtSent-V2 beats the same tuned
> MMseqs2 at every SCOPe-40 cutoff (R@1 0.5256 vs 0.5029, R@10 0.7073 vs 0.5637, MAP 0.4955
> vs 0.3100). On your first question, structural supervision is a large part but not all of
> the effect: the submitted ablation drops remote homology from +40.5% to +15.3% without
> AFDB while STRING removal is what collapses PPI (+5.3% → −0.5%), i.e. each source leaves
> a distinct fingerprint on a distinct task family.

### Yi1G — lead with the control, then the confound, immediately

> All three pretraining corpora were re-filtered against the benchmark test sets at 40%
> identity / 80% coverage of the test sequence and the model retrained from scratch:
> Pfam 28,530,684 → 27,929,772, AFDB 135,404,259 → 126,301,607, STRING 76,070,154 →
> 71,891,417, with 0 flagged sequences surviving in the three parquet files training
> actually opened, negative controls at 0 hits and positive controls at 3,244/3,244 and
> 3,022/3,022. **On the task the corpus was filtered against, performance went up**: remote
> homology 0.6587 → 0.6668 (3-NN) and 0.6899 → 0.7016 (linear), SCOPe-40 Recall@10 0.6529 →
> 0.7073. We state the confound in the same breath: the retrain also changed the effective
> batch (7×1024 vs 1×1024), dropped synthetic hard negatives, used proportional sampling and
> one epoch, so the *only* claim we make is the one you asked about — removing the
> 40%/80% overlap does not remove the gain.

---

## 4. Concessions: loud and early vs. needless

### Concede loudly, in the first 1,500 characters

1. **The linear probe kills the general-purpose claim.** (HNXd sentence 2; one clause in the
   other two.) Buys everything else.
2. **The submitted model loses to tuned MMseqs2 at R@1** (0.4490 vs 0.5029). Concede for V1
   in all three responses, claim the win for V2 in all three. Never blur which model.
3. **"100,000 sequences at the superfamily level" is wrong** — 2,207 sequences, family field;
   the 100,000 is the evaluator's `max_samples` cap echoed into the table. Disclosing it
   yourself converts an apparent 45× misreport into a logging artifact. Let a reviewer find
   it and it is a fabrication accusation.
4. **The remote-homology split is not hierarchy-disjoint** — pooled TAPE holdouts
   (718 + 1,254 + 1,272 = 3,244), no column marking which, and the pooled 457-class macro
   AUC is not comparable to published TAPE per-holdout accuracies.
5. **The PPI decontamination text does not match the released code** (`easy-search` at 40%
   with hit-ID removal, not `easy-linclust` at 50% with cluster removal).
6. **Eq. 1 is malformed.** One sentence, no defense.
7. **The ablations do not establish the submitted configuration** — no-hard-negatives
   improves 20/23 vs the full model's 16/23. Say the submitted config is not validated as
   optimal and drop the claim, rather than explaining it away.
8. **V2 ≠ decontamination-only.** Four changes, named, in the same sentence as the result.

These eight cost you nothing you can keep anyway, and they are the entire basis on which
the reviewer will believe items you *do* claim.

### Being conceded needlessly in the current draft — cut

- **The n=155 / n=92 query-filtered SCOPe analysis** (draft Yi1G §1). Weak, superseded, and
  actively harmful: 92 queries with a 57/92 = 0.620 ceiling, R@1 ties. You now have a
  2,207-query corpus-level control plus identity stratification. Delete the subset tables
  entirely; keeping them invites "your top-1 does not survive decontamination" when the
  strong answer says the opposite.
- **All `[[RESULT: paired bootstrap …]]` placeholders.** NEW_EVIDENCE §8 says these do not
  exist. Do not promise them. Replace with a retraction that is deliverable today: *sub-1%
  Table 2 deltas are withdrawn as evidence; we report absolute deltas and the two effect
  sizes that are not in dispute (median +0.0075 across tasks vs +0.1232 on SCOPe-40 R@10 —
  two orders of magnitude apart).* A retraction beats a promised CI.
- **`[[RESULT: 5- or 10-seed few-shot …]]`.** Same. No seeds exist. Retract Table 5's
  relative-percentage framing outright — say the −126.9% class of cell is an artifact of
  relative change against a near-zero baseline and should never have been printed, and that
  Table 5 is withdrawn pending absolute multi-seed reporting. Reviewers accept a withdrawn
  table; they do not accept an unfulfilled promise.
- **"If the linear probes do not support the label-scarcity claim, we narrow that claim."**
  The probes are done. Conditional hedging about a completed experiment reads as evasion.
  State the outcome.
- **"the downstream split is disjoint in its evaluation hierarchy"** (draft Yi1G §1). This is
  not a concession, it is a false statement — see concession 4 above. Fix, don't soften.
- **Repeating "we claim no superiority to ProtTucker/Foldseek/PLMSearch/DHR/ProTrek" three
  times per response.** Once, tersely, per response.

---

## 5. Where this loses, and the pre-emptive move

**Loss 1 — the AC reads your own linear table as a negative result.** 12/20 losses is a
paper-killer if it arrives unframed.
*Pre-empt:* frame it before they compute it, and frame it as a **measurement of a different
quantity**, not as an invalid probe. The line: a linear head measures how much task signal
can be extracted from residue-averaged features by supervised adaptation; 3-NN measures
whether the relation is already local in the frozen space. ProtSent optimizes the second.
Reinforce with the fact that MMseqs2 also beats every embedding model on EC/GO-MF under a
linear probe — those tasks are homology-transfer tasks, not representation tasks. Never
argue the linear probe is wrong; commit to reporting both permanently.

**Loss 2 — "you changed four things and credited decontamination."**
*Pre-empt:* the confound sentence goes in the same sentence as the headline number, in all
three responses (see §3 Yi1G). Claim only "removing the overlap did not cost performance."

**Loss 3 — "35M only; where are the real baselines?"**
*Pre-empt:* state the absences as a list, with reasons, before anyone asks — no 150M
decontaminated model, no matched ProtTucker/Foldseek/PLMSearch/DHR/ProTrek (checkpoint
network-blocked, and no published number is protocol-comparable), no SaProt/ProSST
substitution (needs residue-level structure tokens for the full Pfam and STRING corpora),
no end-to-end fine-tuning sweep. Then lead the one strong baseline you *do* have, with its
flags. Also state the **recall ceiling of 0.7671** (only 1,693 of 2,207 queries have any
non-self same-family neighbour) — without it, R@10 0.7073 reads as mediocre; with it, it is
92% of achievable.

**Loss 4 — a number in the rebuttal that a reviewer cannot reconcile.** This is the live
danger in the current draft and it is severe:
- The MMseqs2 table (R@1 0.3539, MAP 0.1795, and R@10 = R@30 = 0.3856 to four decimals) is a
  default-sensitivity run with a truncated hit list. The released repo contains a
  reproducible 0.5029/0.3100 at `-s 7.5`. **Publishing the weakest possible baseline while a
  stronger one ships in your own repo is a self-inflicted integrity finding.** Replace it.
- The geometry paragraph (silhouette −0.148 → 0.039, NMI 0.852 → 0.893, ARI 0.165 → 0.313,
  intra/inter 0.701 → 0.418, hierarchy-depth Spearman −0.247 → −0.561) **is not in
  NEW_EVIDENCE.md or REBUTTAL_LEAKAGE.md.** Either it gets verified into NEW_EVIDENCE before
  posting, or it comes out. It is the answer to HNXd's first request, so verifying it is
  high value — but citing it unverified violates the project's own rule and is exactly the
  kind of number a hostile reviewer will ask to see.
- The "reproduced" 150M/35M retrieval table in the draft (0.4237/0.5908/…) matches neither
  the paper's Table 3 nor NEW_EVIDENCE §3. Pick one provenance.
*Pre-empt:* one canonical four-arm SCOPe table (MMseqs2 / ESM-2 35M / V1 / V2), pasted
**identically** into all three responses, sourced only from NEW_EVIDENCE §3.

**Loss 5 — the three responses disagree.** Reviewers read each other's threads; the AC reads
all three. The R@1-vs-MMseqs2 question and the "does ProtSent beat ESM-2" question must have
one answer, word for word, in all three.

**Loss 6 — the 10,000-char limit.** With the new material the draft will overflow. Budget:
~2,500 chars of canonical opening + shared table (identical across responses), ~7,000 of
reviewer-specific. Cutting the n=92 analysis and the CI/seed promises pays for most of it.

**Loss 7 — HNXd's two specific asks (CIs, seed variability) go unanswered.** He named them as
his reconsideration criteria. You cannot deliver either.
*Pre-empt:* convert both from "we will measure" to "we withdraw" (see §4). Then spend the
recovered space on the geometry analysis, which is his *first* ask and which you can
actually deliver if it gets verified.

---

## 6. The 150M decontaminated model

**Do not promise it. Do not imply it is running. Use its absence as a scoping statement.**

Concretely:
- Never write "the 150M retrain is in progress and will appear in the camera-ready."
  Reviewers discount promises to zero, and ACs treat an unfulfilled promise as a reason to
  reject rather than a reason to wait.
- Instead, state once per response, in the scope/limitations block: *"The decontaminated
  retrain exists at 35M only. Every claim in this response is therefore a 35M claim; the
  150M results in the submitted paper were trained on the uncontrolled corpus and we are not
  defending them here."*

That last clause is the important one and it has a real cost you should accept consciously:
**it retires the paper's biggest headline numbers** (+105% remote homology, +19.9% R@1 at
150M). Take the hit. Those numbers are exactly the ones Yi1G's leakage objection targets,
they are the ones you cannot currently control, and defending them is what makes the whole
response look like advocacy. The 35M V2 story is smaller, fully controlled, and beats a
tuned alignment baseline — that is a paper. The 150M story is bigger and undefended — that
is a reject.

Corollary for the revision: **rewrite the abstract at 35M, retrieval-first.** Something in
the shape of *"contrastive fine-tuning turns a 35M sequence-only pLM into a homology-search
model competitive with tuned MMseqs2, at the cost of no improvement under a trained linear
readout"* — a smaller, sharper, defensible paper.

---

## Action list, ranked

1. Verify or delete the geometry metrics (silhouette/NMI/ARI/hierarchy-depth). Blocker for
   HNXd's headline ask.
2. Replace every MMseqs2 number in the draft with the `-s 7.5` run and state the flags.
3. Delete the n=155 / n=92 subset analysis and all CI/seed placeholders; replace with
   explicit withdrawals of the sub-1% Table 2 deltas and of Table 5's relative framing.
4. Write one canonical opening paragraph + one canonical four-arm SCOPe table; paste
   identically into all three responses; then write reviewer-specific bodies around it.
5. Insert the four proactive error disclosures and the V2 confound sentence before any
   reviewer-specific argument.
6. Add the explicit "not available" list, including the 150M scoping statement, once per
   response.
7. Character-count each response before posting.
