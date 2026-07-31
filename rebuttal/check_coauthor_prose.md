# Prose check: COAUTHOR_BRIEF.md

Baseline: 18,632 chars / 2,847 words / 365 lines. Tables are 3,274 chars of that and are
untouched, so the editable prose is ~15,358 chars.

**Headline: the 15–20% suspicion is wrong.** The zero-information-loss ceiling is
**1,222 chars (6.6% of the file, ~8% of the prose)**. The vocabulary tells are simply not
present — `crucially`, `importantly`, `it is worth noting`, `comprehensive`, `robust`,
`leverages`, `delve`, `landscape`, `underscores`, `furthermore`, `moreover`, `notably`,
`seamless`, `nuanced`, `myriad`, `paradigm` all return zero hits. What is there is a
smaller, structural set: sentences that announce the next sentence, three places where a
table is restated in prose immediately below it, and one section (§5.1) whose opening
paragraph is a third statement of a finding already made in §1 and §3.

Applying all 20 edits: **2,643 words** (down from 2,847, −204 words).

---

## Edits, biggest saving first

### 1. −317 (total 317) — L234–240, §5.1: opening paragraph is the third statement of the same finding

The claim already appears in §1 item 1 ("does not survive a linear probe … neutral-to-worse
than the stock backbone across 23 tasks") and again in §3 ("**The probe decides the
headline.**"). The §5.1 heading already says it a fourth time. Then
"There is a legitimate, measured mitigation." announces the sentence after it.

**Delete:**
> Across 23 tasks, both ProtSent models are neutral-to-worse than the stock backbone under a
> trained linear readout, while winning under 3-NN. This is the single most important honesty
> constraint on anything we write.
>
> There is a legitimate, measured mitigation. **Both probes pool the final layer**, which is
> the measurement point most favourable to a model whose top of stack was never touched — the
> contrastive objective only ever sees the final layer.

**Replace with:**
> **Both probes pool the final layer**, which favours the stock backbone — contrastive
> training only ever reshaped ProtSent's final layer.

The rewritten clause also fixes a real readability bug: "the measurement point most
favourable to a model whose top of stack was never touched" takes two passes to work out
that the model in question is stock ESM-2, not ProtSent.

### 2. −110 (total 427) — L3–5, intro: first sentence tells the reader who they are

**Delete:**
> Written for a coauthor who knows we wrote a rebuttal and retrained decontaminated V2
> models, and needs the rest. Everything here is measured on our hardware and reproducible
> from the repo; paths are given so you can check any number yourself.

**Replace with:**
> Every number here is measured on our hardware and reproducible from the repo; paths are
> given so you can check any of them yourself.

### 3. −94 (total 521) — L306–309, §6 item 1: 34-word restatement of §1's closing line

§1 already closes with "ProtSent is a retrieval / metric-space method, not a general-purpose
embedding upgrade." §6's job is the action, not the re-argument. "strong, defensible, and
survives every control" is a triad where the third member is the only measured one.

**Delete:**
> The evidence supports a retrieval / metric-space
>    contribution: a sequence-only embedding whose nearest-neighbour structure is
>    substantially better for homology and structural retrieval, and which is neutral under a
>    trained readout. That claim is strong, defensible, and survives every control we ran.

**Replace with:**
> Claim a retrieval / metric-space contribution instead:
>    nearest-neighbour structure substantially better for homology and structural retrieval,
>    neutral under a trained readout. It survives every control we ran.

### 4. −87 (total 608) — L290–291, §5.2: recommendation stated in three places

The whitened-baseline recommendation appears here, again two sentences later ("goes to you
and into the paper revision"), and again as §6 item 3. §6 is its home.

**Delete (whole sentence):**
> A whitened-baseline
> control is, in my view, something the camera-ready should contain.

### 5. −69 (total 677) — L299–300, §5.3: pure cross-reference

"Real, expected, and an argument for the filtering" is the third time this is said (§1 item
3: "evidence the leakage was real and the filtering works"; §3: "That is an argument *for*
the decontamination"). Only the presentation instruction is new.

**Delete:**
> Covered in §3. Real, expected, and an argument for the filtering — but it must be presented
> with the linear-probe reversal alongside it, or it reads as a straight regression.

**Replace with:**
> See §3. Always present it with the linear-probe reversal alongside, or it reads as a
> straight regression.

*Aggressive variant (−185 instead of −69):* delete §5.3 and its heading entirely, move the
presentation instruction into §3 next to "At 35M, decontamination **improved** this task",
and retitle §5 "The uncomfortable findings, in detail". Costs the 1/2/3 symmetry with §1;
not counted in the running total.

### 6. −66 (total 743) — L293–295, §5.2: double hedge on a decision already made

**Delete:**
> Current decision: this stays **out of the rebuttal** (no reviewer asked, and it opens a new
> front) and goes to you and into the paper revision. I think that is defensible; I also
> think it is the cheapest experiment a skeptical reader could run against us.

**Replace with:**
> Current decision: **out of the rebuttal** (no reviewer asked, and it opens a new front),
> into the paper revision. It is also the cheapest experiment a skeptical reader could run
> against us.

"I think that is defensible" hedges a decision you have already taken; "I also think it is"
hedges a factual claim about experiment cost.

### 7. −61 (total 804) — L267–268, §5.2: restates the table directly above it

The table two lines up gives 0.848 and 0.896; "cosine ~0.85–0.90" is those numbers rounded.

**Delete:**
> Random protein pairs in the stock model have cosine ~0.85–0.90 — the space is a narrow
> cone. ProtSent removes that almost entirely.

**Replace with:**
> The stock space is a narrow cone; ProtSent removes it almost entirely.

### 8. −59 (total 863) — L130–132, §3: restates the table, plus "because it is true"

Both numbers are in the table 20 lines above (HMMER 0.6970, MMseqs2 0.6556) and are
retained there. "worth conceding, because it is true and it makes the rest credible" — the
first reason is not a reason.

**Delete:**
> We ran HMMER because Yi1G named it. It is the stronger baseline (R@1 0.6970 vs MMseqs2's
> 0.6556) and it beats vanilla ESM-2 150M at top-1 by +0.144 — worth conceding, because it
> is true and it makes the rest credible.

**Replace with:**
> We ran HMMER because Yi1G named it. It is the stronger baseline and beats vanilla ESM-2
> 150M at top-1 by +0.144 — worth conceding; it makes the rest credible.

*If you would rather not rely on the reader's eye travelling back to the table, keep the
parenthetical; the edit is then −33.*

### 9. −44 (total 907) — L84–85, §2: defensive lead-in

**Delete:**
> The two config changes are not arbitrary and are worth saying out loud in the rebuttal:
> dropping hard negatives and using proportional sampling are

**Replace with:**
> Say both config changes out loud in the rebuttal: dropping hard negatives and
> proportional sampling are

### 10. −40 (total 947) — L311, §6 item 2: dead closer

**Delete:**
> That is a clean, quantitative headline.

The preceding sentence ("It beats both alignment baselines, including HMMER at top-1, with
paired CIs") already establishes that it is clean and quantitative.

### 11. −37 (total 984) — L258, §5.2: repeats §1 item 2 verbatim

§1 item 2 already says "Stock ESM-2 embeddings are severely anisotropic".

**Delete:**
> Stock ESM-2 embeddings are severely anisotropic on the SCOPe gallery:

**Replace with:**
> Anisotropy on the SCOPe gallery:

### 12. −35 (total 1,019) — L152–157, §3: 57-word sentence, split into three

**Delete:**
> We independently re-derived the 150M numbers (`verify_remote_homology.py`) because the
> linear macro-F1 deficit against vanilla looked suspicious. Outcome: it reproduces, it is
> statistically real (V2 − vanilla macro-F1 −0.0262 [−0.0450, −0.0071]; accuracy unresolved
> at −0.0008), **but it is mostly a rare-class artifact** — the test set has 457 classes with
> median support 3 and 209 classes with ≤2 examples, and restricting to classes with ≥3 test
> examples shrinks the gap from −0.0257 to −0.0036.

**Replace with:**
> We re-derived the 150M numbers (`verify_remote_homology.py`) because the linear macro-F1
> deficit against vanilla looked suspicious. It reproduces and is statistically real (V2 −
> vanilla macro-F1 −0.0262 [−0.0450, −0.0071]; accuracy unresolved at −0.0008). But it is
> mostly a rare-class artifact: the test set has 457 classes, median support 3, 209 classes
> with ≤2 examples, and restricting to classes with ≥3 test examples shrinks the gap from
> −0.0257 to −0.0036.

All eight numbers retained. "independently" is doing no work next to "we re-derived".

### 13. −34 (total 1,053) — L19, §1: sentence announcing the next sentence

**Delete:**
> We answered it the expensive way.

The next sentence ("re-filtered the entire pretraining corpus … retrained both models from
scratch") conveys "expensive" without being told to.

### 14. −32 (total 1,085) — L196–197, §4: meta-commentary on the bullet list below it

**Delete:**
> Two halves, and
> quoting only one would misrepresent it:

**Replace with:**
> Two halves; quote both:

### 15. −31 (total 1,116) — L181–182, §3: self-congratulation

**Delete:**
> This is the generality-accuracy trade-off jVGf asked for, measured rather than
> asserted.

**Replace with:**
> This is the generality-accuracy trade-off jVGf asked for.

The two F1 pairs in the preceding sentence are the measurement; saying so is redundant.

### 16. −28 (total 1,144) — L100–102, §3

**Delete:**
> (the other 514 are
> unachievable for any method — worth stating, since it means R@K is capped at 0.767 on the
> full set)

**Replace with:**
> (the other 514 are
> unachievable for any method, so R@K is capped at 0.767 on the full set)

### 17. −26 (total 1,170) — L201–203, §4: explains ±0.0000

**Delete:**
> With a fixed test split and a deterministic probe over deterministic
>   embeddings there is nothing to vary. So the uncertainty that matters for the main tables
>   is *which proteins are in the test set*, which the bootstrap estimates.

**Replace with:**
> Fixed test split, deterministic probe,
>   deterministic embeddings — nothing to vary. The uncertainty that matters for the main
>   tables is *which proteins are in the test set*, which the bootstrap estimates.

### 18. −26 (total 1,196) — L23, §1: signpost

**Delete:**
> **The headline is good.** ProtSent-V2-150M

**Replace with:**
> ProtSent-V2-150M

Optional. It is a signpost, and §1 uses bolded lead-ins consistently, so keeping it is
defensible; but "strongest model we have ever measured" is not a sentence that needs to be
told it is good news.

### 19. −19 (total 1,215) — L216–219, §4: 42-word sentence

**Delete:**
> The planned binning was impossible — the [0, 0.2) identity
> bin is empty and median max-identity is 0.908, because AFDB covers essentially all of
> UniProt (true of ESM-2's own UniRef50 too, so it is a property of corpus coverage, not of
> us).

**Replace with:**
> The planned binning was impossible: the [0, 0.2) identity bin
> is empty and median max-identity is 0.908, because AFDB covers essentially all of UniProt.
> That is true of ESM-2's own UniRef50 too — corpus coverage, not us.

### 20. −7 (total 1,222) — L250–251, §5.1: claim stronger than the numbers allow

This is the only place in the document where a claim outruns its evidence, so it is worth
the edit despite the trivial saving. "at that layer ProtSent-V2 leads" rests on 0.7500 vs
0.7400 and 0.7357 — a 1.0–1.4 point margin with no CI, in a section whose whole purpose is
to soften a negative result. A reviewer will notice. Also, "the final-layer linear probe
understates every model" is stated here and again as §6 item 4.

**Delete:**
> final layer, and at that layer ProtSent-V2 leads. Same pattern at 35M. So the final-layer
> linear probe understates every model and is not the instrument to settle this on.

**Replace with:**
> final layer; at that layer ProtSent-V2 leads by 1.0–1.4 points (no CI computed). Same
> pattern at 35M. The final-layer probe is not the instrument to settle this on.

---

## Running total

**1,222 chars, 204 words.** 18,632 → 17,410 chars (−6.6%); **2,847 → 2,643 words**.

Verified after applying all 20: no file path or script name lost, all 64 table rows intact,
§6 still has six numbered items, §8 byte-identical, and every number removed from prose
(0.6970, 0.6556, 0.848/0.896) still present in the table it was restating.

---

## Not cuts — three things to check before sending

These are not prose problems, but a coauthor reading carefully will hit all three.

1. **The same quantities disagree between §3 and §5.2.** SCOPe-40 ESM-2 150M is 0.5535 /
   0.7702 / 0.4236 in §3 and 0.5529 / 0.7702 / 0.4242 in §5.2; ProtSent-V2 150M is 0.7431 /
   0.9368 / 0.7046 in §3 and 0.7425 / 0.9374 / 0.7048 in §5.2. Remote-homology 3-NN is
   0.5194 / 0.6612 in §3 and 0.5200 / 0.6606 in §5.2. Presumably different code paths or
   query sets, but as written it reads as sloppiness. Either reconcile or add one clause
   saying why §5.2 differs.

2. **The 15,000,000 in the row arithmetic appears nowhere else.** §2 gives STRING rows-after
   as 71,891,417, then the verification sums 27,929,772 + 126,301,607 + **15,000,000**. The
   arithmetic is correct and matches the trainer log, but the reader has no way to know
   STRING was subsampled to 15M. One clause fixes it.

3. **"several tasks" with two examples.** §3: "Alignment beats the best embedding model
   outright on several tasks — enzyme class … and GO molecular function … are not close."
   If the count is known, give it ("on 5 of 23 tasks"); if only two, say two.

## One table opportunity (no character saving)

§6 item 6 is a 72-word sentence carrying three independent claimed-vs-actual pairs — a
paragraph doing a table's job badly. Same length as a table but far more scannable:

| the paper says | actual |
|---|---|
| SCOPe eval: 100,000 seqs, superfamily level | 2,207 seqs, **family** level (100,000 was a sampling cap echoed into the results table) |
| remote-homology split is hierarchy-disjoint | TAPE's three holdouts pooled: 718 fold + 1,254 superfamily + 1,272 family = 3,244 |
| PPI decontamination as described | does not match what `data_prep.py` does |

A table inside a numbered list is awkward in Markdown; if that bothers you, leave item 6 as
prose but break it into three sentences.

## Checked and left alone

- **§1 as a whole.** It restates §3/§5/§6, but that is the point of a summary for a busy
  reader, and it is the shortest part of the document per fact carried. The only sprawl in
  it was the signpost (edit 18).
- **§2 config list** ("CachedMultipleNegativesRankingLoss, 1024 contrastive batch per
  device, …"). It looks like a paragraph doing a table's job, but as a comma list it is
  shorter than any table of it would be.
- **§4 seed numbers** and the few-shot bullet — four values inline, shorter than a table.
- **§7 and §8** — untouched, as instructed. §8 has no fat in it.
- **"This is the finding I would most want a reviewer not to discover first"** (§5.2) —
  reads as voice, not performance. Keep.
- **Vocabulary.** Zero hits across the full tell list. Whoever wrote this did not let a
  model near it.
