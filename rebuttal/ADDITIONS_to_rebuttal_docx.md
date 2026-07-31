# Additions for "ProtSent - Neurips Rebuttal.docx"

Documentation and forward plan. Not reply text — the postable response is
`rebuttal/FOLLOWUP_openreview.md`. Paste these as new sections.

---

## A. Status as of this round

Reviewer HNXd has **raised their score**, stating the revised narrative, claims and
evidence are satisfactory, that the SCOPe-40 analyses address embedding-space organization
and retrieval, that the linear-probe results justify the narrowed framing, and that the
confidence intervals, repeated few-shot runs and absolute scores resolve their statistical
concerns including the erroneous +244.5% cell.

Their one open point is procedural rather than technical: the claims changed enough that
another review round may be warranted, and they leave that to the AC. That is answered in
Part 1 of the follow-up response — the short form is that the scope narrowed onto results
the submission already contained, and every correction was self-reported.

**Implication for effort allocation.** HNXd is addressed. The remaining leverage is jVGf,
who stated that positioning against structure-informed models plus the
generality–accuracy trade-off would move them to accept. The trade-off is measured; the
positioning is a writing task. That is the cheapest remaining score movement and it needs
no compute.

---

## B. Artifacts produced this round

| artifact | location |
|---|---|
| ProtSent-V2-35M weights | huggingface.co/GrimSqueaker/ProtSent-V2-35M |
| ProtSent-V2-150M weights | huggingface.co/GrimSqueaker/ProtSent-V2-150M |
| Code, results, documentation | github.com/oriel9p/ProtSent, branch `rebuttal` |
| Decontaminated corpus | `/storage/users/ddofer/data/protsent-data-dc40` |
| Coauthor briefing | `COAUTHOR_BRIEF.md` |
| Which model is which | `RUNS.md` |
| Evidence pack (every citable number) | `rebuttal/NEW_EVIDENCE.md` |
| Methods and controls record | `REBUTTAL_LEAKAGE.md` |

Both model cards state the decontamination procedure, the three training sources, the full
training configuration, and the limitations including linear-probe neutrality.

---

## C. Paper changes required for a camera-ready

Ordered by how load-bearing they are.

1. **Retire the general-purpose framing** in the abstract and introduction. The supported
   claim is a retrieval and metric-space contribution: a sequence-only embedding whose
   nearest-neighbour structure is substantially better for homology and structural
   retrieval, and which is neutral under a trained linear readout.
2. **Lead the retrieval claim with R@10 and MAP rather than top-1.** The depth and MAP
   advantages survive every control we ran; the top-1 advantage is scale-dependent and, at
   150M, is the metric most sensitive to a simple re-conditioning of the baseline.
3. **Replace Table 3.** Its numbers come from the pre-audit evaluation path. The corrected
   SCOPe-40 table with all four arms, alignment baselines and paired confidence intervals
   supersedes it; this is a replacement, not an erratum.
4. **Re-run or relabel Tables 4 and 7.** They ablate settings that the shipped V2
   configuration now adopts, so "Full model (ProtSent)" in those tables is no longer the
   released model.
5. **Correct three evaluation descriptions.** SCOPe retrieval is 2,207 sequences at the
   **family** level, not 100,000 at superfamily — the 100,000 was a sampling cap echoed
   into the results table. The remote-homology split is **not** hierarchy-disjoint; it is
   TAPE's three holdouts pooled (718 fold + 1,254 superfamily + 1,272 family = 3,244). The
   PPI decontamination description does not match `data_prep.py`, which uses `easy-search`
   at 40% identity with `--cov-mode 1 -c 0.8`, removing hit query IDs rather than
   `easy-linclust` clusters at 50%.
6. **Withdraw the label-scarcity claim** as phrased. Our own few-shot measurements
   contradict the proposed mechanism: a trained linear head beats 3-NN in almost every
   model/task/N cell tested, including N=50.
7. **Report the pooling layer and its sensitivity.** All probes pool the final layer, and a
   layer sweep shows an intermediate layer is materially better on remote homology for
   every model tested including the untuned backbone.
8. **State that V2 is a three-source model.** The decontamination covered Pfam, AFDB and
   STRING; the DMS/ProteinGym source was not decontaminated and so is absent from V2.
   This confounds V1-vs-V2 comparisons on fitness tasks.

---

## D. Experiments still outstanding

Ranked by value per GPU-hour.

| experiment | cost | what it buys |
|---|---|---|
| Positioning + trade-off discussion for jVGf | writing only | The stated condition for jVGf moving to accept |
| Remote homology on the 718 fold-level holdouts only | ~1 h | Tests whether the flagship gain survives at the hierarchy level the paper claims |
| Per-task bootstrap CIs on the 23-task table | ~2 h | Closes the half of HNXd's CI request that retrieval CIs did not cover |
| Layer sweep across the full suite, with CIs | ~4 h | Turns the layer observation into a result rather than an anecdote |
| Unfiltered-corpus retrain at the V2 configuration | ~11 h at 35M, ~26 h at 150M | The single control that would make the decontamination claim a clean ablation |
| Whitened-vanilla baseline in the paper | ~1 h | Pre-empts the strongest cheap objection to the contribution |
| Matched runs of ProtTucker / Foldseek / PLMSearch / DHR / ProTrek | days, external deps | Not planned; we claim no superiority to them |

**The whitened-baseline control deserves its own note.** Stock ESM-2 embeddings are
severely anisotropic, and simply whitening them recovers a large fraction of ProtSent's
k-NN advantage — on remote homology it closes nearly all of it, and on SCOPe it closes the
top-1 gap at 150M. ProtSent retains a significant advantage in ranking depth and MAP at
both scales, so the contribution survives, but it is narrower than the paper currently
implies. A reviewer can run this in an afternoon. We judged it out of scope for the
rebuttal, which answers the reviews; it belongs in the camera-ready, where it strengthens
the narrow claim more than it costs the broad one.

---

## E. Known confounds, to be stated in the paper rather than discovered

- ProtSent-V2 differs from V1 in the corpus **and** in: no synthetic hard negatives,
  proportional sampling, no DMS source, larger effective batch, Matryoshka heads. It is
  not a single-variable ablation of decontamination.
- The 150M used a smaller per-cluster pair budget (k=5 vs k=8) and therefore saw 31% fewer
  training pairs than the 35M — 23.9M vs 34.8M. Cross-scale comparisons are confounded by
  data budget as well as capacity.
- Only the remote-homology and PPI test sets were decontamination targets. The other
  benchmark test sets were not filtered against, so no corpus-wide leakage claim is
  supported.
- SCOPe-40 was deliberately not a filter target, because it has no train/test split and
  filtering against it would remove nearly all domain sequences. The identity-stratified
  analysis substitutes for it.
- Full-data evaluation is near-deterministic across seeds on seven of eight tasks tested;
  Thermostability (FLIP) is the exception (sd 0.013–0.017) because its split is a seeded
  re-split.
