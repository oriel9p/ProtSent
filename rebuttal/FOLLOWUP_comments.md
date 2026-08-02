# ProtSent — follow-up comments to the posted rebuttal (submission 28056)

Additional OpenReview comments, posted **after** the rebuttal already on the thread. They do
not restate it. Each block between the markers is one paste unit.

State of the thread as of 2026-08-02: HNXd is satisfied and raising their score; jVGf asked
for a direct comparison against structure-distillation models and we replied on 01 Aug that
we would "add one compatible structure-informed sequence baseline in the revision under the
same frozen-embedding and leakage-control protocol". **That baseline is now run**, which makes
the jVGf comment the one that matters. Yi1G named ProtTucker as a missing baseline; that is
now run too.

Order of importance: jVGf, then Yi1G, then HNXd (short — do not destabilise a reviewer who
has already moved).

Sources: `results/benchmarks/ism/`, `ISM_COMPARISON.json`, `scope40_bootstrap_ci_ism.json`,
`cath_eat/cath_levels.json`, `v2_150m/`, `alignment_paired_ci_150m.json`. Interpretation
rails in `RUNS.md` §ISM-C-300M.

---

## Follow-up to Reviewer jVGf — the structure-distillation baseline we promised

<!-- BEGIN jVGf_followup -->
We said we would add a compatible structure-informed sequence baseline under our own frozen-embedding protocol. It is done, and we report it as obtained.

ISM-C-300M (Ouyang-Zhang et al., your reference 3) is structure-distilled from ESM-C-300M. Vanilla ESM-C-300M is therefore its matched control at identical architecture, parameter count and tokenizer, and we ran both through the same pipeline as ProtSent: 23 tasks, both probes, test split, plus SCOPe-40 with 10,000-resample paired intervals.

Each method against **its own** frozen base, which is the only comparison that controls the backbone:

| | SCOPe-40 dR@1 | dR@10 | dMAP |
| --- | --- | --- | --- |
| ISM-C - ESM-C (structure distillation) | +0.060 [+0.034, +0.085] | +0.078 [+0.053, +0.103] | +0.053 [+0.038, +0.067] |
| ProtSent-V2 - ESM-2, 150M | +0.190 [+0.165, +0.214] | +0.167 [+0.148, +0.187] | +0.281 [+0.264, +0.297] |

On the 20 tasks with a defined main metric, against its own base, ISM-C is 7 win / 1 tie / 12 lose under a 3-NN probe and 7 / 3 / 10 under a linear probe; ProtSent-V2-150M is 10 / 3 / 7 and 4 / 4 / 12. No 3-NN record here resolves statistically, so we draw no comparative claim from the tallies. The supportable statement is narrow: under a 3-NN probe ProtSent is net positive where ISM-C is net negative, and under a linear probe both are net negative. The generality cost you saw us concede is not peculiar to us; it is what this family of methods trades.

The Pareto answer you asked for is in *which* tasks each one moves. Against its own base, ISM-C gains on structure- and solubility-flavoured tasks — cloning +0.180, fluorescence +0.173, solubility +0.091, material production +0.082, SCOPe-40 +0.080, remote homology +0.048 — and loses on function and fitness: stability -0.114, EC -0.113, temperature stability -0.086, GB1 variant effect -0.079, GO molecular function -0.079. Both probes agree on the direction. Our own per-source ablations move a different set: STRING drives PPI, the DMS objective drives fluorescence, AlphaFold DB drives remote homology and EC.

So the two methods do not sit at different points on one curve; they shape different neighbourhoods. A user whose notion of "similar" is structural should use a structure-distilled encoder, and on that task ISM-C improves where it is meant to. A user whose notion of similarity mixes family, interaction and mutational fitness has no structure teacher to distil from, and that is the need ProtSent fills. That is our answer to when and why.

Two honesty notes. This crosses model families and scale: ESM-C is a separate pretraining run and a weak foundation for nearest-neighbour transfer here, scoring below our 35M base on SCOPe-40, so we take only the ISM-C minus ESM-C delta from those rows and claim no head-to-head win. And ESM-S, S-PLM and Magneton remain unrun; ISM was the one whose weights we could obtain and load inside the discussion window.
<!-- END jVGf_followup -->

---

## Follow-up to Reviewer Yi1G — ProtTucker, and the 150M

<!-- BEGIN Yi1G_followup -->
Two items from your review that were open when we replied are now closed.

**ProtTucker (item 7).** We ran its own benchmark: CATH v4.3 midnight-zone transfer on the authors' published splits, moving a CATH label from 69,605 lookup domains to 219 queries filtered so that no alignment-detectable relative exists in the lookup set, by 1-NN over mean-pooled embeddings. Our pipeline reproduces their published MMseqs2 baseline at 34.7% against their 35%, which is what licenses the rest. Accuracy (%) at Class / Architecture / Topology / Homologous superfamily:

| | C | A | T | H |
| --- | --- | --- | --- | --- |
| ESM-2 35M | 78.5 | 54.3 | 42.4 | 40.7 |
| ProtSent-V2 35M | 82.2 | 64.4 | 53.3 | 56.7 |
| ESM-2 150M | 74.0 | 53.0 | 41.0 | 43.3 |
| ProtSent-V2 150M | 84.0 | 69.9 | 57.1 | 62.7 |

Each method over its own frozen base: ProtTucker gains +5 / +8 / +7 / +12 over ProtT5, ProtSent-V2 gains +3.7 / +10.1 / +10.9 / +16.0 at 35M and +10.0 / +16.9 / +16.1 / +19.4 at 150M. The shape is the same, the gain growing as the level gets harder. In absolute terms ProtSent-V2-150M is indistinguishable from raw ProtT5 at every level (against 84 / 67 / 57 / 64) at roughly 20x fewer parameters, and on the four-level mean sits between ProtTucker(ProSE-MT) and ProtTucker(ESM-1b). H-level intervals are ±8 for us and ±6 to ±8 for them, so we claim the pattern and the size of the two large deltas, not a ranking against adjacent rows. Profile HMMs still lead at the finest level, 77 against 62.7, consistent with alignment leading at top-1 in our earlier reply.

**The 150M (item 1).** The retrain on the decontaminated corpora, still running when we replied, has finished on the same filtered corpus verified the same way. On SCOPe-40 over the 1,693 eligible queries, 10,000 paired resamples, ProtSent-V2-150M minus ESM-2 150M is +0.190 [+0.165, +0.214] at Recall@1 and +0.281 [+0.264, +0.297] at MAP. Filtering cost the larger model nothing: it leads the submitted unfiltered 150M by +0.081 [+0.060, +0.102] at Recall@1, and on CATH, which we never filtered against, by 4.7 points at the Homologous-superfamily level. Your item 8 verdict is unchanged at this scale, since under a linear probe the 150M loses to its own base on 12 of 20 comparable tasks.

One limit we would rather state than have found: CATH test219 was never filtered against our pretraining corpora, so we cannot call those queries unseen in the sense your Weakness 1 asks about. Three things bound it. The queries have no alignment-detectable relative in the lookup set. ProtTucker is supervised on CATH labels directly, its strongest configuration adding 11M Gene3D sequences, while our supervision never names a CATH class. And the frozen base and the ProtSent arm share identical pretraining exposure, so the delta between them is what the contrastive stage bought. A CATH-specific filter needs a fresh search against a 126M-sequence corpus; we can run it during the discussion period if it would decide your assessment.
<!-- END Yi1G_followup -->

---

## Follow-up to Reviewer HNXd — closing the one item we left in flight

<!-- BEGIN HNXd_followup -->
We are grateful for the reviewer's re-assessment. One loop we left open in our reply is now closed, and we report it because it was promised rather than to reopen the discussion.

The 150M model, which was still being re-run on the decontaminated corpora when we replied, has finished. Every conclusion we drew at 35M holds at the larger scale. On SCOPe-40, over the same 1,693 eligible queries and 10,000 paired resamples, ProtSent-V2-150M minus ESM-2 150M is +0.190 [+0.165, +0.214] at Recall@1, +0.167 [+0.148, +0.187] at Recall@10 and +0.281 [+0.264, +0.297] at MAP. The withdrawal stands with it: under a linear probe the 150M model loses to its own base on 12 of 20 comparable tasks, so the narrowed claim is the same claim at both scales.

The retrieval and clustering result also now replicates outside our own harness, on the CATH v4.3 midnight-zone benchmark of Heinzinger et al. using the authors' published splits: ProtSent-V2 reaches 56.7 ± 8.0% at the Homologous-superfamily level against 40.7 ± 7.8% for its frozen base at 35M, and 62.7 ± 7.8% against 43.3 ± 8.2% at 150M, with our pipeline reproducing that paper's published MMseqs2 baseline at 34.7% against 35%. Numbers are in our follow-up to Reviewer Yi1G.
<!-- END HNXd_followup -->
