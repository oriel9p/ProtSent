# ProtSent — NeurIPS 2026 follow-up comments (submission 28056)

Three additional OpenReview comments, one per reviewer, to post **after** the main responses
in `FINAL_rebuttal.md`. They add the CATH midnight-zone benchmark (ProtTucker / EAT protocol)
and the ESM-C / ISM-C structure-distillation control. The ProtSent-V2-150M results are in the
main responses, not here.

Sources: `results/benchmarks/cath_eat/cath_levels.json`, `results/benchmarks/cath_eat_controls/`.

---

## Follow-up to Reviewer jVGf — ProtTucker and a matched structure-distillation control

<!-- BEGIN jVGf_followup -->
We have run the comparison we said we most wanted to add, on ProtTucker's own benchmark.

CATH v4.3 midnight-zone transfer, using the authors' published splits: transfer a CATH label from 69,605 lookup domains to 219 queries filtered so that no alignment-detectable relative exists in the lookup set, by 1-NN over mean-pooled embeddings. It is their protocol rather than ours, and our pipeline reproduces their published MMseqs2 baseline at 34.7% against their 35%.

Accuracy (%) at Class / Architecture / Topology / Homologous superfamily:

| | C | A | T | H |
| --- | --- | --- | --- | --- |
| ESM-2 35M | 78.5 | 54.3 | 42.4 | 40.7 |
| ProtSent-V1 35M (submitted) | 81.7 | 64.4 | 45.7 | 50.7 |
| ProtSent-V2 35M (decontaminated) | 82.2 | 64.4 | 53.3 | 56.7 |
| ESM-2 150M | 74.0 | 53.0 | 41.0 | 43.3 |
| ProtSent-V1 150M (submitted) | 83.6 | 68.0 | 56.2 | 58.0 |
| ProtSent-V2 150M (decontaminated) | 84.0 | 69.9 | 57.1 | 62.7 |
| ESM-C 300M | 66.7 | 36.1 | 21.9 | 18.7 |
| ISM-C 300M (structure-distilled) | 80.8 | 47.9 | 29.5 | 25.3 |

Decontamination did not cost the method anything on a benchmark we did not filter against: V2 leads the submitted V1 at both scales, by 6.0 and 4.7 points at the Homologous-superfamily level.

Against ProtTucker, the comparison that controls the backbone is each method's gain over its own frozen base. ProtTucker gains +5 / +8 / +7 / +12 over ProtT5; ProtSent-V2 gains +3.7 / +10.1 / +10.9 / +16.0 at 35M and +10.0 / +16.9 / +16.1 / +19.4 at 150M. The shape is the same, the gain growing as the level gets harder, and both our H-level gains exceed the published +12. In absolute terms ProtSent-V2-150M is indistinguishable from raw ProtT5 at every level (84.0 / 69.9 / 57.1 / 62.7 against 84 / 67 / 57 / 64) at roughly 20x fewer parameters, and on the four-level mean sits between ProtTucker(ProSE-MT) and ProtTucker(ESM-1b). H-level intervals are ±8 for us and ±6 to ±8 for them, so we claim the pattern and the size of the two large deltas, not a ranking against adjacent rows.

ISM-C against its own vanilla base, ESM-C, is the matched structure-distillation control we said we lacked: same architecture, same parameter count, our probe. Distillation gains +14.1 / +11.8 / +7.6 / +6.6, largest at Class and smallest at Homologous superfamily, which is the reverse of our ordering. That is evidence for the design-axis claim we could previously only assert: a single structural teacher buys most where the distinction is coarse, while heterogeneous relation supervision keeps buying at the level where fold recognition is hard.

We do not read this as beating ISM. Both ESM-C arms score below our 35M base at every level but Class, so ESM-C is a weak foundation for 1-NN transfer here and the ISM-C minus ESM-C delta is the only comparison we take from those two rows. Profile HMMs still lead at the finest level, 77 against 62.7, as alignment leads at top-1 in our main response.
<!-- END jVGf_followup -->

---

## Follow-up to Reviewer HNXd — the retrieval result on an external benchmark

<!-- BEGIN HNXd_followup -->
One addition bearing on your first question, the direct retrieval and embedding-organisation measurement: it now replicates outside our own harness.

On the CATH v4.3 midnight-zone benchmark of Heinzinger et al., using the authors' published splits and 1-NN annotation transfer to 219 queries with no alignment-detectable relative in the lookup set, ProtSent-V2 reaches 56.7 ± 8.0% at the Homologous-superfamily level against 40.7 ± 7.8% for its frozen base at 35M, and 62.7 ± 7.8% against 43.3 ± 8.2% at 150M. Our pipeline reproduces that paper's published MMseqs2 baseline at 34.7% against 35%, so the splits, labels and scoring are theirs and not ours.

That benchmark also separates the method from scale: stock ESM-2 scores 40.7 at 35M, 43.3 at 150M and 42.7 at 650M, so 18x the parameters buys about two points where contrastive post-training on the 35M buys sixteen. The full table is in our follow-up to jVGf.
<!-- END HNXd_followup -->

---

## Follow-up to Reviewer Yi1G — the ProtTucker gap in item 7

<!-- BEGIN Yi1G_followup -->
Item 7 named ProtTucker as the real gap, its protocol being ours. It is now closed.

We ran the CATH v4.3 midnight-zone benchmark on the authors' published splits, reproducing their MMseqs2 baseline at 34.7% against 35%, and report ProtSent at both scales alongside a matched structure-distillation control; the numbers are in our follow-up to jVGf. The alignment position is unchanged there too: profile HMMs still lead at the finest CATH level, 77 against our 62.7.

One limit, which we would rather state than have found. CATH test219 was never filtered against our pretraining corpora, so we cannot call those queries unseen in the sense your Weakness 1 asks about. Three things bound what that leaves. The queries are selected to have no alignment-detectable relative in the lookup set, so the task is not solvable by the similarity a residual near-duplicate would supply. ProtTucker is supervised on CATH labels directly, its strongest configuration adding 11M Gene3D sequences, while our supervision never names a CATH class. And the frozen base and the ProtSent arm share identical pretraining exposure, so the delta between them is what the contrastive stage bought. A CATH-specific identity filter needs a fresh search against a 126M-sequence corpus; we can run it during discussion if it decides your assessment.
<!-- END Yi1G_followup -->
