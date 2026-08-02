# ProtSent — NeurIPS 2026 follow-up comments (submission 28056)

Three additional OpenReview comments, one per reviewer, to be posted **after** the main
responses in `FINAL_rebuttal.md`. They do not replace anything there. Each block between
the markers is a self-contained paste unit.

New since the main response: the CATH v4.3 midnight-zone benchmark (ProtTucker / EAT
protocol, 8 arms), ESM-C and ISM-C as a matched structure-distillation control, and the
completed ProtSent-V2-150M retrain on the decontaminated corpora with a full sweep and
paired bootstrap intervals.

Sources: `results/benchmarks/cath_eat/cath_levels.json`,
`results/benchmarks/cath_eat_controls/`, `results/benchmarks/v2_150m/`,
`results/benchmarks/scope40_bootstrap_ci_150m_v2.json`,
`results/benchmarks/whiten_scope_control.json`.

---

## Follow-up to Reviewer jVGf

<!-- BEGIN jVGf_followup -->
Since our response we have run the comparison we called the one we would most want to add, and the matched structure-distillation control we said we lacked.

ProtTucker's own benchmark now has ProtSent rows: CATH v4.3 midnight-zone transfer on the authors' published splits, transferring a CATH label from 69,605 lookup domains to 219 queries filtered so that no alignment-detectable relative exists in the lookup set, by 1-NN over mean-pooled embeddings. That is their protocol rather than ours. Our pipeline reproduces their published MMseqs2 baseline at 34.7% against their 35% at the Homologous-superfamily level, which is what licenses the rest, and 3-mer frequency vectors score 0.00, matching their random floor.

Accuracy (%) at Class / Architecture / Topology / Homologous superfamily, with the 95% interval at H:

| | C | A | T | H |
| --- | --- | --- | --- | --- |
| ESM-2 35M | 78.5 | 54.3 | 42.4 | 40.7 ± 7.8 |
| ProtSent-V2 35M | 82.2 | 64.4 | 53.3 | 56.7 ± 8.0 |
| ESM-2 150M | 74.0 | 53.0 | 41.0 | 43.3 ± 8.2 |
| ProtSent-V2 150M | 84.0 | 69.9 | 57.1 | 62.7 ± 7.8 |
| ESM-C 300M | 66.7 | 36.1 | 21.9 | 18.7 ± 6.3 |
| ISM-C 300M (structure-distilled) | 80.8 | 47.9 | 29.5 | 25.3 ± 7.2 |

Against ProtTucker, the comparison that controls the backbone is each method's gain over its own frozen base. ProtTucker gains +5 / +8 / +7 / +12 over ProtT5; ProtSent-V2 gains +3.7 / +10.1 / +10.9 / +16.0 at 35M and +10.0 / +16.9 / +16.1 / +19.4 at 150M. The shape is the same, the gain growing as the level gets harder, and both our H-level gains exceed the published +12. In absolute terms ProtSent-V2-150M is indistinguishable from raw ProtT5 at every level within our intervals (84.0 / 69.9 / 57.1 / 62.7 against 84 / 67 / 57 / 64) at roughly 20× fewer parameters; on the four-level mean it is above ProtTucker(ProtBERT) at 52 and ProtTucker(ProSE-MT) at 66, and below ProtTucker(ESM-1b) at 71 and ProtTucker(ProtT5) at 76. H-level intervals are ±8 for us and ±6 to ±8 for them, so we claim the pattern and the size of the two large deltas, not a ranking against adjacent rows.

On structure-informed pLMs, ISM-C against its own vanilla base, ESM-C, is the matched control: same architecture, same parameter count, our probe. Structure distillation gains +14.1 / +11.8 / +7.6 / +6.6, largest at Class and smallest at Homologous superfamily, which is the reverse of our ordering. That is the evidence we previously lacked for treating relation type as a design axis rather than asserting it: a single structural teacher buys most where the distinction is coarse, while heterogeneous relation supervision keeps buying at the level where fold recognition is hard.

One reading we do not draw. Both ESM-C variants score below our 35M ESM-2 base on this probe, so nothing here says ProtSent beats ISM. ESM-C is a separate pretraining run with different embedding geometry and a weak foundation for Euclidean 1-NN transfer; the ISM-C minus ESM-C delta is the only comparison we take from those two rows.

Our position is unchanged where it should be. Profile HMMs still lead at the finest level, 77 against our 62.7, consistent with alignment leading at top-1 in our main response. We beat MMseqs2 at all four levels, and beat profile HMMs at Class (84.0 against 70) and Architecture (69.9 against 60) only.
<!-- END jVGf_followup -->

---

## Follow-up to Reviewer HNXd

<!-- BEGIN HNXd_followup -->
Two developments since our response bear directly on the analyses you named.

First, the 150M retrain on the decontaminated corpora has finished, so the results we withdrew for want of a model can be replaced by measured ones rather than left deleted. On SCOPe-40 over the same 1,693 eligible queries, 10,000 paired resamples, ProtSent-V2-150M minus ESM-2 150M is +0.190 [+0.165, +0.214] at Recall@1, +0.167 [+0.148, +0.187] at Recall@10 and +0.281 [+0.264, +0.297] at MAP. Absolute values are 0.743 / 0.937 / 0.705 against the backbone's 0.553 / 0.770 / 0.424. The withdrawal of the general-purpose claim is untouched by this: under a linear probe the 150M model loses to its own base on 12 of 20 comparable tasks, 4 winning and 4 tied, the same verdict as at 35M.

Second, the geometry result you asked us to measure directly now has an external check, on a benchmark we did not build. On the CATH v4.3 midnight-zone transfer task of Heinzinger et al., using the authors' published splits and 1-NN annotation transfer, ProtSent-V2 reaches 56.7 ± 8.0% at the Homologous-superfamily level against 40.7 ± 7.8% for its frozen base at 35M, and 62.7 ± 7.8% against 43.3 ± 8.2% at 150M. Our pipeline reproduces that paper's published MMseqs2 baseline at 34.7% against 35%, so the splits, labels and scoring are theirs rather than ours. Details are in our follow-up to jVGf.

That benchmark also answers a question our own table could not. Contrastive post-training is not a proxy for scale here: stock ESM-2 scores 40.7 at 35M, 43.3 at 150M and 42.7 at 650M, so 18× the parameters buys about two points, while post-training the 35M buys sixteen.

Two controls on the geometry, one of which cuts against us.

L2-normalising the embeddings, which gives the untuned model the direction-only geometry our cosine objective trains for, is worth +2.7 at 35M and +1.3 at 150M to the baselines and nothing to ProtSent, 0.0 and -0.7, so the CATH gap is not a normalisation artefact.

Whitening is the stronger transform, and centring plus whitening the baseline recovers most of our top-1 advantage on SCOPe-40: ProtSent-V2-150M raw minus ESM-2 150M whitened is +0.009 [-0.011, +0.029] at Recall@1, unresolved. What survives it is depth, +0.022 [+0.009, +0.034] at Recall@10 and +0.077 [+0.064, +0.091] at MAP. We report this because it is the same boundary we drew in our response, that the supportable claim is ranking depth and MAP rather than top-1, now tested against a stronger null than we had then.
<!-- END HNXd_followup -->

---

## Follow-up to Reviewer Yi1G

<!-- BEGIN Yi1G_followup -->
Two updates, one of which reverses a withdrawal, and one limit we would rather state than have found.

We withdrew the 150M results because no 150M model existed on the decontaminated corpora. That retrain has now finished, on the same filtered corpus verified the same way, with zero flagged sequences surviving in all three training files, so the reason for the withdrawal is gone and we can put measured numbers where the deleted ones were. On SCOPe-40 over the 1,693 eligible queries, 10,000 paired resamples, ProtSent-V2-150M minus ESM-2 150M is +0.190 [+0.165, +0.214] at Recall@1 and +0.281 [+0.264, +0.297] at MAP. Filtering did not cost the larger model anything: the decontaminated 150M leads the submitted unfiltered one by +0.081 [+0.060, +0.102] at Recall@1 and +0.061 [+0.048, +0.074] at MAP. The linear-probe verdict is unchanged at the larger scale, 4 win / 4 tie / 12 lose against its own base, so restoring the scale does not restore the general-purpose claim.

On baselines, item 7 named ProtTucker as the real gap, its protocol being ours. That gap is now closed: we ran the CATH v4.3 midnight-zone benchmark on the authors' published splits, reproducing their MMseqs2 baseline at 34.7% against 35%, and report ProtSent at both scales alongside a matched structure-distillation control. Numbers are in our follow-up to jVGf. The top-1 position is unchanged there too. ProtSent-V2-150M reaches Recall@1 0.743 on SCOPe-40, above phmmer's 0.697 at default settings, but phmmer run at maximum sensitivity reaches 0.753, so we continue to claim ranking depth rather than top-1.

The limit. CATH test219 was never decontaminated against our pretraining corpora, so we cannot claim those queries are unseen in the sense your Weakness 1 asks about. Three things bound what that leaves. The queries are selected to have no alignment-detectable relative in the lookup set, so the task is not solvable by the kind of similarity a residual near-duplicate would supply. ProtTucker is supervised on CATH labels directly, its strongest configuration adding 11M Gene3D sequences, while our supervision is Pfam, Foldseek and STRING co-membership and never names a CATH class, so the comparison is not tilted our way by label access. And the frozen base and the ProtSent arm share identical pretraining exposure, so the delta between them is what the contrastive stage bought regardless of what either has seen. A CATH-specific identity filter would need a fresh search against a 126M-sequence corpus; we can run it during discussion if it is the item that decides your assessment.
<!-- END Yi1G_followup -->
