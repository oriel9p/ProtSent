# Late-interaction pilot (35M) — SUMMARY

> **SUPERSEDED — corrected 2026-08-25.** This file describes the 35M pilot only, and a blind audit
> found three of its claims contradicted by the CSVs in this same directory. They are struck
> through below. **[`../RESULTS.md`](../RESULTS.md) is the current source**, generated from the
> CSVs rather than hand-written, so it cannot drift from the data the way this file did.
>
> - "Best configuration = native 480-D" is **wrong**: `protsent_late_proj128` scores .7057
>   superfamily against 480-D's .6984, paired **+.0074 [+.0035, +.0113]**, CI clear of zero.
> - "Degraded neither" is **wrong**: the 64-D arm is −.0110 [−.0155, −.0067] against doing
>   nothing, and 31k further steps at 35M cost −.0152 superfamily / −.0194 fold.
> - "13 comparable tasks (5 excluded ... SS3)" is **wrong twice**: 20 − 5 = 15, and SS3 was not
>   excluded. The two tasks that vanished (Residue Conservation, Intrinsic Disorder) produced
>   **empty metric cells in all four arms** — an unreported failure, not an exclusion.


Rendered report (same content, readable): `report.html`, published at
https://claude.ai/code/artifact/17bbc760-13fd-492c-a9c7-06682e47eaa6

**Setup.** Two arms trained with an identical recipe from different starting points: `Synthyra/ESM2-35M`
(vanilla) and `GrimSqueaker/ProtSent-V2-35M` (already contrastively post-trained on 34.8M pairs), each plus a
fresh 64-D residue projection (30,720 weights), `<cls>`/`<eos>` masked on both sides, per-residue L2 +
MaxSim, `CachedMultiVectorMultipleNegativesRankingLoss` (scale 1.0), round-robin over decontaminated
Pfam / AFDB / STRING-15M pairs, **2,000 steps = 512k pairs, ~46 min on 2×A100 each**. Backbone LR 1e-5,
projection LR 1e-4, bf16, max_seq_length 512, seed 42. Optimiser sees 256 pairs/step (2×128); in-batch
negatives are 128 (`gather_across_devices` off). Step-0 dense-view parity: **exact (0.0)** on both backbones.

SCOPe-40 numbers below are **eligible-query** metrics (labels with ≥1 non-self gallery member;
n = 1,693 / 2,020 / 2,116 at family / superfamily / fold).

## The finding: the head and the backbone are different things

Scoring each *trained* backbone twice — through its 64-D head, and over its own native 480-D residues —
separates two effects that earlier revisions of this summary confounded. Paired bootstrap, superfamily
eligible AP, n = 2,020, every interval excluding zero:

| Comparison | Δ | 95% CI |
|---|---|---|
| ProtSent-Late (480-D) − ProtSent-V2 + MaxSim | +.0152 | [+.0117, +.0187] |
| ProtSent-Late (480-D) − ProtSent-Late (64-D) | +.0262 | [+.0221, +.0306] |
| ProtSent-Late (64-D) − ProtSent-V2 + MaxSim | −.0110 | [−.0155, −.0067] |
| ESM-2-Late (480-D) − ESM-2 + MaxSim | +.1204 | [+.1102, +.1306] |
| ESM-2-Late (64-D) − ESM-2-Late (480-D) | +.0766 | [+.0690, +.0841] |

~~**Late training improved both backbones and degraded neither.**~~ *(false — see banner)* Late training improved both backbones; the 64-D arm on ProtSent is significantly WORSE than not training. The 64-D projection is *load-bearing* when
the backbone's residues have never been contrastively trained — it is the space the objective optimised — and
*lossy* when they have, because it bottlenecks representations 34.8M pairs already shaped. Family level agrees
on every sign; the only comparison that loses significance there is ProtSent-Late (64-D) − ProtSent-V2 +
MaxSim, at −.0036 [−.0099, +.0030].

~~**Best configuration measured: ProtSent-V2 + late training, scored at native 480-D**~~ *(false — 128-D wins, see banner)* — superfamily eligible
MAP .698, against .683 for the untouched backbone and .672 for the 64-D arm.

## A. Does MaxSim help without any training? Yes, on both backbones

Eligible family MAP, pooled cosine → MaxSim over native residues: ESM-2 .421 → .545; ProtSent-V2 .646 → .684.
Same direction at superfamily and fold. On CATH, rescored in one pipeline so every pair is McNemar-testable, the
gain is significant for ESM-2 (43.3 → 54.7, +11.3, p=.002) and directional but not significant for ProtSent-V2
(56.7 → 61.3, +4.7, p=.10) on that benchmark's 150 queries. Cost is scoring time and index size, not model changes.

## B. Does late training help beyond that? Yes for both — but read it at 480-D

At 64-D the ProtSent arm looks flat (.684 → .681 family) because the head's loss cancels the training's gain.
At 480-D the training gain is visible and significant on both backbones (table above).

## C. Which init is better? For the late model itself, they converge

ESM-2-Late (64-D) and ProtSent-Late (64-D) land within .002 MAP of each other at superfamily. ProtSent's
advantage is that it needs **zero** training to reach .683. Curves plateau by ~step 1,500 from both inits.

## D. Did late training damage the foundation embedding? Depends on the probe

ProtBench transfer suite, test split, seed 42, 20k cap, 13 comparable tasks (5 excluded because ProtBench
forces linear for them in both passes: EC, the three scope40 rows, SS3). Ties are |Δ| < 0.005:

| Probe | ESM-2-Late vs ESM-2 | ProtSent-Late vs ProtSent-V2 |
|---|---|---|
| KNN (k=3) | 7 W / 1 T / 5 L | 4 W / 5 T / 4 L |
| Linear | 7 W / 2 T / 4 L | **10 W / 2 T / 1 L** |

The Stability task carries the disagreement: KNN −.162 (ESM-2) / −.045 (ProtSent) becomes linear −.020 /
**+.072**. "Late training costs fitness and stability tasks" was largely a KNN-probe artefact. The consistent
reading across probes: neighbourhood geometry roughly unchanged, more information linearly decodable.

## E. Probe dependence

See D. KNN and linear are reported side by side and never averaged.

## Other measured results

- **Two-stage retrieval.** On ProtSent-V2's cosine shortlist, MaxSim reranking the top 10 recovers MAP .675 of
  the .684 full value — 1/220th of the scoring work. On ESM-2's weaker shortlist, rerank@100 reaches only .593
  of .698. Fixes scoring cost, **not** storage.
- **Cost.** Scoring the 2,207² SCOPe matrix: cosine 0.03 s, 64-D MaxSim 8.1 s, native 480-D MaxSim 41 s, on a
  shared 3–8 s encode. Storage per protein (mean domain ~154 residues): 480 numbers dense, 9,878 at 64-D
  (20.6×), 74,088 at 480-D (154×). Training is ~21% slower than dense (sdpa both).
- **Flash attention was never enabled** (inferred from the environment, NOT recorded: these pilot `runtime.json` files predate the `attn_implementation` field) — `kernels` 0.12.3 is below the 0.15.2 floor and `flash-attn` is
  absent, so every run silently used sdpa. FA3 measures 1.97× on Pfam, 1.48× on STRING (≈1.30× unpadding ×
  1.14× kernel), peak VRAM 10.7 → 4.2 GB. A paired quality A/B is running before any recipe change; the PR is
  held because it bundles a dependency floor with an optimiser-precision change.
- **Symmetrisation is a trap.** Raw ½(S+Sᵀ) collapses (MaxSim rows scale with query length). Length-normalised
  symmetrisation still costs every arm except the untouched ProtSent-V2: −.275 and −.128 MAP for the trained
  arms, −.078 for untrained ESM-2, +.003 for ProtSent-V2.
- **Contamination.** SCOPe-40 is *not* decontaminated (corpus filtered against remote-homology and PPI test
  splits only; median SCOPe domain sits at 0.91 max identity to training data). Restricting to <40% identity
  queries (n=164) preserves the ordering: ESM-2-Late .676 > ProtSent-V2 + MaxSim .663 > ProtSent-V2 cosine
  .619 > ESM-2 cosine .333.
- **Alignment baselines** (SCOPe: family level only; re-levelling needs a phmmer/MMseqs2 rerun): phmmer
  max-sensitivity .753 / .898 / .607 beats every arm here at R@1 while losing on depth and MAP. On CATH, published
  CATH-Gene3D HMMER (77) and ProtTucker (76) are far above our best arm (61.3) — that table compares small ESM-2
  models, it is not a claim against alignment.
- **CATH protocol effect, measured.** Rescoring ESM-2's dense arm in our own pipeline gives 43.3 against the
  published EAT number's 40.7, so 2.6 points of the old "+14.0 dense → MaxSim" headline was pure cosine-vs-euclidean
  and the real paired gain is +11.3. On ProtSent-V2 the two rules agree to a hundredth of a point.

## Caveats

Single seed, single run per arm; transfer tables have no CIs. CATH's test_h is 150 queries (one protein = 0.67 points); the
cross-protocol problem is now fixed by a single-pipeline eight-arm rerun, but at that sample size only comparisons
involving the ESM-2 dense arm reach significance — every trained-vs-untrained and every head-size pair sits between
p=.23 and p=1.00, so CATH corroborates SCOPe rather than testing anything independently.
No hard-negative mining. AFDB/STRING pairs come from the group-sorted prefix, so 2,000 steps saw a narrow
slice of clusters. Built on V2, not the shipped V2.5. No PROTOCOL-benchmark comparison.

## Go / no-go: GO, with the recommendation changed

1. ~~**Ship ProtSent-V2 + late training scored at 480-D.**~~ Superseded: ship the **128-D** head, no new training needed
   beyond the 1.5 GPU-h already spent, cost is index size.
2. **The head-size ablation now has a sharp hypothesis:** 128-D should recover part of the .026 the 64-D head
   discards on a ProtSent backbone and should matter far less on ESM-2. Running.
3. **Hard-negative mining** is the largest untouched lever.

Dropped from earlier versions of this summary: "lower the backbone LR to avoid the stability regression" (a
KNN-probe artefact) and "the projection is lossy, use none" (true only for an already-contrastive backbone).
