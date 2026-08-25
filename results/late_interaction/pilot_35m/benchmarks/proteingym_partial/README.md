# ProteinGym — PARTIAL, quarantined

Quarantined 2026-08-25. These files are **not results**. They are kept because the *paired*
comparisons inside them are still sound and because re-deriving them costs GPU time.

## Why partial

| limitation | effect |
|---|---|
| `--max_variants_per_assay 500` | ~4.3% of the 2.47M ProteinGym variants scored |
| 512-residue truncation | ~4.7% of mutations land past the cut; 55 of 217 substitution assays have a WT longer than 510 aa; 7 assays drop entirely |
| plain mean over groups | the leaderboard aggregates by UniProt ID, then by the 5 `coarse_selection_type` categories ("corrected average") |
| clinical indels | scored as a per-gene mean AUC; the reference protocol is one total AUC over the pooled variants |

Recomputing the corrected average from the npz files on disk moves `proj128_late` on
dms_substitutions from 0.3303 to **0.3170**, and `protsent_v2_dense` from 0.2818 to **0.2598**.
That correction is free (no rescoring); the coverage and clinical-indels problems are not.

## Why the truncation drops variants rather than mis-scoring them

A variant whose mutated position falls past the 512-residue cut is byte-identical to the
truncated wild type, so its score is the self-similarity: a block of exact ties that drags the
correlation toward zero. Dropping them is the lesser evil, but it changes which variants each
assay is scored on, which is exactly why the absolute numbers are not comparable. 55 of the 217
substitution assays have a wild type longer than 510 aa.

## What is still usable

**Paired deltas only** — MaxSim vs pooled cosine at identical weights, and checkpoint-vs-checkpoint.
Both sides ran under the same cap, the same truncation and the same assay set, so the difference is
a scoring/weights effect, not a protocol artifact. Those are the `paired comparison` rows.

## What is not usable

Any **absolute** number, and any comparison against the ProteinGym leaderboard or against another
paper. Do not put these values in a table next to published ProteinGym scores.

## Rerun

~1.0 h per arm at full coverage (measured: 245-280k residues/s, 920M residues total), 1024-residue
context. Deferred: the scoring path is being optimised in a separate session. When it lands:

    STAGES=proteingym ./run_after_training.sh    # writes to ../ , not here

Delete this directory once the full-coverage numbers exist.
