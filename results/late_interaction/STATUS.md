# Late-interaction campaign — what is usable, what is dead, what needs rerunning

Hand-maintained ledger. `RESULTS.md` next to it is generated and holds the numbers; this holds
their standing. Read this before trusting, citing, or rerunning anything here.

Last reviewed 2026-08-25 (evening).

## In flight

**`vanilla35m_clean`** — the corrected recipe from vanilla `facebook/esm2_t12_35M_UR50D`
(same weights as the Synthyra repackage, but the native load path that takes flash+compile):
4-GPU DDP, batch 256/device (global 1024), `--gather_across_devices` (1023 in-batch negatives),
LR **5e-5 single group** (`--proj_lr 0`), **constant_with_warmup** (500), proportional over the
full uncapped mixture (34.76M pairs, disjoint sampling), seed 42, 6,500 steps ≈ 3 h.
Constant LR ⇒ checkpoints comparable at any step. Next: `v2_35m_clean`, identical command from
`GrimSqueaker/ProtSent-V2-35M`, compared at matched steps; if V2 leads, continue it.
References: `esm2_late` 0.6000 fold (old recipe from the same vanilla base),
`protsent_late_proj128` 0.7057 sfam, `esm2_zeroshot` 0.4735 floor.

## Audited and settled 2026-08-25

- **"Sorted corpora give semi-hard in-batch negatives, don't shuffle" — myth.** V2 shuffled
  twice (`--pair_dataset_shuffle` default True + ST RandomSampler; `--batch_sampler none` means
  "ST default", not "sequential"). Sorting is load-bearing only for the O(1)-memory streaming
  pair builder. The docstring that claimed otherwise is corrected in place.
- **NO_DUPLICATES hang**: cause is duplicate *multiplicity* (7 copies/sequence at k=8
  combinations) × O(remaining) rescans in ST's sampler — not file sort order. The fix
  (`027ef89`, stop auto-selecting) and the diversity fix (`5f2d408`, `--pair_sampling disjoint`,
  2.6× distinct sequences at equal budget) are both pushed and default on this branch.
  Neither is on `master`.
- **No DMS contamination in the late-interaction path**: dc40 has no DMS file and V2-35M never
  trained on `dms_cosent` (V2.5 did — see the ProteinGym contamination note in memory/RUNS).

## Usable as-is

| result | where | note |
|---|---|---|
| SCOPe-40 retrieval, all arms | `pilot_35m/scope/scope_hierarchy.csv` | fp32-verified weights, paired bootstrap over queries |
| SCOPe checkpoint curves | `pilot_35m/scope/per_query_*@*.npz`, `pilot_150m/scope/` | `@0` for the 35M arm is byte-identical to `protsent_late_proj128` (verified, max diff 0.0) |
| CATH midnight-zone | `pilot_35m/benchmarks/cath_*` | valid but underpowered: 150 queries |
| ProteinGym **paired deltas** | `pilot_35m/benchmarks/proteingym_partial/` | same protocol both sides; see that dir's README |
| Pooled ProtBench (knn/linear) | `pilot_35m/benchmarks/{knn,linear}/` | only covers `protsent_late` (2k pilot) and `esm2_late`; phase-2 arms landing now |

## Deleted, deliberately — do not restore

| artifact | why |
|---|---|
| `*_bf16bug` arms (models + rows) | AdamW params in bf16; a 1e-5 update is below the representable spacing, so 2.4% of backbone elements could move vs 93.4% in fp32. Frozen-backbone data wearing a trained label. |
| `protsent_late_capped_flash` | same bug |
| 12 mislabelled npz from those runs | had no CSV row, so any glob-based analysis would silently ingest them |

`build_late_results.py` still filters both names. That guard exists so a CSV restored from git
cannot quietly put them back; it is not evidence the files are still around.

## Quarantined — real, but not what it looks like

**ProteinGym, whole run** → `pilot_35m/benchmarks/proteingym_partial/`. Scored at 500 variants per
assay (~4.3% coverage) and truncated at 512 residues. Paired deltas hold; absolute values are
internal ranking only and must never sit beside a leaderboard number. Rerun ~1.0 h/arm at full
coverage. Deferred: the scoring path is being optimised elsewhere.

## Retracted claims

**"More late-interaction training makes it worse."** Retracted 2026-08-25. The row was labelled
`31k steps − 4k steps`; the two `runtime.json` files show two runs differing in five ways at once
(sampler, pool size, world size, attention backend, `--compile`). Decisively, the 15.5x increase in
pairs seen carries **1.04x** the Pfam exposure (170,667 → 177,441), and Pfam is where SCOPe's
supervision comes from. The curve is U-shaped, not monotonic-down: all the loss lands by step
5,000, and 5,000 → 30,000 *gains* +0.0104 fold. **Nothing in this campaign isolates step count, so
no claim about it holds in either direction.**

**"The 150M arm gains where the 35M loses" (scale-dependent sign flip).** Retracted. Its `@0`
baseline is a random projection — see below. Against its real parent it is −0.0158 sfam: same sign
and magnitude as the 35M arm.

## Known-confounded arms

**`protsent_late_150m_prop`** is *not* a full continuation. Its parent `protsent_late_150m` saved a
64-D head; this run asked for 128-D, so the head could not be carried over and started random
(`logs/queue_b.log:1525`). Verified from the weights: at checkpoint-25000 the head's absmax is
0.0781 against a fresh-init bound of 0.03953, so it is trained — but it was trained *from scratch*,
not continued. The backbone is correctly inherited.

Read it as "150M late backbone + a 128-D head trained 30k steps". Compare it against
`protsent_late_150m/late`, never against its own `@0`. `run_after_training.sh` now scores both
phase-1 parents in the same sweep so the comparison is paired.

This can no longer happen silently: a head-width mismatch raises unless `--allow_head_reinit` is
passed (`test_head_size_mismatch_refuses_to_silently_reinitialise`).

## Open, unresolved

| gap | cost to close |
|---|---|
| Nothing isolates step count | 12.7 h — rerun proj128's exact recipe for 31k steps. Or 1.7 h for a discriminating probe: 4k steps of the *proportional* recipe from the proj128 checkpoint; landing near 0.599 implicates the mixture, near 0.629 the step count. |
| ProteinGym coverage | ~1.0 h/arm |
| n=1 seed everywhere | run-to-run variance is unmeasured for SCOPe; `run_seed_variability.sh` omits it, and `seed_variability.json` varies only the probe seed over fixed embeddings. The paired flash A/B moved SCOPe by up to 0.0045 at n=1, so −0.015 is probably not noise — but "probably" is the whole claim. ~9 seeds/arm to resolve <0.005. |
| Head-size confound in the 35M pilot | the 64-D arm ran 2,000 steps x batch 256, the 128-D arm 4,000 x 128 |
| No ESM-2 arm at the recommended config | — |
| ProtBench sample cap unpinned | 20k vs 100k moves Stability by +0.247 |
