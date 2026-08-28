# Late-interaction campaign — what is usable, what is dead, what needs rerunning

Hand-maintained ledger. `RESULTS.md` next to it is generated and holds the numbers; this holds
their standing. Read this before trusting, citing, or rerunning anything here.

Last reviewed 2026-08-27 (early morning). The r2 campaign is complete: all four init x size
arms trained to 10,000, plus a proj_dim ablation. See "r2 findings" below.


## Run naming convention (adopted 2026-08-25)

**`late-r{N}-{base}-{size}`** — recipe version, base model, parameter count.

| field | values |
|---|---|
| `r{N}` | **r1** = original recipe: cosine schedule, LR 1e-5 backbone / 1e-4 head (two param groups), `combinations` pair sampling. **r2** = clean recipe: `constant_with_warmup`, single LR group at 5e-5, `disjoint` pair sampling, per-rank negatives at batch 256. |
| `{base}` | `esm2` = vanilla `facebook/esm2_t12_35M_UR50D` (identical weights to `Synthyra/ESM2-35M`). `protsentv2` = `GrimSqueaker/ProtSent-V2-*`. |
| `{size}` | `35m`, `150m` |

Example: `late-r2-protsentv2-35m` = clean recipe, ProtSent-V2 base, 35M.

The two runs predating this convention keep their names (renaming a live run breaks its
trainer, watcher, snapshotter and mark-bench at once); the legend below maps every run to what
it actually is. **Use the convention for every new run.**

## Run legend — what each name on disk actually is

| run dir | recipe | base | size | steps | standing |
|---|---|---|---|---|---|
| `vanilla35m_clean` *(= `late-r2-esm2-35m`)* | **r2** | vanilla ESM-2 | 35M | 10,000 | **done**, sfam 0.7040 @10k |
| `late-r2-protsentv2-35m` | **r2** | ProtSent-V2 | 35M | 10,000 | **done**, sfam 0.6939 @10k. Deadlocked twice on `--mini_batch_num_tokens` before running on `--mini_batch_size` |
| `late-r2-esm2-150m` | **r2** | vanilla ESM-2 | 150M | 10,000 | **done**, sfam 0.7288 @10k (0.7403 @8k) — best late model |
| `late-r2-protsentv2-150m` | **r2** | ProtSent-V2 | 150M | 11,000 | **done**, sfam 0.7235 @11k |
| `late-r2-esm2-150m-proj640` | **r2** | vanilla ESM-2 | 150M | 2,000 | **done**, proj_dim ablation: 640-d loses to 128-d |
| `protsent_late_proj128` | r1 | ProtSent-V2 | 35M | 4,000 | phase-1 reference, sfam 0.7057 |
| `protsent_late` | r1 | ProtSent-V2 | 35M | 2,000 | early pilot |
| `protsent_late_swap` | r1 | ProtSent-V2 | 35M | 4,000 | pair-symmetry ablation |
| `protsent_late_35m_prop` | r1 phase 2 | *continues* `protsent_late_proj128` | 35M | 31,000 | mixture-confounded, see retraction above |
| `esm2_late` | r1 | vanilla ESM-2 | 35M | 2,000 | vanilla-base reference, fold 0.6000 |
| `protsent_late_150m` | r1 | ProtSent-V2 | 150M | 5,000 | phase-1 150M, sfam 0.7376 — superseded by `late-r2-esm2-150m` |
| `protsent_late_150m_prop` | r1 phase 2 | *continues* `protsent_late_150m` | 150M | 30,000 | head-reinit confounded, see above |
| `esm2_late_150m` | r1 | vanilla ESM-2 | 150M | 5,000 | vanilla-base 150M reference |

## r2 findings (2026-08-27) — paired bootstraps, not marginal CIs

Regenerate any of these with `python analyze_paired_effects.py --level superfamily`. They are
paired over the same SCOPe-40 queries; the marginal CIs in `scope_hierarchy.csv` overlap for
several contrasts that are separated cleanly here, so **do not compare arms by eye off that file**.

| effect | dAP (sfam) | CI95 | sig |
|---|---|---|---|
| ProtSent-V2 pretraining, 35M, frozen MaxSim | +0.2096 | [+0.1971,+0.2216] | yes |
| ProtSent-V2 pretraining, 150M, frozen MaxSim | +0.2743 | [+0.2626,+0.2863] | yes |
| MaxSim - cosine, same ESM2-35M weights | +0.0754 | [+0.0672,+0.0835] | yes |
| MaxSim - cosine, same V2-150M weights | +0.0490 | [+0.0444,+0.0533] | yes |
| **scale, ESM2, frozen MaxSim 35M->150M** | **-0.0286** | [-0.0415,-0.0160] | yes |
| scale, ProtSent-V2, frozen MaxSim 35M->150M | +0.0360 | [+0.0273,+0.0447] | yes |
| init after training: ESM2 - V2, 35M @10k | +0.0101 | [+0.0054,+0.0150] | yes |
| init after training: ESM2 - V2, 150M | +0.0173 | [+0.0125,+0.0221] | yes |
| late training on V2-35M vs frozen | +0.0108 | [+0.0047,+0.0169] | yes |
| late training on V2-150M vs frozen | +0.0038 | [-0.0022,+0.0098] | **no** |
| proj_dim 128 - 640, identical recipe @2000 | +0.0104 | [+0.0072,+0.0138] | yes |

Four conclusions, and the two that are easy to overstate:

1. **Pretraining is the dominant effect**, ~5x the next largest lever. MaxSim is how it is cashed in.
2. **Scale is an interaction, not a main effect.** Frozen ESM2 gets *worse* at 150M; frozen ProtSent
   gets better. `campaign/token_spread.py` measures why: ESM2's effective residue rank falls with
   scale (12.10 -> 10.54) while ProtSent's rises (15.50 -> 22.50), and MaxSim needs residues to be
   individually distinguishable. The negative ESM2 number was predicted from the rank before it was
   measured.
3. **128-d beats 640-d** at matched steps and identical flags, so the compression claim is a win,
   not a trade-off: 5x smaller index *and* higher MAP.
4. **"Late training adds nothing on top of ProtSent" is size-dependent** — true at 150M (ns here,
   and significantly *negative* on few-shot), false at 35M (+0.0108). Do not state it generally.

Benchmarks that do NOT separate these arms: the 6-task cheap sweep (all 8 cells within 0.024),
ProteinGym (every CI overlaps), CATH (n=150). They are do-no-harm checks, not evidence. SCOPe and
few-shot remote homology are the two that discriminate.

Never run on any r2 arm: the 22-task `paper` suite (`TASKS=paper ./run_late_bench.sh`), and
ProtBench's own ProteinGym path -- our ProteinGym numbers come from `late_interaction_eval.py`.

Results directories follow the same split: `clean_35m/` holds r2 output, `pilot_35m/` and
`pilot_150m/` hold r1. Per-query npz are prefixed with the run name, so they never collide.

## Log caveat

`logs/queue_clean.log` contains **two** runs concatenated: a `--gather_across_devices` variant
launched 13:42 and killed at ~step 150 (its 5.18 s/it figures are not the current config), then
the current run from 14:26. The gathered variant left no checkpoints, no `runtime.json` and no
result rows — verified — so only the log is shared. It was not split because the live trainer
holds an open fd at a byte offset in it.

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
