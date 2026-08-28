# Audit: "more late-interaction training makes it worse"

Audited 2026-08-25. Every number recomputed from the per-query / per-assay `.npz` on disk.

## Verdict: CONTAMINATED — must rerun

The numbers reproduce exactly (SCOPe fold −0.0194, superfamily −0.0152; ProteinGym −0.0051\*\*,
−0.0129\*\*, −0.0027, −0.0125). The **label** is false. `build_late_results.py:77` names the pair
`("protsent_late_35m_prop_late", "proj128_late", "31k steps − 4k steps")`, but per the two
`runtime.json` files these are different *runs*, not two checkpoints of one. The continuation changed
sampler (round_robin → proportional), pool (capped 2M/file → 19.0M AFDB + 15.0M STRING), global batch
(128 → 256, `world_size` 1 → 2), backend (sdpa → vllm-flash-attn3) and `--compile` — all at once with
the step count. And the extra steps bought no extra supervision of the kind SCOPe measures:
round-robin gave the 4k arm 1/3 Pfam (≈170,667 Pfam pairs of 512,000); proportional gives the 31k arm
2.24% Pfam (≈177,441 of 7,936,000). **The 15.5× extra pairs are ~all AFDB/STRING; Pfam exposure is
flat.** Within the continuation, more steps is strictly *better*. The loss is a mixture effect taken
in the first 5,000 steps, not a step-count effect. The 150M "sign flip" is **REFUTED** outright.

## Provenance

Fixes: **(a)** bf16 masters `4dbc9aa` 08-24 21:18; **(b)** head-restore `8963237` 21:43; **(c)**
seed-reaches-head `4dbc9aa` 21:18.

| evidence | file | a | b | c | usable |
|---|---|---|---|---|---|
| 4k arm `proj128` | `models/late_interaction/protsent_late_proj128/runtime.json` (08-24 14:32) | before, but *predates the bug* (`702d6c6`, 17:05); RUNS.md:923 records fp32/sdpa, cos-vs-base 0.836 | before, N/A (hub model, no `1_Dense`) | **before** — head is an unseeded draw | yes |
| 31k arm `35m_prop` | `models/late_interaction/protsent_late_35m_prop/runtime.json` (08-25 05:13) | after | after — `logs/queue_a.log:1302` "continuing from the saved 128-D projection head" | after | yes |
| 35M SCOPe curve | `pilot_35m/scope/per_query_protsent_late_35m_prop@{0..30000}.npz` | after | after | after | yes; `@0` AP array is byte-identical to `per_query_protsent_late_proj128.npz` (max diff 0.0), confirming `@0` = the 4k checkpoint |
| ProteinGym | `pilot_35m/benchmarks/proteingym_partial/proteingym_*_{proj128_late,protsent_late_35m_prop_late}.npz` (08-25 08:13–08:35) | after | after | after | paired rows only; the `c8a4bad` length-division bug does **not** touch this pair (both sides MaxSim, where `sim/lens` is intended) |
| 150M continuation | `logs/queue_b.log:1525` — "saved projection is (64, 640) but this run asks for (128, 640); **keeping the fresh head**" | after | fires the *mismatch* branch | after | **no** — not a step-count contrast |
| pooled ProtBench | `pilot_35m/benchmarks/{knn,linear}/*.csv` | — | — | — | **absent**: covers `protsent_late` (2k pilot) and `esm2_late` only |

**Could not verify:** what path `proj128_late` points to in the ProteinGym files. Those npz predate the
provenance columns `c8a4bad` added; the name exists only as hand-typed text in `report_proteingym.py:38`
and no log of the invocation survives. The SCOPe identity *is* verified. The tree also mutated during
this audit: a parallel session quarantined the ProteinGym files to `proteingym_partial/` at 10:37–10:39,
and `run_late_bench.sh` (PID 1717363) is generating the missing pooled numbers now.

## Recomputed curve — SCOPe-40 eligible MAP, 20k-resample paired bootstrap over queries

35M continuation; `@0` = the 4k `proj128` checkpoint.

| step | fold | sfam | family | Δ fold vs @0 | Δ sfam vs @0 |
|---|---|---|---|---|---|
| 0 | 0.6288 | 0.7057 | 0.7087 | — | — |
| 5,000 | 0.5990 | 0.6798 | 0.6907 | −0.0298 [−0.0331,−0.0265] | −0.0260 [−0.0296,−0.0223] |
| 10,000 | 0.6044 | 0.6834 | 0.6951 | −0.0244 [−0.0275,−0.0213] | −0.0224 [−0.0261,−0.0188] |
| 15,000 | **0.6146** | **0.6943** | 0.7048 | −0.0142 [−0.0171,−0.0112] | −0.0114 [−0.0147,−0.0080] |
| 20,000 | 0.6108 | 0.6920 | 0.7058 | −0.0180 [−0.0211,−0.0150] | −0.0137 [−0.0175,−0.0101] |
| 25,000 | 0.6087 | 0.6901 | 0.7059 | −0.0201 [−0.0233,−0.0169] | −0.0156 [−0.0194,−0.0118] |
| 30,000 | 0.6094 | 0.6905 | 0.7062 | −0.0194 [−0.0226,−0.0162] | −0.0152 [−0.0190,−0.0113] |

**U-shaped, then flat — not monotonic-down.** All the loss is taken by step 5,000 (LR there is 9.39e-6,
94% of peak, per `train_log.csv`). From 5,000 on, more steps helps significantly: @30000 − @5000 =
**+0.0104** fold, **+0.0108** sfam, **+0.0155** family. Family never significantly loses at all
(@30000 vs @0: −0.0025 [−0.0084,+0.0034]). A small real late drift exists: @30000 − @15000 = −0.0052 fold.

**150M.** Against its own `@0` it gains +0.0065 sfam, as claimed — but `@0` is a *fresh random 128-D
head* on a backbone trained with a 64-D one, and that reset alone costs −0.0223 sfam. Against the
checkpoint it actually continued from (`protsent_late_150m/late`, sfam 0.7376) the arm at 25k steps is
**−0.0158 [−0.0198,−0.0119] sfam / −0.0221 fold** — same sign, same magnitude, same 15,000-step peak as
the 35M arm. Both arms agree; there is no scale-dependent flip.

**Noise floor.** The query bootstrap above (±~0.003 paired) is the wrong floor — it holds the run fixed.
The run-to-run floor is **unmeasured in this repo**: `results/benchmarks/seeds/seed_variability.json`
varies only the *probe* seed over fixed embeddings (sd = 0.0000 on most tasks), and
`run_seed_variability.sh` deliberately omits `scope40_retrieval`. The only proxy is the paired flash A/B
(RUNS.md:902–904), where a nuisance-config change moved SCOPe by up to 0.0045 at n=1. −0.015 is ~3× that,
so probably not pure noise — but n=1 on both sides, and that note says ~9 seeds/arm to resolve <0.005.

**Cross-benchmark.** ProteinGym agrees in sign on the same confounded pair (recomputed): dms_subs −0.0051
[−0.0090,−0.0012] (95 win/115 loss); clinical_subs −0.0129 [−0.0177,−0.0082] (449/705, plus **1,072 exact
ties** — per-gene AUCs on ~17 variants); dms_indels −0.0027 ns; clinical_indels −0.0125 ns (45/53 ties).
Effect sizes are tiny (|d|/sd 0.10–0.22) beside MaxSim-vs-cosine's +0.0258/+0.0552 on the same assays.
This is two views of one confounded weight pair, not two independent confirmations.

## What would settle it

**Rerun `proj128`'s exact recipe for 31,000 steps** — round-robin, capped pool
(`--max_pairs_per_file 2000000 --string_max_pairs 2000000`), single GPU, batch 128, sdpa, no compile,
`--save_steps 5000` + `watch_curve`. Only that isolates *steps* from mixture, batch and backend. At
proj128's measured 0.6775 steps/s: **≈12.7 h on one A100**. Checkpoint scoring is already automatic (~40 s each).

Cheaper:
- **Free, today:** relabel the claim to what the data supports — "the phase-2 proportional recipe costs
  −0.015 sfam against the phase-1 capped round-robin arm at equal Pfam exposure."
- **≈1.7 h:** 4,000 steps of the *proportional* recipe from `protsent_late_proj128/late` at batch 128,
  one GPU, sdpa. Landing near 0.599 implicates the mixture; near 0.629 implicates step count.
- **Drop the 150M row** until it is rerun with `--proj_dim 64` (or `protsent_late_150m` re-exported at
  128-D). Its baseline is currently a different model.
