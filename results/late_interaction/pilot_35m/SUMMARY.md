# Late-interaction pilot (35M) — SUMMARY

**Setup.** Two arms trained with the identical recipe: `Synthyra/ESM2-35M` (vanilla) and
`GrimSqueaker/ProtSent-V2-35M`, each + fresh 64-D residue projection, `<cls>`/`<eos>` masked,
per-residue L2 + MaxSim, CachedMultiVectorMNRL (scale 1.0), round-robin over decontaminated
Pfam / AFDB / STRING-15M pairs, 2,000 steps @ 128/device on 2×A100 = **512k pairs, ~46 min per arm**
(bs sweep: 80.6 pairs/s single-GPU, 155–190 on 2 GPUs; peak 12.4 GB). Backbone LR 1e-5,
projection LR 1e-4, bf16, max_seq_length 512, seed 42. Step-0 dense-view parity: **exact (0.0)** both arms.
All numbers below: SCOPe-40 (`tattabio/scope40_test`, n=2,207), **eligible-query** metrics
(labels with ≥1 non-self gallery member; n=1,693 / 2,020 / 2,116 at family / superfamily / fold).
Full tables: `scope/scope_hierarchy.csv`, curves `scope/scope_checkpoint_curve.csv`,
paired bootstrap `scope/scope_pairwise_bootstrap.json`, pooled `benchmarks/knn/`.

## A. Does zero-shot late interaction help? — Yes, on both bases

MaxSim over native 480-D residue states vs pooled cosine, no training at all:

| eligible | ESM2 cos | ESM2 MaxSim | ProtSent cos | ProtSent MaxSim |
|---|---|---|---|---|
| family R@1 / MAP | .499 / .421 | **.601 / .545** | .685 / .646 | **.726 / .684** |
| superfamily R@1 / MAP | .698 / .398 | **.719 / .474** | .857 / .654 | **.878 / .683** |
| fold MAP | .358 | **.404** | .582 | **.601** |

Cost: ~30 s vs ~3 s scoring for the full 2,207² matrix (480-D, exact, one A100).

## B. Does late training help beyond MaxSim alone? — Depends entirely on the init

- **ESM2:** hugely. Family eligible R@1 .601 → **.742**, MAP .545 → **.698**; superfamily R@1
  .719 → **.896**; fold MAP .404 → **.600**. 2,000 steps turn vanilla ESM-2 into the best or
  near-best retrieval arm in the pilot.
- **ProtSent:** essentially no. Trained 64-D MaxSim lands at / slightly under its own zero-shot
  480-D MaxSim (superfamily eligible MAP .672 vs .683; paired Δap −0.011 [−0.015, −0.007];
  family Δ ≈ 0 n.s.). What training buys here is **compression**: 64-D vs 480-D residue vectors
  (7.5× smaller) and ~3× faster scoring (11 s vs 30 s) at equal quality. Note the random 64-D
  projection at step 0 costs ~0.08 MAP; training recovers it and stops.

## C. Does ProtSent provide a better initialization? — For the late model itself, no

Both arms converge to the same place; ESM2-Late even edges ahead on eligible R@1
(family .742 vs .719, superfamily .896 vs .880, fold .901 vs .898), ProtSent-Late marginally
ahead on fold MAP. Curves (superfamily eligible MAP): ESM2 .44 → .67, ProtSent .61 → .67, both
plateau by ~step 1,500. The contrastive signal, not the init, sets the late-model ceiling —
ProtSent's advantage is that it needs **zero** training to get there.

## D. Did late training damage the foundation embedding? — ProtSent: no; ESM2: mixed

Pooled 480-D mean-pool dense views, ProtBench `--fast --eval_split test -p knn --knn_k 3
--seed 42 -n 20000`, 18 usable tasks (conservation_flip / disprot lack local data in every arm —
dropped symmetrically; ties = |Δ| < 0.005):

- **ProtSent-Late dense view vs ProtSent-V2: 5 win / 8 tie / 5 loss** — preserved. Worst:
  Stability −.046, GB1 −.021; best: SS3 +.034, Solubility +.012, PPI +.011.
- **ESM2-Late dense view vs ESM2: 10 win / 1 tie / 7 loss** — net positive and
  retrieval-flavoured (pooled SCOPe R@10 +.04…+.06, Fluorescence +.054, GB1 +.053,
  Peptide-HLA +.046, Remote Homology +.028) but with real fitness/stability regressions
  (Stability −.162, β-lactamase −.054).

Few-shot Remote Homology (3-NN vote, shared seeded draws, N=1000): ESM2 .406 (cos) → .440
(zero-shot MaxSim) → **.548** (trained MaxSim); ProtSent .512 → .546 → .537.

## E. Probe-type dependence

Linear probes not run in the 3-hour window (KNN is the paper-style primary). `run_late_pilot_bench.sh`
runs the matching linear pass; keep it a separate table if run.

## Go / no-go

**GO, with a reframed target.** The pilot's strongest, bootstrap-supported findings:

1. **Zero-shot residue MaxSim is a free upgrade on ProtSent** (Δ vs pooled cosine at every level;
   e.g. family eligible ap +.038 with CI well clear of 0). Ship it as a scoring option — no training.
2. **The late head is a cheap retrieval specializer for vanilla ESM-2** (~45 GPU-minutes to match a
   34.8M-pair contrastive model on SCOPe retrieval) — a strong "capability per FLOP" result, with the
   known cost that pooled fitness/stability tasks suffer (partial-GO pattern: lower backbone LR or
   freeze lower layers next time).
3. **Training the head on top of ProtSent buys compression, not accuracy** (64-D ≈ native 480-D).
   A follow-up could freeze the backbone and train only the projection — likely enough, even cheaper.

Caveats: single seed, one run per arm, 2,000 steps, AFDB/STRING pair caps take the group-sorted
prefix (~500k AFDB clusters), family-level legacy numbers kept only for continuity with the paper.
