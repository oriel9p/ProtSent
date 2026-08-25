# ProteinGym-derived evaluation of late-interaction (MaxSim) scoring

**TL;DR.** Across all four ProteinGym-derived variant-effect settings (DMS + clinical,
substitutions + indels), residue-level MaxSim scoring beats pooled-cosine scoring of the *same
model* by margins whose 95% CIs exclude zero — the largest being +0.20 Spearman on DMS indels.
The effect is a property of the *scoring geometry*, not of our contrastive training: untrained
MaxSim over a model's native residue embeddings captures most of the gain.

## Protocol (and why these numbers are NOT leaderboard-comparable)

We score each variant by its similarity to its own wild type: `score(v) = sim(v, WT)`, where
`sim` is either mean-MaxSim over per-residue embeddings (late interaction) or cosine over
mean-pooled embeddings. For DMS sets we report mean per-assay Spearman against `DMS_score`; for
clinical sets, mean per-protein-group AUC against pathogenicity (score negated: more WT-like ⇒
less pathogenic). Data: `OATML-Markslab/ProteinGym_v1`; up to 500 variants per group, seeded;
sequences truncated at 512 residues; variants whose truncated form is byte-identical to the
truncated WT are excluded and counted per row (~4.7% of substitution mutations; leaving them in
creates blocks of exact ties that depress Spearman).

### Comparability to the leaderboard, itemised

Checked against ProteinGym's own scoring code (`proteingym/performance_DMS_benchmarks.py`,
`performance_clinical_benchmarks.py` on `main`), not the paper.

| Dimension | ProteinGym leaderboard | Ours | Comparable? |
|---|---|---|---|
| Score | mutant-vs-WT **log-likelihood ratio** from a generative/masked LM | **similarity to WT** in embedding space | **No — different quantity** |
| DMS metric | per-assay Spearman | per-assay Spearman | Yes |
| DMS aggregation | assay → **mean per UniProt_ID** → **mean per function category** → mean | now reported both ways (`mean_score`, `corrected_average`) | Yes, via `corrected_average` |
| Clinical substitutions | **AUC per gene, then averaged** | per-group mean AUC | Yes |
| Clinical indels | **all genes pooled, one global AUC** | now reported both ways (`aggregation=pooled`) | Yes, via `pooled` |
| Assay coverage | all 217 DMS substitution assays | **210** (7 dropped, see below) | **No — biased subset** |
| Variants per assay | all (2.47M total) | ≤500, seeded | **No — subsampled** |
| Sequence length | full | truncated at 512 residues | **No** |

**What we drop, exactly.** 7 of 217 DMS-substitution assays are excluded entirely, and they are
not a random sample: every one is a long protein whose mutated positions all lie past our
512-residue truncation, leaving zero variants distinguishable from wild type —
`A4_HUMAN_Seuma_2022` (770 aa), `CAPSD_AAV2S_Sinai_2021` (735), `ERBB2_HUMAN_Elazar_2016` (1255),
`KCNH2_HUMAN_Kozek_2020` (1159), `POLG_HCVJF_Qi_2014` (3033), `SCN5A_HUMAN_Glazer_2019` (2016),
`UBE4B_MOUSE_Starita_2013` (1173). Within the surviving 210 assays a further ~4.7% of mutations
are dropped for the same reason (counted per row as `n_variants_dropped_truncated`). **This biases
our averages upward relative to a full-length model**, since long proteins are excluded rather
than scored badly.

**Bottom line: the metric and aggregation now match; the score, the coverage and the truncation do
not.** These numbers are therefore *not* leaderboard entries. For scale only, leaderboard
zero-shot DMS-substitution averages run ~0.40–0.52 Spearman (ESM-2-650M ≈ 0.40, GEMME 0.455,
TranceptEVE ≈ 0.46); our best similarity-based score is ~0.33 (0.32 corrected) from a 35M encoder
on 210/217 assays. The tables' purpose is **ranking our own arms under one fixed protocol**, where
every comparison is paired and every arm suffers the identical truncation.

## Findings

1. **MaxSim > pooled cosine, everywhere, same model.** The paired within-model delta is positive
   with CI clear of zero on all four variants for every backbone tested. Largest on indels
   (+0.197 Spearman DMS, +0.14 AUC clinical): pooling averages an indel's frameshifted residues
   into one vector, while MaxSim aligns the unshifted prefix/suffix residue-by-residue. (Length
   alone is not the signal: |Δlength| scores 0.096 Spearman / 0.532 AUC.)
2. **Most of the gain needs no training.** Untrained MaxSim over native residue embeddings
   (480-D/640-D) captures the bulk of the improvement on every backbone — e.g. vanilla ESM-2 35M
   goes 0.248 (cosine) → 0.341 (untrained MaxSim) on DMS substitutions.
3. **Contrastive sentence training helps cosine, not MaxSim.** ProtSent-V2's pooled cosine beats
   vanilla ESM-2's (0.282 vs 0.248), consistent with its training objective; under MaxSim the
   ranking flips (0.316 vs 0.341) — sentence-level contrastive training slightly *blurs*
   residue-level identity.
4. **Late-interaction training compresses, not improves.** The trained 128-D head (proj128)
   scores within noise of untrained 480-D MaxSim on ProteinGym while using 3.75× less storage;
   its value here is compression + the SCOPe retrieval gains, not variant effects.
5. **More contrastive pretraining moves the two views in opposite directions.** The 31k-step
   proportional continuation improved its pooled view (+0.018 Spearman) and *degraded* its
   MaxSim view (−0.005; clinical −0.013) — the same pattern as SCOPe (−0.015 superfamily MAP).
   Sentence-level contrastive training trades residue-level structure for pooled quality.

Tables: `PROTEINGYM_TABLES.md` (generated by `report_proteingym.py`; regenerate after any rerun).
Per-group score vectors: `proteingym_<variant>_<model>.npz` — all deltas above are paired over
identical group sets with bootstrap CIs (2,000 resamples).

Caveats: clinical_indels has only 53 groups (wide CIs); DMS indels 63 assays; the 150M rows
use a shorter late-training recipe (64-D, 5k steps) than the 35M rows, and
`protsent_late_150m_prop` (128-D, 30k) completes the grid when its run finishes.
