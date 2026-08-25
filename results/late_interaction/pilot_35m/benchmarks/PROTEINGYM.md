# ProteinGym-derived evaluation of late-interaction (MaxSim) scoring

**TL;DR.** On **substitutions** (DMS and clinical), residue-level MaxSim scoring beats
pooled-cosine scoring of the *same weights* by margins whose 95% CIs exclude zero — up to
+0.093 Spearman on vanilla ESM-2 35M. The effect is a property of the *scoring geometry*, not of
our contrastive training: untrained MaxSim over a model's native residue embeddings captures most
of it, and late-interaction training adds little (or hurts, on vanilla ESM-2).

> ### ⚠️ The two INDEL comparisons are withdrawn pending a rerun
> A length division was applied to **both** scorers, not just MaxSim. `maxsim_against_one` returns
> a **sum** over query residues, so dividing by length is the intended mean-MaxSim;
> `cosine_matrix` normalizes, so cosine is **already length-invariant** and dividing it by length
> is a corruption. Where every variant in a group has the same length — all substitution sets —
> it is a constant per group and changes no rank, so **the substitution numbers stand**. Indel
> sets have varying lengths, so their pooled-cosine baseline was damaged and the two indel deltas
> (+0.197 Spearman, +0.142 AUC) are **not interpretable**. Fixed in `late_interaction_eval.py`
> (divide only when `scoring == "maxsim"`); the affected rows need rescoring before use.

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
`UBE4B_MOUSE_Starita_2013` (1173). Variants lost to truncation *within* the surviving groups,
per variant set (`n_variants_dropped_truncated` / `n_variants_scored`):

| variant set | dropped | scored | loss |
|---|---:|---:|---:|
| clinical_indels | 47 | 251 | **15.8%** |
| clinical_substitutions | 17,775 | 38,207 | **31.8%** |
| dms_indels | 26 | 11,501 | **0.2%** |
| dms_substitutions | 4,595 | 97,553 | **4.5%** |

**This biases our averages upward** relative to a full-length model: long proteins are excluded
rather than scored badly. The clinical-substitution loss is ~7× the DMS one, so the clinical AUCs
carry that bias hardest.

**clinical_indels is too small to headline.** Of 1,555 protein groups, **53** survive (both classes
present, ≥2 variants visible past truncation) — 3.4% of the benchmark — holding **251 variants
total, median 3 per group**. A per-group AUC on 3 points can only take values in {0, 0.5, 1}. Treat
that row as a smoke test, not a result.

**Bottom line: the metric and aggregation now match; the score, the coverage and the truncation do
not.** These numbers are therefore *not* leaderboard entries. For scale only, leaderboard
zero-shot DMS-substitution averages run ~0.40–0.52 Spearman (ESM-2-650M ≈ 0.40, GEMME 0.455,
TranceptEVE ≈ 0.46); our best similarity-based score is ~0.33 (0.32 corrected) from a 35M encoder
on 210/217 assays. The tables' purpose is **ranking our own arms under one fixed protocol**, where
every comparison is paired and every arm suffers the identical truncation.

## Findings

All numbers below are DMS-substitutions Spearman over 210 assays unless stated; every delta is
paired over identical group sets with a 2,000-resample bootstrap. Full tables in
`PROTEINGYM_TABLES.md`.

1. **MaxSim beats pooled cosine on identical weights, on every backbone and every variant.**
   Vanilla ESM-2 35M: 0.248 → 0.341 (**+0.093** [+0.083, +0.104]). ProtSent-V2 35M: 0.282 → 0.316
   (**+0.034**). Late-31k arm: 0.300 → 0.325 (**+0.026**). Clinical substitutions AUC, same
   pattern: ESM-2 0.598 → 0.741. Nothing differs between the compared arms except the scoring
   function, so the mechanism — per-residue alignment instead of a single averaged vector — is the
   only available explanation.

2. **Most of that gain needs no training at all.** The largest single effect in these tables is
   untrained MaxSim over a vanilla ESM-2's native 480-D residue embeddings (+0.093). Late-interaction
   *training* adds only +0.014 on top of untrained MaxSim for ProtSent-V2 ([+0.009, +0.020]), and on
   vanilla ESM-2 it **hurts**: −0.015 ([−0.028, −0.003]). The trained 128-D head's value here is
   compression (3.75× smaller index at parity), not accuracy.

3. **Sentence-level contrastive training helps cosine and hurts MaxSim.** ProtSent-V2 beats vanilla
   ESM-2 under cosine (0.282 vs 0.248) — its objective — but *loses* under MaxSim
   (0.316 vs 0.341, **−0.025** [−0.038, −0.012]). V2.5 improves cosine further (0.308). Pooling
   into one vector is what V2 optimises; doing so appears to blur residue-level identity.

4. **More contrastive pretraining moves the two views apart.** The 31k-step proportional
   continuation improved its pooled view (0.282 → 0.300) while degrading its MaxSim view
   (0.330 → 0.325, **−0.005** [−0.009, −0.001]; clinical substitutions **−0.013**). The same sign
   appears on SCOPe (−0.015 superfamily MAP, −0.019 fold). Four benchmarks, one direction.

5. **Indels — withdrawn.** See the warning above. The MaxSim side of those numbers is sound; the
   pooled-cosine side was divided by a varying length and must be rescored before any indel claim
   is made. The mechanism remains plausible (pooling averages frameshifted residues into one
   vector while MaxSim still aligns the unshifted prefix and suffix) — it is simply not yet
   measured. Note also that the length-only control quoted earlier tested `|Δlength|` against the
   label, which is the wrong control: the right one is the cosine arm rescored without the
   division.

6. **Model size does not rescue pooling.** ESM-2 150M under cosine scores 0.244, no better than
   35M's 0.248 — while switching 35M to MaxSim gets 0.341. The scoring geometry matters more here
   than a 4× parameter increase.

**Reproducing these numbers.** Regenerate the tables with
`uv run --no-sync python report_proteingym.py > PROTEINGYM_TABLES.md`; they are derived from the
per-assay `.npz` files, never hand-edited. Known gaps in the artifacts, stated rather than hidden:
the CSV rows predate the `aggregation`, `corrected_average`, `kind`, `path`, `proj_dim` and
`dataset_revision` columns the code now emits, so arm identity for the existing rows still rests
on the operator-chosen `--models NAME=KIND:PATH` label; the ProteinGym snapshot was **not pinned**
for these runs (`--dataset_revision` now exists for that); per-arm CIs used 1,000 bootstrap
resamples while the paired deltas use 2,000; and no multiplicity correction is applied across the
comparisons. `../RESULTS.md` predates the corrected-average work and still describes a plain
group mean — this file supersedes it on that point.

Tables: `PROTEINGYM_TABLES.md` (generated by `report_proteingym.py`; regenerate after any rerun).
Per-group score vectors: `proteingym_<variant>_<model>.npz` — all deltas above are paired over
identical group sets with bootstrap CIs (2,000 resamples).

Caveats: clinical_indels has only 53 groups (wide CIs); DMS indels 63 assays; the 150M rows
use a shorter late-training recipe (64-D, 5k steps) than the 35M rows, and
`protsent_late_150m_prop` (128-D, 30k) completes the grid when its run finishes.
