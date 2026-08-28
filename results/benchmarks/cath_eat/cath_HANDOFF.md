# Handoff: CATH midnight-zone (ProtTucker/EAT) benchmark added


---

## 1. Why this was run

`rebuttal/FINAL_rebuttal.md` already concedes the gap in our own words:

- `:137` — ProtTucker is "the closest analogue to our protocol… the comparison we would most
  want to add"
- `:216` — "the real gap, its protocol being ours"

Both ProtSent and ProtTucker contrastively post-train a **frozen** pLM's per-protein
embeddings and score by 1-NN annotation transfer. We had no CATH data, no EAT code and no
numbers. Now we do, for both scales and both ProtSent versions.

Reference: Heinzinger, Littmann, Sillitoe, Bordin, Orengo, Rost, *Contrastive learning on
protein embeddings enlightens midnight zone*, NAR Genomics and Bioinformatics 4(2) lqac043,
doi:10.1093/nargab/lqac043. Code: github.com/Rostlab/EAT (GPL-3.0; we reimplemented the
scorer rather than vendoring it).

## 2. What the benchmark is

Transfer a CATH v4.3 label from **69,605 lookup domains** to **219 query domains** that were
filtered so **no alignment-detectable relative exists in the lookup set** (HVAL ≤ 0). That
filter is what makes it a midnight-zone test rather than a lookup exercise.

Method: mean-pooled per-protein embedding, 1 nearest neighbour by **Euclidean** distance,
transfer its label. Scored at all four CATH levels (Class / Architecture / Topology /
Homologous superfamily) over the queries answerable at each level: **219 / 219 / 210 / 150**.

Splits are the authors' own, verbatim from Rostlab/EAT, so numbers sit on the same axis as
their Table 1. Data: HF dataset `GrimSqueaker/cath43-eat` (already published; built by
`ProtBench/scripts/build_cath_eat_dataset.py`).

## 3. The validity check that licenses everything else

**MMseqs2 scores 34.7 here; the paper reports 35.** Same tool, same flags (`-s 7.5 -e 10`),
our pipeline, their data. Reproducing a published baseline within rounding validates splits,
labels, the H-level denominator of 150, the transfer rule and the scoring.

Floor check: 3-mer frequency vectors score **0.00**, matching the published random baseline of
0 at H — no amino-acid-composition shortcut in the task.

## 4. Results — full C/A/T/H, accuracy %, ±95% CI

95% CIs are 1.96 × bootstrap SE over 1,000 resamples (the paper's procedure; EAT's code
defaults to 10,000 and returns a bare SE their tables multiply by hand).

| arm | dim | C | A | T | H | mean |
|---|---|---|---|---|---|---|
| ESM2-35M (frozen base) | 480 | 78.5 ± 5.4 | 54.3 ± 6.5 | 42.4 ± 6.7 | 40.7 ± 7.8 | 54.0 |
| ProtSent-V1-35M | 480 | 81.7 ± 5.1 | 64.4 ± 6.4 | 45.7 ± 6.7 | 50.7 ± 8.0 | 60.6 |
| **ProtSent-V2-35M** | 480 | 82.2 ± 5.0 | 64.4 ± 6.7 | 53.3 ± 6.8 | **56.7 ± 8.0** | 64.1 |
| ESM2-150M (frozen base) | 640 | 74.0 ± 6.0 | 53.0 ± 6.6 | 41.0 ± 6.5 | 43.3 ± 8.2 | 52.8 |
| ProtSent-V1-150M | 640 | 83.6 ± 4.9 | 68.0 ± 6.3 | 56.2 ± 6.7 | 58.0 ± 8.0 | 66.4 |
| **ProtSent-V2-150M** | 640 | 84.0 ± 4.8 | 69.9 ± 6.3 | 57.1 ± 6.8 | **62.7 ± 7.8** | 68.4 |
| ESM-C-300M (vanilla) | 960 | 66.7 ± 6.1 | 36.1 ± 6.4 | 21.9 ± 5.5 | 18.7 ± 6.3 | 35.8 |
| ISM-C-300M (struct-distilled) | 960 | 80.8 ± 5.3 | 47.9 ± 6.6 | 29.5 ± 6.1 | 25.3 ± 7.2 | 45.9 |

Alignment baselines, H level only (the task scores `cath_h`; their C/A/T were not re-measured):

| baseline | H | note |
|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 34.7 | reproduces paper's 35 |
| phmmer (single-sequence) | 38.0 | **not** the paper's HMMER row |
| 3-mer frequencies | 0.00 | floor |

## 5. Comparison to the paper — three tiers, keep them separate

### Tier 1: true reproduction
MMseqs2 34.7 vs 35. Only row where we re-ran their method.

**Do not compare our phmmer 38.0 to their HMMER 77.** Theirs used CATH-Gene3D *profile* HMMs;
ours is single-sequence phmmer. Different method, not a failed reproduction.

### Tier 2: same benchmark, different backbone
Their Table 1 already compares ProtBERT / ESM-1b / ProtT5 / ProSE across architectures and
scales, so adding our rows is the table's intended use. Caveat: their embedding code, not ours.

| row | C | A | T | H | mean |
|---|---|---|---|---|---|
| Random | 29 | 9 | 1 | 0 | 10 |
| MMseqs2 | 52 | 36 | 29 | 35 | 38 |
| HMMER (CATH-Gene3D profiles) | 70 | 60 | 59 | 77 | 67 |
| ProtBERT raw | 67 | 38 | 22 | 18 | 36 |
| ESM-1b raw (650M) | 79 | 61 | 50 | 57 | 62 |
| ProtT5 raw (3B) | 84 | 67 | 57 | 64 | 68 |
| ProtTucker(ProtBERT) | 81 | 52 | 37 | 39 | 52 |
| ProtTucker(ProSE-MT) | 87 | 68 | 53 | 55 | 66 |
| ProtTucker(ESM-1b) | 87 | 68 | 59 | 70 | 71 |
| ProtTucker(ProtT5) | 89 | 75 | 64 | 76 | 76 |
| ProtTucker(ProtT5, train11M) | 88 | 77 | 68 | 79 | 78 |
| **ProtSent-V2-150M (ours)** | **84.0** | **69.9** | **57.1** | **62.7** | **68.4** |
| **ProtSent-V2-35M (ours)** | **82.2** | **64.4** | **53.3** | **56.7** | **64.1** |

Readings that hold:

- **ProtSent-V2-150M matches raw ProtT5 at every level** (84.0/69.9/57.1/62.7 vs 84/67/57/64;
  mean 68.4 vs 68) at **~20× fewer parameters**. Differences are inside the CIs.
- ProtSent-V2-150M's mean exceeds ProtTucker(ProtBERT) 52 and ProtTucker(ProSE-MT) 66; it is
  below ProtTucker(ESM-1b) 71 and ProtTucker(ProtT5) 76.
- Versus alignment, we reproduce the paper's own shape: embeddings win at coarse levels,
  alignment wins at fine. We beat HMMER-profiles at C (84.0 vs 70) and A (69.9 vs 60), lose at
  T (57.1 vs 59), lose clearly at H (62.7 vs 77). **Consistent with our standing position that
  alignment leads at top-1** — this benchmark does not overturn it.
- We beat MMseqs2 at every level.

### Tier 3: method effect with backbone controlled (cleanest)

| gain over its **own** frozen base | C | A | T | H |
|---|---|---|---|---|
| ProtTucker − ProtT5 (published) | +5 | +8 | +7 | **+12** |
| ProtSent-V2-35M − ESM2-35M | +3.7 | +10.1 | +10.9 | **+16.0** |
| ProtSent-V2-150M − ESM2-150M | +10.0 | +16.9 | +16.1 | **+19.4** |
| ISM-C − ESM-C (structure distillation) | +14.1 | +11.8 | +7.6 | +6.6 |

The gain **grows with level difficulty** — the same signature ProtTucker shows. Both our
H-level gains exceed the published +12.

**Statistical caveat that must travel with these numbers:** H-level CIs are ±8 (ours) and ±6–8
(theirs), and they are marginal, not paired. A single-row difference under ~8 points at H is
not resolvable. What is defensible is the *pattern* (gain rising with difficulty) and the size
of the two ProtSent deltas, not fine-grained rankings between adjacent rows.

## 6. Two attacks on the result — run, not just anticipated

**(a) It is not embedding normalisation.** ProtSent trains with a cosine objective while the
probe scores Euclidean distance on raw embeddings, so "ProtSent just has better norms" is the
first objection. Giving vanilla the same direction-only geometry for free:

| arm | raw H | L2-normalised H |
|---|---|---|
| ESM2-35M | 40.7 | 43.3 |
| ESM2-150M | 43.3 | 44.7 |
| ProtSent-V2-35M | 56.7 | 56.7 |
| ProtSent-V2-150M | 62.7 | 62.0 |

Normalisation is worth +2.7 / +1.3 to the vanilla bases and **nothing** to ProtSent. Scoring
vanilla at its best, the gap still stands at **+13.3** (35M) and **+18.0** (150M). Neither
model has a `Normalize` module — both are Transformer + mean pooling, identical pipelines.

This is the CATH counterpart to the SCOPe-40 whitening control, and it lands the opposite way:
on SCOPe-40 whitening recovered most of the kNN gain, here normalisation recovers almost none.
Worth stating explicitly rather than letting the two look inconsistent.

**(b) ESM2 scale does not buy this.** Our vanilla 35M (40.7) sat oddly close to ProtBench's
recorded ESM2-650M (42.7). Measured directly:

| model | H |
|---|---|
| ESM2-35M (FastPLM load path) | 40.7 |
| ESM2-35M (`facebook/`, plain HF path) | 40.7 |
| ESM2-150M | 43.3 |
| ESM2-650M (`facebook/`) | 42.7 |

Two conclusions. The two load paths give **identical** results on the same weights → no
code-path artifact. And the flatness is real: ESM2 goes 40.7 → 43.3 → 42.7 from 35M to 650M,
non-monotone. **18× the parameters buys ~2 points; contrastive post-training on the 35M buys
16.** Also confirms ProtBench's documented 42.7 was correct.

## 7. ESM-C / ISM-C (reviewer jVGf's structure-informed-pLM ask)

`Synthyra/ESMplusplus_small` *is* vanilla ESM-C-300M, so it is the matched control for
ISM-C-300M at fixed architecture and parameter count.

- ISM-C beats its own vanilla base at every level (+14.1 / +11.8 / +7.6 / +6.6) — structure
  distillation works, same direction as ProtSent's deltas, smaller at H.
- **Both 300M ESM-C variants score below our 35M ESM2 vanilla (40.7).** ESM-C and ESM2 are
  different pretraining runs with different embedding geometry; this is not a contradiction,
  but it means ESM-C is a weak foundation for 1-NN Euclidean transfer on CATH and does not
  raise the bar our arms must clear. Do not spin this as "we beat ESM-C" — it is a
  cross-family, cross-geometry comparison.

## 8. Open risk — state it before a reviewer finds it

**CATH-specific decontamination was never run.** `decontam_report.json` shows the 40% identity
/ 80% coverage filtering targeted the benchmark test sets in use at the time
(`remote_homology` / fold-prediction, PPI) — **not** CATH test219. So we cannot currently claim
test219 has no relatives in the AFDB / Pfam / STRING data ProtSent was contrastively trained on.

Measuring it needs a fresh MMseqs2 search against a ~126M-sequence corpus; the decontamination
run kept hit tables but no reusable index. Not started (multi-hour job, deliberately not
launched unprompted).

Two mitigations, to state rather than hide:

- ProtTucker is **explicitly supervised on CATH labels** (its best row adds 11M Gene3D
  sequences). ProtSent's supervision is Pfam / AFDB / STRING clustering — correlated with
  structural family but never naming a CATH label. Ours is the *weaker* supervision.
- The vanilla base and the ProtSent arm share identical pretraining exposure; only the
  contrastive stage differs. The delta is what that stage bought.

## 9. Artifacts on disk

Results (ProtSent repo):
```
results/benchmarks/cath_eat/CATH_COMPARISON.md   <- narrative writeup, the main document
results/benchmarks/cath_eat/CATH_LEVELS.md       <- generated C/A/T/H table + paper Table 1
results/benchmarks/cath_eat/cath_levels.json     <- machine-readable, incl. both masking counts
results/benchmarks/cath_eat/<arm>/*.csv          <- per-arm suite output, 8 arms
results/benchmarks/cath_eat_controls/<arm>/*.csv <- L2 / scaling / kmer controls, 7 arms
logs/bench_cath/*.log                            <- every run's log
```
Arm tags: `esm2_35m`, `protsent_v1_35m`, `protsent_v2_35m`, `esm2_150m`, `protsent_v1_150m`,
`protsent_v2_150m`, `esmc_300m`, `ismc_300m`.

Scripts created (ProtSent repo, uncommitted):
```
run_benchmarks_cath.sh            <- 8-arm H-level sweep, skip-if-complete
run_benchmarks_cath_controls.sh   <- L2, scaling, floor controls
```

Scripts created / changed (ProtBench repo, uncommitted):
```
cath_levels.py                        NEW  full C/A/T/H scorer, has --selfcheck
tests/test_unknown_residue_tokens.py  NEW  11 tests for the OOV fix
model_utils.py                        MOD  +_FallbackVocab, +patch_unknown_residue_tokens
protein_benchmark_suite.py            MOD  load_model wrapper + embed_dataset signature guard
tests/test_embed_cache_{reuse,pid_scoping}.py  MOD  stubs declare the legacy signature
```
Full ProtBench suite: **169 passed**, 0 failed, after the changes.

## 10. Two ProtBench bugs found and fixed (both were silent)

The suite catches per-task exceptions into an `Error` column and **still exits 0**, so a sweep
reports success with an empty results table. Both of these presented as "it worked".

1. **`embed_dataset` signature drift.** The suite called the legacy ESM++ signature
   (`sequences=`, `max_len=`); current FastPLM builds ship `embed_dataset(inputs, *, pooling=,
   ...)`, raising `missing 1 required positional argument: 'inputs'`. Fixed by inspecting the
   signature and falling through to the generic batched path (ported from ProtSent).
2. **Out-of-vocabulary residues.** FastPLM's tokenizer raises `KeyError` on any character
   absent from the vocabulary instead of falling back to unk. Lookup69k domain **`9pcyA00`
   carries two NUL bytes at residue 99** (corrupt in Rostlab/EAT's own FASTA) and killed the
   entire task. Ported `patch_unknown_residue_tokens`, applied in a `load_model` wrapper so it
   covers all eight load paths.

**Trap worth recording:** probing `AutoTokenizer` reports that problem #2 does not exist. The
embedding path uses `model.tokenizer`, a *different object*, and that one raises. Checking the
wrong tokenizer cost a failed sweep.

Third gotcha, config not code: `from_pretrained_with_flash` auto-selects `flash_attention_2`
whenever flash-attn is importable, but the models are fp32-resident → `'flash_attention_2'
supports only manifest-declared dtype(s) bfloat16`. Both scripts export
`PROTEIN_BENCH_ATTN_IMPLEMENTATION=sdpa`, which is also what our existing numbers ran under.

## 11. Protocol details worth carrying into the paper text

- **Probe.** `-p knn --knn_k 1` = `KNeighborsClassifier(n_neighbors=1, metric="euclidean",
  algorithm="brute")` with **no** feature standardisation — literally EAT's method. The linear
  probe standardises features and would fit ~6,500 classes over 69,605 rows; a different
  experiment, not comparable.
- **`-e test` is mandatory.** The suite defaults to `--eval_split validation`, and this dataset
  *has* a validation split (EAT's val200), so a bare run silently scores the wrong thing.
- **The hierarchy cascade is free.** The paper counts a hit at H only if C, A, T also matched.
  Labels are dotted *prefix* strings, so exact match at a level already implies every coarser
  level matched. No extra logic needed.
- **Masking.** We use the paper's rule (singleton across lookup ∪ test), reproducing their
  219/219/210/150 exactly. A stricter reading (label must actually be *in* the lookup set)
  gives **208** at T, identical at the other three levels. Both recorded in `cath_levels.json`
  as `n_answerable` / `n_answerable_strict`. Asserted at runtime — a mismatch aborts.
- **Two independent code paths agree.** The suite's `cath_eat` task and the standalone
  `cath_levels.py` compute H separately; equality is asserted per arm. All 8 arms: OK.
- **Truncation.** `max_length=1024` clips 7 of 69,605 lookup sequences (0.01%) and 0 queries.
  The paper did not truncate.

## 12. Cost

~10 GPU-min for the six ESM2 arms, ~6 for the two 300M arms, ~10 for controls. **Under one
hour total**, against the 10 GPU-hour ceiling that was set. Whole cost is embedding 69,605
lookup sequences once per model; `--cache_embeddings` lets the H-level sweep and the
full-table scorer share embeddings instead of paying twice.

All of it ran on **GPU 2 only**. GPUs 0,1,3–7 ran an unrelated protJepa job throughout and
were never touched.

## 13. Reproduce

```bash
bash run_benchmarks_cath.sh                       # H level, 8 arms
python /home/ddofer/ProtBench/cath_levels.py      # full C/A/T/H, same 8 arms
python /home/ddofer/ProtBench/cath_levels.py --selfcheck
bash run_benchmarks_cath_controls.sh              # L2 / scaling / floor
```

## 14. Suggested next steps (none started)

1. **CATH decontamination measurement** (§8) — the one real gap. Multi-hour MMseqs2 job.
2. **C/A/T for the alignment baselines** — we only have H. Needs `cath_c/a/t` task variants;
   would complete the MMseqs2 reproduction across all four levels.
3. **ProtSent-V1 vs V2 framing** — V2 beats V1 at both scales (56.7 vs 50.7; 62.7 vs 58.0),
   consistent with V2's decontaminated retrain being a genuine improvement, though the gaps are
   inside the CIs individually.
