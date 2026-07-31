# Hostile fact-check: `COAUTHOR_BRIEF.md`

Every number, claim, path and script name checked against the repo on 2026-07-31
(`master`, working tree). Classification: **WRONG** (repo says something else),
**UNSUPPORTED** (no source in repo), **OVERSTATED / UNDERSTATED** (data licenses
less/more than is claimed), **STALE**, **VERIFIED**.

Headline: the retrieval tables, the bootstrap CIs, the anisotropy table, the
whitening control and the decontamination accounting **all check out**. The
damage is concentrated in §5.1 (layer sweep), one aggregate cell in §3, the
150M wall clock, and two credibility claims that are false as written.

---

## Ranked findings

### 1. §5.1 — the layer-sweep mitigation is overstated three ways, and the counterexample is in the same file

> "Layer 20 of 30 (~2/3 depth) is best for **every** model and is worth 5–6 points
> over the final layer, and at that layer ProtSent-V2 leads. Same pattern at 35M.
> So the final-layer linear probe understates every model and is not the
> instrument to settle this on."

Repeated in §6 item 4: "The final-layer default understates every model by 5–6
points on remote homology."

Source: `results/benchmarks/layer_probe_sweep_150m.json`,
`results/benchmarks/layer_probe_sweep.json`.

**(a) "5–6 points" is WRONG.** Layer 20 − layer 30 on remote homology:

| model | layer 20 | layer 30 | gain |
|---|---:|---:|---:|
| ESM-2 150M | 0.7357 | 0.6717 | **+6.4 pt** |
| ProtSent-V1 150M | 0.7400 | 0.7093 | **+3.1 pt** |
| ProtSent-V2 150M | 0.7500 | 0.7040 | **+4.6 pt** |

Correct range is **3.1–6.4 points**; only one of three models exceeds 5. And the
gain is *largest for stock ESM-2* — which cuts against the argument the sentence
is making.

**(b) "best for every model" holds on ONE task; the same JSON contains the
counterexample.** `layer_probe_sweep_150m.json` also has `stability`:

| model | layer 10 | layer 15 | layer 20 | layer 28 | layer 30 |
|---|---:|---:|---:|---:|---:|
| ESM-2 150M | 0.4490 | 0.2659 | **0.2188** | 0.5484 | **0.5866** |
| ProtSent-V1 150M | **0.6284** | 0.4596 | 0.4356 | 0.5583 | 0.5103 |
| ProtSent-V2 150M | **0.6229** | 0.5119 | 0.3844 | 0.4528 | 0.5340 |

On stability, layer 20 is the **worst** layer for ESM-2-150M and the final layer
is its **best**. The generalisation "the final-layer linear probe understates
every model" is contradicted by the other half of the one file it cites. The
sweep was run on exactly 2 tasks (`--tasks stability remote_homology`,
`layer_probe_sweep.py:142`); the brief generalises from n=1.

**(c) "Same pattern at 35M" is WRONG.** `layer_probe_sweep.json` (layers
4/6/8/10/12, remote homology): ESM-2-35M peaks at layer **6** (0.6703, final
0.6373 → +3.3 pt); ProtSent-V2-35M peaks at layer **8** (0.7033, final 0.6803 →
+2.3 pt). Different best layers, so the "one common layer is best for everyone"
pattern does *not* reproduce, and the gain is 2–3 points, not 5–6. V1-35M was
never in the 35M sweep at all (`layer_probe_sweep.py:42-45` has only ESM-2-35M
and ProtSent-V2).

**Fix:** "On remote homology, a mid-stack layer beats the final layer for every
model by 3–6 points (largest for stock ESM-2), and at layer 20 ProtSent-V2 leads.
The effect is task-dependent — on stability the final layer is best for stock
ESM-2 — so this is a caveat on the probe protocol, not a general correction."

---

### 2. §5.1 — the "layer 30 (benchmark default)" column is not the benchmark, and contradicts §3 by 7.8 points

Table header: `| remote homology, linear probe | layer 10 | layer 20 | layer 30 (benchmark default) |`

Same model, same task, same probe, two numbers in the same document:

| model | §5.1 "layer 30 (benchmark default)" | §3 table, "linear acc" | source of the §3 value |
|---|---:|---:|---|
| ESM-2 150M | 0.6717 | **0.7500** | `v2_150m/esm2_150m_linear/bench_Synthyra_ESM2-150M.csv` |
| ProtSent-V1 150M | 0.7093 | **0.7401** | `v2_150m/protsent_v1_150m_linear/…csv` |
| ProtSent-V2 150M | 0.7040 | **0.7503** | `v2_150m/protsent_v2_150m_linear/…csv` |

A reviewer who reads both sections sees a 7.8-point discrepancy on the vanilla
baseline and stops trusting the section.

Cause, from `layer_probe_sweep.py`: the sweep subsamples train to 8,000 and test
to **3,000 of 3,244** (`:143-144`, `:104-108`), pools raw `AutoModel`
`hidden_states[l]` (`:72-77`), and uses a plain multinomial
`LogisticRegression(max_iter=2000)` with `StandardScaler` (`:84-91`). The
benchmark uses the SentenceTransformer encoder over the full split with a
`OneVsRestClassifier(… liblinear)` probe. The layer-30 column is a *different
protocol at the same depth*, not the benchmark default.

**Fix:** relabel the column "final layer (sweep protocol)" and add one line: "the
sweep uses an 8k/3k subsample and its own probe, so its final-layer column is not
the benchmark number in §3; only the within-sweep layer comparison is meaningful."

---

### 3. §3 — "12 / 4 / 7 under 3-NN (median +0.0055)" does not reproduce, and UNDERSTATES the result

> "V2-150M vs V1-150M: 12 / 4 / 7 under 3-NN (median +0.0055), 7 / 6 / 10 under
> linear (median −0.0045)."

Traces only to `RUNS.md:220` and `rebuttal/NEW_EVIDENCE.md:877`. There is **no
JSON or CSV** in `results/` backing it (`comparison.json` / `COMPARISON.md` are
35M-only). Recomputed from `results/benchmarks/v2_150m/*/*.csv` with
`build_comparison.py`'s rules (declared `main_metric`, `TIE_TOL = 0.005`):

| probe | convention | result |
|---|---|---|
| 3-NN | declared metrics, 20 comparable tasks | **12 win / 4 tie / 4 lose, median +0.0090** |
| 3-NN | Accuracy fallback for the 3 AUC-less tasks, 23 tasks | **13 / 5 / 5, median +0.0087** |
| linear | declared metrics, 20 tasks | 5 / 5 / 10, median −0.0052 |
| linear | Accuracy fallback, 23 tasks | **7 / 6 / 10, median −0.0047** ← matches the brief |

The linear line reproduces exactly under the 23-task fallback convention. Under
that *same* convention the 3-NN line is **13 / 5 / 5, median +0.0087** — not
12/4/7, median +0.0055. As printed, "12 / 4 / 7" also sums to 23 while its win
and tie counts match the 20-task computation, i.e. it is internally impossible
under either convention.

**Damage:** this understates our own model. It reports 7 losses where the data
shows 4–5, and roughly half the median gain. It is the only 150M aggregate in the
document. Under 3-NN, V2-150M loses to V1-150M on exactly five tasks: Cloning
(−0.0162), Fluorescence (−0.0780), Solubility AUC (−0.0222), GB1 (−0.0283),
Remote Homology (−0.0435 acc). There is no way to reach seven.

**Fix:** "13 win / 5 tie / 5 lose under 3-NN (median +0.0087), 7 / 6 / 10 under
linear (median −0.0047)", and state the convention (Accuracy substituted for the
three tasks where multiclass AUC is unavailable). Also correct `RUNS.md:220-222`
and `NEW_EVIDENCE.md:877`, which carry the same error.

---

### 4. §4 — the seed-variability paragraph cherry-picks, in a sentence that promises it does not

> "**Seed variability** … Two halves, and quoting only one would misrepresent it:
> *Full-data evaluation is essentially deterministic.* Across 5 seeds: remote
> homology 0.5835±0.0000, metal-ion binding 0.7402±0.0000, solubility
> 0.5102±0.0000, stability 0.6435±0.0001. With a fixed test split and a
> deterministic probe over deterministic embeddings there is nothing to vary."

The four quoted values are **VERIFIED** (`results/benchmarks/seeds/seed_variability.json`).
But that file has **8 tasks per arm**, and one is not deterministic:

| task | ESM-2 35M | ProtSent-V1 | ProtSent-V2 |
|---|---|---|---|
| **Thermostability (FLIP)** | 0.4427 ± **0.0126** | 0.4696 ± **0.0172** | 0.4568 ± **0.0156** |

Two orders of magnitude above every other row, on means of 0.44–0.47. The
explanatory clause "there is nothing to vary" is directly false for it. The
reason is documented in `results/benchmarks/COMPARISON.md`: thermostability is
`auto_split=True`, "the suite makes a seeded 80/20 split of *train* and calls the
20% the test split" — so the split itself moves with the seed. (Fluorescence,
subcellular localisation and GB1 were also measured and are all sd = 0.0000; only
thermostability is the exception.)

A reviewer opening the cited file finds the exception in ten seconds, in a
paragraph that opened by warning against selective quotation.

**Fix:** "7 of the 8 tasks measured are exactly deterministic across 5 seeds
(sd = 0.0000–0.0001). The exception is thermostability (sd 0.013–0.017), whose
test split is itself a seeded 80/20 resplit of train — which is the point: what
varies is which proteins are in the test set, and the bootstrap estimates that."

---

### 5. §2 / §8 — "~17 h total" for the 150M retrain is WRONG; the real cost is ~26 h

> §2 table: `| ProtSent-V2-150M | decontaminated | 6x B300 | 3,890 (MAX_PAIRS_PER_CLUSTER=5) | ~17 h total |`
> §8: "This is the one control that would make the decontamination claim
> airtight; it is ~17 h of GPU time."

From the training logs:

| log | start | end / last step | duration |
|---|---|---|---|
| `logs/esm2_150m/protsent_esm2_150m_v2.log` | Wed Jul 29 16:14:26 | step **2583/3890** at elapsed **19:45:50**, then machine restart | 19 h 46 m |
| `logs/esm2_150m/protsent_esm2_150m_v2_resume.log` | Thu Jul 30 13:45:06 | crashed, `exitcode: 1` | ~24 m wasted |
| `logs/esm2_150m/protsent_esm2_150m_v2_resume2.log` | Thu Jul 30 14:09:04 | `3890/3890 [6:06:27]`, `train_runtime 2.199e+04` | 6 h 06 m |

**Actual ≈ 25 h 52 m of training** (19h46 + 6h06), ~28 h elapsed Jul 29 16:14 →
Jul 30 20:16. "~17 h" is `3890 × 15.74 s/it` — an extrapolation from the *fastest*
segment's tqdm average, which itself is inflated by the fast-forward over the
first 2,500 skipped steps. Segment 1 ran at **27.5 s/it**; that rate over 3,890
steps is 29.7 h.

**Damage:** the cell sits in a column headed "wall clock" next to the 35M's
genuinely measured "10 h 53 m" (`REBUTTAL_LEAKAGE.md:191`, `train_runtime`
3.917e+04 s), so it reads as measured. Worse, §8 uses it to price the single
control that would make the decontamination claim airtight — understating that
control by ~50% (≈157 GPU-hours on 6 B300s, not ≈102).

**Fix:** "~26 h wall clock (19 h 46 m before a machine restart at step 2583,
resumed from checkpoint-2500 for a further 6 h 06 m)" and in §8 "~26 h on 6
B300s, ≈150 GPU-hours".

---

### 6. §7 — "Every analysis script has a `--selfcheck`" is WRONG for the one that matters most

> "Every analysis script has a `--selfcheck` that runs its assertions on
> synthetic data."

`probe_gap_analysis.py` has none:

```
$ python probe_gap_analysis.py --selfcheck
probe_gap_analysis.py: error: unrecognized arguments: --selfcheck
```

It is the script behind **all of §5.2** — the anisotropy table and the
remote-homology whitening numbers, i.e. the most damaging finding in the
document, the one where "verified, not assumed" carries the most weight.

I ran the other ten named analysis scripts; **all pass**: `bootstrap_ci.py`,
`layer_probe_sweep.py`, `whiten_scope_control.py`, `verify_remote_homology.py`,
`scope_identity_partial.py`, `fewshot_seeds.py`, `verify_training_corpus.py`,
`scope_identity_correlation.py`, `mmseqs_baseline.py`, `hmmer_baseline.py` — each
prints `selfcheck ok`. (`decontaminate_pretrain.py` also has none, but it is a
data job, not an analysis script; the sentence as written still covers it.)

**Fix:** either add a `_selfcheck()` to `probe_gap_analysis.py` (its `whiten()`
and `spectrum_stats()` are trivially testable on synthetic anisotropic data —
`whiten_scope_control.py:130-155` is a ready template), or write "every analysis
script except `probe_gap_analysis.py` has a `--selfcheck`". The first is ~15
lines and is the better answer, given what §5.2 rests on.

---

### 7. Header — the two Hugging Face model links are UNSUPPORTED by the repo

> "Models: [GrimSqueaker/ProtSent-V2-35M](…), [GrimSqueaker/ProtSent-V2-150M](…)"

Grepping the whole repo for `GrimSqueaker` returns only two *dataset* ids in
`benchmark_tasks.py:220,230` (`SignalP_Binary`, `ProFET_NP_SP_Cleaved`) and one
note in `rebuttal/REVIEWS_and_notes.md:121`. Nothing references those two model
repos, and there is no upload script. `RUNS.md:10,13` gives the V2 models only as
local paths (`models/protsent_esm2_35m_v3/final`,
`models/protsent_esm2_150m_v2/final`); every script that loads them
(`probe_gap_analysis.py:36,38`, `whiten_scope_control.py:44,46`,
`run_benchmarks_150m.sh`) uses the local path. V1 is on the Hub as
`oriel9p/protsent-esm2-*`; V2 appears never to have been pushed.

I cannot check the Hub from here (no network). **Click both links before sending** —
if the models were not uploaded, this is the coauthor's first action and it 404s.

---

### 8. §2 — "what the paper's own ablations already favoured" is OVERSTATED for proportional sampling

> "dropping hard negatives and using proportional sampling are **what the paper's
> own ablations already favoured** (20/23 tasks at +7.9% without hard negatives vs
> 16/23 at +6.7% with). That is also the direct answer to Yi1G's complaint that
> the ablations do not support the submitted defaults — we acted on them."

The parenthetical supports the hard-negatives half only (**VERIFIED** — paper
Table 4, via `rebuttal/check_facts.md:224`, `RUNS.md:52`). For sampling, the
repo's own text says the opposite of "favoured":

> `rebuttal/REVIEWS_and_notes.md:412` (and `DRAFT_rebuttal.md:171`): "proportional
> sampling (+7.0%) is effectively comparable to round-robin (+6.7%). … we will
> **not claim** that either hard negatives or round-robin sampling is validated as
> generally superior."

A 0.3-point difference is a tie, and we have already committed in writing not to
call it a preference. Presenting a tie as "what the ablations favoured", in the
sentence billed as "the direct answer to Yi1G's complaint that the ablations do
not support the defaults", hands Yi1G the same complaint back.

**Fix:** "dropping hard negatives is what the paper's own ablations favoured
(20/23 at +7.9% vs 16/23 at +6.7%); proportional vs round-robin was a tie in the
ablation (+7.0% vs +6.7%), and we chose proportional for the retrain without
claiming the ablation settles it."

---

### 9. §3 vs §5.2 — the same five quantities appear with two different values

§3's SCOPe table takes the ESM-2/ProtSent rows from the benchmark CSVs, but the
MMseqs2 row and every CI from the bootstrap JSONs, which were computed on an
independent re-encoding and differ by ~1 query (1/1693 = 0.00059):

| quantity | §3 | §5.2 | CSV | bootstrap JSON |
|---|---:|---:|---:|---:|
| ESM-2 150M R@1 | 0.5535 | 0.5529 | 0.55346 | 0.55286 |
| ESM-2 150M MAP | 0.4236 | 0.4242 | 0.42359 | 0.42424 |
| ProtSent-V2 150M R@1 | 0.7431 | 0.7425 | 0.74306 | 0.74247 |
| ProtSent-V2 150M R@10 | 0.9368 | 0.9374 | 0.9368 | 0.93739 |
| ProtSent-V2 150M MAP | 0.7046 | 0.7048 | 0.70458 | 0.70481 |
| V2-150M remote hom. 3-NN | 0.6612 | 0.6606 ("raw") | 0.66122 | 0.66060 (`probe_gap_analysis.json`) |
| ESM-2 150M remote hom. 3-NN | 0.5194 | 0.5200 | 0.51942 | 0.52004 |

Every individual value is sourced and correct; the problem is that both appear.
Consequence: **subtracting §3's own table cells does not reproduce §3's own CI
point estimates** — 0.6852 − 0.5854 = +0.0998 vs the quoted +0.0986;
0.7431 − 0.6615 = +0.0816 vs +0.0809. A coauthor checking the arithmetic will
think one of them is wrong.

**Fix:** one footnote under the §3 table — "CIs are computed by `bootstrap_ci.py`
on an independent re-encoding of the same gallery; point estimates there differ
from the table by at most one query (±0.0006), so table subtraction will not
reproduce the deltas exactly."

---

### 10. §3 / §5.1 — "23 tasks" should be 20 (and the probe contrast rests on 17)

> §3: "### Aggregate over 23 tasks, both probes"; §5.1: "Across 23 tasks, both
> ProtSent models are neutral-to-worse…"

The counts quoted are verbatim from `results/benchmarks/COMPARISON.md` and are
**VERIFIED** (11/3/6 +0.0075; 10/3/7 +0.0041; 4/4/12 −0.0139; 2/7/11 −0.0107) —
but each is explicitly "**of 20 comparable tasks**": `antibiotic_resistance`,
`remote_homology` and `temperature_stability` have no multiclass AUC in any
embedding arm and are excluded. Every count in the brief sums to 20, under a
heading that says 23.

Additionally, 3 of those 20 are the *same measurement* in both probe tables —
`COMPARISON.md`: "The requested probe was ignored on: `ec_classification`,
`go_mf`, `scope40_retrieval` … Those rows are therefore **identical** in the kNN
and linear tables — they are one measurement printed twice, not two." So "the
probe decides the headline" rests on **17** genuinely paired tasks.

(§8's "the other 21 task test sets" is consistent with a 23-task suite and is fine.)

---

### 11. §4 — the few-shot deltas quoted are V1's, not V2's, and the experiment is 35M-only

> "on remote homology at N=1000, ProtSent leads ESM-2 by +0.133 (k-NN) and
> +0.089 (linear)"

`results/benchmarks/fewshot_seeds.json`, N=1000, remote homology:

| arm | k-NN | vs ESM-2 | linear | vs ESM-2 |
|---|---:|---:|---:|---:|
| ESM-2-35M | 0.1850 | — | 0.2878 | — |
| **ProtSent-V1** | 0.3185 | **+0.1335** | 0.3772 | **+0.0894** |
| **ProtSent-V2** | 0.2893 | **+0.1043** | 0.3552 | **+0.0674** |

The quoted +0.133 / +0.089 are **ProtSent-V1**'s. In a document whose subject is
V2, unqualified "ProtSent" reads as V2, and the higher pair is quoted. The whole
few-shot study is also **35M only** (`fewshot_seeds.json` has no 150M arm), which
the brief does not say.

**Fix:** "at N=1000 ProtSent-V2-35M leads ESM-2-35M by +0.104 (k-NN) and +0.067
(linear); V1-35M leads by +0.134 / +0.089. Few-shot was run at 35M only."

---

### 12. §1 — "re-filtered … against the benchmark test sets" contradicts §8

> §1: "We re-filtered the entire pretraining corpus against the benchmark test
> sets and retrained both models from scratch"
> §8: "Only `remote_homology` and `ppi_bernett` test sets were decontamination
> targets. The other 21 task test sets were **not** filtered against."

Two of twenty-three. §2 and §8 are precise and correct; §1 is the sentence a
reviewer quotes back. §2's own lead-in has the same slack ("Every pretraining
source searched against the benchmark test sequences", before naming the two
targets). One word fixes §1: "against the two benchmark test sets under
suspicion".

---

### 13. §5.2 — "both landing at the linear-probe score" is OVERSTATED by ~1.5 points, and mixes two distance metrics

> "whitening the vanilla embeddings takes ESM-2 150M k-NN from 0.5200 to
> **0.7346**, against ProtSent-V2's 0.6606 raw / 0.7343 whitened — both landing at
> the linear-probe score."

`results/benchmarks/probe_gap_analysis.json` — all four quoted numbers
**VERIFIED** (0.52004, 0.73459, 0.66060, 0.73428). But the linear-probe scores
are **0.7506** (ESM-2 150M) and **0.7497** (V2-150M): the whitened k-NN lands
~1.5 points *below*, not at.

Second, the comparison mixes metrics. `probe_gap_analysis.py:123-133`: `knn()`
defaults to **cosine**, so the whitened figures are cosine k-NN, while the "raw"
figures quoted are `knn_euclidean_raw`. On cosine throughout, V2-150M raw is
**0.6788**, not 0.6606 — which shrinks the "whitening recovers it all" gap from
7.4 to 5.6 points. Direction unchanged and the choice is conservative for the
argument being made, but anyone re-running the script sees both numbers.

**Fix:** "…to 0.7346, against ProtSent-V2's 0.6606 raw (0.6788 under cosine) /
0.7343 whitened — both within ~1.5 points of the linear-probe score (0.7506 /
0.7497)."

---

### 14. §2 — the STRING row and the row arithmetic contradict each other

The decontamination table ends STRING at **71,891,417**; two paragraphs later the
arithmetic uses **15,000,000** for STRING. Both are right — `REBUTTAL_LEAKAGE.md:901`
records "a deliberate STRING subsample (71,891,417 → 15,000,000, seed 42) taken to
fit a [step budget]", and the training file is `stringdb_train_15M.parquet` — but
the brief never says so, and as written the table and the sum disagree.

**Fix:** add "(subsampled to 15,000,000 pairs, seed 42, for the training mix)" to
the STRING row.

---

### 15. Rounding and attribution errors (low damage, but they are wrong)

| # | brief | correct | source |
|---|---|---|---|
| a | "shrinks the gap from **−0.0257** to **−0.0036**" | −0.0256 → −0.0037 (0.492944−0.518561; 0.705055−0.708730) | `verify_remote_homology_150m.json` |
| b | "R@10 −0.002, **p=0.93**" | p = **0.92** (0.9246) | `scope_identity_partial_150m_v2.json` |
| c | "median max-identity is 0.908" | 0.907725 ✓ — but that is identity to the **unfiltered** corpus; after dc40 it is 0.895 | `scope_identity_correlation_v2.json`; `REBUTTAL_LEAKAGE.md:604` |
| d | "HNXd made **three** specific requests" | four bolded items follow, and the fourth (identity-stratified SCOPe) is **Yi1G's**, not HNXd's. HNXd actually asked four things — CIs, seed variability, a linear baseline under label scarcity, *and* absolute scores in Table 5 — the last of which is not in the list | `rebuttal/REVIEWS_actual.md:24-26,35-37` |
| e | "(the −126.9% cell)" cited as explained by few-shot noise | the cell is Table 5 **Enzyme Catalytic Efficiency**, Spearman; the few-shot seed study covers remote_homology / solubility / metal_ion_binding / stability only. Plausible inference, unmeasured cell | `rebuttal/check_facts.md:214`; `fewshot_seeds.json` |
| f | "beats 3-NN in **almost every** model/task/N cell" | 42 of 48. All six exceptions are ProtSent arms (metal-ion V1 N=50 & N=1000, V2 N=1000; solubility V1 N=100; stability V2 N=50 & N=100) — mildly in our favour, worth one clause | `fewshot_seeds.json` |
| g | "on solubility and metal-ion binding it does not win" | metal-ion at N=1000 under k-NN: V1 +0.005, V2 +0.005 over ESM-2. Immaterial, but not zero | `fewshot_seeds.json` |

Note (a) and (b) both round in the direction that flatters the claim.

---

## VERIFIED — checked and correct

Everything below traces to a named file with a matching value.

**§2 decontamination.** Pfam 28,530,684 → 27,929,772 (2.11%), AFDB 135,404,259 →
126,301,607 (6.72%), STRING 76,070,154 → 71,891,417 (5.49%) — `REBUTTAL_LEAKAGE.md:453-455`,
`NEW_EVIDENCE.md:26-28`; all three percentages recomputed and correct.
Filter targets `fold_prediction[test]` 3,244 and `bernett_gold_ppi[test]` 3,022 —
`REBUTTAL_LEAKAGE.md:160,267-268`. 40% identity / 80% coverage / `--cov-mode 1` —
`decontaminate_pretrain.py:535-541` defaults `--min-seq-id 0.4 --cov 0.8 --cov-mode 1`,
docstring `:5-13`. Prefilter: exhaustive/GPU 100% recall, k-mer `-s 5.7` at 89.4%
for AFDB — `decontaminate_pretrain.py:277-281,509-512`, `REBUTTAL_LEAKAGE.md:481`.
**"Zero flagged sequences survived in any of the three files"** —
`training_corpus_verification.json`: `leaked_total: 0` for all three,
`all_clean: true`; `verify_training_corpus.py` semi-joins against the recorded
removal lists, and passes `--selfcheck`. Row arithmetic 27,929,772 + 126,301,607
+ 15,000,000 = 169,231,379 — arithmetic correct, matches the trainer's `total=`
(`REBUTTAL_LEAKAGE.md:233`). "SCOPe-40 deliberately not a filter target" —
`REBUTTAL_LEAKAGE.md:276,499`.

**§2 retrains.** 35M: 7× B300, 4,850 steps, 1 epoch over 169,231,379 rows,
`train_runtime` 3.917e+04 s = 10 h 53 m — `REBUTTAL_LEAKAGE.md:191`. 150M: 3,890
steps, `MAX_PAIRS_PER_CLUSTER=5`, 6 GPUs — `RUNS.md:13,54,116` (wall clock: see
finding 5). Config, both scripts: `cached_mnrl`
(`train_esm2_35m.sh:43`/`150m:81`), `BATCH_SIZE=1024` per device (`:41`/`:79`),
`--no_gather_across_devices` (`:92`/`:129`), `--matryoshka_dims 64 128 256`
(`:93`/`:130`), `flash_attention_2` (`:80`/`:109`), `MAX_SEQ_LENGTH=512`
(`:57`/`:88`), `--multi_dataset_sampler proportional` (`:110`/`:148`), no hard
negatives (`train_esm2_150m.sh:66`). The V1-vs-V2 confound caveat is correctly
stated and matches `RUNS.md:197-198`.

**§3 SCOPe table.** Every cell sourced. 1,693 of 2,207 eligible and the 0.767 cap
(1693/2207 = 0.76711) ✓. HMMER 0.6970 / 0.7809 / 0.4747 ✓ `hmmer_scope40.json`.
MMseqs2 0.6556 / 0.7401 / 0.4098 ✓ `scope40_bootstrap_ci.json`. All embedding
rows ✓ `v3/` and `v2_150m/` CSVs (provenance caveat: finding 9).

**§3 bootstrap CIs — all four rows correct, and "every delta excludes zero" is
correctly stated.**
V2-35M − V1-35M R@1 +0.0986 [+0.0762, +0.1211] ✓, MAP +0.0943 [+0.0814, +0.1074] ✓
(`scope40_bootstrap_ci.json`, both `excludes_zero: true`).
V2-150M − V1-150M +0.0809 [+0.0602, +0.1022] ✓, +0.0607 [+0.0477, +0.0735] ✓
(`scope40_bootstrap_ci_150m.json`).
V2-150M − HMMER +0.0455 [+0.0219, +0.0691] ✓, +0.2301 [+0.2111, +0.2492] ✓
(`alignment_paired_ci_150m.json`, sign-flipped from `HMMER - ProtSent-V2-150M`).
V2-150M − MMseqs2 +0.0868 [+0.0620, +0.1116] ✓, +0.2950 [+0.2751, +0.3144] ✓.
All eight intervals exclude zero — the claim is exactly right.
The 35M HMMER tie −0.0124 [−0.0372, +0.0124] ✓ `alignment_paired_ci.json`,
`excludes_zero: false` — "statistically tied" is the correct reading, and the
scale-dependence warning is right. HMMER over vanilla ESM-2 150M at top-1 +0.144
✓ (0.14412). `bootstrap_ci.py` exists, 10,000 resamples (`n_boot: 10000` in every
JSON), paired, `--selfcheck` passes.

**§3 remote-homology table — all 24 cells correct.** `v2_150m/*_{knn,linear}/*.csv`
and `v3/*`: ESM-2 150M 0.51942/0.27641/0.75/0.51622; V1 0.70469/0.42974/0.74014/0.47751;
V2 0.66122/0.38854/0.75031/0.49413. 35M rows likewise. The 4.4-point drop and the
linear reversal are both real and correctly characterised.

**§3 verification paragraph.** V2 − vanilla macro-F1 −0.0262 [−0.0450, −0.0071]
excluding zero ✓, accuracy unresolved at −0.0008 ✓, 457 classes ✓, 209 classes
with ≤2 examples (457 − 248) ✓ — `verify_remote_homology_150m.json`.
`verify_remote_homology.py` exists and passes `--selfcheck`. (Rounding: finding 15a.)

**§3 aggregate 35M.** 11/3/6 +0.0075, 10/3/7 +0.0041, 4/4/12 −0.0139, 2/7/11
−0.0107 ✓ verbatim from `COMPARISON.md`. (Task-count caveat: finding 10.)

**§3 alignment across the benchmark.** EC 0.710 vs 0.598 ✓, GO-MF 0.585 vs 0.459
✓ `COMPARISON.md` (0.7103 / 0.5984; 0.5850 / 0.4590 — 0.598 and 0.459 are stock
ESM-2 35M, the best embedding arm on those rows). `mmseqs_baseline.py` and
`hmmer_baseline.py` exist, both pass `--selfcheck`, both JSONs carry 24 task
entries with `no_hit_counts_as_failure` semantics.

**§4 identity-stratified analysis.** `[0, 0.2)` bin empty (`n: 0`) ✓
`scope_identity_correlation.json`. Raw Spearman negative at both scales ✓.
Partial, 35M: MAP −0.081, p=9.0e-4 ✓ (−0.08059, 0.000904)
`scope_identity_partial_v2.json`. Partial, 150M: MAP −0.037, p=0.12 ✓ (−0.03743,
0.1237); R@10 −0.002 ✓ (−0.0023) `scope_identity_partial_150m_v2.json`. Both
scripts exist and pass `--selfcheck`. The reasoning ("memorisation predicts
positive; neither is positive") is sound and the recommended phrasing matches the
data.

**§4 seeds/few-shot.** The four quoted seed values ✓ (see finding 4 for what is
missing). Stability at N=100: ±0.20 on means 0.28–0.40 ✓ `fewshot_seeds.json`
(linear sd 0.2087/0.1770/0.1996; V2 k-NN 0.4010±0.2017). `run_seed_variability.sh`
and `fewshot_seeds.py` exist; `fewshot_seeds.py --selfcheck` passes.
**HNXd did propose the label-scarcity framing** — `REVIEWS_actual.md:13` verbatim:
"a stronger framing would be that, when labeled data is scarce, standard linear
classifiers and fine-tuning pipelines degrade substantially, while k-NN remains
competitive"; and `:24` asks for the linear baseline. Withdrawing it is correct.

**§5.2 anisotropy — all twelve cells correct** (`probe_gap_analysis.json`,
`mean_cos_random_pair` / `centered_participation_ratio` / `centered_n_dims_for_95pct`):
ESM-2 35M 0.8477 / 7.88 / 112; V2-35M 0.1521 / 52.54 / 148; ESM-2 150M 0.8960 /
10.60 / 126; V2-150M 0.1746 / 43.36 / 144. "cosine ~0.85–0.90" ✓.

**§5.2 whitening control — all values and all three significance calls correct**
(`whiten_scope_control.json`): ESM-2 150M raw 0.5529/0.7702/0.4242 ✓; whitened
0.7336/0.9155/0.6276 ✓; V2-150M raw 0.7425/0.9374/0.7048 ✓. Paired: R@1 +0.0089
`excludes_zero: false` → "unresolved" ✓; R@10 +0.0219 true ✓; MAP +0.0772 true ✓.
35M margins +0.063 / +0.050 / +0.114, all `excludes_zero: true` ✓. **The claim
that the whitening is fit on the same gallery it is applied to is true and is the
generous setting** — `whiten_scope_control.py:19-22` and `:90` (`whiten(emb)` on
the gallery), `probe_gap_analysis.py:158` (`whiten(X, X)`). §1's "all of the
top-1 gain at 150M" is exactly right.

**§5.1 the probe claim itself.** "Both probes pool the final layer" ✓ —
`protein_benchmark_suite.py:1400,1405` uses `last_hidden_state` /
`hidden_states[-1]`. "compared **at a common layer**, never at each model's own
best" ✓ — `layer_probe_sweep.py:128-129` selects
`{n//3, n//2, 2n//3, n-2, n}`, identical for all models of a given depth; the
three columns quoted are layers 10/20/30 for all three arms. All nine table cells
match `layer_probe_sweep_150m.json` exactly. (The claims *drawn* from them:
findings 1 and 2.)

**§6 item 6.** SCOPe is 2,207 sequences at family level and 100,000 was the
evaluator's `max_samples` cap ✓ `rebuttal/check_facts.md:214`. TAPE holdouts
718 + 1,254 + 1,272 = 3,244 ✓ `COMPARISON.md` "Metrics that are not comparable to
published literature". PPI decontamination description mismatch ✓
`REBUTTAL_LEAKAGE.md:400-406` (code uses `easy-search` at 40% with `--cov-mode 1
-c 0.8`, not `easy-linclust` at 50%).

**§7 paths — all exist.** `rebuttal/FINAL_rebuttal.md`, `rebuttal/NEW_EVIDENCE.md`,
`REBUTTAL_LEAKAGE.md`, `rebuttal/REVIEWS_actual.md`, `RUNS.md`,
`results/benchmarks/COMPARISON.md`, `results/benchmarks/v3/`,
`results/benchmarks/v2_150m/`, `/storage/users/ddofer/data/protsent-data-dc40`.
All fourteen named scripts exist. `run_benchmarks_150m.sh` does what the brief
implies — 4 arms × {kNN, linear}, `--eval_split test`, one code path.

**§1 reviewer scores.** "All three reviewers scored 2: Reject, confidence 4" ✓
`rebuttal/REVIEWS_actual.md:3,40-41,83-84,115-116`.

**Branch.** `origin/rebuttal` exists and is level with `master`
(`git log origin/rebuttal..master` is empty); it contains `probe_gap_analysis.py`,
`whiten_scope_control.py`, `layer_probe_sweep.py`, `verify_remote_homology.py`
and all five analysis JSONs. **But `COAUTHOR_BRIEF.md` itself is untracked** and
is on no branch — commit it before pointing anyone at the repo.

---

## What to change before sending

Must fix: findings 1–7 (layer-sweep magnitude and generality; the "benchmark
default" column; the 12/4/7 cell; the seed cherry-pick; the 17 h; the
`--selfcheck` claim; the Hub links).
Should fix: 8–14.
Nice: 15.

Two of these understate our own result (finding 3, and the metric mix in 13). The
rest overstate. Findings 1, 2 and 4 are the ones a hostile reviewer can reach
using only the files the brief itself cites.
