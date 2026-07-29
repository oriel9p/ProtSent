# ProtSent — NeurIPS 2026 rebuttal (variant: surgical / maximum density)

Submission 28056. Post each block under its own review. No links, no attachments.

---

## Response to Reviewer HNXd

<!-- 6,744 characters -->

Both probes are now complete on the **test** split for four arms, and they disagree. Against ESM-2 35M over the 20 tasks comparable in both arms: **3-NN — ProtSent wins 11 / ties 3 / loses 6, median delta +0.0075. Linear probe — wins 4 / ties 4 / loses 12, median delta -0.0139.** The linear probe does not support a general-purpose superiority claim and we withdraw it. What survives both probes is structural retrieval and remote homology.

Arms: MMseqs2 (`-s 7.5 -e 10`); ESM-2 35M; **V1** = the submitted `protsent-esm2-35M`; **V2** = a retrain on a corpus decontaminated against the benchmark test sets. Tie band ±0.005 on the task's main metric. 20 of 23 tasks are comparable because `antibiotic_resistance`, `remote_homology` and `temperature_stability` yield no multiclass AUC in any embedding arm (the probe's `predict_proba` lacks a column for a test-only class); they are excluded from every count, not scored zero. **All numbers below are 35M.** The decontaminated retrain exists only at 35M; the 150M results in the submitted paper were trained on the unfiltered corpus and we do not defend them here.

### 1. Direct retrieval / embedding-space organisation

Retrieval, yes. Clustering-geometry metrics (silhouette, NMI, ARI), no — we did not run them.

SCOPe-40, test split, 2,207-sequence gallery, self excluded, no-hit queries scored as failures. **Recall is capped at 0.7671**: only 1,693 of 2,207 queries have any non-self same-family gallery member, so V1's R@30 of 0.7100 is 92.6% of attainable, not 71%.

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 `-s 7.5 -e 10` | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 (decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Remote homology, test split: kNN accuracy ESM-2 0.5835 / V1 0.6587 / V2 0.6668; linear-probe accuracy 0.6868 / 0.6899 / 0.7016. This is the one task where the fine-tune beats the backbone under both probes.

Two disclosures from the same audit. (a) The submitted text says SCOPe-40 retrieval uses "100,000 proteins" at superfamily level. The code evaluates the **family** field on **2,207** sequences; 100,000 is the evaluator's `max_samples` cap echoed into the results table. (b) A separate superfamily-level evaluation still improves: R@1 0.639 → 0.726 at 35M.

### 2. Linear/ridge probe, and fine-tuning

Frozen logistic-regression and ridge probes on identical splits are complete: the 4/4/12 record above. Representative absolute values, test split:

| task | metric | probe | ESM-2 | V1 | V2 | MMseqs2 |
|---|---|---|---|---|---|---|
| Stability (BIOMAP) | Spearman | linear | 0.4395 | 0.5110 | 0.3878 | 0.5817 |
| Stability (BIOMAP) | Spearman | 3-NN | 0.6435 | 0.5638 | 0.5961 | 0.5817 |
| Variant Effect GB1 | Spearman | linear | 0.8163 | 0.8247 | 0.8126 | 0.7166 |
| Fluorescence | Spearman | linear | 0.5913 | 0.5912 | 0.5883 | 0.3863 |
| AAV (FLIP) | Spearman | linear | 0.5639 | 0.4362 | 0.2471 | 0.4024 |
| Binary Subcell. Loc. | AUC | linear | 0.9572 | 0.9404 | 0.9093 | 0.6834 |

On your Stability example specifically: our reported figure is Spearman, not the accuracy the source paper reports, so the two are not the same quantity. Under our own protocol the probe gap runs opposite to your expectation (3-NN 0.6435 beats linear 0.4395 for ESM-2), and a tuned alignment baseline beats every embedding arm under the linear probe on that task.

Fine-tuning / LoRA sweep: **we did not run it.**

Label-scarcity claim: no few-shot linear-probe baseline exists, so the claim that kNN-on-ProtSent beats a trained head under label scarcity is unsupported by anything we ran. **We withdraw it.**

### 3. 95% confidence intervals by bootstrapping over predictions

Run exactly as you specified, on the retrieval task where every metric is a mean over per-query values so resampling needs no refit: 10,000 resamples over the 1,693 eligible queries. **Quote the paired intervals** — the same queries are scored by every method, so overlapping marginal intervals do not mean a difference is unresolved.

Paired per-query differences, 95% CI:

| comparison | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 − ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 − ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 − MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

Against interest, from the same run: **MMseqs2 beats ProtSent-V1 at top-1 by +0.0697 [+0.0413, +0.0975]** and beats ESM-2 35M by +0.1565 [+0.1276, +0.1855]. The submitted model's defensible retrieval claim is ranking depth, not top-1. Only V2 passes alignment at R@1. And MMseqs2 vs ESM-2 at Recall@10 (−0.0213 [−0.0484, +0.0047]) and MAP (−0.0125 [−0.0351, +0.0102]) is **unresolved** — we do not claim an untuned pLM beats alignment at depth.

Caveat: this resamples the query set, so it quantifies benchmark-composition uncertainty, not training-seed variance.

For the Table 2 per-task deltas, a bootstrap requires refitting the probe per resample and **we did not run it**. In its place: the win/tie/lose records and medians above, with an explicit ±0.005 tie band. Every sub-1% Table 2 delta falls inside that band. **We withdraw all of them as claimed improvements.**

### 4. Multi-seed few-shot variability

**We did not run it.** One training seed, one few-shot sampling seed. We are not able to distinguish the small few-shot deltas from seed noise, and we do not ask you to.

### 5. Absolute scores in Table 5

We do not have the absolute few-shot scores in a form we can quote here, and we will not report a number we have not re-measured. The structural point is yours and we concede it: relative change has unbounded magnitude when the denominator is near zero, which is what produces −126.9% and +244.5% in the same table. **We withdraw Table 5's relative framing.** Combined with item 4, the few-shot section is not evidence for anything at present and should be read as such.

One implementation detail relevant to small N: the few-shot probe uses `n_neighbors = max(1, min(3, train_size))`, so k is 3 except when fewer than 3 training examples exist.

You said you would consider raising your score given the probe comparison, confidence intervals and variability analysis. Two of the three are above with numbers; the third we did not run and have replaced with an explicit withdrawal of the claims it would have tested. If that trade is enough, please raise the score. If not, name the single item that decides it.

---

## Response to Reviewer jVGf

<!-- 6,291 characters -->

Both questions answered with measurements, then the two things we did not run.

**(1) Structural supervision is a large share of the gain, not all of it.** Removing AFDB drops the mean relative gain over ESM-2 from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and remote homology from +40.5% to +15.3% (35M, kNN probe, submitted ablation).

**(2) On the generality-accuracy curve, a tuned alignment baseline beats the best embedding model outright on 3 of 23 tasks under a 3-NN probe and 6 of 23 under a linear probe.** Measured, not asserted.

Naming: **V1** = the submitted `protsent-esm2-35M`; **V2** = a retrain on a corpus decontaminated against the benchmark test sets at 40% identity / 80% coverage. All numbers here are 35M — the retrain exists only at that scale, and we are not defending the paper's 150M numbers, which were trained on the unfiltered corpus.

### 1. Beyond structural-information injection

Each supervision source leaves a fingerprint on a different task family (35M, kNN, vs stock ESM-2):

| ablation | tasks improved /23 | mean Δ% | signature effect |
|---|---|---|---|
| full ProtSent | 16 | +6.7 | — |
| w/o AlphaFold DB | 13 | +3.2 | remote homology +40.5% → +15.3% |
| w/o Pfam families | 15 | +4.6 | — |
| w/o STRING | 17 | +5.9 | PPI +5.3% → −0.5% |
| w/o DMS (CoSENT) | 15 | +5.8 | fluorescence +15.6% → +10.4% |

The claim we can support is a sequence-level metric space shaped jointly by family, structural-cluster, interaction and fitness-order relations. Removing structure supervision halves the gain; removing interaction supervision destroys PPI transfer and leaves the rest largely intact. That dissociation is the difference from structure-distillation work.

We will position ProtSent against ESM-S, S-PLM, ISM, Magneton, SaProt, ProSST and ProTrek in related work and claim no superiority to any of them, since we ran none of them.

**SaProt / ProSST as backbones: we did not run it.** Their inputs are residue-level structure tokens, which would have to be generated for the full Pfam and STRING corpora; that is a data-preparation job, not a backbone swap.

### 2. Where ProtSent sits on the generality-accuracy trade-off

Alignment baseline, all 23 tasks, same metric definitions as the embedding path (family-level Recall@K with self excluded; per-class max bitscore for classification so AUC stays comparable; 1-NN by bitscore for regression). **No-hit queries are counted as failures, not dropped.** Flags: `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`. The `-s 5.7` variant is much weaker (SCOPe R@1 0.3847), so any MMseqs2 number must state its sensitivity.

SCOPe-40, test split, 2,207-sequence gallery, self excluded. **Recall is capped at 0.7671** — only 1,693 of 2,207 queries have a non-self same-family gallery member.

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 `-s 7.5 -e 10` | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 (decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Read that honestly: **alignment beats the submitted model at top-1** (0.5029 vs 0.4490; paired 95% CI on the difference +0.0697 [+0.0413, +0.0975], 10,000 resamples over the 1,693 eligible queries). The submitted paper's defensible retrieval claim is ranking depth, not top-1. V2 is the first arm to pass alignment at every cutoff (V2 − MMseqs2 R@1 +0.0289 [+0.0035, +0.0544], MAP +0.2356 [+0.2159, +0.2551]).

Where alignment wins outright: `ec_classification` F1_Macro 0.710 vs 0.598 (ESM-2) / 0.562 (V1); `go_mf` 0.585 vs 0.459 / 0.443; `beta_lactamase_peer` Spearman 0.803 vs 0.727 / 0.768. Under a linear probe it additionally beats every embedding arm on `enzyme_catalytic_efficiency`, `optimal_ph` and `stability`. Where it fails: DeepSol solubility AUC 0.4185, **below chance** — the nearest homolog's solubility label is anti-correlated, and `rhla_enzyme_mutations` returns 0% hit coverage because its inputs are 6-residue strings.

The cost side of the trade-off is now measured too. Against ESM-2 35M over the 20 tasks comparable in both probes (3 multiclass tasks yield no AUC in any embedding arm and are excluded; tie band ±0.005): **3-NN, V1 wins 11 / ties 3 / loses 6, median +0.0075; linear probe, V1 wins 4 / ties 4 / loses 12, median −0.0139.** We withdraw the general-purpose superiority claim. The trade-off is therefore not "general vs specialised" but "one frozen embedding that is strong for retrieval and neighbourhood transfer, neutral-to-worse under a trained readout, against an alignment tool that is strong where homology transfers the label and useless where it does not."

**ProTrek, ProtTucker, Foldseek, PLMSearch, DHR: we did not run them.** ProtTucker specifically is blocked, not skipped — the checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster, and is not mirrored on HuggingFace.

### 3. CoSENT on DMS data

It is ordinal over pair similarities, not a regression onto absolute cosine values. Each training row is a (wild type, mutant) pair carrying a within-assay normalised fitness score. Within a batch, for two pairs p and q with score(p) > score(q), the loss penalises `exp(λ·(cos_q − cos_p))`; there is no absolute similarity target and nothing pulls all high-fitness variants to one point. So the objective you describe as more reasonable — preserving fitness-induced ordering of WT-mutant distances — is the one implemented.

The misreading traces to one wrong sentence in our Section 3.3, "This auxiliary loss operates on single proteins rather than pairs", which contradicts our own appendix. It is wrong; we remove it.

Real limitation: the configuration is WT-anchored, so mutant-mutant geometry is not directly optimised.

### 4. Minor

The broken citation on line 21 is Heinzinger et al. 2022 (ProtTucker), already in the bibliography; the `?` is a rendering failure, not a missing reference.

You said clarifying these two axes could take you to accept. Both are above as measurements, including the two places where the measurement goes against us. If that is enough, please raise the score. If not, name the one comparison that decides it.

---

## Response to Reviewer Yi1G

<!-- 9,920 characters -->

Eight items, your numbering. Four of them are concessions that the paper is wrong.

Naming: **V1** = the submitted `protsent-esm2-35M`; **V2** = a rebuttal-period retrain on the decontaminated corpus. All numbers are 35M, test split; no decontaminated 150M model exists.

### 1. Leakage

We re-filtered **all three** pretraining corpora against the benchmark test sets and retrained from scratch. MMseqs2 `easy-search`, corpus as query, test set as target, `--min-seq-id 0.4 --cov-mode 1 -c 0.8 -e 1e-3`: 40% identity, 80% coverage **of the test sequence**.

Pfam 28,530,684 → 27,929,772 (−600,912, 2.11%) and AFDB 135,404,259 → 126,301,607 (−9,102,652, 6.72%), both against the fold_prediction test set (3,244 sequences); STRING 76,070,154 → 71,891,417 pairs (−4,178,737, 5.49%) against the Bernett PPI test set (3,022).

Controls: 1,000 random sequences from each *filtered* corpus vs its test set → **0 hits**; test sets vs themselves → 3,244/3,244 and 3,022/3,022 at `fident` 1.000. Verification is on the exact parquets the training log shows being opened, semi-joined against the removal lists: **0 flagged sequences survived** in any of the three (STRING on both pair columns). Row arithmetic closes: 27,929,772 + 126,301,607 + 15,000,000 (a compute-budget STRING subsample, not a leakage control) = 169,231,379 = the trainer's logged `total=`.

**Removing every pretraining sequence within 40% identity / 80% coverage of the remote-homology test set improved remote-homology accuracy: kNN 0.6587 → 0.6668, linear probe 0.6899 → 0.7016 (ESM-2 35M: 0.5835 / 0.6868).** SCOPe-40 improved too: R@1 0.4490 → 0.5256, MAP 0.4226 → 0.4955.

**Confound, up front: V2 differs from V1 in four ways** — decontaminated corpus, 7×1024 effective batch vs 1×1024, no synthetic hard negatives, proportional sampling. The supported claim is the narrow one you asked about: removing the overlap did not cost performance, not that filtering caused the gain.

**SCOPe was deliberately not filtered against.** It has no train/test split, its median maximum identity to a comprehensive corpus is 0.89, and **no** SCOPe sequence falls below 20% — its domains come from PDB entries whose parents are in UniProt, which AFDB covers. Filtering at 40% would delete essentially every structured domain. ESM-2's UniRef50 pretraining carries the identical exposure, so the model-vs-model delta is the valid measurement.

We tested memorisation directly instead: if the gain came from memorising pretraining neighbours, queries with a *closer* neighbour would gain *more*. Per-query gain over ESM-2, 1,693 eligible SCOPe queries, binned by maximum identity to the pretraining corpus:

| max identity | n | V1 ΔR@10 | V2 ΔR@10 | V1 ΔMAP | V2 ΔMAP |
|---|---|---|---|---|---|
| [0.2, 0.4) | 164 | +0.0915 | +0.1524 | +0.1856 | +0.2859 |
| [0.4, 0.7) | 315 | +0.1016 | +0.1810 | +0.1453 | +0.2417 |
| [0.7, 1.0] | 1,214 | +0.0865 | +0.1565 | +0.1169 | +0.2099 |

Per-query Spearman between identity and gain: R@10 −0.038 (both models); MAP −0.114 (V1) / −0.116 (V2), p < 3e-6. **Null to negative.** Memorisation predicts the opposite sign.

**PPI — the paper's description of this control is wrong.** The appendix says `easy-linclust` at 50% identity with cluster-level removal. The released code uses `easy-search` (STRING as query, Bernett test as target) at 40% identity, `--cov-mode 1 -c 0.8`, removing hit query IDs, not clusters. Your requested stricter analysis is the completed 40% pass above.

**Remote homology — the paper's split claim is also wrong.** We wrote that the test split is hierarchy-disjoint. It is TAPE remote homology repackaged: the *pooled* concatenation of TAPE's three holdouts (718 fold + 1,254 superfamily + 1,272 family = 3,244), no column marking which, so two thirds is not fold-disjoint. The corpus-level decontamination above is the real control, and our pooled 457-class macro AUC is not comparable to published per-holdout top-1 accuracies.

**Also wrong: "100,000 sequences at the superfamily level."** The code evaluates the SCOPe **family** field on **2,207** sequences; 100,000 is the evaluator's `max_samples` cap echoed into the results table. A separate superfamily evaluation still improves (R@1 0.639 → 0.726).

### 2. DMS objective

The ordering objective you describe is the one implemented. Each row is a (WT, mutant) pair with within-assay normalised fitness; for two pairs p, q in a batch with score(p) > score(q), CoSENT penalises `exp(λ·(cos_q − cos_p))`. No absolute similarity target, nothing collapses high-fitness variants to a point. Section 3.3's "This auxiliary loss operates on single proteins rather than pairs" contradicts our own appendix; it is wrong and we remove it. Limitation: WT-anchored, so mutant-mutant geometry is not directly optimised.

### 3. MNRL batch semantics and Eq. 1

Your suspicion is correct and the paper is misleading. The submitted 35M run is per-device batch 64 with gradient accumulation 16; Table 6's "effective batch size 1024" is the optimiser batch. **Gradient accumulation does not share in-batch negatives across micro-steps, so the contrastive batch is 64** — and 16 for the 150M run (per-device 16, accumulation 64). We will report the two as separate rows. Round-robin sampling also means a step draws from one source, so negatives are within-source. V2 uses `CachedMultipleNegativesRankingLoss` with a true 1,024 contrastive batch per rank, `mini_batch_size=256` partitioning only the forward/backward.

Eq. 1 is malformed. The numerator must use the positive paired with anchor *i*; the denominator ranges over the positive members of all *N* pairs in the batch; the superscript **+** denotes the positive member of a pair.

### 4. Pair-level tasks

PPI: each partner is embedded independently and the two vectors concatenated (`np.concatenate([emb[s1], emb[s2]])`) before the same probe is fitted. Peptide-HLA is **not** a two-input task in our implementation — the dataset supplies one `seq` field (a pipe-joined HLA pseudosequence and peptide, ~44 characters), embedded as a single string. `ppi_bernett` is excluded from the alignment baseline and the paired probe sweep, so it has no linear-probe row in any arm.

### 5. k-NN regression

`KNeighborsRegressor(n_neighbors=3, metric="minkowski")` with no `weights` argument: **uniform averaging, Euclidean distance.** The few-shot path clamps `n_neighbors = max(1, min(3, train_size))`, so k is 3 except when fewer than 3 training examples exist.

### 6. Ablations

They do not, and we drop the claim. 35M, kNN, vs stock ESM-2: full model 16/23 improved, mean +6.7%; **w/o hard negatives 20/23, +7.9%**; proportional sampling 16/23, +7.0%. The submitted configuration is not the argmax of its own ablation table. V2 was accordingly trained with **no hard negatives and proportional sampling**.

### 7. Baselines

MMseqs2, all 23 tasks, same metric definitions as the embedding path, no-hit queries scored as failures, flags `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`. SCOPe-40, test split, 2,207 gallery, self excluded; **Recall capped at 0.7671** (only 1,693 queries have a non-self same-family match):

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 `-s 7.5` | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Alignment beats the *submitted* model at top-1; only V2 passes it. Remote homology, MMseqs2: AUC 0.6523, hit coverage 0.8893. Across the benchmark, alignment beats the best embedding arm outright on 3 tasks under kNN (`ec_classification` 0.710 vs 0.598/0.562, `go_mf` 0.585 vs 0.459/0.443) and 6 under a linear probe.

**HMMER, Foldseek, PLMSearch, DHR, ProTrek: we did not run them**, and claim no superiority to any. ProtTucker is blocked rather than skipped: its checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster, and is not mirrored on HuggingFace. ProtTucker's own protocol removes >20%-PIDE overlap from the *supervised* split only; our filter above is applied to the pretraining corpus itself. Redl et al. is cited as the closest sentence-transformer antecedent.

### 8. Statistical evidence

Retrieval metrics are means over per-query values, so a bootstrap needs no refit: 10,000 resamples over the 1,693 eligible queries, **paired** (the same queries are scored by every method).

| comparison | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 − ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 − ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 − MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

Against interest, same run: MMseqs2 − V1 at Recall@1 is +0.0697 [+0.0413, +0.0975]; MMseqs2 vs ESM-2 at Recall@10 and MAP is unresolved, both intervals spanning zero.

For the Table 2 per-task deltas a bootstrap requires refitting the probe per resample, and **we did not run it.** In its place, both probes on the test split against ESM-2 35M over the 20 comparable tasks (3 multiclass tasks yield no AUC in any embedding arm and are excluded, not zeroed; tie band ±0.005): **3-NN — V1 11 win / 3 tie / 6 lose, median +0.0075; V2 10/3/7, median +0.0041. Linear probe — V1 4/4/12, median −0.0139; V2 2/7/11, median −0.0107.** Every sub-1% Table 2 delta lies inside the tie band; **we withdraw all of them, and the general-purpose superiority claim with them.** What survives both probes is structural retrieval and remote homology.

Six of your eight items are answered with measurements; four descriptions in the paper are conceded wrong. If that resolves your concerns, please raise the score. Otherwise name the single item that remains decisive.
