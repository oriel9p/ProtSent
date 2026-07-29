# ProtSent — NeurIPS 2026 rebuttal, submission 28056

Frame: **evidence first**. Each response opens with what was run during the rebuttal
period and what it showed. Post each block under its own review. Character counts are
stated in the HTML comment at the top of each block and cover the text from that comment
to the end of that block.

---
<!-- BEGIN HNXd -->
<!-- 7,289 characters -->

**What we ran during the rebuttal, and what it showed.** Five things, every one scored on each task's declared **test** split: (1) all three pretraining corpora re-filtered against the benchmark test sets at 40% identity / 80% coverage, and the 35M model retrained from scratch on the filtered corpus; (2) the frozen **linear probe** you asked for (logistic regression / ridge), run beside the 3-NN probe on 23 tasks x 4 model arms; (3) a tuned MMseqs2 alignment baseline under identical metric definitions; (4) 10,000-resample bootstrap intervals over individual SCOPe-40 queries; (5) a per-query analysis of gain versus identity to the pretraining corpus. V1 = the submitted 35M model, V2 = the decontaminated retrain. Every number below is 35M: the retrain exists only at 35M, and we no longer rest the paper's claim on the 150M results, which were trained on the unfiltered corpus.

The linear probe changed the headline, and it changed it against us. That result is first.

**1. Linear probe vs 3-NN (your Q2), and the claim we withdraw**

23 tasks, test split, frozen features, tie band +/-0.005 on each task's main metric, 20 tasks comparable in both arms:

| probe | ProtSent-V1 vs ESM-2 35M | ProtSent-V2 vs ESM-2 35M |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.0075 | 10 / 3 / 7, median +0.0041 |
| linear | 4 / 4 / 12, median -0.0139 | 2 / 7 / 11, median -0.0107 |

Under a trained linear readout ProtSent loses to the backbone it fine-tuned. We withdraw the general-purpose superiority claim; the paper's contribution is what survives both probes.

Two things do survive both. Remote homology (3,244-sequence test split, accuracy): ESM-2 35M 0.5835 / V1 0.6587 / V2 0.6668 under 3-NN, and 0.6868 / 0.6899 / 0.7016 under the linear probe. SCOPe-40 retrieval: table in section 2.

Your Stability example is the right diagnosis but the probe is not the explanation. Stability (Biomap), test split, Spearman: ESM-2 35M scores **0.6435** under 3-NN and **0.4395** under the linear probe (V1: 0.5638 and 0.5110). The probe moves that task by more than ProtSent does, in the direction opposite to the one your comparison assumes. We do not claim comparability with the 69.08 linear / 77.69 LoRA figures: different backbone and readout, and we ran no fine-tuning. Two further reporting notes so these cells reconcile with the submission: this sweep passes `--eval_split test` for every arm, whereas submitted Table 2 used the suite default (validation, falling back to 4-fold CV on train for the 6 tasks that declare no validation split), so the numbers here are not the submitted cells; and `ec_classification`, `go_mf` and SCOPe use a built-in evaluator that ignores the requested probe, so those three rows are identical in both tables.

**Label scarcity.** We did not run few-shot linear baselines and we did not run fine-tuning. Without them the label-scarcity claim is unsupported, so we withdraw it rather than defer it.

**2. Direct retrieval evaluation (your Q1)**

SCOPe-40, 2,207-sequence gallery, cosine 1-NN ranking, self-matches excluded, queries with no hit counted as failures, family-level relevance:

| method | Recall@1 | Recall@10 | Recall@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10 --max-seqs 300`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Recall@K here is upper-bounded at **0.7671**: only 1,693 of the 2,207 queries have any non-self same-family protein in the gallery, and the remaining 514 are singleton families that no method can hit. V1's 0.7100 at Recall@30 is 92.6% of attainable, V2's 0.7390 is 96.3%.

A tuned alignment baseline beats the submitted model at top-1 (0.5029 vs 0.4490). Only the decontaminated retrain passes it (0.5256). For the paper as submitted the defensible retrieval claim is ranking depth (Recall@10/@30, MAP), not top-1, and we state it that way.

We did not compute clustering indices (silhouette, NMI, ARI) or an embedding-anisotropy analysis. The retrieval evaluation above is the first branch of your question; we did not run the second.

**3. Bootstrap confidence intervals (your Q3)**

You asked for 95% intervals by bootstrapping over individual predictions. Retrieval admits that exactly: every metric is a mean over per-query values, so resampling queries gives the sampling distribution with no refitting. 10,000 resamples over the 1,693 eligible SCOPe-40 queries. Because every method scores the same queries, the informative quantity is the **paired** per-query difference, not overlapping marginal intervals:

| comparison | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 35M | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 35M | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - V1 | +0.0986 [+0.0762, +0.1211] | +0.0709 [+0.0555, +0.0862] | +0.0943 [+0.0814, +0.1074] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

Every one of those intervals excludes zero. Three intervals that cut the other way, reported for the same reason: MMseqs2 beats ESM-2 35M at top-1 by +0.1565 [+0.1276, +0.1855]; MMseqs2 beats V1 at top-1 by +0.0697 [+0.0413, +0.0975]; and MMseqs2 vs ESM-2 35M is **unresolved** at depth, Recall@10 -0.0213 [-0.0484, +0.0047] and MAP -0.0125 [-0.0351, +0.0102]. An untuned pLM is not better than alignment at finding homologs; the contrastive model is, at depth and significantly.

Caveat we state rather than bury: this resamples the query set, so it quantifies uncertainty over which proteins are in the benchmark. It does not quantify training-seed variance.

**Table 2 deltas.** We did not run per-task bootstrap resampling on the 23-task table, so we make no interval claim there. Instead we retract the sub-1% cells as claims: in the paired sweep above, differences inside a +/-0.005 band are counted as ties and not described as improvements, which is why the counts read 11/3/6 rather than a win total.

**4. Table 5, few-shot (your Q4 and Q5)**

We did not run a multi-seed few-shot sweep, and we do not have absolute-value replacements for Table 5 to offer here. Rather than defend it, we retract Table 5's relative framing: cells such as -126.9% (Enzyme Catalytic Efficiency, N=50) and +244.5% (Remote Homology, N=100) are ratios against a near-zero baseline Spearman/F1 and are not interpretable as effect sizes. The few-shot claim comes out of the paper's claim set.

**What we did not run, in one place:** multi-seed training or few-shot runs; per-task bootstrap on Table 2; fine-tuning or LoRA of any arm; a 150M model on the decontaminated corpus; clustering-geometry indices; matched runs of ProtTucker, Foldseek, PLMSearch, DHR or ProTrek.

We have handed you the negative result your review predicted, and the retrieval claim that survives it is narrower and better controlled than the submitted one. If that resolves your concerns we ask you to raise your score. If it does not, name the single item that is still decisive and we will answer it in discussion.
<!-- END HNXd -->

---
<!-- BEGIN jVGf -->
<!-- 7,049 characters -->

**What we ran during the rebuttal.** (1) A tuned MMseqs2 alignment baseline across the whole benchmark, scored under the same metric definitions as the embedding path, which puts a measured point on the generality-accuracy curve instead of an asserted one. (2) A frozen linear probe beside the 3-NN probe on all 23 tasks, test split. (3) Re-filtering of all three pretraining corpora at 40% identity / 80% coverage against the benchmark test sets, and a 35M retrain on the result (V2; V1 = the submitted model). Everything below is 35M; the decontaminated retrain exists only at that scale and we no longer rest the claim on the paper's 150M numbers.

**1. Where ProtSent sits on the generality-accuracy trade-off (your weakness 2)**

Measured, not argued. `mmseqs_baseline.py`, 23 tasks, test split, flags `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`; family-level Recall@K with self excluded for retrieval, per-class max bitscore for classification so AUC stays comparable, 1-NN by bitscore for regression. **Queries with no hit are scored as failures, not dropped.**

Alignment wins outright on 3 of 23 tasks under the 3-NN probe and 6 under a linear probe. EC classification F1-macro: MMseqs2 **0.710** vs ESM-2 35M 0.598 and ProtSent-V1 0.562. GO molecular function F1-macro: **0.585** vs 0.459 and 0.443. Beta-lactamase (PEER) Spearman: **0.803** vs 0.727 and 0.768. Under the linear probe alignment additionally wins enzyme catalytic efficiency, optimal pH and stability. Alignment also fails hard where homology does not transfer the label: DeepSol solubility AUC **0.4185**, below chance.

Structural retrieval is where the embedding pays for itself. SCOPe-40, same 2,207-sequence gallery for every row, self excluded, no-hit queries counted as failures:

| method | Recall@1 | Recall@10 | Recall@30 | MAP |
|---|---:|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Recall@K is upper-bounded at 0.7671 (only 1,693 of 2,207 queries have a non-self same-family neighbour). Paired per-query bootstrap, 10,000 resamples over those 1,693 queries: V2 - MMseqs2 is +0.0289 [+0.0035, +0.0544] at Recall@1, +0.1819 [+0.1607, +0.2026] at Recall@10, +0.2356 [+0.2159, +0.2551] on MAP. The submitted model **loses** top-1 to a tuned MMseqs2 (0.4490 vs 0.5029, paired difference +0.0697 [+0.0413, +0.0975] in alignment's favour) and leads only at depth. We say so rather than let the -s 5.7 default make alignment look weaker than it is; at `-s 5.7` the same baseline scores Recall@1 0.3847, and any MMseqs2 number should carry its sensitivity flag.

So the position on the curve, stated as a scope rather than a boast: one frozen sequence embedding, competitive-to-better than alignment on structural retrieval at depth and on remote homology (accuracy 0.5835 ESM-2 / 0.6587 V1 / 0.6668 V2 under 3-NN; MMseqs2 reaches 0.6523 macro AUC there with 0.8893 hit coverage), roughly a wash against the untuned backbone on general property prediction under a trained readout (linear probe, 20 comparable tasks: V1 4 win / 4 tie / 12 lose, median -0.0139), and beaten by alignment on the tasks where label transfer by homology is close to solved.

We did not run ProTrek, ProtTucker, Foldseek, PLMSearch or DHR. For ProtTucker specifically the checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster and not mirrored on HuggingFace. We claim no superiority to any of them.

**2. Does ProtSent do more than inject structure? (your weakness 1 and Q1)**

Partly, and the ablations bound how much. From the submitted 35M ablations (kNN probe, tasks improved out of 23 and mean relative change against stock ESM-2): full model 16/23, +6.7%. Without AlphaFold DB: 13/23, +3.2%, and remote homology falls from +40.5% to +15.3%. Structural supervision is the largest single contributor, and we no longer describe the gain as independent of it.

What the other sources do is separable, and each moves a different task family: without Pfam 15/23, +4.6%; without StringDB 17/23, +5.9% but PPI goes from +5.3% to -0.5%; without DMS 15/23, +5.8%, with the loss concentrated in fitness regression. The claim we defend is a sequence-level metric space shaped jointly by family, structural-cluster, interaction and fitness-order relations — not structure injection alone, and not a general-purpose improvement.

One new datum bears directly on this. AFDB took the largest decontamination cut of the three corpora — 9,102,652 sequences (6.72%) removed for falling within 40% identity / 80% coverage of the remote-homology test set — and remote homology **improved** afterwards (3-NN accuracy 0.6587 -> 0.6668; linear probe 0.6899 -> 0.7016). The structural gain is not near-duplicate transfer from AFDB. Note that V2 differs from V1 in more than the filter: 7x1024 effective batch vs 1x1024, no synthetic hard negatives, proportional sampling, one epoch.

We did not run the joint no-AFDB/no-Pfam ablation you asked for.

We also did not run ProtSent on SaProt or ProSST. That is not a backbone swap at the data level: both consume residue-level structure-derived tokens, which would have to be generated for the full Pfam and STRING corpora (28.5M and 76.1M rows) before a single training step. It did not fit the window. We will position the paper against ESM-S, S-PLM, ISM and Magneton in related work without claiming superiority to them, since we have no matched runs.

**3. CoSENT on DMS data (your Q3)**

It is ordinal over pairs, and the submission described it wrongly. Each training row is a (wild type, mutant) **pair** carrying a within-assay z-scored, clipped and rescaled fitness value. CoSENT compares pairs within a batch: if pair p has a higher fitness label than pair q, the loss pushes cos(WT_p, mutant_p) above cos(WT_q, mutant_q). There is no absolute similarity target and nothing pulls high-fitness mutants toward a common point.

The sentence that produced your reading is ours, not a misreading: "This auxiliary loss operates on single proteins rather than pairs" in the data section contradicts our own appendix and the released `data_prep.py`. It is deleted.

The real limitation is narrower and we state it: the configuration is WT-anchored, so mutant-mutant geometry is never directly optimised — only each mutant's distance to its wild type.

**Minor.** The "?" on line 21 is a broken citation key, not a missing reference: Heinzinger et al. (2022) is cited in related work and present in the bibliography. Fixed.

The measured trade-off, the source-by-source ablation fingerprints and the corrected CoSENT description are the two axes you named plus the mechanism question. If they resolve your concerns we ask you to raise your score. If one of the two axes still reads as unanswered, say which, and we will put the remaining number on it in discussion.
<!-- END jVGf -->

---
<!-- BEGIN Yi1G -->
<!-- 9,848 characters -->

**We removed the leakage and retrained.** All three pretraining corpora were re-filtered against the benchmark **test** sets with MMseqs2 `easy-search`, corpus-as-query, at 40% identity / 80% coverage of the test sequence (`--cov-mode 1 -c 0.8 -e 1e-3`), and the 35M model was retrained from scratch on the result. V1 = the submitted model, V2 = the retrain. Your eight points in order; everything is 35M, test split.

**1. Leakage (AFDB/SCOPe, STRING/PPI)**

Filter targets: the fold-prediction test set (3,244 sequences) for Pfam and AFDB, the Bernett gold PPI test set (3,022) for STRING. Rows before -> after: Pfam 28,530,684 -> 27,929,772 (-600,912, 2.11%); AFDB 135,404,259 -> 126,301,607 (-9,102,652, 6.72%); STRING 76,070,154 -> 71,891,417 (-4,178,737, 5.49%).

Controls: 1,000 random sequences from each *filtered* corpus, re-searched against its test set, return **0 hits**; positive controls self-hit 3,244/3,244 and 3,022/3,022 at fident 1.000. The filter was then verified on the exact parquet files the training job opened, semi-joined against the removal lists: **0** flagged survivors in pfam_sorted (27,929,772 rows), **0** in afdb_sorted (126,301,607) and **0** in stringdb_train_15M (15,000,000, both pair columns). Those rows sum to 169,231,379, exactly the total the trainer logged.

**Performance went up, on the task the corpus was filtered against.** Remote homology (3,244-sequence test split, accuracy): ESM-2 35M 0.5835, V1 0.6587, V2 **0.6668** under 3-NN; 0.6868 / 0.6899 / **0.7016** under a linear probe. SCOPe-40, 2,207-sequence gallery, self excluded, no-hit queries scored as failures:

| method | Recall@1 | Recall@10 | MAP |
|---|---:|---:|---:|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.3230 |
| ProtSent-V1 (submitted) | 0.4490 | 0.6529 | 0.4226 |
| ProtSent-V2 (decontaminated) | 0.5256 | 0.7073 | 0.4955 |

Recall@K is upper-bounded at 0.7671: only 1,693 of 2,207 queries have a non-self same-family neighbour.

**V2 differs from V1 in more than the filter** — 7x1024 effective batch vs 1x1024, no synthetic hard negatives, proportional sampling, one epoch — so the improvement is not attributable to decontamination. The supported claim is the one you asked about: removing the overlapping pretraining sequences does not remove the gain.

**SCOPe was deliberately not used as a filter target, and we should have said why.** It has no train/test split — the benchmark is leave-one-out self-retrieval over the whole set — so filtering at 40% against all of SCOPe deletes essentially every structured domain from the corpus. Nor is it separable in principle: the median maximum identity of a SCOPe sequence to our corpus is 0.908 and **none** falls below 20%, because AFDB covers essentially all of UniProt. ESM-2's UniRef50 pretraining carries the same exposure, so the model-vs-model delta is the measurable quantity.

We tested the memorisation prediction directly instead: if the gain came from memorising pretraining neighbours, queries with a *closer* pretraining neighbour would gain more. Per-query MAP gain over ESM-2 35M, binned by maximum identity to the pretraining corpus (same identity table, same 1,693 queries, both models): +0.1856 (V1) and +0.2859 (V2) in the [0.2, 0.4) bin (n=164), against +0.1169 and +0.2099 in the [0.7, 1.0] bin (n=1,214). Spearman between identity and gain is -0.038 for Recall@10 in both models and -0.114 / -0.116 for MAP, p < 3e-6. Null to negative — the advantage shrinks slightly with proximity to the pretraining corpus, and memorisation predicts the opposite sign.

**PPI.** The <40% analysis you asked for is done corpus-wide rather than as a subset check: 4,178,737 STRING pairs, 319,282 unique sequences removed, controls as above. The submission describes this step wrongly: appendix 12.4 says `easy-linclust` at 50% identity with cluster-level removal, while the released `data_prep.py` uses `easy-search` (STRING as query, Bernett test as target) at 40%, `--cov-mode 1 -c 0.8`, removing hit query IDs. The code is the more sensitive choice; the text is wrong and corrected.

**Two further corrections.** (i) SCOPe-40 retrieval is not "100,000 sequences at the superfamily level": the code evaluates the **family** field on **2,207** sequences, and the 100,000 is the evaluator's `max_samples` cap echoed into the results table. A separate superfamily evaluation still improves at 35M, Recall@1 0.639 -> 0.726. (ii) The remote-homology split is not superfamily-disjoint as the submission states: it is TAPE remote homology repackaged, three holdouts pooled (718 fold + 1,254 superfamily + 1,272 family = 3,244) with no column marking which. We therefore do not offer that split as a leakage control — the corpus filtering above is the control — and its pooled 457-class macro AUC is not comparable to published per-holdout top-1 accuracies.

**2. DMS objective**

The ordering objective you describe is the one implemented; the paper described it wrongly. Each row is a (WT, mutant) pair with a within-assay z-scored fitness value, and CoSENT compares pairs within the batch: a higher-fitness pair p is pushed to have cos(WT_p, mutant_p) above cos(WT_q, mutant_q) for a lower-fitness pair q. No absolute cosine target, and high-fitness variants are not pulled to a point. The sentence "This auxiliary loss operates on single proteins rather than pairs" contradicts our own appendix and the released code; it is deleted. Remaining limitation: the setup is WT-anchored, so mutant-mutant distances are never directly optimised.

**3. MNRL batch semantics and Eq. 1**

The answer is the unflattering one. Submitted 35M config (Table 6): per-device batch 64, gradient accumulation 16, "effective batch size 1024". Gradient accumulation does not share in-batch negatives across micro-batches — each micro-step computes its own MNRL loss over 64 examples — so **the contrastive batch of the submitted model is 64, i.e. 63 in-batch negatives per anchor, not 1023**. Table 6's "effective batch size" is the optimizer batch; the submission never distinguished the two, and we correct it. Round-robin sampling draws each step from one source, so those negatives are within-source. V2 instead uses CachedMultipleNegativesRankingLoss with a true contrastive batch of 1024 per rank.

Eq. 1: the numerator should use the positive paired with anchor i, the denominator should range over the positive members of all N pairs in the contrastive batch, and the superscript + denotes the positive member of a pair. Corrected.

**4. Pair-level tasks**

PPI: each partner is embedded independently and the two vectors are **concatenated** in the given order (no symmetrisation, no product or difference) before the same probe is applied. Peptide-HLA is not a two-input task here: the dataset supplies one `seq` field holding a pipe-joined HLA pseudo-sequence and peptide (~44 characters), embedded as a single string. Both are now specified.

**5. k-NN regression**

`KNeighborsRegressor(n_neighbors=3, metric="minkowski")` with the default `weights="uniform"`: uniform averaging over 3 neighbours, Euclidean distance, and `n_neighbors = max(1, min(3, train_size))` at very small training sizes.

**6. Ablations**

They do not establish the submitted configuration and we stop claiming they do. Removing synthetic hard negatives improves 20/23 tasks at mean +7.9%, against 16/23 and +6.7% for the submitted configuration; proportional sampling gives 16/23 and +7.0% against round-robin's +6.7%. We acted on that rather than argued with it: V2 is trained without synthetic hard negatives and with proportional sampling.

**7. Baselines**

MMseqs2 now runs across the whole benchmark under identical metric definitions, no-hit queries scored as failures, flags `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`. SCOPe-40 is in the table above (alignment beats every arm at top-1 except V2); remote homology macro AUC 0.6523 at 0.8893 hit coverage; across all tasks alignment beats the best embedding model outright on 3 tasks under 3-NN (EC F1-macro 0.710 vs 0.598) and 6 under a linear probe.

We did not run ProtTucker, Foldseek, PLMSearch, DHR or HMMER, and claim no superiority to them; the ProtTucker checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster. ProtTucker is the closest published analogue and Redl et al. (2023) the closest SentenceTransformers antecedent; both get an explicit comparison in related work.

**8. Statistical evidence**

Bootstrap, 10,000 resamples over the 1,693 eligible SCOPe-40 queries, paired per-query differences (every method scores the same queries): V1 - ESM-2 MAP +0.1289 [+0.1129, +0.1447]; V2 - ESM-2 MAP +0.2232 [+0.2082, +0.2383]; V2 - MMseqs2 Recall@1 +0.0289 [+0.0035, +0.0544], MAP +0.2356 [+0.2159, +0.2551]. All exclude zero. The other way: MMseqs2 - V1 at Recall@1 is +0.0697 [+0.0413, +0.0975], and MMseqs2 - ESM-2 is unresolved at depth (Recall@10 -0.0213 [-0.0484, +0.0047]).

For the 23-task table we did not run per-task bootstrap and make no interval claim there. The small Table 2 differences are retracted as claims: the re-run sweep counts anything inside +/-0.005 as a tie, giving V1 vs ESM-2 35M 11 win / 3 tie / 6 lose (median +0.0075) under 3-NN and 4 / 4 / 12 (median -0.0139) under a linear probe. Under a trained linear readout ProtSent loses to its own backbone. Structural retrieval and remote homology survive both probes; the general-purpose claim does not, so we withdraw it. We did not run multiple training seeds.

The leakage question is answered by removal and retraining rather than by argument, and four description errors are corrected above. If that resolves your concerns we ask you to raise your score; if one of the eight remains decisive, name it and we will answer it in discussion.
<!-- END Yi1G -->
