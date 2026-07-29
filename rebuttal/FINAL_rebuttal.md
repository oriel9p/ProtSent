# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — postable

Paste unit = everything strictly between `<!-- BEGIN X -->` and `<!-- END X -->`.

V1 = the submitted 35M model. V2 = a 35M retrained during the rebuttal on the
decontaminated corpora. There is no 150M model on decontaminated data.

---

## Response to Reviewer HNXd

<!-- character count of the pasted body below: 9353 (limit 10,000) -->
<!-- BEGIN HNXd -->
All five analyses are run. Two answer against us, so they go first.

Under a trained linear probe, ProtSent loses to its own untuned backbone, stock ESM-2 35M, on 11 of 20 comparable tasks (test split, one seed, tie band 0.005 absolute). The general-purpose claim is withdrawn. The label-scarcity framing you proposed does not hold in our data: a trained linear head beats the 3-NN probe in nearly every model, task and N cell we measured, including N=50. We report that rather than adopt the framing.

What survives is a retrieval and clustering result, and it is now measured the way you asked. Family ARI over the SCOPe-40 embedding space rises from 0.054 (stock ESM-2 35M) to 0.507 (ProtSent-V2 35M).

Naming: V1 is the submitted 35M model. V2 is a 35M retrained on corpora decontaminated at 40% identity / 80% coverage against the benchmark test sets, using the configuration the paper's own ablations favour. All 150M results are withdrawn, including the abstract's +105% and +19.9%, because no 150M model exists on the decontaminated corpora. Every number below is `--eval_split test`; the submitted tables use the suite's default split and are not cell-comparable.

### 1. Retrieval and clustering, measured directly (Q1)

2,207 SCOPe-40 domains, their 917 true families, cosine distance, frozen mean-pooled embeddings.

| measure | ESM-2 35M | ProtSent-V2 35M |
|---|---|---|
| silhouette (family) | -0.143 | +0.053 |
| ARI vs true families | 0.054 | 0.507 |
| NMI vs true families | 0.823 | 0.917 |
| Spearman(distance, shared hierarchy) | -0.105 | -0.210 |

ARI of 0.054 is near chance: clustering the untuned space at the true number of families recovers almost nothing. Silhouette changes sign, so families go from overlapping to separated. Your distance-versus-property-similarity request is the last row, with SCOPe hierarchy depth as the property. Mean pairwise distance should fall as two domains share more of the hierarchy. For ProtSent-V2 it does, strictly, across all five levels (0.865 down to 0.299). For ESM-2 35M it does not: domains sharing two levels sit further apart (0.146) than domains sharing one (0.140).

Retrieval, same gallery, leave-one-out, self excluded, no-hit scored as a failure. Only 1,693 of 2,207 queries have a non-self same-family neighbour, so every row is those.

| method, 1,693 eligible queries | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer, `-E 10`) | 0.697 | 0.781 | 0.798 | 0.475 |
| ESM-2 35M | 0.499 | 0.761 | 0.834 | 0.421 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.926 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.963 | 0.646 |

Alignment is the better top-1 method, and the submitted V1 loses top-1 to both tools. Our advantage is ranking depth, and it holds on precision as well as recall: precision@10 is 0.367 for V2 against 0.245 for phmmer and 0.279 for ESM-2 35M, on an attainable ceiling of 0.516. Depth is not bought by returning more candidates. Part of that depth margin is candidate coverage rather than ranking quality: 691 of the 2,207 queries return no phmmer hit at all and score zero at every K. We state that as a limit on our own result.

The gain is not proximity to pretraining data. SCOPe-40 cannot be filtered at corpus level, so we filtered the benchmark instead: on the 164 eligible queries below 40% identity to our corpus, V2 minus HMMER is +0.116 [+0.049, +0.189] Recall@10 and +0.140 [+0.075, +0.207] MAP. The margin does not shrink as the queries get cleaner.

### 2. A linear classifier on the frozen backbone (Q2)

Test split, frozen embeddings, scikit-learn defaults (`StandardScaler` plus liblinear logistic regression, or `Ridge(alpha=1.0)`); 3-NN is `n_neighbors=3`, uniform.

| probe, 20 comparable tasks | V1 vs ESM-2 35M | V2 vs ESM-2 35M |
|---|---|---|
| 3-NN | 11 win / 3 tie / 6 lose, median +0.007 | 10 / 3 / 7, median +0.004 |
| linear | 4 / 4 / 12, median -0.014 | 2 / 7 / 11, median -0.011 |

No setting of the tie band turns the linear record into a win. Three tasks fall outside the 20 because one-vs-rest AUC is undefined when the test split holds a class absent from train, and that drops remote homology, our best task, from the tally. Remote homology separately (457 pooled classes, test split): 3-NN accuracy 0.584 for ESM-2 35M, 0.659 for V1, 0.667 for V2; linear accuracy 0.687, 0.690, 0.702; linear macro-F1 0.441, 0.428, 0.453. V1 sits below the untuned backbone on linear macro-F1. Only V2 improves on both metrics under both probes.

We ran no fine-tuning sweep, and we doubt it rescues the general-purpose claim when a frozen linear probe already beats us on 11 of 20.

Your hypothesis about the Stability gap is tested, and the probe is not the cause. The `biomap-research/stability_prediction` labels are continuous floats from -1.680 to 2.150 scored by Spearman, so our 0.588 is a correlation, not the accuracy that the 69.08% linear and 77.69% LoRA figures report. We withdraw that comparison as non-commensurate. And on that task the 3-NN probe scores higher than our linear probe for every arm (ESM-2 35M Spearman 0.643 with 3-NN against 0.440 linear, test split), so the probe change you proposed moves the number the wrong way.

One confound we found in our own protocol: both probes pool the final layer, which is the worst layer for both models on remote homology. Sweeping the pooled layer with a linear probe (8,000 train / 3,000 test), ESM-2 35M peaks at layer 6 with accuracy 0.670 and V2 peaks at layer 8 with 0.703. V2 at its worst useful layer, 0.680 at the final layer, still beats ESM-2 at its best. The advantage is not a layer choice.

### 3. Bootstrap confidence intervals (Q3)

Retrieval metrics are per-query means, so resampling the 1,693 eligible queries needs no refitting. 10,000 paired resamples, the same queries scoring every method.

| paired difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V2 - ESM-2 35M | +0.185 [+0.162, +0.210] | +0.161 [+0.141, +0.180] | +0.223 [+0.208, +0.238] |
| V2 - HMMER | -0.012 [-0.037, +0.012] | +0.141 [+0.120, +0.162] | +0.171 [+0.151, +0.191] |
| V2 - MMseqs2 | +0.029 [+0.004, +0.054] | +0.182 [+0.161, +0.203] | +0.236 [+0.216, +0.255] |

We do not claim to beat alignment at top-1. V2 ties phmmer there, and its +0.029 edge over MMseqs2 clears zero by 0.004 across three uncorrected comparisons, so we do not lean on that either. V1 minus HMMER at Recall@1 is -0.111 [-0.139, -0.083], an outright loss.

No intervals exist for the 23-task table; your objection stands. The per-task numbers are measurements and we report them as such, but we draw no inferential claim from the aggregate.

### 4 and 5. Few-shot absolute scores and seed variability (Q5, Q4)

Table 5 is re-run for four tasks, not defended as it stands. Its relative cells were computed against near-zero Spearman baselines, and the estimator was not constant, since the code sets `n_neighbors = max(1, min(3, train_size))`. Re-run with absolute scores, a fixed estimator, five training-subset draws and the full test split. Remote homology accuracy, mean and SD over 5 seeds:

| N | ESM-2, 3-NN | V1, 3-NN | V2, 3-NN | ESM-2, linear | V1, linear | V2, linear |
|---|---|---|---|---|---|---|
| 50 | 0.061 (0.010) | 0.055 (0.008) | 0.045 (0.009) | 0.121 (0.003) | 0.159 (0.004) | 0.145 (0.005) |
| 250 | 0.148 (0.002) | 0.223 (0.011) | 0.200 (0.010) | 0.310 (0.007) | 0.394 (0.012) | 0.368 (0.013) |
| 1000 | 0.185 (0.002) | 0.318 (0.015) | 0.289 (0.016) | 0.288 (0.014) | 0.377 (0.008) | 0.355 (0.009) |

Your suspicion about the +244.5% cell was right. It is remote homology at N=100 under 3-NN, and re-run properly it is accuracy 0.116 for ESM-2 35M against 0.135 for V1, so +0.019 absolute and +16.8% relative. The gain is real and task-bound: on metal-ion binding at N=1000 under a linear head, ESM-2 35M reaches accuracy 0.666 (SD 0.001) against V1 at 0.637 (0.004) and V2 at 0.595 (0.001). Your concern about spread is confirmed: Biomap stability at N=100 has SD near 0.20 on means of 0.28 to 0.40, so the spread is as large as the effect.

Full-data evaluation is close to deterministic. Five seeds across 8 tasks and 3 model arms under 3-NN give a median SD of 0.000 over 24 rows, because fixed embeddings and a fixed test split make that probe deterministic. Only thermostability subsamples, at SD 0.013 to 0.017. Two caveats: one training run exists per model, so training-seed variance is unmeasured, and a near-trough checkpoint of V2 differs from the final one by 0.005 to 0.008 on every structural metric, so no structural delta below 0.010 is resolved.

**What the paper claims after all this.** Contrastive fine-tuning on multi-relational protein pairs makes structural family membership recoverable from a frozen 35M sequence-only embedding without labels: adjusted Rand index 0.054 to 0.507, silhouette crossing zero, and SCOPe-40 retrieval that leads its backbone by +0.185 Recall@1 and +0.223 MAP with intervals excluding zero. It survives decontamination of the corpus and of the benchmark. It does not make the model a better general-purpose encoder, and we no longer say it does.

That is a narrower paper than we submitted and a better-supported one. If it is what you asked for, we ask you to raise your score; if one item remains decisive, name it and we answer it in discussion.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 8866 (limit 10,000) -->
<!-- BEGIN jVGf -->
Structural supervision is the single largest contributor, as you suspected, and our own Table 4 says so against the paper's own text. Removing AlphaFold DB drops improved tasks from 16 of 23 to 13 and the mean relative gain from +6.7% to +3.2%, while removing Pfam drops them to 15 and +4.6%. The sentence calling Pfam "the dominant contrastive signal" is contradicted by the table beneath it. That is our error, and no reviewer caught it.

Two withdrawals before the evidence. Under a trained linear probe ProtSent loses to stock ESM-2 35M on 12 of 20 comparable tasks (submitted model) and 11 of 20 (retrained), so the general-purpose framing is withdrawn. And no 150M model exists on the decontaminated corpora, so the submitted 150M numbers are withdrawn too.

Naming: V1 is the submitted 35M model, V2 a 35M retrained during the rebuttal on corpora decontaminated at 40% identity / 80% coverage, with the configuration our own ablations favour. Every rebuttal number is `--eval_split test`; the submitted tables use the suite's default split and we never mix the two.

### 1. Where ProtSent sits on the generality-accuracy curve (Q3, W2)

Both alignment pipelines were run over the whole benchmark under identical metric definitions, with no-hit queries scored as failures: MMseqs2 at `-s 7.5 -e 10 --max-seqs 300` (the default `-s 5.7` gives a much weaker baseline, SCOPe-40 R@1 0.385, so any MMseqs2 number needs its sensitivity stated) and HMMER phmmer at `-E 10` through the same scoring code. HMMER wins on 12 of the 22 tasks both finished, so neither is the weak opponent, and we quote whichever is better per task.

Alignment beats the best embedding arm outright on 3 of 23 tasks under a 3-NN probe and 6 under a linear probe, by large margins. EC classification F1-macro: 0.723 for HMMER against 0.598 for ESM-2 35M, 0.562 for V1 and 0.592 for V2. GO molecular function F1-macro: 0.605 for HMMER against 0.459, 0.443 and 0.455. Beta-lactamase fitness Spearman: 0.803 for MMseqs2 against 0.727, 0.768 and 0.715. Where annotation transfers by homology, alignment is better, and it is not close.

The other end of the curve is coverage, and it is the sharper half of the answer. Alignment returns nothing at all for a large share of queries, and that share grows exactly where the task is hard.

HMMER returns no hit for 47.6% of the remote-homology test set and 31.3% of the SCOPe-40 gallery. On `rhla_enzyme_mutations`, which is 6-residue mutation-site strings, hit coverage is 0.004 for HMMER and 0.000 for MMseqs2: both tools fail completely. On DeepSol solubility MMseqs2 scores AUC 0.418, below chance.

An embedding always returns a ranked list, so its metric is never a property of a fallback. That said, the three alignment wins above stand at coverage 0.945, 0.901 and 1.000, so they are real wins and not coverage artifacts.

SCOPe-40, family level, 2,207-domain gallery, leave-one-out, self excluded, no-hit as failure. Only 1,693 queries have a non-self same-family neighbour, and every row uses those.

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer) | 0.697 | 0.781 | 0.798 | 0.475 |
| ESM-2 35M | 0.499 | 0.761 | 0.834 | 0.421 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.926 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.963 | 0.646 |

Paired bootstrap, 10,000 resamples, the same queries scoring every method. We do not beat alignment at top-1: V2 minus HMMER at Recall@1 is -0.012 [-0.037, +0.012], unresolved, and V1 minus HMMER is -0.111 [-0.139, -0.083], a clear loss. The embedding wins at depth against both: V2 minus HMMER is +0.141 [+0.120, +0.162] at Recall@10 and +0.171 [+0.151, +0.191] at MAP; V2 minus MMseqs2 is +0.182 [+0.161, +0.203] and +0.236 [+0.216, +0.255].

Two limits on that depth result, both ours.

First, part of the margin is candidate coverage rather than ranking. 691 of 2,207 queries return no phmmer hit at all and score zero at every K, and both tools flatten from Recall@10 to Recall@30 (both +0.017) where the embeddings do not (+0.073 for ESM-2 35M, +0.041 for V2). We did not re-run either tool with the threshold lifted, so the depth margin is an upper bound.

Second, SCOPe-40 cannot be filtered at corpus level, so we filtered the benchmark. On the 164 eligible queries below 40% identity to our corpus, V2 minus HMMER holds at +0.116 [+0.049, +0.189] Recall@10 and +0.140 [+0.075, +0.207] MAP. That bounds identity-level exposure only: supervision is Foldseek-cluster and Pfam-family co-membership, so a training pair sharing a query's fold at 15% identity survives any identity threshold. We did not run that fold-exclusion control.

The trade-off, plainly: alignment wins single-best-hit retrieval and homology-transferable annotation; the embedding wins ranking depth and returns a usable ranking where alignment returns nothing.

### 2. Is this more than injecting structure? (Q1, W1)

Partly not, and the AFDB ablation above quantifies how much. We did not run the joint no-AFDB/no-Pfam ablation you asked for, and two single-factor ablations do not substitute for it.

What we can put against it is the non-structural half on its own. Each source fingerprints a different task family (submitted model, default split, single run, mean relative change over 23 tasks): without Pfam the model still improves 15 of 23 at +4.6%; removing StringDB takes PPI from +5.3% to -0.5% while overall quality holds at 17 of 23 and +5.9%; removing the DMS CoSENT objective takes fluorescence from +15.6% to +10.4%.

In absolute decontaminated numbers, GB1 variant effect Spearman under a 3-NN probe on the test split, mean of 5 seeds, is 0.658 for ESM-2 35M, 0.711 for V1 and 0.781 for V2, at SD 0.000. Fitness order and interaction are relations no structure teacher supplies, so a structure-distillation model has no PPI dial and no fitness dial.

The limit on that argument: those ablation percentages are single-run relative changes on the default split, the convention we withdraw for small cells elsewhere. They support the direction of source-specific effects and nothing finer, and they cannot show that the sources do not interfere.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton, ProTrek (W1)

ESM-S, S-PLM, ISM and Magneton inject structure into a sequence model by distilling a structure encoder or structural tokens: one relation type, one teacher. ProtSent supervises a heterogeneous relation graph over sequences — Pfam family co-membership, Foldseek cluster co-membership, STRING interaction, DMS fitness order — with no structure encoder at training or at inference. The claim is that relation type is a design axis, evidenced by each source moving a different task family above. It is not a claim of superiority: we have no matched runs against any of them. ProTrek is the trimodal point on the same curve, and we expect it to beat a 35M sequence-only encoder at retrieval.

Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek. ProtTucker is the closest analogue to our protocol, contrastive fine-tuning of frozen embeddings for remote homology, and it is the gap we would most want to close.

We also did not apply ProtSent to SaProt or ProSST (Q2). The blocker is data, not code: both consume residue-level structure tokens, and we have no predicted structures for the Pfam and STRING sequences, which are most of the corpus.

### 4. The CoSENT objective on DMS data (Q4)

Our text is wrong: the paper says the DMS loss "operates on single proteins rather than pairs." The released code writes `(sentence_0, sentence_1, score)` rows, which are wild-type, mutant, and the within-assay normalised fitness rescaled to [0,1]. CoSENT is ordinal over those pairs exactly as over sentence pairs: within a batch, if pair p scores above pair q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no absolute cosine target and no term pulling high-fitness mutants together, so it does not flatten an assay. The real limitation is that pairing is wild-type-anchored, so mutant-to-mutant geometry is constrained only indirectly.

The "?" at line 21 is a broken citation key, not a missing reference. Heinzinger et al. 2022 and Redl et al. 2023 are both in Related Work.

**Where that leaves the paper.** Not "better than structure distillation", which we cannot show, and not a general-purpose encoder, which the linear probe refutes. What is left is a measured point on the curve you asked about: a sequence-only 35M encoder, no structure at inference, that trades general-purpose accuracy for retrieval geometry, and a demonstration that the relation *types* you supervise on are separable knobs. That is the insight we think is worth the page.

We ask you to raise your score on it. If the missing no-AFDB/no-Pfam ablation is the decisive item, say so and we report it in discussion.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9736 (limit 10,000) -->
<!-- BEGIN Yi1G -->
Your leakage objection was correct and we treated it as decisive: all three pretraining corpora re-filtered at 40% identity / 80% coverage, the model retrained from scratch, benchmarks re-run, and verification on the files training actually opened finding 0 flagged sequences surviving. Running HMMER, as you asked, then cost us a claim. ProtSent-V2 minus phmmer at SCOPe-40 family Recall@1 is -0.012, 95% CI [-0.037, +0.012] — a tie, not a win.

V1 is the submitted 35M model, V2 the retrain on the filtered corpora. Every number below is on `--eval_split test`. All 150M results are withdrawn, because no 150M model exists on the decontaminated corpora.

### 1. Leakage

MMseqs2 `easy-search`, corpus as query, test set as target, 40% identity / 80% coverage, dropping any corpus sequence with a hit.

| corpus | rows before | rows after | removed |
|---|---|---|---|
| Pfam | 28,530,684 | 27,929,772 | 2.11% |
| AFDB | 135,404,259 | 126,301,607 | 6.72% |
| STRING | 76,070,154 | 71,891,417 | 5.49% |

Pfam and AFDB were filtered against the `remote_homology` test set (3,244 sequences), STRING against `ppi_bernett` test (3,022 sequences). Those two were the only filter targets; the other 21 benchmark test sets were not filtered, and that is the scope limit. Verification semi-joined the training parquets against the removal lists: 0 flagged sequences survived, in all three files. The row counts sum independently to the 169,231,379 total in the training log.

Removing every pretraining sequence within 40% identity of the remote-homology test set improved that task. Remote homology accuracy, test split: 3-NN 0.584 for ESM-2 35M, 0.659 for V1, 0.667 for V2; linear 0.687, 0.690, 0.702; linear macro-F1 0.441, 0.428, 0.453, where V1 sits below the untuned backbone.

What that does not establish: V2 retrains on the filtered corpora with the configuration our own ablations favour, so V1 against V2 is not a controlled decontamination ablation, and no unfiltered retrain at that configuration exists. The supportable claim is the weaker and sufficient one. Decontaminating the corpus did not cost performance on the filtered task.

PPI: the filter you asked for was run, at 40% identity rather than the 50% you named, removing 4,178,737 STRING pairs. The downstream number does not exist, because `ppi_bernett` is pair-input and not in the 23-task sweep, so the paper's +5.3% AUC remains a pre-decontamination V1 result. That is the open half of this weakness.

SCOPe-40 cannot be filtered at corpus level. It has no train/test split, and the median maximum identity of a SCOPe-40 domain to our corpus is 0.908, so filtering against it would remove essentially every structured domain. We filtered the benchmark instead: drop the queries with a close pretraining neighbour, re-score every arm on what remains. Paired V2 minus HMMER, 10,000 resamples:

| eligible queries kept | R@1 | R@10 | MAP |
|---|---|---|---|
| identity below 0.4 (n=164) | -0.043 [-0.128, +0.043] | +0.116 [+0.049, +0.189] | +0.140 [+0.075, +0.207] |
| below 0.7 (n=479) | -0.027 [-0.073, +0.017] | +0.127 [+0.090, +0.165] | +0.154 [+0.117, +0.190] |
| all (n=1,693) | -0.012 [-0.037, +0.012] | +0.141 [+0.120, +0.162] | +0.171 [+0.151, +0.191] |

The conclusion does not move. On the 164 queries furthest from anything we trained on, V2 still ties phmmer at top-1 and still leads at depth, and the margin does not shrink as the queries get cleaner. Per-query Spearman between maximum identity to the corpus and gain in average precision is -0.116 (p=1.6e-06), and -0.081 (p=9.0e-04) after controlling for baseline score. It is negative where memorisation predicts positive.

What this cannot rule out: supervision is Foldseek-cluster and Pfam-family co-membership, so a training pair sharing a query's fold at 15% identity survives any identity filter. Excluding queries whose fold appears in our training clusters would separate the two. We did not run it, and can in discussion.

### 2. The DMS objective

Implemented as you describe, and our text ("operates on single proteins rather than pairs") is wrong. Rows are wild-type, mutant, and within-assay normalised fitness in [0,1]. CoSENT ranks pairs within a batch: if mutant a outscores b, the loss pushes cos(WT, a) above cos(WT, b). No absolute target, nothing collapsing high-fitness variants onto the wild type. The pairing is wild-type-anchored, so mutant-to-mutant distances are constrained only indirectly.

### 3. MNRL batch semantics and Eq. 1

Correct, and a real error. The submitted 1,024 is an optimizer batch from gradient accumulation (35M: 64 per device times 16 steps; 150M: 16 times 64), and accumulation does not share in-batch negatives. Each MNRL call therefore saw 64 examples at 35M and 16 at 150M, the likeliest explanation for the 150M results we no longer defend. The retrain uses a true 1,024 batch per device. In Eq. 1 the numerator takes the positive paired with anchor i and the denominator the positives of all N pairs.

### 4 and 5. Pair-level tasks and k-NN regression

PPI partners are embedded independently and concatenated before the probe. Peptide-HLA is not two-input here: the dataset supplies one `seq` field holding a pipe-joined `HLA_pseudoseq|peptide` string. k-NN regression is uniform, `KNeighborsRegressor(n_neighbors=3)`, an unweighted mean over 3 neighbours. In the few-shot code the estimator is `n_neighbors = max(1, min(3, train_size))`, a different estimator in the smallest cells, which is one reason Table 5 is re-run.

### 6. The ablations do not support the defaults

They do not, and we acted on it. Removing synthetic hard negatives improves 20 of 23 tasks at mean +7.9% against 16 of 23 and +6.7% for the submitted configuration, and proportional sampling gives +7.0% against round-robin's +6.7% (relative gain over ESM-2 35M, submitted model, default split, one run each). V2 uses neither submitted default.

The consequence goes against us and we state it: those ablations were scored on these same benchmarks, so V2's configuration was chosen with benchmark results in view. That is a selection channel the corpus filter does not touch, and we do not call V2's 23-task numbers a clean held-out measurement. SCOPe-40 entered that aggregate as one task of 23, not as the criterion, and both alignment baselines were run after the configuration was fixed.

### 7. Baselines

HMMER (phmmer, `-E 10`, top 300 hits per query, no-hit scored as failure) was run on the same gallery through the same scoring code as MMseqs2. Both take sequence only, as we do, so they bound what a sequence-only encoder has to beat. SCOPe-40, family level, leave-one-out over 2,207 domains; the 1,693 eligible queries are those with a non-self same-family neighbour.

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.757 | 0.410 |
| HMMER (phmmer) | 0.697 | 0.781 | 0.798 | 0.475 |
| ESM-2 35M | 0.499 | 0.761 | 0.834 | 0.421 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.926 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.963 | 0.646 |

Paired bootstrap, 10,000 resamples. At Recall@1, V1 minus HMMER is -0.111 [-0.139, -0.083], an outright loss, and V2 minus MMseqs2 is +0.029 [+0.004, +0.054], clearing zero by 0.004 across three uncorrected comparisons, so we do not lean on it. At depth, V2 minus HMMER is +0.141 [+0.120, +0.162] at Recall@10 and +0.171 [+0.151, +0.191] at MAP.

Alignment remains the better top-1 method, and 691 of 2,207 queries return no phmmer hit at all, so part of our depth margin is candidate coverage rather than ranking. Across the benchmark alignment beats the best embedding arm on 3 of 23 tasks under 3-NN and 6 under a linear probe, including EC classification F1-macro 0.723 for HMMER against 0.598 for ESM-2 35M.

Not run: ProtTucker, Foldseek, PLMSearch, DHR, ProTrek, Redl et al. 2023. Foldseek and ProTrek consume structure at query time, so losing to them says nothing about a sequence-only encoder. ProtTucker is the real gap, since its protocol is ours.

### 8. Statistical evidence

On Table 2 you are right: no intervals, one seed per cell, and a 0.005 tie band narrower than the 0.005 to 0.008 spread between V2's final and near-trough checkpoints. A 5-seed sweep gives median SD 0.000 across 24 rows, which shows only that a 3-NN probe is deterministic on fixed embeddings; training-seed variance is unmeasured, since one training run exists per model. Over the 20 tasks scorable for all arms, V1 beats ESM-2 35M 11/3/6 under 3-NN but 4/4/12 under a linear probe, and V2 is 10/3/7 and 2/7/11. The general-purpose claim is withdrawn on that basis.

### Errors in our own submission

The PPI decontamination text does not match the code, which uses `easy-search` at 40% identity removing hit query IDs, not `easy-linclust` at 50% with cluster-level removal. The remote-homology test split is not hierarchy-disjoint: it is TAPE's three holdouts pooled (718 fold, 1,254 superfamily, 1,272 family), so its pooled macro AUC is not comparable to published per-holdout accuracies. SCOPe retrieval evaluates the family field over 2,207 sequences, not superfamily over 100,000; the 100,000 is the evaluator's sample cap echoed into the table.

Weakness 1 now reduces to the residual we name: untested fold-level overlap on SCOPe-40, and no post-filter PPI measurement.

What survives your eight points is a smaller claim than we submitted, measured on a corpus we filtered and verified: a frozen 35M sequence-only encoder whose geometry recovers structural family membership without labels, holding on the queries furthest from our training data. We ask you to raise your score on that. If the fold-exclusion control is what decides it, say so and we run it in discussion.
<!-- END Yi1G -->
