# ProtSent — NeurIPS 2026 rebuttal (submission 28056) — postable


---

## Response to Reviewer HNXd

<!-- character count of the pasted body below: 9924 (limit 10,000) -->
<!-- BEGIN HNXd -->
We thank the reviewer for naming which analyses would change the assessment. All five are run.

Under a trained linear probe, ProtSent loses to stock ESM-2 35M, on 11 of 20 comparable tasks (test split, single seed, tie band 0.005 absolute), so we withdraw the general-purpose embedding claim. What survives is a retrieval and clustering result, now measured as asked.

A note on naming, since this rebuttal introduces a second model. **V1** is the submitted model; **V2** is the retrain on corpora decontaminated at 40% identity and 80% coverage, using the configuration our own ablations favour. Both exist at 35M and 150M. The submitted 150M numbers, including the +105% and +19.9% in the abstract, are withdrawn and replaced by measurements on the decontaminated retrain. Every number below is on the held-out test split, which is not comparable cell-by-cell with the submitted tables.

### 1. Direct retrieval and embedding-space organisation (Q1)

We have now measured the geometry directly, on 2,207 SCOPe-40 domains against their 917 true families, using cosine distance over frozen mean-pooled embeddings.

| measure | ESM-2 35M | ProtSent-V2 35M |
| --- | --- | --- |
| silhouette (family) | -0.143 | +0.053 |
| adjusted Rand index | 0.054 | 0.507 |
| normalised mutual information | 0.823 | 0.917 |
| Spearman(distance, shared hierarchy) | -0.105 | -0.210 |

An ARI of 0.054 is close to chance: clustering the untuned space at the true number of families recovers almost nothing. The silhouette changes sign, so families go from overlapping to separated. The final row is the distance-versus-property-similarity analysis the reviewer suggested, with SCOPe hierarchy depth as the property: mean pairwise distance should fall as two domains share more of the hierarchy. For ProtSent-V2 it does so strictly across all five levels (0.865 down to 0.299); for ESM-2 35M it does not, since domains sharing two levels sit further apart (0.146) than domains sharing one (0.140).

For retrieval itself we use the same gallery, leave-one-out. Only 1,693 of the 2,207 queries have a non-self same-family neighbour, so all rows below are computed over those.

| method | R@1 | R@10 | MAP |
| --- | --- | --- | --- |
| HMMER (phmmer, `-E 10`) | 0.697 | 0.781 | 0.475 |
| MMseqs2 (`-s 7.5 -e 10`) | 0.656 | 0.740 | 0.410 |
| ESM-2 35M | 0.499 | 0.761 | 0.421 |
| ProtSent-V1 35M | 0.585 | 0.851 | 0.551 |
| ProtSent-V2 35M | 0.685 | 0.922 | 0.646 |

Alignment remains the better method at top-1, and V1 loses top-1 to both tools. Our advantage is in ranking depth, and it holds for precision as well as recall: precision@10 is 0.367 for V2 against 0.245 for phmmer, on an attainable ceiling of 0.516.

Finally, the gain is not explained by proximity to pretraining data. SCOPe-40 cannot be decontaminated at the corpus level, so we filtered the benchmark instead: restricted to the 164 eligible queries below 40% identity to our corpus, V2 minus HMMER is +0.116 [+0.049, +0.189] Recall@10 and +0.140 [+0.075, +0.207] MAP, so the margin does not shrink as queries become cleaner.

### 2. Linear and ridge probes on the frozen backbone (Q2)

We agree a conventional learned readout is needed to contextualise the 3-NN probe, which was intended to measure neighbourhood quality rather than to claim state-of-the-art prediction. Throughout, "linear probe" means logistic regression or ridge fitted on frozen embeddings, at scikit-learn defaults, on identical splits; the encoder is never updated.

| probe, 20 comparable tasks | V1 vs ESM-2 35M | V2 vs ESM-2 35M |
| --- | --- | --- |
| 3-NN | 11 win / 3 tie / 6 lose, median +0.007 | 10 / 3 / 7, median +0.004 |
| linear | 4 / 4 / 12, median -0.014 | 2 / 7 / 11, median -0.011 |

The pattern repeats at 150M, where V2 is 10 / 3 / 7 under a 3-NN probe and 4 / 4 / 12 under a linear probe, so scale does not rescue the linear record. No setting of the tie band turns the linear record into a win. Three of the 23 tasks fall outside the 20 because one-vs-rest AUC is undefined when the test split contains a class absent from training, and this excludes remote homology, our strongest task, from the tally. Reported separately, remote homology gives 3-NN accuracy 0.584 for ESM-2 35M, 0.659 for V1 and 0.667 for V2; linear accuracy 0.687, 0.690 and 0.702; linear macro-F1 0.441, 0.428 and 0.453. V1 sits below the untuned backbone on linear macro-F1, and only V2 improves on both metrics under both probes. The abstract's +105% at 150M, restated on decontaminated data as absolute scores, is 3-NN accuracy 0.519 for ESM-2 150M against 0.661 for V2, with linear macro-F1 0.516 against 0.494: a large neighbourhood gain, not a decodability one.

We did not run a fine-tuning sweep.

On the level gap identified for Stability, we tested the hypothesis and the probe is not the cause. The Biomap stability labels are continuous values from -1.680 to 2.150 scored by Spearman correlation, so our 0.588 is a correlation rather than the accuracy that the published 69.08% linear and 77.69% LoRA figures report; we withdraw that comparison as non-commensurate. Moreover, on that task the 3-NN probe scores higher than our linear probe for every model (ESM-2 35M reaches Spearman 0.643 with 3-NN against 0.440 with the linear probe), so the probe change would move the number in the wrong direction.

We also found a confound in our own protocol: both probes pool the final layer, which is the worst layer for both models on remote homology. Sweeping the pooled layer, ESM-2 35M peaks at layer 6 with accuracy 0.670 and V2 at layer 8 with 0.703; V2 at the final layer, 0.680, still exceeds ESM-2 at its best. The advantage is not an artefact of layer choice.

### 3. Confidence intervals (Q3)

We agree that sub-1% differences should not be presented as established improvements. Retrieval metrics are per-query means, so resampling the 1,693 eligible queries gives exact intervals without refitting. Using 10,000 paired resamples, the same queries scoring every method:

| paired difference | Recall@1 | Recall@10 | MAP |
| --- | --- | --- | --- |
| V2 - ESM-2 35M | +0.185 [+0.162, +0.210] | +0.161 [+0.141, +0.180] | +0.223 [+0.208, +0.238] |
| V2 - HMMER | -0.012 [-0.037, +0.012] | +0.141 [+0.120, +0.162] | +0.171 [+0.151, +0.191] |
| V2 - MMseqs2 | +0.029 [+0.004, +0.054] | +0.182 [+0.161, +0.203] | +0.236 [+0.216, +0.255] |

The same test at 150M gives V2 minus ESM-2 150M +0.190 [+0.165, +0.214], +0.167 [+0.148, +0.187] and +0.281 [+0.264, +0.297], so the retrieval result holds at the larger scale.

We do not claim to beat alignment at top-1: V2 ties phmmer there, and its +0.029 advantage over MMseqs2 clears zero by only 0.004 across three uncorrected comparisons, so we do not rely on it either. V1 minus HMMER at Recall@1 is -0.111 [-0.139, -0.083], an outright loss.

Intervals do not exist for the 23-task table, and the reviewer's objection stands there. The per-task values are measurements and we report them as such, but we draw no inferential claim from the aggregate.

### 4 and 5. Few-shot variability and absolute scores (Q4, Q5)

We agree on both points, and Table 5 is re-run rather than defended. Its relative cells were computed against near-zero Spearman baselines, and the estimator was not held constant, since the code sets the neighbour count to the smaller of 3 and the training size. We re-ran four tasks with absolute scores, a fixed estimator, five draws and the full test split. Remote homology accuracy, mean and standard deviation over 5 seeds:

| N | ESM-2, 3-NN | V1, 3-NN | V2, 3-NN | ESM-2, linear | V1, linear | V2, linear |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.061 (0.010) | 0.055 (0.008) | 0.045 (0.009) | 0.121 (0.003) | 0.159 (0.004) | 0.145 (0.005) |
| 250 | 0.148 (0.002) | 0.223 (0.011) | 0.200 (0.010) | 0.310 (0.007) | 0.394 (0.012) | 0.368 (0.013) |
| 1000 | 0.185 (0.002) | 0.318 (0.015) | 0.289 (0.016) | 0.288 (0.014) | 0.377 (0.008) | 0.355 (0.009) |

The reviewer's suspicion about the +244.5% cell was correct. It is remote homology at N=100 under a 3-NN probe, and re-run properly it becomes accuracy 0.116 for ESM-2 35M against 0.135 for V1, that is +0.019 absolute and +16.8% relative. The gain is real but task-bound: on metal-ion binding at N=1000 under the linear probe, ESM-2 35M reaches 0.666 (SD 0.001) against 0.637 (0.004) for V1 and 0.595 (0.001) for V2. The concern about spread is likewise confirmed, since Biomap stability at N=100 has a standard deviation near 0.20 on means between 0.28 and 0.40.

Full-data evaluation is close to deterministic. Five seeds across 8 tasks and 3 models under a 3-NN probe give a median standard deviation of 0.000 over 24 rows, because fixed embeddings and a fixed test split make that probe deterministic; only thermostability subsamples, at 0.013 to 0.017. Two caveats remain: one training run exists per model, so training-seed variance is unmeasured, and a near-trough checkpoint of V2 differs from the final one by 0.005 to 0.008 on every structural metric, so we treat no structural difference below 0.010 as resolved.

### What the paper now claims

Contrastive fine-tuning on multi-relational protein pairs makes structural family membership recoverable from a frozen 35M sequence-only embedding without labels: adjusted Rand index rises from 0.054 to 0.507, the silhouette crosses zero, and SCOPe-40 retrieval leads the backbone by +0.185 Recall@1 and +0.223 MAP with intervals excluding zero. This survives decontamination of both the corpus and the benchmark. It does not make the model a better general-purpose encoder, and we no longer claim that it does.

That is a narrower paper than the one we submitted, and a better-supported one. If these analyses address the criteria the reviewer identified, we would appreciate an updated assessment; if one point remains decisive, please indicate which, and we will address it during the discussion period.
<!-- END HNXd -->

---

## Response to Reviewer jVGf

<!-- character count of the pasted body below: 9524 (limit 10,000) -->
<!-- BEGIN jVGf -->
We thank the reviewer for identifying the two axes that would change the assessment. We address both, and we begin with a correction that goes against us.

Structural supervision is indeed the single largest contributor, as the reviewer suspected, and our own Table 4 says so against the paper's own text. Removing AlphaFold DB reduces improved tasks from 16 of 23 to 13 and the mean relative gain from +6.7% to +3.2%, whereas removing Pfam reduces them to 15 and +4.6%. The sentence describing Pfam as "the dominant contrastive signal" is contradicted by the table beneath it. That is our error, and no reviewer caught it.

Two withdrawals should also precede the evidence. Under a trained linear probe, ProtSent loses to stock ESM-2 35M on 12 of 20 comparable tasks for the submitted model and 11 of 20 for the retrained one, so we withdraw the general-purpose framing. The submitted 150M numbers are likewise withdrawn, replaced by measurements on a decontaminated 150M retrain.

On naming: **V1** is the submitted 35M model, and **V2** a 35M model retrained during the rebuttal on corpora decontaminated at 40% identity and 80% coverage, using the configuration our own ablations favour. All rebuttal numbers are on the held-out test split; the submitted tables use the suite's default split, and we do not mix the two.

### 1. Where ProtSent sits on the generality-accuracy trade-off (Q3, W2)

We agree that this is the important missing piece, and we have now measured it rather than asserted it. Both alignment pipelines were run over the whole benchmark under identical metric definitions, with no-hit queries scored as failures: MMseqs2 at high sensitivity (`-s 7.5 -e 10 --max-seqs 300`, since the default `-s 5.7` gives a much weaker baseline of SCOPe-40 R@1 0.385, so any MMseqs2 number needs its sensitivity stated) and HMMER phmmer at `-E 10` through the same scoring code. HMMER wins on 12 of the 22 tasks both completed, so neither is the weaker opponent, and we quote whichever tool is better for each task.

Alignment beats every 35M arm we scored on 3 of the 20 comparable tasks under a 3-NN probe and 6 under a linear probe. On enzyme-class prediction, F1-macro is 0.723 for HMMER against 0.598 for ESM-2 35M, 0.562 for V1 and 0.592 for V2. On GO molecular function it is 0.605 against 0.459, 0.443 and 0.455. On beta-lactamase fitness, Spearman is 0.803 for MMseqs2 against 0.727, 0.768 and 0.715. Where annotation transfers by homology, alignment is better, and on the two annotation-transfer tasks the gap exceeds 0.12.

The other end of the trade-off is coverage. Where a query has no alignable relative, alignment returns nothing and the score becomes a property of the fallback. On the RhlA enzyme-mutation task, whose sequences are 6-residue mutation-site strings, hit coverage is 0.004 for HMMER and 0.000 for MMseqs2, so both tools fail outright. On DeepSol solubility, MMseqs2 scores an AUC of 0.418, below chance. An embedding always returns a ranked list. We should add, however, that the three alignment wins above occur at coverage 0.945, 0.901 and 1.000, so they are genuine wins and not coverage artefacts.

On SCOPe-40 family retrieval (2,207-domain gallery, leave-one-out, self excluded, no-hit as failure, over the 1,693 queries with a non-self same-family neighbour), the full table appears in our response to Reviewer Yi1G, item 7. Against phmmer, the stronger of the two tools, ProtSent-V2 gives R@1 0.685 against 0.697, R@10 0.922 against 0.781, and MAP 0.646 against 0.475.

Using 10,000 paired bootstrap resamples with the same queries scoring every method, we do not beat alignment at top-1: V2 minus HMMER at Recall@1 is -0.012 [-0.037, +0.012], unresolved, and V1 minus HMMER is -0.111 [-0.139, -0.083], a clear loss. The embedding wins at depth against both tools: V2 minus HMMER is +0.141 [+0.120, +0.162] at Recall@10 and +0.171 [+0.151, +0.191] at MAP, and V2 minus MMseqs2 is +0.182 [+0.161, +0.203] and +0.236 [+0.216, +0.255].

Two limits on that depth result should be stated, and both are ours. First, part of the margin is candidate coverage rather than ranking: 691 of the 2,207 queries return no phmmer hit at all and score zero at every cutoff, and both tools flatten between Recall@10 and Recall@30 (both +0.017) where the embeddings do not (+0.073 for ESM-2 35M and +0.041 for V2), so part of the depth margin is an upper bound. Second, SCOPe-40 cannot be decontaminated at the corpus level, so we filtered the benchmark instead; restricted to the 164 eligible queries below 40% identity to our corpus, V2 minus HMMER holds at +0.116 [+0.049, +0.189] Recall@10 and +0.140 [+0.075, +0.207] MAP. That bounds identity-level exposure only, since supervision comes from Foldseek-cluster and Pfam-family co-membership and a training pair sharing a query's fold at 15% identity survives any identity threshold. We did not run that fold-exclusion control.

Stated plainly, the trade-off is this: alignment wins single-best-hit retrieval and homology-transferable annotation, while the embedding wins ranking depth and returns a usable ranking where alignment returns nothing at all.

### 2. Does ProtSent go beyond injecting structural information? (Q1, W1)

Partly it does not, and the AlphaFold DB ablation above quantifies how much. We did not run the joint no-AFDB/no-Pfam ablation the reviewer requested, and we accept that two single-factor ablations do not substitute for it.

What we can offer against that gap is the non-structural half of the supervision on its own. Each source leaves a distinct fingerprint on a distinct task family (submitted model, default split, single run, mean relative change over 23 tasks): without Pfam the model still improves 15 of 23 tasks at +4.6%; removing STRING moves PPI from +5.3% to -0.5% while overall quality holds at 17 of 23 and +5.9%; and removing the DMS objective reduces fluorescence from +15.6% to +10.4%. In absolute decontaminated numbers, GB1 variant-effect Spearman under a 3-NN probe on the test split, averaged over 5 seeds, is 0.658 for ESM-2 35M, 0.711 for V1 and 0.781 for V2, at a standard deviation of 0.000. Fitness ordering and physical interaction are relations that no structure teacher supplies, so a pure structure-distillation model has neither a PPI dial nor a fitness dial.

The limit on that argument is that those ablation percentages are single-run relative changes on the default split, the same convention we withdraw for small cells elsewhere; they support the direction of source-specific effects and nothing finer.

### 3. Positioning against ESM-S, S-PLM, ISM, Magneton and ProTrek (W1)

We agree the paper needs this context. ESM-S, S-PLM, ISM and Magneton inject structure into a sequence model by distilling a structure encoder or structural tokens: one relation type, one teacher. ProtSent instead supervises a heterogeneous relation graph over sequences — Pfam family co-membership, Foldseek cluster co-membership, STRING interaction and DMS fitness ordering — with no structure encoder at training or at inference. Our claim is that relation type is a design axis, evidenced by each source moving a different task family as shown above. It is not a claim of superiority: we have no matched runs against any of these models. ProTrek is the trimodal point on the same curve, and we would expect it to beat a 35M sequence-only encoder at retrieval.

We did not run ProtTucker, Foldseek, PLMSearch, DHR or ProTrek. ProtTucker is the closest analogue to our protocol, being contrastive fine-tuning of frozen embeddings for remote homology, and it is the comparison we would most want to add. We also did not apply ProtSent to SaProt or ProSST (Q2). The blocker there is data rather than code: both consume residue-level structure tokens, and we have no predicted structures for the Pfam and STRING sequences, which constitute most of the corpus.

### 4. The CoSENT objective on DMS data (Q4)

The reviewer is right that the paper's description is unclear, and in fact it is wrong: the text states that the DMS loss "operates on single proteins rather than pairs." The released code writes (wild-type, mutant, score) rows, where the score is the within-assay normalised fitness rescaled to [0,1]. CoSENT is ordinal over those pairs exactly as it is over sentence pairs: within a batch, if pair p scores above pair q, the loss pushes the wild-type-to-mutant cosine of p above that of q. There is no absolute cosine target and no term pulling high-fitness mutants together, so the objective does not flatten an assay. The real limitation is that the pairing is wild-type-anchored, so mutant-to-mutant geometry is constrained only indirectly.

On the minor note, the "?" at line 21 is a broken citation key rather than a missing reference; Heinzinger et al. 2022 and Redl et al. 2023 both appear in Related Work.

### Where this leaves the paper

Not "better than structure distillation", which we cannot show, and not a general-purpose encoder, which the linear probe refutes. What remains is a measured position on the curve the reviewer asked about: a sequence-only 35M encoder that uses no structure at inference, trades general-purpose accuracy for retrieval geometry, and demonstrates that the relation types used for supervision act as separable knobs.

If this addresses the two axes the reviewer identified, we would appreciate an updated assessment. If the missing no-AFDB/no-Pfam ablation is the decisive item, please say so and we will report it during the discussion period.
<!-- END jVGf -->

---

## Response to Reviewer Yi1G

<!-- character count of the pasted body below: 9977 (limit 10,000) -->
<!-- BEGIN Yi1G -->
We thank the reviewer for a detailed review. The leakage objection was correct and we treated it as decisive: all three pretraining corpora were re-filtered, the model retrained from scratch, the benchmarks re-run. Running HMMER, as asked, then cost us a claim: ProtSent-V2 minus phmmer at SCOPe-40 family Recall@1 is -0.012, 95% CI [-0.037, +0.012], a tie rather than a win.

Throughout, **V1** is the submitted model and **V2** the retrain on the filtered corpora, both at 35M and 150M. All numbers below are on the held-out test split. The submitted 150M numbers are replaced by measurements on the decontaminated retrain.

### 1. Potential train-test leakage

We used MMseqs2 easy-search, corpus as query and test set as target, dropping any corpus sequence with a hit.

| corpus | rows before | rows after | removed |
| --- | --- | --- | --- |
| Pfam | 28,530,684 | 27,929,772 | 2.11% |
| AlphaFold DB | 135,404,259 | 126,301,607 | 6.72% |
| STRING | 76,070,154 | 71,891,417 | 5.49% |

At 40% identity and 80% coverage, Pfam and AlphaFold DB were filtered against the remote-homology test set (3,244 sequences), STRING against the Bernett PPI test set (3,022 sequences). Those two were the only filter targets; the remaining 21 benchmark test sets were not filtered, which is the scope limit of this control. Verification semi-joined the training files against the removal lists, finding 0 flagged sequences surviving in all three, with row counts summing independently to the 169,231,379 total in the training log.

Removing every pretraining sequence within 40% identity of the remote-homology test set improved that task. For ESM-2 35M, V1 and V2, accuracy is 0.584 / 0.659 / 0.667 under a 3-NN probe and 0.687 / 0.690 / 0.702 under a linear probe, with linear macro-F1 0.441 / 0.428 / 0.453, where V1 sits below the untuned backbone. This does not establish causation: V2 also retrains with the configuration our ablations favour, so V1 against V2 is not a controlled decontamination ablation and no unfiltered retrain at that configuration exists. The supportable claim is the weaker one: decontamination did not cost performance on the filtered task.

On PPI, the stricter analysis requested was run at 40% identity rather than 50%, removing 4,178,737 STRING pairs. The downstream number does not exist: the Bernett task is pair-input and not part of the 23-task sweep, so the paper's +5.3% AUC remains a pre-decontamination V1 result. That is the open half of this weakness.

SCOPe-40 cannot be decontaminated at the corpus level: it has no train/test split, and the median maximum identity of a domain to our corpus is 0.908, so filtering against it would remove essentially every structured domain from training. We filtered the benchmark instead, dropping queries with a close pretraining neighbour and re-scoring every model on what remains. Paired V2 minus HMMER, 10,000 resamples:

| eligible queries kept | R@1 | R@10 | MAP |
| --- | --- | --- | --- |
| identity below 0.4 (n=164) | -0.043 [-0.128, +0.043] | +0.116 [+0.049, +0.189] | +0.140 [+0.075, +0.207] |
| below 0.7 (n=479) | -0.027 [-0.073, +0.017] | +0.127 [+0.090, +0.165] | +0.154 [+0.117, +0.190] |
| all (n=1,693) | -0.012 [-0.037, +0.012] | +0.141 [+0.120, +0.162] | +0.171 [+0.151, +0.191] |

The conclusion does not move: on the 164 queries furthest from anything we trained on, V2 still ties phmmer at top-1 and leads at depth. The per-query Spearman between maximum identity to the corpus and gain in average precision is -0.116, and -0.081 after controlling for baseline score, both p < 1e-03 — negative, where memorisation would predict positive. Filtering also cost the larger model nothing: the decontaminated 150M leads the submitted unfiltered one by +0.081 [+0.060, +0.102] Recall@1.

What this cannot rule out is fold-level overlap. Supervision comes from Foldseek-cluster and Pfam-family co-membership, so a training pair sharing a query's fold at 15% identity survives any identity filter. Excluding queries whose fold appears among our training clusters would separate the two explanations; we did not run that control, and can during discussion.

### 2. The DMS objective's biological assumption

The objective is implemented as the reviewer describes, and our text ("operates on single proteins rather than pairs") is wrong. Rows are wild-type, mutant, and within-assay normalised fitness in [0,1]; CoSENT ranks those pairs within a batch, with no absolute target and nothing collapsing high-fitness variants onto the wild type. The pairing is wild-type-anchored, so mutant-to-mutant distances are constrained only indirectly. Full mechanism in our reply to jVGf, item 4.

### 3. MNRL batch semantics and Eq. 1

The reviewer is right, and this is a real error. The 1,024 reported is an optimizer batch formed by gradient accumulation (64 per device over 16 steps at 35M; 16 over 64 at 150M), and accumulation does not share in-batch negatives. Each MNRL call therefore saw 64 examples at 35M and 16 at 150M — the likeliest explanation for the 150M results we no longer defend. The retrain uses a true 1,024-example batch per device. In Eq. 1 the numerator takes the positive paired with anchor i, the denominator the positives of all N pairs.

### 4-5. Pair-level tasks and k-NN regression

For PPI the two partners are embedded independently and concatenated before the probe. Peptide-HLA is not two-input here: the dataset supplies one field holding a pipe-joined HLA pseudo-sequence and peptide. The k-NN regressor uses uniform weighting over 3 neighbours with Euclidean distance. In the few-shot code the neighbour count is the smaller of 3 and the training size, so the estimator differs in the smallest cells — one reason Table 5 is re-run.

### 6. Ablations and the default design

They do not, and we have acted on it. Removing synthetic hard negatives improves 20 of 23 tasks at a mean of +7.9%, against 16 of 23 and +6.7% for the submitted configuration, and proportional sampling gives +7.0% against round-robin's +6.7%. V2 uses neither submitted default. The consequence goes against us: those ablations were scored on these same benchmarks, so V2's configuration was chosen with benchmark results in view. That is a selection channel the corpus filter does not address, and we do not present V2's 23-task numbers as a clean held-out measurement. SCOPe-40 entered that aggregate as one task among 23 rather than as the criterion, and both alignment baselines were run after the configuration was fixed.

### Mapping heterogeneous relations into one space

The cost of the shared space is the linear-probe record in item 8: compressing family, structural-cluster, interaction and fitness relations into one metric buys neighbourhood structure and loses decodable property information. The per-source ablations show each relation still moves its own task family, so the space has not collapsed to one signal. ProtSent is reliable for retrieval, clustering and nearest-neighbour transfer over structure and family, and not where one would otherwise fit a supervised model.

### 7. Baselines

HMMER (phmmer at `-E 10`, no-hit scored as failure) was run on the same gallery through the same scoring code as MMseqs2. Both take sequence input only, as we do, so together they bound what a sequence-only encoder must beat. On SCOPe-40 at family level over 1,693 eligible queries, R@1 / R@10 / MAP are 0.697 / 0.781 / 0.475 for phmmer, 0.656 / 0.740 / 0.410 for MMseqs2, 0.499 / 0.761 / 0.421 for ESM-2 35M, 0.585 / 0.851 / 0.551 for V1 and 0.685 / 0.922 / 0.646 for V2.

By paired bootstrap over 10,000 resamples, V1 minus HMMER at Recall@1 is -0.111 [-0.139, -0.083], an outright loss, while V2 minus HMMER is +0.141 [+0.120, +0.162] at Recall@10 and +0.171 [+0.151, +0.191] at MAP. Alignment remains better at top-1, and 691 queries return no phmmer hit at all, so part of our depth margin reflects coverage rather than ranking. Across the benchmark alignment beats every 35M arm on 3 of 20 comparable tasks under a 3-NN probe and 6 under a linear probe; details in our reply to jVGf.

We did not run ProtTucker, Foldseek, PLMSearch, DHR, ProTrek or Redl et al. 2023; Foldseek and ProTrek consume structure at query time. ProtTucker is the real gap, its protocol being ours.

### 8. Statistical evidence

The reviewer is right about Table 2: no intervals, one seed per cell, and a 0.005 tie band narrower than the 0.005 to 0.008 spread between V2's final and near-trough checkpoints. A 5-seed sweep gives a median standard deviation of 0.000 across 24 rows, showing only that a 3-NN probe is deterministic on fixed embeddings; training-seed variance is unmeasured, since one training run exists per model. Over the 20 tasks scorable for all models, V1 beats ESM-2 35M 11/3/6 under a 3-NN probe but 4/4/12 under a linear probe, and V2 is 10/3/7 and 2/7/11. We withdraw the general-purpose claim on that basis.

### Errors we found in our submission

The PPI decontamination description does not match the code, which uses easy-search removing hit query IDs, not easy-linclust at 50% with cluster removal. The remote-homology test split is not hierarchy-disjoint: it is TAPE's three holdouts pooled (718 fold, 1,254 superfamily, 1,272 family). SCOPe retrieval evaluates the family field over 2,207 sequences, not superfamily over 100,000 — that figure was the evaluator's sample cap echoed into the table.

Weakness 1 now reduces to two residuals we have named: untested fold-level overlap on SCOPe-40, and no post-filter PPI measurement. What survives is a smaller claim than we submitted, measured on a corpus we filtered and verified: a frozen 35M sequence-only encoder whose geometry recovers structural family membership without labels, and holds up on the queries furthest from our training data. If that addresses the concerns raised we would appreciate an updated assessment; if the fold-exclusion control decides it, please say so and we will run it in discussion.
<!-- END Yi1G -->
