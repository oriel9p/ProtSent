# ProtSent — NeurIPS 2026 rebuttal (submission 28056)
## Variant: LEAD WITH THE REFRAME

Post each response under its own review. No links, no attachments. Character
counts are stated per response and exclude the HTML comment line itself.

---

## Response to Reviewer HNXd

<!-- character count: 7,972 (body below, excluding this comment line) -->

You identified the real problem: the paper sits between two narratives. We resolved it by measurement, not by rewording, and the resolution costs us the broader claim.

Both probes are now complete on all 23 tasks, test split, for vanilla ESM-2 35M and ProtSent, and they disagree. Over the 20 tasks comparable in both arms, against ESM-2 35M: under the 3-NN probe ProtSent wins 11 / ties 3 / loses 6, median delta +0.0075; under a frozen linear probe it wins 4 / ties 4 / loses 12, median delta -0.0139. The general-purpose superiority claim does not survive the linear probe, and we withdraw it.

What survives both probes is narrower and is what the paper should have claimed: contrastive training over multiple relation types reorganizes the metric space so that family, fold and interaction relations become locally recoverable. The gain is in retrieval geometry and ranking depth, not in beating a trained head on arbitrary property prediction. Everything below is that claim being tested.

(Tie band ±0.005 on each task's main metric. 20 of 23 because three multiclass tasks — antibiotic resistance, remote homology, temperature stability — cannot produce a comparable AUC in the embedding arms when the test split contains a class absent from the probe's training split; they are excluded from the counts rather than scored either way.)

### 1. Direct retrieval evaluation (your question 1)

SCOPe-40, family-level, 2,207-sequence gallery, self-matches excluded, queries with no hit scored as failures. **Recall@K here is upper-bounded at 0.7671**: only 1,693 of the 2,207 queries have any non-self same-family protein in the gallery, so 514 are unachievable for any method.

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (the submitted model) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained, decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

V1's R@30 of 0.7100 is 92.6% of the attainable maximum. V2 is a retrain on a corpus filtered at 40% identity / 80% coverage against the benchmark test sets (details in our response to Yi1G); it differs from V1 in more than filtering — 7x1024 effective batch, no synthetic hard negatives, proportional sampling — so we do not attribute its margin to decontamination alone.

The pattern is the one the reframe predicts: the advantage grows with depth (MAP +0.10 over ESM-2 for V1, +0.17 for V2) rather than being a top-1 effect. It is also not uniform — a tuned MMseqs2 beats the submitted model at top-1 (0.5029 vs 0.4490), and only the retrain passes it.

**We did not compute clustering-geometry statistics** (silhouette, NMI, ARI). You asked for either a retrieval/clustering evaluation or a geometry analysis; we ran the first, plus the per-query analysis in section 3.

### 2. Linear probes and label scarcity (your question 2)

Frozen logistic-regression / ridge probes are complete on identical splits; the aggregate is the 4/4/12 above. On remote homology (457-class pooled TAPE holdouts, accuracy): ESM-2 35M 0.6868, ProtSent-V1 0.6899, ProtSent-V2 0.7016 under the linear probe; 0.5835 / 0.6587 / 0.6668 under 3-NN. The structural signal is present under both. Elsewhere the linear probe favours ESM-2 — for example AAV fitness Spearman 0.5639 vs 0.4362, beta-lactamase 0.6639 vs 0.5762 — while 3-NN favours ProtSent on the same tasks (AAV 0.4667 vs 0.5553). Contrastive training makes relations local; it does not add information a trained head could not already extract.

That also explains the level gap you flagged against the literature: every number in our tables is a frozen 35M backbone under a 3-NN or linear probe, not a LoRA-tuned larger model. We should not have printed those next to published fine-tuned numbers, and we no longer do.

**We did not run a fine-tuning sweep**, and **we have no linear-probe few-shot baseline**. The label-scarcity claim therefore has no supporting control and we withdraw it rather than defend it.

### 3. 95% confidence intervals (your question 3)

Retrieval answers your request exactly, because every metric is a mean over per-query values: resampling queries gives the sampling distribution with no refitting. 10,000 bootstrap resamples over the 1,693 eligible queries, **paired** (the same queries score every method):

| difference | Recall@1 | Recall@10 | MAP |
|---|---|---|---|
| V1 - ESM-2 | +0.0868 [+0.0614, +0.1122] | +0.0898 [+0.0685, +0.1105] | +0.1289 [+0.1129, +0.1447] |
| V2 - ESM-2 | +0.1855 [+0.1618, +0.2097] | +0.1607 [+0.1412, +0.1802] | +0.2232 [+0.2082, +0.2383] |
| V2 - V1 | +0.0986 [+0.0762, +0.1211] | +0.0709 [+0.0555, +0.0862] | +0.0943 [+0.0814, +0.1074] |
| V2 - MMseqs2 | +0.0289 [+0.0035, +0.0544] | +0.1819 [+0.1607, +0.2026] | +0.2356 [+0.2159, +0.2551] |

Three results from the same procedure that cut against us, reported because they come from the same table: MMseqs2 beats ESM-2 35M at top-1 by +0.1565 [+0.1276, +0.1855]; MMseqs2 beats ProtSent-V1 at top-1 by +0.0697 [+0.0413, +0.0975]; and MMseqs2 vs ESM-2 at Recall@10 (-0.0213 [-0.0484, +0.0047]) and MAP (-0.0125 [-0.0351, +0.0102]) is unresolved, so we do not claim an untuned pLM beats alignment at depth. ProtSent does, significantly.

This bootstrap quantifies which proteins are in the benchmark. It does not quantify training-seed variance.

**We have no bootstrap intervals for the 23-task Table 2 deltas** — those metrics come from probes fit per task, and refitting 10,000 times across 23 tasks x 4 arms x 2 probes did not fit the window. Your objection stands on its own without them: each Table 2 cell is one run, and any delta inside our ±0.005 tie band is unresolved. We report those as ties, not as improvements, and we no longer bold sub-1% positives.

### 4. Seed variability in the few-shot evaluation (your question 4)

**We did not run a multi-seed few-shot sweep.** We will not present single-seed few-shot numbers as evidence, so Table 5's claims are withdrawn rather than restated.

The one variance measurement we do have is checkpoint sensitivity: a near-trough checkpoint of the retrain (step 4,000, taken where the 3-cycle cosine schedule bottoms) differs from the final checkpoint by 0.005-0.008 on every structural metric above. That bounds one nuisance factor; it is not a seed replicate.

### 5. Table 5 absolute scores (your question 5)

You are pointing at an arithmetic artifact. A relative change of -126.9% on a Spearman correlation is a sign flip whose magnitude is 0.269x the baseline, not a 127-point drop; +244.5% on a near-zero baseline is likewise unbounded. A second mechanism compounds it: at very small N the probe silently reduces k, since the code sets `n_neighbors = max(1, min(3, train_size))`, so the smallest few-shot cells are not even the same estimator. Reporting relative change over near-zero denominators, from single runs, with a varying k, was the wrong instrument. We withdraw the few-shot relative framing entirely rather than re-present it with absolute values from the same single-seed run.

### Errors we found in our own submission

All new evidence above is 35M; there is no decontaminated 150M model, so we are not defending the submitted 150M numbers. The SCOPe evaluation uses the **family** field on **2,207** sequences; the text says superfamily and 100,000. The 100,000 is the evaluator's `max_samples` cap echoed into the results table. A separate superfamily evaluation still improves (R@1 0.639 -> 0.726 at 35M). Also: our remote-homology test split is TAPE's three holdouts pooled (718 fold + 1,254 superfamily + 1,272 family), not hierarchy-disjoint as stated, and its 457-class macro AUC is not comparable to published per-holdout accuracies.

If the withdrawn claims and the evidence above resolve your concerns, we ask you to raise your score. If one issue remains decisive, name it and we will answer it in discussion.

---

## Response to Reviewer jVGf

<!-- character count: 7,590 (body below, excluding this comment line) -->

Your two axes turned out to be one axis, and answering it changed what we claim. The submitted paper argued for a general-purpose embedding. Measured against a properly tuned alignment baseline and against a linear probe, that claim fails. What holds is narrower: multi-relational contrastive training buys **retrieval geometry** — ranking depth on structural and homology search — and buys nothing on tasks where a trained head already extracts the label. The position on the generality-accuracy curve is below.

### 1. Where ProtSent sits on the trade-off (your weakness 2 / question 3)

We ran MMseqs2 as a full alternative pipeline across all 23 benchmark tasks under identical metric definitions — family-level Recall@K with self excluded, per-class max bitscore for classification so AUC stays comparable, 1-NN by bitscore for regression. Queries with no hit are scored as failures, not dropped. Flags: `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3` (the default `-s 5.7` gives a much weaker baseline, SCOPe R@1 0.3847; any MMseqs2 number needs its sensitivity stated).

Alignment beats the best embedding model outright on **3 of 23 tasks under a 3-NN probe** (EC classification, GO molecular function, beta-lactamase) and **6 under a linear probe** (those plus enzyme catalytic efficiency, optimal pH, stability). The margins are not small: EC classification F1-macro 0.710 for MMseqs2 vs 0.598 for ESM-2 35M and 0.562 for ProtSent; GO-MF 0.585 vs 0.459 / 0.443. Where annotation transfers by homology, alignment is simply better and we say so.

The other end of the curve, SCOPe-40 (2,207-sequence gallery, self excluded, no-hit = failure; **Recall@K is upper-bounded at 0.7671** because only 1,693 queries have a non-self same-family neighbour):

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 (`-s 7.5 -e 10`) | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 35M (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 35M (retrained, decontaminated) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Paired bootstrap over the 1,693 eligible queries, 10,000 resamples: MMseqs2 **beats the submitted model** at top-1 by +0.0697 [+0.0413, +0.0975]; the retrained model leads MMseqs2 at every cutoff, R@1 +0.0289 [+0.0035, +0.0544], R@10 +0.1819 [+0.1607, +0.2026], MAP +0.2356 [+0.2159, +0.2551]. And MMseqs2 falls below chance on DeepSol solubility (AUC 0.4185), where there is no homology signal to transfer.

That is the trade-off, measured rather than asserted: alignment wins single-best-hit and homology-transferable annotation; the embedding wins ranking depth, and it is the only one of the two that produces a fixed-width vector usable on tasks with no alignment signal. We are not claiming it dominates.

The same trade-off appears against the untuned backbone, and it is the reason we withdraw the general-purpose framing. Both probes are now complete on all 23 tasks, test split. Over the 20 tasks comparable in both arms, against ESM-2 35M, the submitted model is 11 win / 3 tie / 6 lose under a 3-NN probe (median delta +0.0075) and **4 / 4 / 12 under a frozen linear probe** (median -0.0139); the retrain is 10/3/7 and 2/7/11. Contrastive training makes relations locally recoverable; it does not add information that a trained head could not already extract from mean-pooled ESM-2 features. On remote homology, where it does add something, the gain holds under both probes (3-NN accuracy 0.5835 -> 0.6668, linear 0.6868 -> 0.7016, ESM-2 to retrain). Tie band ±0.005; 3 of 23 tasks are excluded from both counts because their multiclass AUC is undefined in the embedding arms.

**Not run, and not implied:** ProTrek, Foldseek, PLMSearch, DHR. ProtTucker specifically: its checkpoint is served only from rostlab.org and zenodo.org, both unreachable from our cluster and not mirrored on HuggingFace, and no published SCOPe-40 or fold number we found is protocol-comparable to ours. We claim no superiority to any of these.

### 2. Is this more than structural-information injection? (your weakness 1 / question 1)

Partly not, and the ablation says how much. Removing AFDB drops the mean relative gain from +6.7% to +3.2%, improved tasks from 16/23 to 13/23, and the remote-homology gain from +40.5% to +15.3%. Structural supervision is the single largest contributor.

The remainder is not structure. Each source leaves a distinct fingerprint on a distinct task family: without Pfam the model still improves 15/23 tasks (mean +4.6%); removing STRING moves PPI from +5.3% to -0.5% while leaving most other tasks intact; removing DMS reduces fitness-related gains, including fluorescence. A structure-distillation model does not have a PPI dial or a fitness dial. The claim we can support is a sequence-level metric space shaped jointly by family, structural-cluster, interaction and fitness-order relations.

**We did not run the joint no-AFDB/no-Pfam ablation** you asked for. The single-source ablations above are what exists.

ESM-S, S-PLM, ISM and Magneton belong in the related work as the structure-injection line, and we will position ProtSent against them as a different supervision graph rather than a better one; we have no matched runs and claim no superiority.

**We did not apply ProtSent to SaProt or ProSST.** That is not a backbone swap at the data level — both require residue-level structure-derived tokens for the entire Pfam and STRING corpora (over 100M sequences), which we could not prepare in this window.

### 3. The CoSENT objective on DMS data (your question 4)

Your reading is a reasonable reading of our text, and our text is wrong. The paper says the DMS loss "operates on single proteins rather than pairs." The released code writes `dms_cosent.parquet` as `(sentence_0, sentence_1, score)` rows: `sentence_0` is the wild-type, `sentence_1` the mutant, and `score` is the within-assay normalized fitness rescaled to [0,1] (clinical rows map benign to 1.0, pathogenic to 0.0).

CoSENT is then ordinal over those pairs, exactly as you describe for sentences. Within a batch, if pair p has a higher score than pair q, the loss pushes cos(WT_p, mut_p) above cos(WT_q, mut_q). There is no absolute cosine target and no term pulling high-fitness mutants to a common point, so it does not "pull all high-fitness variants close to the wild type" in the sense that would flatten the assay.

The real limitation is narrower than the one in the review and we will state it: the pairing is **wild-type-anchored**, so mutant-mutant geometry within an assay is only constrained indirectly, through each mutant's distance to the WT.

### 4. Scale, and the missing reference

The evidence above is 35M only. The retrained decontaminated model exists at 35M; **there is no 150M model on the decontaminated corpus**, and the 150M numbers in the submission were trained on the uncontrolled corpus, so we are not defending them. Every claim we now make is a 35M claim.

The "?" on line 21 is a broken citation key, not a missing reference — Heinzinger et al. 2022 (ProtTucker) and Redl et al. 2023 are both discussed in Related Work. We fix the key.

The retrain differs from the submitted model in more than decontamination (7x1024 effective batch, no synthetic hard negatives, proportional sampling, one epoch), so we attribute its margin to the retrain as a whole, not to filtering.

If the measured trade-off and the ablation fingerprints answer your two axes, we ask you to raise your score. If one of the two remains unanswered, say which and we will address it in discussion.

---

## Response to Reviewer Yi1G

<!-- character count: 9,864 (body below, excluding this comment line) -->

Your leakage objection was correct and we treated it as decisive. We re-filtered all three pretraining corpora against the benchmark test sets, retrained from scratch on the result, and re-ran everything. The outcome changed what we claim: the general-purpose framing is withdrawn; what remains is a multi-relational metric space whose advantage is retrieval geometry and ranking depth. Items in your order.

### 1. Leakage

**Decontamination, completed.** MMseqs2 `easy-search`, pretraining corpus as query, benchmark test set as target, 40% identity / 80% coverage of the test sequence (`--min-seq-id 0.4 --cov-mode 1 -c 0.8 -e 1e-3`). Any pretraining sequence with any hit is dropped.

| corpus | filtered against | rows before | rows after | removed |
|---|---|---|---|---|
| Pfam | fold_prediction test (3,244 seqs) | 28,530,684 | 27,929,772 | 600,912 (2.11%) |
| AFDB | fold_prediction test (3,244 seqs) | 135,404,259 | 126,301,607 | 9,102,652 (6.72%) |
| STRING | bernett_gold_ppi test (3,022 seqs) | 76,070,154 | 71,891,417 | 4,178,737 (5.49%) |

Controls: negative controls (1,000 random sequences from each *filtered* corpus vs its test set, plus a stricter exhaustive-GPU re-run for AFDB) return **0 hits**; positive controls self-hit 3,244/3,244 and 3,022/3,022 at fident 1.000. The filter was then verified on the parquet files the training job actually opened, by semi-join against the recorded removal lists: **0 flagged sequences survived** in any of the three, both columns checked for STRING pairs. Row arithmetic closes independently: 27,929,772 + 126,301,607 + 15,000,000 = 169,231,379, the total in the training log (STRING was subsampled to 15M rows for compute budget, which is not a leakage control and we do not present it as one).

**Retraining on that corpus improved the filtered task.** Remote homology (457-class, accuracy): ESM-2 35M 0.5835, submitted ProtSent 0.6587, retrained ProtSent **0.6668** under 3-NN; 0.6868 / 0.6899 / **0.7016** under a linear probe. Removing every pretraining sequence within 40% identity of that test set did not cost performance on it. Caveat: the retrain also changed batch (7x1024 effective vs 1x1024), dropped synthetic hard negatives, and used proportional sampling over one epoch, so the delta is not attributable to filtering alone. The sufficient claim: the structural advantage is not an artifact of the flagged overlap. The retrain exists at 35M only; the submitted 150M results were trained on the unfiltered corpus and we do not defend them.

**SCOPe-40 cannot be decontaminated, by us or anyone.** It has no train/test split — the benchmark is leave-one-out self-retrieval over all 2,207 domains — so filtering against it would remove essentially every structured domain. Measured: median maximum identity of a SCOPe-40 sequence to a comprehensive corpus is 0.89, and **no** SCOPe sequence falls below 20%, because AFDB covers essentially all of UniProt and SCOPe domains come from PDB entries with UniProt parents. ESM-2's UniRef50 pretraining carries the same exposure, so the model-vs-model delta is the valid measurement.

So we tested it directly: if the gain came from memorizing pretraining neighbours, queries with a closer pretraining neighbour would gain more. Per-query gain over ESM-2, 1,693 eligible queries, binned by maximum identity to the pretraining corpus:

| max identity | n | submitted dRecall@10 | retrained dRecall@10 |
|---|---|---|---|
| [0.2, 0.4) | 164 | +0.0915 | +0.1524 |
| [0.4, 0.7) | 315 | +0.1016 | +0.1810 |
| [0.7, 1.0] | 1,214 | +0.0865 | +0.1565 |

Per-query Spearman between identity and gain: Recall@10 -0.038 for both models; MAP -0.114 / -0.116, p < 3e-6. The correlation is null to negative — the advantage does not grow with proximity to pretraining data. Memorization predicts the opposite sign.

**Two corrections you should have from us.** (i) Our paper describes PPI decontamination as `easy-linclust` at 50% identity with cluster-level removal. The code runs `easy-search`, STRING as query and Bernett test as target, at 40% identity, `--cov-mode 1 -c 0.8`, removing hit query IDs — stricter than the text, and the text is wrong. (ii) Our remote-homology split is not hierarchy-disjoint as claimed; it is TAPE's three holdouts pooled (718 fold + 1,254 superfamily + 1,272 family = 3,244) with no column marking which, and its 457-class macro AUC is not comparable to published per-holdout accuracies. The corpus-level filtering above is the real control, not the split.

### 2. DMS objective

The ordering objective you describe is the one implemented; our text describes it wrongly ("operates on single proteins rather than pairs"). Each row is (wild-type, mutant, within-assay normalized fitness in [0,1]). CoSENT ranks pairs within a batch: if mutant a has higher fitness than mutant b, the loss pushes cos(WT, a) above cos(WT, b). No absolute similarity target, no term collapsing high-fitness variants together. The genuine limitation: the pairing is WT-anchored, so mutant-mutant distances are constrained only indirectly.

### 3. MNRL batch semantics and Eq. 1

Your suspicion is correct, and this is a real error. For the submitted models the 1,024 is an **optimizer** batch reached by gradient accumulation (35M: per-device 64 x 16 steps; 150M: 16 x 64). Gradient accumulation does not share in-batch negatives across micro-batches, so each MNRL loss call saw **64** examples at 35M and **16** at 150M, not 1,024; the paper's "effective batch size" conflated the two. The retrained model does use a true 1,024-example contrastive batch per device via CachedMultipleNegativesRankingLoss, where `mini_batch_size` only partitions the forward/backward inside one loss call. Round-robin sampling also means a submitted-model step drew from one source, not a mixture.

Eq. 1: the numerator should use the positive paired with anchor i, the denominator ranges over the positive members of all N pairs in the loss batch, and the superscript + denotes the positive member of a pair.

### 4. Pair-level tasks

PPI: each partner is embedded independently and the two vectors are concatenated (`np.concatenate([emb[s1], emb[s2]])`) before the probe; the probe is unchanged. Peptide-HLA is not a two-input task in our implementation — the dataset supplies a single `seq` field, so no combination operator is applied. Neither was stated in the paper.

### 5. k-NN regression

`KNeighborsRegressor(n_neighbors=3, metric="minkowski")` with the default uniform weighting, i.e. unweighted mean over 3 Euclidean neighbours. One further detail that matters for the few-shot table: at small N the code sets `n_neighbors = max(1, min(3, train_size))`, so the smallest few-shot cells are not the same estimator. Combined with relative changes over near-zero Spearman baselines (which is what produces cells like -126.9%, a sign flip of magnitude 0.269x baseline), the few-shot table is not interpretable and we withdraw its claims rather than restate them.

### 6. Ablations

Agreed, and we no longer claim otherwise. Removing Pfam hard negatives improves 20/23 tasks at mean +7.9% against 16/23 and +6.7% for the submitted configuration; proportional sampling gives +7.0% vs round-robin's +6.7%. The submitted configuration is not established as optimal by our own ablations. The retrained model accordingly uses **no** synthetic hard negatives and proportional sampling.

### 7. Baselines

MMseqs2, run as a full alternative pipeline over all 23 tasks with identical metric definitions (self excluded; no-hit queries scored as failures, not dropped), `-s 7.5 -e 10 --max-seqs 300 --alignment-mode 3`. SCOPe-40, same 2,207 gallery, **Recall@K upper-bounded at 0.7671** (only 1,693 queries have a non-self same-family neighbour):

| method | R@1 | R@10 | R@30 | MAP |
|---|---|---|---|---|
| MMseqs2 | 0.5029 | 0.5637 | 0.5641 | 0.3100 |
| ESM-2 35M | 0.3829 | 0.5840 | 0.6398 | 0.3230 |
| ProtSent-V1 (submitted) | 0.4490 | 0.6529 | 0.7100 | 0.4226 |
| ProtSent-V2 (retrained) | 0.5256 | 0.7073 | 0.7390 | 0.4955 |

Alignment beats the submitted model at top-1 and beats the best embedding model outright on 3 of 23 tasks under 3-NN and 6 under a linear probe (EC F1-macro 0.710 vs 0.598/0.562; GO-MF 0.585 vs 0.459/0.443). On remote homology MMseqs2 reaches AUC 0.6523.

**Not run:** ProtTucker (checkpoint served only from rostlab.org and zenodo.org, both unreachable from our cluster, not mirrored on HuggingFace; no protocol-comparable published number to cite instead), Foldseek, PLMSearch, DHR, ProTrek. We claim no superiority to any of them. Redl et al. and Heinzinger et al. are cited; we will compare method-to-method in text, not by number.

### 8. Statistical evidence

Paired bootstrap over the 1,693 eligible SCOPe queries, 10,000 resamples: submitted model minus ESM-2 is +0.0868 [+0.0614, +0.1122] at R@1 and +0.1289 [+0.1129, +0.1447] at MAP; retrained minus ESM-2 is +0.1855 [+0.1618, +0.2097] and +0.2232 [+0.2082, +0.2383]. Against interest from the same procedure: MMseqs2 beats ESM-2 at top-1 by +0.1565 [+0.1276, +0.1855], and MMseqs2 vs ESM-2 at Recall@10 and MAP is **unresolved** (intervals span zero).

For the 23-task table we have no bootstrap intervals — those metrics require refitting a probe per resample. Your objection holds without them: each cell is a single run, and we now treat any delta inside a ±0.005 band as a tie rather than an improvement. On that basis, against ESM-2 35M over the 20 comparable tasks, the submitted model is 11 win / 3 tie / 6 lose under 3-NN (median +0.0075) and **4 / 4 / 12 under a linear probe** (median -0.0139). That is why the general-purpose claim is withdrawn.

If the completed decontamination, the retrained model, and the corrections above resolve your concerns, we ask you to raise your score. If one of the eight remains decisive, name it and we will answer it in discussion.
