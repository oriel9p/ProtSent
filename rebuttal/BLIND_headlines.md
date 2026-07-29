# Additional experiments run during a conference rebuttal period

You are seeing only the headline and the first one or two paragraphs of each
experiment. You have not seen the paper, the reviews, or the code.

---

### 1. The entire pretraining corpus was re-filtered against the benchmark test sets, and the model retrained from scratch on the result

All three pretraining sources — 28.5M Pfam rows, 135.4M AlphaFold-DB rows, 76.1M
STRING protein-pair rows — were searched against the benchmark test sequences
with MMseqs2 and every training sequence within 40% identity at 80% coverage was
removed (2.11% / 6.72% / 5.49% respectively). The model was then retrained from
scratch on the filtered corpus, one epoch over 169.2M rows, ~11 hours on 7 GPUs.

The filtering was verified on the exact files the training job opened, by
semi-joining each against the list of removed sequences: 0 of 27,929,772 Pfam
rows, 0 of 126,301,607 AlphaFold rows, and 0 of 15,000,000 STRING pairs
contained a flagged sequence.

### 2. Performance on the task that was filtered against went up, not down

Remote homology detection is the benchmark whose test set the corpus was
decontaminated against. After removing every pretraining sequence within 40%
identity of it, accuracy rose from 0.6587 to 0.6668 under a nearest-neighbour
probe and from 0.6899 to 0.7016 under a linear probe. The untuned backbone
scores 0.5835 and 0.6868.

### 3. On structural retrieval, the decontaminated model overtook a tuned alignment search on every metric

Against a tuned MMseqs2 sequence search on the same 2,207-domain gallery, the
originally submitted model won at ranking depth but *lost* at top-1 (Recall@1
0.4490 vs 0.5029). The retrained decontaminated model wins at every cutoff:
Recall@1 0.5256, Recall@10 0.7073, MAP 0.4955, against alignment's 0.5029 /
0.5637 / 0.3100.

### 4. The benefit does not track proximity to pretraining data

For every query, the maximum sequence identity to the pretraining corpus was
computed and correlated against that query's individual retrieval gain. If the
model were memorising near-duplicates, queries with a closer pretraining
neighbour would gain more. The correlation is null to slightly negative
(Spearman -0.116 on average precision, p < 3e-6), and the largest gains occur in
the *lowest*-identity bin (+0.286 average precision at 20-40% identity, versus
+0.210 above 70%).

### 5. A sequence-alignment baseline was run across all 23 benchmark tasks, and it wins several of them

Rather than comparing only against neural baselines, MMseqs2 alignment was
scored on every task under identical metric definitions, with queries returning
no hit counted as failures. Alignment beats the best embedding model outright on
3 tasks under a nearest-neighbour probe and 6 under a linear probe — including
enzyme-class prediction (F1-macro 0.710 vs 0.598) and GO molecular function
(0.585 vs 0.459).

### 6. Both probe types are now reported separately, and they disagree

Every task was evaluated twice, once with a 3-nearest-neighbour probe and once
with a trained linear probe, on held-out test splits. Against the untuned
backbone across 20 comparable tasks, the model wins 10 / ties 3 / loses 7 under
nearest-neighbour, and wins 2 / ties 7 / loses 11 under a linear probe.

### 7. An audit of the submitted code found three description errors in the paper, now corrected

The evaluation described as covering 100,000 sequences at the superfamily level
in fact covers 2,207 sequences at the family level; the 100,000 was a sampling
cap echoed into the results table. The remote-homology test split described as
hierarchy-disjoint is a pooled concatenation of three different holdout levels.
The protein-interaction decontamination described in the paper does not match
the released code's actual procedure.
