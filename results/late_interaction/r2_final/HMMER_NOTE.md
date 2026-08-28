# ProtSent + MaxSim vs phmmer — measured, and NOT recommended for the paper

Same SCOPe-40 corpus (2,207 domains), family level, eligible queries, self excluded.
Numbers: `hmmer_vs_maxsim.csv`.

| method | R@1 | MAP | no-hit queries | wall |
|---|---|---|---|---|
| phmmer (HMMER3) | 0.7525 | 0.6067 | **691 / 2207** | **4.2 s** (32 CPU) |
| ProtSent-150M cosine | 0.7431 | 0.7045 | 0 | 9.0 s (1 GPU) |
| ProtSent-150M **MaxSim** | **0.7744** | **0.7496** | 0 | 46.8 s (1 GPU) |
| ESM-2-150M MaxSim | 0.6804 | 0.6058 | 0 | — |

## Why this is not a paper claim

It looked like one: MaxSim moves ProtSent from below phmmer on R@1 (0.7431) to above it (0.7744),
and vanilla ESM-2 under MaxSim does not get there. Three things kill it.

1. **No paired test is possible.** No per-query phmmer hits were stored -- `hmmer_maxsens.json` keeps
   aggregates only. The one paired comparison on record (`hmmer_maxsens_paired.json`, ProtSent-V2-35M
   cosine vs phmmer, hit@1 delta -0.068) has a CI half-width of about 0.024. The +0.022 margin here
   is smaller than that, so it would very likely not exclude zero.
2. **phmmer is 5-11x faster**, depending on which MaxSim configuration you compare against:

   | | dim | encode | score | total |
   |---|---|---|---|---|
   | phmmer, 32 CPU threads | - | - | - | **4.2 s** |
   | MaxSim, 128-d projected arm | 128 | 9.3 s | 11.1 s | 20.4 s |
   | MaxSim, frozen ProtSent | 640 | 9.9 s | 34.2 s | 44.1 s |

   Encoding is only about 22% of the total; MaxSim scoring dominates and scales with dimension, so
   the 128-d projected arm scores 3x faster than the 640-d unprojected one on identical work. The
   frozen-ProtSent row is the paper-relevant one and is the slowest. No free speedup is available:
   bf16 would roughly halve the scoring time but ST upcasts to fp32 only AFTER the token-token
   matmul, so bf16 quantises each per-residue maximum before summing, and this repo pins fp32 for
   that reason. Lower dimension is the real lever and it requires a trained projection.
3. The margin is one point comparison at one level on one corpus.

## What IS real, and is about coverage rather than accuracy

phmmer returns nothing at all for **691 of 2,207 queries (31%)**. That is why its MAP (0.6067) sits
0.14 below ProtSent's (0.7496) while its R@1 is comparable: when it hits it ranks well, and when it
misses it misses completely. Embedding retrieval always returns a full ranking.

That is a genuine difference and is defensible, but it is a statement about recall coverage on remote
homologs, not "we beat HMMER". Making it properly needs the per-query phmmer hits stored so the
comparison can be paired.
