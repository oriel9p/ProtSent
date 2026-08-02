# CATH v4.3 midnight-zone annotation transfer (C/A/T/H)

1-NN Euclidean over mean-pooled per-protein embeddings, lookup69k -> test219,
scored per CATH level over the queries answerable at that level.
Errors are 95% CIs (1.96 x bootstrap SE, 1,000 resamples).

## Our models

| Model | C | A | T | H |
|---|---|---|---|---|
| esm2_35m | 78.5 ± 5.4 | 54.3 ± 6.5 | 42.4 ± 6.7 | 40.7 ± 7.8 |
| protsent_v1_35m | 81.7 ± 5.1 | 64.4 ± 6.4 | 45.7 ± 6.7 | 50.7 ± 8.0 |
| protsent_v2_35m | 82.2 ± 5.0 | 64.4 ± 6.7 | 53.3 ± 6.8 | 56.7 ± 8.0 |
| esm2_150m | 74.0 ± 6.0 | 53.0 ± 6.6 | 41.0 ± 6.5 | 43.3 ± 8.2 |
| protsent_v1_150m | 83.6 ± 4.9 | 68.0 ± 6.3 | 56.2 ± 6.7 | 58.0 ± 8.0 |
| protsent_v2_150m | 84.0 ± 4.8 | 69.9 ± 6.3 | 57.1 ± 6.8 | 62.7 ± 7.8 |
| esmc_300m | 66.7 ± 6.1 | 36.1 ± 6.4 | 21.9 ± 5.5 | 18.7 ± 6.3 |
| ismc_300m | 80.8 ± 5.3 | 47.9 ± 6.6 | 29.5 ± 6.1 | 25.3 ± 7.2 |

Answerable queries per level: C 219 / A 219 / T 210 / H 150 (of 219).

## Heinzinger et al. 2022, Table 1

Same splits and same scoring, but their models and their embedding code.
A row here is NOT a like-for-like comparison against a row above; the
like-for-like comparison is each ProtSent arm against its own frozen base.

| Method | C | A | T | H |
|---|---|---|---|---|
| Random | 29 | 9 | 1 | 0 |
| MMseqs2 (sequence) | 52 | 36 | 29 | 35 |
| HMMER (CATH-Gene3D profiles) | 70 | 60 | 59 | 77 |
| ProtBERT (raw) | 67 | 38 | 22 | 18 |
| ESM-1b (raw) | 79 | 61 | 50 | 57 |
| ProtT5 (raw) | 84 | 67 | 57 | 64 |
| ProtTucker(ESM-1b) | 87 | 68 | 59 | 70 |
| ProtTucker(ProtT5) | 89 | 75 | 64 | 76 |
