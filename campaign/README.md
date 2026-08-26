# Campaign orchestration, as run

These are the scripts that actually produced `results/late_interaction/clean_*` and `r2_final`.
They ran from a scratch directory during the session and are committed here verbatim so the
campaign is reproducible; the waiting logic matches sibling scripts by basename (`ps | grep`),
so running them from this directory behaves the same.

Order, each waiting on the previous so training never contends with benchmarking:

| script | what it does |
|---|---|
| `finish_campaign.sh` | esm2-150m gate at 10k + sweep, then the 35M arm, then CATH + ProteinGym for every arm |
| `relaunch_35m.sh` | retries `late-r2-protsentv2-35m` with `--mini_batch_size` after two `--mini_batch_num_tokens` DDP deadlocks |
| `proj640_ablation.sh` | matched 640-d control for the 128-d arm: isolates dimension from training |
| `close_gaps.sh` | ESM2-150M zero-shot (the missing 2x2 cell) + two-stage rerank at every SCOPe level |
| `token_spread.py` | residue geometry: intra-sequence cosine and effective rank, the mechanism behind the pretraining effect |

Analysis that regenerates the paper numbers from saved per-query vectors, no GPU:
`python analyze_paired_effects.py --level superfamily`
