Per-query vectors from runs that trained under the bf16 master-weights bug: AdamW's parameters
were bf16, where a 1e-5 update is below the representable spacing, so only 2.4% of backbone
elements could move. Their CSV rows were relabelled `*_bf16bug`; these files were not, so a
glob over `per_query_*.npz` would silently mix frozen-backbone runs into a paired analysis.

Kept as evidence of the bug. Do not use as results. See the bf16 section of RUNS.md.
