#!/usr/bin/env python3
"""Generate the HF dataset card for the decontaminated ProtSent pretraining data.

Reads decontam_report.json (written by decontaminate_pretrain.py) so the card can
never drift from the numbers actually produced.

    python write_dataset_card.py [--out-dir /storage/users/ddofer/data/protsent-data-dc40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Internal redundancy reduction is a separate, pre-existing step per corpus; it is
# NOT the test-set filter. Recorded here so the card states both honestly.
INTERNAL_REDUNDANCY = {
    "pfam": "MMseqs2 `easy-linclust` at 70% identity / 80% coverage, representatives only "
    "(`data_prep.py: prep_pfam_full`)",
    "afdb": "None of our own. Inherits AFDB50 cluster ids (<50% identity between cluster "
    "representatives) plus Foldseek structural clusters. Within-cluster members are kept "
    "deliberately — they are the positive pairs for contrastive training",
    "stringdb": "MMseqs2 cascaded clustering, `linclust` 65%/85% then `cluster` 50%/75%; "
    "same-cluster pairs dropped and one pair kept per cluster-pair (`data_prep.py: prep_stringdb`)",
}

SOURCE = {
    "pfam": "Pfam-A full (EBI), family/clan labelled",
    "afdb": "AlphaFold DB clustered sequences, Foldseek structural clusters (Barrio-Hernandez et al.)",
    "stringdb": "STRING v12.0 physical links, combined_score >= 400",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("/storage/users/ddofer/data/protsent-data-dc40"),
    )
    args = ap.parse_args()

    report = json.loads((args.out_dir / "decontam_report.json").read_text())
    corpora = report["corpora"]

    rows = []
    for name, s in corpora.items():
        rows.append(
            f"| `{s['output'].split('/')[-1]}` | {s['rows_before']:,} | {s['rows_after']:,} | "
            f"{s['rows_removed']:,} | {s['pct_removed']:.3f}% | {s['test_set']} | "
            f"{s.get('prefilter', report.get('prefilter', 'n/a'))} |"
        )
    table = "\n".join(rows)

    detail = []
    for name, s in corpora.items():
        groups = (
            f"{s['groups_before']:,} → {s['groups_after']:,}"
            if s.get("groups_before") is not None
            else "n/a (pair table)"
        )
        detail.append(
            f"### {name}\n\n"
            f"- Source: {SOURCE.get(name, 'n/a')}\n"
            f"- Internal redundancy reduction: {INTERNAL_REDUNDANCY.get(name, 'n/a')}\n"
            f"- Test-set filter: removed anything matching **{s['test_set']}** at "
            f"≥{report['min_seq_id']:.0%} identity over ≥{report['coverage']:.0%} of the test sequence\n"
            f"- Unique sequences searched: {s['unique_sequences_searched']:,}; "
            f"leaked: {s['leaked_unique_sequences']:,}\n"
            f"- Rows: {s['rows_before']:,} → {s['rows_after']:,} "
            f"({s['rows_removed']:,} removed, {s['pct_removed']:.3f}%)\n"
            f"  - by direct sequence match: {s.get('rows_removed_by_sequence_hit', 0):,}\n"
            f"  - as newly-stranded singleton clusters: {s.get('rows_removed_as_new_singletons', 0):,}\n"
            f"- Groups (clusters/families): {groups}\n"
        )

    card = f"""# ProtSent pretraining data — benchmark-decontaminated

Pretraining corpora for ProtSent contrastive protein embedding models, with
sequences similar to the evaluation test sets removed.

**Every corpus here is filtered at ≥{report['min_seq_id']:.0%} sequence identity over
≥{report['coverage']:.0%} coverage of the benchmark test sequence** (MMseqs2
`--min-seq-id {report['min_seq_id']} --cov-mode {report['cov_mode']} -c {report['coverage']}`).

## Contents

| file | rows before | rows after | removed | % | filtered against | prefilter |
|---|---|---|---|---|---|---|
{table}

## Method

`mmseqs easy-search`, with the **pretraining corpus as the query and the benchmark
test set as the target**. That orientation matters: each corpus sequence only needs
*any* hit to be dropped, so the default `--max-seqs` prefilter cap is harmless. In the
reverse orientation the cap silently truncates at 300 hits per test sequence, and the
E-value threshold also scales with target-database size, both of which under-remove.

`--cov-mode 1` is coverage of the *target* (the test sequence), so a long corpus protein
that merely **contains** a test-length domain is still caught.

### Why not linclust / linsearch

Measured on 200k Pfam sequences against the 3,244-sequence remote-homology test set,
identical thresholds:

| method | time | recall vs exhaustive |
|---|---|---|
| `easy-linclust` @40% over the union | 6 s | 15.6% |
| `easy-linsearch` @40% | 6 s | 0% |
| `easy-search -s 5.7` | 14 s | 89.4% |
| `easy-search -s 7.5` | 21 s | 97.6% |
| `easy-search` exhaustive / GPU prefilter | 94 s | 100% (identical hit sets) |

Clustering is structurally unfit for decontamination: cluster membership is exclusive,
so a corpus↔test similarity is lost whenever the corpus sequence has a closer corpus
neighbour. `easy-linsearch` returns nothing even at 90% identity here, because
`createlinindex` samples only 21 k-mers per sequence from a 3,244-sequence target index.

### Per-corpus detail

{"".join(detail)}

## Provenance

- Reproduce with `decontaminate_pretrain.py` in the ProtSent repo.
- `decontam/` holds the audit trail: the test-set FASTAs, every hit with its identity
  and coverage (`*_hits.tsv.gz`), and the exact leaked sequences (`*_leaked_sequences.parquet`).
- Row order is preserved from the source parquets (already group-sorted), and clusters
  left with a single member after filtering are dropped, since they yield no positive pairs.
- Schemas are unchanged from the source parquets and are drop-in for `protein_pipeline.py train`.

## Caveats

- Internal redundancy cutoffs differ per corpus (see above); they are a separate,
  pre-existing step and are **not** harmonised.
- Decontamination targets the benchmark test splits listed above only. Other evaluation
  sets are not filtered against.
"""

    residual = [
        (n, s) for n, s in corpora.items()
        if "kmer" in str(s.get("prefilter", report.get("prefilter", "")))
    ]
    if residual:
        card += "\n### Residual leakage\n\n"
        for n, s in residual:
            card += (
                f"- **{n}** used `{s.get('prefilter')}`, measured at **89.4% recall** "
                f"relative to the exhaustive prefilter. Roughly **10.6% of sequences that "
                f"would match {s['test_set']}** at this threshold therefore remain. The other "
                f"corpora were filtered at 100% recall.\n"
            )

    (args.out_dir / "README.md").write_text(card)
    print(f"wrote {args.out_dir / 'README.md'} ({len(card):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
