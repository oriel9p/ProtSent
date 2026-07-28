#!/usr/bin/env python3
"""Per-SCOPe-sequence maximum identity to the ProtSent pretraining corpus.

Consumes the streaming reductions written by
``decontam_work/scope_strat/reduce.sh`` and emits
``scope40_max_identity.parquet`` plus the binned distribution table.

Two identity metrics are carried, because the mmseqs search is deliberately
permissive (``--min-seq-id 0 -c 0 -e 10``) so the low-identity tail survives:

``max_ident_*``       fident * tcov -- identities over the FULL length of the
                     SCOPe sequence. This is the headline number. Raw mmseqs
                     ``fident`` alone is identity over the aligned region only,
                     which at -c 0 is dominated by ~10-residue noise hits
                     scoring fident=1.0.
``max_fident_hicov_*`` raw fident, counting only hits covering >= 80% of the
                     SCOPe sequence -- directly comparable to the threshold the
                     decontamination run used (``--cov-mode 1 -c 0.8``).

``*_dc`` columns are the same quantities against the decontaminated (dc40)
corpus. SCOPe sequences with no hit get 0.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

WORK = Path("/storage/users/ddofer/data/decontam_work/scope_strat")
CORPORA = ("afdb", "pfam", "stringdb")
BINS = [0.0, 0.2, 0.4, 0.7, 1.0001]
BIN_LABELS = ["[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.7)", "[0.7, 1.0]"]
# reduce.sh column order, after `target`
METRICS = ["max_ident", "max_fident_hicov", "n_hits",
           "max_ident_dc", "max_fident_hicov_dc", "n_hits_dc"]


def read_fasta(path: Path) -> pl.DataFrame:
    """>{idx:08d}|{family} -> (target, scope_idx, family, sequence)."""
    rows, header, chunks = [], None, []

    def flush():
        if header is not None:
            rows.append((header, *header.split("|", 1), "".join(chunks)))

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    flush()
    return pl.DataFrame(
        rows, schema=["target", "scope_idx", "family", "sequence"], orient="row"
    ).with_columns(pl.col("scope_idx").cast(pl.Int32))


def read_summary(corpus: str) -> pl.DataFrame:
    """Missing corpus -> empty frame, so partial runs still produce a table."""
    cols = ["target"] + [f"{m}_{corpus}" for m in METRICS]
    path = WORK / f"{corpus}.summary.tsv"
    if not path.exists():
        print(f"WARNING: {path.name} missing -- {corpus} treated as zero hits")
        return pl.DataFrame(
            schema={cols[0]: pl.String, **{c: pl.Float64 for c in cols[1:]}}
        )
    return pl.read_csv(path, separator="\t", has_header=False, new_columns=cols)


def bin_table(values: np.ndarray) -> list[tuple[str, int, float]]:
    counts = np.histogram(values, bins=BINS)[0]
    n = len(values)
    return [(lbl, int(c), 100 * c / n) for lbl, c in zip(BIN_LABELS, counts)]


def main() -> int:
    df = read_fasta(WORK / "scope40.fasta")
    for corpus in CORPORA:
        df = df.join(read_summary(corpus), on="target", how="left")
    df = df.fill_null(0)

    for m in METRICS:
        cols = [f"{m}_{c}" for c in CORPORA]
        agg = pl.sum_horizontal if m.startswith("n_hits") else pl.max_horizontal
        df = df.with_columns(agg(cols).alias(f"{m}_overall"))

    out = df.select(
        "scope_idx", "family", "sequence",
        *[f"{m}_overall" for m in METRICS],
        *[f"{m}_{c}" for m in METRICS for c in CORPORA],
    ).sort("scope_idx")
    out.write_parquet(WORK / "scope40_max_identity.parquet")

    print(f"\nSCOPe-40 sequences: {out.height}")
    for label, col in [
        ("max_ident (fident x tcov, full corpus)", "max_ident_overall"),
        ("max_ident (fident x tcov, decontaminated)", "max_ident_dc_overall"),
        ("max_fident @ tcov>=0.8 (full corpus)", "max_fident_hicov_overall"),
        ("max_fident @ tcov>=0.8 (decontaminated)", "max_fident_hicov_dc_overall"),
    ]:
        print(f"\n{label}")
        print(f"{'bin':<12}{'n':>8}{'%':>9}")
        for lbl, c, pct in bin_table(out[col].to_numpy()):
            print(f"{lbl:<12}{c:>8}{pct:>8.2f}%")
    print(f"\nwrote {WORK / 'scope40_max_identity.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
