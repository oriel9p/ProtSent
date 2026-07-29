#!/usr/bin/env python
"""Prove the corpus ProtSent-V2 actually trained on contains no leaked sequence.

The decontamination step wrote its outputs to protsent-data-dc40/ and recorded a
per-corpus list of the sequences it removed. That is a record of *intent*. This
checks the *result*: it re-reads the exact parquet files the training job opened
(taken from the training log, not assumed) and semi-joins them against those
removal lists. Any non-zero count means a sequence MMseqs2 flagged as matching a
benchmark test sequence at >=40% identity / >=80% coverage survived into
training, and the decontamination claim in the rebuttal does not hold.

The STRING file needs this most: stringdb_train_15M.parquet is a 15M-pair
subsample created two days after the filtering run, and nothing in the data
directory records which parent it was drawn from. Membership is the only way to
tell a subsample of the filtered file from a subsample of the original.

Sequences are compared exactly, matching how the filter removed them (by
sequence hit, not by alignment).

Usage:
    python verify_training_corpus.py
    python verify_training_corpus.py --selfcheck
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

DATA = Path("/storage/users/ddofer/data/protsent-data-dc40")
LEAKED = DATA / "decontam"

# (training file, sequence columns, matching leaked-sequence list)
CHECKS = [
    ("pfam_sorted.parquet", ["sequence"], "pfam_leaked_sequences.parquet"),
    ("afdb_sorted.parquet", ["sequence"], "afdb_leaked_sequences.parquet"),
    ("stringdb_train_15M.parquet", ["seq1", "seq2"], "stringdb_leaked_sequences.parquet"),
]


def count_leaked(train_file: Path, seq_cols: list[str], leaked_file: Path) -> dict:
    """Rows in `train_file` whose sequence appears in `leaked_file`.

    Streamed: afdb_sorted is 126M rows / 12 GB and will not fit in memory as a
    materialised join.
    """
    leaked = pl.scan_parquet(leaked_file).select("sequence").unique()

    n_rows = pl.scan_parquet(train_file).select(pl.len()).collect(engine="streaming").item()
    hits = {}
    for col in seq_cols:
        n = (
            pl.scan_parquet(train_file)
            .select(pl.col(col).alias("sequence"))
            .join(leaked, on="sequence", how="semi")
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        hits[col] = int(n)

    return {
        "train_file": str(train_file),
        "leaked_list": str(leaked_file),
        "rows": int(n_rows),
        "leaked_rows_per_column": hits,
        "leaked_total": int(sum(hits.values())),
        "clean": sum(hits.values()) == 0,
    }


def main() -> int:
    report = {"data_dir": str(DATA), "checks": []}
    ok = True
    for name, cols, leaked_name in CHECKS:
        train_file, leaked_file = DATA / name, LEAKED / leaked_name
        if not train_file.exists():
            print(f"MISSING {train_file}", file=sys.stderr)
            return 2
        print(f"checking {name} ({', '.join(cols)}) against {leaked_name} ...", flush=True)
        res = count_leaked(train_file, cols, leaked_file)
        report["checks"].append(res)
        ok &= res["clean"]
        status = "CLEAN" if res["clean"] else f"LEAKED {res['leaked_total']}"
        print(f"  {res['rows']:,} rows -> {status}  {res['leaked_rows_per_column']}", flush=True)

    report["all_clean"] = ok
    out = Path("results/benchmarks/training_corpus_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{'ALL CLEAN' if ok else 'LEAKAGE FOUND'} -- wrote {out}")
    return 0 if ok else 1


def _selfcheck() -> None:
    """A planted leak must be caught, and a clean file must pass."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        pl.DataFrame({"sequence": ["AAA", "BBB"]}).write_parquet(p / "leaked.parquet")

        pl.DataFrame({"sequence": ["CCC", "DDD", "EEE"]}).write_parquet(p / "clean.parquet")
        r = count_leaked(p / "clean.parquet", ["sequence"], p / "leaked.parquet")
        assert r["clean"] and r["rows"] == 3, r

        pl.DataFrame({"sequence": ["CCC", "AAA", "EEE"]}).write_parquet(p / "dirty.parquet")
        r = count_leaked(p / "dirty.parquet", ["sequence"], p / "leaked.parquet")
        assert not r["clean"] and r["leaked_total"] == 1, r

        # Pair files: a leak in either column counts, and both are reported.
        pl.DataFrame(
            {"seq1": ["CCC", "AAA"], "seq2": ["BBB", "DDD"]}
        ).write_parquet(p / "pairs.parquet")
        r = count_leaked(p / "pairs.parquet", ["seq1", "seq2"], p / "leaked.parquet")
        assert r["leaked_rows_per_column"] == {"seq1": 1, "seq2": 1}, r
        assert r["leaked_total"] == 2, r
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
