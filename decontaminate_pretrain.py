#!/usr/bin/env python3
"""Decontaminate ProtSent pretraining corpora against benchmark test sets.

Removes any pretraining sequence that aligns to a benchmark test sequence at
>= --min-seq-id identity with >= --cov coverage of the test sequence
(``--cov-mode 1``), using MMseqs2 ``easy-search``.

Orientation is deliberate: the pretraining corpus is the QUERY and the test set
is the TARGET. Each pretraining sequence only needs *any* hit to be dropped, so
the default ``--max-seqs`` prefilter cap is harmless. The reverse orientation
would silently truncate at 300 hits per test sequence and under-remove.

``--cov-mode 1`` is coverage of the target (= the test sequence), so a long
pretraining protein that merely *contains* a test-length domain is still caught.
This matches the convention already used in ``data_prep.py``.

Usage:
    python decontaminate_pretrain.py --corpus pfam
    python decontaminate_pretrain.py --corpus all --gpu
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SRC_DIR = Path("/storage/users/ddofer/data")
OUT_DIR = SRC_DIR / "protsent-data-dc40"
WORK_DIR = SRC_DIR / "decontam_work"
HF_HUB = Path("/storage/models/hf_home/hub")

# Benchmark test sets, read straight from the local HF snapshot cache so this
# runs fully offline (huggingface.co is unreachable from this host).
TEST_SETS = {
    "fold": {
        "glob": str(
            HF_HUB
            / "datasets--biomap-research--fold_prediction/snapshots/*/data/test-*.parquet"
        ),
        "cols": ["seq"],
        "prefix": "FOLD",
        "task": "remote_homology (biomap-research/fold_prediction[test])",
    },
    "bernett": {
        "glob": str(
            HF_HUB
            / "datasets--Synthyra--bernett_gold_ppi/snapshots/*/data/test-*.parquet"
        ),
        "cols": ["SeqA", "SeqB"],
        "prefix": "BERNETT",
        "task": "ppi_bernett (Synthyra/bernett_gold_ppi[test])",
    },
}

# group_col is the finest grouping level, mirroring `_sort_and_save`'s
# "drop clusters with a single member" rule (data_prep.py:306-310). Removing
# leaked members can strand new singletons, which yield zero contrastive pairs.
CORPORA = {
    "pfam": {
        "src": SRC_DIR / "pfam_sorted.parquet",
        "out": "pfam_sorted.parquet",
        "seq_cols": ["sequence"],
        "group_col": "group_id",
        "test_set": "fold",
    },
    "afdb": {
        "src": SRC_DIR / "afdb_sorted.parquet",
        "out": "afdb_sorted.parquet",
        "seq_cols": ["sequence"],
        "group_col": "group_id",
        "test_set": "fold",
    },
    "stringdb": {
        "src": SRC_DIR / "stringdb_train.parquet",
        "out": "stringdb_train.parquet",
        "seq_cols": ["seq1", "seq2"],
        "group_col": None,  # pair table, no cluster column persisted
        "test_set": "bernett",
    },
}


def resolve_mmseqs() -> str:
    """Locate the mmseqs binary, preferring the bundled one (as data_prep.py does)."""
    bundled = Path(__file__).resolve().parent / "tools" / "mmseqs" / "bin" / "mmseqs"
    if bundled.exists():
        logger.info("Using bundled mmseqs: %s", bundled)
        return str(bundled)
    try:
        result = subprocess.run(
            ["mmseqs", "version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("Using system mmseqs: %s", result.stdout.strip())
            return "mmseqs"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError(
        "MMseqs2 not found. Install the GPU build:\n"
        "  curl -L -o mmseqs.tar.gz https://github.com/soedinglab/MMseqs2/"
        "releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz\n"
        "  tar xzf mmseqs.tar.gz && ln -s $PWD/mmseqs <repo>/tools/mmseqs"
    )


def build_test_fasta(name: str, out_path: Path) -> int:
    """Write the benchmark test split to FASTA. Returns the unique-sequence count.

    Header convention matches data_prep.py:1079-1082 (dedupe -> sort ->
    >PREFIX_%08d) so the output is reproducible and hashable.
    """
    spec = TEST_SETS[name]
    files = sorted(glob.glob(spec["glob"]))
    if not files:
        raise FileNotFoundError(
            f"No cached parquet for test set '{name}' at {spec['glob']}. "
            "huggingface.co is unreachable from this host, so the snapshot must "
            "already be in /storage/models/hf_home."
        )

    seqs: set[str] = set()
    for f in files:
        table = pq.read_table(f, columns=spec["cols"])
        for col in spec["cols"]:
            seqs.update(s for s in table.column(col).to_pylist() if s)

    ordered = sorted(seqs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for i, seq in enumerate(ordered, start=1):
            fh.write(f">{spec['prefix']}_{i:08d}\n{seq}\n")
    logger.info(
        "Test set '%s': %d unique sequences from %d file(s) -> %s",
        name,
        len(ordered),
        len(files),
        out_path,
    )
    return len(ordered)


def build_unique_sequences(src: Path, seq_cols: list[str], out_parquet: Path) -> int:
    """Stream the corpus to a deduplicated single-column ``sequence`` parquet."""
    if out_parquet.exists():
        n = pq.ParquetFile(out_parquet).metadata.num_rows
        logger.info("Reusing %s (%s unique sequences)", out_parquet.name, f"{n:,}")
        return n

    lf = pl.scan_parquet(src)
    if len(seq_cols) == 1:
        uniq = lf.select(pl.col(seq_cols[0]).alias("sequence"))
    else:
        uniq = pl.concat(
            [lf.select(pl.col(c).alias("sequence")) for c in seq_cols], how="vertical"
        )
    uniq = uniq.drop_nulls().unique(subset="sequence")

    tmp = out_parquet.with_suffix(".tmp.parquet")
    t0 = time.time()
    uniq.sink_parquet(tmp)
    tmp.replace(out_parquet)
    n = pq.ParquetFile(out_parquet).metadata.num_rows
    logger.info(
        "Unique sequences in %s: %s (%.1f min)", src.name, f"{n:,}", (time.time() - t0) / 60
    )
    return n


def write_fasta_shards(uniq_parquet: Path, shard_dir: Path, shard_size: int) -> list[Path]:
    """One streaming pass over the unique sequences -> sharded FASTA.

    The FASTA header is the 0-based row index in ``uniq_parquet``, so hits map
    straight back without storing an id column.
    """
    done_marker = shard_dir / ".shards_complete"
    if done_marker.exists():
        shards = sorted(shard_dir.glob("shard_*.fasta"))
        logger.info("Reusing %d existing FASTA shard(s) in %s", len(shards), shard_dir)
        return shards

    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards: list[Path] = []
    handle = None
    shard_rows = 0
    global_idx = 0
    t0 = time.time()

    pf = pq.ParquetFile(uniq_parquet)
    for batch in pf.iter_batches(batch_size=250_000, columns=["sequence"]):
        if handle is None or shard_rows >= shard_size:
            if handle is not None:
                handle.close()
            path = shard_dir / f"shard_{len(shards):04d}.fasta"
            handle = open(path, "wb")
            shards.append(path)
            shard_rows = 0
        seqs = batch.column("sequence").to_pylist()
        handle.write(
            b"".join(
                b">%d\n%s\n" % (global_idx + i, s.encode())
                for i, s in enumerate(seqs)
            )
        )
        global_idx += len(seqs)
        shard_rows += len(seqs)
    if handle is not None:
        handle.close()

    done_marker.touch()
    logger.info(
        "Wrote %d FASTA shard(s), %s sequences (%.1f min)",
        len(shards),
        f"{global_idx:,}",
        (time.time() - t0) / 60,
    )
    return shards


def run_search(
    mmseqs: str,
    query_fasta: Path,
    target_fasta: Path,
    out_tsv: Path,
    tmp_dir: Path,
    *,
    min_seq_id: float,
    cov: float,
    cov_mode: int,
    threads: int,
    prefilter: str,
    sensitivity: float,
    timeout: int,
    device: Optional[str] = None,
) -> None:
    """Run one mmseqs easy-search shard. Idempotent: skips if out_tsv exists."""
    if out_tsv.exists():
        logger.info("Reusing existing hits: %s", out_tsv.name)
        return

    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs, "easy-search",
        str(query_fasta), str(target_fasta), str(out_tsv), str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "--cov-mode", str(cov_mode),
        "-c", str(cov),
        "--alignment-mode", "3",
        "-e", "1e-3",
        "--threads", str(threads),
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue",
        "--remove-tmp-files", "-v", "3",
    ]
    # Measured recall against the exhaustive prefilter, 200k Pfam seqs vs the 3,244
    # fold-test seqs (see decontam_report.json "prefilter"):
    #   gpu / exhaustive   100%    (identical hit sets)
    #   kmer -s 7.5        97.6%
    #   kmer -s 5.7        89.4%   ~7x faster than exhaustive
    #   easy-linclust@0.4  15.6%   structurally unfit: cluster membership is
    #                              exclusive, so a corpus<->test link is lost
    #                              whenever the corpus seq has a closer neighbour
    if prefilter == "gpu":
        cmd += ["--gpu", "1"]
    elif prefilter == "exhaustive":
        cmd += ["--prefilter-mode", "1"]
    else:
        cmd += ["-s", str(sensitivity)]

    env = None
    if device is not None and prefilter == "gpu":
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}

    t0 = time.time()
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.CalledProcessError as exc:
        out_tsv.unlink(missing_ok=True)
        raise RuntimeError(
            f"mmseqs easy-search failed ({' '.join(cmd)}):\n{exc.stderr[-4000:]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        out_tsv.unlink(missing_ok=True)
        raise RuntimeError(
            f"mmseqs easy-search timed out after {timeout}s ({' '.join(cmd)})"
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "  %s%s -> %s hits (%.1f min)",
        query_fasta.name,
        f" [gpu {device}]" if device else "",
        f"{sum(1 for _ in open(out_tsv)):,}",
        (time.time() - t0) / 60,
    )


def collect_leaked_sequences(
    hit_files: list[Path], uniq_parquet: Path, out_parquet: Path
) -> pl.Series:
    """Map hit query ids (row indices) back to sequences."""
    indices: set[int] = set()
    for path in hit_files:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    indices.add(int(line.split("\t", 1)[0]))

    if not indices:
        logger.warning("No hits at all — nothing will be removed.")
        empty = pl.DataFrame({"sequence": []}, schema={"sequence": pl.String})
        empty.write_parquet(out_parquet)
        return empty["sequence"]

    leaked = (
        pl.scan_parquet(uniq_parquet)
        .with_row_index("__idx")
        .filter(pl.col("__idx").is_in(sorted(indices)))
        .select("sequence")
        .collect()
    )
    leaked.write_parquet(out_parquet)
    logger.info("Leaked unique sequences: %s", f"{leaked.height:,}")
    return leaked["sequence"]


def filter_corpus(
    src: Path,
    dst: Path,
    seq_cols: list[str],
    group_col: Optional[str],
    leaked: pl.Series,
) -> dict:
    """Drop leaked rows, then re-drop groups left with a single member.

    Row order is preserved (the sources are already sorted and both operations
    are order-preserving), so the group-contiguity the training pair-builder
    relies on survives untouched.
    """
    rows_before = pq.ParquetFile(src).metadata.num_rows

    def keep_expr():
        expr = ~pl.col(seq_cols[0]).is_in(leaked)
        for col in seq_cols[1:]:
            expr = expr & ~pl.col(col).is_in(leaked)
        return expr

    lf = pl.scan_parquet(src).filter(keep_expr())

    singletons_dropped = 0
    if group_col is not None:
        # One pass gives both the post-sequence-filter row count and the groups
        # that still have >1 member. is_in (rather than a join) is used because
        # it is guaranteed order-preserving.
        counts = lf.group_by(group_col).len().collect()
        after_seq_filter = int(counts["len"].sum())
        valid = counts.filter(pl.col("len") > 1)[group_col]
        lf = lf.filter(pl.col(group_col).is_in(valid))
    else:
        after_seq_filter = None

    tmp = dst.with_suffix(".tmp.parquet")
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lf.sink_parquet(tmp)
    tmp.replace(dst)
    rows_after = pq.ParquetFile(dst).metadata.num_rows

    if after_seq_filter is not None:
        singletons_dropped = after_seq_filter - rows_after

    logger.info(
        "%s: %s -> %s rows (removed %s, %.4f%%) in %.1f min",
        src.name,
        f"{rows_before:,}",
        f"{rows_after:,}",
        f"{rows_before - rows_after:,}",
        100 * (rows_before - rows_after) / max(1, rows_before),
        (time.time() - t0) / 60,
    )
    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "pct_removed": round(100 * (rows_before - rows_after) / max(1, rows_before), 4),
        "rows_removed_by_sequence_hit": (
            rows_before - after_seq_filter if after_seq_filter is not None else rows_before - rows_after
        ),
        "rows_removed_as_new_singletons": singletons_dropped,
    }


def group_stats(path: Path, group_col: Optional[str]) -> Optional[int]:
    if group_col is None:
        return None
    return (
        pl.scan_parquet(path).select(pl.col(group_col).n_unique()).collect().item()
    )


def process_corpus(name: str, args, mmseqs: str) -> dict:
    cfg = CORPORA[name]
    src: Path = cfg["src"]
    if not src.exists():
        raise FileNotFoundError(src)

    logger.info("=" * 78)
    logger.info("CORPUS: %s  (%s)", name, src)
    logger.info("=" * 78)

    work = WORK_DIR / name
    work.mkdir(parents=True, exist_ok=True)
    decontam_dir = OUT_DIR / "decontam"
    decontam_dir.mkdir(parents=True, exist_ok=True)

    test_name = cfg["test_set"]
    test_fasta = decontam_dir / f"{test_name}_test.fasta"
    if not test_fasta.exists():
        build_test_fasta(test_name, test_fasta)
    n_test = sum(1 for line in open(test_fasta) if line.startswith(">"))

    uniq_parquet = work / "uniq.parquet"
    n_uniq = build_unique_sequences(src, cfg["seq_cols"], uniq_parquet)

    shards = write_fasta_shards(uniq_parquet, work / "shards", args.shard_size)

    devices = (
        args.gpu_devices.split(",")
        if (args.prefilter == "gpu" and args.gpu_devices)
        else [None]
    )
    hit_files = [work / f"hits_{i:04d}.tsv" for i in range(len(shards))]

    def search_shard(i: int) -> None:
        logger.info("Searching shard %d/%d ...", i + 1, len(shards))
        run_search(
            mmseqs,
            shards[i],
            test_fasta,
            hit_files[i],
            work / f"tmp_{i:04d}",
            min_seq_id=args.min_seq_id,
            cov=args.cov,
            cov_mode=args.cov_mode,
            threads=args.threads,
            prefilter=args.prefilter,
            sensitivity=args.sensitivity,
            timeout=args.timeout,
            device=devices[i % len(devices)],
        )

    if len(devices) > 1 and len(shards) > 1:
        logger.info("Searching %d shard(s) across GPUs %s", len(shards), devices)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for _ in pool.map(search_shard, range(len(shards))):
                pass
    else:
        for i in range(len(shards)):
            search_shard(i)

    leaked = collect_leaked_sequences(
        hit_files, uniq_parquet, decontam_dir / f"{name}_leaked_sequences.parquet"
    )

    merged_hits = decontam_dir / f"{name}_hits.tsv"
    with open(merged_hits, "wb") as out:
        out.write(b"query\ttarget\tfident\talnlen\tqcov\ttcov\tevalue\n")
        for path in hit_files:
            with open(path, "rb") as fh:
                shutil.copyfileobj(fh, out)
    subprocess.run(["gzip", "-f", str(merged_hits)], check=True)

    groups_before = group_stats(src, cfg["group_col"])
    stats = filter_corpus(
        src, OUT_DIR / cfg["out"], cfg["seq_cols"], cfg["group_col"], leaked
    )
    groups_after = group_stats(OUT_DIR / cfg["out"], cfg["group_col"])

    stats.update(
        {
            "corpus": name,
            # Recorded per corpus: corpora may be filtered in separate invocations
            # with different prefilters, and the card must not mislabel them.
            "prefilter": (
                f"kmer -s {args.sensitivity} (89.4% recall vs exhaustive)"
                if args.prefilter == "kmer"
                else f"{args.prefilter} (exhaustive ungapped, 100% recall)"
            ),
            "source": str(src),
            "output": str(OUT_DIR / cfg["out"]),
            "test_set": TEST_SETS[test_name]["task"],
            "test_sequences": n_test,
            "unique_sequences_searched": n_uniq,
            "leaked_unique_sequences": leaked.len(),
            "groups_before": groups_before,
            "groups_after": groups_after,
            "shards": len(shards),
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default="all", choices=[*CORPORA, "all"],
        help="Which pretraining corpus to decontaminate.",
    )
    parser.add_argument("--min-seq-id", type=float, default=0.4)
    parser.add_argument("--cov", type=float, default=0.8)
    parser.add_argument(
        "--cov-mode", type=int, default=1,
        help="1 = coverage of the target (the test sequence).",
    )
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument(
        "--prefilter", default="gpu", choices=["gpu", "exhaustive", "kmer"],
        help="gpu/exhaustive = 100%% recall (identical hit sets, ~O(corpus x test) cost); "
        "kmer = -s <sensitivity>, ~7x faster at 89-98%% recall.",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Deprecated alias for --prefilter gpu (the default).",
    )
    parser.add_argument(
        "--gpu-devices", default="4,5,6,7",
        help="Comma-separated CUDA device ids; shards are searched in parallel, one per device.",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=5.7,
        help="-s value for --prefilter kmer. 5.7 = 89.4%% recall, 7.5 = 97.6%%.",
    )
    parser.add_argument("--shard-size", type=int, default=10_000_000)
    parser.add_argument("--timeout", type=int, default=86_400)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    globals()["OUT_DIR"] = args.out_dir
    mmseqs = resolve_mmseqs()

    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    report = {
        "min_seq_id": args.min_seq_id,
        "coverage": args.cov,
        "cov_mode": args.cov_mode,
        "prefilter": (
            f"kmer -s {args.sensitivity}"
            if args.prefilter == "kmer"
            else f"{args.prefilter} (exhaustive ungapped, 100% recall)"
        ),
        "search": "mmseqs easy-search",
        "orientation": "pretrain corpus = query, benchmark test set = target",
        "corpora": {},
    }

    report_path = args.out_dir / "decontam_report.json"
    if report_path.exists():
        report["corpora"] = json.loads(report_path.read_text()).get("corpora", {})

    for name in names:
        report["corpora"][name] = process_corpus(name, args, mmseqs)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 108)
    print(f"{'corpus':<10} {'rows before':>15} {'rows after':>15} {'removed':>12} {'%':>8} "
          f"{'uniq searched':>15} {'leaked':>9} {'groups before/after':>22}")
    print("-" * 108)
    for name, s in report["corpora"].items():
        grp = (
            f"{s['groups_before']:,} / {s['groups_after']:,}"
            if s.get("groups_before") is not None
            else "n/a (pair table)"
        )
        print(
            f"{name:<10} {s['rows_before']:>15,} {s['rows_after']:>15,} "
            f"{s['rows_removed']:>12,} {s['pct_removed']:>7.4f}% "
            f"{s['unique_sequences_searched']:>15,} {s['leaked_unique_sequences']:>9,} {grp:>22}"
        )
    print("=" * 108)
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
