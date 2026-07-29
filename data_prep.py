"""
Protein data preparation utilities.

This module handles dataset download, wrangling, validation, and Parquet export
for training the Protein-SBERT pipeline.
"""

import argparse
import gzip
import hashlib
import itertools
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional, cast

import numpy as np
import polars as pl
import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ProteinPipeline")
logging.getLogger("httpx").setLevel(logging.WARNING)

_SPLIT_COLUMN_CANDIDATES = ("stage", "split", "set", "fold")
_TRAIN_SPLIT_VALUES = {"train", "training"}
_TEST_SPLIT_VALUES = {"test", "testing"}
_AFDB_ALLOWED_CLU_FLAGS = (1, 2)


PFAM_HN_AA = "ACDEFGHIKLMNPQRSTVWY"
PFAM_HN_AA_INDEX = {aa: i for i, aa in enumerate(PFAM_HN_AA)}
# Substitutions are drawn only from residues this common in the background
# proteome, so a negative keeps a natural amino-acid composition instead of
# filling up with the rarest residues the profile happens to disfavour.
PFAM_HN_MIN_BACKGROUND_FREQ = 0.04


def _hmm_match_log_odds(hmm, log_background: np.ndarray) -> np.ndarray:
    """Per-match-state log-odds matrix for one profile HMM.

    Args:
        hmm: A ``pyhmmer.plan7.HMM``.
        log_background: log2 background frequencies, shape (20,).

    Returns:
        Array of shape (model_len, 20) holding log2(emission) - log2(background).
    """
    emissions = np.asarray(hmm.match_emissions, dtype=np.float32)
    # Row 0 is the BEGIN state, which emits nothing.
    emissions = np.maximum(emissions[1:, : len(PFAM_HN_AA)], 1e-9)
    return np.log2(emissions) - log_background[np.newaxis, :]


def _alignment_position_map(domain) -> dict[int, int]:
    """Map sequence offsets to HMM match states using the real alignment.

    A Pfam domain is not gaplessly aligned to its model: one insert or delete
    shifts every downstream residue, so ``i -> i`` is wrong for most sequences.
    Walk the alignment HMMER produced instead.

    Args:
        domain: A ``pyhmmer.plan7.Domain`` from a search hit.

    Returns:
        Mapping of 0-based sequence offset to 0-based match-state index,
        containing only positions aligned to a match state.
    """
    alignment = domain.alignment
    position_map: dict[int, int] = {}
    seq_idx = alignment.target_from - 1
    hmm_idx = alignment.hmm_from - 1
    for hmm_char, target_char in zip(
        alignment.hmm_sequence, alignment.target_sequence
    ):
        if hmm_char == "." and target_char != "-":
            seq_idx += 1  # insertion relative to the model
        elif hmm_char != "." and target_char == "-":
            hmm_idx += 1  # deletion relative to the model
        else:
            position_map[seq_idx] = hmm_idx
            seq_idx += 1
            hmm_idx += 1
    return position_map


def _rank_substitutions(
    sequence: str,
    position_map: dict[int, int],
    log_odds: np.ndarray,
    allowed_aa: np.ndarray,
) -> list[tuple[float, int, str]]:
    """Rank aligned positions by how much the profile score they can destroy.

    Args:
        sequence: The wild-type sequence.
        position_map: Sequence offset -> match state, from the alignment.
        log_odds: Match-state log-odds, shape (model_len, 20).
        allowed_aa: Indices of substitutions permitted at any position.

    Returns:
        ``(delta, offset, replacement)`` tuples sorted most-damaging first.
        The sort is total, so the ranking is reproducible without a seed.
    """
    ranked: list[tuple[float, int, str]] = []
    model_len = log_odds.shape[0]
    for offset, state in position_map.items():
        if state >= model_len or offset >= len(sequence):
            continue
        wild_type = sequence[offset]
        wt_index = PFAM_HN_AA_INDEX.get(wild_type)
        if wt_index is None:
            continue
        delta = log_odds[state] - log_odds[state, wt_index]
        best = allowed_aa[int(np.argmin(delta[allowed_aa]))]
        if best == wt_index:
            continue
        ranked.append((float(delta[best]), offset, PFAM_HN_AA[best]))
    # Ties break on offset then residue, never on dict or thread ordering.
    ranked.sort()
    return ranked


def _apply_substitutions(
    sequence: str, ranked: list[tuple[float, int, str]], count: int
) -> str:
    """Apply the ``count`` most damaging ranked substitutions to a sequence."""
    residues = list(sequence)
    for _, offset, replacement in ranked[:count]:
        residues[offset] = replacement
    return "".join(residues)


def _hmm_search_named(hmm, named_sequences, evalue_z: float):
    """Search one HMM against named sequences with a pinned E-value database size.

    ``Z`` is pinned so a reported E-value depends only on the sequence and the
    model, not on how many candidates happen to share the batch.

    Args:
        hmm: A ``pyhmmer.plan7.HMM``.
        named_sequences: ``(name, sequence)`` pairs.
        evalue_z: Effective database size used for E-value calculation.

    Returns:
        Mapping of name to ``(bit_score, evalue, top_domain)``. Names that did
        not produce a hit are absent.
    """
    from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequence
    from pyhmmer.plan7 import Pipeline

    alphabet = Alphabet.amino()
    block = DigitalSequenceBlock(
        alphabet,
        [
            TextSequence(name=name.encode(), sequence=sequence).digitize(alphabet)
            for name, sequence in named_sequences
        ],
    )
    # Report everything: the caller decides what counts as a miss, and a
    # filtered-out hit would look like a negative when it is not.
    pipeline = Pipeline(
        alphabet,
        E=1e9,
        incE=1e9,
        Z=evalue_z,
        domZ=evalue_z,
        bias_filter=False,
        F1=1.0,
        F2=1.0,
        F3=1.0,
    )
    results = {}
    for hit in pipeline.search_hmm(hmm, block):
        name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
        results[name] = (
            hit.score,
            hit.evalue,
            hit.domains[0] if hit.domains else None,
        )
    return results


def _generate_family_negatives(
    hmm,
    sequences: list[str],
    log_background: np.ndarray,
    max_evalue: float = 1.0,
    evalue_z: float = 1e6,
    max_mutation_fraction: float = 0.5,
    min_aligned_positions: int = 20,
) -> list[str | None]:
    """Generate one verified hard negative per sequence for a single family.

    A negative is accepted only when HMMER, re-run on the mutant, no longer
    reports it as a hit for this family above ``max_evalue``. The smallest
    mutation count that clears the bar is chosen, so the negative stays as
    close to the wild type as the criterion allows.

    Args:
        hmm: The family's profile HMM.
        sequences: Wild-type sequences belonging to the family.
        log_background: log2 background residue frequencies, shape (20,).
        max_evalue: A mutant is a negative once its E-value exceeds this.
        evalue_z: Effective database size for E-value calculation.
        max_mutation_fraction: Never mutate more than this fraction of the
            aligned positions; sequences that cannot clear the bar within the
            budget get ``None``.
        min_aligned_positions: Skip sequences aligning to fewer match states.

    Returns:
        One negative (or ``None``) per input sequence, in input order.
    """
    from pyhmmer.plan7 import Background

    negatives: list[str | None] = [None] * len(sequences)
    if not sequences:
        return negatives

    background_freq = np.maximum(
        np.array(
            [Background(_amino_alphabet()).residue_frequencies[i]
             for i in range(len(PFAM_HN_AA))],
            dtype=np.float32,
        ),
        1e-9,
    )
    allowed_aa = np.flatnonzero(background_freq >= PFAM_HN_MIN_BACKGROUND_FREQ)
    log_odds = _hmm_match_log_odds(hmm, log_background)

    wild_type_hits = _hmm_search_named(
        hmm, [(str(i), s) for i, s in enumerate(sequences)], evalue_z
    )

    ranked_by_index: dict[int, list[tuple[float, int, str]]] = {}
    budget: dict[int, int] = {}
    for index, sequence in enumerate(sequences):
        hit = wild_type_hits.get(str(index))
        if hit is None or hit[2] is None:
            continue  # not recognised as a member; nothing to break
        position_map = _alignment_position_map(hit[2])
        if len(position_map) < min_aligned_positions:
            continue
        ranked = _rank_substitutions(sequence, position_map, log_odds, allowed_aa)
        cap = int(len(ranked) * max_mutation_fraction)
        if cap < 1:
            continue
        ranked_by_index[index] = ranked
        budget[index] = cap

    def probe(counts: dict[int, int]) -> dict[int, bool]:
        """Mutate at the given counts and report which cleared max_evalue."""
        named = [
            (f"{index}|{count}", _apply_substitutions(
                sequences[index], ranked_by_index[index], count))
            for index, count in counts.items()
        ]
        hits = _hmm_search_named(hmm, named, evalue_z)
        cleared = {}
        for index, count in counts.items():
            hit = hits.get(f"{index}|{count}")
            # No hit at all is the strongest possible pass.
            cleared[index] = hit is None or hit[1] > max_evalue
        return cleared

    # Only sequences that clear the bar at full budget can clear it at all.
    feasible = probe(budget)
    low = {i: 1 for i, ok in feasible.items() if ok}
    high = {i: budget[i] for i in low}
    best = dict(high)

    while True:
        counts = {
            i: (low[i] + high[i]) // 2 for i in low if low[i] < high[i]
        }
        if not counts:
            break
        cleared = probe(counts)
        for index, count in counts.items():
            if cleared[index]:
                high[index] = count
                best[index] = count
            else:
                low[index] = count + 1

    for index, count in best.items():
        negatives[index] = _apply_substitutions(
            sequences[index], ranked_by_index[index], count
        )
    return negatives


def _amino_alphabet():
    """Return the pyhmmer amino-acid alphabet."""
    from pyhmmer.easel import Alphabet

    return Alphabet.amino()


def _download_file(url: str, dest_path: Path, attempts: int = 3) -> None:
    """Download a file with wget (preferred) or requests fallback.

    Args:
        url: Source URL to download from.
        dest_path: Local filesystem destination.
        attempts: Number of full download attempts before failing.

    Raises:
        RuntimeError: If the download fails via both methods.
    """
    if dest_path.exists():
        if dest_path.stat().st_size > 0:
            logger.info("Found existing: %s", dest_path.name)
            return
        logger.warning("Removing empty partial download: %s", dest_path)
        dest_path.unlink()

    logger.info("Downloading: %s", dest_path.name)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                [
                    "wget",
                    "-c",
                    "--tries=3",
                    "--timeout=30",
                    "--read-timeout=300",
                    "-O",
                    str(dest_path),
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if (
                result.returncode == 0
                and dest_path.exists()
                and dest_path.stat().st_size > 0
            ):
                logger.info("Downloaded: %s", dest_path.name)
                return
            if result.returncode != 0:
                logger.warning(
                    "wget attempt %d/%d failed for %s: %s",
                    attempt,
                    attempts,
                    dest_path.name,
                    result.stderr.strip()[-500:],
                )
        except FileNotFoundError as exc:
            last_error = exc
            logger.info("wget not available, using requests for %s", dest_path.name)
        except subprocess.SubprocessError as exc:
            last_error = exc
            logger.warning(
                "wget attempt %d/%d failed for %s: %s",
                attempt,
                attempts,
                dest_path.name,
                exc,
            )

        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
        try:
            with requests.get(
                url, stream=True, timeout=(30, 300), headers=headers
            ) as r:
                if r.status_code == 416 and existing_size > 0:
                    logger.info("Downloaded: %s", dest_path.name)
                    return
                r.raise_for_status()

                append = existing_size > 0 and r.status_code == 206
                if existing_size > 0 and not append:
                    logger.info(
                        "Server did not honor resume for %s; restarting download",
                        dest_path.name,
                    )
                    existing_size = 0

                content_length = int(r.headers.get("content-length", 0))
                total_size = existing_size + content_length if content_length > 0 else 0
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    initial=existing_size if total_size > existing_size else 0,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    mininterval=5.0,
                    desc=f"Downloading {dest_path.name}",
                    dynamic_ncols=True,
                    leave=False,
                ) as progress:
                    with open(dest_path, "ab" if append else "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            progress.update(len(chunk))

            if dest_path.exists() and dest_path.stat().st_size > 0:
                logger.info("Downloaded: %s", dest_path.name)
                return
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "requests attempt %d/%d failed for %s: %s",
                attempt,
                attempts,
                dest_path.name,
                exc,
            )

        if attempt < attempts:
            time.sleep(min(2**attempt, 30))

    if dest_path.exists() and dest_path.stat().st_size == 0:
        dest_path.unlink()
    raise RuntimeError(f"Failed to download {url} to {dest_path}") from last_error


def _load_afdb_foldseek_map_lazy(
    tsv_path: Path,
    allowed_clu_flags: tuple[int, ...] = _AFDB_ALLOWED_CLU_FLAGS,
) -> pl.LazyFrame:
    """Load AFDB structural mapping lazily from Steinegger TSV.

    Args:
        tsv_path: Path to 1-AFDBClusters-repId_entryId_cluFlag_taxId.tsv.gz.
        allowed_clu_flags: cluFlag values to include in the mapping.

    Returns:
        LazyFrame with columns: foldseek_rep, entry_id, clu_flag.
    """
    return (
        pl.scan_csv(
            tsv_path,
            separator="\t",
            has_header=False,
            new_columns=["foldseek_rep", "entry_id", "clu_flag", "tax_id"],
        )
        .with_columns(pl.col("clu_flag").cast(pl.Int8, strict=False))
        .filter(pl.col("clu_flag").is_in(list(allowed_clu_flags)))
        .select(["foldseek_rep", "entry_id", "clu_flag"])
    )


class DataPrep:
    _BENCHMARK_OVERLAP_PREFIXES = (
        "GB1_",
        "GFP_AEQVI_",
    )

    # Benchmark test splits that pretraining corpora are filtered against.
    # Keyed by short name -> (HF dataset, split, sequence columns, FASTA prefix).
    _BENCHMARK_TEST_SETS = {
        "bernett": ("Synthyra/bernett_gold_ppi", "test", ("protein1_sequence", "protein2_sequence", "SeqA", "SeqB"), "BERNETT"),
        "fold": ("biomap-research/fold_prediction", "test", ("seq",), "FOLD"),
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._mmseqs_bin_cache: Optional[str] = None

    def _resolve_mmseqs_binary(self) -> str:
        """Locate the mmseqs binary, preferring a bundled copy over PATH."""
        if self._mmseqs_bin_cache is not None:
            return self._mmseqs_bin_cache

        bundled = Path(self.data_dir).parent / "tools" / "mmseqs" / "bin" / "mmseqs"
        if not bundled.exists():
            bundled = (
                Path(__file__).resolve().parent / "tools" / "mmseqs" / "bin" / "mmseqs"
            )
        if bundled.exists():
            self._mmseqs_bin_cache = str(bundled)
            logger.info("Using bundled mmseqs: %s", self._mmseqs_bin_cache)
            return self._mmseqs_bin_cache

        try:
            result = subprocess.run(
                ["mmseqs", "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._mmseqs_bin_cache = "mmseqs"
                logger.info("Using system mmseqs: %s", result.stdout.strip())
                return self._mmseqs_bin_cache
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise RuntimeError(
            "MMseqs2 not found. Install the GPU build:\n"
            "  curl -L -o mmseqs.tar.gz https://github.com/soedinglab/MMseqs2/"
            "releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz\n"
            "  tar xzf mmseqs.tar.gz && ln -s $PWD/mmseqs <repo>/tools/mmseqs\n"
            "Or via bioconda: conda install -c bioconda mmseqs2"
        )

    def _build_benchmark_test_fasta(self, test_set: str, out_path: Path) -> list[str]:
        """Write a benchmark test split to FASTA (deduped, sorted). Returns the sequences."""
        dataset_name, split, seq_cols, prefix = self._BENCHMARK_TEST_SETS[test_set]
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {dataset_name}[{split}] for decontamination"
            ) from e

        present = [c for c in seq_cols if c in ds.column_names]
        if not present:
            raise RuntimeError(
                f"{dataset_name}[{split}] is missing supported sequence columns; "
                f"available columns: {ds.column_names}"
            )
        logger.info("Using %s sequence columns: %s", dataset_name, ", ".join(present))

        sequences: set[str] = set()
        for row in ds:
            for col in present:
                value = str(row.get(col, "") or "").strip()
                if value:
                    sequences.add(value)

        if not sequences:
            raise RuntimeError(f"{dataset_name}[{split}] has no non-empty sequences")

        ordered = sorted(sequences)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for index, sequence in enumerate(ordered, start=1):
                f.write(f">{prefix}_{index:08d}\n{sequence}\n")
        logger.info("%s test set: %d unique sequences -> %s", test_set, len(ordered), out_path)
        return ordered

    def _mmseqs_leaked_query_ids(
        self,
        query_fasta: Path,
        target_fasta: Path,
        hits_tsv: Path,
        tmp_dir: Path,
        *,
        min_seq_id: float,
        cov: float = 0.8,
        cov_mode: int = 1,
        threads: int = 48,
        gpu: bool = False,
        timeout: int = 86_400,
    ) -> set[str]:
        """Search query_fasta against target_fasta; return query ids that hit.

        The corpus is the QUERY and the test set is the TARGET so that each corpus
        sequence only needs *any* hit — the default --max-seqs prefilter cap would
        otherwise silently truncate hits in the reverse orientation. cov-mode 1 is
        coverage of the target (the test sequence), so a long corpus protein that
        merely contains a test-length domain is still caught.

        easy-search (not easy-linclust) is used deliberately: linclust's k-mer
        prefilter loses sensitivity below ~50% identity, and decontamination needs
        recall, not speed.
        """
        mmseqs_bin = self._resolve_mmseqs_binary()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        hits_tsv.unlink(missing_ok=True)

        cmd = [
            mmseqs_bin, "easy-search",
            str(query_fasta), str(target_fasta), str(hits_tsv), str(tmp_dir),
            "--min-seq-id", str(min_seq_id),
            "--cov-mode", str(cov_mode),
            "-c", str(cov),
            "--alignment-mode", "3",
            "-e", "1e-3",
            "--threads", str(threads),
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue",
            "--remove-tmp-files", "-v", "3",
        ]
        cmd += ["--gpu", "1"] if gpu else ["-s", "7.5"]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"mmseqs easy-search failed ({' '.join(cmd)}):\n{e.stderr[-4000:]}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"mmseqs easy-search timed out after {timeout}s ({' '.join(cmd)})"
            ) from e
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        leaked: set[str] = set()
        with open(hits_tsv) as f:
            for line in f:
                if line.strip():
                    leaked.add(line.split("\t", 1)[0])
        return leaked

    def _decontaminate_dataframe(
        self,
        df: pl.DataFrame,
        seq_col: str,
        test_set: str,
        *,
        min_seq_id: float,
        cov: float = 0.8,
        threads: int = 48,
        gpu: bool = False,
        label: str = "corpus",
    ) -> pl.DataFrame:
        """Drop rows whose sequence aligns to a benchmark test sequence.

        Test-set leakage removal only — this is independent of, and applied after,
        each corpus's own internal redundancy reduction.
        """
        work_dir = Path(self.data_dir) / "decontam" / label
        work_dir.mkdir(parents=True, exist_ok=True)
        test_fasta = work_dir / f"{test_set}_test.fasta"
        self._build_benchmark_test_fasta(test_set, test_fasta)

        unique_seqs = df.select(pl.col(seq_col).alias("sequence")).drop_nulls().unique()
        query_fasta = work_dir / "query.fasta"
        with open(query_fasta, "wb") as f:
            for i, seq in enumerate(unique_seqs["sequence"].to_list()):
                f.write(b">%d\n%s\n" % (i, seq.encode()))

        logger.info(
            "Decontaminating %s: %d unique sequences vs %s test set at %.0f%% id / %.0f%% cov",
            label, unique_seqs.height, test_set, 100 * min_seq_id, 100 * cov,
        )
        leaked_ids = self._mmseqs_leaked_query_ids(
            query_fasta,
            test_fasta,
            work_dir / "hits.tsv",
            work_dir / "mmseqs_tmp",
            min_seq_id=min_seq_id,
            cov=cov,
            threads=threads,
            gpu=gpu,
        )

        if not leaked_ids:
            logger.info("%s: no test-set leakage found", label)
            return df

        leaked_seqs = unique_seqs["sequence"].gather(sorted(int(i) for i in leaked_ids))
        before = df.height
        df = df.filter(~pl.col(seq_col).is_in(leaked_seqs))
        logger.info(
            "%s decontamination: removed %d/%d rows (%.4f%%) matching %d leaked sequences",
            label, before - df.height, before,
            100.0 * (before - df.height) / max(1, before), leaked_seqs.len(),
        )
        if df.height == 0:
            raise RuntimeError(f"{label} decontamination removed every row")
        return df

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            "seq": "sequence",
            "sequence_aa": "sequence",
            "aa_seq": "sequence",
            "ur50_id": "cluster_id",
            "uniref50_id": "cluster_id",
            "repId": "cluster_id",
            "RepId": "cluster_id",
            "family": "family_id",
            "clan": "clan_id",
        }
        rename = {
            src: dst
            for src, dst in rename_map.items()
            if src in df.columns and dst not in df.columns
        }
        if rename:
            df = df.rename(rename)
        return df

    def _validate_df(
        self,
        df: pl.DataFrame,
        required_cols: Iterable[str],
        output_name: str,
        group_col: Optional[str] = None,
    ) -> pl.DataFrame:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {output_name}: {', '.join(missing)}"
            )

        before = len(df)
        df = df.filter(
            pl.col("sequence").is_not_null() & (pl.col("sequence").str.len_bytes() > 0)
        )
        if group_col:
            df = df.filter(pl.col(group_col).is_not_null())

        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "Filtered %d invalid rows for %s (null/empty sequence or group)",
                dropped,
                output_name,
            )

        return df

    def _sort_and_save(
        self,
        df: pl.DataFrame,
        output_name: str,
        hierarchical: bool,
        sort_cols: Optional[str | list[str]] = None,
        shuffle_before_sort: bool = False,
        drop_redundant_cluster_id: bool = False,
    ):
        path = os.path.join(self.data_dir, output_name)
        logger.info("Sorting %d sequences -> %s", len(df), output_name)

        df = self._normalize_columns(df)

        # Validation and cleaning
        if hierarchical:
            df = self._validate_df(
                df,
                required_cols=["sequence", "clan_id", "family_id"],
                output_name=output_name,
                group_col="family_id",
            )
        else:
            df = self._validate_df(
                df,
                required_cols=["sequence", "cluster_id"],
                output_name=output_name,
                group_col="cluster_id",
            )

        if len(df) == 0:
            logger.warning("%s is empty after validation.", output_name)
            return

        # Filter to clusters with >1 member (for the finest level)
        finest_col = "cluster_id" if not hierarchical else "family_id"
        counts = df.group_by(finest_col).len()
        valid_groups = counts.filter(pl.col("len") > 1).select(finest_col)
        df = df.join(valid_groups, on=finest_col, how="semi")

        if len(df) == 0:
            logger.warning("%s has no groups with >1 member.", output_name)
            return

        # Add canonical group_id column (= finest grouping level)
        if hierarchical:
            df = df.with_columns(pl.col("family_id").alias("group_id"))
        else:
            df = df.with_columns(pl.col("cluster_id").alias("group_id"))

        if drop_redundant_cluster_id and "cluster_id" in df.columns:
            mismatch_count = df.filter(
                pl.col("cluster_id") != pl.col("group_id")
            ).height
            if mismatch_count == 0:
                df = df.drop("cluster_id")
                logger.info(
                    "Dropped redundant cluster_id column; group_id is canonical."
                )
            else:
                logger.warning(
                    "Keeping cluster_id due to %d mismatched rows against group_id.",
                    mismatch_count,
                )

        if sort_cols is None:
            resolved_sort_cols = (
                ["clan_id", "family_id"] if hierarchical else ["cluster_id"]
            )
        elif isinstance(sort_cols, str):
            resolved_sort_cols = [sort_cols]
        else:
            resolved_sort_cols = list(sort_cols)

        # Hierarchical sort: branch (clan/fold) -> leaf (family/cluster)
        if hierarchical:
            logger.info("Hierarchical sort: %s", " -> ".join(resolved_sort_cols))
        else:
            logger.info("Flat sort: %s", " -> ".join(resolved_sort_cols))

        if shuffle_before_sort:
            df = (
                df.with_row_index("__shuffle_idx")
                .sample(fraction=1.0, shuffle=True, seed=42)
                .sort([*resolved_sort_cols, "__shuffle_idx"])
                .drop("__shuffle_idx")
            )
        else:
            df = df.sort(resolved_sort_cols)

        df = df.rechunk()
        df.write_parquet(path)
        logger.info("Saved: %s (SORTED - do not shuffle during training)", path)

    def prep_nvidia(self, limit_gb: int = 5):
        logger.info("ETL: Nvidia UniRef | Limit: %s GB", limit_gb)
        n_shards = max(1, int(limit_gb * 5))
        dfs = []
        for i in range(n_shards):
            try:
                p = hf_hub_download(
                    repo_id="nvidia/esm2_uniref_pretraining_data",
                    filename=f"train/{i:03d}.parquet",
                    repo_type="dataset",
                    local_dir=self.data_dir,
                )
                dfs.append(
                    pl.read_parquet(p).select(
                        [pl.col("sequence"), pl.col("ur50_id").alias("cluster_id")]
                    )
                )
            except Exception as e:
                logger.warning(
                    "Failed to load shard %d: %s. Using partial dataset from %d shards.",
                    i,
                    e,
                    len(dfs),
                )
                break
        if dfs:
            self._sort_and_save(pl.concat(dfs), "nvidia_sorted.parquet", False)

    def prep_afdb(
        self,
        limit_gb: int = 25,
        decontaminate: bool = True,
        decontam_min_seq_id: float = 0.4,
        threads: int = 48,
        mmseqs_gpu: bool = False,
    ) -> None:
        """Download AFDB sequences and label them with Foldseek structural cluster IDs.

        Two-step ETL:
        1. Download the Steinegger Lab structural cluster mapping
                     (1-AFDBClusters-repId_entryId_cluFlag_taxId.tsv.gz, cluFlag in {1,2}).
        2. Lazy-scan sequences from ``willdaspit/afdb_clustered_seqs`` on HuggingFace.
                    3. Inner-join on HF.repId (AFDB50 representative) == Steinegger.entry_id to
              assign each sequence its Foldseek structural cluster representative as
              ``cluster_id`` while also preserving the AFDB50 representative as
              ``afdb50_cluster_id``.

        The resulting parquet groups proteins by 3D structural similarity (2.3M
        Foldseek clusters) rather than sequence identity (52M AFDB50 clusters),
        producing structurally diverse but related positives for contrastive training.

        Args:
            limit_gb: Approximate output size cap in GB.
                Controls max_rows ceiling (default 25 => ~50M sequences).
                Set <= 0 for an uncapped full join.
            decontaminate: Drop sequences similar to the remote-homology benchmark
                test split (biomap-research/fold_prediction[test]).
            decontam_min_seq_id: Identity cutoff for that filter (default 0.40,
                at 80% coverage of the test sequence).
            threads: Threads for the MMseqs2 decontamination search.
            mmseqs_gpu: Use the MMseqs2 GPU prefilter for that search.
        """
        logger.info(
            "ETL: AFDB Structural Clusters (Foldseek) | Target: ~%s GB", limit_gb
        )

        tsv_filename = "1-AFDBClusters-repId_entryId_cluFlag_taxId.tsv.gz"
        tsv_path = Path(self.data_dir) / tsv_filename
        _download_file(
            "https://afdb-cluster.steineggerlab.workers.dev/v6/" + tsv_filename,
            tsv_path,
        )

        logger.info(
            "Loading Foldseek structural mappings (cluFlag in %s)...",
            _AFDB_ALLOWED_CLU_FLAGS,
        )
        foldseek_map_lf = _load_afdb_foldseek_map_lazy(
            tsv_path, allowed_clu_flags=_AFDB_ALLOWED_CLU_FLAGS
        )
        map_stats = cast(
            pl.DataFrame,
            foldseek_map_lf.select(
                pl.len().alias("n_rows"),
                pl.col("foldseek_rep").n_unique().alias("n_clusters"),
            ).collect(),
        )
        map_flag_counts = cast(
            pl.DataFrame,
            foldseek_map_lf.group_by("clu_flag").len().sort("clu_flag").collect(),
        )
        n_map_rows = cast(int, map_stats["n_rows"][0])
        n_map_clusters = cast(int, map_stats["n_clusters"][0])
        logger.info(
            "Structural mapping: %d entries across %d Foldseek clusters",
            n_map_rows,
            n_map_clusters,
        )
        logger.info(
            "Structural mapping cluFlag distribution: %s",
            map_flag_counts.to_dicts(),
        )

        hf_glob = "hf://datasets/willdaspit/afdb_clustered_seqs/**/*.parquet"

        schema_cols = set(
            cast(pl.DataFrame, pl.scan_parquet(hf_glob).head(1).collect()).columns
        )
        required_cols = {"sequence", "repId", "plddt", "fragment"}
        if missing := required_cols - schema_cols:
            raise ValueError(
                f"HF dataset missing expected columns: {missing}. Found: {sorted(schema_cols)}"
            )
        logger.info("HF schema validated. Columns: %s", sorted(schema_cols))

        sample_ids = (
            pl.scan_parquet(hf_glob)
            .filter((pl.col("plddt") > 70) & (pl.col("fragment") == 0))
            .select("repId")
            .head(10_000)
            .collect()
        )
        sample_ids_df = cast(pl.DataFrame, sample_ids)
        matched_sample_df = cast(
            pl.DataFrame,
            sample_ids_df.lazy()
            .join(
                foldseek_map_lf.select(pl.col("entry_id").alias("repId")).unique(),
                on="repId",
                how="inner",
            )
            .select(pl.len().alias("n"))
            .collect(),
        )
        matched_sample = cast(int, matched_sample_df["n"][0])
        match_rate = matched_sample / max(1, len(sample_ids_df))
        logger.info("Mini-join match rate: %.1f%% on 10k sample", match_rate * 100)
        if match_rate < 0.5:
            logger.warning(
                "Low match rate (%.1f%%). Verify HF.repId format matches Steinegger entryId.",
                match_rate * 100,
            )

        joined_lf = (
            pl.scan_parquet(hf_glob)
            .filter((pl.col("plddt") > 70) & (pl.col("fragment") == 0))
            .select(["sequence", "repId"])
            .join(
                foldseek_map_lf.select(
                    [pl.col("foldseek_rep"), pl.col("entry_id").alias("repId")]
                ),
                on="repId",
                how="inner",
            )
            .select(
                [
                    pl.col("sequence"),
                    pl.col("foldseek_rep").alias("cluster_id"),
                    pl.col("repId").alias("afdb50_cluster_id"),
                ]
            )
        )

        if limit_gb > 0:
            max_rows = limit_gb * 2_000_000
            logger.info("Running full join (max_rows=%d)...", max_rows)
            joined_lf = joined_lf.head(max_rows)
        else:
            logger.info("Running full join without row cap (limit_gb <= 0)")

        df = joined_lf.collect()

        df_result = cast(pl.DataFrame, df)
        if len(df_result) == 0:
            logger.error(
                "Join produced 0 rows. Check that HF.repId accession format "
                "matches Steinegger entryId column in %s.",
                tsv_filename,
            )
            return

        logger.info(
            "Collected %d sequences across %d structural clusters",
            len(df_result),
            df_result["cluster_id"].n_unique(),
        )
        n_struct_clusters = df_result["cluster_id"].n_unique()
        n_afdb50_reps = df_result["afdb50_cluster_id"].n_unique()
        logger.info(
            "Label granularity: %d structural targets vs %d AFDB50 representatives",
            n_struct_clusters,
            n_afdb50_reps,
        )
        if n_struct_clusters >= n_afdb50_reps:
            logger.warning(
                "Structural target count is not coarser than AFDB50 representatives. "
                "Validate mapping semantics."
            )

        # Remove anything similar to the remote-homology benchmark test set.
        # AFDB has no internal identity clustering of its own — it inherits AFDB50
        # cluster ids (<50% identity between representatives, stricter than the 70%
        # used for Pfam), and within-cluster members are deliberately kept because
        # they are the positive pairs for contrastive training.
        if decontaminate:
            df_result = self._decontaminate_dataframe(
                df_result,
                "sequence",
                "fold",
                min_seq_id=decontam_min_seq_id,
                threads=threads,
                gpu=mmseqs_gpu,
                label="afdb",
            )

        self._sort_and_save(
            df_result,
            "afdb_sorted.parquet",
            hierarchical=False,
            sort_cols="afdb50_cluster_id",
            shuffle_before_sort=True,
            drop_redundant_cluster_id=True,
        )

    def prep_pfam_full(
        self,
        min_seq_id: float = 0.7,
        threads: int = 40,
        fast: bool = False,
        decontaminate: bool = True,
        decontam_min_seq_id: float = 0.4,
        mmseqs_gpu: bool = False,
    ):
        """
        ETL Pipeline for Pfam-A Full (Large Scale with MMseqs2).

        Process:
        1. Downloads Pfam-A.fasta.gz (50M+ seqs) + Pfam-A.clans.tsv.gz
        2. Internal redundancy reduction: MMseqs2 linclust at ``min_seq_id``
           identity / 80% coverage, keeping representatives only
        3. Falls back to full dataset if MMseqs2 unavailable
        4. Maps Family -> Clan hierarchy
        5. Removes sequences similar to the remote-homology benchmark test split
        6. Sorts by Clan -> Family for streaming dataloader compatibility

        Args:
            min_seq_id: MMseqs2 *internal* redundancy identity threshold (default 0.70)
            threads: Number of threads for MMseqs2 (default 40)
            fast: If True, limit to 50k sequences for testing
            decontaminate: Drop sequences similar to the remote-homology benchmark
                test split (biomap-research/fold_prediction[test]).
            decontam_min_seq_id: Identity cutoff for that filter (default 0.40,
                at 80% coverage of the test sequence).
            mmseqs_gpu: Use the MMseqs2 GPU prefilter for that search.
        """

        logger.info("ETL: Pfam-A Full (with MMseqs2 de-duplication)")

        # --- Config & URLs ---
        pfam_base = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release"
        files = {
            "fasta": ("Pfam-A.fasta.gz", f"{pfam_base}/Pfam-A.fasta.gz"),
            "clans": ("Pfam-A.clans.tsv.gz", f"{pfam_base}/Pfam-A.clans.tsv.gz"),
        }

        # --- Paths ---
        data_path = Path(self.data_dir)
        raw_fasta = data_path / files["fasta"][0]
        clans_tsv = data_path / files["clans"][0]
        mmseqs_prefix = data_path / "pfam_clustered"
        mmseqs_tmp = data_path / "mmseqs_tmp"
        mmseqs_out_fasta = Path(f"{mmseqs_prefix}_rep_seq.fasta")

        # --- 1. Download ---
        for fname, url in files.values():
            _download_file(url, data_path / fname)

        # --- 2. MMseqs2 Clustering (Optional but Recommended) ---
        use_mmseqs = False
        input_fasta = raw_fasta

        # Skip MMseqs2 in fast mode (does not make sense to cluster 50M seqs to extract 50k)
        if fast:
            logger.info("Fast mode: skipping MMseqs2 clustering, using raw FASTA")
            use_mmseqs = False
        else:
            # Check if MMseqs2 is available
            try:
                result = subprocess.run(
                    ["mmseqs", "version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    use_mmseqs = True
                    logger.info("MMseqs2 found: %s", result.stdout.strip())
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning(
                    "MMseqs2 not found in PATH. Using full dataset (will be slower)."
                )
                logger.warning("Install: conda install -c bioconda mmseqs2")

        if use_mmseqs and not mmseqs_out_fasta.exists():
            logger.info(
                "Running MMseqs2 clustering (identity: %s, threads: %s)...",
                min_seq_id,
                threads,
            )
            logger.info("This will take 10-30 minutes for 50M sequences...")

            mmseqs_tmp.mkdir(exist_ok=True)

            cmd = [
                "mmseqs",
                "easy-linclust",
                str(raw_fasta),
                str(mmseqs_prefix),
                str(mmseqs_tmp),
                "--min-seq-id",
                str(min_seq_id),
                "--cov-mode",
                "1",  # Target coverage
                "-c",
                "0.8",  # 80% alignment coverage
                "--threads",
                str(threads),
                "-v",
                "3",  # Show progress info
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("MMseqs2 clustering complete")

                # Cleanup tmp directory
                shutil.rmtree(mmseqs_tmp, ignore_errors=True)
                logger.info("Cleaned up temporary files")

                input_fasta = mmseqs_out_fasta

            except subprocess.CalledProcessError as e:
                logger.error("MMseqs2 failed: %s", e.stderr)
                logger.warning("Falling back to full dataset")
                use_mmseqs = False
        elif mmseqs_out_fasta.exists():
            logger.info("Using existing clustered sequences")
            input_fasta = mmseqs_out_fasta
            use_mmseqs = True

        # --- 3. Parse FASTA Headers ---
        logger.info("Parsing sequences from: %s", input_fasta.name)

        seqs = []
        fam_ids = []
        domain_ids = []  # Store full ID including domain boundaries

        current_seq_parts = []
        current_fam = None
        current_domain_id = None

        # Open with gzip if needed
        open_func = gzip.open if str(input_fasta).endswith(".gz") else open

        with open_func(input_fasta, "rt", encoding="latin-1", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if line.startswith(">"):
                    # Save previous sequence
                    if current_fam and current_seq_parts:
                        seqs.append("".join(current_seq_parts))
                        fam_ids.append(current_fam)
                        domain_ids.append(current_domain_id)
                        current_seq_parts = []

                    # Parse header: ">A0A067SRH6_GALM3/383-505 A0A067SRH6.1 PF26733.1;03009_C;"
                    # Format: [0]=domain_id [1]=uniprot.version [2]=PFxxxxx.version;family_name;
                    try:
                        # Remove leading ">"
                        header = line[1:]
                        parts = header.split()

                        if len(parts) >= 3:
                            # First part is the domain ID (e.g., "A0A067SRH6_GALM3/383-505")
                            current_domain_id = parts[0]

                            # Third part contains PF ID (e.g., "PF26733.1;03009_C;")
                            pf_part = parts[2].split(";")[0]  # Remove family name
                            current_fam = pf_part.split(".")[0]  # Remove version
                        elif len(parts) >= 2:
                            # Fallback: try parts[1] in case header format varies
                            current_domain_id = parts[0]
                            pf_part = parts[1].split(";")[0]
                            current_fam = pf_part.split(".")[0]
                        else:
                            # Malformed header, skip
                            current_fam = None
                            current_domain_id = None
                    except Exception:
                        # Parsing error, skip this sequence
                        current_fam = None
                        current_domain_id = None
                else:
                    # Sequence line
                    current_seq_parts.append(line)

                # Fast mode: limit sequences
                if fast and len(seqs) >= 50000:
                    break

            # Flush last sequence
            if current_fam and current_seq_parts:
                seqs.append("".join(current_seq_parts))
                fam_ids.append(current_fam)
                domain_ids.append(current_domain_id)

        logger.info("Loaded %d domain sequences", len(seqs))

        if len(seqs) == 0:
            logger.error("No sequences parsed. Check FASTA format.")
            return

        # --- 4. Create DataFrame ---
        df = pl.DataFrame(
            {"sequence": seqs, "family_id": fam_ids, "domain_id": domain_ids}
        )

        # Free memory
        del seqs, fam_ids, domain_ids

        logger.info("Unique families: %d", df["family_id"].n_unique())

        # --- 5. Load and Join Clan Hierarchy ---
        logger.info("Mapping families to clans...")

        # Read clans file: family_id, clan_id, clan_name, fam_name, fam_desc
        try:
            clans_df = pl.read_csv(
                clans_tsv,
                separator="\t",
                has_header=False,
                new_columns=[
                    "family_id",
                    "clan_id",
                    "clan_name",
                    "fam_name",
                    "fam_desc",
                ],
            ).select(["family_id", "clan_id"])

            logger.info("Loaded %d family->clan mappings", len(clans_df))

            # Join
            df = df.join(clans_df, on="family_id", how="left")

            # Handle orphans: families without clans become their own "clan"
            df = df.with_columns(
                [pl.col("clan_id").fill_null(pl.col("family_id")).alias("clan_id")]
            )

            orphans = df.filter(pl.col("clan_id") == pl.col("family_id")).height
            logger.info(
                "Orphan families (no clan): %d (%.1f%%)",
                orphans,
                100 * orphans / len(df),
            )

        except Exception as e:
            logger.error("Failed to load clans file: %s", e)
            logger.warning("Using family_id as clan_id (no hierarchy)")
            df = df.with_columns([pl.col("family_id").alias("clan_id")])

        # --- 6. Validation Stats ---
        logger.info("Dataset Statistics:")
        logger.info("Total sequences: %d", len(df))
        logger.info("Unique clans: %d", df["clan_id"].n_unique())
        logger.info("Unique families: %d", df["family_id"].n_unique())

        # Check multi-family clans
        fam_per_clan = df.group_by("clan_id").agg(
            pl.col("family_id").n_unique().alias("n_fams")
        )
        multi_fam_clans = fam_per_clan.filter(pl.col("n_fams") > 1).height
        logger.info(
            "Multi-family clans: %d / %d (%.1f%%)",
            multi_fam_clans,
            fam_per_clan.height,
            100 * multi_fam_clans / fam_per_clan.height,
        )
        logger.info("Avg families/clan: %.2f", fam_per_clan["n_fams"].mean())

        # --- 7. Sort & Save ---
        logger.info("Sorting by Clan -> Family and filtering singleton families...")
        before_save = len(df)
        if decontaminate:
            df = self._decontaminate_dataframe(
                df,
                "sequence",
                "fold",
                min_seq_id=decontam_min_seq_id,
                threads=threads,
                gpu=mmseqs_gpu,
                label="pfam",
            )

        self._sort_and_save(df, "pfam_sorted.parquet", True)
        logger.info("Final size before singleton filter: %d sequences", before_save)

        if use_mmseqs:
            reduction = 100 * (1 - before_save / 50_000_000)
            logger.info("Size reduction: ~%.0f%% (MMseqs2 clustering)", reduction)

    def prep_stringdb(
        self,
        min_seq_id: float = 0.5,
        threads: int = 48,
        min_combined_score: int = 400,
        max_rows: int = 0,
        min_seq_len: int = 10,
        max_seq_len: int = 1024,
        cleanup_mode: str = "aggressive",
        decontam_min_seq_id: float = 0.4,
        mmseqs_gpu: bool = False,
    ):
        """
        ETL Pipeline for STRING-DB Protein-Protein Interactions.

        Process:
        1. Downloads STRING-DB v12.0 sequences and physical links
        2. Pre-filters proteins to those appearing in high-score links
        3. Bernett test decontamination before main clustering
        4. Two-stage MMseqs2 clustering: linclust@65% → cascaded cluster@min_seq_id
        5. Streaming ETL to deduplicate by cluster-pair and produce
           a training Parquet with pre-built (seq1, seq2) pairs

        The output Parquet has columns:
            seq1, seq2

        It is consumed directly by protein_pipeline.py as a PPI pair dataset.

        Args:
            min_seq_id: MMseqs2 cascaded-cluster sequence identity threshold (default 0.50).
                This is the *internal* redundancy reduction cutoff, not the test-set one.
            threads: Number of threads for MMseqs2 (default 48)
            decontam_min_seq_id: Identity cutoff for removing sequences similar to the
                Bernett test split (default 0.40, at 80% coverage of the test sequence)
            mmseqs_gpu: Use the MMseqs2 GPU prefilter for the decontamination search
            min_combined_score: Minimum combined_score filter for links
                (default 400 = STRING-DB medium confidence)
            max_rows: Maximum interaction pairs to keep (0 = no limit)
            min_seq_len: Minimum sequence length to keep (default 10)
            max_seq_len: Maximum sequence length to keep (default 1024)
            cleanup_mode: Cleanup policy for intermediate files:
                - "keep": keep reusable artifacts (filtered_proteins.fa, clu45.tsv)
                - "aggressive": keep only gz downloads + output parquet
        """
        logger.info("ETL: STRING-DB PPI (v12.0)")

        stringdb_dir = Path(self.data_dir) / "stringdb"
        stringdb_dir.mkdir(parents=True, exist_ok=True)

        # --- URLs ---
        base_url = "https://stringdb-downloads.org/download"
        files_to_download = {
            "sequences": (
                "protein.sequences.v12.0.fa.gz",
                f"{base_url}/protein.sequences.v12.0.fa.gz",
            ),
            "links": (
                "protein.physical.links.full.v12.0.txt.gz",
                f"{base_url}/protein.physical.links.full.v12.0.txt.gz",
            ),
        }

        # --- 1. Download ---
        for fname, url in files_to_download.values():
            dest = stringdb_dir / fname
            _download_file(url, dest)

        # --- 2. Decompress FASTA ---
        fasta_gz = stringdb_dir / "protein.sequences.v12.0.fa.gz"
        fasta_plain = stringdb_dir / "protein.sequences.v12.0.fa"
        if not fasta_plain.exists() and fasta_gz.exists():
            logger.info("Decompressing %s ...", fasta_gz.name)
            subprocess.run(
                ["gunzip", "-k", str(fasta_gz)],
                check=True,
                capture_output=True,
            )
            logger.info("Decompressed to %s", fasta_plain.name)

        # --- 2.5. Pre-filter: keep only proteins in high-score links ---
        # Scanning links first keeps only the ~20-30M proteins that actually
        # participate in interactions above the score threshold, instead of
        # all 59M sequences. This roughly halves clustering time.
        filtered_fasta = stringdb_dir / "filtered_proteins.fa"
        decontaminated_fasta = stringdb_dir / "decontaminated_proteins.fa"
        bernett_test_fasta = stringdb_dir / "bernett_test.fasta"
        decontam_hits_tsv = stringdb_dir / "decontam_hits.tsv"
        decontam_hash_file = stringdb_dir / ".decontam_bernett_sha256"
        decontam_signature_file = stringdb_dir / ".decontam_input_signature"
        links_gz = stringdb_dir / "protein.physical.links.full.v12.0.txt.gz"
        clu_tsv = stringdb_dir / "clu45.tsv"
        threshold_file = stringdb_dir / ".filtered_score_threshold"

        def _remove_decontam_outputs(remove_sidecars: bool = False) -> None:
            for file_path in [
                decontaminated_fasta,
                decontam_hits_tsv,
                bernett_test_fasta,
            ]:
                file_path.unlink(missing_ok=True)
            if remove_sidecars:
                decontam_hash_file.unlink(missing_ok=True)
                decontam_signature_file.unlink(missing_ok=True)

        # Invalidate stale filtered FASTA if score threshold changed
        if filtered_fasta.exists() and min_combined_score > 0:
            if threshold_file.exists():
                prev_threshold = int(threshold_file.read_text().strip())
                if prev_threshold != min_combined_score:
                    logger.info(
                        "Score threshold changed (%d → %d), "
                        "regenerating filtered FASTA ...",
                        prev_threshold,
                        min_combined_score,
                    )
                    filtered_fasta.unlink()
                    _remove_decontam_outputs(remove_sidecars=True)
                    if clu_tsv.exists():
                        logger.info(
                            "Also removing stale cluster file: %s", clu_tsv.name
                        )
                        clu_tsv.unlink()

        if not filtered_fasta.exists() and min_combined_score > 0:
            logger.info("Pre-filtering proteins by score >= %d ...", min_combined_score)
            wanted: set[str] = set()
            n_scanned = 0
            with gzip.open(links_gz, "rt") as f:
                next(f)  # skip header
                for line in f:
                    n_scanned += 1
                    if n_scanned % 100_000_000 == 0:
                        logger.info(
                            "  Scanned %dM links, %d unique proteins so far",
                            n_scanned // 1_000_000,
                            len(wanted),
                        )
                    fields = line.split()
                    if int(fields[-1]) >= min_combined_score:
                        wanted.add(fields[0])
                        wanted.add(fields[1])

            logger.info(
                "Found %d proteins in %dM links with score >= %d",
                len(wanted),
                n_scanned // 1_000_000,
                min_combined_score,
            )

            # Write filtered FASTA (only proteins in high-score interactions)
            logger.info("Writing filtered FASTA ...")
            n_written = 0
            with open(fasta_plain, "r") as fin, open(filtered_fasta, "w") as fout:
                keep = False
                for line_fa in fin:
                    if line_fa[0] == ">":
                        pid = line_fa[1:].split(None, 1)[0]
                        keep = pid in wanted
                        if keep:
                            n_written += 1
                    if keep:
                        fout.write(line_fa)
            del wanted
            logger.info("Filtered FASTA: %d proteins written", n_written)
            # Record the threshold used for staleness detection
            threshold_file.write_text(str(min_combined_score))
        elif filtered_fasta.exists():
            logger.info("Using existing filtered FASTA: %s", filtered_fasta.name)

        # --- 2.75. Bernett test decontamination ---
        decontam_source_fasta = (
            filtered_fasta if filtered_fasta.exists() else fasta_plain
        )

        logger.info("Preparing Bernett test decontamination set ...")
        sorted_bernett_sequences = self._build_benchmark_test_fasta(
            "bernett", bernett_test_fasta
        )

        bernett_sha256 = hashlib.sha256(
            "\n".join(sorted_bernett_sequences).encode("utf-8")
        ).hexdigest()

        source_stat = decontam_source_fasta.stat()
        # The cutoff is part of the signature: without it, lowering the threshold
        # would be a silent no-op against an existing decontaminated FASTA.
        input_signature = (
            f"{decontam_source_fasta.name}:{source_stat.st_size}:{source_stat.st_mtime_ns}"
            f":minid={decontam_min_seq_id}:cov=0.8:covmode=1:method=easy-search"
        )
        previous_hash = (
            decontam_hash_file.read_text().strip()
            if decontam_hash_file.exists()
            else ""
        )
        previous_signature = (
            decontam_signature_file.read_text().strip()
            if decontam_signature_file.exists()
            else ""
        )
        regenerate_decontam = (
            (not decontaminated_fasta.exists())
            or previous_hash != bernett_sha256
            or previous_signature != input_signature
        )

        if regenerate_decontam:
            logger.info("Regenerating Bernett decontaminated FASTA ...")
            clu_tsv.unlink(missing_ok=True)

            # Search the STRING proteins (query) against the Bernett test split
            # (target). easy-search rather than easy-linclust: linclust's k-mer
            # prefilter loses sensitivity below ~50% identity, and this needs recall.
            removed_ids = self._mmseqs_leaked_query_ids(
                decontam_source_fasta,
                bernett_test_fasta,
                decontam_hits_tsv,
                stringdb_dir / "mmseqs_tmp",
                min_seq_id=decontam_min_seq_id,
                cov=0.8,
                cov_mode=1,
                threads=threads,
                gpu=mmseqs_gpu,
            )

            total_proteins = 0
            kept_proteins = 0
            with (
                open(decontam_source_fasta, "r") as fin,
                open(decontaminated_fasta, "w") as fout,
            ):
                keep_record = False
                for line_fa in fin:
                    if line_fa.startswith(">"):
                        total_proteins += 1
                        protein_id = line_fa[1:].split(None, 1)[0]
                        keep_record = protein_id not in removed_ids
                        if keep_record:
                            kept_proteins += 1
                    if keep_record:
                        fout.write(line_fa)

            removed_proteins = total_proteins - kept_proteins
            removed_pct = (
                (100.0 * removed_proteins / total_proteins) if total_proteins else 0.0
            )
            logger.info(
                "Bernett decontamination (>=%.0f%% id, >=80%% coverage of the test sequence): "
                "Bernett sequences=%d, kept proteins=%d, removed=%d/%d proteins (%.3f%%)",
                100 * decontam_min_seq_id,
                len(sorted_bernett_sequences),
                kept_proteins,
                removed_proteins,
                total_proteins,
                removed_pct,
            )

            if kept_proteins == 0:
                raise RuntimeError(
                    "Bernett decontamination produced an empty decontaminated FASTA"
                )

            decontam_hash_file.write_text(bernett_sha256)
            decontam_signature_file.write_text(input_signature)
        else:
            logger.info(
                "Using existing decontaminated FASTA: %s", decontaminated_fasta.name
            )

        # --- 3. Two-stage MMseqs2 clustering ---
        # Stage 1: linclust at 65% identity (fast, linear-time near-duplicate removal)
        # Stage 2: cascaded cluster at target identity on the reduced representative set
        # This is much faster than running cascaded cluster on all proteins directly.
        # Reference: https://github.com/VarunUllanat/mint

        def _cleanup_stringdb_intermediates(
            remove_final_artifacts: bool = False,
        ) -> None:
            shutil.rmtree(stringdb_dir / "mmseqs_tmp", ignore_errors=True)

            for file_path in [
                stringdb_dir / "linclust65_all_seqs.fasta",
                stringdb_dir / "linclust65_rep_seq.fasta",
                stringdb_dir / "linclust65_cluster.tsv",
                decontam_hits_tsv,
                bernett_test_fasta,
                stringdb_dir / "clu45_reps.tsv",
                stringdb_dir / "clu45",
                stringdb_dir / "clu45.dbtype",
                stringdb_dir / "clu45.index",
            ]:
                file_path.unlink(missing_ok=True)

            # Full decompressed FASTA is redundant once filtered FASTA exists
            if filtered_fasta.exists():
                (stringdb_dir / "protein.sequences.v12.0.fa").unlink(missing_ok=True)

            for pattern in [
                "repDB*",
                "seqDB*",
                "linclust65*",
                "clu45.*",
            ]:
                for file_path in stringdb_dir.glob(pattern):
                    if not file_path.is_file():
                        continue
                    if file_path.name == "clu45.tsv":
                        continue
                    file_path.unlink(missing_ok=True)

            if cleanup_mode == "aggressive" and remove_final_artifacts:
                for file_path in [
                    filtered_fasta,
                    decontaminated_fasta,
                    clu_tsv,
                    stringdb_dir / ".filtered_score_threshold",
                    decontam_hash_file,
                    decontam_signature_file,
                    stringdb_dir / "stringdb_prep_run.log",
                ]:
                    file_path.unlink(missing_ok=True)

        if not clu_tsv.exists():
            mmseqs_bin = self._resolve_mmseqs_binary()

            tmp_dir = stringdb_dir / "mmseqs_tmp"
            if tmp_dir.exists():
                logger.info("Removing old tmp directory: %s", tmp_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Input preference: decontaminated -> filtered -> full
            cluster_input = (
                decontaminated_fasta
                if decontaminated_fasta.exists()
                else filtered_fasta
                if filtered_fasta.exists()
                else fasta_plain
            )
            logger.info("Clustering input: %s", cluster_input.name)

            try:
                # --- Stage 1: linclust at 65% identity (fast dedup) ---
                linclust_prefix = stringdb_dir / "linclust65"
                linclust_reps = stringdb_dir / "linclust65_rep_seq.fasta"
                linclust_cluster_tsv = stringdb_dir / "linclust65_cluster.tsv"

                if not linclust_reps.exists():
                    logger.info(
                        "Stage 1/2: linclust at 65%% identity "
                        "(fast dedup, threads=%d)...",
                        threads,
                    )
                    subprocess.run(
                        [
                            mmseqs_bin,
                            "easy-linclust",
                            str(cluster_input),
                            str(linclust_prefix),
                            str(tmp_dir),
                            "--min-seq-id",
                            "0.65",
                            "--cov-mode",
                            "1",
                            "-c",
                            "0.85",
                            "--threads",
                            str(threads),
                            "--remove-tmp-files",
                            "-v",
                            "3",
                        ],
                        check=True,
                        timeout=7200,  # 2h for fast linclust
                    )
                    logger.info("Stage 1 complete: %s", linclust_reps.name)
                else:
                    logger.info(
                        "Using existing linclust65 representatives: %s",
                        linclust_reps.name,
                    )

                # Count representatives after stage 1
                with open(linclust_reps) as fh:
                    n_reps = sum(1 for ln in fh if ln[0] == ">")
                logger.info("Stage 1 reduced to %d representatives", n_reps)

                # --- Stage 2: cascaded cluster at target identity ---
                logger.info(
                    "Stage 2/2: cascaded cluster at %.0f%% identity (threads=%d)...",
                    min_seq_id * 100,
                    threads,
                )
                db_path = stringdb_dir / "repDB"
                clu_prefix = stringdb_dir / "clu45"

                # Clean stale tmp for stage 2
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                # Step 2a: createdb from linclust representatives
                logger.info("  createdb ...")
                subprocess.run(
                    [mmseqs_bin, "createdb", str(linclust_reps), str(db_path)],
                    check=True,
                    timeout=3600,
                )

                # Step 2b: cascaded clustering (smaller input → much faster)
                logger.info("  cluster (cascaded) ...")
                subprocess.run(
                    [
                        mmseqs_bin,
                        "cluster",
                        str(db_path),
                        str(clu_prefix),
                        str(tmp_dir),
                        "--min-seq-id",
                        str(min_seq_id),
                        "--cov-mode",
                        "1",
                        "-c",
                        "0.75",
                        "--threads",
                        str(threads),
                        "--remove-tmp-files",
                        "-v",
                        "3",
                    ],
                    check=True,
                    timeout=18000,  # 5h for cascaded clustering
                )

                # Step 2c: export stage-2 cluster TSV (rep90 assignments only)
                clu45_reps_tsv = stringdb_dir / "clu45_reps.tsv"
                logger.info("  createtsv ...")
                subprocess.run(
                    [
                        mmseqs_bin,
                        "createtsv",
                        str(db_path),
                        str(db_path),
                        str(clu_prefix),
                        str(clu45_reps_tsv),
                    ],
                    check=True,
                    timeout=3600,
                )

                # --- Compose two-stage mapping → final clu45.tsv ---
                # Stage 2: rep90 → rep45  |  Stage 1: member → rep90
                # Final:   member → rep45
                logger.info("Composing two-stage cluster mapping ...")
                stage2_map: dict[str, str] = {}
                with open(clu45_reps_tsv, "r") as f:
                    for line_tsv in f:
                        rep45, rep70 = line_tsv.rstrip().split("\t")
                        stage2_map[rep70] = rep45

                n_composed = 0
                with open(linclust_cluster_tsv, "r") as fin, open(clu_tsv, "w") as fout:
                    for line_tsv in fin:
                        rep70, member = line_tsv.rstrip().split("\t")
                        rep45 = stage2_map.get(rep70, rep70)
                        fout.write(f"{rep45}\t{member}\n")
                        n_composed += 1

                n_final_clusters = len(set(stage2_map.values()))
                del stage2_map
                logger.info(
                    "Clustering complete: %d proteins → %d clusters → %s",
                    n_composed,
                    n_final_clusters,
                    clu_tsv.name,
                )

                _cleanup_stringdb_intermediates()

            except subprocess.CalledProcessError as e:
                logger.error("MMseqs2 clustering command failed")
                logger.error("STDOUT: %s", e.stdout)
                logger.error("STDERR: %s", e.stderr)
                raise RuntimeError(
                    f"MMseqs2 clustering failed. Check logs above for details.\n"
                    f"Command: {' '.join(e.cmd)}\n"
                    f"Exit code: {e.returncode}"
                ) from e
            except subprocess.TimeoutExpired as e:
                logger.error("MMseqs2 clustering timed out after 5 hours")
                raise RuntimeError(
                    "MMseqs2 clustering timed out. The dataset may be too large.\n"
                    "Consider reducing the dataset size or increasing the timeout."
                ) from e
        else:
            logger.info("Found existing cluster assignments: %s", clu_tsv)

        # Cleanup stale intermediates from previous/incomplete runs
        _cleanup_stringdb_intermediates()

        # --- 4. Polars-based ETL ---
        # Parse FASTA (manual, no lib reader), load clusters as DataFrame,
        # stream score-filtered links, then use Polars joins for cluster
        # assignment, deduplication, and sequence lookup.
        # Reference: https://github.com/VarunUllanat/mint/blob/main/stringdb.py

        # Input preference for ETL: decontaminated -> filtered -> full
        fasta_path = (
            decontaminated_fasta
            if decontaminated_fasta.exists()
            else filtered_fasta
            if filtered_fasta.exists()
            else stringdb_dir / "protein.sequences.v12.0.fa"
        )
        links_path = stringdb_dir / "protein.physical.links.full.v12.0.txt.gz"
        cluster_path = stringdb_dir / "clu45.tsv"
        output_path = stringdb_dir / "stringdb_train.parquet"

        # Avoid stale output ambiguity when re-running
        output_path.unlink(missing_ok=True)

        for p in [fasta_path, links_path, cluster_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Run with correct data_dir to download and cluster."
                )

        # 4a. Parse FASTA → dict (no Polars reader for FASTA format)
        logger.info("Parsing sequences from %s ...", fasta_path.name)
        seqs: dict[str, str] = {}
        current_id: Optional[str] = None
        parts: list[str] = []
        with open(fasta_path, "r") as f:
            for line in f:
                if line[0] == ">":
                    if current_id is not None:
                        seqs[current_id] = "".join(parts)
                    current_id = line[1:].split(None, 1)[0]
                    parts = []
                else:
                    parts.append(line.rstrip())
            if current_id is not None:
                seqs[current_id] = "".join(parts)
        logger.info("Loaded %d sequences", len(seqs))

        # 4b. Load cluster assignments as Polars DataFrame
        clusters = pl.read_csv(
            cluster_path,
            separator="\t",
            has_header=False,
            new_columns=["cluster_rep", "protein_id"],
        )
        logger.info(
            "Loaded %d cluster assignments (%d unique clusters)",
            len(clusters),
            clusters["cluster_rep"].n_unique(),
        )

        # 4c. Stream score-filtered links into Polars DataFrame
        # (gzip streaming required — file is too large for pl.read_csv)
        logger.info(
            "Streaming links from %s (min_score=%d) ...",
            links_path.name,
            min_combined_score,
        )
        p1_col: list[str] = []
        p2_col: list[str] = []
        sc_col: list[int] = []
        n_lines = 0
        with gzip.open(links_path, "rt") as f:
            next(f)  # skip header
            for line in f:
                n_lines += 1
                if n_lines % 100_000_000 == 0:
                    logger.info(
                        "  Scanned %dM links, %d above threshold",
                        n_lines // 1_000_000,
                        len(p1_col),
                    )
                fields = line.split()
                score = int(fields[-1])
                if score >= min_combined_score:
                    p1_col.append(fields[0])
                    p2_col.append(fields[1])
                    sc_col.append(score)

        logger.info(
            "Score filter: %dM links → %d passed (score >= %d)",
            n_lines // 1_000_000,
            len(p1_col),
            min_combined_score,
        )
        links = pl.DataFrame(
            {"protein1": p1_col, "protein2": p2_col, "combined_score": sc_col}
        )
        del p1_col, p2_col, sc_col

        # 4d. Join clusters → filter self-cluster → deduplicate canonical pairs
        #     Keep the highest-scoring representative for each cluster pair.
        n_before = len(links)
        df = (
            links.join(clusters, left_on="protein1", right_on="protein_id")
            .rename({"cluster_rep": "clu1"})
            .join(clusters, left_on="protein2", right_on="protein_id")
            .rename({"cluster_rep": "clu2"})
            .filter(pl.col("clu1") != pl.col("clu2"))
            .with_columns(
                pl.min_horizontal("clu1", "clu2").alias("_lo"),
                pl.max_horizontal("clu1", "clu2").alias("_hi"),
            )
            .sort("combined_score", descending=True)
            .unique(subset=["_lo", "_hi"], keep="first")
            .drop("_lo", "_hi")
        )
        n_after_cluster_dedup = len(df)
        logger.info(
            "Cluster join+dedup: %d score-filtered links -> %d unique cross-cluster pairs (dropped %d)",
            n_before,
            n_after_cluster_dedup,
            n_before - n_after_cluster_dedup,
        )
        del links

        if max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows)
            logger.info("Capped to max_rows=%d", max_rows)

        # 4e. Join sequences
        seq_df = pl.DataFrame(
            {"protein_id": list(seqs.keys()), "sequence": list(seqs.values())}
        )
        del seqs
        n_pre_join1 = len(df)
        df = df.join(seq_df, left_on="protein1", right_on="protein_id").rename(
            {"sequence": "seq1"}
        )
        n_post_join1 = len(df)
        df = df.join(seq_df, left_on="protein2", right_on="protein_id").rename(
            {"sequence": "seq2"}
        )
        n_post_join2 = len(df)
        logger.info(
            "Sequence joins: %d -> %d (protein1) -> %d (protein2), dropped %d pairs",
            n_pre_join1,
            n_post_join1,
            n_post_join2,
            n_pre_join1 - n_post_join2,
        )
        del seq_df

        # 4f. Filter by sequence length
        n_before_len = len(df)
        df = df.filter(
            (pl.col("seq1").str.len_bytes() >= min_seq_len)
            & (pl.col("seq1").str.len_bytes() <= max_seq_len)
            & (pl.col("seq2").str.len_bytes() >= min_seq_len)
            & (pl.col("seq2").str.len_bytes() <= max_seq_len)
        )
        logger.info(
            "Sequence length filter (%d-%d AA): %d → %d pairs (dropped %d)",
            min_seq_len,
            max_seq_len,
            n_before_len,
            len(df),
            n_before_len - len(df),
        )

        # Add group_id and shuffle rows so the output parquet is cluster-unordered.
        # This ensures any prefix or row-group sample drawn at training time is
        # representative of the full dataset, avoiding cluster-prefix bias.
        df = df.with_columns(pl.col("clu1").alias("group_id")).sample(
            fraction=1.0, shuffle=True, seed=42
        )

        logger.info(
            "ETL: %dM links → %d score-filtered → %d unique cross-cluster pairs",
            n_lines // 1_000_000,
            n_before,
            len(df),
        )

        # Save compact training parquet (only columns needed by training)
        train_df = df.select(["seq1", "seq2"])
        logger.info("Saving %d interaction pairs to %s", len(train_df), output_path)
        train_df.write_parquet(str(output_path))

        # Output sanity checks
        total_pairs = len(train_df)
        if total_pairs > 120_000_000:
            logger.warning(
                "Suspiciously high STRING-DB pair count: %d (>120M). "
                "Reference baseline is ~96M at 50%% clustering with no linclust.",
                total_pairs,
            )

        invalid_seq_pairs = train_df.filter(
            pl.col("seq1").is_null()
            | pl.col("seq2").is_null()
            | (pl.col("seq1").str.len_bytes() == 0)
            | (pl.col("seq2").str.len_bytes() == 0)
        )
        if len(invalid_seq_pairs) > 0:
            raise ValueError(
                f"Found {len(invalid_seq_pairs)} output pairs with null/empty sequences"
            )

        output_size_gb = output_path.stat().st_size / (1024**3)
        seq1_len = pl.col("seq1").str.len_bytes()
        seq2_len = pl.col("seq2").str.len_bytes()
        len_stats = train_df.select(
            seq1_len.min().alias("seq1_min"),
            seq1_len.max().alias("seq1_max"),
            seq1_len.mean().alias("seq1_mean"),
            seq2_len.min().alias("seq2_min"),
            seq2_len.max().alias("seq2_max"),
            seq2_len.mean().alias("seq2_mean"),
        ).row(0)
        logger.info("Output parquet size: %.2f GB", output_size_gb)
        logger.info(
            "Seq length stats - seq1[min=%d max=%d mean=%.1f], seq2[min=%d max=%d mean=%.1f]",
            len_stats[0],
            len_stats[1],
            len_stats[2],
            len_stats[3],
            len_stats[4],
            len_stats[5],
        )

        # Stats
        logger.info("--- STRING-DB PPI Dataset Statistics ---")
        logger.info("Total pairs: %d", len(df))
        logger.info("Unique clusters (group_id): %d", df["group_id"].n_unique())
        logger.info("Unique proteins (protein1): %d", df["protein1"].n_unique())
        logger.info("Unique proteins (protein2): %d", df["protein2"].n_unique())
        logger.info(
            "Score range: %d - %d (mean: %.1f)",
            df["combined_score"].min(),
            df["combined_score"].max(),
            df["combined_score"].mean(),
        )
        logger.info("STRING-DB PPI dataset ready: %s", output_path)
        _cleanup_stringdb_intermediates(remove_final_artifacts=True)

    def prep_dms(
        self,
        max_pairs_per_assay: int = 0,
        min_mutations_per_assay: int = 1,
        intra_pairs: bool = False,
        intra_pairs_per_assay: int = 8_000,
        deduplicate_benchmarks: bool = True,
        drop_benchmark_test_fold: bool = True,
        force: bool = False,
    ) -> None:
        """Prepare ProteinGym DMS and clinical similarity-training outputs.

        Writes dms_cosent.parquet with paired sentence_0, sentence_1, score rows
        for --dms_file / CoSENTLoss, where higher score means the mutant should
        embed closer to target_seq, be more WT-like, and be more functional.
        Continuous DMS rows use within-assay normalized DMS_score scaled to
        [0,1], while clinical rows map benign -> 1.0 and pathogenic -> 0.0 for
        similarity training. Also writes dms_sequences.parquet with unique
        sequences only for optional --simcse_files, not as a replacement for
        dms_cosent.parquet.

          By default this keeps the usable ProteinGym train rows while:
          1) excluding known benchmark-overlap assays (GB1_ and GFP_AEQVI_), and
          2) dropping the supervised benchmark test fold for groups with >=10
              rows, using the same deterministic 80/20 per-group split logic as
              protein_benchmark_suite.py (RandomState(42)).

          If a recognized split column is present in the raw ProteinGym data
          (stage/split/set/fold), explicit test rows are dropped directly.
        """
        output_path = os.path.join(self.data_dir, "dms_cosent.parquet")
        simcse_path = os.path.join(self.data_dir, "dms_sequences.parquet")
        if force:
            for path in (output_path, simcse_path):
                if os.path.exists(path):
                    os.remove(path)

        if os.path.exists(output_path):
            logger.info(
                "DMS parquet already exists: %s (use --force to regenerate)",
                output_path,
            )
            return

        result_frames: list[pl.DataFrame] = []
        simcse_frames: list[pl.DataFrame] = []
        forbidden_pair_frames: list[pl.DataFrame] = []
        logger.info(
            "DMS similarity polarity: higher scores mean closer/more functional; "
            "clinical benign/pathogenic are mapped to 1/0 for similarity training"
        )

        def _empty_forbidden_pairs() -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "sentence_0": pl.String,
                    "sentence_1": pl.String,
                }
            )

        def _load_proteingym_split(split_name: str):
            """Load a ProteinGym split from HF or an older local cache."""

            try:
                return load_dataset(
                    "OATML-Markslab/ProteinGym_v1",
                    name=split_name,
                    split="train",
                )
            except Exception as name_error:
                logger.info(
                    "  Falling back to cached data_dir loading for %s: %s",
                    split_name,
                    name_error,
                )
                return load_dataset(
                    "OATML-Markslab/ProteinGym_v1",
                    data_dir=split_name,
                    split="train",
                )

        def _drop_benchmark_test_rows(
            df: pl.DataFrame,
            *,
            group_col: str,
            split_name: str,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            """Drop benchmark test rows/sequences using explicit split metadata or fallback split."""
            if not drop_benchmark_test_fold or len(df) == 0:
                return df, _empty_forbidden_pairs()

            for split_col in _SPLIT_COLUMN_CANDIDATES:
                if split_col not in df.columns:
                    continue

                normalized = (
                    pl.col(split_col)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .str.strip_chars()
                    .fill_null("")
                )
                present_values = set(
                    df.select(normalized.alias("_split"))["_split"].to_list()
                )
                has_recognized_splits = bool(
                    present_values & (_TRAIN_SPLIT_VALUES | _TEST_SPLIT_VALUES)
                )
                if not has_recognized_splits:
                    continue

                before = len(df)
                test_rows = df.filter(normalized.is_in(sorted(_TEST_SPLIT_VALUES)))
                forbidden_pairs = _empty_forbidden_pairs()
                if len(test_rows) > 0 and {
                    "mutated_sequence",
                    "target_seq",
                }.issubset(set(test_rows.columns)):
                    forbidden_pairs = test_rows.select(
                        pl.col("mutated_sequence").alias("sentence_0"),
                        pl.col("target_seq").alias("sentence_1"),
                    ).unique()
                filtered = df.filter(~normalized.is_in(sorted(_TEST_SPLIT_VALUES)))

                if (
                    len(test_rows) > 0
                    and "mutated_sequence" in df.columns
                    and group_col in df.columns
                ):
                    test_keys = test_rows.select(
                        [group_col, "mutated_sequence"]
                    ).unique()
                    filtered = filtered.join(
                        test_keys,
                        on=[group_col, "mutated_sequence"],
                        how="anti",
                    )

                logger.info(
                    "  Dropped %d rows from %s using explicit split column '%s'",
                    before - len(filtered),
                    split_name,
                    split_col,
                )
                return filtered, forbidden_pairs

            if group_col not in df.columns:
                logger.warning(
                    "  Could not drop benchmark test fold for %s: missing group column '%s'",
                    split_name,
                    group_col,
                )
                return df, _empty_forbidden_pairs()

            group_values = np.asarray(df[group_col].to_list())
            mutant_values = (
                np.asarray(df["mutated_sequence"].to_list())
                if "mutated_sequence" in df.columns
                else None
            )
            target_values = (
                np.asarray(df["target_seq"].to_list())
                if "target_seq" in df.columns
                else None
            )
            keep_mask = np.ones(len(df), dtype=bool)
            dropped_groups = 0
            forbidden_pairs_list: list[tuple[str, str]] = []

            for group_value in np.unique(group_values):
                group_indices = np.where(group_values == group_value)[0]
                group_size = len(group_indices)
                # Match supervised ProteinGym benchmark behavior: groups with
                # <10 rows are skipped during evaluation and are left intact here.
                if group_size < 10:
                    continue

                shuffled = np.random.RandomState(42).permutation(group_size)
                split_idx = int(group_size * 0.8)
                test_local_indices = shuffled[split_idx:]
                if len(test_local_indices) == 0:
                    continue

                if mutant_values is None:
                    keep_mask[group_indices[test_local_indices]] = False
                else:
                    test_sequences = set(
                        mutant_values[group_indices[test_local_indices]]
                    )
                    group_mutants = mutant_values[group_indices]
                    keep_mask[
                        group_indices[np.isin(group_mutants, list(test_sequences))]
                    ] = False
                    if target_values is not None:
                        group_targets = target_values[group_indices]
                        for sequence, target in zip(group_mutants, group_targets):
                            if sequence in test_sequences:
                                forbidden_pairs_list.append(
                                    (str(sequence), str(target))
                                )
                dropped_groups += 1

            filtered = df.filter(pl.Series("_keep_supervised_train", keep_mask))
            logger.info(
                "  Dropped %d rows from %s via deterministic supervised fold filter "
                "(%d groups affected)",
                len(df) - len(filtered),
                split_name,
                dropped_groups,
            )
            forbidden_pairs = _empty_forbidden_pairs()
            if forbidden_pairs_list:
                forbidden_pairs = pl.DataFrame(
                    {
                        "sentence_0": [seq for seq, _ in forbidden_pairs_list],
                        "sentence_1": [target for _, target in forbidden_pairs_list],
                    }
                ).unique()
            return filtered, forbidden_pairs

        def _sample_intra_pairs(df: pl.DataFrame) -> pl.DataFrame:
            """Build optional mutant-mutant pairs scored by fitness similarity."""

            if not intra_pairs:
                return pl.DataFrame(
                    schema={
                        "sentence_0": pl.String,
                        "sentence_1": pl.String,
                        "score": pl.Float64,
                    }
                )

            intra_rows: list[pl.DataFrame] = []
            for _, assay_df in df.group_by("DMS_id"):
                assay_size = len(assay_df)
                if assay_size < 2:
                    continue

                sequences = assay_df["mutated_sequence"].to_list()
                scores = assay_df["score"].to_list()
                max_possible_pairs = assay_size * (assay_size - 1) // 2
                if max_possible_pairs == 0:
                    continue

                if (
                    intra_pairs_per_assay <= 0
                    or intra_pairs_per_assay >= max_possible_pairs
                ):
                    pair_indices = list(itertools.combinations(range(assay_size), 2))
                else:
                    rng = np.random.default_rng(seed=42)
                    pair_index_set: set[tuple[int, int]] = set()
                    while len(pair_index_set) < intra_pairs_per_assay:
                        batch_size = min(intra_pairs_per_assay * 2, max_possible_pairs)
                        idx_a = rng.integers(0, assay_size, size=batch_size)
                        idx_b = rng.integers(0, assay_size, size=batch_size)
                        for left_idx, right_idx in zip(idx_a, idx_b):
                            if left_idx == right_idx:
                                continue
                            left = int(left_idx)
                            right = int(right_idx)
                            pair = (left, right) if left < right else (right, left)
                            pair_index_set.add(pair)
                            if len(pair_index_set) >= intra_pairs_per_assay:
                                break
                    pair_indices = list(pair_index_set)

                intra_rows.append(
                    pl.DataFrame(
                        {
                            "sentence_0": [sequences[left] for left, _ in pair_indices],
                            "sentence_1": [
                                sequences[right] for _, right in pair_indices
                            ],
                            "score": [
                                1.0 - abs(scores[left] - scores[right])
                                for left, right in pair_indices
                            ],
                        }
                    )
                )

            if not intra_rows:
                return pl.DataFrame(
                    schema={
                        "sentence_0": pl.String,
                        "sentence_1": pl.String,
                        "score": pl.Float64,
                    }
                )

            return pl.concat(intra_rows, how="vertical")

        # --- DMS splits (continuous scores) ---
        for split_name in ["DMS_substitutions", "DMS_indels"]:
            logger.info("Loading ProteinGym %s ...", split_name)
            try:
                ds = _load_proteingym_split(split_name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", split_name, e)
                continue

            ds_columns = (
                ds.column_names if hasattr(ds, "column_names") else list(ds.keys())
            )

            dms_data: dict[str, list[object]] = {
                "DMS_id": ds["DMS_id"],
                "mutated_sequence": ds["mutated_sequence"],
                "target_seq": ds["target_seq"],
                "DMS_score": ds["DMS_score"],
            }
            for split_col in _SPLIT_COLUMN_CANDIDATES:
                if split_col in ds_columns:
                    dms_data[split_col] = ds[split_col]

            df = pl.DataFrame(dms_data)
            logger.info("  Loaded %d rows from %s", len(df), split_name)

            if deduplicate_benchmarks:
                before = len(df)
                overlap_mask = pl.lit(False)
                for prefix in self._BENCHMARK_OVERLAP_PREFIXES:
                    overlap_mask = overlap_mask | pl.col("DMS_id").str.starts_with(
                        prefix
                    )
                df = df.filter(~overlap_mask)
                logger.info(
                    "  Removed %d benchmark-overlap rows from %s",
                    before - len(df),
                    split_name,
                )

            df, forbidden_pairs = _drop_benchmark_test_rows(
                df,
                group_col="DMS_id",
                split_name=split_name,
            )
            if len(forbidden_pairs) > 0:
                forbidden_pair_frames.append(forbidden_pairs)

            # Per-assay Z-score normalization
            df = df.with_columns(
                pl.col("DMS_score")
                .sub(pl.col("DMS_score").mean().over("DMS_id"))
                .truediv(
                    pl.col("DMS_score").std().over("DMS_id").clip(lower_bound=1e-8)
                )
                .clip(-3.0, 3.0)
                .add(3.0)
                .truediv(6.0)
                .alias("score")
            )

            # Filter small assays
            assay_counts = df.group_by("DMS_id").len()
            keep_assays = assay_counts.filter(pl.col("len") >= min_mutations_per_assay)[
                "DMS_id"
            ]
            df = df.filter(pl.col("DMS_id").is_in(keep_assays.implode()))
            logger.info(
                "  After filtering (min %d mutations): %d rows, %d assays",
                min_mutations_per_assay,
                len(df),
                df["DMS_id"].n_unique(),
            )

            if max_pairs_per_assay > 0:
                df = df.group_by("DMS_id").map_groups(
                    lambda group_df: group_df.sample(
                        n=min(len(group_df), max_pairs_per_assay),
                        shuffle=True,
                        seed=42,
                    )
                )
                logger.info(
                    "  Applied assay cap (%d): %d rows remain",
                    max_pairs_per_assay,
                    len(df),
                )

            result_frames.append(
                df.select(
                    pl.col("mutated_sequence").alias("sentence_0"),
                    pl.col("target_seq").alias("sentence_1"),
                    pl.col("score").cast(pl.Float64),
                )
            )
            simcse_frames.append(
                pl.concat(
                    [
                        df.select(pl.col("mutated_sequence").alias("sequence")),
                        df.select(pl.col("target_seq").alias("sequence")),
                    ],
                    how="vertical",
                )
            )

            intra_df = _sample_intra_pairs(df)
            if len(intra_df) > 0:
                result_frames.append(intra_df)
                logger.info(
                    "  Added %d intra-assay pairs from %s",
                    len(intra_df),
                    split_name,
                )

        # --- Clinical splits (binary labels) ---
        for split_name in ["clinical_substitutions", "clinical_indels"]:
            logger.info("Loading ProteinGym %s ...", split_name)
            try:
                ds = _load_proteingym_split(split_name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", split_name, e)
                continue

            ds_columns = (
                ds.column_names if hasattr(ds, "column_names") else list(ds.keys())
            )

            label_map = {"Pathogenic": 0.0, "Benign": 1.0, "0": 1.0, "1": 0.0}
            clinical_data: dict[str, list[object] | list[str]] = {
                "mutated_sequence": list(ds["mutated_sequence"]),
                "target_seq": list(ds["target_seq"]),
                "annotation": [str(label) for label in ds["annotation"]],
                "protein_id": (
                    list(ds["protein_id"])
                    if "protein_id" in ds_columns
                    else [""] * len(ds["annotation"])
                ),
            }
            for split_col in _SPLIT_COLUMN_CANDIDATES:
                if split_col in ds_columns:
                    clinical_data[split_col] = list(ds[split_col])

            clinical_df = pl.DataFrame(clinical_data)
            clinical_df, forbidden_pairs = _drop_benchmark_test_rows(
                clinical_df,
                group_col="protein_id",
                split_name=split_name,
            )
            if len(forbidden_pairs) > 0:
                forbidden_pair_frames.append(forbidden_pairs)
            clinical_df = clinical_df.with_columns(
                pl.col("annotation")
                .replace(label_map)
                .cast(pl.Float64, strict=False)
                .alias("score")
            ).filter(pl.col("score").is_not_null())

            result_frames.append(
                clinical_df.select(
                    pl.col("mutated_sequence").alias("sentence_0"),
                    pl.col("target_seq").alias("sentence_1"),
                    pl.col("score"),
                )
            )
            simcse_frames.append(
                pl.concat(
                    [
                        clinical_df.select(
                            pl.col("mutated_sequence").alias("sequence")
                        ),
                        clinical_df.select(pl.col("target_seq").alias("sequence")),
                    ],
                    how="vertical",
                )
            )
            logger.info(
                "  Added %d clinical pairs from %s",
                len(clinical_df),
                split_name,
            )

        if not result_frames:
            raise ValueError("No DMS/clinical data loaded. Check network and dataset.")

        result_df = pl.concat(result_frames, how="vertical").sample(
            fraction=1.0,
            shuffle=True,
            seed=42,
        )
        if drop_benchmark_test_fold and forbidden_pair_frames:
            forbidden_pairs = pl.concat(forbidden_pair_frames, how="vertical").unique()
            before_global = len(result_df)
            result_df = result_df.join(
                forbidden_pairs,
                on=["sentence_0", "sentence_1"],
                how="anti",
            )
            logger.info(
                "Globally removed %d benchmark test-fold pairs from final CoSENT data",
                before_global - len(result_df),
            )
        n_unique_wt = result_df["sentence_1"].n_unique()
        logger.info(
            "DMS dataset: %d total pairs, %d unique wildtype proteins",
            len(result_df),
            n_unique_wt,
        )
        logger.info(
            "Score stats: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
            result_df["score"].min(),
            result_df["score"].max(),
            result_df["score"].mean(),
            result_df["score"].std(),
        )
        result_df.write_parquet(output_path)
        logger.info("Saved DMS CoSENT parquet: %s", output_path)

        simcse_df = (
            pl.concat(simcse_frames, how="vertical")
            .unique(subset=["sequence"])
            .sort("sequence")
        )
        simcse_df.write_parquet(simcse_path)
        logger.info(
            "Saved DMS SimCSE sequence parquet: %s (%d unique sequences)",
            simcse_path,
            len(simcse_df),
        )

    def prep_pfam_hard_negatives(
        self,
        max_evalue: float = 1.0,
        evalue_z: float = 1e6,
        max_mutation_fraction: float = 0.5,
        min_aligned_positions: int = 20,
        force: bool = False,
        max_total_rows: int = 0,
        max_seqs_per_family: int = 100,
        workers: int = 0,
    ) -> None:
        """Generate HMM-verified hard negatives for Pfam domains.

        For each anchor the profile is used to rank aligned positions by how
        much score a substitution there destroys, the most damaging ones are
        applied, and HMMER is re-run on the mutant. A mutant is only kept once
        the family no longer recognises it above ``max_evalue``, and the
        smallest mutation count that clears the bar is chosen, so the negative
        stays as similar to the anchor as the criterion permits.

        This replaces an earlier scheme that applied six substitutions chosen
        for a fixed total log-odds drop. That produced sequences which HMMER
        still matched to the source family with E-values around 1e-8, i.e.
        false negatives, and it mapped sequence offsets onto match states
        directly, which is only correct for a gaplessly aligned domain.

        Positions are ranked by a total order and no sampling is involved, so
        repeated runs over the same input produce byte-identical output.

        Output is pfam_hard_negatives.parquet containing the core columns plus
        ``hard_negative``. Anchors with no verified negative are dropped, so
        the column is never null.

        Args:
            max_evalue: A mutant counts as a negative once its E-value against
                its own family exceeds this (default 1.0 = no meaningful hit).
            evalue_z: Effective database size for E-value calculation. Pinned
                so acceptance does not depend on batch size.
            max_mutation_fraction: Cap on the fraction of aligned positions
                that may be mutated (default 0.5, bounding identity loss).
            min_aligned_positions: Skip anchors aligning to fewer match states.
            force: Overwrite existing output if it exists.
            max_total_rows: Global cap on selected rows (0 = uncapped).
            max_seqs_per_family: Cap per-family rows (0 = uncapped).
            workers: Thread workers (0 = auto).

        Raises:
            FileNotFoundError: If pfam_sorted.parquet is not found.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import pyhmmer
        from pyhmmer.easel import Alphabet
        from pyhmmer.plan7 import HMMFile

        data_path = Path(self.data_dir)
        source_parquet = data_path / "pfam_sorted.parquet"
        hmm_gz_path = data_path / "Pfam-A.hmm.gz"
        output_path = data_path / "pfam_hard_negatives.parquet"

        if output_path.exists() and not force:
            logger.info(
                "Output already exists: %s (use --force to overwrite)",
                output_path,
            )
            return

        if not source_parquet.exists():
            raise FileNotFoundError(
                f"Source parquet not found: {source_parquet}. "
                "Run `python data_prep.py --dataset pfam` first."
            )

        hmm_url = (
            "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
        )
        _download_file(hmm_url, hmm_gz_path)

        source_schema = pl.read_parquet_schema(source_parquet)
        keep_cols = [
            c
            for c in ["sequence", "family_id", "clan_id", "group_id", "domain_id"]
            if c in source_schema
        ]
        if "sequence" not in keep_cols or "family_id" not in keep_cols:
            raise ValueError("pfam_sorted.parquet must contain sequence and family_id")

        logger.info("Selecting source rows (5 < len < 1024, capped) ...")
        selected_lf = (
            pl.scan_parquet(source_parquet)
            .select([pl.col(c) for c in keep_cols])
            .filter(pl.col("sequence").str.len_bytes() > 5)
            .filter(pl.col("sequence").str.len_bytes() < 1024)
        )
        if max_seqs_per_family > 0:
            # maintain_order keeps selection reproducible across runs.
            selected_lf = selected_lf.group_by("family_id", maintain_order=True).head(
                max_seqs_per_family
            )

        selected_df = cast(pl.DataFrame, selected_lf.collect())
        if max_total_rows > 0 and selected_df.height > max_total_rows:
            selected_df = selected_df.sample(n=max_total_rows, seed=42, shuffle=True)
        if "group_id" not in selected_df.columns:
            selected_df = selected_df.with_columns(
                pl.col("family_id").alias("group_id")
            )
        selected_df = selected_df.sort("family_id", maintain_order=True)
        selected_df = selected_df.with_row_index("__row_idx")
        total_selected = selected_df.height
        wanted_families = set(selected_df["family_id"].unique().to_list())
        logger.info(
            "Selected %d rows across %d families",
            total_selected,
            len(wanted_families),
        )

        logger.info("Loading Pfam HMM profiles...")
        alphabet = Alphabet.amino()
        background = pyhmmer.plan7.Background(alphabet)
        log_background = np.log2(
            np.maximum(
                np.array(
                    [
                        background.residue_frequencies[i]
                        for i in range(len(PFAM_HN_AA))
                    ],
                    dtype=np.float32,
                ),
                1e-9,
            )
        )

        hmm_dict = {}
        with HMMFile(hmm_gz_path) as hmm_file:
            for hmm in hmm_file:
                raw_acc = hmm.accession
                if raw_acc is None:
                    continue
                acc = raw_acc.decode() if isinstance(raw_acc, bytes) else raw_acc
                family_id = acc.split(".")[0]
                if family_id in wanted_families:
                    hmm_dict[family_id] = hmm
        logger.info(
            "Loaded %d family profiles covering %d of %d selected families",
            len(hmm_dict),
            len(hmm_dict),
            len(wanted_families),
        )
        if not hmm_dict:
            logger.error("No HMM profiles matched the selected families; aborting.")
            return

        def _process_family(
            family_id: str, row_indices: list[int], sequences: list[str]
        ) -> list[tuple[int, str | None]]:
            """Generate verified negatives for one family."""
            negatives = _generate_family_negatives(
                hmm_dict[family_id],
                sequences,
                log_background,
                max_evalue=max_evalue,
                evalue_z=evalue_z,
                max_mutation_fraction=max_mutation_fraction,
                min_aligned_positions=min_aligned_positions,
            )
            return list(zip(row_indices, negatives))

        if workers <= 0:
            workers = min(os.cpu_count() or 1, 32)

        hard_negative_col = cast(list[str | None], [None] * total_selected)
        family_batches = (
            selected_df.group_by("family_id", maintain_order=True)
            .agg(
                pl.col("__row_idx").alias("row_indices"),
                pl.col("sequence").alias("sequences"),
            )
            .iter_rows(named=True)
        )

        logger.info(
            "Generating negatives (workers=%d, max_evalue=%s, max_mut_frac=%.2f)...",
            workers,
            max_evalue,
            max_mutation_fraction,
        )
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row in family_batches:
                family_id = row["family_id"]
                if family_id not in hmm_dict:
                    continue
                futures.append(
                    executor.submit(
                        _process_family,
                        family_id,
                        row["row_indices"],
                        row["sequences"],
                    )
                )
            for done, future in enumerate(as_completed(futures), start=1):
                for row_idx, negative in future.result():
                    hard_negative_col[row_idx] = negative
                if done % 250 == 0:
                    logger.info("Processed %d/%d families", done, len(futures))

        result_df = (
            selected_df.with_columns(
                pl.Series("hard_negative", hard_negative_col, dtype=pl.Utf8),
            )
            .drop("__row_idx")
            .filter(pl.col("hard_negative").is_not_null())
        )
        logger.info(
            "Anchors with a verified hard negative: %d / %d (%.1f%%)",
            result_df.height,
            total_selected,
            100 * result_df.height / max(1, total_selected),
        )
        if result_df.height:
            stats = result_df.select(
                (
                    100.0
                    * pl.struct("sequence", "hard_negative")
                    .map_elements(
                        lambda r: sum(
                            a != b
                            for a, b in zip(r["sequence"], r["hard_negative"])
                        )
                        / max(1, len(r["sequence"])),
                        return_dtype=pl.Float64,
                    )
                ).alias("pct_mutated")
            )["pct_mutated"]
            logger.info(
                "Mutated fraction: median %.1f%% (identity to anchor ~%.1f%%)",
                stats.median(),
                100 - stats.median(),
            )

        sort_cols = [c for c in ["clan_id", "family_id"] if c in result_df.columns]
        if sort_cols:
            result_df = result_df.sort(sort_cols, maintain_order=True)

        result_df = result_df.rechunk()
        result_df.write_parquet(output_path)
        logger.info("Saved: %s (%d rows)", output_path, len(result_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare protein datasets")
    parser.add_argument(
        "--dataset",
        choices=["pfam", "nvidia", "afdb", "stringdb", "dms", "pfam_hard_negatives"],
        required=True,
    )
    parser.add_argument(
        "--limit_gb",
        type=int,
        default=30,
        help=(
            "Approximate output size cap in GB for capped ETLs. "
            "For AFDB, set <= 0 for an uncapped full join."
        ),
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--min_combined_score",
        type=int,
        default=400,
        help="STRING-DB: minimum combined_score filter (default 400 = medium confidence, 0 = no filter)",
    )
    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.5,
        help="STRING-DB: MMseqs2 cascaded-cluster sequence identity threshold for "
        "*internal* redundancy reduction (default 0.5)",
    )
    parser.add_argument(
        "--decontam_min_seq_id",
        type=float,
        default=0.4,
        help="Identity cutoff for removing training sequences similar to the benchmark "
        "test split, at 80%% coverage of the test sequence (default 0.4). "
        "Applies to stringdb (vs Bernett PPI test) and afdb/pfam (vs remote-homology test).",
    )
    parser.add_argument(
        "--no_decontaminate",
        action="store_true",
        help="Skip benchmark test-set decontamination for afdb/pfam (not recommended).",
    )
    parser.add_argument(
        "--mmseqs_gpu",
        action="store_true",
        help="Use the MMseqs2 GPU prefilter for the decontamination search. Only worth it "
        "for large target databases; with a few thousand test sequences the CPU path "
        "(-s 7.5, many threads) is faster.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="STRING-DB: maximum interaction pairs (0 = no limit)",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="STRING-DB: minimum sequence length in amino acids (default 10)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="STRING-DB: maximum sequence length in amino acids (default 1024)",
    )
    parser.add_argument(
        "--stringdb_cleanup",
        choices=["keep", "aggressive"],
        default="aggressive",
        help="STRING-DB cleanup mode: keep reusable artifacts or aggressively delete intermediates",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared outputs for the selected dataset",
    )
    parser.add_argument(
        "--min_mutations_per_assay",
        type=int,
        default=1,
        help="DMS: minimum mutations per assay to keep (default 1 = keep all usable assays)",
    )
    parser.add_argument(
        "--max_pairs_per_assay",
        type=int,
        default=0,
        help="DMS: cap mutant-WT rows per assay (0 = no cap)",
    )
    parser.add_argument(
        "--dms_intra_pairs",
        action="store_true",
        help="DMS: add mutant-mutant pairs within each assay",
    )
    parser.add_argument(
        "--dms_intra_pairs_per_assay",
        type=int,
        default=5000,
        help="DMS: max mutant-mutant pairs per assay when --dms_intra_pairs is enabled",
    )
    parser.add_argument(
        "--deduplicate_benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "DMS: drop known benchmark-overlap assays (GB1_, GFP_AEQVI_) "
            "(default on; use --no-deduplicate_benchmarks to disable)"
        ),
    )
    parser.add_argument(
        "--drop_benchmark_test_fold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "DMS: drop ProteinGym supervised benchmark test fold rows from "
            "CoSENT training data. Uses explicit split metadata when present, "
            "otherwise deterministic per-group 80/20 split (default on)."
        ),
    )
    parser.add_argument(
        "--hard_negative_max_evalue",
        type=float,
        default=1.0,
        help=(
            "pfam_hard_negatives: a mutant is accepted as a negative once its "
            "E-value against its own family exceeds this (default 1.0)"
        ),
    )
    parser.add_argument(
        "--hard_negative_evalue_z",
        type=float,
        default=1e6,
        help=(
            "pfam_hard_negatives: effective database size for E-value "
            "calculation, pinned so acceptance is batch-independent "
            "(default 1e6)"
        ),
    )
    parser.add_argument(
        "--hard_negative_max_mut_frac",
        type=float,
        default=0.5,
        help=(
            "pfam_hard_negatives: cap on the fraction of aligned positions "
            "that may be mutated (default 0.5)"
        ),
    )
    parser.add_argument(
        "--max_total_rows",
        type=int,
        default=0,
        help=(
            "pfam_hard_negatives: global cap on selected rows before generation "
            "(default 0 = uncapped)"
        ),
    )
    parser.add_argument(
        "--max_seqs_per_family",
        type=int,
        default=100,
        help=(
            "pfam_hard_negatives: per-family cap before generation "
            "(default 100, 0 = uncapped)"
        ),
    )
    parser.add_argument(
        "--min_aligned_positions",
        type=int,
        default=20,
        help=(
            "pfam_hard_negatives: skip anchors aligning to fewer match states "
            "(default 20)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pfam_hard_negatives: worker threads (default 0 = auto)",
    )

    args = parser.parse_args()
    dp = DataPrep(data_dir=args.data_dir)

    if args.dataset == "nvidia":
        dp.prep_nvidia(args.limit_gb)
    elif args.dataset == "pfam":
        dp.prep_pfam_full(
            fast=args.fast,
            decontaminate=not args.no_decontaminate,
            decontam_min_seq_id=args.decontam_min_seq_id,
            mmseqs_gpu=args.mmseqs_gpu,
        )
    elif args.dataset == "afdb":
        dp.prep_afdb(
            args.limit_gb,
            decontaminate=not args.no_decontaminate,
            decontam_min_seq_id=args.decontam_min_seq_id,
            mmseqs_gpu=args.mmseqs_gpu,
        )
    elif args.dataset == "stringdb":
        dp.prep_stringdb(
            min_seq_id=args.min_seq_id,
            min_combined_score=args.min_combined_score,
            max_rows=args.max_rows,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            cleanup_mode=args.stringdb_cleanup,
            decontam_min_seq_id=args.decontam_min_seq_id,
            mmseqs_gpu=args.mmseqs_gpu,
        )
    elif args.dataset == "dms":
        dp.prep_dms(
            max_pairs_per_assay=args.max_pairs_per_assay,
            min_mutations_per_assay=args.min_mutations_per_assay,
            intra_pairs=args.dms_intra_pairs,
            intra_pairs_per_assay=args.dms_intra_pairs_per_assay,
            deduplicate_benchmarks=args.deduplicate_benchmarks,
            drop_benchmark_test_fold=args.drop_benchmark_test_fold,
            force=args.force,
        )
    elif args.dataset == "pfam_hard_negatives":
        dp.prep_pfam_hard_negatives(
            max_evalue=args.hard_negative_max_evalue,
            evalue_z=args.hard_negative_evalue_z,
            max_mutation_fraction=args.hard_negative_max_mut_frac,
            min_aligned_positions=args.min_aligned_positions,
            force=args.force,
            max_total_rows=args.max_total_rows,
            max_seqs_per_family=args.max_seqs_per_family,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
