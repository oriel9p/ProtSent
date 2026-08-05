"""Pair sampling: same budget per cluster, more distinct sequences.

The default changed from "draw k members, emit every C(k, 2) pair among them"
to "spend that same budget on disjoint pairs". The point of the change is
coverage, and the constraint is that it must not alter corpus size -- step
counts and LR schedules are derived from it.
"""

from __future__ import annotations

import random
import sys
from math import comb
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from protein_pipeline import _build_pair_dataset  # noqa: E402


def _corpus(tmp_path: Path, cluster_sizes: list[int]) -> str:
    """Group-sorted parquet with one distinct sequence per member."""
    seqs, groups = [], []
    for gi, n in enumerate(cluster_sizes):
        for j in range(n):
            seqs.append(f"SEQ{gi}_{j}")
            groups.append(f"g{gi}")
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"sequence": seqs, "group_id": groups}), path)
    return str(path)


def _build(path: str, k: int, mode: str):
    return _build_pair_dataset(
        file_paths=[path],
        seq_col="sequence",
        group_col="group_id",
        max_pairs_per_cluster=k,
        max_pairs=0,
        pair_sampling=mode,
    )


@pytest.mark.parametrize("mode", ["disjoint", "combinations"])
def test_pair_count_is_the_budget(tmp_path: Path, mode: str) -> None:
    """Corpus size must not depend on the sampling mode: step counts ride on it."""
    sizes = [2, 3, 8, 50, 500]
    k = 8
    random.seed(0)
    ds = _build(_corpus(tmp_path, sizes), k, mode)
    expected = sum(comb(min(n, k), 2) for n in sizes)
    assert len(ds) == expected


def test_disjoint_covers_more_of_a_large_cluster(tmp_path: Path) -> None:
    """The whole point: same budget, more distinct sequences.

    One 500-member cluster at k=8 has a 28-pair budget. combinations spends it
    on 8 sequences; disjoint spends it on 56.
    """
    k = 8
    path = _corpus(tmp_path, [500])

    random.seed(0)
    combo = _build(path, k, "combinations")
    random.seed(0)
    disjoint = _build(path, k, "disjoint")

    assert len(combo) == len(disjoint) == comb(k, 2)

    def distinct(ds):
        return len(set(ds["sentence_0"]) | set(ds["sentence_1"]))

    assert distinct(combo) == k
    assert distinct(disjoint) == 2 * comb(k, 2)
    assert distinct(disjoint) > distinct(combo)


def test_pairs_never_join_different_clusters(tmp_path: Path) -> None:
    """A pair is a positive; crossing clusters would be a mislabelled one."""
    random.seed(0)
    ds = _build(_corpus(tmp_path, [4, 9, 40]), 8, "disjoint")
    for a, b in zip(ds["sentence_0"], ds["sentence_1"]):
        assert a.split("_")[0] == b.split("_")[0]
        assert a != b  # and never pairs a sequence with itself


def test_small_clusters_are_exhaustive_either_way(tmp_path: Path) -> None:
    """Below the cap the budget is every pair, so both modes must agree."""
    random.seed(0)
    a = _build(_corpus(tmp_path, [3]), 8, "combinations")
    random.seed(0)
    b = _build(_corpus(tmp_path, [3]), 8, "disjoint")
    norm = lambda ds: {frozenset((x, y)) for x, y in zip(ds["sentence_0"], ds["sentence_1"])}
    assert norm(a) == norm(b) == {frozenset({"SEQ0_0", "SEQ0_1"}),
                                 frozenset({"SEQ0_0", "SEQ0_2"}),
                                 frozenset({"SEQ0_1", "SEQ0_2"})}
