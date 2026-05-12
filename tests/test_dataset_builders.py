"""Offline unit tests for dataset builder functions.

Tests _load_ppi_pair_dataset, _build_pair_dataset, and _build_label_dataset
using small synthetic parquet files. No GPU required.

Run:
    pytest tests/test_dataset_builders.py -v
    pytest tests/test_dataset_builders.py -v -s -m benchmark
"""

import random
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_pipeline import (
    _build_label_dataset,
    _build_pair_dataset,
    _load_ppi_pair_dataset,
)

random.seed(42)
_AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand_seq(length: int = 60) -> str:
    return "".join(random.choices(_AA, k=length))


# ---------------------------------------------------------------------------
# Fixtures: small synthetic parquet files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ppi_parquet(tmp_path_factory):
    """500-pair PPI parquet with seq1/seq2 plus an extra column."""
    tmp = tmp_path_factory.mktemp("ppi")
    path = str(tmp / "ppi_test.parquet")
    n = 500
    pq.write_table(
        pa.table(
            {
                "seq1": [_rand_seq() for _ in range(n)],
                "seq2": [_rand_seq() for _ in range(n)],
                "extra_col": list(range(n)),
            }
        ),
        path,
    )
    return path


@pytest.fixture(scope="module")
def ordered_ppi_parquet(tmp_path_factory):
    """Ordered PPI parquet with deterministic sequence IDs and many row groups."""
    tmp = tmp_path_factory.mktemp("ordered_ppi")
    path = str(tmp / "ordered_ppi_test.parquet")
    n = 500
    pq.write_table(
        pa.table(
            {
                "seq1": [f"SEQ_A_{i:04d}" for i in range(n)],
                "seq2": [f"SEQ_B_{i:04d}" for i in range(n)],
            }
        ),
        path,
        row_group_size=50,
    )
    return path


@pytest.fixture(scope="module")
def cluster_parquet(tmp_path_factory):
    """200-row cluster parquet (10 groups × 20 seqs), sorted by group_id."""
    tmp = tmp_path_factory.mktemp("cluster")
    path = str(tmp / "cluster_test.parquet")
    rows = [
        {"sequence": _rand_seq(), "group_id": f"group_{g:03d}"}
        for g in range(10)
        for _ in range(20)
    ]
    rows.sort(key=lambda r: r["group_id"])
    pq.write_table(
        pa.table(
            {
                "sequence": [r["sequence"] for r in rows],
                "group_id": [r["group_id"] for r in rows],
            }
        ),
        path,
    )
    return path


@pytest.fixture(scope="module")
def label_parquet(tmp_path_factory):
    """156-row label parquet: 5 common families (30 each) + 3 rare (2 each), shuffled."""
    tmp = tmp_path_factory.mktemp("label")
    path = str(tmp / "label_test.parquet")
    rows = [
        {"sequence": _rand_seq(), "family_id": f"fam_{i:03d}"}
        for i in range(5)
        for _ in range(30)
    ] + [
        {"sequence": _rand_seq(), "family_id": f"rare_{i:03d}"}
        for i in range(3)
        for _ in range(2)
    ]
    random.shuffle(rows)
    pq.write_table(
        pa.table(
            {
                "sequence": [r["sequence"] for r in rows],
                "family_id": [r["family_id"] for r in rows],
            }
        ),
        path,
    )
    return path


# ---------------------------------------------------------------------------
# _load_ppi_pair_dataset
# ---------------------------------------------------------------------------


class TestLoadPpiPairDataset:
    def test_loads_all_pairs(self, ppi_parquet):
        ds = _load_ppi_pair_dataset([ppi_parquet])
        assert len(ds) == 500

    def test_column_names(self, ppi_parquet):
        ds = _load_ppi_pair_dataset([ppi_parquet])
        assert set(ds.column_names) == {"sentence_0", "sentence_1"}

    def test_no_extra_columns(self, ppi_parquet):
        """extra_col from the source parquet must not appear in output."""
        ds = _load_ppi_pair_dataset([ppi_parquet])
        assert "extra_col" not in ds.column_names

    def test_sequences_are_strings(self, ppi_parquet):
        ds = _load_ppi_pair_dataset([ppi_parquet])
        assert isinstance(ds[0]["sentence_0"], str)
        assert isinstance(ds[0]["sentence_1"], str)

    def test_respects_max_pairs_cap(self, ppi_parquet):
        """When cap < total rows, generator path is used."""
        ds = _load_ppi_pair_dataset([ppi_parquet], max_pairs=100)
        assert len(ds) == 100

    def test_cap_equal_to_total_uses_fast_path(self, ppi_parquet):
        """cap == total rows should use the fast parquet path (still 500 rows)."""
        ds = _load_ppi_pair_dataset([ppi_parquet], max_pairs=500)
        assert len(ds) == 500

    def test_cap_larger_than_total_uses_fast_path(self, ppi_parquet):
        ds = _load_ppi_pair_dataset([ppi_parquet], max_pairs=9999)
        assert len(ds) == 500

    def test_capped_sampling_avoids_prefix_bias(self, ordered_ppi_parquet):
        ds = _load_ppi_pair_dataset([ordered_ppi_parquet], max_pairs=60)
        sampled_indices = [int(row["sentence_0"].split("_")[-1]) for row in ds]

        assert len(sampled_indices) == 60
        assert sampled_indices != list(range(60))
        assert max(sampled_indices) >= 100

    def test_capped_sampling_is_deterministic(self, ordered_ppi_parquet):
        ds1 = _load_ppi_pair_dataset([ordered_ppi_parquet], max_pairs=60)
        ds2 = _load_ppi_pair_dataset([ordered_ppi_parquet], max_pairs=60)

        assert ds1["sentence_0"] == ds2["sentence_0"]
        assert ds1["sentence_1"] == ds2["sentence_1"]


# ---------------------------------------------------------------------------
# _build_pair_dataset
# ---------------------------------------------------------------------------


class TestBuildPairDataset:
    def test_generates_pairs(self, cluster_parquet):
        ds = _build_pair_dataset(
            [cluster_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=0,
        )
        # 10 groups × C(20, 2) = 10 × 190 = 1900 pairs
        assert len(ds) == 1900
        assert set(ds.column_names) == {"sentence_0", "sentence_1"}

    def test_respects_max_pairs(self, cluster_parquet):
        ds = _build_pair_dataset(
            [cluster_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=100,
        )
        assert len(ds) == 100

    def test_respects_max_pairs_per_cluster(self, cluster_parquet):
        ds = _build_pair_dataset(
            [cluster_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=5,
            max_pairs=0,
        )
        # 10 groups × C(5, 2) = 10 × 10 = 100 pairs
        assert len(ds) == 100

    def test_sequences_are_strings(self, cluster_parquet):
        ds = _build_pair_dataset(
            [cluster_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=5,
            max_pairs=0,
        )
        assert isinstance(ds[0]["sentence_0"], str)
        assert isinstance(ds[0]["sentence_1"], str)


# ---------------------------------------------------------------------------
# _build_label_dataset
# ---------------------------------------------------------------------------


class TestBuildLabelDataset:
    def test_basic_load(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=1,
        )
        # 5×30 + 3×2 = 156 total rows; all labels pass min_count=1
        assert len(ds) == 156
        assert set(ds.column_names) == {"sentence", "label"}

    def test_filters_rare_labels(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=10,
        )
        # Only the 5 families with 30 samples survive; 3×2 rare families dropped
        assert len(ds) == 150
        assert ds.features["label"].num_classes == 5

    def test_caps_samples_per_label(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=1,
            max_samples_per_label=10,
        )
        # 5×10 + 3×2 = 56   (rare families have only 2, so capped at 2)
        assert len(ds) == 56

    def test_combined_filter_and_cap(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=10,
            max_samples_per_label=5,
        )
        # 5 labels pass filter, each capped at 5 → 25 rows
        assert len(ds) == 25
        assert ds.features["label"].num_classes == 5

    def test_respects_max_rows(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=50,
            min_label_count=1,
        )
        assert len(ds) <= 50

    def test_labels_are_class_encoded(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=1,
        )
        assert hasattr(ds.features["label"], "num_classes")
        assert ds.features["label"].num_classes >= 1

    def test_raises_on_no_valid_labels(self, tmp_path):
        """All labels are singletons, min_label_count=10 → ValueError."""
        path = str(tmp_path / "singletons.parquet")
        pq.write_table(
            pa.table(
                {
                    "sequence": [_rand_seq() for _ in range(5)],
                    "family_id": [f"unique_{i}" for i in range(5)],
                }
            ),
            path,
        )
        with pytest.raises(ValueError, match="No labels"):
            _build_label_dataset(
                [path],
                "sequence",
                "family_id",
                max_rows=0,
                min_label_count=10,
            )

    def test_sentences_are_strings(self, label_parquet):
        ds = _build_label_dataset(
            [label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=1,
        )
        assert isinstance(ds[0]["sentence"], str)


# ---------------------------------------------------------------------------
# Benchmarks (opt-in: pytest -m benchmark -s)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def large_label_parquet(tmp_path_factory):
    """100K-row label parquet: 200 common (500 each) + 50 rare (3 each)."""
    tmp = tmp_path_factory.mktemp("bench_label")
    path = str(tmp / "bench_label.parquet")
    seqs, labels = [], []
    for fam in range(200):
        fid = f"fam_{fam:04d}"
        for _ in range(500):
            seqs.append(_rand_seq(120))
            labels.append(fid)
    for i in range(50):
        for _ in range(3):
            seqs.append(_rand_seq(120))
            labels.append(f"rare_{i}")
    combined = list(zip(seqs, labels))
    random.shuffle(combined)
    seqs, labels = zip(*combined)
    pq.write_table(pa.table({"sequence": list(seqs), "family_id": list(labels)}), path)
    return path


@pytest.fixture(scope="module")
def large_ppi_parquet(tmp_path_factory):
    """200K-pair PPI parquet."""
    tmp = tmp_path_factory.mktemp("bench_ppi")
    path = str(tmp / "bench_ppi.parquet")
    n = 200_000
    pq.write_table(
        pa.table(
            {
                "seq1": [_rand_seq(120) for _ in range(n)],
                "seq2": [_rand_seq(120) for _ in range(n)],
            }
        ),
        path,
    )
    return path


@pytest.mark.benchmark
class TestBenchmark:
    def test_label_dataset_filter_only(self, large_label_parquet):
        t0 = time.perf_counter()
        ds = _build_label_dataset(
            [large_label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=5,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  build_label (filter only, 100K): {elapsed:.3f}s → {len(ds):,} rows")
        assert len(ds) > 0

    def test_label_dataset_filter_and_cap(self, large_label_parquet):
        t0 = time.perf_counter()
        ds = _build_label_dataset(
            [large_label_parquet],
            "sequence",
            "family_id",
            max_rows=0,
            min_label_count=5,
            max_samples_per_label=100,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  build_label (filter+cap, 100K): {elapsed:.3f}s → {len(ds):,} rows")
        assert len(ds) > 0

    def test_ppi_no_cap(self, large_ppi_parquet):
        t0 = time.perf_counter()
        ds = _load_ppi_pair_dataset([large_ppi_parquet])
        elapsed = time.perf_counter() - t0
        print(f"\n  load_ppi (no cap, 200K): {elapsed:.3f}s → {len(ds):,} rows")
        assert len(ds) == 200_000

    def test_ppi_with_cap(self, large_ppi_parquet):
        t0 = time.perf_counter()
        ds = _load_ppi_pair_dataset([large_ppi_parquet], max_pairs=50_000)
        elapsed = time.perf_counter() - t0
        print(f"\n  load_ppi (cap=50K, 200K total): {elapsed:.3f}s → {len(ds):,} rows")
        assert len(ds) == 50_000


# ---------------------------------------------------------------------------
# _build_pair_dataset – hard negatives
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster_with_hard_negatives_parquet(tmp_path_factory):
    """Cluster parquet with a hard_negative column.

    5 groups × 10 seqs. Some rows have hard negatives, some are null.
    """
    tmp = tmp_path_factory.mktemp("hard_neg")
    path = str(tmp / "hn_test.parquet")
    seqs, groups, hard_negatives = [], [], []
    for g in range(5):
        for i in range(10):
            seqs.append(_rand_seq(40))
            groups.append(f"grp_{g:02d}")
            hard_negatives.append(_rand_seq(40) if i < 6 else None)
    table = pa.table(
        {
            "sequence": seqs,
            "group_id": groups,
            "hard_negative": hard_negatives,
        }
    )
    pq.write_table(table, path)
    return path


class TestBuildPairDatasetHardNegatives:
    def test_hard_negatives_disabled_ignores_columns(
        self, cluster_with_hard_negatives_parquet
    ):
        """With hard_negatives=False, output only has sentence_0/sentence_1."""
        ds = _build_pair_dataset(
            [cluster_with_hard_negatives_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=0,
            hard_negatives=False,
        )
        assert set(ds.column_names) == {"sentence_0", "sentence_1"}
        assert len(ds) > 0

    def test_hard_negatives_enabled_adds_columns(
        self, cluster_with_hard_negatives_parquet
    ):
        """With hard_negatives=True, output includes sentence_2."""
        ds = _build_pair_dataset(
            [cluster_with_hard_negatives_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=0,
            hard_negatives=True,
        )
        assert "sentence_0" in ds.column_names
        assert "sentence_1" in ds.column_names
        assert "sentence_2" in ds.column_names
        assert "sentence_3" not in ds.column_names

    def test_hard_negatives_rows_have_anchor_content(
        self, cluster_with_hard_negatives_parquet
    ):
        """Each row should have non-empty sentence_0 and sentence_1."""
        ds = _build_pair_dataset(
            [cluster_with_hard_negatives_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=0,
            hard_negatives=True,
        )
        for i in range(min(10, len(ds))):
            assert len(ds[i]["sentence_0"]) > 0
            assert len(ds[i]["sentence_1"]) > 0

    def test_hard_negatives_respects_max_pairs(
        self, cluster_with_hard_negatives_parquet
    ):
        ds = _build_pair_dataset(
            [cluster_with_hard_negatives_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=20,
            hard_negatives=True,
        )
        assert len(ds) == 20

    def test_no_hard_neg_columns_with_flag_true(self, cluster_parquet):
        """When parquet has no hard_negative_* columns, hard_negatives=True still works (just 2-col output)."""
        ds = _build_pair_dataset(
            [cluster_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=5,
            max_pairs=0,
            hard_negatives=True,
        )
        assert set(ds.column_names) == {"sentence_0", "sentence_1"}
        assert len(ds) > 0

    def test_all_rows_materialize_with_consistent_schema(
        self, cluster_with_hard_negatives_parquet
    ):
        """Every row must have all declared columns (no missing keys)."""
        ds = _build_pair_dataset(
            [cluster_with_hard_negatives_parquet],
            "sequence",
            "group_id",
            max_pairs_per_cluster=50,
            max_pairs=0,
            hard_negatives=True,
        )
        expected_cols = set(ds.column_names)
        for i in range(len(ds)):
            assert set(ds[i].keys()) == expected_cols
            for col in expected_cols:
                assert isinstance(ds[i][col], str)
