"""Regression tests for ProteinGym DMS preparation.

These tests are offline and mock Hugging Face dataset loading.
"""

import gzip
import subprocess
import sys
from pathlib import Path
from typing import cast

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_prep import (
    DataPrep,
    _download_file,
    _load_afdb_foldseek_map_lazy,
)


def _mock_load_dataset(repo_id: str, name: str, split: str) -> dict[str, list[object]]:
    assert repo_id == "OATML-Markslab/ProteinGym_v1"
    assert split == "train"

    if name == "DMS_substitutions":
        return {
            "DMS_id": ["GB1_overlap", "GB1_overlap", "ASSAY_A", "ASSAY_A"],
            "mutated_sequence": ["AAA", "AAB", "AAC", "AAD"],
            "target_seq": ["WT1", "WT1", "WT2", "WT2"],
            "DMS_score": [0.1, 0.9, 0.2, 0.8],
        }

    if name == "DMS_indels":
        return {
            "DMS_id": ["INDEL_A", "INDEL_A", "INDEL_A"],
            "mutated_sequence": ["IAA", "IAB", "IAC"],
            "target_seq": ["WT3", "WT3", "WT3"],
            "DMS_score": [0.4, 0.7, 0.6],
        }

    if name == "clinical_substitutions":
        return {
            "mutated_sequence": ["CAA", "CAB", "CAC"],
            "target_seq": ["WT4", "WT4", "WT5"],
            "annotation": ["Pathogenic", "Benign", "Unknown"],
            "protein_id": ["P1", "P1", "P2"],
        }

    if name == "clinical_indels":
        return {
            "mutated_sequence": ["CIA", "CIB"],
            "target_seq": ["WT6", "WT6"],
            "annotation": ["0", "1"],
            "protein_id": ["P3", "P3"],
        }

    raise AssertionError(name)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_load_afdb_foldseek_map_lazy_filters_flags(tmp_path: Path) -> None:
    tsv_path = tmp_path / "map.tsv.gz"
    rows = [
        "repA\tentryA\t1\t9606\n",
        "repA\tentryB\t2\t9606\n",
        "repB\tentryC\t1\t3702\n",
    ]
    with gzip.open(tsv_path, "wt") as f:
        f.writelines(rows)

    flag1_df = cast(
        pl.DataFrame,
        _load_afdb_foldseek_map_lazy(tsv_path, allowed_clu_flags=(1,)).collect(),
    )
    assert flag1_df.shape[0] == 2
    assert set(flag1_df["entry_id"].to_list()) == {"entryA", "entryC"}

    flag12_df = cast(
        pl.DataFrame,
        _load_afdb_foldseek_map_lazy(tsv_path, allowed_clu_flags=(1, 2)).collect(),
    )
    assert flag12_df.shape[0] == 3
    assert set(flag12_df["entry_id"].to_list()) == {"entryA", "entryB", "entryC"}


def test_download_file_replaces_zero_byte_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dest_path = tmp_path / "download.bin"
    dest_path.write_bytes(b"")

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("-O") + 1])
        output_path.write_bytes(b"payload")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("data_prep.subprocess.run", fake_run)

    _download_file("https://example.test/download.bin", dest_path)

    assert dest_path.read_bytes() == b"payload"


def test_prep_dms_deduplicates_known_overlap_prefixes_by_default(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms()

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")

    # 2 non-overlap DMS substitution rows + 3 DMS indel rows + 4 clinical rows.
    assert len(result_df) == 9
    assert "WT1" not in result_df["sentence_1"].to_list()
    assert result_df["sentence_1"].n_unique() == 4
    assert set(result_df.columns) == {"sentence_0", "sentence_1", "score"}
    min_score = cast(float, result_df["score"].min())
    max_score = cast(float, result_df["score"].max())
    assert min_score >= 0.0
    assert max_score <= 1.0


def test_prep_dms_keeps_full_train_rows_when_dedup_disabled(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True, deduplicate_benchmarks=False)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")

    # 4 DMS substitution rows + 3 DMS indel rows + 4 clinical labeled rows.
    assert len(result_df) == 11
    assert "WT1" in result_df["sentence_1"].to_list()


def test_prep_dms_adds_intra_pairs_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True, intra_pairs=True, intra_pairs_per_assay=2)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")

    # Base rows 9 + 1 intra pair from ASSAY_A + 2 from INDEL_A.
    assert len(result_df) == 12
    min_score = cast(float, result_df["score"].min())
    max_score = cast(float, result_df["score"].max())
    assert min_score >= 0.0
    assert max_score <= 1.0


def test_prep_dms_writes_unique_simcse_sequences(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True)

    simcse_df = pl.read_parquet(data_dir / "dms_sequences.parquet")

    assert simcse_df.columns == ["sequence"]
    assert simcse_df["sequence"].n_unique() == len(simcse_df)
    assert set(simcse_df["sequence"].to_list()) >= {
        "AAC",
        "IAC",
        "WT2",
        "WT6",
    }


def test_sort_and_save_supports_afdb50_ordering_and_structural_grouping(
    data_dir: Path,
) -> None:
    prep = DataPrep(str(data_dir))
    df = pl.DataFrame(
        {
            "sequence": ["SEQ_A", "SEQ_B", "SEQ_C", "SEQ_D", "SEQ_E"],
            "cluster_id": ["struct_1", "struct_1", "struct_2", "struct_2", "singleton"],
            "afdb50_cluster_id": ["af50_b", "af50_a", "af50_a", "af50_b", "af50_c"],
        }
    )

    prep._sort_and_save(
        df,
        "afdb_sorted.parquet",
        hierarchical=False,
        sort_cols="afdb50_cluster_id",
        shuffle_before_sort=True,
        drop_redundant_cluster_id=True,
    )

    result_df = pl.read_parquet(data_dir / "afdb_sorted.parquet")

    assert set(result_df.columns) == {
        "sequence",
        "afdb50_cluster_id",
        "group_id",
    }

    cluster_sizes = result_df.group_by("group_id").len().sort("group_id")
    assert cast(int, cluster_sizes["len"].min()) >= 2
    assert "singleton" not in result_df["group_id"].to_list()

    afdb50_values = result_df["afdb50_cluster_id"].to_list()
    assert afdb50_values == sorted(afdb50_values)


def test_prep_dms_continuous_scores_preserve_expected_polarity(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")
    assay_rows = result_df.filter(pl.col("sentence_1") == "WT2")

    score_by_mutant = dict(
        zip(
            assay_rows["sentence_0"].to_list(),
            assay_rows["score"].to_list(),
            strict=True,
        )
    )

    assert cast(float, score_by_mutant["AAD"]) > cast(float, score_by_mutant["AAC"])


def test_prep_dms_clinical_scores_align_text_and_numeric_labels(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("data_prep.load_dataset", _mock_load_dataset)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")

    text_rows = result_df.filter(pl.col("sentence_1") == "WT4")
    text_score_by_mutant = dict(
        zip(
            text_rows["sentence_0"].to_list(),
            text_rows["score"].to_list(),
            strict=True,
        )
    )

    numeric_rows = result_df.filter(pl.col("sentence_1") == "WT6")
    numeric_score_by_mutant = dict(
        zip(
            numeric_rows["sentence_0"].to_list(),
            numeric_rows["score"].to_list(),
            strict=True,
        )
    )

    assert cast(float, text_score_by_mutant["CAB"]) > cast(
        float, text_score_by_mutant["CAA"]
    )
    assert cast(float, numeric_score_by_mutant["CIA"]) > cast(
        float, numeric_score_by_mutant["CIB"]
    )


def test_prep_dms_drops_supervised_test_fold_for_large_groups(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    def _mock_large_assay(
        repo_id: str,
        name: str,
        split: str,
    ) -> dict[str, list[object]]:
        assert repo_id == "OATML-Markslab/ProteinGym_v1"
        assert split == "train"

        if name == "DMS_substitutions":
            return {
                "DMS_id": ["ASSAY_BIG"] * 10,
                "mutated_sequence": [f"AA{i}" for i in range(10)],
                "target_seq": ["WT_BIG"] * 10,
                "DMS_score": [float(i) for i in range(10)],
            }
        if name == "DMS_indels":
            return {
                "DMS_id": ["INDEL_SMALL"] * 3,
                "mutated_sequence": ["I1", "I2", "I3"],
                "target_seq": ["WT_I"] * 3,
                "DMS_score": [0.1, 0.2, 0.3],
            }
        if name == "clinical_substitutions":
            return {
                "mutated_sequence": ["C1", "C2"],
                "target_seq": ["WT_C", "WT_C"],
                "annotation": ["Benign", "Pathogenic"],
                "protein_id": ["P1", "P1"],
            }
        if name == "clinical_indels":
            return {
                "mutated_sequence": ["CI1"],
                "target_seq": ["WT_CI"],
                "annotation": ["1"],
                "protein_id": ["P2"],
            }
        raise AssertionError(name)

    monkeypatch.setattr("data_prep.load_dataset", _mock_large_assay)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True, deduplicate_benchmarks=False)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")
    wt_big_rows = result_df.filter(pl.col("sentence_1") == "WT_BIG")

    # For group size 10, supervised benchmark uses 80/20 split; 2 rows must be dropped.
    assert len(wt_big_rows) == 8


def test_prep_dms_prefers_explicit_split_column_for_fold_dropping(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> None:
    def _mock_explicit_split(
        repo_id: str,
        name: str,
        split: str,
    ) -> dict[str, list[object]]:
        assert repo_id == "OATML-Markslab/ProteinGym_v1"
        assert split == "train"

        if name == "DMS_substitutions":
            return {
                "DMS_id": ["ASSAY_STAGE"] * 4,
                "mutated_sequence": ["S1", "S2", "S3", "S4"],
                "target_seq": ["WT_STAGE"] * 4,
                "DMS_score": [0.1, 0.2, 0.3, 0.4],
                "stage": ["train", "train", "test", "test"],
            }
        if name == "DMS_indels":
            return {
                "DMS_id": ["I_STAGE"],
                "mutated_sequence": ["I1"],
                "target_seq": ["WT_I"],
                "DMS_score": [0.5],
                "stage": ["train"],
            }
        if name == "clinical_substitutions":
            return {
                "mutated_sequence": ["C1"],
                "target_seq": ["WT_C"],
                "annotation": ["Benign"],
                "protein_id": ["P1"],
                "stage": ["train"],
            }
        if name == "clinical_indels":
            return {
                "mutated_sequence": ["CI1"],
                "target_seq": ["WT_CI"],
                "annotation": ["1"],
                "protein_id": ["P2"],
                "stage": ["test"],
            }
        raise AssertionError(name)

    monkeypatch.setattr("data_prep.load_dataset", _mock_explicit_split)

    prep = DataPrep(str(data_dir))
    prep.prep_dms(force=True, deduplicate_benchmarks=False)

    result_df = pl.read_parquet(data_dir / "dms_cosent.parquet")
    stage_rows = result_df.filter(pl.col("sentence_1") == "WT_STAGE")

    # Explicit test rows are removed even though group size is <10.
    assert len(stage_rows) == 2


# The fixed Delta-S regime this used to assert on is gone; hard negatives are
# now verified against the family HMM. See tests/test_pfam_hard_negatives.py.
