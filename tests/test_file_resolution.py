"""Unit tests for file path resolution helpers in protein_pipeline.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_pipeline import _expand_paths


def test_expand_paths_resolves_data_dir_fallback(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    expected = data_dir / "pfam_sorted.parquet"
    expected.touch()

    resolved = _expand_paths(["pfam_sorted.parquet"], data_dir=str(data_dir))

    assert resolved == [str(expected)]


def test_expand_paths_raises_for_unresolved_entry(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        _expand_paths(["missing.parquet"], data_dir=str(tmp_path / "data"))

    msg = str(exc.value)
    assert "Some --files entries could not be resolved" in msg
    assert "missing.parquet" in msg


def test_expand_paths_raises_when_any_entry_is_unresolved(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    valid = data_dir / "afdb_sorted.parquet"
    valid.touch()

    with pytest.raises(FileNotFoundError) as exc:
        _expand_paths(
            ["afdb_sorted.parquet", "data/data/pfam_sorted.parquet"],
            data_dir=str(data_dir),
        )

    msg = str(exc.value)
    assert "Some --files entries could not be resolved" in msg
    assert "data/data/pfam_sorted.parquet" in msg


def test_expand_paths_raises_for_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_parquets"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc:
        _expand_paths([str(empty_dir)], data_dir=str(tmp_path / "data"))

    msg = str(exc.value)
    assert "directory contains no .parquet files" in msg
    assert str(empty_dir) in msg
