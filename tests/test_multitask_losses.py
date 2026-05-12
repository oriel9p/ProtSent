"""Regression tests for multi-task helpers.

These tests cover DMS CoSENT data loading and batch-size resolution.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_pipeline import (
    _estimate_multidataset_steps_per_epoch,
    _load_dms_dataset,
    _resolve_dms_train_batch_size,
)


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def test_resolve_dms_batch_size_uses_target_on_single_gpu() -> None:
    """Single-GPU runs should use the DMS target batch size directly."""

    train_dataset = {
        "a": _SizedDataset(1200),
        "b": _SizedDataset(800),
    }

    resolved = _resolve_dms_train_batch_size(
        base_batch_size=1024,
        dms_batch_size=128,
        mnrl_mini_batch_size=256,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        world_size=1,
        sampler_mode="round_robin",
        drop_last=True,
    )

    assert resolved == 128


def test_resolve_dms_batch_size_ddp_round_robin_adjusts_for_divisibility() -> None:
    """DDP round-robin should pick a size that yields divisible global batches."""

    train_dataset = {
        "afdb": _SizedDataset(6000),
        "stringdb": _SizedDataset(6000),
        "simcse": _SizedDataset(12000),
        "dms_cosent": _SizedDataset(12000),
    }

    resolved = _resolve_dms_train_batch_size(
        base_batch_size=1024,
        dms_batch_size=128,
        mnrl_mini_batch_size=256,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        world_size=3,
        sampler_mode="round_robin",
        drop_last=True,
    )

    _, global_batches, _ = _estimate_multidataset_steps_per_epoch(
        train_dataset=train_dataset,  # type: ignore[arg-type]
        per_device_batch_size=resolved,
        world_size=3,
        sampler_mode="round_robin",
        drop_last=True,
    )

    assert 128 <= resolved <= 1024
    assert global_batches % 3 == 0


def test_load_dms_dataset_respects_max_rows(tmp_path: Path) -> None:
    """Smoke runs should be able to cap DMS construction without loading all rows."""

    file_path = tmp_path / "dms.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_0": [f"AAA{i}" for i in range(8)],
                "sentence_1": [f"BBB{i}" for i in range(8)],
                "score": [float(i) / 10.0 for i in range(8)],
                "unused": list(range(8)),
            }
        ),
        file_path,
    )

    dataset = _load_dms_dataset(str(file_path), max_rows=3)

    assert len(dataset) == 3
    assert set(dataset.column_names) == {"sentence_0", "sentence_1", "score"}
    assert dataset[0]["sentence_0"] == "AAA0"
    assert dataset[2]["score"] == pytest.approx(0.2)
