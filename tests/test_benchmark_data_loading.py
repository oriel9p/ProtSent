"""Tests for benchmark dataset loading edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import protein_benchmark_suite as benchmark_suite
from benchmark_tasks import TASKS
from protein_benchmark_suite import prepare_data


def test_prepare_data_supports_stage_column_split(monkeypatch) -> None:
    """Stage-column datasets should be split without random auto-splitting."""
    dataset = datasets.Dataset.from_dict(
        {
            "protein": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
            "label": [0.1, 0.2, 0.3, 0.4, 0.5],
            "stage": ["train", "test", "train", "valid", "test"],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["beta_lactamase_peer"]
    train_seqs, train_labels, test_seqs, test_labels, extra_data, metadata = (
        prepare_data(cfg, eval_split="test")
    )

    assert train_seqs == ["AAAA", "CCCC"]
    assert train_labels == [0.1, 0.3]
    assert test_seqs == ["BBBB", "EEEE"]
    assert test_labels == [0.2, 0.5]
    assert extra_data is None
    assert metadata["eval_strategy"] == "test_split_column"


def test_prepare_data_compacts_whitespace_delimited_sequences(monkeypatch) -> None:
    """Whitespace-delimited amino-acid sequences should be compacted before embedding."""
    train_dataset = datasets.Dataset.from_dict(
        {
            "prot_seq": ["A A A A", "B B B B"],
            "localization": [0, 1],
        }
    )
    test_dataset = datasets.Dataset.from_dict(
        {
            "prot_seq": ["C C C C", "D D D D"],
            "localization": [1, 0],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["binary_subcellular_localization"]
    train_seqs, train_labels, test_seqs, test_labels, extra_data, metadata = (
        prepare_data(cfg, eval_split="test")
    )

    assert train_seqs == ["AAAA", "BBBB"]
    assert test_seqs == ["CCCC", "DDDD"]
    assert train_labels == [0, 1]
    assert test_labels == [1, 0]
    assert extra_data is None
    assert metadata["resolved_eval_split"] == "test"


def test_prepare_data_prefers_validation_split_when_available(monkeypatch) -> None:
    """Validation mode should use the explicit validation split if present."""
    train_dataset = datasets.Dataset.from_dict(
        {
            "seq": ["AAAA", "BBBB", "CCCC"],
            "label": [1, 0, 1],
        }
    )
    validation_dataset = datasets.Dataset.from_dict(
        {
            "seq": ["DDDD", "EEEE"],
            "label": [0, 1],
        }
    )
    test_dataset = datasets.Dataset.from_dict(
        {
            "seq": ["FFFF"],
            "label": [1],
        }
    )
    dataset_dict = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
    )

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["metal_ion_binding"]
    train_seqs, train_labels, eval_seqs, eval_labels, extra_data, metadata = (
        prepare_data(
            cfg,
            eval_split="validation",
        )
    )

    assert train_seqs == ["AAAA", "BBBB", "CCCC"]
    assert train_labels == [1, 0, 1]
    assert eval_seqs == ["DDDD", "EEEE"]
    assert eval_labels == [0, 1]
    assert extra_data is None
    assert metadata["resolved_eval_split"] == "validation"
    assert metadata["eval_strategy"] == "validation_split"
    assert metadata["cv_fallback"] is False


def test_prepare_data_uses_cv_fallback_when_validation_missing(monkeypatch) -> None:
    """Validation mode should switch to CV when no validation split exists."""
    train_dataset = datasets.Dataset.from_dict(
        {
            "seq": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    test_dataset = datasets.Dataset.from_dict(
        {
            "seq": ["GGGG", "HHHH"],
            "label": [1, 0],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["metal_ion_binding"]
    train_seqs, train_labels, eval_seqs, eval_labels, extra_data, metadata = (
        prepare_data(
            cfg,
            eval_split="validation",
        )
    )

    assert train_seqs == ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]
    assert train_labels == [1, 0, 1, 0, 1, 0]
    assert eval_seqs is None
    assert eval_labels is None
    assert extra_data is None
    assert metadata["resolved_eval_split"] == "validation"
    assert metadata["eval_strategy"] == "validation_cv4_train"
    assert metadata["cv_fallback"] is True


def test_prepare_data_stage_column_uses_validation_rows(monkeypatch) -> None:
    """Validation mode should read split-column validation rows when configured."""
    dataset = datasets.Dataset.from_dict(
        {
            "protein": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
            "label": [0.1, 0.2, 0.3, 0.4, 0.5],
            "stage": ["train", "test", "train", "valid", "test"],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["beta_lactamase_peer"]
    train_seqs, train_labels, eval_seqs, eval_labels, extra_data, metadata = (
        prepare_data(
            cfg,
            eval_split="validation",
        )
    )

    assert train_seqs == ["AAAA", "CCCC"]
    assert train_labels == [0.1, 0.3]
    assert eval_seqs == ["DDDD"]
    assert eval_labels == [0.4]
    assert extra_data is None
    assert metadata["eval_strategy"] == "validation_split_column"


def test_prepare_data_respects_configured_benchmark_seed(monkeypatch) -> None:
    """Changing the benchmark seed should change sampled rows."""

    train_dataset = datasets.Dataset.from_dict(
        {
            "seq": [f"TR{i}" for i in range(8)],
            "label": [i % 2 for i in range(8)],
        }
    )
    test_dataset = datasets.Dataset.from_dict(
        {
            "seq": [f"TE{i}" for i in range(8)],
            "label": [(i + 1) % 2 for i in range(8)],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["metal_ion_binding"]

    monkeypatch.setattr(benchmark_suite, "BENCHMARK_SEED", 7)
    train_seqs_seed_7, _, eval_seqs_seed_7, _, _, _ = prepare_data(
        cfg,
        max_samples=3,
        eval_split="test",
    )

    monkeypatch.setattr(benchmark_suite, "BENCHMARK_SEED", 11)
    train_seqs_seed_11, _, eval_seqs_seed_11, _, _, _ = prepare_data(
        cfg,
        max_samples=3,
        eval_split="test",
    )

    assert train_seqs_seed_7 != train_seqs_seed_11
    assert eval_seqs_seed_7 != eval_seqs_seed_11


def test_parse_args_accepts_seed(monkeypatch) -> None:
    """CLI parsing should expose the benchmark seed override."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["protein_benchmark_suite.py", "--seed", "17"],
    )

    args = benchmark_suite.parse_args()

    assert args.seed == 17


def test_parse_args_accepts_seed_list(monkeypatch) -> None:
    """CLI parsing should expose optional multi-seed benchmark execution."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["protein_benchmark_suite.py", "--seed_list", "17,19,23"],
    )

    args = benchmark_suite.parse_args()

    assert args.seed_list == "17,19,23"
