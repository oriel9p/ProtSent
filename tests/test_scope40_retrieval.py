"""Offline tests for the SCOPe-40 retrieval benchmark path."""

import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_comparison import get_best_metric_for_task
from benchmark_tasks import TASKS
from protein_benchmark_suite import (
    ResultTracker,
    effective_probe_type,
    evaluate_retrieval,
    prepare_data,
)


def test_evaluate_retrieval_handles_duplicate_embeddings() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.array(["fam_a", "fam_a", "fam_b", "fam_b"])

    metrics = evaluate_retrieval(embeddings, labels)

    assert metrics["Recall@1"] == 1.0
    assert metrics["Recall@10"] == 1.0
    assert metrics["Recall@30"] == 1.0


def test_evaluate_retrieval_returns_zero_without_family_matches() -> None:
    embeddings = np.eye(3, dtype=float)
    labels = np.array(["fam_a", "fam_b", "fam_c"])

    metrics = evaluate_retrieval(embeddings, labels, k_list=(1, 2))

    assert metrics == {"Recall@1": 0.0, "Recall@2": 0.0}


def test_prepare_data_retrieval_uses_single_gallery_split(monkeypatch) -> None:
    dataset = datasets.Dataset.from_dict(
        {
            "id": ["q1", "q2", "q3"],
            "sequence": ["AAAA", "BBBB", "CCCC"],
            "family": ["a.1.1.1", "a.1.1.1", "b.2.2.2"],
        }
    )
    dataset_dict = datasets.DatasetDict({"train": dataset})

    def _fake_load_dataset(*args, **kwargs):
        return dataset_dict

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = TASKS["scope40_retrieval"]
    train_seqs, train_labels, test_seqs, test_labels, extra_data, metadata = (
        prepare_data(cfg)
    )

    assert train_seqs == ["AAAA", "BBBB", "CCCC"]
    assert train_seqs == test_seqs
    assert train_labels == ["a.1.1.1", "a.1.1.1", "b.2.2.2"]
    assert train_labels == test_labels
    assert extra_data is None
    assert metadata["eval_strategy"] == "retrieval_unchanged"


def test_get_best_metric_for_task_prefers_recall_at_10() -> None:
    # Test that Recall@10 is selected when higher-priority metrics are absent
    row = pd.Series({"Recall@1": 0.4, "Recall@10": 0.7, "Recall@30": 0.9})

    metric_name, metric_value = get_best_metric_for_task(row)

    assert metric_name == "Recall@10"
    assert metric_value == 0.7


def test_result_tracker_save_rounds_metrics_to_five_decimals(tmp_path) -> None:
    tracker = ResultTracker("unit/test-model")
    tracker.add(
        "Task A",
        {"AUC": 0.9217079755651822, "AP": 0.9300937801243615, "MSE": 0.038045919},
        samples=1000,
    )

    output_path = tracker.save(str(tmp_path))

    assert output_path is not None
    saved_df = pd.read_csv(output_path)
    assert saved_df.loc[0, "AUC"] == pytest.approx(0.92171)
    assert saved_df.loc[0, "AP"] == pytest.approx(0.93009)
    assert saved_df.loc[0, "MSE"] == pytest.approx(0.03805)


def test_result_tracker_preserves_distinct_probe_rows(tmp_path) -> None:
    tracker = ResultTracker("unit/test-model")
    tracker.add("Task A", {"AUC": 0.9}, samples=1000, probe="linear")
    tracker.add("Task A", {"AUC": 0.8}, samples=1000, probe="histgb")

    output_path = tracker.save(str(tmp_path))

    assert output_path is not None
    saved_df = pd.read_csv(output_path)
    assert set(saved_df["Probe"]) == {"linear", "histgb"}
    assert len(saved_df) == 2


def test_result_tracker_preserves_distinct_seed_rows(tmp_path) -> None:
    """Rows for different benchmark seeds should coexist in one results CSV."""

    tracker = ResultTracker("unit/test-model")
    tracker.add("Task A", {"AUC": 0.9}, samples=1000, benchmark_seed=1)
    tracker.add("Task A", {"AUC": 0.8}, samples=1000, benchmark_seed=73)

    output_path = tracker.save(str(tmp_path))

    assert output_path is not None
    saved_df = pd.read_csv(output_path)
    assert set(saved_df["BenchmarkSeed"].astype(str)) == {"1", "73"}
    assert len(saved_df) == 2


def test_effective_probe_type_defaults_to_linear_on_ignored_modes() -> None:
    assert effective_probe_type(TASKS["scope40_retrieval"], "knn") == "linear"
    assert effective_probe_type(TASKS["go_mf"], "histgb") == "linear"
    assert (
        effective_probe_type(TASKS["proteingym_dms_substitutions_zeroshot"], "knn")
        == "linear"
    )
    assert effective_probe_type(TASKS["solubility"], "knn") == "knn"
