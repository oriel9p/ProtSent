"""Lightweight offline tests — no GPU, no network required.

Covers:
  - All TASKS entries have valid configurations
  - TaskConfig rejects invalid problem_type
  - METRIC_PRIORITY is consistent
  - find_result_file raises for non-existent paths
  - FAST_TASKS are all valid keys in TASKS

Run:
    pytest tests/test_benchmark_tasks.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_comparison import METRIC_PRIORITY, compare_benchmarks, find_result_file
from benchmark_tasks import (
    DEFAULT_TASKS,
    FAST_TASKS,
    TASKS,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# TaskConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,cfg", list(TASKS.items()), ids=list(TASKS.keys()))
def test_task_has_valid_problem_type(key, cfg):
    valid = {"binary", "multiclass", "multilabel", "regression", "retrieval"}
    assert cfg.problem_type in valid, (
        f"{key}: invalid problem_type '{cfg.problem_type}'"
    )


@pytest.mark.parametrize("key,cfg", list(TASKS.items()), ids=list(TASKS.keys()))
def test_task_has_non_empty_dataset(key, cfg):
    assert cfg.dataset, f"{key}: empty dataset"


@pytest.mark.parametrize("key,cfg", list(TASKS.items()), ids=list(TASKS.keys()))
def test_task_has_non_empty_input_map(key, cfg):
    assert cfg.input_map, f"{key}: empty input_map"


@pytest.mark.parametrize("key,cfg", list(TASKS.items()), ids=list(TASKS.keys()))
def test_task_has_non_empty_label_col(key, cfg):
    assert cfg.label_col, f"{key}: empty label_col"


def test_taskconfig_rejects_invalid_problem_type():
    with pytest.raises(ValueError, match="problem_type"):
        TaskConfig(
            name="test",
            dataset="test/test",
            input_map={"seq": "seq"},
            label_col="label",
            problem_type="invalid_type",
            main_metric="AUC",
        )


def test_tasks_minimum_count():
    """Sanity: at least 15 tasks defined."""
    assert len(TASKS) >= 15


# ---------------------------------------------------------------------------
# FAST_TASKS
# ---------------------------------------------------------------------------


def test_fast_tasks_are_valid_keys():
    for key in FAST_TASKS:
        assert key in TASKS, f"FAST_TASKS contains unknown key: '{key}'"


def test_fast_tasks_include_requested_new_benchmarks() -> None:
    assert "beta_lactamase_peer" in FAST_TASKS
    assert "binary_subcellular_localization" in TASKS


# ---------------------------------------------------------------------------
# METRIC_PRIORITY
# ---------------------------------------------------------------------------


def test_metric_priority_non_empty():
    assert len(METRIC_PRIORITY) > 0


def test_metric_priority_contains_expected_metrics():
    for expected in ("AUC", "Accuracy", "Spearman", "F1_Macro", "Recall@10"):
        assert expected in METRIC_PRIORITY, f"'{expected}' missing from METRIC_PRIORITY"


def test_scope40_retrieval_is_registered_and_opt_in_only():
    cfg = TASKS["scope40_retrieval"]
    assert cfg.problem_type == "retrieval"
    assert cfg.main_metric == "Recall@10"
    assert "scope40_retrieval" not in FAST_TASKS
    assert "scope40_retrieval" not in DEFAULT_TASKS


def test_classification_tasks_prefer_auc_main_metric() -> None:
    auc_task_keys = [
        "ppi_bernett",
        "solubility",
        "peptide_hla",
        "metal_ion_binding",
        "material_production",
        "binary_subcellular_localization",
        "remote_homology",
        "subcellular_loc",
        "ec_classification",
        "antibiotic_resistance",
        "temperature_stability",
    ]
    for key in auc_task_keys:
        assert TASKS[key].main_metric == "AUC"


# ---------------------------------------------------------------------------
# find_result_file
# ---------------------------------------------------------------------------


def test_find_result_file_raises_for_nonexistent_model():
    with pytest.raises(FileNotFoundError):
        find_result_file("nonexistent_model_xyz_abc", output_dir="/tmp")


def test_find_result_file_raises_for_missing_dir():
    with pytest.raises(FileNotFoundError):
        find_result_file("some_model", output_dir="/nonexistent_dir_xyz_abc")


def test_find_result_file_direct_csv(tmp_path):
    """Direct CSV path should be returned as-is."""
    csv = tmp_path / "bench_test.csv"
    csv.write_text("Task,Accuracy\nsolubility,0.85\n")
    result = find_result_file(str(csv))
    assert result == csv


def test_compare_benchmarks_defaults_to_five_decimal_rounding(tmp_path) -> None:
    model1 = tmp_path / "bench_model1.csv"
    model2 = tmp_path / "bench_model2.csv"

    model1.write_text(
        "Model,Task,AUC\nmodel_1,Task A,0.876549\n",
        encoding="utf-8",
    )
    model2.write_text(
        "Model,Task,AUC\nmodel_2,Task A,0.812341\n",
        encoding="utf-8",
    )

    comparison_df = compare_benchmarks(str(model1), str(model2))

    assert comparison_df.loc[0, "Metric"] == "AUC"
    assert comparison_df.loc[0, "Best_AUC"] == pytest.approx(0.87655)
    assert comparison_df.loc[0, "Other_AUC"] == pytest.approx(0.81234)


def test_compare_benchmarks_marks_ties_explicitly(tmp_path) -> None:
    model1 = tmp_path / "bench_model1.csv"
    model2 = tmp_path / "bench_model2.csv"

    model1.write_text(
        "Model,Task,AUC\nmodel_1,Task A,0.75\n",
        encoding="utf-8",
    )
    model2.write_text(
        "Model,Task,AUC\nmodel_2,Task A,0.75\n",
        encoding="utf-8",
    )

    comparison_df = compare_benchmarks(str(model1), str(model2))

    assert comparison_df.loc[0, "Winner"] == "tie"
    assert comparison_df.loc[0, "Δ_AUC"] == pytest.approx(0.0)


def test_compare_benchmarks_keeps_probe_rows_separate(tmp_path) -> None:
    model1 = tmp_path / "bench_model1.csv"
    model2 = tmp_path / "bench_model2.csv"

    model1.write_text(
        "Model,Task,Samples,Date,Probe,EvalMode,AUC\n"
        "model_1,Task A,Full,2026-03-15,linear,standard,0.90\n"
        "model_1,Task A,Full,2026-03-15,histgb,standard,0.70\n",
        encoding="utf-8",
    )
    model2.write_text(
        "Model,Task,Samples,Date,Probe,EvalMode,AUC\n"
        "model_2,Task A,Full,2026-03-15,linear,standard,0.80\n"
        "model_2,Task A,Full,2026-03-15,histgb,standard,0.60\n",
        encoding="utf-8",
    )

    comparison_df = compare_benchmarks(str(model1), str(model2))

    assert sorted(comparison_df["Probe"].tolist()) == ["histgb", "linear"]
    assert len(comparison_df) == 2


def test_compare_benchmarks_keeps_eval_split_rows_separate(tmp_path) -> None:
    model1 = tmp_path / "bench_model1.csv"
    model2 = tmp_path / "bench_model2.csv"

    model1.write_text(
        "Model,Task,Samples,Date,Probe,EvalMode,EvalSplit,EvalStrategy,AUC\n"
        "model_1,Task A,Full,2026-03-15,linear,standard,test,test_split,0.90\n"
        "model_1,Task A,Full,2026-03-15,linear,standard,validation,validation_cv4_train,0.70\n",
        encoding="utf-8",
    )
    model2.write_text(
        "Model,Task,Samples,Date,Probe,EvalMode,EvalSplit,EvalStrategy,AUC\n"
        "model_2,Task A,Full,2026-03-15,linear,standard,test,test_split,0.80\n"
        "model_2,Task A,Full,2026-03-15,linear,standard,validation,validation_cv4_train,0.60\n",
        encoding="utf-8",
    )

    comparison_df = compare_benchmarks(str(model1), str(model2))

    assert len(comparison_df) == 2
    assert set(comparison_df["EvalSplit"].tolist()) == {"test", "validation"}


def test_compare_benchmarks_uses_latest_historical_row(tmp_path) -> None:
    model1 = tmp_path / "bench_model1.csv"
    model2 = tmp_path / "bench_model2.csv"

    model1.write_text(
        "Model,Task,Samples,Date,AUC\n"
        "model_1,Task A,Full,2026-03-14,0.10\n"
        "model_1,Task A,Full,2026-03-15,0.95\n",
        encoding="utf-8",
    )
    model2.write_text(
        "Model,Task,Samples,Date,AUC\nmodel_2,Task A,Full,2026-03-15,0.80\n",
        encoding="utf-8",
    )

    comparison_df = compare_benchmarks(str(model1), str(model2))

    assert comparison_df.loc[0, "Best_AUC"] == pytest.approx(0.95)
