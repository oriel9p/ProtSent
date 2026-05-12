"""Tests for JEPA ablation benchmark reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_ablation_report import (
    build_experiment_summary_dataframe,
    build_slide_markdown,
    build_task_relative_dataframe,
    plot_experiment_summary,
    plot_task_relative_performance,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to a benchmark CSV fixture."""

    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_task_relative_dataframe_merges_csvs_and_computes_deltas(
    tmp_path: Path,
) -> None:
    """Relative benchmark deltas should be computed across merged CSV inputs."""

    csv_a = tmp_path / "group_a.csv"
    csv_b = tmp_path / "group_b.csv"
    common = {
        "Samples": "50000",
        "Date": "2026-03-21",
        "Probe": "linear",
        "EvalMode": "standard",
        "EvalSplit": "test",
        "EvalStrategy": "test_split",
        "benchmark_seed": "1",
    }
    _write_csv(
        csv_a,
        [
            {
                **common,
                "Task": "Solubility (DeepSol)",
                "AUC": 0.80,
                "experiment": "baseline",
            },
            {
                **common,
                "Task": "Solubility (DeepSol)",
                "AUC": 0.84,
                "experiment": "mlm_only",
            },
        ],
    )
    _write_csv(
        csv_b,
        [
            {
                **common,
                "Task": "Fluorescence (TAPE)",
                "MSE": 2.0,
                "experiment": "baseline",
            },
            {
                **common,
                "Task": "Fluorescence (TAPE)",
                "MSE": 1.5,
                "experiment": "jepa_masked_l1",
            },
        ],
    )

    result = build_task_relative_dataframe(
        [str(csv_a), str(csv_b)],
        experiments=["mlm_only", "jepa_masked_l1"],
    )

    assert result["experiment"].tolist() == ["mlm_only", "jepa_masked_l1"]
    solubility_row = result[result["Task"] == "Solubility (DeepSol)"].iloc[0]
    fluorescence_row = result[result["Task"] == "Fluorescence (TAPE)"].iloc[0]
    assert solubility_row["Metric"] == "AUC"
    assert float(solubility_row["RelativeDeltaPct"]) == pytest.approx(5.0)
    assert fluorescence_row["Metric"] == "MSE"
    assert float(fluorescence_row["RelativeDeltaPct"]) == pytest.approx(25.0)


def test_summary_plot_and_slide_markdown_render(tmp_path: Path) -> None:
    """Summary helpers should create non-empty figures and slide markdown."""

    task_df = pd.DataFrame(
        [
            {
                "Task": "Solubility (DeepSol)",
                "TaskGroup": "Binary",
                "experiment": "mlm_only",
                "ExperimentLabel": "MLM Only",
                "Metric": "AUC",
                "ValueMean": 0.84,
                "ValueStd": 0.0,
                "BaselineMean": 0.80,
                "BaselineStd": 0.0,
                "RelativeDeltaPct": 5.0,
                "SeedCount": 1,
                "Samples": "50000",
                "Probe": "linear",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
            },
            {
                "Task": "Fluorescence (TAPE)",
                "TaskGroup": "Regression",
                "experiment": "jepa_masked_l1",
                "ExperimentLabel": "JEPA Masked L1",
                "Metric": "MSE",
                "ValueMean": 1.50,
                "ValueStd": 0.0,
                "BaselineMean": 2.00,
                "BaselineStd": 0.0,
                "RelativeDeltaPct": 25.0,
                "SeedCount": 1,
                "Samples": "50000",
                "Probe": "linear",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
            },
        ]
    )
    summary_df = build_experiment_summary_dataframe(task_df)
    task_png = plot_task_relative_performance(
        task_df,
        str(tmp_path / "task.png"),
        "Task Relative Performance",
    )
    summary_png = plot_experiment_summary(
        summary_df,
        str(tmp_path / "summary.png"),
        "Experiment Summary",
    )

    markdown = build_slide_markdown(task_df, summary_df)

    assert task_png.exists()
    assert task_png.stat().st_size > 0
    assert summary_png.exists()
    assert summary_png.stat().st_size > 0
    assert "Best mean downstream change vs baseline" in markdown
    assert "MLM Only" in markdown or "JEPA Masked L1" in markdown
