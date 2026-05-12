"""Unit tests for grouped relative benchmark plotting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_relative_plot import build_relative_dataframe, plot_relative_performance


def test_build_relative_dataframe_handles_higher_is_better_metric(
    tmp_path: Path,
) -> None:
    """Relative delta should be positive when trained AUC exceeds baseline."""
    baseline_csv = tmp_path / "baseline.csv"
    trained_csv = tmp_path / "trained.csv"

    baseline_df = pd.DataFrame(
        [
            {
                "Task": "Solubility (DeepSol)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Date": "2026-03-20",
                "AUC": 0.80,
            }
        ]
    )
    trained_df = pd.DataFrame(
        [
            {
                "Task": "Solubility (DeepSol)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Date": "2026-03-20",
                "AUC": 0.88,
            }
        ]
    )

    baseline_df.to_csv(baseline_csv, index=False)
    trained_df.to_csv(trained_csv, index=False)

    result = build_relative_dataframe(str(baseline_csv), str(trained_csv))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Metric"] == "AUC"
    assert row["TaskGroup"] == "Binary"
    assert float(row["RelativeDeltaPct"]) == pytest.approx(10.0)


def test_build_relative_dataframe_handles_mse_direction(
    tmp_path: Path,
) -> None:
    """Relative delta should be positive when trained MSE is lower."""
    baseline_csv = tmp_path / "baseline.csv"
    trained_csv = tmp_path / "trained.csv"

    baseline_df = pd.DataFrame(
        [
            {
                "Task": "Fluorescence (TAPE)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Date": "2026-03-20",
                "MSE": 2.0,
            }
        ]
    )
    trained_df = pd.DataFrame(
        [
            {
                "Task": "Fluorescence (TAPE)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Date": "2026-03-20",
                "MSE": 1.5,
            }
        ]
    )

    baseline_df.to_csv(baseline_csv, index=False)
    trained_df.to_csv(trained_csv, index=False)

    result = build_relative_dataframe(str(baseline_csv), str(trained_csv))

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Metric"] == "MSE"
    assert row["TaskGroup"] == "Regression"
    assert float(row["RelativeDeltaPct"]) == pytest.approx(25.0)


def test_plot_relative_performance_writes_png(tmp_path: Path) -> None:
    """Plot helper should render and save a non-empty PNG."""
    plot_df = pd.DataFrame(
        [
            {
                "Task": "Solubility (DeepSol)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Metric": "AUC",
                "BaselineValue": 0.8,
                "TrainedValue": 0.84,
                "RelativeDeltaPct": 5.0,
                "TaskGroup": "Binary",
            },
            {
                "Task": "Fluorescence (TAPE)",
                "Samples": "Full",
                "Probe": "knn",
                "EvalMode": "standard",
                "EvalSplit": "test",
                "EvalStrategy": "test_split",
                "Metric": "Spearman",
                "BaselineValue": 0.31,
                "TrainedValue": 0.34,
                "RelativeDeltaPct": 9.6774193548,
                "TaskGroup": "Regression",
            },
        ]
    )

    output_png = tmp_path / "relative_plot.png"
    output_path = plot_relative_performance(
        plot_df,
        str(output_png),
        "Relative Performance vs Baseline",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
