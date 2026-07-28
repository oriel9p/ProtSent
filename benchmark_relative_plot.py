#!/usr/bin/env python3
"""Generate grouped relative benchmark plots against a baseline model.

This script aligns two benchmark CSVs, computes per-task relative changes
against the baseline score, and writes:
1) A PNG figure with task-level and group-level summaries.
2) A CSV table with the computed relative deltas.

For higher-is-better metrics (AUC, F1, Spearman, ...):
    relative_delta_pct = 100 * (trained - baseline) / abs(baseline)
For MSE (lower is better):
    relative_delta_pct = 100 * (baseline - trained) / abs(baseline)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_comparison import (
    DEFAULT_RESULT_EVAL_MODE,
    DEFAULT_RESULT_EVAL_SPLIT,
    DEFAULT_RESULT_EVAL_STRATEGY,
    DEFAULT_RESULT_PROBE,
    METRIC_PRIORITY,
    RESULT_IDENTITY_COLUMNS,
    find_result_file,
)
from benchmark_tasks import TASKS
from benchmark_utils import (
    TASK_GROUP_COLORS,
    first_common_metric,
    prepare_result_df,
    relative_delta_pct,
    task_group_map,
)
from benchmark_plotting import save_figure


def build_relative_dataframe(
    baseline_csv: str,
    trained_csv: str,
    *,
    output_dir: str = "results/benchmarks",
) -> pd.DataFrame:
    """Compute relative per-task gains versus baseline.

    Args:
        baseline_csv: Baseline benchmark CSV path or model identifier.
        trained_csv: Trained benchmark CSV path or model identifier.
        output_dir: Directory used when model identifiers are passed.

    Returns:
        Dataframe with baseline/trained metric values and relative delta percent.

    Raises:
        ValueError: If no comparable rows are found.
    """
    baseline_path = find_result_file(baseline_csv, output_dir=output_dir)
    trained_path = find_result_file(trained_csv, output_dir=output_dir)

    baseline_df = prepare_result_df(pd.read_csv(baseline_path))
    trained_df = prepare_result_df(pd.read_csv(trained_path))

    baseline_rows = {
        tuple(row[column] for column in RESULT_IDENTITY_COLUMNS): row
        for _, row in baseline_df.iterrows()
    }
    trained_rows = {
        tuple(row[column] for column in RESULT_IDENTITY_COLUMNS): row
        for _, row in trained_df.iterrows()
    }

    task_to_group = task_group_map()
    comparison_rows: list[dict[str, object]] = []

    for identity in sorted(set(baseline_rows) & set(trained_rows)):
        baseline_row = baseline_rows[identity]
        trained_row = trained_rows[identity]
        # Find first metric with finite values in BOTH rows
        metric = None
        for _m in METRIC_PRIORITY:
            if _m in baseline_row.index and _m in trained_row.index:
                try:
                    v1, v2 = float(baseline_row[_m]), float(trained_row[_m])
                    if np.isfinite(v1) and np.isfinite(v2):
                        metric = _m
                        break
                except (ValueError, TypeError):
                    continue
        if metric is None:
            continue

        baseline_value = float(baseline_row[metric])
        trained_value = float(trained_row[metric])

        denom = max(abs(baseline_value), EPSILON)
        if metric == "MSE":
            relative_delta_pct = 100.0 * (baseline_value - trained_value) / denom
        else:
            relative_delta_pct = 100.0 * (trained_value - baseline_value) / denom

        task_name = str(identity[0])
        comparison_rows.append(
            {
                "Task": task_name,
                "Samples": str(identity[1]),
                "Probe": str(identity[2]),
                "EvalMode": str(identity[3]),
                "EvalSplit": str(identity[4]),
                "EvalStrategy": str(identity[5]),
                "Metric": metric,
                "BaselineValue": baseline_value,
                "TrainedValue": trained_value,
                "RelativeDeltaPct": relative_delta_pct,
                "TaskGroup": task_to_group.get(task_name, "Other"),
            }
        )

    if not comparison_rows:
        raise ValueError(
            "No comparable rows found between baseline and trained benchmark CSVs."
        )

    result = pd.DataFrame(comparison_rows)
    group_order = [
        "Binary",
        "Multiclass",
        "Multilabel",
        "Regression",
        "Retrieval",
        "Other",
    ]
    result["TaskGroup"] = pd.Categorical(
        result["TaskGroup"], categories=group_order, ordered=True
    )
    result = result.sort_values(
        ["TaskGroup", "RelativeDeltaPct", "Task"], ascending=[True, False, True]
    )
    return result.reset_index(drop=True)


def plot_relative_performance(df: pd.DataFrame, output_png: str, title: str) -> Path:
    """Render and save a grouped relative-performance figure.

    Args:
        df: Dataframe from ``build_relative_dataframe``.
        output_png: Destination PNG path.
        title: Figure title.

    Returns:
        Saved figure path.
    """
    plot_df = df.copy()
    plot_df["Color"] = (
        plot_df["TaskGroup"].astype(str).map(TASK_GROUP_COLORS).fillna("#7f7f7f")
    )

    group_summary = (
        plot_df.groupby("TaskGroup", observed=True)["RelativeDeltaPct"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    fig, (ax_tasks, ax_groups) = plt.subplots(
        2,
        1,
        figsize=(18, 12),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )

    y_positions = np.arange(len(plot_df))
    ax_tasks.barh(
        y_positions, plot_df["RelativeDeltaPct"], color=plot_df["Color"], alpha=0.92
    )
    ax_tasks.axvline(0.0, color="black", linewidth=1.1, linestyle="--")
    ax_tasks.set_yticks(y_positions)
    ax_tasks.set_yticklabels(plot_df["Task"], fontsize=8)
    ax_tasks.invert_yaxis()
    ax_tasks.set_xlabel("Relative change vs baseline (%)")
    ax_tasks.set_title(title)
    ax_tasks.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)

    legend_labels = [
        label
        for label in [
            "Binary",
            "Multiclass",
            "Multilabel",
            "Regression",
            "Retrieval",
            "Other",
        ]
        if label in set(plot_df["TaskGroup"].astype(str))
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TASK_GROUP_COLORS[label], alpha=0.92)
        for label in legend_labels
    ]
    ax_tasks.legend(handles, legend_labels, title="Task group", loc="lower right")

    ax_groups.bar(
        group_summary["TaskGroup"].astype(str),
        group_summary["mean"],
        color=[
            TASK_GROUP_COLORS.get(str(group), "#7f7f7f")
            for group in group_summary["TaskGroup"]
        ],
        alpha=0.92,
    )
    ax_groups.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_groups.set_ylabel("Mean relative change (%)")
    ax_groups.set_xlabel("Task group")
    ax_groups.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    for idx, row in group_summary.iterrows():
        ax_groups.text(
            idx,
            float(row["mean"]),
            f"n={int(row['count'])}",
            ha="center",
            va="bottom" if float(row["mean"]) >= 0 else "top",
            fontsize=9,
        )

    output_path = save_figure(fig, output_png)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the relative benchmark plot tool."""
    parser = argparse.ArgumentParser(
        description="Generate a grouped relative benchmark figure vs baseline"
    )
    parser.add_argument(
        "--baseline_csv", required=True, help="Baseline benchmark CSV path"
    )
    parser.add_argument(
        "--trained_csv", required=True, help="Trained benchmark CSV path"
    )
    parser.add_argument("--output_png", required=True, help="Output PNG path")
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional output CSV path for relative delta table",
    )
    parser.add_argument(
        "--title",
        default="Relative Performance vs Baseline",
        help="Figure title",
    )
    parser.add_argument(
        "--output_dir",
        default="results/benchmarks",
        help="Search directory used when baseline/trained identifiers are model names",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    relative_df = build_relative_dataframe(
        args.baseline_csv,
        args.trained_csv,
        output_dir=args.output_dir,
    )

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        relative_df.to_csv(output_csv_path, index=False)

    output_png_path = plot_relative_performance(
        relative_df, args.output_png, args.title
    )
    print(f"Saved grouped relative benchmark figure: {output_png_path}")


if __name__ == "__main__":
    main()
