#!/usr/bin/env python3
"""Build lab-facing JEPA ablation summaries and plots.

This module merges one or more ablation benchmark CSVs, compares each
experiment against a baseline experiment, and writes:

1. A per-task grouped bar chart of relative performance vs baseline.
2. An experiment-level summary chart with mean relative gain and task wins.
3. CSV tables for task-level and experiment-level summaries.
4. Slide-ready markdown with benchmark and optional LM takeaways.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from benchmark_comparison import (
    METRIC_PRIORITY,
    RESULT_IDENTITY_COLUMNS,
)
from benchmark_utils import (
    first_common_metric,
    prepare_result_df,
    relative_delta_pct,
    task_group_map,
)
from benchmark_plotting import save_figure

DEFAULT_EXPERIMENT_ORDER = [
    "baseline",
    "mlm_only",
    "jepa_masked_l1",
    "jepa_all_position",
    "jepa_global_pool",
    "jepa_cls",
    "jepa_hybrid",
]
EXPERIMENT_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "mlm_only": "MLM Only",
    "jepa_masked_l1": "JEPA Masked L1",
    "jepa_all_position": "JEPA All Position",
    "jepa_global_pool": "JEPA Global Pool",
    "jepa_cls": "JEPA CLS",
    "jepa_hybrid": "JEPA Hybrid",
}


def _display_name(experiment: str) -> str:
    """Return a readable experiment label."""

    return EXPERIMENT_DISPLAY_NAMES.get(
        experiment, experiment.replace("_", " ").title()
    )


def _ordered_experiments(
    experiments: Sequence[str] | None,
    available: Sequence[str],
    baseline_experiment: str,
    include_baseline: bool,
) -> list[str]:
    """Resolve experiment order for summaries and plots."""

    if experiments:
        ordered = [name for name in experiments if name in set(available)]
    else:
        preferred = [
            name for name in DEFAULT_EXPERIMENT_ORDER if name in set(available)
        ]
        extras = [name for name in available if name not in set(preferred)]
        ordered = [*preferred, *sorted(extras)]

    if include_baseline:
        if baseline_experiment in set(available) and baseline_experiment not in ordered:
            ordered = [baseline_experiment, *ordered]
    else:
        ordered = [name for name in ordered if name != baseline_experiment]

    return ordered


def load_ablation_benchmark_frame(benchmark_csvs: Sequence[str]) -> pd.DataFrame:
    """Load and merge one or more ablation benchmark CSVs."""

    frames = [
        prepare_result_df(
            pd.read_csv(path),
            extra_defaults={"benchmark_seed": "0", "experiment": "unknown"},
            extra_dedup_columns=["experiment", "benchmark_seed"],
        )
        for path in benchmark_csvs
    ]
    if not frames:
        raise ValueError("At least one benchmark CSV is required.")
    return pd.concat(frames, ignore_index=True)


def build_task_relative_dataframe(
    benchmark_csvs: Sequence[str],
    *,
    baseline_experiment: str = "baseline",
    experiments: Sequence[str] | None = None,
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Build task-level relative benchmark results versus baseline."""

    combined = load_ablation_benchmark_frame(benchmark_csvs)
    available = combined["experiment"].unique().tolist()
    ordered_experiments = _ordered_experiments(
        experiments,
        available,
        baseline_experiment,
        include_baseline,
    )
    if baseline_experiment not in set(available):
        raise ValueError(f"Baseline experiment '{baseline_experiment}' not found.")

    task_group_lookup = task_group_map()
    baseline_df = combined[combined["experiment"] == baseline_experiment]
    rows: list[dict[str, object]] = []

    for identity, baseline_group in baseline_df.groupby(
        list(RESULT_IDENTITY_COLUMNS), sort=True
    ):
        identity_mask = np.logical_and.reduce(
            [
                combined[column].eq(value)
                for column, value in zip(RESULT_IDENTITY_COLUMNS, identity)
            ]
        )
        identity_df = combined[identity_mask]
        task_name = str(identity[0])
        task_group = task_group_lookup.get(task_name, "Other")

        baseline_metric = None
        for metric in METRIC_PRIORITY:
            if (
                metric in baseline_group.columns
                and baseline_group[metric].notna().any()
            ):
                baseline_metric = metric
                break
        if baseline_metric is None:
            continue

        baseline_values = baseline_group[baseline_metric].dropna().astype(float)
        baseline_mean = float(baseline_values.mean())
        baseline_std = (
            float(baseline_values.std(ddof=0)) if len(baseline_values) > 1 else 0.0
        )

        if include_baseline:
            rows.append(
                {
                    "Task": task_name,
                    "TaskGroup": task_group,
                    "experiment": baseline_experiment,
                    "ExperimentLabel": _display_name(baseline_experiment),
                    "Metric": baseline_metric,
                    "ValueMean": baseline_mean,
                    "ValueStd": baseline_std,
                    "BaselineMean": baseline_mean,
                    "BaselineStd": baseline_std,
                    "RelativeDeltaPct": 0.0,
                    "SeedCount": int(len(baseline_values)),
                    **{
                        column: value
                        for column, value in zip(RESULT_IDENTITY_COLUMNS, identity)
                    },
                }
            )

        for experiment in ordered_experiments:
            if experiment == baseline_experiment:
                continue
            experiment_group = identity_df[identity_df["experiment"] == experiment]
            if experiment_group.empty:
                continue

            metric = first_common_metric(
                baseline_group.columns, experiment_group.columns
            )
            if metric is None:
                continue

            experiment_values = experiment_group[metric].dropna().astype(float)
            if experiment_values.empty:
                continue

            experiment_mean = float(experiment_values.mean())
            experiment_std = (
                float(experiment_values.std(ddof=0))
                if len(experiment_values) > 1
                else 0.0
            )
            rows.append(
                {
                    "Task": task_name,
                    "TaskGroup": task_group,
                    "experiment": experiment,
                    "ExperimentLabel": _display_name(experiment),
                    "Metric": metric,
                    "ValueMean": experiment_mean,
                    "ValueStd": experiment_std,
                    "BaselineMean": baseline_mean,
                    "BaselineStd": baseline_std,
                    "RelativeDeltaPct": relative_delta_pct(
                        metric, baseline_mean, experiment_mean
                    ),
                    "SeedCount": int(len(experiment_values)),
                    **{
                        column: value
                        for column, value in zip(RESULT_IDENTITY_COLUMNS, identity)
                    },
                }
            )

    if not rows:
        raise ValueError("No comparable ablation rows found.")

    result = pd.DataFrame(rows)
    experiment_order = (
        ordered_experiments
        if include_baseline
        else [name for name in ordered_experiments if name != baseline_experiment]
    )
    if include_baseline and baseline_experiment not in experiment_order:
        experiment_order = [baseline_experiment, *experiment_order]
    result["experiment"] = pd.Categorical(
        result["experiment"], categories=experiment_order, ordered=True
    )
    result["TaskGroup"] = pd.Categorical(
        result["TaskGroup"],
        categories=[
            "Binary",
            "Multiclass",
            "Multilabel",
            "Regression",
            "Retrieval",
            "Other",
        ],
        ordered=True,
    )
    return result.sort_values(["TaskGroup", "Task", "experiment"]).reset_index(
        drop=True
    )


def build_experiment_summary_dataframe(task_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate task-level relative results into experiment-level summaries."""

    comparable = task_df.copy()
    task_key_columns = list(RESULT_IDENTITY_COLUMNS)
    winner_indices = comparable.groupby(task_key_columns, observed=True)[
        "RelativeDeltaPct"
    ].idxmax()
    win_counts = comparable.loc[winner_indices, "experiment"].value_counts().to_dict()

    summary = (
        comparable.groupby(["experiment", "ExperimentLabel"], observed=True)
        .agg(
            MeanRelativeDeltaPct=("RelativeDeltaPct", "mean"),
            MedianRelativeDeltaPct=("RelativeDeltaPct", "median"),
            StdRelativeDeltaPct=(
                "RelativeDeltaPct",
                lambda values: float(np.std(values, ddof=0)),
            ),
            TaskCount=("Task", "count"),
            ImprovedTaskCount=(
                "RelativeDeltaPct",
                lambda values: int((values > 0).sum()),
            ),
            DeclinedTaskCount=(
                "RelativeDeltaPct",
                lambda values: int((values < 0).sum()),
            ),
        )
        .reset_index()
    )
    summary["TaskWinCount"] = summary["experiment"].map(
        lambda name: int(win_counts.get(name, 0))
    )
    return summary.sort_values(
        ["MeanRelativeDeltaPct", "TaskWinCount"], ascending=[False, False]
    ).reset_index(drop=True)


def _color_lookup(labels: Sequence[str]) -> dict[str, str]:
    """Assign stable colors to experiment labels."""

    cmap = plt.get_cmap("tab10")
    return {label: cmap(idx % 10) for idx, label in enumerate(labels)}


def plot_task_relative_performance(
    task_df: pd.DataFrame,
    output_png: str,
    title: str,
) -> Path:
    """Render a grouped task-level relative performance plot."""

    if task_df.empty:
        raise ValueError("Task dataframe is empty.")

    task_order = task_df[["Task", "TaskGroup"]].drop_duplicates().reset_index(drop=True)
    labels = task_df["ExperimentLabel"].drop_duplicates().tolist()
    colors = _color_lookup(labels)
    pivot_df = (
        task_df.pivot_table(
            index="Task",
            columns="ExperimentLabel",
            values="RelativeDeltaPct",
            aggfunc="mean",
        )
        .reindex(task_order["Task"].tolist())
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(22, 9), constrained_layout=True)
    x_positions = np.arange(len(task_order))
    bar_width = 0.82 / max(len(labels), 1)

    for idx, label in enumerate(labels):
        offsets = x_positions - 0.41 + bar_width / 2 + idx * bar_width
        ax.bar(
            offsets,
            pivot_df[label].to_numpy(dtype=float),
            width=bar_width,
            label=label,
            color=colors[label],
            alpha=0.92,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_order["Task"], rotation=50, ha="right", fontsize=9)
    ax.set_ylabel("Relative change vs baseline (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    current_group = None
    group_start = 0
    for idx, (_, row) in enumerate(task_order.iterrows()):
        group = str(row["TaskGroup"])
        if current_group is None:
            current_group = group
            group_start = idx
            continue
        if group != current_group:
            if (group_start // 2) % 2 == 0:
                ax.axvspan(group_start - 0.5, idx - 0.5, color="#f7f7f7", zorder=0)
            current_group = group
            group_start = idx
    if current_group is not None and (group_start // 2) % 2 == 0:
        ax.axvspan(group_start - 0.5, len(task_order) - 0.5, color="#f7f7f7", zorder=0)

    ax.legend(ncol=min(4, max(len(labels), 1)), loc="upper left")

    output_path = save_figure(fig, output_png)
    return output_path


def plot_experiment_summary(
    summary_df: pd.DataFrame,
    output_png: str,
    title: str,
) -> Path:
    """Render an experiment-level task-wins bar chart."""

    if summary_df.empty:
        raise ValueError("Summary dataframe is empty.")

    labels = summary_df["ExperimentLabel"].tolist()
    colors = _color_lookup(labels)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    x_positions = np.arange(len(summary_df))
    bar_colors = [colors[label] for label in labels]
    bars = ax.bar(x_positions, summary_df["TaskWinCount"], color=bar_colors, alpha=0.92)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_ylabel("Task wins (#)")
    ax.set_xlabel("Experiment")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = save_figure(fig, output_png)
    return output_path


def plot_performance_heatmap(
    task_df: pd.DataFrame,
    output_png: str,
    title: str,
) -> Path:
    """Render a heatmap of per-task relative performance vs baseline.

    Rows are experiments (ordered by mean relative delta, best first).
    Columns are tasks. Cell colour encodes RelativeDeltaPct on a diverging
    red-yellow-green scale centred at zero.

    Args:
        task_df: Output of build_task_relative_dataframe.
        output_png: Absolute path for the saved figure.
        title: Figure title.

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If task_df is empty.
    """

    if task_df.empty:
        raise ValueError("Task dataframe is empty.")

    pivot = task_df.pivot_table(
        index="ExperimentLabel",
        columns="Task",
        values="RelativeDeltaPct",
        aggfunc="mean",
    )
    exp_order = (
        task_df.groupby("ExperimentLabel", observed=True)["RelativeDeltaPct"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot = pivot.reindex(exp_order)

    data = pivot.to_numpy(dtype=float)
    abs_max = max(float(np.nanmax(np.abs(data))), 1.0)

    n_exp, n_task = data.shape
    fig, ax = plt.subplots(
        figsize=(max(10, n_task * 1.15), max(3.5, n_exp * 0.85)),
        constrained_layout=True,
    )
    im = ax.imshow(
        data,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
    )
    ax.set_xticks(np.arange(n_task))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=50, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_exp))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Relative change vs baseline (%)")

    for row_idx in range(n_exp):
        for col_idx in range(n_task):
            val = data[row_idx, col_idx]
            if np.isnan(val):
                continue
            luminance_threshold = abs_max * 0.55
            text_color = "white" if abs(val) > luminance_threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    ax.set_title(title)
    output_path = save_figure(fig, output_png)
    return output_path


def load_lm_summary(
    lm_csv: str,
    *,
    experiments: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the optional ablation LM metric summary."""

    lm_df = pd.read_csv(lm_csv)
    if experiments:
        lm_df = lm_df[lm_df["experiment"].isin(experiments)]
    lm_df = lm_df.copy()
    lm_df["ExperimentLabel"] = lm_df["experiment"].map(_display_name)
    return lm_df.sort_values("mlm_loss_mean").reset_index(drop=True)


def build_slide_markdown(
    task_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    lm_df: Optional[pd.DataFrame] = None,
    title: str = "Synthyra JEPA Ablation Summary",
) -> str:
    """Create short slide-ready markdown content."""

    top_experiment = summary_df.iloc[0]
    top_task_rows = task_df.sort_values("RelativeDeltaPct", ascending=False).head(3)
    bottom_task_rows = task_df.sort_values("RelativeDeltaPct", ascending=True).head(3)

    lines = [
        f"# {title}",
        "",
        "## Headline",
        (
            f"- Best mean downstream change vs baseline: {top_experiment['ExperimentLabel']} "
            f"({float(top_experiment['MeanRelativeDeltaPct']):+.2f}% across {int(top_experiment['TaskCount'])} tasks)."
        ),
        (
            f"- Most task wins: {summary_df.sort_values('TaskWinCount', ascending=False).iloc[0]['ExperimentLabel']} "
            f"({int(summary_df.sort_values('TaskWinCount', ascending=False).iloc[0]['TaskWinCount'])} wins)."
        ),
        "- Benchmark summary is relative to the frozen Synthyra baseline; positive percentages mean better than baseline.",
        "",
        "## Biggest Gains",
    ]
    for _, row in top_task_rows.iterrows():
        lines.append(
            f"- {row['ExperimentLabel']} on {row['Task']}: {float(row['RelativeDeltaPct']):+.2f}% ({row['Metric']})."
        )

    lines.append("")
    lines.append("## Biggest Drops")
    for _, row in bottom_task_rows.iterrows():
        lines.append(
            f"- {row['ExperimentLabel']} on {row['Task']}: {float(row['RelativeDeltaPct']):+.2f}% ({row['Metric']})."
        )

    lines.append("")
    lines.append("## LM Check")
    if lm_df is not None and not lm_df.empty:
        best_lm = lm_df.sort_values("mlm_loss_mean").iloc[0]
        baseline_lm = lm_df[lm_df["experiment"] == "baseline"]
        if not baseline_lm.empty:
            baseline_loss = float(baseline_lm.iloc[0]["mlm_loss_mean"])
            delta = baseline_loss - float(best_lm["mlm_loss_mean"])
            lines.append(
                f"- Lowest validation MLM loss: {best_lm['ExperimentLabel']} ({float(best_lm['mlm_loss_mean']):.4f}, {delta:+.4f} vs baseline)."
            )
        else:
            lines.append(
                f"- Lowest validation MLM loss: {best_lm['ExperimentLabel']} ({float(best_lm['mlm_loss_mean']):.4f})."
            )
    else:
        lines.append("- No LM summary CSV provided.")

    lines.extend(
        [
            "",
            "## Caveats",
            "- Current benchmark report uses one completed benchmark seed per experiment, so benchmark-side variance is not estimated here.",
            "- Mean relative deltas are averaged across mixed task metrics after converting each task to a baseline-relative percentage.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ablation reporting tool."""

    parser = argparse.ArgumentParser(
        description="Build JEPA ablation lab summary plots"
    )
    parser.add_argument(
        "--benchmark_csvs",
        nargs="+",
        required=True,
        help="One or more ablation benchmark CSVs",
    )
    parser.add_argument("--lm_csv", default="", help="Optional ablation LM metrics CSV")
    parser.add_argument(
        "--baseline_experiment",
        default="baseline",
        help="Baseline experiment identifier",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiments to plot. Default: all present experiments.",
    )
    parser.add_argument(
        "--include_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the baseline as zero-delta bars in the task plot.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where outputs are written"
    )
    parser.add_argument(
        "--task_title",
        default="JEPA Ablations: Relative Benchmark Performance vs Baseline",
        help="Task figure title",
    )
    parser.add_argument(
        "--summary_title",
        default="JEPA Ablations: Summary vs Baseline",
        help="Summary figure title",
    )
    parser.add_argument(
        "--slide_title",
        default="Synthyra JEPA Ablation Summary",
        help="Slide markdown title",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    task_df = build_task_relative_dataframe(
        args.benchmark_csvs,
        baseline_experiment=args.baseline_experiment,
        experiments=args.experiments,
        include_baseline=args.include_baseline,
    )
    summary_df = build_experiment_summary_dataframe(task_df)
    lm_df = (
        load_lm_summary(args.lm_csv, experiments=args.experiments)
        if args.lm_csv
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_csv = output_dir / "task_relative_summary.csv"
    summary_csv = output_dir / "experiment_summary.csv"
    task_png = output_dir / "task_relative_performance.png"
    summary_png = output_dir / "experiment_summary.png"
    heatmap_png = output_dir / "task_performance_heatmap.png"
    slide_md = output_dir / "slide_summary.md"

    task_df.to_csv(task_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_task_relative_performance(task_df, str(task_png), args.task_title)
    plot_experiment_summary(summary_df, str(summary_png), args.summary_title)
    plot_performance_heatmap(task_df, str(heatmap_png), args.task_title)
    slide_md.write_text(
        build_slide_markdown(task_df, summary_df, lm_df=lm_df, title=args.slide_title),
        encoding="utf-8",
    )

    if lm_df is not None:
        lm_df.to_csv(output_dir / "lm_summary.csv", index=False)

    print(f"Saved task summary CSV: {task_csv}")
    print(f"Saved experiment summary CSV: {summary_csv}")
    print(f"Saved task plot: {task_png}")
    print(f"Saved summary plot: {summary_png}")
    print(f"Saved heatmap: {heatmap_png}")
    print(f"Saved slide markdown: {slide_md}")


if __name__ == "__main__":
    main()
