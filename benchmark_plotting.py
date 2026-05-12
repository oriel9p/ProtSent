"""Shared plotting utilities for benchmark reports.

This module provides reusable matplotlib helpers for creating consistent
benchmark visualization plots across:
  - benchmark_relative_plot.py
  - benchmark_ablation_report.py

Usage:
    from benchmark_plotting import save_figure, TASK_GROUP_COLORS

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, y, color=[TASK_GROUP_COLORS[g] for g in groups])
    save_figure(fig, "output.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from benchmark_utils import TASK_GROUP_COLORS

# Re-export commonly used constants
__all__ = ["TASK_GROUP_COLORS", "save_figure"]


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 180) -> Path:
    """Save a matplotlib figure to disk and close it.

    Args:
        fig: Matplotlib figure object
        output_path: Destination file path
        dpi: Dots per inch for the output image

    Returns:
        Pathlib Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
