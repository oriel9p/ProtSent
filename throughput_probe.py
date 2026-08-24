#!/usr/bin/env python
"""Steady-state training throughput, measured after warmup.

A 15-step probe blends compile/warmup cost into the rate: three 10s compile steps
in front of ten 0.5s steps report 47.5 pairs/s where the model actually sustains
256. Every throughput number here is taken over post-warmup steps only, and the
blended figure is reported beside it so the distortion stays visible.
"""

from __future__ import annotations

import statistics


def steady_state_rate(step_times: list[float], pairs_per_step: int, warmup: int) -> dict:
    """Throughput over the steps after `warmup`, plus the blended figure for contrast.

    Args:
        step_times: wall-clock seconds per optimizer step, in order.
        pairs_per_step: pairs consumed per step (per_device_batch * world_size).
        warmup: leading steps to discard (compile, cudnn autotune, allocator warmup).
    """
    if len(step_times) <= warmup:
        raise ValueError(f"need more than {warmup} steps, got {len(step_times)}")
    measured = step_times[warmup:]
    total = sum(measured)
    # A recompile (torch.compile re-tracing a new padded length) shows up as an
    # isolated multi-second step inside an otherwise flat window. Averaging hides
    # it, so flag any step past 2x the median rather than reporting a clean mean.
    median = statistics.median(measured)
    outliers = [t for t in measured if t > 2 * median]
    return {
        "steady": not outliers,
        "outlier_steps": len(outliers),
        "pairs_per_s": pairs_per_step * len(measured) / total,
        "blended_pairs_per_s": pairs_per_step * len(step_times) / sum(step_times),
        "measured_steps": len(measured),
        "warmup_steps": warmup,
        "s_per_step_median": median,
        "s_per_step_max": max(measured),
        "warmup_s": sum(step_times[:warmup]),
    }
