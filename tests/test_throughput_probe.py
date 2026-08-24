"""Steady-state throughput measurement, separating warmup (compile) from measured steps."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from throughput_probe import steady_state_rate  # noqa: E402


def test_discards_warmup_steps_so_compile_does_not_depress_the_rate():
    # 3 compile-dominated steps at 10s, then 10 steady steps at 0.5s, batch=128.
    # Steady state is 128 / 0.5 = 256 pairs/s. The blended average over all 13
    # steps would be 13*128 / (3*10 + 10*0.5) = 1664 / 35 = 47.5 pairs/s, i.e.
    # a 5.4x understatement -- which is what a 15-step probe reports for a
    # compiled run.
    step_times = [10.0, 10.0, 10.0] + [0.5] * 10
    out = steady_state_rate(step_times, pairs_per_step=128, warmup=3)
    assert out["pairs_per_s"] == pytest.approx(256.0)
    assert out["blended_pairs_per_s"] == pytest.approx(47.5428, abs=1e-3)
    assert out["measured_steps"] == 10


def test_flags_recompiles_hiding_inside_the_measured_window():
    # Steady 0.5s steps with two 6s spikes: torch.compile re-tracing because the
    # padded sequence length changed. The mean alone looks merely mediocre, so the
    # probe must say the window was not steady.
    step_times = [10.0] * 3 + [0.5] * 8 + [6.0] + [0.5] * 5 + [6.0] + [0.5] * 5
    out = steady_state_rate(step_times, pairs_per_step=128, warmup=3)
    assert out["steady"] is False
    assert out["outlier_steps"] == 2


def test_clean_window_is_reported_steady():
    out = steady_state_rate([10.0] * 3 + [0.5] * 10, pairs_per_step=128, warmup=3)
    assert out["steady"] is True
    assert out["outlier_steps"] == 0
