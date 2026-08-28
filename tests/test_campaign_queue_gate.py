"""The step gate must be able to fire while training is still alive."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "run_campaign_queue.sh"


def _run_stage_body() -> str:
    """Executable lines of run_stage only.

    Comments are stripped because the fix's own comment explains the ordering and mentions
    wait_for_pid, which a naive substring search finds before the real call -- the test would then
    fail on correct code for describing itself.
    """
    t = SCRIPT.read_text()
    start = t.index("run_stage() {")
    body = t[start:t.index("\n}\n", start)]
    return "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))


def test_step_gate_is_armed_before_the_blocking_wait():
    """`stop_at_and_bench.sh` stops a run at TARGET; it must start before we block on the trainer.

    run_stage launches training with --max_steps MAXSTEPS (30,000, which only shapes the
    constant_with_warmup schedule) and relies on the gate to stop it at TARGET (10,000). But
    wait_for_pid blocks until the trainer exits on its own, so a gate invoked after it cannot stop
    anything -- by the time it runs, training has already gone to MAXSTEPS. That is not a wrong
    constant, it is an unreachable guard: every arm silently trains 3x its intended budget.
    """
    body = _run_stage_body()
    gate = body.find("stop_at_and_bench.sh")
    wait = body.find("wait_for_pid")
    assert gate != -1, "run_stage no longer invokes the step gate at all"
    assert wait != -1, "run_stage no longer waits on the trainer"
    assert gate < wait, (
        "stop_at_and_bench.sh is invoked AFTER wait_for_pid, so it cannot stop training at TARGET; "
        "training runs to --max_steps instead"
    )


def test_gate_is_backgrounded_so_it_can_run_alongside_training():
    """Arming it first is not enough -- a synchronous call would block before training starts."""
    body = _run_stage_body()
    line = next(l for l in body.splitlines() if "stop_at_and_bench.sh" in l)
    following = body[body.index(line):body.index(line) + 400]
    assert "&" in following.split("\n")[0] or "&" in following.split("\n")[1], (
        f"the gate call must be backgrounded, got: {line.strip()}"
    )
