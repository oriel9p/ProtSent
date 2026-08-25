"""Behavioural tests for stop_at_and_bench.sh.

The script runs unattended for hours and then deletes its own inputs, so the failure mode that
matters is not a crash -- it is reporting success after a sweep that produced nothing. That is
the same class of bug as cmd_watch_curve silently writing zero rows (see
test_save_per_query_creates_its_own_output_dir); this file pins the shell side of it.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "stop_at_and_bench.sh"
RUN = "testrun"
MARKS = "1000 4000"


def _fixture(tmp_path: Path, *, bench_exit: int = 0) -> Path:
    """A miniature repo: the real script, fake tools, and a complete target checkpoint."""
    shutil.copy(SCRIPT, tmp_path / "stop_at_and_bench.sh")
    os.chmod(tmp_path / "stop_at_and_bench.sh", 0o755)

    d = tmp_path / "models/late_interaction" / RUN
    ck = d / "checkpoint-10000"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (ck / "optimizer.pt").write_text("x")
    for n in MARKS.split():
        (d / "snapshots" / f"step-{n}").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    # Fake run_late_bench.sh: records the OUT it was handed, then exits as configured.
    bench = tmp_path / "run_late_bench.sh"
    bench.write_text(f'#!/usr/bin/env bash\necho "$OUT" >> "{tmp_path}/outs.txt"\nexit {bench_exit}\n')
    os.chmod(bench, 0o755)

    # Fake `uv`: the script calls `uv run --no-sync python - <snap> <dense>`; just make the dir.
    binp = tmp_path / "bin"
    binp.mkdir()
    (binp / "uv").write_text('#!/usr/bin/env bash\nmkdir -p "${@: -1}"\n')
    os.chmod(binp / "uv", 0o755)
    return tmp_path


def _run(tmp_path: Path, **env_extra) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{tmp_path}/bin:{os.environ['PATH']}",
           "RUN": RUN, "TARGET": "10000", "MARKS": MARKS, "SETTLE": "0",
           "TRAIN_PID": "999999",  # dead: the kill branch is skipped
           **env_extra}
    return subprocess.run(["bash", str(tmp_path / "stop_at_and_bench.sh")],
                          capture_output=True, text=True, env=env, timeout=180)


def _dense_dirs(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "models/late_interaction" / RUN / "snapshots").glob("*-dense"))


def test_a_failed_sweep_does_not_report_success(tmp_path):
    """`wait` discards subshell exit codes, so a sweep where every benchmark failed still
    printed "all marks benchmarked" and exited 0. Unattended, that is indistinguishable from
    a real sweep until someone opens the results dir."""
    t = _fixture(tmp_path, bench_exit=1)
    r = _run(t)
    assert "all marks benchmarked" not in r.stdout, "reported success after every benchmark failed"
    assert r.returncode != 0, "exited 0 after every benchmark failed"


def test_a_failed_sweep_keeps_the_dense_views_for_a_retry(tmp_path):
    """Cleanup ran unconditionally, deleting the sweep's inputs before anyone could retry."""
    t = _fixture(tmp_path, bench_exit=1)
    _run(t)
    assert _dense_dirs(t), "deleted the dense views after a failed sweep"


def test_a_successful_sweep_reports_and_cleans_up(tmp_path):
    t = _fixture(tmp_path, bench_exit=0)
    r = _run(t)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all marks benchmarked" in r.stdout
    assert not _dense_dirs(t), "left the disposable dense views behind after a clean sweep"


def test_results_dir_is_overridable_for_runs_that_are_not_r2_35m(tmp_path):
    """clean_35m names the recipe and size, so both r2 35M runs share it by design -- but a
    150M r2 run must not land there."""
    t = _fixture(tmp_path, bench_exit=0)
    custom = str(tmp_path / "elsewhere")
    _run(t, OUTDIR=custom)
    seen = (t / "outs.txt").read_text()
    assert custom in seen, f"OUTDIR override ignored; benchmarks wrote to {seen!r}"


def test_it_refuses_to_benchmark_while_training_still_holds_the_gpus(tmp_path):
    """If a rank survives the kill (D-state on NCCL is the realistic case), benchmarking anyway
    produces contention-skewed numbers that look valid."""
    t = _fixture(tmp_path, bench_exit=0)
    # pkill stubbed to a no-op, and a live pid: training "survives" the stop.
    (t / "bin" / "pkill").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(t / "bin" / "pkill", 0o755)
    proc = subprocess.Popen(["sleep", "120"])
    try:
        r = _run(t, TRAIN_PID=str(proc.pid))
        assert r.returncode != 0, "benchmarked while training was still running"
        assert "all marks benchmarked" not in r.stdout
    finally:
        proc.kill()


def test_a_slow_sweep_is_flagged(tmp_path):
    """Marks already run one per GPU, so a slow sweep is a per-mark cost, not a scheduling
    problem -- but it has to be visible in an unattended overnight log to be actionable."""
    t = _fixture(tmp_path, bench_exit=0)
    r = _run(t, SWEEP_WARN_S="-1")  # fake bench is instant; elapsed is 0s
    assert "WARN: sweep took" in r.stdout, r.stdout
    assert "sweep wall clock:" in r.stdout


def test_a_fast_sweep_is_not_flagged(tmp_path):
    t = _fixture(tmp_path, bench_exit=0)
    r = _run(t, SWEEP_WARN_S="99999")
    assert "WARN: sweep took" not in r.stdout
