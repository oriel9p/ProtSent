"""MaxSim-kNN rows need an interval, and per-query predictions to pair on.

Table B's "27 of 30" was point estimates only: nothing in the file said whether any individual
win was resolvable. The SCOPe and few-shot tables both carry bootstrap intervals over their own
queries; this is the same estimator over the eval split.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_rows_carry_a_bootstrap_interval():
    src = (ROOT / "maxsim_knn_bench.py").read_text()
    assert "boot_ci" in src, "no bootstrap over the eval queries"
    assert "ci95" in src, "rows do not report an interval"


def test_per_query_predictions_are_saved_for_paired_tests():
    src = (ROOT / "maxsim_knn_bench.py").read_text()
    assert "np.savez" in src, "per-query predictions discarded, so arms cannot be paired"
    assert "merged" in src, "per-query file overwrites instead of merging across invocations"
