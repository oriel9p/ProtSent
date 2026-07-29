"""The sweep must judge an arm by its latest run, not by its whole history.

protein_benchmark_suite.py APPENDS to its results CSV. After the '|' / '#'
tokenizer bug was fixed and esm2_35m/knn was re-measured, the file held 46 rows:
23 from the broken run (2 with errors) and 23 clean ones. The sweep counted
error rows across the whole file and reported "FAILED: 2 task(s) errored" for an
arm that had in fact just succeeded on all 23.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bench_arm_status import arm_status

COLS = ["Model", "Task", "Date", "Accuracy", "Error"]


def _csv(tmp_path, rows):
    p = tmp_path / "bench.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return p


def test_stale_error_rows_do_not_fail_an_arm_that_was_re_measured(tmp_path):
    """The exact esm2_35m/knn shape: broken run, then a clean rerun appended."""
    rows = [
        ["m", "peptide_hla", "2026-07-28", None, "'|'"],
        ["m", "thermostability", "2026-07-28", None, "'#'"],
        ["m", "solubility", "2026-07-28", 0.63, None],
        ["m", "peptide_hla", "2026-07-29", 0.70, None],
        ["m", "thermostability", "2026-07-29", 0.44, None],
        ["m", "solubility", "2026-07-29", 0.63, None],
    ]
    st = arm_status(_csv(tmp_path, rows), expected_tasks=3)
    assert st.complete
    assert st.failed_tasks == []
    assert st.clean_tasks == 3


def test_task_that_never_succeeded_is_reported(tmp_path):
    rows = [
        ["m", "peptide_hla", "2026-07-28", None, "'|'"],
        ["m", "peptide_hla", "2026-07-29", None, "'|'"],
        ["m", "solubility", "2026-07-29", 0.63, None],
    ]
    st = arm_status(_csv(tmp_path, rows), expected_tasks=2)
    assert not st.complete
    assert st.failed_tasks == ["peptide_hla"]


def test_missing_task_makes_the_arm_incomplete(tmp_path):
    """Fewer tasks than requested is a failure even when every row is clean.

    A crash partway through the sweep leaves a short but error-free CSV.
    """
    rows = [["m", "solubility", "2026-07-29", 0.63, None]]
    st = arm_status(_csv(tmp_path, rows), expected_tasks=3)
    assert not st.complete
    assert st.clean_tasks == 1


def test_absent_csv_is_incomplete_not_an_exception(tmp_path):
    st = arm_status(tmp_path / "nope.csv", expected_tasks=3)
    assert not st.complete
    assert st.clean_tasks == 0


def test_csv_without_an_error_column_is_read_as_all_clean(tmp_path):
    """Older CSVs predate the Error column; absent means no failures."""
    p = tmp_path / "bench.csv"
    pd.DataFrame(
        [["m", "solubility", "2026-07-29", 0.63]],
        columns=["Model", "Task", "Date", "Accuracy"],
    ).to_csv(p, index=False)
    st = arm_status(p, expected_tasks=1)
    assert st.complete


def test_empty_csv_is_incomplete(tmp_path):
    st = arm_status(_csv(tmp_path, []), expected_tasks=1)
    assert not st.complete


@pytest.mark.parametrize("expected", [0, -1])
def test_nonpositive_expectation_is_rejected(expected):
    with pytest.raises(ValueError):
        arm_status("ignored.csv", expected_tasks=expected)
