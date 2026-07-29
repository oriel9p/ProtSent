#!/usr/bin/env python
"""Decide whether one benchmark arm succeeded, from its appended results CSV.

protein_benchmark_suite.py appends to its CSV rather than overwriting, and it
catches per-task exceptions into an "Error" column while still exiting 0. Two
consequences the sweep has to handle:

* exit code alone proves nothing -- a whole arm can report rc=0 with every row
  a failure;
* error rows are permanent. After the '|' / '#' tokenizer bug was fixed and
  esm2_35m/knn was re-measured, the CSV held 23 broken rows and 23 clean ones,
  and counting error rows over the whole file reported "FAILED: 2 task(s)
  errored" for an arm that had just succeeded on all 23.

So an arm is complete when every requested task has at least one error-free row
*somewhere* in the file, and a task has failed only when it has no clean row at
all. Row counts cannot express either condition.

Usage (exit 0 = complete, 1 = not):
    python bench_arm_status.py <results.csv> <expected_task_count>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ArmStatus:
    complete: bool
    clean_tasks: int
    failed_tasks: list[str] = field(default_factory=list)


def arm_status(csv_path: str | Path, expected_tasks: int) -> ArmStatus:
    if expected_tasks <= 0:
        raise ValueError(f"expected_tasks must be positive, got {expected_tasks}")

    path = Path(csv_path)
    if not path.is_file():
        return ArmStatus(complete=False, clean_tasks=0)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return ArmStatus(complete=False, clean_tasks=0)
    if df.empty or "Task" not in df.columns:
        return ArmStatus(complete=False, clean_tasks=0)

    err_col = next((c for c in df.columns if "rror" in c), None)
    clean = df[df[err_col].isna()] if err_col else df

    clean_names = set(clean["Task"].dropna())
    failed = sorted(set(df["Task"].dropna()) - clean_names)
    return ArmStatus(
        complete=len(clean_names) >= expected_tasks,
        clean_tasks=len(clean_names),
        failed_tasks=failed,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    st = arm_status(sys.argv[1], int(sys.argv[2]))
    if st.failed_tasks:
        print(f"never succeeded: {', '.join(st.failed_tasks)}", file=sys.stderr)
    print(f"{st.clean_tasks} task(s) with a clean result")
    raise SystemExit(0 if st.complete else 1)
