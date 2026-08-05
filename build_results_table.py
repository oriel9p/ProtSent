"""Consolidate every benchmark arm into one normalised table.

The measured numbers live in ~90 files of three shapes -- per-arm suite CSVs,
bootstrap-CI JSONs, and alignment-baseline JSONs -- with names that encode the
run rather than the model. That is fine as an audit trail and unreadable as a
result. This emits one long-format CSV plus a Markdown view of it, so a reader
gets every model x probe x task in one place without opening any of them.

Reads only. Writes results/RESULTS.csv and results/RESULTS.md.

    uv run --no-sync python build_results_table.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
BENCH = ROOT / "results" / "benchmarks"

# Suite arms, as (display model, scale, results dir prefix). The directory names
# carry historical run names -- "protsent_old" is V1, "protsent_v3" is V2 -- so
# the mapping to paper names has to be explicit or the table lies.
ARMS = [
    ("ESM-2 35M", "35M", BENCH / "v3", "esm2_35m"),
    ("ProtSent-V1", "35M", BENCH / "v3", "protsent_old"),
    ("ProtSent-V2", "35M", BENCH / "v3", "protsent_v3"),
    ("ProtSent-V2.5", "35M", BENCH / "v3", "protsent_v2p5"),
    ("ProtSent-V2.5-noGOR", "35M", BENCH / "v3", "protsent_v2p5_nogor"),
    ("ESM-2 150M", "150M", BENCH / "v2_150m", "esm2_150m"),
    ("ProtSent-V1", "150M", BENCH / "v2_150m", "protsent_v1_150m"),
    ("ProtSent-V2", "150M", BENCH / "v2_150m", "protsent_v2_150m"),
    ("ProtSent-V2.5", "150M", BENCH / "v2_150m", "protsent_v2p5_150m"),
    # Different model family and a third scale. CLAUDE.md: this is NOT a
    # controlled comparison against the ProtSent line -- it crosses family and
    # scale at once, and raw mean-pooled ESM-C is weak at retrieval to begin
    # with, below ESM-2 35M. Kept in the table because the numbers are real and
    # a reader looking for them should not have to hunt; labelled so nobody
    # reads a head-to-head into it.
    ("ESM-C 300M (uncontrolled)", "300M", BENCH / "ism", "esmc_300m"),
    ("ISM-C 300M (uncontrolled)", "300M", BENCH / "ism", "ismc_300m"),
]

RETRIEVAL_TASK = "SCOPe-40 Structural Retrieval"
# Three tasks have an undefined one-vs-rest AUC because a test-split class is
# absent from train; published tallies cover the other 20 and report these
# separately (CLAUDE.md).
UNDEFINED_AUC = {
    "Antibiotic Resistance",
    "Remote Homology (Fold)",
    "Temperature Stability",
}
PRIMARY = ["AUC", "Spearman", "F1_Macro", "Accuracy"]
FALLBACK = ["Accuracy", "Spearman", "F1_Macro"]


def _load_arm(directory: Path, prefix: str, probe: str) -> pd.DataFrame | None:
    files = glob.glob(str(directory / f"{prefix}_{probe}" / "*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "Error" in df:
        df = df[df["Error"].isna()]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # The suite appends and dedups keeping newest, so a rerun leaves stale rows.
    return df.sort_values("Date").drop_duplicates("Task", keep="last").set_index("Task")


def _primary_metric(row: pd.Series, task: str) -> tuple[str | None, float | None]:
    if task == RETRIEVAL_TASK:
        return "eligible_MAP", float(row["eligible_MAP"])
    for metric in FALLBACK if task in UNDEFINED_AUC else PRIMARY:
        if metric in row and pd.notna(row[metric]):
            return metric, float(row[metric])
    return None, None


def suite_rows() -> list[dict]:
    out: list[dict] = []
    for model, scale, directory, prefix in ARMS:
        for probe in ("knn", "linear"):
            df = _load_arm(directory, prefix, probe)
            if df is None:
                continue
            for task in df.index:
                metric, value = _primary_metric(df.loc[task], task)
                if metric is None:
                    continue
                out.append(
                    {
                        "model": model,
                        "scale": scale,
                        "probe": probe,
                        "task": task,
                        "metric": metric,
                        "value": round(value, 4),
                        "auc_undefined": task in UNDEFINED_AUC,
                    }
                )
    return out


def retrieval_rows() -> list[dict]:
    """SCOPe-40, eligible-query figures only.

    Unrestricted values are exactly eligible x 1693/2207 -- a query with no
    same-family protein in the gallery cannot succeed at any k -- so reporting
    both invites mixing them (CLAUDE.md).
    """
    out: list[dict] = []
    for model, scale, directory, prefix in ARMS:
        df = _load_arm(directory, prefix, "knn")  # retrieval ignores --probe_type
        if df is None or RETRIEVAL_TASK not in df.index:
            continue
        row = df.loc[RETRIEVAL_TASK]
        for col, name in (
            ("eligible_Recall@1", "R@1"),
            ("eligible_Recall@10", "R@10"),
            ("eligible_Recall@30", "R@30"),
            ("eligible_MAP", "MAP"),
        ):
            if col in row and pd.notna(row[col]):
                out.append(
                    {
                        "model": model,
                        "scale": scale,
                        "metric": name,
                        "value": round(float(row[col]), 4),
                        "source": "suite",
                    }
                )
    # Alignment baselines. hmmer_maxsens.json is the filters-off run: the
    # default-filter run is weaker, and quoting it once produced a published
    # claim that ProtSent beats profile search at top-1, which it does not.
    hmmer = json.loads((BENCH / "hmmer_maxsens.json").read_text())["eligible"]
    for key, name in (("hit1", "R@1"), ("hit10", "R@10"), ("hit30", "R@30"), ("ap", "MAP")):
        out.append(
            {
                "model": "HMMER (phmmer, filters off)",
                "scale": "n/a",
                "metric": name,
                "value": round(hmmer[key], 4),
                "source": "hmmer_maxsens.json",
            }
        )
    # MMseqs2 from the 2026-07-31 scoring, not mmseqs_baseline.json, whose hit30
    # differs by 0.021 under an earlier scoring (CLAUDE.md).
    mm = json.loads((BENCH / "scope40_bootstrap_ci_150m.json").read_text())["marginal"]["MMseqs2"]
    for key, name in (("hit1", "R@1"), ("hit10", "R@10"), ("hit30", "R@30"), ("ap", "MAP")):
        out.append(
            {
                "model": "MMseqs2 (-s 7.5)",
                "scale": "n/a",
                "metric": name,
                "value": round(mm[key]["mean"], 4),
                "source": "scope40_bootstrap_ci_150m.json",
            }
        )
    return out


def ci_rows() -> list[dict]:
    """Paired bootstrap deltas, which are the only resolved comparisons we have."""
    out: list[dict] = []
    for path in sorted(BENCH.glob("scope40_bootstrap_ci*.json")):
        data = json.loads(path.read_text())
        for comparison, metrics in data.get("paired", {}).items():
            for metric, v in metrics.items():
                if not isinstance(v, dict) or "lo" not in v:
                    continue
                lo, hi = v["lo"], v["hi"]
                out.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        # No silent 0.0 default: an unrecognised key would read
                        # as "no effect" rather than "not parsed".
                        "delta": round(v["mean"] if "mean" in v else v["delta"], 4),
                        "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4),
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "source": path.name,
                    }
                )
    return out


def _md(df: pd.DataFrame, index_names: list[str]) -> str:
    """Markdown table from a frame. Local, so pandas' optional tabulate dep stays optional."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |",
            "|" + "|".join("---" if c in index_names else "---:" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                 for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render(suite: pd.DataFrame, retrieval: pd.DataFrame, cis: pd.DataFrame) -> str:
    lines = [
        "# ProtSent benchmark results",
        "",
        "Generated by `build_results_table.py`. Long-format data in `RESULTS.csv`.",
        "",
        "SCOPe-40 figures are eligible-query only (n=1,693 of 2,207). The 23-task",
        "suite reports its primary metric per task; the three tasks with an undefined",
        "one-vs-rest AUC are marked and excluded from published tallies.",
        "",
        "**No inferential claim follows from the 23-task aggregates below.** A sign",
        "test resolves almost none of the win/tie/loss records, so comparative",
        "adjectives over them are not supportable.",
        "",
        "**ESM-C / ISM-C rows are not a controlled comparison.** They cross model",
        "family and scale at once, and raw mean-pooled ESM-C is weak at retrieval to",
        "begin with -- below ESM-2 35M. They do not establish anything about",
        "contrastive post-training versus structure distillation.",
        "",
        "CATH / ProtTucker-EAT results are a separate protocol and live in",
        "`results/benchmarks/cath_eat/CATH_COMPARISON.md` and `CATH_LEVELS.md`.",
        "",
        "## SCOPe-40 retrieval",
        "",
    ]
    piv = retrieval.pivot_table(
        index=["scale", "model"], columns="metric", values="value"
    ).reindex(columns=["R@1", "R@10", "R@30", "MAP"])
    lines += [_md(piv.round(4), ["scale", "model"]), ""]

    if not cis.empty:
        lines += ["## Paired bootstrap comparisons", "",
                  _md(cis.set_index("comparison"), ["comparison", "metric", "source"]), ""]

    # Iterate the scales the data actually has. A literal list here silently
    # dropped the 300M arms from the Markdown while leaving them in the CSV.
    for scale in sorted(suite.scale.unique(), key=lambda s: int(s.rstrip("M"))):
        for probe in ("knn", "linear"):
            sub = suite[(suite.scale == scale) & (suite.probe == probe)]
            if sub.empty:
                continue
            lines += [f"## {scale}, {probe} probe", ""]
            table = sub.pivot_table(index=["task", "metric"], columns="model", values="value")
            lines += [_md(table.round(4), ["task", "metric"]), ""]
    return "\n".join(lines)


def selfcheck() -> int:
    """Guard the two mistakes this table exists to prevent."""
    retrieval = pd.DataFrame(retrieval_rows())
    hm = retrieval[(retrieval.model.str.startswith("HMMER")) & (retrieval.metric == "R@1")]
    assert abs(float(hm.value.iloc[0]) - 0.7525) < 1e-3, f"HMMER R@1 wrong: {hm.value.iloc[0]}"
    mm = retrieval[(retrieval.model.str.startswith("MMseqs2")) & (retrieval.metric == "R@1")]
    assert abs(float(mm.value.iloc[0]) - 0.6556) < 1e-3, f"MMseqs2 R@1 wrong: {mm.value.iloc[0]}"
    suite = pd.DataFrame(suite_rows())
    assert not suite.empty, "no suite rows"
    assert suite.value.between(-1, 1).all(), "metric outside [-1, 1]"
    print(f"selfcheck ok: {len(suite)} suite rows, {len(retrieval)} retrieval rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()

    suite = pd.DataFrame(suite_rows())
    retrieval = pd.DataFrame(retrieval_rows())
    cis = pd.DataFrame(ci_rows())

    out_csv = ROOT / "results" / "RESULTS.csv"
    long = pd.concat(
        [
            suite.assign(kind="suite"),
            retrieval.assign(kind="retrieval", probe="n/a", task=RETRIEVAL_TASK),
        ],
        ignore_index=True,
    )
    long.to_csv(out_csv, index=False)
    (ROOT / "results" / "RESULTS.md").write_text(render(suite, retrieval, cis) + "\n")
    print(f"wrote {out_csv} ({len(long)} rows) and results/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
