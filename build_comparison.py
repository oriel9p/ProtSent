#!/usr/bin/env python3
"""Build the paired benchmark comparison tables for the rebuttal.

Merges four arms per task, per probe:
    MMseqs2 (alignment baseline)  results/benchmarks/mmseqs_baseline.json
    ESM-2 35M (untuned base)      results/benchmarks/v3/esm2_35m_<probe>/*.csv
    ProtSent-V1-35M (published)   results/benchmarks/v3/protsent_old_<probe>/*.csv
    ProtSent-V2-35M (decontam.)   results/benchmarks/v3/protsent_v3_<probe>/*.csv

Idempotent: re-run it whenever another arm lands. Missing cells stay missing.

    uv run --no-sync python build_comparison.py
    uv run --no-sync python build_comparison.py --selfcheck
"""

import os

# 256-core box, OpenBLAS built for fewer threads: uncapped fits die with
# "corrupted size vs. prev_size" (SIGABRT). Must be set before numpy is imported.
for _v in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "32")

import argparse
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Paper naming. On disk the V2 arm is the RUN_NAME `protsent_v3`
# (models/protsent_esm2_35m_v3), inherited from an older script.
ARMS = [
    ("mmseqs", "MMseqs2"),
    ("esm2_35m", "ESM-2 35M"),
    ("protsent_old", "ProtSent-V1"),
    ("protsent_v3", "ProtSent-V2"),
]
EMBED_ARMS = ["esm2_35m", "protsent_old", "protsent_v3"]
PROBES = ["knn", "linear"]

# All main metrics in this suite are higher-is-better (AUC, F1_Macro, Spearman,
# Recall@10). Deltas are therefore plain signed differences.
TIE_TOL = 0.005

# EvalStrategy values that mean "a real held-out test split shipped with the dataset".
HELD_OUT_STRATEGIES = {"test_split", "test_split_column"}
# Retrieval has no train/test probe at all; flagged as a protocol note, not a split defect.
PROTOCOL_STRATEGIES = {"retrieval_unchanged", "proteingym_unchanged"}

SCOPE_METHOD_TO_TAG = {
    "ESM-2 35M": "esm2_35m",
    "ProtSent 35M (published)": "protsent_old",
}


def _num(x):
    """Float, or None for NaN / None / non-numeric."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def load_mmseqs(path: Path) -> dict:
    """task_key -> mmseqs row. Empty dict if the baseline has not been written."""
    if not path.exists():
        return {}
    return {r["task_key"]: r for r in json.loads(path.read_text())}


def load_arm(v3_dir: Path, tag: str, probe: str) -> dict:
    """Task display name -> latest row, for one <tag>_<probe> results dir.

    The suite appends history to a stable per-model CSV and dedups keeping the
    newest row last, so last-occurrence-wins is the newest run.
    """
    rows = {}
    for csv in sorted((v3_dir / f"{tag}_{probe}").glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:  # half-written file: the sweep is still running
            continue
        for _, r in df.iterrows():
            if "Task" in r:
                rows[str(r["Task"])] = r
    return rows


def build_probe_rows(bench: Path, probe: str) -> list:
    """One merged record per task, for a single probe protocol."""
    mmseqs = load_mmseqs(bench / "mmseqs_baseline.json")
    arms = {tag: load_arm(bench / "v3", tag, probe) for tag in EMBED_ARMS}

    # Task registry: the mmseqs baseline declares task_key/name/metric/type for
    # every paired task. Anything a CSV reports that is not in it still gets a row.
    by_name = {r["Task"]: k for k, r in mmseqs.items()}
    meta = {
        k: (r["Task"], r["problem_type"], r["main_metric"]) for k, r in mmseqs.items()
    }
    for tag in EMBED_ARMS:
        for name in arms[tag]:
            if name not in by_name:
                by_name[name] = name
                meta.setdefault(name, (name, "?", "?"))

    scope_fallback = load_scope_fallback(bench)

    out = []
    for key, (name, ptype, metric) in meta.items():
        vals, strategies, notes, recorded_probes, no_metric = {}, [], {}, [], []
        m = mmseqs.get(key)
        vals["mmseqs"] = _num(m.get(metric)) if m else None
        for tag in EMBED_ARMS:
            r = arms[tag].get(name)
            vals[tag] = _num(r.get(metric)) if r is not None else None
            if r is not None and "EvalStrategy" in r:
                strategies.append(str(r["EvalStrategy"]))
            if r is not None and "Probe" in r:
                recorded_probes.append(str(r["Probe"]))
            if vals[tag] is None and key == "scope40_retrieval":
                fb = scope_fallback.get(tag)
                if fb is not None:
                    vals[tag] = fb
                    notes[tag] = "from scope40_table.json (standalone retrieval run)"
                    # Same evaluator as the sweep; the standalone CSVs record this too.
                    strategies.append("retrieval_unchanged")
            # "The arm has not run yet" and "the arm ran but the metric could not
            # be computed" are both a blank cell, and they mean opposite things.
            if r is not None and vals[tag] is None:
                no_metric.append(tag)

        if all(v is None for v in vals.values()):
            continue  # nothing measured anywhere; listed as "no data" instead

        out.append(
            {
                "task_key": key,
                "task": name,
                "problem_type": ptype,
                "main_metric": metric,
                "probe": probe,
                # Kept per-arm-union: one arm can error (task_exception) while
                # another succeeds, and that must not be hidden by the survivor.
                "eval_strategies": sorted(set(strategies)),
                # Retrieval and multilabel tasks use a built-in evaluator and
                # ignore the requested probe: the CSV then records a different
                # probe than the directory it lives in.
                "probe_ignored": bool(recorded_probes) and probe not in recorded_probes,
                "values": vals,
                "ran_but_no_main_metric": no_metric,
                "delta_v1_minus_esm2": _delta(vals["protsent_old"], vals["esm2_35m"]),
                "delta_v2_minus_esm2": _delta(vals["protsent_v3"], vals["esm2_35m"]),
                "source_notes": notes,
            }
        )
    out.sort(key=lambda r: (r["problem_type"], r["task_key"]))
    return out


def load_scope_fallback(bench: Path) -> dict:
    """Recall@10 per tag from the standalone SCOPe-40 run, if present."""
    p = bench / "scope40_table.json"
    if not p.exists():
        return {}
    return {
        SCOPE_METHOD_TO_TAG[r["method"]]: _num(r.get("Recall@10"))
        for r in json.loads(p.read_text())
        if r.get("method") in SCOPE_METHOD_TO_TAG
    }


def _delta(a, b):
    return None if a is None or b is None else a - b


def summarize(rows: list) -> dict:
    """Win/tie/loss vs ESM-2 per ProtSent arm, plus the MMseqs2-wins count."""
    s = {"n_tasks": len(rows), "tie_tol": TIE_TOL}
    for arm, field in (("protsent_old", "delta_v1_minus_esm2"), ("protsent_v3", "delta_v2_minus_esm2")):
        d = [r[field] for r in rows if r[field] is not None]
        s[arm] = {
            "n_compared": len(d),
            "beats": sum(x > TIE_TOL for x in d),
            "ties": sum(abs(x) <= TIE_TOL for x in d),
            "loses": sum(x < -TIE_TOL for x in d),
            "median_delta": statistics.median(d) if d else None,
        }
    wins, tasks = 0, []
    for r in rows:
        mm = r["values"]["mmseqs"]
        emb = [r["values"][t] for t in EMBED_ARMS if r["values"][t] is not None]
        if mm is not None and emb and mm - max(emb) > TIE_TOL:
            wins += 1
            tasks.append(r["task_key"])
    s["mmseqs_beats_best_embedding"] = {"n": wins, "tasks": tasks}
    return s


def _fmt(v):
    return "--" if v is None else f"{v:.4f}"


def _cell(r, tag):
    """`n/a` = the arm ran and the metric was not computable; `--` = not run yet."""
    if r["values"][tag] is None and tag in r["ran_but_no_main_metric"]:
        return "n/a"
    return _fmt(r["values"][tag])


def _fmt_delta(v):
    return "--" if v is None else f"{v:+.4f}"


def render_table(rows: list) -> str:
    head = (
        "| Task | Problem type | Main metric | MMseqs2 | ESM-2 35M | ProtSent-V1 | "
        "ProtSent-V2 | V1-minus-ESM2 | V2-minus-ESM2 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        "| {task} | {ptype} | {metric} | {mm} | {esm} | {v1} | {v2} | {d1} | {d2} |\n".format(
            task=r["task"],
            ptype=r["problem_type"],
            metric=r["main_metric"],
            mm=_fmt(r["values"]["mmseqs"]),
            esm=_cell(r, "esm2_35m"),
            v1=_cell(r, "protsent_old"),
            v2=_cell(r, "protsent_v3"),
            d1=_fmt_delta(r["delta_v1_minus_esm2"]),
            d2=_fmt_delta(r["delta_v2_minus_esm2"]),
        )
        for r in rows
    )
    return head + body


def summary_line(s: dict) -> str:
    parts = []
    for arm, label in (("protsent_old", "ProtSent-V1"), ("protsent_v3", "ProtSent-V2")):
        a = s[arm]
        med = "--" if a["median_delta"] is None else f"{a['median_delta']:+.4f}"
        parts.append(
            f"**{label} vs ESM-2 35M**: {a['beats']} beats / {a['ties']} ties / "
            f"{a['loses']} loses (of {a['n_compared']} comparable tasks), median signed delta {med}"
        )
    mm = s["mmseqs_beats_best_embedding"]
    parts.append(
        f"**MMseqs2 beats the best embedding model on {mm['n']} task(s)**"
        + (f" ({', '.join(mm['tasks'])})" if mm["tasks"] else "")
    )
    return ". ".join(parts) + "."


def _metric_of(all_rows: dict, task_key: str) -> str:
    for rows in all_rows.values():
        for r in rows:
            if r["task_key"] == task_key:
                return r["main_metric"]
    return "?"


def caveats(all_rows: dict, bench: Path) -> dict:
    """Everything the 'read this before quoting' section is built from."""
    split_flags, protocol, errored, probe_ignored, no_metric = {}, {}, {}, {}, {}
    for probe, rows in all_rows.items():
        for r in rows:
            if r["probe_ignored"]:
                probe_ignored.setdefault(r["task_key"], set()).add(probe)
            for tag in r["ran_but_no_main_metric"]:
                no_metric.setdefault(r["task_key"], set()).add(f"{probe}:{tag}")
            for st in r["eval_strategies"]:
                if st == "task_exception":
                    # The suite catches per-task errors and still exits 0: a run
                    # can look clean while a task never produced a number.
                    errored.setdefault(r["task_key"], set()).add(probe)
                elif st in PROTOCOL_STRATEGIES:
                    protocol.setdefault(r["task_key"], st)
                elif st not in HELD_OUT_STRATEGIES:
                    split_flags.setdefault(r["task_key"], set()).add(f"{probe}:{st}")
    mmseqs = load_mmseqs(bench / "mmseqs_baseline.json")
    seen = {r["task_key"] for rows in all_rows.values() for r in rows}
    no_data = sorted(set(mmseqs) - seen)
    return {
        "non_heldout_eval": split_flags,
        "protocol_notes": protocol,
        "errored_tasks": errored,
        "probe_ignored": probe_ignored,
        "ran_but_no_main_metric": no_metric,
        "no_data_tasks": no_data,
    }


def render_md(all_rows: dict, summaries: dict, bench: Path) -> str:
    cav = caveats(all_rows, bench)
    split_flags = cav["non_heldout_eval"]
    protocol = cav["protocol_notes"]
    errored = cav["errored_tasks"]
    probe_ignored = cav["probe_ignored"]
    no_metric = cav["ran_but_no_main_metric"]
    no_data = cav["no_data_tasks"]
    L = []
    A = L.append
    A("# Paired benchmark comparison: MMseqs2 vs ESM-2 35M vs ProtSent\n")
    A(
        "Generated by `build_comparison.py` (re-run it to refresh; it is idempotent "
        "and fills cells in as arms land).\n"
    )
    A("**Naming.** V1 = the published `oriel9p/protsent-esm2-35M`, trained on the "
      "unfiltered corpus. V2 = `ProtSent-V2-35M (decontaminated)`, the retrain on the "
      "decontaminated corpus; on disk it is the RUN_NAME `protsent_esm2_35m_v3` / tag "
      "`protsent_v3`, inherited from an older script.\n")
    A(
        "All arms are scored on each task's declared **test** split with that task's "
        "declared `main_metric`; every metric here is higher-is-better.\n"
    )
    A(
        "Empty cells: **`--`** = that run has not landed yet (or the task was not in the "
        "sweep). **`n/a`** = the run completed but the declared main metric could not be "
        "computed for it; see 'Main metric unavailable' below. Neither is ever a zero, and "
        "neither counts as a loss in the summaries.\n"
    )

    for probe in PROBES:
        rows = all_rows[probe]
        A(f"\n## {probe} probe\n")
        if not rows:
            A("_No results yet for this probe._\n")
            continue
        A(render_table(rows))
        A(f"\n**Summary ({probe}).** {summary_line(summaries[probe])}")
        A(
            f"Beats/ties/loses use a tie band of +/-{TIE_TOL} on the main metric; "
            "tasks with a missing cell in either arm are excluded from the count.\n"
        )

    A("\n## Read this before quoting\n")
    A(
        "**kNN and linear are separate protocols.** They are reported as two tables on "
        "purpose. Do not average them or mix cells between them.\n"
    )
    A("### Eval splits that are not a real held-out test set\n")
    if split_flags:
        for k in sorted(split_flags):
            A(f"- **{k}** -- `EvalStrategy` = {', '.join(sorted(split_flags[k]))}. "
              "Not a dataset-shipped held-out test set.")
        A("")
    else:
        A("- None detected in the results present so far (checked the `EvalStrategy` "
          "column of every CSV merged here). Re-check once the remaining arms land.\n")
    A(
        "- **thermostability** is configured `auto_split=True`: the suite makes a seeded "
        "80/20 split of *train* and calls the 20% the test split (`EvalStrategy = "
        "test_random_split`). It is not a held-out test set, and MMseqs2 is scored on the "
        "same construction, so the arms are comparable to each other but not to published "
        "FLIP numbers.\n"
    )
    if protocol or probe_ignored:
        A("### Protocol notes (not split defects)\n")
        for k, st in sorted(protocol.items()):
            A(f"- **{k}** -- `EvalStrategy` = `{st}` (no train/test probe fit; the "
              "task's own evaluator scores it).")
        if probe_ignored:
            A(
                "- **The requested probe was ignored on: "
                + ", ".join(f"`{k}`" for k in sorted(probe_ignored))
                + "**. Retrieval and multilabel tasks use a built-in linear evaluator "
                "regardless of `-p knn`, and the CSV records `Probe=linear` even inside a "
                "`*_knn/` directory. Those rows are therefore **identical** in the kNN and "
                "linear tables -- they are one measurement printed twice, not two."
            )
        A("")
    if errored:
        A("### Tasks that errored during the sweep (cell is `--`, not a result)\n")
        for k in sorted(errored):
            A(f"- **{k}** -- `EvalStrategy = task_exception` in: {', '.join(sorted(errored[k]))}.")
        A("")
    if no_metric:
        A("### Main metric unavailable (`n/a` cells) -- do NOT read these as a model failure\n")
        for k in sorted(no_metric):
            A(f"- **{k}** -- no `{_metric_of(all_rows, k)}` from: "
              f"{', '.join(sorted(no_metric[k]))}.")
        A(
            "\nThe usual cause is the multiclass AUC path: the probe's `predict_proba` "
            "returns one column per class **seen in train**, and `roc_auc_score(..., "
            "multi_class='ovr')` refuses when the test set contains a class the probe never "
            "saw (\"Number of classes in y_true not equal to the number of columns in "
            "'y_score'\"). The suite logs a warning, drops AUC, and keeps Accuracy/F1. "
            "MMseqs2 scores those same tasks by a different route and does have an AUC, so "
            "the MMseqs2 column is populated where the embedding columns are `n/a`. "
            "**That is not MMseqs2 winning** -- there is no paired number to compare, and "
            "these tasks are excluded from every count in the summary lines. Quote "
            "Accuracy / F1_Macro from the raw CSVs if you need those tasks.\n"
        )
    A("### Metrics that are not comparable to published literature\n")
    A(
        "- **remote_homology** -- our test split is TAPE's three holdouts *pooled* "
        "(718 fold + 1,254 superfamily + 1,272 family = 3,244 sequences), scored as a "
        "457-class macro one-vs-rest AUC. Published TAPE remote-homology numbers are "
        "**per-holdout top-1 accuracy**. The two are not comparable; quote this number only "
        "against the other arms in this table.\n"
    )
    A(
        "- **thermostability** -- see above; the 80/20 train resplit is not the FLIP test set.\n"
    )
    if no_data:
        A("### Tasks with no data in any arm (omitted from the tables)\n")
        for k in no_data:
            A(f"- **{k}**")
        A(
            "\n`rhla_enzyme_mutations` is expected here: its sequences are 6-residue "
            "mutation-site strings, too short for MMseqs2 k-mers (hit coverage 0.0), so it "
            "is excluded from the paired sweep.\n"
        )
    A(
        "\n### Provenance of individual cells\n"
        "- Cells marked in `comparison.json` under `source_notes` did not come from the "
        "paired sweep. Currently that is only the SCOPe-40 row when the sweep has not "
        "reached it, backfilled from `scope40_table.json` (same evaluator, standalone run).\n"
    )
    return "\n".join(L) + "\n"


def build(bench: Path) -> tuple:
    all_rows = {p: build_probe_rows(bench, p) for p in PROBES}
    summaries = {p: summarize(all_rows[p]) for p in PROBES}
    return all_rows, summaries


def write_outputs(bench: Path) -> tuple:
    all_rows, summaries = build(bench)
    (bench / "COMPARISON.md").write_text(render_md(all_rows, summaries, bench))
    cav = {
        k: ({kk: sorted(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
        for k, v in caveats(all_rows, bench).items()
    }
    cav["not_literature_comparable"] = {
        "remote_homology": "pooled TAPE holdouts (718 fold + 1254 superfamily + 1272 family), "
        "457-class macro AUC; published numbers are per-holdout top-1 accuracy",
        "thermostability": "seeded 80/20 resplit of train, not the FLIP test set",
    }
    (bench / "comparison.json").write_text(
        json.dumps(
            {
                "naming": {
                    "ProtSent-V1": "oriel9p/protsent-esm2-35M (published, unfiltered corpus)",
                    "ProtSent-V2": "ProtSent-V2-35M (decontaminated); on disk protsent_esm2_35m_v3 / tag protsent_v3",
                },
                "tie_tol": TIE_TOL,
                "arms": dict(ARMS),
                "tables": all_rows,
                "summary": summaries,
                "caveats": cav,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return all_rows, summaries


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #


def selfcheck() -> None:
    """Exercise the merge on synthetic rows: missing cells, deltas, newest-row-wins."""
    tmp = Path(tempfile.mkdtemp(prefix="cmp_selfcheck_"))
    try:
        (tmp / "v3").mkdir(parents=True)
        (tmp / "mmseqs_baseline.json").write_text(
            json.dumps(
                [
                    {"task_key": "t_auc", "Task": "T AUC", "main_metric": "AUC",
                     "problem_type": "binary", "AUC": 0.60},
                    {"task_key": "t_spear", "Task": "T Spear", "main_metric": "Spearman",
                     "problem_type": "regression", "Spearman": 0.90},
                    {"task_key": "t_empty", "Task": "T Empty", "main_metric": "AUC",
                     "problem_type": "binary", "AUC": None},
                ]
            )
        )

        def csv(tag, probe, text):
            d = tmp / "v3" / f"{tag}_{probe}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"bench_{tag}.csv").write_text(text)

        hdr = "Model,Task,Date,Probe,EvalSplit,EvalStrategy,AUC,Spearman\n"
        # esm2: both tasks. Two dated rows for T AUC -- the newest (last) must win.
        csv("esm2_35m", "knn", hdr
            + "e,T AUC,2026-07-01,knn,test,test_split,0.11,\n"
            + "e,T AUC,2026-07-02,knn,test,test_split,0.70,\n"
            + "e,T Spear,2026-07-02,knn,test,test_random_split,,0.50\n")
        # V1: beats on one, loses on the other.
        csv("protsent_old", "knn", hdr
            + "o,T AUC,2026-07-02,knn,test,test_split,0.80,\n"
            + "o,T Spear,2026-07-02,knn,test,test_random_split,,0.45\n")
        # V2 arm: T AUC errored (suite writes the row and still exits 0).
        csv("protsent_v3", "knn", hdr + "v,T AUC,2026-07-02,knn,test,task_exception,,\n")

        rows = build_probe_rows(tmp, "knn")
        got = {r["task_key"]: r for r in rows}

        assert "t_empty" not in got, "all-missing task must be dropped, not rendered"
        assert set(got) == {"t_auc", "t_spear"}, got.keys()

        a = got["t_auc"]
        assert a["values"]["esm2_35m"] == 0.70, "newest row must win"
        assert a["values"]["protsent_old"] == 0.80
        assert a["values"]["protsent_v3"] is None, "absent arm must stay None"
        assert a["values"]["mmseqs"] == 0.60
        assert abs(a["delta_v1_minus_esm2"] - 0.10) < 1e-9
        assert a["delta_v2_minus_esm2"] is None, "delta against a missing arm must be None"

        s = got["t_spear"]
        assert s["main_metric"] == "Spearman" and s["values"]["esm2_35m"] == 0.50
        assert abs(s["delta_v1_minus_esm2"] + 0.05) < 1e-9

        summ = summarize(rows)
        v1 = summ["protsent_old"]
        assert (v1["n_compared"], v1["beats"], v1["ties"], v1["loses"]) == (2, 1, 0, 1), v1
        assert abs(v1["median_delta"] - 0.025) < 1e-9, v1  # median of +0.10 and -0.05
        assert summ["protsent_v3"]["n_compared"] == 0
        assert summ["protsent_v3"]["median_delta"] is None
        # mmseqs 0.90 > best embedding 0.50 on t_spear; 0.60 < 0.80 on t_auc.
        assert summ["mmseqs_beats_best_embedding"] == {"n": 1, "tasks": ["t_spear"]}

        # Tie band: a delta inside +/-TIE_TOL counts as a tie, not a win.
        tie = [{"delta_v1_minus_esm2": TIE_TOL / 2, "delta_v2_minus_esm2": None,
                "values": {"mmseqs": None, "esm2_35m": 1.0, "protsent_old": 1.0,
                           "protsent_v3": None}}]
        assert summarize(tie)["protsent_old"]["ties"] == 1

        # Rendering must never print a missing cell as a number.
        md = render_table(rows)
        assert "--" in md and "nan" not in md.lower()

        # Caveat bucketing: an errored arm must not be hidden by a healthy one,
        # and a seeded train resplit must be flagged as not-held-out.
        cav = caveats({"knn": rows}, tmp)
        assert cav["errored_tasks"] == {"t_auc": {"knn"}}, cav
        # V2 ran T AUC but wrote no AUC -> "n/a", not "not run yet".
        assert cav["ran_but_no_main_metric"] == {"t_auc": {"knn:protsent_v3"}}, cav
        assert _cell(got["t_auc"], "protsent_v3") == "n/a"
        assert _cell(got["t_spear"], "protsent_v3") == "--", "absent arm stays --"
        assert cav["non_heldout_eval"] == {"t_spear": {"knn:test_random_split"}}, cav
        assert cav["protocol_notes"] == {} and cav["no_data_tasks"] == ["t_empty"], cav
        # A CSV row recording a different probe than its directory must be flagged.
        assert cav["probe_ignored"] == {}, cav
        csv("esm2_35m", "linear", hdr + "e,T AUC,2026-07-02,linear,test,test_split,0.70,\n")
        csv("protsent_old", "linear", hdr + "o,T AUC,2026-07-02,knn,test,test_split,0.80,\n")
        lin = caveats({"linear": build_probe_rows(tmp, "linear")}, tmp)
        assert lin["probe_ignored"] == {}, "probe recorded by at least one arm -> no flag"
        csv("esm2_35m", "linear", hdr + "e,T AUC,2026-07-02,knn,test,test_split,0.70,\n")
        lin = caveats({"linear": build_probe_rows(tmp, "linear")}, tmp)
        assert lin["probe_ignored"] == {"t_auc": {"linear"}}, lin

        # Empty inputs must not explode (the sweep is mid-flight most of the time).
        empty = Path(tempfile.mkdtemp(prefix="cmp_empty_"))
        try:
            assert build_probe_rows(empty, "knn") == []
            assert summarize([])["protsent_old"]["median_delta"] is None
        finally:
            shutil.rmtree(empty)
        print("selfcheck OK")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default=str(ROOT / "results" / "benchmarks"))
    ap.add_argument("--selfcheck", action="store_true", help="verify merge logic on synthetic rows")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    bench = Path(args.bench_dir)
    all_rows, summaries = write_outputs(bench)
    for probe in PROBES:
        print(f"[{probe}] {len(all_rows[probe])} task rows")
        print("  " + summary_line(summaries[probe]))
    print(f"wrote {bench/'COMPARISON.md'} and {bench/'comparison.json'}")


if __name__ == "__main__":
    main()
