#!/usr/bin/env python3
"""Compare ISM-C-300M against its matched control and against our own arms.

Reviewer jVGf asked us to position ProtSent against structure-informed protein
LMs. ISM is the only one of the four they named with usable public weights.
This builds two tables from already-computed results, running nothing:

  1. Structure distillation, at fixed scale and architecture:
        ISM-C-300M  vs  vanilla ESM-C-300M
     Same parameter count, same tokenizer, same code path. The difference is
     the distillation and nothing else, so this is the only pairing that
     licenses a claim about ISM specifically.

  2. Where both sit relative to our arms and the alignment baselines. These
     differ in BOTH family and scale -- ESM-C 300M against ESM-2 at 150M and
     35M -- so they are context, not a controlled comparison. No claim of the
     form "contrastive training beats structure distillation" follows from them
     alone: raw mean-pooled ESM-C is simply weak at retrieval, below even ESM-2
     35M. Separating the two needs ProtSent post-training on an ESM-C backbone.

Deliberately not folded into build_comparison.py: that script is hard-wired to
the 35M directory tree, a fixed three-arm EMBED_ARMS, and named
delta_v{1,2}_minus_esm2 fields. Generalising it for a differently-shaped
question would cost more than a separate reader, and would risk the numbers
already published from it.

HMMER is read from hmmer_maxsens.json, the FILTERS-OFF run (eligible R@1
0.7525). The default-filter run is weaker and quoting it once produced a
published false claim that we beat profile search at top-1. We do not.

    uv run --no-sync python ism_comparison.py
    uv run --no-sync python ism_comparison.py --selfcheck
"""

import os

for _v in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "32")

import argparse
import json
from pathlib import Path

from build_comparison import _num, load_arm, load_mmseqs

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "results" / "benchmarks"

# (results subdirectory, tag, display name). Order is the table order.
ARMS = [
    ("ism", "esmc_300m", "ESM-C 300M"),
    ("ism", "ismc_300m", "ISM-C 300M"),
    ("v2_150m", "esm2_150m", "ESM-2 150M"),
    ("v2_150m", "protsent_v1_150m", "ProtSent-V1 150M"),
    ("v2_150m", "protsent_v2_150m", "ProtSent-V2 150M"),
    ("v3", "esm2_35m", "ESM-2 35M"),
    ("v3", "protsent_v3", "ProtSent-V2 35M"),
]
# The controlled pair: everything else in the table is context.
CONTROL, TREATMENT = "esmc_300m", "ismc_300m"
PROBES = ["knn", "linear"]
TIE_TOL = 0.005

SCOPE_TASK = "SCOPe-40 Structural Retrieval"
# Restricted to the 1,693 of 2,207 queries that have a non-self same-family
# protein in the gallery. The unrestricted numbers count impossible queries as
# failures and are not comparable across methods with different hit coverage.
SCOPE_METRICS = [
    ("eligible_Recall@1", "R@1"),
    ("eligible_Recall@10", "R@10"),
    ("eligible_MAP", "MAP"),
]
# SCOPe-40's declared main_metric is the UNRESTRICTED Recall@10, which counts the
# 514 queries with no same-family protein in the gallery as failures. Reporting
# that in the per-task delta table while the retrieval table above reports the
# eligible-only figure puts two different numbers under one task name with no
# label saying why. Use the eligible metric in both. (It does not change the
# sign here -- ISM-C leads by +0.0612 unrestricted and +0.0797 eligible.)
SCOPE_DELTA_METRIC = "eligible_Recall@10"

# Three multiclass tasks -- Antibiotic Resistance, Remote Homology (Fold),
# Temperature Stability -- declare AUC as their main metric but the suite leaves
# that column empty for them, recording Accuracy/F1 instead. Comparing on the
# declared metric therefore drops them from every tally, silently and uniformly
# across arms. Remote Homology is the paper's flagship task, so dropping it is
# not acceptable. Fall back to Accuracy, which is populated, and say so.
METRIC_FALLBACK = "Accuracy"


def _metric_value(row, metric):
    """Value on `metric`, falling back to Accuracy when the column is empty.

    Returns (value, metric_actually_used) so callers can label the row honestly.
    """
    v = _num(row.get(metric))
    if v is not None:
        return v, metric
    v = _num(row.get(METRIC_FALLBACK))
    return (v, METRIC_FALLBACK) if v is not None else (None, metric)


def collect(probe: str) -> dict:
    """tag -> {task name -> row} for every arm that has results for this probe."""
    return {tag: load_arm(BENCH / sub, tag, probe) for sub, tag, _ in ARMS}


def _mmseqs_scope_later(metric: str = "hit10"):
    """MMseqs2's SCOPe-40 figure from the later of the two scorings, or None."""
    p = BENCH / "scope40_bootstrap_ci_150m.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text()).get("marginal", {}).get("MMseqs2")
    return _num(m[metric]["mean"]) if m and metric in m else None


def scope_rows() -> list:
    """SCOPe-40 retrieval, all methods. Retrieval ignores the probe flag."""
    rows = []
    # MMseqs2 on SCOPe-40 was scored twice, by two implementations that disagree:
    # mmseqs_baseline.json (2026-07-29) gives eligible 0.6556/0.7348/0.7354/0.4041,
    # bootstrap_ci.py's own hit-table scoring (2026-07-31) gives
    # 0.6556/0.7401/0.7566/0.4098. hit30 differs by 0.021, so this is a genuine
    # rescoring, not rounding. Use the LATER one -- it is also the row already
    # published in FINAL_rebuttal.md, so quoting the other would put two numbers
    # for one method in front of the same reviewer.
    mm_ci = BENCH / "scope40_bootstrap_ci_150m.json"
    if mm_ci.exists():
        m = json.loads(mm_ci.read_text())["marginal"].get("MMseqs2")
        if m:
            rows.append(
                ("MMseqs2 (-s 7.5)", [_num(m[k]["mean"]) for k in ("hit1", "hit10", "ap")])
            )
    hp = BENCH / "hmmer_maxsens.json"
    if hp.exists():
        e = json.loads(hp.read_text())["eligible"]
        rows.append(("HMMER (phmmer, filters off)", [_num(e.get(k)) for k in ("hit1", "hit10", "ap")]))

    arms = collect("knn")
    for _, tag, label in ARMS:
        r = arms.get(tag, {}).get(SCOPE_TASK)
        if r is not None:
            rows.append((label, [_num(r.get(k)) for k, _ in SCOPE_METRICS]))
    return rows


def distillation_delta(probe: str) -> list:
    """Per-task ISM-C minus vanilla ESM-C, on the task's own main metric."""
    arms = collect(probe)
    mmseqs = load_mmseqs(BENCH / "mmseqs_baseline.json")
    metric_of = {r["Task"]: r["main_metric"] for r in mmseqs.values()}

    out = []
    control, treated = arms.get(CONTROL, {}), arms.get(TREATMENT, {})
    for task in sorted(set(control) & set(treated)):
        metric = SCOPE_DELTA_METRIC if task == SCOPE_TASK else metric_of.get(task)
        if metric is None:
            continue
        a, used_a = _metric_value(treated[task], metric)
        b, used_b = _metric_value(control[task], metric)
        # Both arms must be scored on the SAME metric or the delta is meaningless.
        if a is None or b is None or used_a != used_b:
            continue
        out.append({"task": task, "metric": used_a, "ismc": a, "esmc": b, "delta": a - b})
    out.sort(key=lambda r: r["delta"])
    return out


def cross_model(probe: str) -> list:
    """One row per task with every arm side by side, on a shared metric."""
    arms = collect(probe)
    metric_of = {
        r["Task"]: r["main_metric"] for r in load_mmseqs(BENCH / "mmseqs_baseline.json").values()
    }
    mmseqs = {r["Task"]: r for r in load_mmseqs(BENCH / "mmseqs_baseline.json").values()}
    present = [(tag, label) for _, tag, label in ARMS if arms.get(tag)]
    if not present:
        return []

    out = []
    for task in sorted(set.intersection(*[set(arms[t]) for t, _ in present])):
        metric = SCOPE_DELTA_METRIC if task == SCOPE_TASK else metric_of.get(task)
        if metric is None:
            continue
        vals, used = {}, set()
        for tag, label in present:
            v, m = _metric_value(arms[tag][task], metric)
            vals[label] = v
            if v is not None:
                used.add(m)
        if len(used) != 1:  # arms scored on different metrics are not comparable
            continue
        mm_row = mmseqs.get(task)
        vals["MMseqs2"] = _metric_value(mm_row, metric)[0] if mm_row else None
        if task == SCOPE_TASK:
            # Same two-implementation disagreement as in scope_rows(); take the
            # later scoring here too, or one table says 0.740 and the other 0.735.
            vals["MMseqs2"] = _mmseqs_scope_later() or vals["MMseqs2"]
        out.append({"task": task, "metric": used.pop(), "values": vals})
    return out


HEAD_TO_HEAD = "protsent_v2_150m"
# The three AUC-undefined tasks are excluded here for the same reason as in the
# trade-off table: FINAL_rebuttal.md:52 already tells reviewers they are out.
_AUC_UNDEFINED = {
    "Antibiotic Resistance",
    "Remote Homology (Fold)",
    "Temperature Stability",
}


def head_to_head(probe: str) -> list:
    """ProtSent-V2-150M minus ESM-C and minus ISM-C, per task.

    A direct read of how our best model sits against the new baselines. Both
    columns cross model family and scale, so this is a comparison of levels, not
    a controlled experiment -- see the module docstring.
    """
    arms = collect(probe)
    metric_of = {
        r["Task"]: r["main_metric"] for r in load_mmseqs(BENCH / "mmseqs_baseline.json").values()
    }
    ours = arms.get(HEAD_TO_HEAD, {})
    out = []
    for task in sorted(set(ours) & set(arms.get(CONTROL, {})) & set(arms.get(TREATMENT, {}))):
        if task in _AUC_UNDEFINED:
            continue
        metric = SCOPE_DELTA_METRIC if task == SCOPE_TASK else metric_of.get(task)
        if metric is None:
            continue
        v, m0 = _metric_value(ours[task], metric)
        e, m1 = _metric_value(arms[CONTROL][task], metric)
        i, m2 = _metric_value(arms[TREATMENT][task], metric)
        if None in (v, e, i) or not (m0 == m1 == m2):
            continue
        out.append(
            {"task": task, "metric": m0, "ours": v, "esmc": e, "ismc": i,
             "d_esmc": v - e, "d_ismc": v - i}
        )
    out.sort(key=lambda r: -r["d_ismc"])
    return out


def tally(deltas: list) -> dict:
    d = [r["delta"] for r in deltas]
    return {
        "n": len(d),
        "ismc_wins": sum(x > TIE_TOL for x in d),
        "ties": sum(abs(x) <= TIE_TOL for x in d),
        "ismc_loses": sum(x < -TIE_TOL for x in d),
        "median_delta": (sorted(d)[len(d) // 2] if d else None),
    }


def _f(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def render(report: dict) -> str:
    L = ["# ISM-C-300M on the ProtSent benchmark suite", ""]
    L += [
        "ISM-C-300M is a structure-distilled ESM-C-300M. `Synthyra/ESMplusplus_small`",
        "is vanilla ESM-C-300M: same architecture, parameter count, tokenizer and",
        "code path, so that pairing isolates the distillation and nothing else.",
        "",
        "The ProtSent and ESM-2 rows differ from the ESM-C rows in BOTH family and",
        "scale. They are context, not a controlled comparison, and no claim of the",
        'form "contrastive training beats structure distillation" follows from them',
        "alone -- raw mean-pooled ESM-C is simply weak at retrieval, below even",
        "ESM-2 35M. Separating the two needs ProtSent post-training on ESM-C.",
        "",
        "Three of the 20 rows below (EC, GO, SCOPe-40) are probe-invariant by",
        "construction: multilabel and retrieval tasks use a built-in evaluator and",
        "ignore the --probe_type flag, so their knn and linear numbers are identical.",
        "",
        "## SCOPe-40 structural retrieval",
        "",
        "Test split, self excluded, restricted to the 1,693 of 2,207 queries with a",
        "non-self same-family protein in the gallery.",
        "",
        "| method | R@1 | R@10 | MAP |",
        "|---|---|---|---|",
    ]
    for label, vals in report["scope40"]:
        L.append(f"| {label} | " + " | ".join(_f(v) for v in vals) + " |")

    for probe in PROBES:
        deltas = report["distillation"][probe]
        t = report["tally"][probe]
        L += [
            "",
            f"## Structure distillation on the 23-task suite ({probe} probe)",
            "",
            f"ISM-C beats vanilla ESM-C on **{t['ismc_wins']}** tasks, ties on"
            f" **{t['ties']}**, loses on **{t['ismc_loses']}** of {t['n']}"
            f" (tie tolerance {TIE_TOL}). Median delta {_f(t['median_delta'])}.",
            "",
            "| task | metric | ESM-C 300M | ISM-C 300M | delta |",
            "|---|---|---|---|---|",
        ]
        for r in deltas:
            L.append(
                f"| {r['task']} | {r['metric']} | {_f(r['esmc'])} | {_f(r['ismc'])} "
                f"| {r['delta']:+.3f} |"
            )

    for probe in PROBES:
        rows = report["head_to_head"][probe]
        if not rows:
            continue
        d_e = [r["d_esmc"] for r in rows]
        d_i = [r["d_ismc"] for r in rows]

        def rec(d):
            w = sum(x > TIE_TOL for x in d)
            lo = sum(x < -TIE_TOL for x in d)
            return f"{w}W/{len(d) - w - lo}T/{lo}L, median {sorted(d)[len(d) // 2]:+.3f}"

        L += [
            "",
            f"## ProtSent-V2 150M against the ESM-C arms ({probe} probe)",
            "",
            f"Against ESM-C: {rec(d_e)}. Against ISM-C: {rec(d_i)}. Both columns cross",
            "model family and scale, so these are levels, not a controlled comparison.",
            "",
            "| task | metric | ProtSent-V2 150M | ESM-C | ISM-C | vs ESM-C | vs ISM-C |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            L.append(
                f"| {r['task']} | {r['metric']} | {_f(r['ours'])} | {_f(r['esmc'])} "
                f"| {_f(r['ismc'])} | {r['d_esmc']:+.3f} | {r['d_ismc']:+.3f} |"
            )

    labels = [label for _, _, label in ARMS] + ["MMseqs2"]
    for probe in PROBES:
        rows = report["cross_model"][probe]
        if not rows:
            continue
        L += [
            "",
            f"## Every arm side by side ({probe} probe)",
            "",
            "| task | metric | " + " | ".join(labels) + " |",
            "|---|---|" + "---|" * len(labels),
        ]
        for r in rows:
            L.append(
                f"| {r['task']} | {r['metric']} | "
                + " | ".join(_f(r["values"].get(k)) for k in labels)
                + " |"
            )
    return "\n".join(L) + "\n"


def build() -> dict:
    return {
        "scope40": scope_rows(),
        "distillation": {p: distillation_delta(p) for p in PROBES},
        "tally": {p: tally(distillation_delta(p)) for p in PROBES},
        "cross_model": {p: cross_model(p) for p in PROBES},
        "head_to_head": {p: head_to_head(p) for p in PROBES},
    }


def _selfcheck() -> None:
    assert tally([])["n"] == 0
    t = tally([{"delta": 0.02}, {"delta": -0.02}, {"delta": 0.0}, {"delta": 0.004}])
    assert (t["ismc_wins"], t["ties"], t["ismc_loses"]) == (1, 2, 1), t
    assert _f(None) == "--" and _f(0.5) == "0.500"
    # A missing arm must yield an empty comparison, not a crash or a silent zero.
    assert distillation_delta("nonexistent-probe") == []
    # SCOPe-40 must be compared on the eligible metric in BOTH tables, or one task
    # name carries two different numbers with nothing saying which is which.
    scope = [r for r in distillation_delta("knn") if r["task"] == SCOPE_TASK]
    if scope:
        assert scope[0]["metric"] == SCOPE_DELTA_METRIC, scope[0]
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BENCH / "ISM_COMPARISON.md")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        raise SystemExit(0)

    report = build()
    args.out.write_text(render(report))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(render(report))
    print(f"wrote {args.out}")
