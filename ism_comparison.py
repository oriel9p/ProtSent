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

  2. Where both sit relative to our arms and the alignment baselines. These are
     NOT scale-matched -- ISM-C is 300M against our 150M and 35M -- so they are
     reported as context, not as a controlled comparison.

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


def collect(probe: str) -> dict:
    """tag -> {task name -> row} for every arm that has results for this probe."""
    return {tag: load_arm(BENCH / sub, tag, probe) for sub, tag, _ in ARMS}


def scope_rows() -> list:
    """SCOPe-40 retrieval, all methods. Retrieval ignores the probe flag."""
    rows = []
    mm = load_mmseqs(BENCH / "mmseqs_baseline.json").get("scope40_retrieval")
    if mm:
        rows.append(("MMseqs2 (-s 7.5)", [_num(mm.get(k)) for k, _ in SCOPE_METRICS]))
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
        metric = metric_of.get(task)
        if metric is None:
            continue
        a, b = _num(treated[task].get(metric)), _num(control[task].get(metric))
        if a is None or b is None:
            continue
        out.append({"task": task, "metric": metric, "ismc": a, "esmc": b, "delta": a - b})
    out.sort(key=lambda r: r["delta"])
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


def _f(v, nd=4):
    return "--" if v is None else f"{v:.{nd}f}"


def render(report: dict) -> str:
    L = ["# ISM-C-300M on the ProtSent benchmark suite", ""]
    L += [
        "ISM-C-300M is a structure-distilled ESM-C-300M. `Synthyra/ESMplusplus_small`",
        "is vanilla ESM-C-300M: same architecture, parameter count, tokenizer and",
        "code path. That pairing isolates the distillation. The 150M and 35M rows are",
        "context at a different scale, not a controlled comparison.",
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
                f"| {r['delta']:+.4f} |"
            )
    return "\n".join(L) + "\n"


def build() -> dict:
    return {
        "scope40": scope_rows(),
        "distillation": {p: distillation_delta(p) for p in PROBES},
        "tally": {p: tally(distillation_delta(p)) for p in PROBES},
    }


def _selfcheck() -> None:
    assert tally([])["n"] == 0
    t = tally([{"delta": 0.02}, {"delta": -0.02}, {"delta": 0.0}, {"delta": 0.004}])
    assert (t["ismc_wins"], t["ties"], t["ismc_loses"]) == (1, 2, 1), t
    assert _f(None) == "--" and _f(0.5) == "0.5000"
    # A missing arm must yield an empty comparison, not a crash or a silent zero.
    assert distillation_delta("nonexistent-probe") == []
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
