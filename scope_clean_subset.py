#!/usr/bin/env python
"""SCOPe-40 retrieval restricted to queries far from the pretraining corpus.

SCOPe-40 cannot be decontaminated at the corpus level: it has no train/test split,
so removing every pretraining sequence that resembles a SCOPe domain would delete
essentially every structured domain from the corpus. The benchmark side can be
decontaminated instead -- drop the *queries* that have a close pretraining
neighbour and re-score on what remains.

That is a weaker control than corpus filtering and we say so: it bounds
identity-level exposure only. A training pair sharing a query's fold at 15%
identity survives any identity threshold.

Reports every arm on the same query subsets so the comparison stays paired, with
bootstrap CIs on the retained queries.

Usage:
    python scope_clean_subset.py
    python scope_clean_subset.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = [0.4, 0.5, 0.7, 1.01]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/scope40_clean_subset.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import boot_ci, embed_ranking, mmseqs_ranking, per_query_metrics
    from hmmer_baseline import load_scope, phmmer_ranking
    from scope_identity_correlation import load_identities

    seqs, labels = load_scope()
    labels = np.asarray(labels)
    ident = load_identities(seqs)
    n = len(seqs)

    arms: dict[str, dict] = {}
    arms["HMMER"] = per_query_metrics(phmmer_ranking(seqs, cpus=48), labels)
    arms["MMseqs2"] = per_query_metrics(
        mmseqs_ranking(
            Path("/storage/users/ddofer/data/mmseqs_baseline/scope40_retrieval/hits.tsv"), n
        ),
        labels,
    )
    for name, path in [
        ("ESM-2 35M", "/storage/models/ESM2-35M"),
        ("ProtSent-V1", "oriel9p/protsent-esm2-35M"),
        ("ProtSent-V2", "models/protsent_esm2_35m_v3/final"),
    ]:
        arms[name] = per_query_metrics(embed_ranking(path, seqs), labels)

    eligible = arms["ESM-2 35M"]["eligible"]
    report = {"n_queries": n, "identity_median": float(np.median(ident)), "subsets": []}

    for thr in THRESHOLDS:
        sel = eligible & (ident < thr)
        block = {"max_identity_below": thr, "n_eligible_queries": int(sel.sum()), "arms": {}}
        label = "all" if thr > 1 else f"<{thr}"
        print(f"\n=== queries with max identity to pretraining corpus {label} "
              f"({int(sel.sum())} eligible) ===")
        print(f"{'method':14s} {'R@1':>18s} {'R@10':>18s} {'MAP':>18s}")
        for name, m in arms.items():
            row = {}
            cells = []
            for metric, key in (("hit1", "R@1"), ("hit10", "R@10"), ("ap", "MAP")):
                mean, lo, hi = boot_ci(m[metric][sel])
                row[key] = {"mean": mean, "lo": lo, "hi": hi}
                cells.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
            block["arms"][name] = row
            print(f"{name:14s} " + " ".join(f"{c:>18s}" for c in cells))

        # Paired margins against the better alignment tool, on the same queries.
        best = "HMMER"
        block["paired_vs_best_alignment"] = {}
        for name in ("ProtSent-V1", "ProtSent-V2"):
            for metric, key in (("hit1", "R@1"), ("hit10", "R@10"), ("ap", "MAP")):
                d = (arms[name][metric] - arms[best][metric])[sel]
                mean, lo, hi = boot_ci(d)
                block["paired_vs_best_alignment"][f"{name} - {best} / {key}"] = {
                    "delta": mean, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)
                }
        report["subsets"].append(block)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import per_query_metrics

    labels = np.array(["A", "A", "B", "B"])
    ranking = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])
    m = per_query_metrics(ranking, labels)
    ident = np.array([0.9, 0.3, 0.95, 0.2])

    sel_low = m["eligible"] & (ident < 0.4)
    assert sel_low.sum() == 2, sel_low
    assert m["hit1"][sel_low].mean() == 1.0

    # The subset must actually change the population it scores.
    sel_all = m["eligible"] & (ident < 1.01)
    assert sel_all.sum() == 4
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
