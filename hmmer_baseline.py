#!/usr/bin/env python
"""HMMER (phmmer) SCOPe-40 retrieval baseline, scored identically to every other arm.

Reviewer Yi1G named "HMMER/MMseqs2" among the missing baselines. MMseqs2 is
covered by `mmseqs_baseline.py`; this adds the HMMER half. phmmer builds an
implicit profile from each single query sequence and searches it against the
gallery, which is a more sensitive remote-homology detector than MMseqs2's
k-mer prefilter -- it is the harder alignment baseline, which is why it is worth
running rather than assuming it would lose.

Scored exactly like the embedding arms: self-matches removed, family-level
Recall@K over the same 2,207-sequence gallery, and **queries phmmer returns
nothing for count as failures, not dropped** -- their positives simply never
enter the ranked prefix.

Ranking key is bitscore descending, the convention `mmseqs_baseline.py` uses.

Usage:
    python hmmer_baseline.py
    python hmmer_baseline.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_scope() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("tattabio/scope40_test", split="train")
    return list(ds["sequence"]), list(ds["family"])


def phmmer_ranking(seqs: list[str], cpus: int = 32, E: float = 10.0) -> np.ndarray:
    """All-vs-all phmmer. Per query, the gallery order by bitscore descending.

    Queries with no hit keep an arbitrary tail, so their positives land beyond
    any K that is scored -- the "no hit is a failure" convention.
    """
    import pyhmmer
    from pyhmmer.easel import Alphabet, TextSequence

    alphabet = Alphabet.amino()
    digital = [
        TextSequence(sequence=s, name=str(i).encode()).digitize(alphabet)
        for i, s in enumerate(seqs)
    ]

    n = len(seqs)
    ranking = np.zeros((n, n - 1), dtype=int)
    n_no_hit = 0
    for q, hits in enumerate(pyhmmer.hmmer.phmmer(digital, digital, cpus=cpus, E=E)):
        scored = [
            (int(h.name if isinstance(h.name, str) else h.name.decode()), float(h.score))
            for h in hits
        ]
        scored = [(t, s) for t, s in scored if t != q]
        scored.sort(key=lambda x: -x[1])
        ranked = [t for t, _ in scored]
        if not ranked:
            n_no_hit += 1
        seen = set(ranked)
        ranking[q] = np.array(
            ranked + [t for t in range(n) if t != q and t not in seen], dtype=int
        )[: n - 1]
    print(f"queries with no phmmer hit at all: {n_no_hit}/{n}")
    return ranking


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/hmmer_scope40.json")
    ap.add_argument("--cpus", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import boot_ci, per_query_metrics

    seqs, labels = load_scope()
    labels = np.asarray(labels)
    print(f"{len(seqs)} SCOPe-40 sequences; phmmer all-vs-all ...", flush=True)

    ranking = phmmer_ranking(seqs, cpus=args.cpus)
    m = per_query_metrics(ranking, labels)
    el = m["eligible"]

    report = {
        "method": "phmmer (HMMER3, pyhmmer)",
        "flags": {"E": 10.0, "rank_by": "bitscore desc", "self_excluded": True,
                  "no_hit_counts_as_failure": True},
        "n_queries": len(seqs),
        "n_eligible": int(el.sum()),
        "all_queries": {},
        "eligible": {},
    }
    print(f"\n{'metric':7s} {'all':>8s} {'eligible':>9s}  95% CI (eligible)")
    for metric in ("hit1", "hit10", "hit30", "ap"):
        allv = float(m[metric].mean())
        mean, lo, hi = boot_ci(m[metric][el])
        report["all_queries"][metric] = allv
        report["eligible"][metric] = {"mean": mean, "lo": lo, "hi": hi}
        print(f"{metric:7s} {allv:8.4f} {mean:9.4f}  [{lo:.4f}, {hi:.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    """Two families of two near-identical sequences; the partner must rank first."""
    a1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    a2 = a1[:50] + "C" + a1[51:]
    b1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRSDLAWQVQNMLHRYPQVVDMLRRLGLDPQAVE"
    b2 = b1[:-1] + "D"

    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import per_query_metrics

    ranking = phmmer_ranking([a1, a2, b1, b2], cpus=2)
    m = per_query_metrics(ranking, np.array(["A", "A", "B", "B"]))
    assert m["eligible"].all(), m["eligible"]
    assert m["hit1"].tolist() == [1.0, 1.0, 1.0, 1.0], (
        f"phmmer did not rank near-identical partners first: {m['hit1']}"
    )

    # A query with no hit must score zero, not crash or silently drop out.
    lone = "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    r2 = phmmer_ranking([a1, a2, lone], cpus=2)
    m2 = per_query_metrics(r2, np.array(["A", "A", "Z"]))
    assert m2["eligible"].tolist() == [True, True, False]
    assert m2["hit1"][2] == 0.0
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
