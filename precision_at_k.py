#!/usr/bin/env python
"""Precision@K alongside Recall@K, for alignment and embedding retrieval.

Recall@K asks "was a same-family domain found anywhere in the top K". It rewards a
method for returning more candidates, so a search run with its heuristic filters
disabled -- which returns a hit for every query -- can raise Recall@K while getting
*worse* at telling homologs from non-homologs.

Precision@K is the fraction of the top K that are same-family. It is the half of
the picture Recall@K hides, and it is what a practitioner filtering a hit list
actually experiences.

Both are reported over the same 1,693 eligible queries so they are comparable, and
precision is capped by the number of true positives available: a query whose family
has 3 other members cannot exceed 3/10 at K=10, so the mean is reported alongside
its attainable ceiling.

Usage:
    python precision_at_k.py
    python precision_at_k.py --selfcheck
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def precision_recall_at_k(ranking: np.ndarray, labels: np.ndarray, k: int) -> dict:
    n = len(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    fam = dict(zip(uniq.tolist(), cnt.tolist()))

    prec, rec, ceil = [], [], []
    for q in range(n):
        n_rel = fam[labels[q]] - 1
        if n_rel <= 0:
            continue
        top = labels[ranking[q][:k]]
        hits = int((top == labels[q]).sum())
        prec.append(hits / k)
        rec.append(float(hits > 0))
        ceil.append(min(n_rel, k) / k)
    return {"precision": float(np.mean(prec)), "recall": float(np.mean(rec)),
            "precision_ceiling": float(np.mean(ceil)), "n": len(prec)}


def main() -> int:
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import embed_ranking, mmseqs_ranking, per_query_metrics
    from hmmer_baseline import load_scope, phmmer_ranking

    seqs, labels = load_scope()
    labels = np.asarray(labels)
    n = len(seqs)

    arms = {}
    arms["HMMER phmmer (default)"] = phmmer_ranking(seqs, cpus=48)
    arms["MMseqs2 (-s 7.5)"] = mmseqs_ranking(
        Path("/storage/users/ddofer/data/mmseqs_baseline/scope40_retrieval/hits.tsv"), n)
    for name, path in [("ESM-2 35M", "/storage/models/ESM2-35M"),
                       ("ProtSent-V1", "oriel9p/protsent-esm2-35M"),
                       ("ProtSent-V2", "models/protsent_esm2_35m_v3/final")]:
        arms[name] = embed_ranking(path, seqs)

    report = {}
    print(f"\n{'method':24s} " + "  ".join(f"P@{k:<2d}   R@{k:<2d}" for k in (1, 5, 10)))
    for name, rk in arms.items():
        row, cells = {}, []
        for k in (1, 5, 10):
            m = precision_recall_at_k(rk, labels, k)
            row[f"@{k}"] = m
            cells.append(f"{m['precision']:.3f}  {m['recall']:.3f}")
        report[name] = row
        print(f"{name:24s} " + "  ".join(cells))
    c = precision_recall_at_k(arms["ESM-2 35M"], labels, 10)
    print(f"\nattainable mean precision ceiling at K=10: {c['precision_ceiling']:.3f} "
          f"(families are small; most queries cannot fill 10 slots)")

    out = Path("results/benchmarks/precision_at_k.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


def _selfcheck() -> None:
    # 4 domains, families A A B B. Perfect ranking: each query has exactly 1 true
    # positive, so P@1 = 1.0 but P@10-style dilution shows at larger K.
    labels = np.array(["A", "A", "B", "B"])
    ranking = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])
    m1 = precision_recall_at_k(ranking, labels, 1)
    assert m1["precision"] == 1.0 and m1["recall"] == 1.0, m1
    m3 = precision_recall_at_k(ranking, labels, 3)
    assert abs(m3["precision"] - 1 / 3) < 1e-9, m3     # 1 hit out of 3 slots
    assert m3["recall"] == 1.0, m3
    assert abs(m3["precision_ceiling"] - 1 / 3) < 1e-9, m3

    # A ranking that puts the wrong family first loses precision AND recall at K=1.
    bad = np.array([[2, 3, 1], [3, 2, 0], [0, 1, 3], [1, 0, 2]])
    mb = precision_recall_at_k(bad, labels, 1)
    assert mb["precision"] == 0.0 and mb["recall"] == 0.0, mb
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
