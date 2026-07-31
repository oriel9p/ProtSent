#!/usr/bin/env python
"""95% bootstrap confidence intervals on SCOPe-40 retrieval, and paired CIs on deltas.

Reviewer HNXd asked for "95% confidence intervals for the reported metrics,
computed by bootstrapping over individual predictions". Retrieval is the one
place that request is answerable exactly rather than approximately: every metric
here is a mean over per-query values, so resampling queries with replacement
gives the sampling distribution directly, with no refitting.

Two things are reported, and only the second one settles anything:

  marginal CI   the interval for one method's metric. Wide, and overlapping
                marginal intervals do NOT imply the difference is unresolved,
                because the same queries are scored by both methods.
  paired CI     the interval for the per-query DIFFERENCE between two methods.
                This is the test. It is much tighter than the marginals suggest,
                and it is the number to quote when claiming an improvement.

Covers the embedding models. MMseqs2 is scored from its hit table when one is
supplied, so the alignment baseline gets the same treatment.

Usage:
    python bootstrap_ci.py --models ESM-2=/storage/models/ESM2-35M \\
        ProtSent-V1=oriel9p/protsent-esm2-35M \\
        ProtSent-V2=models/protsent_esm2_35m_v3/final
    python bootstrap_ci.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10_000
SEED = 0


def per_query_metrics(ranking: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """hit@1, hit@10, hit@30 and average precision for each query.

    `ranking[q]` is the gallery order for query q, self already removed.
    Queries with no achievable positive get 0 and are marked ineligible.
    """
    n = len(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    fam = dict(zip(uniq.tolist(), cnt.tolist()))

    out = {k: np.zeros(n) for k in ("hit1", "hit10", "hit30", "ap")}
    eligible = np.zeros(n, dtype=bool)
    for q in range(n):
        n_rel = fam[labels[q]] - 1
        if n_rel <= 0:
            continue
        eligible[q] = True
        rel = labels[ranking[q]] == labels[q]
        out["hit1"][q] = float(rel[:1].any())
        out["hit10"][q] = float(rel[:10].any())
        out["hit30"][q] = float(rel[:30].any())
        hr = np.flatnonzero(rel) + 1
        if hr.size:
            out["ap"][q] = float(np.sum(np.arange(1, hr.size + 1) / hr) / n_rel)
    out["eligible"] = eligible
    return out


def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over the query axis."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def embed_ranking(model_name: str, seqs: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
    emb = np.asarray(m.encode(seqs, batch_size=batch_size, show_progress_bar=False))
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)  # self ranks last, then is dropped below
    # Return n-1 columns with self actually REMOVED, matching what
    # per_query_metrics documents and what mmseqs_ranking/phmmer_ranking return.
    # Leaving self in as a trailing column put a self-match into the relevance
    # vector and inflated AP by ~6e-4 for embedding arms only -- alignment arms
    # were unaffected, so embedding-vs-alignment MAP deltas carried that bias.
    # hit@K was never affected, since self sits beyond any K scored.
    return np.argsort(-sim, axis=1)[:, :-1]


def mmseqs_ranking(hits_tsv: Path, n: int) -> np.ndarray:
    """Gallery order per query from an MMseqs2 hit table, self removed.

    Queries the search missed keep an arbitrary tail order; every one of their
    positives therefore lands beyond any K we score, which is the intended
    "no hit counts as a failure" convention.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from mmseqs_baseline import read_hits

    hits = read_hits(hits_tsv)
    ranking = np.zeros((n, n - 1), dtype=int)
    for q in range(n):
        ranked = [t for t, _ in hits.get(q, []) if t != q]
        rest = [t for t in range(n) if t != q and t not in set(ranked)]
        ranking[q] = np.array(ranked + rest, dtype=int)[: n - 1]
    return ranking


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--mmseqs_hits", default=None, help="hit TSV to score as an extra arm")
    ap.add_argument("--hmmer", action="store_true",
                    help="score phmmer as an extra arm (all-vs-all, CPU; ~tens of minutes)")
    ap.add_argument("--hmmer_cpus", type=int, default=48)
    ap.add_argument("--out", default="results/benchmarks/scope40_bootstrap_ci.json")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    from datasets import load_dataset

    ds = load_dataset("tattabio/scope40_test", split="train")
    seqs, labels = list(ds["sequence"]), np.asarray(ds["family"])
    n = len(seqs)
    print(f"{n} SCOPe-40 sequences")

    per_model = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"embedding {name} ...", flush=True)
        per_model[name] = per_query_metrics(embed_ranking(path, seqs), labels)
    if args.mmseqs_hits:
        print("scoring MMseqs2 hit table ...", flush=True)
        per_model["MMseqs2"] = per_query_metrics(mmseqs_ranking(Path(args.mmseqs_hits), n), labels)
    if args.hmmer:
        # phmmer is the stronger alignment baseline and the one Yi1G named. It is
        # computed rather than read from disk because hmmer_baseline.py keeps hits in
        # memory; phmmer_ranking already returns the dense gallery ordering this wants,
        # with no-hit queries left in an arbitrary tail so they score as failures.
        print(f"running all-vs-all phmmer on {n} sequences ...", flush=True)
        sys.path.insert(0, str(Path(__file__).parent))
        from hmmer_baseline import phmmer_ranking

        per_model["HMMER"] = per_query_metrics(phmmer_ranking(seqs, cpus=args.hmmer_cpus), labels)

    eligible = next(iter(per_model.values()))["eligible"]
    print(f"eligible queries: {int(eligible.sum())}/{n}\n")

    report = {"n_queries": n, "n_eligible": int(eligible.sum()),
              "n_boot": args.n_boot, "marginal": {}, "paired": {}}

    print(f"{'method':16s} {'metric':7s} {'mean':>7s}  95% CI (eligible queries)")
    for name, m in per_model.items():
        report["marginal"][name] = {}
        for metric in ("hit1", "hit10", "hit30", "ap"):
            mean, lo, hi = boot_ci(m[metric][eligible], args.n_boot)
            report["marginal"][name][metric] = {"mean": mean, "lo": lo, "hi": hi}
            print(f"{name:16s} {metric:7s} {mean:7.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\n{'comparison':34s} {'metric':7s} {'delta':>8s}  95% CI (paired)   verdict")
    names = list(per_model)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{b} - {a}"
            report["paired"][key] = {}
            for metric in ("hit1", "hit10", "hit30", "ap"):
                diff = (per_model[b][metric] - per_model[a][metric])[eligible]
                mean, lo, hi = boot_ci(diff, args.n_boot)
                sig = lo > 0 or hi < 0
                report["paired"][key][metric] = {
                    "delta": mean, "lo": lo, "hi": hi, "excludes_zero": bool(sig)
                }
                print(f"{key:34s} {metric:7s} {mean:+8.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                      f"{'significant' if sig else 'unresolved'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    # 4 proteins, families A A B B, perfect ranking -> every metric 1.0.
    labels = np.array(["A", "A", "B", "B"])
    ranking = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])
    m = per_query_metrics(ranking, labels)
    assert m["hit1"].tolist() == [1, 1, 1, 1], m["hit1"]
    assert np.allclose(m["ap"], 1.0), m["ap"]
    assert m["eligible"].all()

    # Singleton family: unachievable, must stay 0 and ineligible.
    m2 = per_query_metrics(np.array([[1, 2], [0, 2], [0, 1]]), np.array(["A", "A", "C"]))
    assert m2["eligible"].tolist() == [True, True, False]
    assert m2["hit1"].tolist() == [1.0, 1.0, 0.0]

    # A constant vector has a zero-width CI; a balanced one brackets its mean.
    mean, lo, hi = boot_ci(np.ones(50))
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)
    mean, lo, hi = boot_ci(np.array([0.0, 1.0] * 200))
    assert lo < mean < hi and abs(mean - 0.5) < 1e-9

    # Paired differences: a uniform +0.2 shift must exclude zero.
    _, lo, hi = boot_ci(np.full(300, 0.2))
    assert lo > 0
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
