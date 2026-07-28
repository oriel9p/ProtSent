#!/usr/bin/env python
"""Does ProtSent's SCOPe-40 gain depend on how close the query is to pretraining data?

The planned analysis was to bin Recall@10 by max sequence identity to the
pretraining corpus and show the gain survives in a low-identity bin. That is not
possible here: the [0, 0.2) bin is empty and the median max-identity is 0.89,
because AFDB covers essentially all of UniProt and SCOPe domains come from PDB
entries whose parent sequences are in UniProt. The same is true of ESM-2's
UniRef50 pretraining set, so this is a property of corpus coverage rather than
something specific to ProtSent.

The question the binning was meant to answer can still be answered directly:

    if ProtSent's advantage came from memorizing pretraining neighbours, queries
    with a closer pretraining neighbour would gain more.

So we correlate, per query, the max identity to the pretraining corpus against
the per-query ProtSent-minus-baseline retrieval gain. A correlation near zero is
evidence against memorization and does not need an empty bin to make its point.

Reports Spearman and Pearson on the per-query gain, plus per-bin means over the
bins that are actually populated, for hit@1, hit@10 and average precision.

Usage:
    python scope_identity_correlation.py --models /storage/models/ESM2-35M oriel9p/protsent-esm2-35M
    python scope_identity_correlation.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

IDENT_PARQUET = (
    "/storage/users/ddofer/data/decontam_work/scope_strat/scope40_max_identity.parquet"
)
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.01)]


def compute_per_query(emb: np.ndarray, labels) -> dict[str, np.ndarray]:
    """Per-query hit@1, hit@10 and average precision. Self excluded.

    Mirrors evaluate_retrieval() in protein_benchmark_suite.py, but returns the
    per-query vectors rather than their means, since the whole point is to
    correlate them against something else.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    n = len(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    fam_size = dict(zip(uniq.tolist(), cnt.tolist()))

    nn = NearestNeighbors(n_neighbors=n, metric="cosine", n_jobs=-1).fit(emb)
    _, idx = nn.kneighbors(emb)

    hit1 = np.zeros(n)
    hit10 = np.zeros(n)
    ap = np.zeros(n)
    for q in range(n):
        ranked = idx[q][idx[q] != q]
        rel = labels[ranked] == labels[q]
        n_rel = fam_size[labels[q]] - 1
        if n_rel <= 0:
            continue  # unachievable query; stays 0 everywhere
        hit1[q] = float(rel[:1].any())
        hit10[q] = float(rel[:10].any())
        hr = np.flatnonzero(rel) + 1
        if hr.size:
            ap[q] = float(np.sum(np.arange(1, hr.size + 1) / hr) / n_rel)
    eligible = np.array([fam_size[lab] > 1 for lab in labels.tolist()])
    return {"hit1": hit1, "hit10": hit10, "ap": ap, "eligible": eligible}


def embed(model_name: str, seqs: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    # The base ESM2-35M checkpoint carries FastPLM custom code, so it needs
    # trust_remote_code; the published ProtSent checkpoint is plain ESM and does
    # not. Passing it unconditionally is harmless for the latter.
    m = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
    return np.asarray(m.encode(seqs, batch_size=batch_size, show_progress_bar=False))


def load_scope():
    from datasets import load_dataset

    ds = load_dataset("tattabio/scope40_test", split="train")
    return list(ds["sequence"]), list(ds["family"])


def load_identities(seqs: list[str]) -> np.ndarray:
    """Max identity per query, aligned to `seqs` by SEQUENCE, not by row index."""
    import polars as pl

    df = pl.read_parquet(IDENT_PARQUET)
    col = "max_ident_overall" if "max_ident_overall" in df.columns else None
    if col is None:
        cand = [c for c in df.columns if c.startswith("max_ident") and c.endswith("overall")]
        if not cand:
            raise KeyError(f"no max_ident*overall column in {df.columns}")
        col = cand[0]
    lookup = dict(zip(df["sequence"].to_list(), df[col].to_list()))
    missing = [s for s in seqs if s not in lookup]
    if missing:
        raise RuntimeError(
            f"{len(missing)} SCOPe sequences absent from {IDENT_PARQUET}; "
            "the identity table does not match the benchmark set"
        )
    return np.array([lookup[s] for s in seqs], dtype=float)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--models", nargs=2, metavar=("BASELINE", "PROTSENT"), required=True)
    ap_.add_argument("--out", default="results/benchmarks/scope_identity_correlation.json")
    ap_.add_argument("--batch_size", type=int, default=64)
    args = ap_.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    from scipy.stats import pearsonr, spearmanr

    seqs, labels = load_scope()
    ident = load_identities(seqs)
    print(f"{len(seqs)} SCOPe sequences; identity median {np.median(ident):.3f}")

    base_name, ps_name = args.models
    base = compute_per_query(embed(base_name, seqs, args.batch_size), labels)
    ps = compute_per_query(embed(ps_name, seqs, args.batch_size), labels)

    el = base["eligible"]
    print(f"eligible queries: {int(el.sum())}/{len(seqs)}")

    report: dict = {
        "baseline": base_name,
        "protsent": ps_name,
        "n_queries": len(seqs),
        "n_eligible": int(el.sum()),
        "identity_median": float(np.median(ident)),
        "correlations": {},
        "bins": [],
    }

    for metric in ("hit1", "hit10", "ap"):
        gain = (ps[metric] - base[metric])[el]
        x = ident[el]
        rs, ps_p = spearmanr(x, gain)
        rp, pp = pearsonr(x, gain)
        report["correlations"][metric] = {
            "spearman_r": float(rs), "spearman_p": float(ps_p),
            "pearson_r": float(rp), "pearson_p": float(pp),
            "mean_gain": float(gain.mean()),
        }
        print(f"{metric:6s} gain {gain.mean():+.4f} | spearman r={rs:+.4f} p={ps_p:.3g} "
              f"| pearson r={rp:+.4f} p={pp:.3g}")

    print(f"\n{'bin':12s} {'n':>5s} {'base_h10':>9s} {'ps_h10':>8s} {'gain_h10':>9s} "
          f"{'base_ap':>8s} {'ps_ap':>7s} {'gain_ap':>8s}")
    for lo, hi in BINS:
        sel = el & (ident >= lo) & (ident < hi)
        n = int(sel.sum())
        row = {"bin": f"[{lo}, {hi})", "n": n}
        if n:
            row.update({
                "base_hit10": float(base["hit10"][sel].mean()),
                "protsent_hit10": float(ps["hit10"][sel].mean()),
                "base_ap": float(base["ap"][sel].mean()),
                "protsent_ap": float(ps["ap"][sel].mean()),
            })
            print(f"[{lo}, {hi})   {n:5d} {row['base_hit10']:9.4f} {row['protsent_hit10']:8.4f} "
                  f"{row['protsent_hit10']-row['base_hit10']:+9.4f} "
                  f"{row['base_ap']:8.4f} {row['protsent_ap']:7.4f} "
                  f"{row['protsent_ap']-row['base_ap']:+8.4f}")
        else:
            print(f"[{lo}, {hi})   {n:5d}  (empty)")
        report["bins"].append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    # 4 points: families A A B B. Perfect retrieval -> hit1=hit10=ap=1 for all.
    E = np.array([[1, 0], [0.999, 0.01], [0, 1], [0.01, 0.999]], dtype=float)
    m = compute_per_query(E, ["A", "A", "B", "B"])
    assert m["hit1"].tolist() == [1.0, 1.0, 1.0, 1.0], m["hit1"]
    assert np.allclose(m["ap"], 1.0), m["ap"]
    assert m["eligible"].all()

    # Singleton family is unachievable and must stay 0 without polluting others.
    m2 = compute_per_query(
        np.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=float), ["A", "A", "C"]
    )
    assert m2["hit1"].tolist() == [1.0, 1.0, 0.0], m2["hit1"]
    assert m2["eligible"].tolist() == [True, True, False]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
