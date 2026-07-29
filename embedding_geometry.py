#!/usr/bin/env python
"""Does contrastive fine-tuning reorganise the embedding space, or just move a metric?

Reviewer HNXd asked, as one of the conditions for raising their score, for "either a
direct retrieval/clustering evaluation, or an analysis showing how ProtSent changes the
local and global organization of the protein embedding space". Retrieval is answered
elsewhere; this is the organisation half, which had not been computed.

SCOPe-40 labels are four-level -- class.fold.superfamily.family (e.g. `a.5.6.1`) -- so the
benchmark carries its own ground-truth hierarchy and no external annotation is needed.

Reported per model:

  silhouette          family-level cohesion vs separation under cosine distance. Negative
                      means families overlap more than they separate.
  NMI / ARI           agglomerative clustering at k = the true number of families, scored
                      against the true family labels. NMI is permutation-invariant;
                      ARI is chance-corrected, so a high NMI with a near-zero ARI means
                      the partition is informative but badly aligned.
  intra/inter ratio   mean within-family distance divided by mean between-family distance.
                      Lower is better; this is the quantity contrastive training targets.
  hierarchy Spearman  correlation between pairwise embedding distance and how much SCOPe
                      hierarchy two domains share (0 = different class, 4 = same family).
                      A well-organised space makes this strongly NEGATIVE: more shared
                      hierarchy, less distance. This is the global-organisation measure,
                      and it cannot be gamed by tightening families alone.

Usage:
    python embedding_geometry.py --models ESM-2-35M=/storage/models/ESM2-35M ...
    python embedding_geometry.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def shared_depth(labels: list[str]) -> np.ndarray:
    """Pairwise count of matching leading hierarchy levels, 0..4."""
    parts = np.array([l.split(".") for l in labels])          # (n, 4)
    n = len(labels)
    depth = np.zeros((n, n), dtype=np.int8)
    prefix_match = np.ones((n, n), dtype=bool)
    for lvl in range(parts.shape[1]):
        col = parts[:, lvl]
        prefix_match &= col[:, None] == col[None, :]
        depth += prefix_match
    return depth


def geometry(emb: np.ndarray, labels: list[str], seed: int = 0) -> dict:
    from scipy.stats import spearmanr
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    dist = 1.0 - emb @ emb.T
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    y = np.asarray(labels)
    k = len(set(labels))

    sil = float(silhouette_score(dist, y, metric="precomputed"))
    pred = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                   linkage="average").fit_predict(dist)
    nmi = float(normalized_mutual_info_score(y, pred))
    ari = float(adjusted_rand_score(y, pred))

    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    intra = float(dist[same & off].mean())
    inter = float(dist[~same].mean())

    depth = shared_depth(labels)
    iu = np.triu_indices(len(y), k=1)
    rho, p = spearmanr(dist[iu], depth[iu])

    by_depth = {
        int(d): float(dist[iu][depth[iu] == d].mean())
        for d in range(5)
        if (depth[iu] == d).any()
    }
    return {
        "silhouette": sil, "nmi": nmi, "ari": ari,
        "intra_family_distance": intra, "inter_family_distance": inter,
        "intra_over_inter": intra / inter,
        "hierarchy_spearman": float(rho), "hierarchy_spearman_p": float(p),
        "mean_distance_by_shared_depth": by_depth,
        "n_families": k,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="results/benchmarks/embedding_geometry.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    sys.path.insert(0, str(Path(__file__).parent))
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ds = load_dataset("tattabio/scope40_test", split="train")
    seqs, labels = list(ds["sequence"]), list(ds["family"])
    print(f"{len(seqs)} SCOPe-40 domains, {len(set(labels))} families, 4-level hierarchy")

    report = {"n_domains": len(seqs), "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"\n{name} ...", flush=True)
        m = SentenceTransformer(path, device="cuda", trust_remote_code=True)
        emb = np.asarray(m.encode(seqs, batch_size=64, show_progress_bar=False))
        g = geometry(emb, labels)
        report["models"][name] = g
        print(f"  silhouette {g['silhouette']:+.4f}   NMI {g['nmi']:.4f}   ARI {g['ari']:.4f}")
        print(f"  intra/inter {g['intra_over_inter']:.4f} "
              f"({g['intra_family_distance']:.4f} / {g['inter_family_distance']:.4f})")
        print(f"  distance vs shared hierarchy: spearman {g['hierarchy_spearman']:+.4f}")
        print(f"  mean distance by shared depth: "
              + ", ".join(f"{d}:{v:.3f}" for d, v in g["mean_distance_by_shared_depth"].items()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    labels = ["a.1.1.1", "a.1.1.1", "a.1.1.2", "a.2.1.1", "b.1.1.1", "b.1.1.1"]
    d = shared_depth(labels)
    assert d[0, 1] == 4, d[0, 1]          # identical family
    assert d[0, 2] == 3, d[0, 2]          # same superfamily, different family
    assert d[0, 3] == 1, d[0, 3]          # a.1.1.1 vs a.2.1.1 share only the class
    assert d[0, 4] == 0, d[0, 4]          # different class
    assert (np.diag(d) == 4).all()

    # A perfectly organised space: families tight, hierarchy respected.
    rng = np.random.default_rng(0)
    centres = {"a.1.1.1": [1, 0, 0], "a.1.1.2": [0.9, 0.1, 0], "a.2.1.1": [0, 1, 0],
               "b.1.1.1": [0, 0, 1]}
    emb = np.array([np.array(centres[l], dtype=float) + rng.normal(0, 0.01, 3)
                    for l in labels])
    g = geometry(emb, labels)
    assert g["silhouette"] > 0.5, g["silhouette"]
    assert g["intra_over_inter"] < 0.5, g["intra_over_inter"]
    assert g["hierarchy_spearman"] < -0.5, g["hierarchy_spearman"]

    # A scrambled space must score near chance and must NOT show hierarchy structure.
    emb_rand = rng.normal(0, 1, (len(labels), 16))
    g2 = geometry(emb_rand, labels)
    assert g2["silhouette"] < g["silhouette"]
    assert g2["hierarchy_spearman"] > g["hierarchy_spearman"]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
