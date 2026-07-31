#!/usr/bin/env python
"""Does whitening the vanilla embeddings close ProtSent's SCOPe retrieval gap?

`probe_gap_analysis.py` established that stock ESM-2 embeddings are severely
anisotropic (mean cosine between random SCOPe pairs 0.85 at 35M, 0.90 at 150M;
participation ratio 8-11 of 480/640 dimensions), that ProtSent removes this, and
that on remote homology a whitened vanilla k-NN recovers essentially all of
ProtSent's k-NN advantage. That is a deflating result for the method: a linear
probe can learn any invertible linear map for free, so if contrastive tuning only
whitens, a trained readout should see no benefit -- which is what is measured.

SCOPe-40 retrieval is the case that decides whether the contribution is larger
than whitening, because there the margin is much bigger and there is no trained
readout anywhere in the pipeline. If a whitened vanilla baseline closes the gap,
the honest claim shrinks to "contrastive tuning is a learned whitening". If it
does not, ProtSent reorganises the space in a way no global linear transform of
the original space achieves, and the retrieval claim stands on its own.

The whitening transform is fit on the SAME gallery it is applied to, which is the
most generous possible setting for the baseline -- it is an upper bound on what
whitening can do, not a realistic deployment. Making the baseline as strong as
possible is the point: a gap that survives this survives anything.

Usage:
    python whiten_scope_control.py
    python whiten_scope_control.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

MODELS = {
    "ESM-2-35M": "/storage/models/ESM2-35M",
    "ProtSent-V2-35M": "models/protsent_esm2_35m_v3/final",
    "ESM-2-150M": "Synthyra/ESM2-150M",
    "ProtSent-V2-150M": "models/protsent_esm2_150m_v2/final",
}


def whiten(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Zero-mean, identity-covariance via eigendecomposition (ZCA-style scaling).

    eps floors the eigenvalues so near-null directions -- of which an anisotropic
    encoder has many -- are not amplified into pure noise.
    """
    xc = x - x.mean(0, keepdims=True)
    cov = np.cov(xc, rowvar=False)
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return xc @ v @ np.diag(1.0 / np.sqrt(w))


def rank_by_cosine(emb: np.ndarray) -> np.ndarray:
    """Gallery order per query, self removed -- n-1 columns.

    Self must be dropped, not merely ranked last: per_query_metrics builds its
    relevance vector from every column it is given, so a trailing self-match
    inflates average precision.
    """
    e = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/whiten_scope_control.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")

    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_ci import boot_ci, per_query_metrics
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ds = load_dataset("tattabio/scope40_test", split="train")
    seqs, labels = list(ds["sequence"]), np.asarray(ds["family"])
    print(f"{len(seqs)} SCOPe-40 sequences")

    report, per = {}, {}
    for name, path in MODELS.items():
        m = SentenceTransformer(path, device="cuda", trust_remote_code=True)
        emb = np.asarray(m.encode(seqs, batch_size=args.batch_size, show_progress_bar=False))
        for variant, e in (("raw", emb), ("whitened", whiten(emb))):
            key = f"{name} [{variant}]"
            per[key] = per_query_metrics(rank_by_cosine(e), labels)
        del m

    el = next(iter(per.values()))["eligible"]
    print(f"eligible queries: {int(el.sum())}/{len(seqs)}\n")
    print(f"{'arm':34s} {'R@1':>8s} {'R@10':>8s} {'MAP':>8s}")
    for key, v in per.items():
        row = {mm: float(v[mm][el].mean()) for mm in ("hit1", "hit10", "ap")}
        report[key] = row
        print(f"{key:34s} {row['hit1']:8.4f} {row['hit10']:8.4f} {row['ap']:8.4f}")

    # The comparison that settles it: whitened vanilla vs raw ProtSent, paired.
    print(f"\n{'paired comparison':46s} {'metric':6s} {'delta':>8s}  95% CI")
    pairs = [("ProtSent-V2-35M [raw]", "ESM-2-35M [whitened]"),
             ("ProtSent-V2-35M [whitened]", "ESM-2-35M [whitened]"),
             ("ProtSent-V2-150M [raw]", "ESM-2-150M [whitened]"),
             ("ProtSent-V2-150M [whitened]", "ESM-2-150M [whitened]")]
    report["paired"] = {}
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        report["paired"][f"{a} - {b}"] = {}
        for mm in ("hit1", "hit10", "ap"):
            d = (per[a][mm] - per[b][mm])[el]
            mean, lo, hi = boot_ci(d)
            sig = lo > 0 or hi < 0
            report["paired"][f"{a} - {b}"][mm] = {
                "delta": mean, "lo": lo, "hi": hi, "excludes_zero": bool(sig)}
            print(f"{a+' - '+b:46s} {mm:6s} {mean:+8.4f}  [{lo:+.4f}, {hi:+.4f}] "
                  f"{'significant' if sig else 'unresolved'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    # Anisotropic data: one direction dominates, plus a shared offset.
    x = rng.normal(size=(500, 8)) * np.array([50, 1, 1, 1, 1, 1, 1, 1]) + 20.0
    w = whiten(x)
    assert abs(w.mean()) < 1e-8, w.mean()
    c = np.cov(w, rowvar=False)
    off = c - np.diag(np.diag(c))
    assert np.allclose(np.diag(c), 1.0, atol=1e-6), np.diag(c)
    assert abs(off).max() < 1e-6, abs(off).max()

    # Whitening must destroy the anisotropy that inflates random-pair cosine.
    def mean_cos(a):
        u = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        s = u @ u.T
        return (s.sum() - np.trace(s)) / (len(a) * (len(a) - 1))
    assert mean_cos(x) > 0.5, mean_cos(x)
    assert abs(mean_cos(w)) < 0.1, mean_cos(w)

    # Ranking must exclude self and be a permutation of the others.
    r = rank_by_cosine(np.eye(4))
    assert r.shape == (4, 3), r.shape
    for q in range(4):
        assert q not in r[q].tolist(), (q, r[q])
        assert sorted(r[q].tolist()) == sorted(set(range(4)) - {q})
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
