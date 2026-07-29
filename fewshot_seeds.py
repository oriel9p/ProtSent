#!/usr/bin/env python
"""Few-shot transfer with seed variability: 3-NN versus a trained linear head.

Reviewer HNXd asked for three things this answers together:

  * absolute scores, not only relative percentages (their Table 5 complaint: a
    +244% change from a near-zero baseline can be a tiny absolute gain);
  * a variability analysis over multiple random seeds for the few-shot setting;
  * a linear-classifier baseline, because comparing only to k-NN "does not
    reflect how practitioners typically use ESM2".

They also proposed the framing they would find convincing: that under label
scarcity a standard linear classifier degrades substantially while k-NN stays
competitive because of contrastive alignment. That is a testable claim, and it is
tested here rather than asserted.

Why not use the benchmark suite's --max_samples: it caps the EVAL split as well
as the train split, so the test set would shrink and change with the seed,
conflating train-subsampling variance with test-set variance. Here the test set
is held fixed at full size and only the training subset is resampled, so the
spread reported is exactly the quantity HNXd asked about.

Embeddings are computed once per model and reused across every N and seed, so the
cost is one forward pass per model.

Usage:
    python fewshot_seeds.py --tasks remote_homology solubility metal_ion_binding
    python fewshot_seeds.py --selfcheck
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
    "ProtSent-V1": "oriel9p/protsent-esm2-35M",
    "ProtSent-V2": "models/protsent_esm2_35m_v3/final",
}
SHOTS = [50, 100, 250, 1000]
SEEDS = [0, 1, 2, 3, 4]


def probe_scores(tr_x, tr_y, te_x, te_y, problem: str) -> dict[str, float]:
    """3-NN and a trained linear head on the same split. Returns both scores."""
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    out = {}
    k = min(3, len(tr_y))
    if problem == "regression":
        from scipy.stats import spearmanr
        from sklearn.linear_model import RidgeCV

        knn = KNeighborsRegressor(n_neighbors=k).fit(tr_x, tr_y)
        out["knn"] = float(spearmanr(knn.predict(te_x), te_y).statistic)
        sc = StandardScaler().fit(tr_x)
        lin = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(tr_x), tr_y)
        out["linear"] = float(spearmanr(lin.predict(sc.transform(te_x)), te_y).statistic)
    else:
        from sklearn.linear_model import LogisticRegression

        knn = KNeighborsClassifier(n_neighbors=k).fit(tr_x, tr_y)
        out["knn"] = float((knn.predict(te_x) == te_y).mean())
        sc = StandardScaler().fit(tr_x)
        lin = LogisticRegression(max_iter=3000).fit(sc.transform(tr_x), tr_y)
        out["linear"] = float((lin.predict(sc.transform(te_x)) == te_y).mean())
    return out


def stratified_subset(y: np.ndarray, n: int, seed: int, problem: str) -> np.ndarray:
    """Indices of an n-row training subset.

    Classification keeps at least one example per class where possible, because a
    plain random draw at n=50 can miss classes entirely and produce a degenerate
    fit that is a property of the draw rather than of the embedding.
    """
    rng = np.random.default_rng(seed)
    if problem == "regression" or n >= len(y):
        return rng.permutation(len(y))[:n]
    idx = []
    classes = np.unique(y)
    for c in classes[: n]:
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool))
    remaining = np.setdiff1d(rng.permutation(len(y)), np.array(idx))
    idx = np.array(idx)
    return np.concatenate([idx, remaining[: max(0, n - len(idx))]])


def run_task(task: str, embs: dict, labels, problem: str) -> dict:
    tr_y, te_y = labels
    res = {"task": task, "problem": problem, "by_model": {}}
    print(f"\n=== {task} ({problem}) ===", flush=True)
    for name, (tr_x, te_x) in embs.items():
        res["by_model"][name] = {}
        for n in SHOTS:
            if n > len(tr_y):
                continue
            rows = {"knn": [], "linear": []}
            for s in SEEDS:
                sel = stratified_subset(tr_y, n, s, problem)
                sc = probe_scores(tr_x[sel], tr_y[sel], te_x, te_y, problem)
                for kk in rows:
                    rows[kk].append(sc[kk])
            res["by_model"][name][str(n)] = {
                kk: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "seeds": v}
                for kk, v in rows.items()
            }
            print(f"  {name:12s} N={n:5d}  knn {np.mean(rows['knn']):.4f}+-{np.std(rows['knn']):.4f}"
                  f"   linear {np.mean(rows['linear']):.4f}+-{np.std(rows['linear']):.4f}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+",
                    default=["remote_homology", "solubility", "metal_ion_binding"])
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_test", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="results/benchmarks/fewshot_seeds.json")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_tasks import TASKS
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    loaded = {n: SentenceTransformer(p, device="cuda", trust_remote_code=True)
              for n, p in MODELS.items()}

    report = []
    for task in args.tasks:
        cfg = TASKS[task]
        ds = load_dataset(cfg.dataset)
        seq_col = list(cfg.input_map.values())[0]
        test_key = "test" if "test" in ds else "valid"

        def take(split, n):
            d = ds[split]
            if len(d) > n:
                d = d.shuffle(seed=0).select(range(n))
            return list(d[seq_col]), np.array(d[cfg.label_col])

        tr_seq, tr_y = take("train", args.max_train)
        te_seq, te_y = take(test_key, args.max_test)
        problem = "regression" if cfg.problem_type == "regression" else "classification"

        embs = {}
        for name, m in loaded.items():
            e = np.asarray(m.encode(tr_seq + te_seq, batch_size=args.batch_size,
                                    show_progress_bar=False))
            embs[name] = (e[:len(tr_seq)], e[len(tr_seq):])
        report.append(run_task(task, embs, (tr_y, te_y), problem))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    # Two well-separated classes: both probes should be near-perfect.
    x = np.concatenate([rng.normal(0, 0.3, (200, 4)), rng.normal(4, 0.3, (200, 4))])
    y = np.array([0] * 200 + [1] * 200)
    sc = probe_scores(x[:300], y[:300], x[300:], y[300:], "classification")
    assert sc["knn"] > 0.9 and sc["linear"] > 0.9, sc

    # Stratified subset must cover every class and return exactly n rows.
    y3 = np.array([0] * 50 + [1] * 50 + [2] * 50)
    idx = stratified_subset(y3, 10, 0, "classification")
    assert len(idx) == 10, len(idx)
    assert set(np.unique(y3[idx])) == {0, 1, 2}, np.unique(y3[idx])
    # Different seeds must give different draws, or the "variability" is fake.
    assert not np.array_equal(stratified_subset(y3, 20, 0, "classification"),
                              stratified_subset(y3, 20, 1, "classification"))
    # Regression path returns n indices and ignores stratification.
    assert len(stratified_subset(np.arange(100.0), 15, 0, "regression")) == 15
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
