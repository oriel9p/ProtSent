#!/usr/bin/env python
"""Is ProtSent-V2-150M's lower linear-probe macro-F1 on remote homology real?

The sweep reports, on the same test split with the same seed:

    vanilla ESM-2 150M   linear accuracy 0.7500  weighted-F1 0.7329  macro-F1 0.5162
    ProtSent-V2-150M     linear accuracy 0.7503  weighted-F1 0.7330  macro-F1 0.4941

Accuracy and weighted-F1 are dead ties and only macro-F1 moves, by -0.022. On a
457-class task that is the signature of a difference confined to rare classes, not
a difference in what the representation encodes. Macro-F1 weights a class with
three test examples the same as one with three hundred, and the probe is
one-vs-rest liblinear, so a rare class flipping to all-negative predictions scores
0 for that class and drags the mean.

This script does three things the sweep does not:

  1. Reproduces both numbers independently, to rule out a reporting or
     row-selection error.
  2. Bootstraps the TEST SET, paired, to put a confidence interval on the macro-F1
     difference. Paired because the same test proteins are scored by both models,
     so the comparison is far tighter than two independent intervals suggest.
  3. Reports how many classes each model predicts at all, and the macro-F1
     restricted to classes with enough support to be estimable, which is the
     comparison that is not dominated by rare-class coin flips.

Usage:
    python verify_remote_homology.py
    python verify_remote_homology.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

MODELS = {
    "ESM-2-150M": "Synthyra/ESM2-150M",
    "ProtSent-V1-150M": "oriel9p/protsent-esm2-150M",
    "ProtSent-V2-150M": "models/protsent_esm2_150m_v2/final",
}
N_BOOT = 2000


def fit_predict(tr_x, tr_y, te_x) -> np.ndarray:
    """The suite's linear probe: standardised features, one-vs-rest liblinear."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = OneVsRestClassifier(
        make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear")),
        n_jobs=-1,
    ).fit(tr_x, tr_y)
    return clf.predict(te_x)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def macro_f1_min_support(y_true, y_pred, min_support: int) -> tuple[float, int]:
    """Macro-F1 over classes with at least `min_support` test examples.

    Rare classes are where one-vs-rest predictions are least stable, so this
    separates "the representation is worse" from "a handful of tiny classes
    flipped".
    """
    from sklearn.metrics import f1_score

    counts = Counter(y_true.tolist())
    keep = [c for c, n in counts.items() if n >= min_support]
    if not keep:
        return float("nan"), 0
    scores = f1_score(y_true, y_pred, labels=keep, average=None, zero_division=0)
    return float(np.mean(scores)), len(keep)


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """CI on (b - a) for accuracy and macro-F1, resampling test rows."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_acc, d_mac = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pa, pb = y_true[idx], pred_a[idx], pred_b[idx]
        d_acc.append((pb == yt).mean() - (pa == yt).mean())
        d_mac.append(f1_score(yt, pb, average="macro", zero_division=0)
                     - f1_score(yt, pa, average="macro", zero_division=0))
    out = {}
    for name, v in (("accuracy", d_acc), ("macro_f1", d_mac)):
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[name] = {"delta": float(v.mean()), "lo": float(lo), "hi": float(hi),
                     "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="results/benchmarks/verify_remote_homology_150m.json")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_tasks import TASKS
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    cfg = TASKS["remote_homology"]
    ds = load_dataset(cfg.dataset)
    seq_col = list(cfg.input_map.values())[0]
    tr, te = ds["train"], ds["test"]
    tr_seq, tr_y = list(tr[seq_col]), np.array(tr[cfg.label_col])
    te_seq, te_y = list(te[seq_col]), np.array(te[cfg.label_col])
    print(f"train {len(tr_seq)}  test {len(te_seq)}  classes {len(set(tr_y.tolist()))}")

    sup = Counter(te_y.tolist())
    print(f"test-set class support: median {np.median(list(sup.values())):.0f}, "
          f"{sum(1 for v in sup.values() if v <= 2)} classes with <=2 examples "
          f"of {len(sup)}")

    report = {"n_train": len(tr_seq), "n_test": len(te_seq),
              "n_classes_test": len(sup), "models": {}, "paired_vs_vanilla": {}}
    preds = {}
    for name, path in MODELS.items():
        m = SentenceTransformer(path, device="cuda", trust_remote_code=True)
        e = np.asarray(m.encode(tr_seq + te_seq, batch_size=args.batch_size,
                                show_progress_bar=False))
        p = fit_predict(e[:len(tr_seq)], tr_y, e[len(tr_seq):])
        preds[name] = p
        mt = metrics(te_y, p)
        for ms in (3, 5, 10):
            v, k = macro_f1_min_support(te_y, p, ms)
            mt[f"macro_f1_support>={ms}"] = v
            mt[f"n_classes_support>={ms}"] = k
        mt["n_classes_predicted"] = int(len(set(p.tolist())))
        report["models"][name] = mt
        print(f"\n{name}")
        for k, v in mt.items():
            print(f"   {k:26s} {v:.4f}" if isinstance(v, float) else f"   {k:26s} {v}")

    base = "ESM-2-150M"
    for name in MODELS:
        if name == base:
            continue
        print(f"\npaired bootstrap {name} - {base} ({args.n_boot} resamples of the test set)")
        pb = paired_bootstrap(te_y, preds[base], preds[name], args.n_boot)
        report["paired_vs_vanilla"][f"{name} - {base}"] = pb
        for k, v in pb.items():
            verdict = "significant" if v["excludes_zero"] else "UNRESOLVED"
            print(f"   {k:12s} {v['delta']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]  {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    y = np.array([0] * 50 + [1] * 50 + [2] * 3)  # class 2 is rare
    perfect = y.copy()
    # Miss only the rare class: accuracy barely moves, macro-F1 falls a lot.
    miss_rare = y.copy()
    miss_rare[y == 2] = 0
    m_perfect, m_miss = metrics(y, perfect), metrics(y, miss_rare)
    assert m_perfect["accuracy"] - m_miss["accuracy"] < 0.05, (m_perfect, m_miss)
    assert m_perfect["macro_f1"] - m_miss["macro_f1"] > 0.25, (m_perfect, m_miss)

    # Restricting to well-supported classes must hide that rare-class collapse.
    v_all, _ = macro_f1_min_support(y, miss_rare, 1)
    v_sup, k = macro_f1_min_support(y, miss_rare, 10)
    assert v_sup > v_all and k == 2, (v_all, v_sup, k)

    # Identical predictions must give a delta CI containing zero.
    pb = paired_bootstrap(y, perfect, perfect, n_boot=200)
    assert not pb["macro_f1"]["excludes_zero"], pb
    assert abs(pb["macro_f1"]["delta"]) < 1e-9, pb
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
