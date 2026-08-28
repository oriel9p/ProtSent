#!/usr/bin/env python
"""Run ProtBench tasks with MaxSim as the kNN metric instead of pooled cosine.

ProtBench's knn probe is `KNeighborsClassifier(metric="euclidean", algorithm="brute")` over
mean-pooled embeddings. MaxSim is just a different similarity between two proteins, so it drops
into the same protocol: same splits, same k, same metrics, only the neighbour ranking changes.
That makes MaxSim-vs-cosine a controlled comparison rather than a different experiment.

Reuses ProtBench's `prepare_data` (splits, label maps, top-k label handling) and its
`classification_metrics` so the columns match the rest of the suite. Only the vote is local:
sklearn's `metric="precomputed"` needs a square train x train distance matrix at fit time, which for
fold_prediction is 12,312^2 = 151M MaxSim pairs against the 40M the test x train matrix needs. Four
times the compute to borrow eight lines is not a trade worth making.

Multilabel tasks (EC) are out of scope: a k-vote over label SETS is not what ProtBench does there --
it auto-switches EC to a linear probe -- so scoring it this way would not be comparable.

    python maxsim_knn_bench.py --task remote_homology --models NAME=late:PATH --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import late_interaction as li  # noqa: E402
from late_interaction_eval import append_csv, parse_model_spec  # noqa: E402
from bootstrap_ci import boot_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maxsim_knn")


def knn_predict(sim: np.ndarray, gallery_y: np.ndarray, *, k: int, regression: bool) -> np.ndarray:
    """Predict for each row of `sim` (higher = nearer) from its k nearest gallery items.

    Classification ties break toward the nearest neighbour, matching cmd_fewshot_rh.
    """
    k = min(k, sim.shape[1])
    # argpartition for the top-k, then sort just those: O(n) rather than a full argsort per row.
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(top, np.argsort(-np.take_along_axis(sim, top, 1), axis=1), axis=1)
    if regression:
        return gallery_y[order].mean(axis=1)
    out = []
    for row in order:
        labels = gallery_y[row]
        vals, counts = np.unique(labels, return_counts=True)
        best = counts.max()
        winners = set(vals[counts == best].tolist())
        out.append(next(l for l in labels if l in winners))  # nearest among the tied labels
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="ProtBench task key, e.g. remote_homology")
    ap.add_argument("--models", action="append", required=True, metavar="NAME=KIND:PATH")
    ap.add_argument("--protbench", default="/opt/hpc/ddofer/ProtBench")
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--chunk_elements", type=int, default=50_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", default="results/late_interaction/r2_final/benchmarks")
    args = ap.parse_args()

    sys.path.insert(0, args.protbench)
    import benchmark_tasks as bt
    from protein_benchmark_suite import prepare_data, classification_metrics

    cfg = bt.TASKS[args.task]
    if cfg.problem_type == "multilabel":
        raise SystemExit(f"{args.task} is multilabel; ProtBench scores it with a linear probe, so a "
                         "kNN vote here would not be comparable. Pick a binary/multiclass/regression task.")
    tr_seqs, tr_y, te_seqs, te_y, _mlb, _meta = prepare_data(cfg, eval_split=args.eval_split)
    tr_y, te_y = np.asarray(tr_y), np.asarray(te_y)
    regression = cfg.problem_type == "regression"
    logger.info("%s: %d gallery, %d eval, problem=%s", args.task, len(tr_seqs), len(te_seqs),
                cfg.problem_type)

    rows, per_query = [], {}
    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        mve = (li.load_multivector_encoder(path, device=args.device) if kind == "late"
               else li.build_multivector_encoder(path, proj_dim=0,
                                                 max_seq_length=args.max_seq_length,
                                                 device=args.device)[0])
        mve.max_seq_length = args.max_seq_length
        t0 = time.time()
        sim = li.maxsim_matrix(mve, list(tr_seqs), batch_size=args.batch_size,
                               chunk_elements=args.chunk_elements, queries=list(te_seqs))
        pred = knn_predict(sim, tr_y, k=args.knn_k, regression=regression)
        runtime_s = time.time() - t0

        # Bootstrap over the eval queries, the same axis and estimator the SCOPe and few-shot rows
        # use, so an interval here means what it means there. Per-query values are kept so two arms
        # can be compared paired later; marginal intervals overlap freely between arms a paired test
        # separates cleanly.
        if regression:
            from scipy.stats import pearsonr, spearmanr
            m = {"Spearman": float(spearmanr(te_y, pred).statistic),
                 "Pearson": float(pearsonr(te_y, pred).statistic),
                 "MAE": float(np.abs(te_y - pred).mean())}
            per_q = np.abs(te_y - pred)          # per-query error; pair on this
            _, lo, hi = boot_ci(per_q, n_boot=args.n_boot, seed=42)
            m["MAE_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        else:
            m = classification_metrics(cfg.problem_type, te_y, pred)
            per_q = (pred == te_y).astype(float)  # per-query correctness
            _, lo, hi = boot_ci(per_q, n_boot=args.n_boot, seed=42)
            m["Accuracy_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        per_query[f"{args.task}|{name}"] = per_q
        rows.append({"Task": cfg.name, "task_key": args.task, "model": name, "scoring": "maxsim",
                     "probe": f"knn{args.knn_k}", "n_gallery": len(tr_seqs), "n_eval": len(te_seqs),
                     "runtime_s": round(runtime_s, 2), **m})
        logger.info("%s: %s", name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
        del mve, sim
        torch.cuda.empty_cache()

    append_csv(Path(args.out_dir) / "maxsim_knn_bench.csv", rows)
    # Merge, never overwrite: rows are appended to the CSV across invocations, so rewriting this
    # would delete earlier tasks' vectors and defeat the paired comparison they exist for.
    npz = Path(args.out_dir) / "maxsim_knn_per_query.npz"
    merged = {}
    if npz.exists():
        with np.load(npz) as prev:
            merged.update({k: prev[k] for k in prev.files})
    merged.update(per_query)
    np.savez_compressed(npz, **merged)
    logger.info("wrote %d rows -> %s", len(rows), Path(args.out_dir) / "maxsim_knn_bench.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
