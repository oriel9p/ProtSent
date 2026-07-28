#!/usr/bin/env python
"""MMseqs2-only baseline for the two structural benchmark tasks.

Answers the reviewer question "how much of ProtSent's structural performance is
just sequence similarity?" by scoring the SAME tasks with alignment instead of
embeddings, under the SAME metric definitions:

  scope40_retrieval  family-level Recall@{1,10,30}, self-match excluded --
                     identical to evaluate_retrieval() in
                     protein_benchmark_suite.py:1863-1907, with cosine-NN rank
                     replaced by MMseqs2 bitscore rank.
  remote_homology    multiclass fold prediction. Per-class score = max bitscore
                     over that class's training sequences, which yields a dense
                     score vector, so the task's AUC main metric stays
                     comparable instead of degenerating to hard 1-NN accuracy.

Queries with no hit at all are genuine misses (recall 0 / lowest-rank class),
not dropped rows -- that failure to retrieve is the point of the baseline.

Usage:
    uv run --no-sync python mmseqs_baseline.py --task scope40_retrieval
    uv run --no-sync python mmseqs_baseline.py --task remote_homology
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("mmseqs_baseline")

MMSEQS = Path(__file__).parent / "tools" / "mmseqs" / "bin" / "mmseqs"
# Sensitive search: this is a homology-detection baseline, so recall matters far
# more than runtime. Both task sets are small (~10-15k sequences).
SEARCH_FLAGS = [
    "-s", "7.5",
    "-e", "10",
    "--max-seqs", "300",
    "--alignment-mode", "3",
    "--format-output", "query,target,fident,alnlen,evalue,bits",
]


def write_fasta(seqs: list[str], path: Path) -> None:
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")


def easy_search(query: Path, target: Path, out_tsv: Path, threads: int) -> None:
    """Run mmseqs easy-search; raises with the full command on failure."""
    tmp = Path(tempfile.mkdtemp(prefix="mmseqs_bl_", dir=out_tsv.parent))
    cmd = [
        str(MMSEQS), "easy-search", str(query), str(target), str(out_tsv), str(tmp),
        *SEARCH_FLAGS, "--threads", str(threads), "--remove-tmp-files", "-v", "3",
    ]
    logger.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=24 * 3600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"mmseqs failed ({exc.returncode}): {' '.join(cmd)}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_hits(tsv: Path) -> dict[int, list[tuple[int, float]]]:
    """query_idx -> [(target_idx, bits), ...] sorted by bits descending."""
    hits: dict[int, list[tuple[int, float]]] = {}
    with open(tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            hits.setdefault(int(parts[0]), []).append((int(parts[1]), float(parts[5])))
    for v in hits.values():
        v.sort(key=lambda t: -t[1])
    return hits


def load_task(task: str, max_samples: int | None = None):
    """Load train/test exactly as the benchmark suite does with --eval_split test.

    Delegates to protein_benchmark_suite.prepare_data so the baseline sees byte
    identical inputs to the model side -- including whitespace stripping,
    split_column / auto_split handling and label parsing. Anything else would
    compare the two systems on different data.
    """
    from benchmark_tasks import TASKS
    from protein_benchmark_suite import prepare_data

    cfg = TASKS[task]
    tr_seqs, tr_labels, te_seqs, te_labels, _, meta = prepare_data(
        cfg, max_samples=max_samples, eval_split="test"
    )
    logger.info("split metadata: %s", meta)
    return cfg, (tr_seqs, tr_labels), (te_seqs, te_labels)


def eval_retrieval(hits, labels, k_list=(1, 10, 30)) -> dict[str, float]:
    """Family-level Recall@K, self excluded. Mirrors evaluate_retrieval()."""
    out = {}
    for k in k_list:
        matched = []
        for q in range(len(labels)):
            neigh = [t for t, _ in hits.get(q, []) if t != q][:k]
            matched.append(any(labels[t] == labels[q] for t in neigh))
        out[f"Recall@{k}"] = float(np.mean(matched))
    return out


def eval_regression(hits, train_labels, test_labels) -> dict[str, float]:
    """1-NN by bitscore: predict the target value of the best-aligning train seq.

    Queries with no hit fall back to the training mean -- the least-informative
    guess available, so they neither help nor are silently dropped.
    """
    from scipy.stats import spearmanr

    y_train = np.asarray(train_labels, dtype=np.float64)
    y_true = np.asarray(test_labels, dtype=np.float64)
    fallback = float(y_train.mean())
    y_pred = np.array(
        [y_train[hits[q][0][0]] if hits.get(q) else fallback
         for q in range(len(y_true))]
    )
    covered = np.array([bool(hits.get(q)) for q in range(len(y_true))])
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "Spearman": float(rho) if rho == rho else None,
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "hit_coverage": float(covered.mean()),
    }


def eval_multilabel(hits, train_labels, test_labels) -> dict[str, float]:
    """1-NN by bitscore: copy the best-aligning train sequence's label set."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    mlb.fit([list(x) for x in train_labels] + [list(x) for x in test_labels])
    Y = mlb.transform([list(x) for x in test_labels])
    empty = [()] * 0
    pred = [
        train_labels[hits[q][0][0]] if hits.get(q) else empty
        for q in range(len(test_labels))
    ]
    P = mlb.transform([list(x) for x in pred])
    covered = np.array([bool(hits.get(q)) for q in range(len(test_labels))])
    return {
        "F1_Macro": float(f1_score(Y, P, average="macro", zero_division=0)),
        "F1_Micro": float(f1_score(Y, P, average="micro", zero_division=0)),
        "hit_coverage": float(covered.mean()),
    }


def eval_multiclass(hits, train_labels, test_labels) -> dict[str, float]:
    """Per-class score = max bitscore over that class's training sequences."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    classes = sorted(set(train_labels) | set(test_labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    scores = np.zeros((len(test_labels), len(classes)), dtype=np.float32)
    for q in range(len(test_labels)):
        for t, bits in hits.get(q, []):
            j = cls_idx[train_labels[t]]
            if bits > scores[q, j]:
                scores[q, j] = bits

    y_true = np.array([cls_idx[c] for c in test_labels])
    y_pred = scores.argmax(axis=1)
    covered = (scores.max(axis=1) > 0)

    # Rows with no hit have an all-zero score vector; argmax picks class 0
    # arbitrarily. Left in deliberately -- "no homolog found" is a real error
    # mode of a search baseline and must count against it.
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hit_coverage": float(covered.mean()),
    }
    # Macro one-vs-rest AUC computed per class directly. sklearn's
    # multi_class="ovr" insists the score matrix be probabilities summing to 1,
    # which a per-class max-bitscore matrix is not; normalising it into a
    # softmax only distorts the ranking (bitscores span hundreds, so the softmax
    # saturates to one-hot). AUC only cares about the ordering within each class
    # column, so the raw scores are the right input.
    aucs = []
    for j in range(len(classes)):
        pos = y_true == j
        if pos.any() and not pos.all():  # AUC undefined for a single-class column
            aucs.append(roc_auc_score(pos, scores[:, j]))
    metrics["AUC"] = float(np.mean(aucs)) if aucs else None
    metrics["AUC_n_classes"] = len(aucs)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    help="Any key from benchmark_tasks.TASKS")
    ap.add_argument("--work_dir", default="/storage/users/ddofer/data/mmseqs_baseline")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Subsample cap passed to prepare_data (default: no cap)")
    ap.add_argument("--output", default="results/benchmarks/mmseqs_baseline.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")

    if not MMSEQS.exists():
        logger.error("mmseqs binary not found at %s", MMSEQS)
        return 1

    work = Path(args.work_dir) / args.task
    work.mkdir(parents=True, exist_ok=True)

    cfg, (train_seqs, train_labels), (test_seqs, test_labels) = load_task(
        args.task, args.max_samples
    )
    logger.info("%s [%s]: %d train / %d test sequences",
                cfg.name, cfg.problem_type, len(train_seqs), len(test_seqs))

    q_fa, t_fa = work / "query.fasta", work / "target.fasta"
    write_fasta(test_seqs, q_fa)
    write_fasta(train_seqs, t_fa)

    tsv = work / "hits.tsv"
    if tsv.exists() and tsv.stat().st_size > 0:
        logger.info("reusing existing hits: %s", tsv)
    else:
        easy_search(q_fa, t_fa, tsv, args.threads)
    hits = read_hits(tsv)
    logger.info("%d/%d queries got >=1 hit", len(hits), len(test_seqs))

    evaluator = {
        "retrieval": lambda: eval_retrieval(hits, test_labels),
        "regression": lambda: eval_regression(hits, train_labels, test_labels),
        "multilabel": lambda: eval_multilabel(hits, train_labels, test_labels),
    }.get(cfg.problem_type, lambda: eval_multiclass(hits, train_labels, test_labels))
    metrics = evaluator()

    row = {
        "Model": "MMseqs2 (alignment baseline)",
        "Task": cfg.name,
        "task_key": args.task,
        "main_metric": cfg.main_metric,
        "problem_type": cfg.problem_type,
        "eval_split": "test",  # prepare_data(eval_split="test"); NOT the suite default
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "mmseqs_flags": " ".join(SEARCH_FLAGS),
        **metrics,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing = [r for r in existing if r.get("task_key") != args.task]
    existing.append(row)
    out.write_text(json.dumps(existing, indent=2))

    print(json.dumps(row, indent=2))
    return 0


def _selfcheck() -> None:
    """Metric functions must agree with the suite's definitions."""
    # Recall: q0 and q2 share family "A"; q1 is alone in "B".
    labels = ["A", "B", "A"]
    hits = {0: [(0, 99.0), (1, 5.0), (2, 4.0)], 1: [(1, 99.0), (0, 3.0)], 2: []}
    r = eval_retrieval(hits, labels, k_list=(1, 2))
    # q0@1 -> best non-self is 1 ("B"): miss. q0@2 -> includes 2 ("A"): hit.
    # q1 never matches (no other "B"). q2 has no hits at all: miss.
    assert r["Recall@1"] == 0.0, r
    assert abs(r["Recall@2"] - 1 / 3) < 1e-9, r

    # Multiclass: test0 best-hits a class-"x" train seq, test1 has no hit.
    m = eval_multiclass({0: [(0, 50.0), (1, 10.0)]}, ["x", "y"], ["x", "y"])
    assert m["hit_coverage"] == 0.5, m
    assert m["Accuracy"] == 0.5, m  # test0 correct, test1 falls to class 0 ("x")

    # Regression / multilabel: these were defined but unrouted at one point, so
    # assert they emit the main_metric key their TaskConfigs declare.
    r = eval_regression({0: [(1, 9.0)]}, [0.0, 10.0], [10.0, 0.0])
    assert abs(r["Spearman"] - 1.0) < 1e-9 and r["hit_coverage"] == 0.5, r
    ml = eval_multilabel({0: [(0, 9.0)]}, [["a", "b"]], [["a", "b"], ["c"]])
    assert ml["F1_Micro"] > 0 and ml["hit_coverage"] == 0.5, ml
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
