#!/usr/bin/env python
"""Evaluate dense-cosine vs residue-MaxSim protein retrieval.

Model specs (repeat --models): NAME=KIND:PATH with KIND one of
    dense     mean-pooled SentenceTransformer, cosine scoring
    zeroshot  backbone loaded with proj_dim=0 -> native residue MaxSim
    late      trained MultiVectorEncoder dir -> projected residue MaxSim

Subcommands:
    scope      all-vs-all SCOPe-40, metrics at fold/superfamily/family
               (+ per-query .npz, paired bootstrap vs --reference)
    scope --checkpoints RUN_DIR   MaxSim SCOPe curve over checkpoint-*/ dirs
    fewshot_rh few-shot Remote Homology (fold_prediction) 3-NN vote, N=100/500/1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import late_interaction as li  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("late_interaction_eval")


def parse_model_spec(spec: str):
    name, rest = spec.split("=", 1)
    kind, path = rest.split(":", 1)
    if kind not in {"dense", "zeroshot", "late"}:
        raise ValueError(f"unknown model kind {kind!r} in {spec!r}")
    return name, kind, path


def load_scorer(kind: str, path: str, *, max_seq_length: int, device):
    """Return (score_fn(seqs[, queries]) -> np.ndarray, scoring_label)."""
    if kind == "dense":
        # Same loader as training (handles FastPLM/ESM alike, enforces max_seq_length).
        _, st = li.build_multivector_encoder(path, proj_dim=0, max_seq_length=max_seq_length, device=device)
        return (lambda seqs, queries=None, bs=64: li.cosine_matrix(st, seqs, batch_size=bs, queries=queries)), "cosine"
    if kind == "zeroshot":
        mve, _ = li.build_multivector_encoder(path, proj_dim=0, max_seq_length=max_seq_length, device=device)
    else:
        mve = li.load_multivector_encoder(path, device=device)
        mve.max_seq_length = max_seq_length
    return (
        lambda seqs, queries=None, bs=64, ce=50_000_000: li.maxsim_matrix(
            mve, seqs, batch_size=bs, chunk_elements=ce, queries=queries
        ),
        "maxsim",
    )


def append_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), path)


def cmd_scope(args) -> None:
    seqs, families = li.load_scope40()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [parse_model_spec(s) for s in args.models or []]
    if args.checkpoints:
        run = Path(args.checkpoints)
        ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if (run / "step0" / "late").exists():
            specs.append((f"{run.name}@0", "late", str(run / "step0" / "late")))
        specs += [(f"{run.name}@{c.name.split('-')[1]}", "late", str(c)) for c in ckpts]
        if (run / "late").exists():
            specs.append((f"{run.name}@final", "late", str(run / "late")))

    all_rows, per_query_by_model = [], {}
    for name, kind, path in specs:
        scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
        t0 = time.time()
        sim = scorer(seqs, bs=args.batch_size) if scoring == "cosine" else scorer(
            seqs, bs=args.batch_size, ce=args.chunk_elements
        )
        runtime_s = time.time() - t0
        rows, pq = li.scope_rows(
            sim, families, model=name, scoring=scoring, n_boot=args.n_boot, seed=args.seed,
            runtime_s=round(runtime_s, 2),
        )
        all_rows += rows
        per_query_by_model[name] = pq
        np.savez_compressed(
            out / f"per_query_{name.replace('/', '_')}.npz",
            **{f"{lvl}_{k}": v for lvl, d in pq.items() for k, v in d.items()},
        )
        logger.info("%s (%s): %.1fs scoring; family eligible R@10=%.4f", name, scoring, runtime_s,
                    [r for r in rows if r["level"] == "family"][0].get("eligible_Recall@10", float("nan")))
        del scorer, sim

    append_csv(out / ("scope_checkpoint_curve.csv" if args.checkpoints else "scope_hierarchy.csv"), all_rows)

    if args.reference and args.reference in per_query_by_model:
        ref = per_query_by_model[args.reference]
        deltas = {
            name: {lvl: li.paired_bootstrap(pq[lvl], ref[lvl], n_boot=args.n_boot, seed=args.seed)
                   for lvl in li.SCOPE_LEVELS}
            for name, pq in per_query_by_model.items() if name != args.reference
        }
        p = out / "scope_pairwise_bootstrap.json"
        existing = json.loads(p.read_text()) if p.exists() else {}
        existing[f"vs_{args.reference}"] = deltas
        p.write_text(json.dumps(existing, indent=2))
        logger.info("paired bootstrap vs %s -> %s", args.reference, p)


def cmd_fewshot_rh(args) -> None:
    from datasets import load_dataset

    ds = load_dataset("biomap-research/fold_prediction")
    train, test = ds["train"], ds["test"]
    tr_seqs, tr_y = list(train["seq"]), np.asarray(train["label"])
    te_seqs, te_y = list(test["seq"]), np.asarray(test["label"])
    if args.max_test and len(te_seqs) > args.max_test:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(te_seqs), args.max_test, replace=False)
        te_seqs, te_y = [te_seqs[i] for i in idx], te_y[idx]

    rows = []
    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
        for budget in args.budgets:
            rng = np.random.default_rng(args.seed)  # same draw for every model
            sub = rng.choice(len(tr_seqs), min(budget, len(tr_seqs)), replace=False)
            gallery = [tr_seqs[i] for i in sub]
            gal_y = tr_y[sub]
            t0 = time.time()
            sim = scorer(gallery, queries=te_seqs, bs=args.batch_size)
            order = np.argsort(-sim, axis=1)[:, : args.knn_k]
            preds = []
            for r in order:
                votes = Counter(gal_y[r])
                top = max(votes.values())
                preds.append(next(lbl for lbl in gal_y[r] if votes[lbl] == top))  # tie -> nearest
            preds = np.asarray(preds)
            from sklearn.metrics import accuracy_score, f1_score

            rows.append({
                "model": name, "scoring": scoring, "budget": int(budget), "knn_k": args.knn_k,
                "seed": args.seed, "n_test": len(te_y),
                "accuracy": float(accuracy_score(te_y, preds)),
                "f1_macro": float(f1_score(te_y, preds, average="macro")),
                "runtime_s": round(time.time() - t0, 2),
            })
            logger.info("%s N=%d: acc=%.4f", name, budget, rows[-1]["accuracy"])
        del scorer
    append_csv(Path(args.out_dir) / "late_fewshot_knn.csv", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", action="append", default=[], metavar="NAME=KIND:PATH")
    common.add_argument("--max_seq_length", type=int, default=512)
    common.add_argument("--batch_size", type=int, default=64)
    common.add_argument("--device", default=None)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--out_dir", default="results/late_interaction/pilot_35m/scope")

    ps = sub.add_parser("scope", parents=[common])
    ps.add_argument("--chunk_elements", type=int, default=50_000_000)
    ps.add_argument("--n_boot", type=int, default=1000)
    ps.add_argument("--reference", default=None, help="Model NAME for paired bootstrap deltas")
    ps.add_argument("--checkpoints", default=None, help="Training run dir: evaluate step0 + checkpoint-* + final")
    ps.set_defaults(fn=cmd_scope)

    pf = sub.add_parser("fewshot_rh", parents=[common])
    pf.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000])
    pf.add_argument("--knn_k", type=int, default=3)
    pf.add_argument("--max_test", type=int, default=0, help="Subsample the test split (0 = all)")
    pf.set_defaults(fn=cmd_fewshot_rh)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
