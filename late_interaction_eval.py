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
import csv
import itertools
import logging
import os
import sys
import time
from collections import Counter
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import late_interaction as li  # noqa: E402
from bootstrap_ci import boot_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("late_interaction_eval")

# Mirrors ProtBench's _PROTEINGYM_VARIANTS so the two report on identical data and grouping.
PROTEINGYM_VARIANTS = {
    "dms_substitutions": {"data_dir": "DMS_substitutions", "label": "DMS_score",
                          "group_by": "DMS_id", "mutant": "mutated_sequence", "wt": "target_seq"},
    "dms_indels": {"data_dir": "DMS_indels", "label": "DMS_score",
                   "group_by": "DMS_id", "mutant": "mutated_sequence", "wt": "target_seq"},
}


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
        # ce is accepted and ignored so both scorers share one signature; the caller should not
        # have to know which kind it got in order to call it.
        return (lambda seqs, queries=None, bs=64, ce=None: li.cosine_matrix(
            st, seqs, batch_size=bs, queries=queries)), "cosine"
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


def save_per_query(out: Path, name: str, pq: dict) -> Path:
    """Write the per-query vectors that every paired test downstream reads.

    This file is the interchange format between measurement and analysis (paired bootstrap,
    identity strata, McNemar), so its name and key layout live here rather than being
    restated at each call site -- they had already started to drift.
    """
    path = out / f"per_query_{name.replace('/', '_')}.npz"
    np.savez_compressed(path, **{f"{lvl}_{k}": v for lvl, d in pq.items() for k, v in d.items()})
    return path


def append_csv(path: Path, rows: list[dict]) -> None:
    """Append rows, reusing the file's existing header so columns cannot shift.

    A run that emits more columns than the file already has (bootstrap CIs, say) would
    otherwise write its own wider header order into the middle of the file and silently
    offset every value; that file is rewritten in full instead.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    old_rows: list[dict] = []
    if path.exists():
        with path.open() as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            if set(keys) <= set(header):
                with path.open("a", newline="") as out:
                    csv.DictWriter(out, fieldnames=header).writerows(rows)
                logger.info("wrote %d rows -> %s", len(rows), path)
                return
            old_rows = list(reader)
            keys = sorted(set(keys) | set(header))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(old_rows + rows)
    logger.info("wrote %d rows -> %s", len(rows), path)


def subsample(seqs: list, labels, n: int, seed: int):
    """Take a seeded random n of (seqs, labels), or return them unchanged if n is 0 or too big."""
    if not n or len(seqs) <= n:
        return seqs, labels
    idx = np.random.default_rng(seed).choice(len(seqs), n, replace=False)
    return [seqs[i] for i in idx], labels[idx]


def curve_points(run: Path, prefix: str):
    """Yield ``(name, kind, path)`` for a training run's scoreable points, in training order.

    step0 -> checkpoint-* (numeric order) -> the exported final model. Both the one-shot `scope`
    command and the live `watch_curve` walk exactly this sequence, and used to walk it with two
    hand-written copies that disagreed about the trailing @final point.
    """
    if (run / "step0" / "late").exists():
        yield f"{prefix}@0", "late", str(run / "step0" / "late")
    for ckpt in sorted(run.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1])):
        if (ckpt / "modules.json").exists():  # written last, so its presence means the save finished
            yield f"{prefix}@{ckpt.name.split('-')[1]}", "late", str(ckpt)
    if (run / "late").exists():
        yield f"{prefix}@final", "late", str(run / "late")


def score_scope(name: str, kind: str, path: str, seqs, families, out: Path, args):
    """Score one model on SCOPe-40 and persist its per-query vectors. Returns ``(rows, pq)``.

    The bootstrap and the .npz are computed here or never: a live watcher's checkpoint is
    rotated off disk by the next save, so there is no second chance to rescore it.
    """
    scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
    t0 = time.time()
    sim = scorer(seqs, bs=args.batch_size, ce=args.chunk_elements)
    runtime_s = time.time() - t0
    rows, pq = li.scope_rows(sim, families, model=name, scoring=scoring, n_boot=args.n_boot,
                             seed=args.seed, runtime_s=round(runtime_s, 2))
    save_per_query(out, name, pq)
    family = next(r for r in rows if r["level"] == "family")
    logger.info("%s (%s): %.1fs scoring; family eligible R@10=%.4f", name, scoring, runtime_s,
                family.get("eligible_Recall@10", float("nan")))
    del scorer, sim
    torch.cuda.empty_cache()
    return rows, pq


def cmd_scope(args) -> None:
    seqs, families = li.load_scope40()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [parse_model_spec(s) for s in args.models or []]
    if args.checkpoints:
        run = Path(args.checkpoints)
        specs += list(curve_points(run, run.name))

    all_rows, per_query_by_model = [], {}
    for name, kind, path in specs:
        rows, pq = score_scope(name, kind, path, seqs, families, out, args)
        all_rows += rows
        per_query_by_model[name] = pq

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
    te_seqs, te_y = subsample(te_seqs, te_y, args.max_test, args.seed)

    rows = []
    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
        # One permutation, sliced per budget, so the budgets are nested and the query side is
        # encoded once instead of once per budget. rng.choice per budget gave three unrelated
        # galleries, which meant re-encoding all 3,244 test sequences for each one.
        perm = np.random.default_rng(args.seed).permutation(len(tr_seqs))
        for budget in args.budgets:
            sub = perm[: min(budget, len(tr_seqs))]
            gallery = [tr_seqs[i] for i in sub]
            gal_y = tr_y[sub]
            t0 = time.time()
            sim = scorer(gallery, queries=te_seqs, bs=args.batch_size, ce=args.chunk_elements)
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
        torch.cuda.empty_cache()  # 8 arms in one invocation, each holding a corpus
    append_csv(Path(args.out_dir) / "late_fewshot_knn.csv", rows)


def cmd_proteingym(args) -> None:
    """Zero-shot ProteinGym with MaxSim: score each variant by its similarity to the wild type.

    ProtBench scores an embedding model here as cosine(mutant, WT). The late-interaction analogue
    is MaxSim(mutant, WT), which is what this measures -- same data, same metric, same grouping,
    so the two are directly comparable.

    Scores are MEAN MaxSim (divided by query length), not the raw sum. Raw MaxSim sums over query
    residues, so it scales with mutant length: harmless for substitutions, where every variant in
    an assay is the same length, but a pure length artifact for indels. Dividing is a per-assay
    monotone transform when lengths are equal, so it cannot change the substitution numbers.
    """
    from datasets import load_dataset
    from scipy.stats import spearmanr

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PROTEINGYM_VARIANTS[args.variant]
    ds = load_dataset("OATML-Markslab/ProteinGym_v1", data_dir=cfg["data_dir"], split="train")
    logger.info("%s: %d rows", args.variant, len(ds))

    groups: dict = {}
    for i, g in enumerate(ds[cfg["group_by"]]):
        groups.setdefault(g, []).append(i)
    assays = sorted(groups)
    if args.max_assays:
        assays = assays[: args.max_assays]
    logger.info("%d assays (of %d)", len(assays), len(groups))

    rows = []
    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
        t0 = time.time()
        per_assay = []
        for a in assays:
            idx = groups[a]
            if args.max_variants_per_assay and len(idx) > args.max_variants_per_assay:
                idx = list(np.random.default_rng(args.seed).choice(idx, args.max_variants_per_assay, replace=False))
            mut = [ds[int(i)][cfg["mutant"]] for i in idx]
            y = np.asarray([ds[int(i)][cfg["label"]] for i in idx], dtype=float)
            if len(set(y.tolist())) < 2:
                continue  # a constant label has no rank correlation to report
            wt = ds[int(idx[0])][cfg["wt"]]
            sim = scorer([wt], queries=mut, bs=args.batch_size, ce=args.chunk_elements)[:, 0]
            lens = np.array([min(len(s), args.max_seq_length - 2) for s in mut], dtype=float)
            rho = spearmanr(sim / lens, y).statistic
            if np.isfinite(rho):
                per_assay.append({"assay": a, "spearman": float(rho), "n": len(idx)})
        del scorer
        torch.cuda.empty_cache()
        if not per_assay:
            logger.warning("%s: no scoreable assay", name)
            continue
        rhos = np.array([r["spearman"] for r in per_assay])
        _, lo, hi = boot_ci(rhos, n_boot=args.n_boot, seed=args.seed)
        rows.append({"model": name, "scoring": scoring, "variant": args.variant,
                     "mean_spearman": float(rhos.mean()), "ci95": f"[{lo:.4f}, {hi:.4f}]",
                     "n_assays": len(per_assay), "runtime_s": round(time.time() - t0, 1)})
        logger.info("%s %s: mean Spearman %.4f over %d assays", name, args.variant,
                    rhos.mean(), len(per_assay))
        np.savez_compressed(out / f"proteingym_{args.variant}_{name.replace('/', '_')}.npz",
                            assay=np.array([r["assay"] for r in per_assay]), spearman=rhos)
    append_csv(out / "proteingym_maxsim.csv", rows)


def cmd_watch_curve(args) -> None:
    """Poll a training run for new checkpoint-* dirs and score SCOPe before they are deleted.

    Runs alongside training on the same GPU (small model + 2.2k sequences), so the run can keep
    save_total_limit=1 and still produce a training curve. Idempotent: already-scored steps are
    skipped, so restarting the watcher after a crash is safe.
    """
    run = Path(args.run_dir)
    out = Path(args.out_dir)
    curve = out / "scope_checkpoint_curve.csv"
    seqs, families = li.load_scope40()

    def already() -> set[str]:
        if not curve.exists():
            return set()
        return {r["model"] for r in csv.DictReader(curve.open())}

    def score(name: str, kind: str, path: str) -> None:
        try:
            rows, _ = score_scope(name, kind, path, seqs, families, out, args)
            append_csv(curve, rows)
        except Exception as exc:  # a busy GPU or a half-written checkpoint must not kill the watcher
            logger.warning("curve point %s failed: %s", name, exc)

    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        done = already()
        for name, kind, path in curve_points(run, args.name):
            # @final is handled below, once training has actually stopped.
            if not name.endswith("@final") and name not in done:
                score(name, kind, path)
        if args.follow_pid:
            try:
                os.kill(args.follow_pid, 0)
            except (ProcessLookupError, PermissionError):
                if not (run / "late").exists():
                    logger.warning("trainer %d exited without exporting %s; watcher stopping",
                                   args.follow_pid, run)
                    return
        if (run / "late").exists():  # training finished and exported
            # `late/` is exported from the same weights as the last checkpoint, so scoring it
            # again just spends ~3 GPU-minutes to write a bit-identical row into the curve.
            # Confirmed on the 35M runs: esm2_late@2000 and @final have identical per-query AP
            # vectors at every level. Only score it if no checkpoint was captured at all.
            done = already()
            scored_any = any(m.startswith(f"{args.name}@") and not m.endswith("@final") for m in done)
            if not scored_any and f"{args.name}@final" not in done:
                score(f"{args.name}@final", "late", str(run / "late"))
            logger.info("curve watcher done for %s", args.name)
            return
        time.sleep(args.poll_seconds)
    logger.warning("curve watcher timed out for %s", args.name)


def cmd_cath(args) -> None:
    """CATH midnight-zone (ProtTucker setting): 1-NN superfamily transfer, test_h vs the lookup set."""
    from datasets import load_dataset

    ds = load_dataset("GrimSqueaker/cath43-eat")
    lookup, test = ds["lookup"], ds[args.test_split]
    gal_seqs, gal_y = list(lookup["sequence"]), np.asarray(lookup[args.label_col])
    q_seqs, q_y = list(test["sequence"]), np.asarray(test[args.label_col])
    gal_seqs, gal_y = subsample(gal_seqs, gal_y, args.max_lookup, args.seed)

    rows = []
    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        scorer, scoring = load_scorer(kind, path, max_seq_length=args.max_seq_length, device=args.device)
        t0 = time.time()
        sim = scorer(gal_seqs, queries=q_seqs, bs=args.batch_size, ce=args.chunk_elements)
        pred = gal_y[np.argmax(sim, axis=1)]
        correct = (pred == q_y)
        acc = float(correct.mean())
        # test_h is 150 queries, so one query is 0.67 points and a 3-point gap is five
        # proteins. Save per-query correctness so arms can be compared with McNemar
        # rather than by eyeballing accuracies, and carry the marginal CI in the row.
        _, lo, hi = boot_ci(correct.astype(float), n_boot=args.n_boot, seed=args.seed)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)  # before a 69k-sequence encode is thrown away
        np.savez_compressed(Path(args.out_dir) / f"cath_per_query_{name.replace('/', '_')}.npz",
                            correct=correct, labels=q_y)
        rows.append({
            "model": name, "scoring": scoring, "level": args.label_col, "test_split": args.test_split,
            "accuracy": acc, "ci95": f"[{lo:.4f}, {hi:.4f}]",
            "n_queries": int(len(q_y)), "n_correct": int(correct.sum()),
            "n_lookup": int(len(gal_y)), "runtime_s": round(time.time() - t0, 2),
        })
        logger.info("%s cath %s: acc=%.4f", name, args.test_split, rows[-1]["accuracy"])
        del scorer, sim
        torch.cuda.empty_cache()
    append_csv(Path(args.out_dir) / "cath_eat.csv", rows)
    if args.mcnemar:
        cath_mcnemar(Path(args.out_dir))


def cath_mcnemar(out: Path) -> None:
    """Paired McNemar over every cath_per_query_*.npz in `out`, with an alignment guard.

    A pair is only meaningful if both arms scored the same queries in the same order, so the
    label vectors are compared before any test runs.
    """
    arms = {p.stem.replace("cath_per_query_", ""): np.load(p, allow_pickle=True)
            for p in sorted(out.glob("cath_per_query_*.npz"))}
    if not arms:
        logger.warning("no cath_per_query_*.npz in %s", out)
        return
    labelled = {n: z["labels"] for n, z in arms.items() if "labels" in z}
    ref_name, ref = next(iter(labelled.items()), (None, None))
    for n, lab in labelled.items():
        if not np.array_equal(ref, lab):
            raise SystemExit(f"ABORT: {n} query order differs from {ref_name} -- pairing invalid")
    if len(labelled) < len(arms):
        logger.warning("%d arm(s) carry no labels vector; alignment unverified for those",
                       len(arms) - len(labelled))
    logger.info("%d arms, query alignment verified", len(arms))

    for n, z in arms.items():
        c = z["correct"].astype(bool)
        logger.info("  %-24s %3d/%d  %5.2f%%", n, c.sum(), len(c), 100 * c.mean())
    print(f"{'pair':52s} {'b':>3s} {'c':>3s} {'disc':>5s} {'chi2':>6s} {'p':>7s}")
    for a, b_ in itertools.combinations(arms, 2):
        x, y = arms[a]["correct"].astype(bool), arms[b_]["correct"].astype(bool)
        b = int((x & ~y).sum()); c = int((~x & y).sum())
        if b + c == 0:
            print(f"{a + ' vs ' + b_:52s} identical")
            continue
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        pval = erfc(sqrt(chi / 2))
        print(f"{a + ' vs ' + b_:52s} {b:3d} {c:3d} {b + c:5d} {chi:6.2f} {pval:7.3f}"
              f"{' *' if pval < 0.05 else ''}")


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
    common.add_argument("--chunk_elements", type=int, default=50_000_000,
                        help="MaxSim scoring budget; caps the score tensor built at once")

    ps = sub.add_parser("scope", parents=[common])
    ps.add_argument("--n_boot", type=int, default=1000)
    ps.add_argument("--reference", default=None, help="Model NAME for paired bootstrap deltas")
    ps.add_argument("--checkpoints", default=None, help="Training run dir: evaluate step0 + checkpoint-* + final")
    ps.set_defaults(fn=cmd_scope)

    pf = sub.add_parser("fewshot_rh", parents=[common])
    pf.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000])
    pf.add_argument("--knn_k", type=int, default=3)
    pf.add_argument("--max_test", type=int, default=0, help="Subsample the test split (0 = all)")
    pf.set_defaults(fn=cmd_fewshot_rh)

    pc = sub.add_parser("watch_curve", parents=[common])
    pc.add_argument("--run_dir", required=True)
    pc.add_argument("--name", required=True, help="Curve label, e.g. protsent_late_150m")
    pc.add_argument("--poll_seconds", type=int, default=120)
    pc.add_argument("--max_hours", type=float, default=24.0)
    pc.add_argument("--n_boot", type=int, default=1000)
    pc.add_argument("--follow_pid", type=int, default=0,
                    help="Exit when this PID (the trainer) is gone, so a dead run cannot hold the GPU")
    pc.set_defaults(fn=cmd_watch_curve)

    pg = sub.add_parser("proteingym", parents=[common])
    pg.add_argument("--variant", default="dms_substitutions", choices=sorted(PROTEINGYM_VARIANTS))
    pg.add_argument("--max_assays", type=int, default=0, help="0 = all")
    pg.add_argument("--max_variants_per_assay", type=int, default=2000,
                    help="Seeded subsample per assay; Spearman is stable well below the full set")
    pg.add_argument("--n_boot", type=int, default=1000)
    pg.set_defaults(fn=cmd_proteingym)

    pcath = sub.add_parser("cath", parents=[common])
    pcath.add_argument("--test_split", default="test_h", choices=["test_h", "test219", "test300", "validation"])
    pcath.add_argument("--label_col", default="cath_h")
    pcath.add_argument("--max_lookup", type=int, default=0, help="Subsample the lookup set (0 = all 69,605)")
    pcath.add_argument("--n_boot", type=int, default=1000)
    pcath.add_argument("--mcnemar", action="store_true",
                       help="After scoring, run paired McNemar over every cath_per_query_*.npz in --out_dir")
    pcath.set_defaults(fn=cmd_cath)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
