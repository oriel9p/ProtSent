#!/usr/bin/env python
"""Separate the encode cost from the scoring cost, and price two cheaper alternatives.

The pilot's first timings measured ``encode + score`` together, which understates the
scoring ratio (a shared constant sits in both terms) and hides that pooled-cosine
"scoring" of a 2.2k corpus is a single GEMM. This times the two phases apart on one
GPU, then reuses the same matrices for two zero-extra-cost analyses:

* symmetrised MaxSim, ``(S + S.T) / 2``: does the operator's query/document asymmetry
  cost anything at evaluation time?
* two-stage retrieval: pooled-cosine shortlist of the top-k, MaxSim rerank inside it —
  the standard way late interaction is deployed at scale.

    uv run --no-sync python analyze_maxsim_cost.py --selfcheck
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import late_interaction as li
from late_interaction_eval import parse_model_spec  # noqa: E402
from bootstrap_ci import per_query_metrics  # noqa: E402

RERANK_K = (10, 30, 100)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maxsim_cost")


def metrics(ranking: np.ndarray, labels: np.ndarray) -> dict:
    pq = per_query_metrics(ranking, labels)
    elig = pq["eligible"]
    # Spelled eligible_* to match scope_hierarchy.csv: these are means over queries that
    # have at least one reachable positive, NOT plain Recall over all queries (which is
    # ~2 points lower and would join wrongly against the other tables on a bare "R@1").
    return {"eligible_Recall@1": float(pq["hit1"][elig].mean()),
            "eligible_Recall@10": float(pq["hit10"][elig].mean()),
            "eligible_MAP": float(pq["ap"][elig].mean()), "n_eligible": int(elig.sum())}


def rerank_from(shortlist: np.ndarray, maxsim: np.ndarray, k: int) -> np.ndarray:
    """Reorder the first k of a precomputed cosine ranking by MaxSim; keep the tail as is."""
    m = maxsim.copy()
    np.fill_diagonal(m, -np.inf)
    out = shortlist.copy()
    for q in range(len(shortlist)):
        head = shortlist[q, :k]
        out[q, :k] = head[np.argsort(-m[q, head], kind="stable")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", action="append", default=[], metavar="NAME=KIND:PATH")
    ap.add_argument("--rerank", action="append", default=[], metavar="MAXSIM_ARM=DENSE_ARM",
                    help="Two-stage pairing: score MAXSIM_ARM's rerank over DENSE_ARM's cosine "
                         "shortlist. Repeatable; both names must appear in --models.")
    ap.add_argument("--out_dir", default="results/late_interaction/pilot_35m")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--chunk_elements", type=int, default=50_000_000)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        cos = np.array([[9.0, 0.9, 0.1], [0.9, 9.0, 0.2], [0.1, 0.2, 9.0]])
        maxsim = np.array([[9.0, 0.1, 0.9], [0.1, 9.0, 0.2], [0.9, 0.2, 9.0]])
        # cosine alone ranks doc1 first for query0; MaxSim prefers doc2, and a k=2
        # rerank must surface that while leaving the tail alone.
        assert li.ranking_from_similarity(cos)[0].tolist() == [1, 2]
        assert rerank_from(li.ranking_from_similarity(cos), maxsim, 2)[0].tolist() == [2, 1]
        assert rerank_from(li.ranking_from_similarity(cos), maxsim, 1)[0].tolist() == [1, 2]
        # a positive per-row rescale leaves every row's ranking untouched
        lens = np.array([2.0, 5.0, 11.0])
        assert (li.ranking_from_similarity(maxsim / lens[:, None])
                == li.ranking_from_similarity(maxsim)).all()
        print("selfcheck ok")
        return 0

    pairing = dict(pair.split("=", 1) for pair in args.rerank)
    seqs, families = li.load_scope40()
    fam = np.asarray(families)
    cost_rows, rerank_rows, sym_rows = [], [], []
    cosine_cache: dict[str, np.ndarray] = {}

    for spec in args.models:
        name, kind, path = parse_model_spec(spec)
        if kind == "dense":
            _, model = li.build_multivector_encoder(path, proj_dim=0, max_seq_length=args.max_seq_length,
                                                    device=args.device)
            t0 = time.time()
            emb = model.encode(seqs, batch_size=args.batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
            encode_s = time.time() - t0
            t0 = time.time()
            sim = emb @ emb.T
            score_s = time.time() - t0
            dim, per_protein = emb.shape[1], emb.shape[1]
        else:
            proj = 0 if kind == "zeroshot" else None
            model = (li.build_multivector_encoder(path, proj_dim=0, max_seq_length=args.max_seq_length,
                                                  device=args.device)[0]
                     if proj == 0 else li.load_multivector_encoder(path, device=args.device))
            model.max_seq_length = args.max_seq_length
            t0 = time.time()
            emb = model.encode_document(seqs, batch_size=args.batch_size, show_progress_bar=False)
            torch.cuda.synchronize()
            encode_s = time.time() - t0
            t0 = time.time()
            with torch.no_grad():
                sim_t = model.similarity(emb, emb, chunk_elements=args.chunk_elements)
            torch.cuda.synchronize()
            score_s = time.time() - t0
            sim = sim_t.float().cpu().numpy()
            dim = emb[0].shape[1]
            per_protein = float(np.mean([e.shape[0] for e in emb])) * dim

        cost_rows.append({
            "arm": name, "scoring": "cosine" if kind == "dense" else "maxsim",
            "vector_dim": dim, "numbers_per_protein": round(float(per_protein), 1),
            "encode_s": round(encode_s, 2), "score_s": round(score_s, 3),
            "total_s": round(encode_s + score_s, 2), "n_proteins": len(seqs),
            "pairs_scored": len(seqs) ** 2,
        })
        print(f"{name}: encode {encode_s:.1f}s  score {score_s:.3f}s  ({per_protein:.0f} numbers/protein)")

        if kind == "dense":
            cosine_cache[name] = sim  # keyed by the exact --models name that --rerank refers to
        else:
            base = metrics(li.ranking_from_similarity(sim), fam)
            sym = metrics(li.ranking_from_similarity((sim + sim.T) / 2), fam)
            # Raw MaxSim sums over query residues, so a row's scale is the query's
            # length: averaging with the transpose mixes in a term that scales with
            # the DOCUMENT's length and biases the ranking toward long documents.
            # MeanMaxSim (divide each row by its query length) is scale-free, so its
            # symmetrisation is the meaningful one.
            lens = np.array([min(len(x), args.max_seq_length - 2) for x in seqs], dtype=float)
            mean_sim = sim / lens[:, None]
            # MeanMaxSim on its own is not reported: rescaling a row by a positive constant
            # cannot reorder that row, so every ranking metric is identical to MaxSim's.
            mean_sym = metrics(li.ranking_from_similarity((mean_sim + mean_sim.T) / 2), fam)
            sym_rows.append({"arm": name,
                             **{f"{k}_maxsim": v for k, v in base.items()},
                             **{f"{k}_symmetrised_raw": v for k, v in sym.items()},
                             **{f"{k}_symmetrised_meanmaxsim": v for k, v in mean_sym.items()}})
            partner = pairing.get(name)
            if partner is not None:
                cos = cosine_cache.get(partner)
                if cos is None:
                    raise SystemExit(
                        f"--rerank {name}={partner} but {partner} produced no cosine matrix; "
                        "list it in --models as a dense arm before this one"
                    )
                shortlist = li.ranking_from_similarity(cos)
                for k in RERANK_K:
                    rerank_rows.append({"arm": name, "shortlist_from": partner, "shortlist_k": k,
                                        **metrics(rerank_from(shortlist, sim, k), fam)})
                rerank_rows.append({"arm": name, "shortlist_from": partner,
                                    "shortlist_k": "full", **base})
            else:
                logger.info("no --rerank pairing for %s; skipping its two-stage rows", name)
        del model, emb, sim, sim_t
        torch.cuda.empty_cache()  # free before the next arm's encode is timed

    out = Path(args.out_dir)
    for rows, fname in ((cost_rows, "training/scoring_cost.csv"),
                        (sym_rows, "scope/scope_symmetrised.csv"),
                        (rerank_rows, "scope/scope_two_stage_rerank.csv")):
        if not rows:
            continue
        path = out / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
