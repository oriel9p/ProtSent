"""nDCG / MRR / Precision@k / Recall@k / MAP@k for MaxSim retrieval, computed by sentence-transformers.

`scope_rows` reports MAP and Recall@k. The rest of the standard IR panel is missing, and rather than
reimplement it, this hands an already-ranked list to ST's own
`InformationRetrievalEvaluator.compute_metrics`, which is separable from its search.

Why not use the evaluator end-to-end: SCOPe is all-vs-all, so each query is also a corpus document,
and ST deliberately does not skip self-matches (its information_retrieval.py says doing so "might be
unexpected behaviour if the user just uses sets of integers from 0"). Every query would rank itself
first and shift every metric by a rank. `li.ranking_from_similarity` already drops self, so the
search stays ours -- validated, and the same ranking every other number in results/ came from -- and
only the metric arithmetic is delegated.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

DEFAULT_K = (1, 5, 10, 30)

# NAME COLLISION, read before joining these columns with scope_hierarchy.csv. The `Recall@k` in
# scope_rows is hit-at-least-one-in-top-k, i.e. what IR calls accuracy@k -- and it is the definition
# the ProtSent paper uses ("the fraction of queries for which a protein from the same structural
# superfamily appears among the top-K"). The `recall@k` returned here is the true IR recall,
# retrieved-relevant over total-relevant. They are different estimators that happen to share a name;
# use `accuracy@k` below when comparing against scope_hierarchy.csv or the paper's Table 3.


def ir_metrics_from_ranking(ranking: np.ndarray, labels, *, k_values=DEFAULT_K) -> dict[str, float]:
    """Metrics for one (n, n-1) self-excluded ranking against same-label relevance.

    `ranking[q]` is q's corpus indices best-first. Relevance is "same label as q", which is how
    every SCOPe/CATH level is defined once `li.scope_labels` has truncated to that level.
    """
    labels = np.asarray(labels)
    ks = sorted(set(int(k) for k in k_values))
    qids = [str(i) for i in range(len(ranking))]
    relevant = {
        str(q): {str(d) for d in np.flatnonzero(labels == labels[q]) if d != q}
        for q in range(len(ranking))
    }
    ev = InformationRetrievalEvaluator(
        queries={q: "" for q in qids},
        corpus={str(i): "" for i in range(len(labels))},
        relevant_docs=relevant,
        accuracy_at_k=ks, precision_recall_at_k=ks, mrr_at_k=ks, ndcg_at_k=ks, map_at_k=ks,
    )
    # The evaluator drops queries whose relevant set is empty -- a singleton label has nothing to
    # retrieve -- so results must be indexed by ev.queries_ids, not by the original row order.
    # This is the same "eligible query" rule scope_rows applies, arrived at independently.
    # ST scores descending, so a strictly decreasing stand-in keeps the order we pass in.
    results = [
        [{"corpus_id": str(d), "score": float(-rank)}
         for rank, d in enumerate(ranking[int(qid)])]
        for qid in ev.queries_ids
    ]
    m = ev.compute_metrics(results)
    out = {}
    for k in ks:
        out[f"accuracy@{k}"] = m["accuracy@k"][k]
        out[f"precision@{k}"] = m["precision@k"][k]
        out[f"recall@{k}"] = m["recall@k"][k]
        out[f"mrr@{k}"] = m["mrr@k"][k]
        out[f"ndcg@{k}"] = m["ndcg@k"][k]
        out[f"map@{k}"] = m["map@k"][k]
    return out
