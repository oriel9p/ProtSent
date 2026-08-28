"""IR metrics for MaxSim retrieval, computed by sentence-transformers rather than by hand.

SCOPe is scored all-vs-all, so every query is also a corpus document. ST's IR evaluator
deliberately does NOT skip self-matches (see the NOTE in its information_retrieval.py: it would be
"unexpected behaviour if the user just uses sets of integers from 0"), which means feeding it a
raw all-vs-all corpus puts each query at rank 1 against itself and shifts every metric by one rank.
li.ranking_from_similarity already drops self, so the adapter feeds ST that ranking and gets
nDCG / MRR / Precision@k / Recall@k / MAP@k without reimplementing any of them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ir_metrics import ir_metrics_from_ranking


def test_perfect_ranking_scores_one():
    # 4 items, two labels; each query's two same-label partners rank first.
    labels = np.array(["a", "a", "b", "b"])
    ranking = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])
    m = ir_metrics_from_ranking(ranking, labels, k_values=(1,))
    assert m["accuracy@1"] == 1.0
    assert m["recall@1"] == 1.0  # one relevant partner each, retrieved at rank 1
    assert m["ndcg@1"] == 1.0


def test_worst_ranking_scores_zero_at_one():
    labels = np.array(["a", "a", "b", "b"])
    ranking = np.array([[2, 3, 1], [3, 2, 0], [0, 1, 3], [1, 0, 2]])
    m = ir_metrics_from_ranking(ranking, labels, k_values=(1,))
    assert m["accuracy@1"] == 0.0
    assert m["mrr@1"] == 0.0


def test_the_real_ranking_helper_excludes_self():
    """The adapter is only correct because li.ranking_from_similarity drops the diagonal.

    ST does not skip self-matches, so if that ever changed, every metric here would silently gain a
    free rank-1 hit. Exercise the real helper rather than a hand-written fixture.
    """
    import late_interaction as li

    sim = np.eye(4) + 0.1  # self-similarity is maximal, so self would rank 1st if kept
    ranking = li.ranking_from_similarity(sim)
    assert ranking.shape == (4, 3)
    for q, row in enumerate(ranking):
        assert q not in row, f"query {q} appears in its own ranking"


def test_matches_hand_computed_recall_at_2():
    # query 0 has partners {1,2}; ranking puts one of them at rank 2.
    labels = np.array(["a", "a", "a", "b"])
    ranking = np.array([[3, 1, 2], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    m = ir_metrics_from_ranking(ranking, labels, k_values=(2,))
    # q0: partners {1,2}, top-2 = [3,1] -> 1 of 2 = 0.5
    # q1: partners {0,2}, top-2 = [0,2] -> 2 of 2 = 1.0
    # q2: partners {0,1}, top-2 = [0,1] -> 2 of 2 = 1.0
    # q3: no partners -> excluded from the mean by ST (empty relevant set)
    assert abs(m["recall@2"] - (0.5 + 1.0 + 1.0) / 3) < 1e-9
