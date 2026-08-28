"""MaxSim as the kNN metric on ProtBench's own task setups.

ProtBench's knn probe is KNeighborsClassifier(metric="euclidean") over pooled embeddings. Swapping
in MaxSim needs only the neighbour selection to change, so this reuses ProtBench's prepare_data and
classification_metrics and keeps just the vote local.

sklearn's metric="precomputed" was the obvious drop-in and is rejected on cost: its fit() wants a
square train x train distance matrix, which for fold_prediction is 12,312^2 = 151M MaxSim pairs
against the 40M the test x train matrix actually needs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxsim_knn_bench import knn_predict


def test_classification_votes_the_majority_of_k_neighbours():
    # 3 test rows, 4 gallery items. Higher score = nearer.
    sim = np.array([
        [9.0, 8.0, 1.0, 0.0],   # neighbours 0,1,2 -> labels a,a,b -> a
        [0.0, 1.0, 8.0, 9.0],   # neighbours 3,2,1 -> labels b,b,a -> b
        [9.0, 0.0, 0.5, 8.0],   # neighbours 0,3,2 -> labels a,b,b -> b
    ])
    y = np.array(["a", "a", "b", "b"])
    assert knn_predict(sim, y, k=3, regression=False).tolist() == ["a", "b", "b"]


def test_regression_averages_the_k_neighbours():
    sim = np.array([[9.0, 8.0, 7.0, 0.0]])
    y = np.array([1.0, 2.0, 3.0, 100.0])
    out = knn_predict(sim, y, k=3, regression=True)
    assert abs(out[0] - 2.0) < 1e-9  # mean(1,2,3), the 100 is the far neighbour


def test_k_larger_than_gallery_is_clipped_not_an_error():
    sim = np.array([[1.0, 2.0]])
    y = np.array(["a", "b"])
    assert knn_predict(sim, y, k=5, regression=False).tolist() == ["b"]  # tie -> nearest wins


def test_ties_break_toward_the_nearest_neighbour():
    """One vote each for two labels; the closer neighbour must win, as in fewshot_rh."""
    sim = np.array([[5.0, 4.0]])
    y = np.array(["a", "b"])
    assert knn_predict(sim, y, k=2, regression=False).tolist() == ["a"]
