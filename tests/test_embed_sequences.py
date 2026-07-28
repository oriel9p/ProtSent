"""Tests for embed_sequences deduplication and pair reassembly.

These are offline unit tests using a mock model — no GPU required.

Run:
    pytest tests/test_embed_sequences.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_benchmark_suite import embed_sequences


# ---------------------------------------------------------------------------
# Helpers: create mock model objects
# ---------------------------------------------------------------------------

EMB_DIM = 8


def _make_sbert_mock(emb_dim: int = EMB_DIM):
    """Return a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()

    def _encode(seqs, **kwargs):
        return np.array([_seq_to_emb(s, emb_dim) for s in seqs])

    model.encode = _encode
    return model


def _make_embed_dataset_mock(emb_dim: int = EMB_DIM):
    """Return a (tokenizer, model) mock where model has embed_dataset (ESMplusplus)."""
    tokenizer = MagicMock()
    model = MagicMock()

    def _embed_dataset(sequences, **kwargs):
        max_len = kwargs.get("max_len", 512)
        return {s[:max_len]: torch.tensor(_seq_to_emb(s, emb_dim)) for s in sequences}

    model.embed_dataset = _embed_dataset
    model.tokenizer = tokenizer
    return (tokenizer, model)


def _seq_to_emb(seq: str, dim: int) -> list:
    """Deterministic embedding from sequence string (reproducible via hash)."""
    rng = np.random.RandomState(hash(seq) % (2**31))
    return rng.randn(dim).tolist()


# ---------------------------------------------------------------------------
# Tests: single sequences — no duplicates
# ---------------------------------------------------------------------------


class TestSingleSequences:
    seqs = ["ACDEF", "GHIKL", "MNPQR"]

    def test_sbert_shape(self):
        model = _make_sbert_mock()
        embs = embed_sequences(model, True, self.seqs, "cpu")
        assert embs.shape == (3, EMB_DIM)

    def test_embed_dataset_shape(self):
        model = _make_embed_dataset_mock()
        embs = embed_sequences(model, False, self.seqs, "cpu")
        assert embs.shape == (3, EMB_DIM)


# ---------------------------------------------------------------------------
# Tests: single sequences with duplicates
# ---------------------------------------------------------------------------


class TestDuplicateSequences:
    """When input has duplicate sequences, output must match input length."""

    seqs = ["ACDEF", "GHIKL", "ACDEF", "MNPQR", "GHIKL", "GHIKL"]

    def test_sbert_output_length_matches_input(self):
        model = _make_sbert_mock()
        embs = embed_sequences(model, True, self.seqs, "cpu")
        assert embs.shape[0] == len(self.seqs), (
            f"Expected {len(self.seqs)} rows, got {embs.shape[0]}"
        )

    def test_embed_dataset_output_length_matches_input(self):
        model = _make_embed_dataset_mock()
        embs = embed_sequences(model, False, self.seqs, "cpu")
        assert embs.shape[0] == len(self.seqs), (
            f"Expected {len(self.seqs)} rows, got {embs.shape[0]}"
        )

    def test_duplicate_rows_have_same_embedding(self):
        model = _make_embed_dataset_mock()
        embs = embed_sequences(model, False, self.seqs, "cpu")
        # "ACDEF" appears at index 0 and 2
        np.testing.assert_array_equal(embs[0], embs[2])
        # "GHIKL" appears at index 1, 4, 5
        np.testing.assert_array_equal(embs[1], embs[4])
        np.testing.assert_array_equal(embs[1], embs[5])


# ---------------------------------------------------------------------------
# Tests: pair sequences (PPI)
# ---------------------------------------------------------------------------


class TestPairSequences:
    """Pair inputs must return concatenated embeddings with shape (n_pairs, 2*dim)."""

    pairs = [("ACDEF", "GHIKL"), ("MNPQR", "ACDEF"), ("GHIKL", "MNPQR")]

    def test_sbert_pair_shape(self):
        model = _make_sbert_mock()
        embs = embed_sequences(model, True, self.pairs, "cpu")
        assert embs.shape == (3, 2 * EMB_DIM), (
            f"Expected (3, {2 * EMB_DIM}), got {embs.shape}"
        )

    def test_embed_dataset_pair_shape(self):
        model = _make_embed_dataset_mock()
        embs = embed_sequences(model, False, self.pairs, "cpu")
        assert embs.shape == (3, 2 * EMB_DIM), (
            f"Expected (3, {2 * EMB_DIM}), got {embs.shape}"
        )

    def test_pair_is_concat_of_individual_embeddings(self):
        model = _make_embed_dataset_mock()
        embs = embed_sequences(model, False, self.pairs, "cpu")
        # First pair: ("ACDEF", "GHIKL")
        emb_a = np.array(_seq_to_emb("ACDEF", EMB_DIM))
        emb_b = np.array(_seq_to_emb("GHIKL", EMB_DIM))
        expected = np.concatenate([emb_a, emb_b])
        np.testing.assert_allclose(embs[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: empty input
# ---------------------------------------------------------------------------


def test_empty_input():
    model = _make_sbert_mock()
    embs = embed_sequences(model, True, [], "cpu")
    assert embs.shape == (0,)


def test_sbert_path_respects_requested_max_length() -> None:
    """SentenceTransformer models should use the same truncation limit as HF models."""
    model = _make_sbert_mock()
    model.max_seq_length = 4096

    _ = embed_sequences(model, True, ["ACDEFGHIKLMNPQRSTVWY"], "cpu", max_length=123)

    assert model.max_seq_length == 123
