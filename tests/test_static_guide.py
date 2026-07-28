"""Tests for StaticEmbedding guide helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from static_guide import (  # noqa: E402
    load_static_guide_model,
    patch_static_embedding_for_gist,
    save_static_embedding_model,
)


def _build_toy_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "M": 1,
                "K": 2,
                "T": 3,
                "A": 4,
                "Y": 5,
                "I": 6,
                "Q": 7,
                "R": 8,
                "L": 9,
                "G": 10,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


def _build_toy_model() -> SentenceTransformer:
    tokenizer = _build_toy_tokenizer()
    embedding_weights = torch.arange(44, dtype=torch.float32).reshape(11, 4)
    static_embedding = StaticEmbedding(tokenizer, embedding_weights=embedding_weights)
    return SentenceTransformer(modules=[static_embedding], device="cpu")


def test_patch_static_embedding_for_gist_preserves_encode() -> None:
    model = _build_toy_model()

    patch_static_embedding_for_gist(model)

    assert isinstance(model.tokenizer, PreTrainedTokenizerBase)
    embeddings = model.encode(
        ["M K T", "A Y I"],
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    assert embeddings.shape == (2, 4)


def test_load_static_guide_model_round_trip(tmp_path: Path) -> None:
    model = _build_toy_model()
    patch_static_embedding_for_gist(model)

    sequences = ["M K T", "A Y I", "Q R L G"]
    expected_embeddings = model.encode(
        sequences,
        batch_size=len(sequences),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    save_path = save_static_embedding_model(model, str(tmp_path / "toy_static_guide"))
    reloaded_model = load_static_guide_model(save_path)
    actual_embeddings = reloaded_model.encode(
        sequences,
        batch_size=len(sequences),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    np.testing.assert_allclose(
        actual_embeddings, expected_embeddings, rtol=1e-6, atol=1e-6
    )
