"""Unit tests for custom attention pooling modules."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention_pooling import (
    ContextualAttentionPooling,
    GeneralizedMeanPooling,
    LinearAttentionPooling,
)


@pytest.fixture()
def sample_features() -> dict[str, torch.Tensor]:
    """Return a deterministic token batch with masked padding positions."""
    token_embeddings = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [100.0, 100.0, 100.0],
            ],
            [
                [2.0, 2.0, 2.0],
                [8.0, 8.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.int64)
    return {
        "token_embeddings": token_embeddings.clone(),
        "attention_mask": attention_mask.clone(),
    }


@pytest.mark.parametrize(
    ("pooling_cls", "kwargs"),
    [
        (LinearAttentionPooling, {}),
        (ContextualAttentionPooling, {"activation": "tanh"}),
        (ContextualAttentionPooling, {"activation": "silu"}),
        (GeneralizedMeanPooling, {}),
    ],
)
def test_pooling_modules_emit_sentence_embeddings(
    sample_features: dict[str, torch.Tensor],
    pooling_cls: type[torch.nn.Module],
    kwargs: dict[str, str],
) -> None:
    """Each pooling module should emit finite sentence embeddings."""
    pooling_module = pooling_cls(3, **kwargs)
    output = pooling_module(sample_features)

    sentence_embedding = output["sentence_embedding"]
    assert sentence_embedding.shape == (2, 3)
    assert torch.isfinite(sentence_embedding).all()


def test_attention_pooling_ignores_masked_tokens(
    sample_features: dict[str, torch.Tensor],
) -> None:
    """Masked tokens should not contribute to attention pooling outputs."""
    pooling_module = LinearAttentionPooling(3)
    with torch.no_grad():
        pooling_module.attention.weight.fill_(1.0)

    output = pooling_module(sample_features)
    first_embedding = output["sentence_embedding"][0]
    expected = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(first_embedding, expected, atol=1e-3)


def test_gem_matches_mean_when_p_is_one(
    sample_features: dict[str, torch.Tensor],
) -> None:
    """GeM should reduce to masked mean pooling when p == 1."""
    pooling_module = GeneralizedMeanPooling(3, p=1.0)
    with torch.no_grad():
        pooling_module.p.fill_(1.0)

    output = pooling_module(sample_features)
    first_embedding = output["sentence_embedding"][0]
    expected = torch.tensor([2.5, 3.5, 4.5])
    assert torch.allclose(first_embedding, expected, atol=1e-5)


def test_contextual_attention_rejects_unknown_activation() -> None:
    """Unsupported activations should fail fast."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        ContextualAttentionPooling(3, activation="relu")


@pytest.mark.parametrize(
    ("pooling_cls", "kwargs"),
    [
        (LinearAttentionPooling, {}),
        (ContextualAttentionPooling, {"activation": "gelu"}),
        (GeneralizedMeanPooling, {"p": 2.0}),
    ],
)
def test_pooling_modules_save_and_load(
    pooling_cls: type[Any],
    kwargs: dict[str, float | str],
) -> None:
    """Custom pooling modules should round-trip through SentenceTransformer save hooks."""
    module = pooling_cls(3, **kwargs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        module.save(tmp_dir)
        reloaded = pooling_cls.load(tmp_dir)

        assert reloaded.get_sentence_embedding_dimension() == 3
        assert set(module.state_dict()) == set(reloaded.state_dict())
        for key, value in module.state_dict().items():
            assert torch.allclose(value, reloaded.state_dict()[key])
