"""Tests for the opt-in GOR loss wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gor_loss import LossWithGOR, SubsampledLoss


class _ConstantLoss(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, sentence_features, labels=None):
        return torch.tensor(self.value)


class _FakeModel:
    """Stand-in for a SentenceTransformer: embeds one row per input row."""

    def __call__(self, features):
        return {"sentence_embedding": features["input_ids"].float()}


def _install_fake_gor(monkeypatch: pytest.MonkeyPatch, seen: list, built: list = None):
    from sentence_transformers.sentence_transformer import losses as st_losses

    class _FakeGOR(nn.Module):
        def __init__(self, model, **term_weights):
            super().__init__()
            if built is not None:
                built.append(term_weights)

        def forward(self, sentence_features, labels=None):  # replaced by the wrapper
            raise AssertionError("patched_forward should have replaced this")

        def compute_loss_from_embeddings(self, embeddings, labels=None):
            seen.append([e.shape[0] for e in embeddings])
            return {"gor_mean": torch.tensor(1.5), "gor_second_moment": torch.tensor(0.5)}

    monkeypatch.setattr(
        st_losses, "GlobalOrthogonalRegularizationLoss", _FakeGOR, raising=False
    )


def _features(batch_size: int, dim: int = 4):
    return [{"input_ids": torch.arange(batch_size * dim).reshape(batch_size, dim)}]


def test_loss_with_gor_adds_weighted_regularizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list = []
    _install_fake_gor(monkeypatch, seen)

    loss = LossWithGOR(_FakeModel(), _ConstantLoss(3.0), gor_weight=0.25)

    # 3.0 + 0.25 * (1.5 + 0.5)
    assert loss(_features(8), None).item() == pytest.approx(3.5)


def test_gor_term_weights_reach_the_regularizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingGemma drops the mean term entirely, so it must be reachable
    rather than frozen at the sentence-transformers default of 1.0."""
    seen: list = []
    built: list = []
    _install_fake_gor(monkeypatch, seen, built)

    LossWithGOR(_FakeModel(), _ConstantLoss(0.0), gor_weight=1.0, mean_weight=0.0)
    assert built == [{"mean_weight": 0.0}]


def test_gor_subsamples_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is what keeps peak memory bounded; without it GOR embeds the
    whole 1024-pair batch with grad enabled and OOMs a B300."""
    seen: list = []
    _install_fake_gor(monkeypatch, seen)

    loss = LossWithGOR(
        _FakeModel(), _ConstantLoss(0.0), gor_weight=1.0, mini_batch_size=16, max_samples=32
    )
    loss(_features(256), None)
    assert seen == [[32]]

    seen.clear()
    loss(_features(20), None)  # batch smaller than the cap
    assert seen == [[20]]


def test_subsampled_loss_caps_rows_and_labels() -> None:
    """CoSENT backprops the whole batch, so this cap is what keeps a large
    contrastive batch and a DMS target on the same GPU."""
    seen: list = []

    class _Recorder(nn.Module):
        def forward(self, sentence_features, labels=None):
            seen.append(
                (
                    [f["input_ids"].shape[0] for f in sentence_features],
                    None if labels is None else labels.shape[0],
                )
            )
            return torch.tensor(1.0)

    loss = SubsampledLoss(_Recorder(), max_samples=4)
    assert loss(_features(16) * 2, torch.arange(16.0)).item() == pytest.approx(1.0)
    assert seen == [([4, 4], 4)]

    seen.clear()
    loss(_features(3), torch.arange(3.0))  # batch smaller than the cap
    assert seen == [([3], 3)]

    seen.clear()
    SubsampledLoss(_Recorder(), max_samples=0)(_features(16), None)  # cap disabled
    assert seen == [([16], None)]


def test_loss_with_zero_weight_does_not_require_gor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sentence_transformers.sentence_transformer import losses as st_losses

    monkeypatch.delattr(st_losses, "GlobalOrthogonalRegularizationLoss", raising=False)

    loss = LossWithGOR(object(), _ConstantLoss(3.0), gor_weight=0.0)

    assert loss([], None).item() == pytest.approx(3.0)
