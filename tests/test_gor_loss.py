"""Tests for the opt-in GOR loss wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gor_loss import LossWithGOR


class _ConstantLoss(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, sentence_features, labels=None):
        return torch.tensor(self.value)


def test_loss_with_gor_adds_weighted_regularizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sentence_transformers.sentence_transformer import losses as st_losses

    class _FakeGOR(nn.Module):
        def __init__(self, model):
            super().__init__()

        def forward(self, sentence_features, labels=None):
            return torch.tensor(2.0)

    monkeypatch.setattr(
        st_losses, "GlobalOrthogonalRegularizationLoss", _FakeGOR, raising=False
    )

    loss = LossWithGOR(object(), _ConstantLoss(3.0), gor_weight=0.25)

    assert loss([], None).item() == pytest.approx(3.5)


def test_loss_with_zero_weight_does_not_require_gor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sentence_transformers.sentence_transformer import losses as st_losses

    monkeypatch.delattr(st_losses, "GlobalOrthogonalRegularizationLoss", raising=False)

    loss = LossWithGOR(object(), _ConstantLoss(3.0), gor_weight=0.0)

    assert loss([], None).item() == pytest.approx(3.0)
