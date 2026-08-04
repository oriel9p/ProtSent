"""Batch-sampler and MNRL-direction resolution.

Both defaults were wrong for every run to date, and the fix for one of them
nearly introduced a crash on a path it was not meant to touch, so the mapping
gets a test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_pipeline import (  # noqa: E402
    MNRL_DIRECTIONS_DEFAULT,
    BatchSamplers,
    _mnrl_directions,
    _resolve_batch_sampler,
    _resolve_primary_loss,
)

# Take the enum from the module under test, not a second copy of a
# sentence-transformers-internal import path: if that path moves, the module's own
# guarded import goes None and _resolve_batch_sampler silently returns None, which
# a separately-imported enum would hide behind an unrelated skip.
pytestmark = pytest.mark.skipif(
    BatchSamplers is None, reason="BatchSamplers unavailable in this sentence-transformers"
)


@pytest.mark.parametrize(
    ("effective_loss", "expected"),
    [
        # The bug this fixes lived upstream of here: run_training passed the raw
        # loss_mode, so "multi" matched nothing and every multi-dataset run trained
        # without the sampler the CachedMNRL docs pair with it. This helper now only
        # ever sees an already-resolved loss.
        ("cached_mnrl", "NO_DUPLICATES"),
        ("triplet", "GROUP_BY_LABEL"),
        # A triplet primary under multi-task is masked to "" by the caller, because
        # GroupByLabelBatchSampler raises on a dataset with no label column and a
        # multi-task dict mixes labelled and unlabelled datasets.
        ("", None),
        ("simcse", None),
    ],
)
def test_batch_sampler_resolution(effective_loss, expected) -> None:
    got = _resolve_batch_sampler("auto", effective_loss)
    assert got is (None if expected is None else getattr(BatchSamplers, expected))


def test_explicit_batch_sampler_overrides_auto() -> None:
    assert _resolve_batch_sampler("none", "cached_mnrl") is None


def test_primary_loss_resolution() -> None:
    assert _resolve_primary_loss(SimpleNamespace(multi_primary_loss="triplet")) == "triplet"
    # "auto" defers to the legacy flag.
    assert (
        _resolve_primary_loss(
            SimpleNamespace(multi_primary_loss="auto", multi_mnrl_loss="cached_mnrl")
        )
        == "cached_mnrl"
    )


def test_directions_default_is_symmetric_everywhere() -> None:
    """A programmatic caller must get the same loss as the CLI."""
    assert _mnrl_directions(SimpleNamespace()) == MNRL_DIRECTIONS_DEFAULT
    assert _mnrl_directions(SimpleNamespace(mnrl_directions=["query_to_doc"])) == (
        "query_to_doc",
    )
