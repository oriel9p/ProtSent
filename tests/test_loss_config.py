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
    _mnrl_directions,
    _resolve_batch_sampler,
    _resolve_primary_loss,
)

BatchSamplers = pytest.importorskip(
    "sentence_transformers.sentence_transformer.training_args"
).BatchSamplers


@pytest.mark.parametrize(
    ("loss_mode", "primary", "expected"),
    [
        # The bug: "multi" is a container, so it fell through to None and every
        # multi-dataset run trained without the sampler CachedMNRL wants.
        ("multi", "cached_mnrl", BatchSamplers.NO_DUPLICATES),
        ("multi", "mnrl", BatchSamplers.NO_DUPLICATES),
        ("multi", "cached_gist", BatchSamplers.NO_DUPLICATES),
        # ...but a triplet primary must keep its historical None. A multi-task
        # dict mixes labelled and unlabelled datasets and GroupByLabelBatchSampler
        # raises ValueError on a dataset with no label column.
        ("multi", "triplet", None),
        # Single-loss modes are unchanged.
        ("cached_mnrl", "", BatchSamplers.NO_DUPLICATES),
        ("triplet", "", BatchSamplers.GROUP_BY_LABEL),
        ("simcse", "", None),
    ],
)
def test_batch_sampler_resolution(loss_mode, primary, expected) -> None:
    assert _resolve_batch_sampler("auto", loss_mode, primary) is expected


def test_explicit_batch_sampler_overrides_auto() -> None:
    assert _resolve_batch_sampler("none", "multi", "cached_mnrl") is None
    assert (
        _resolve_batch_sampler("group_by_label", "multi", "cached_mnrl")
        is BatchSamplers.GROUP_BY_LABEL
    )


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
    assert _mnrl_directions(SimpleNamespace(mnrl_directions=None)) == MNRL_DIRECTIONS_DEFAULT
    assert _mnrl_directions(SimpleNamespace(mnrl_directions=["query_to_doc"])) == (
        "query_to_doc",
    )
