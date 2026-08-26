"""Two-stage rerank must report every SCOPe level, not just family.

The rerank table was family-only, which is the level where a pooled shortlist is easiest --
same-family proteins are near-duplicates. The interesting claim is at superfamily and fold, where
the shortlist has to generalise, so the analysis is only usable if it covers all three.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_rerank_rows_cover_every_scope_level():
    src = (ROOT / "analyze_maxsim_cost.py").read_text()
    # The rows must carry a level column; a single hardcoded `fam` cannot.
    assert '"level"' in src, "rerank rows do not record which SCOPe level they are for"
    assert "scope_labels" in src, "rerank does not derive per-level labels via li.scope_labels"


def test_rerank_head_reordering_is_confined_to_the_shortlist():
    """rerank_from must touch only the first k columns and permute, never drop, them."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from analyze_maxsim_cost import rerank_from

    shortlist = np.array([[3, 1, 2, 0], [0, 2, 1, 3]])
    maxsim = np.array(
        [[0.0, 0.1, 0.9, 0.2], [0.5, 0.4, 0.3, 0.2]], dtype=float
    )
    out = rerank_from(shortlist, maxsim, k=2)
    # tail untouched
    assert out[0, 2:].tolist() == [2, 0]
    assert out[1, 2:].tolist() == [1, 3]
    # head is a permutation of the original head
    assert sorted(out[0, :2].tolist()) == sorted(shortlist[0, :2].tolist())
    assert sorted(out[1, :2].tolist()) == sorted(shortlist[1, :2].tolist())
