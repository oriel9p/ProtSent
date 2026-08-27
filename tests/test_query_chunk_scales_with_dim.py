"""The MaxSim query chunk must shrink as the embedding dimension grows.

maxsim_against_one holds one chunk of query embeddings resident. That is chunk x L x dim floats, so
a fixed chunk means an unprojected 640-d arm holds 5x what a 128-d projected arm does. In practice
the ProteinGym zero-shot arms asked for a single 26.6 GiB allocation and OOMed on an 80 GB card even
running alone, while the 128-d trained arms on the same data were fine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from late_interaction import query_chunk_for_dim, _QUERY_CHUNK


def test_projected_dim_keeps_the_full_chunk():
    assert query_chunk_for_dim(128) == _QUERY_CHUNK


def test_unprojected_dims_shrink_roughly_inversely():
    c640 = query_chunk_for_dim(640)
    c480 = query_chunk_for_dim(480)
    assert c640 < c480 < _QUERY_CHUNK
    # residency is chunk*dim; hold it near the 128-d budget rather than 5x over it
    assert c640 * 640 <= _QUERY_CHUNK * 128 * 1.05


def test_never_returns_a_degenerate_chunk():
    assert query_chunk_for_dim(100_000) >= 1
