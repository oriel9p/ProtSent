"""SCOPe rows must say which corpus produced them, not just which level.

Three SCOPe protocols are in circulation for this project: the paper's Section 9 describes a
100,000-protein set, results/RESULTS.md is family-level over tattabio/scope40_test (2,207 domains),
and the late_interaction tables are superfamily-level over that same 2,207. ESM-2 150M R@1 reads
0.423, 0.5535 and 0.7277 across them. Without a corpus column a rerun on a different gallery produces
a file that is indistinguishable from the existing ones except by n_queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from late_interaction import SCOPE_CORPUS, scope_rows


def test_rows_record_the_corpus():
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    fams = ["a.1.1.1", "a.1.1.2", "b.2.2.2"]
    rows, _ = scope_rows(sim, fams, model="m", scoring="maxsim", n_boot=10, seed=0, runtime_s=0.0)
    assert rows, "no rows produced"
    for r in rows:
        assert r["corpus"] == SCOPE_CORPUS, f"row missing corpus: {r.keys()}"


def test_corpus_constant_is_the_one_the_loader_uses():
    src = (Path(__file__).resolve().parent.parent / "late_interaction.py").read_text()
    loader = src[src.index("def load_scope40"):]
    loader = loader[:loader.index("\ndef ")] if "\ndef " in loader else loader
    assert "SCOPE_CORPUS" in loader, "load_scope40 does not use the SCOPE_CORPUS constant"
