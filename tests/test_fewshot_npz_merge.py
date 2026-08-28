"""A second few-shot invocation must not delete the first one's per-query vectors.

Rows are appended to late_fewshot_knn.csv, so arms accumulate there across runs. The .npz was being
rewritten from scratch, so after campaign/fill_fewshot.sh added one arm the file held that arm alone
while the CSV listed ten -- and the paired analysis those vectors exist for silently had nothing left
to pair.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_fewshot_merges_rather_than_overwrites_per_query():
    src = (ROOT / "late_interaction_eval.py").read_text()
    body = src[src.index("def cmd_fewshot_rh"):src.index("PROTEINGYM_REF")]
    assert "merged.update(per_query)" in body, "per-query npz is not merged with the existing file"
    assert "if npz.exists()" in body, "existing per-query npz is not read before writing"
