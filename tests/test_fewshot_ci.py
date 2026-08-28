"""Few-shot remote-homology rows must carry an accuracy CI and per-query correctness.

Point accuracies with no interval invited exactly the mistake the SCOPe analysis already made:
comparing arms by eye and calling small gaps real. The per-query vector is what makes the paired
comparison possible later, and it is cheap -- one bit per test sequence.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_fewshot_rows_carry_an_accuracy_ci():
    src = (ROOT / "late_interaction_eval.py").read_text()
    body = src[src.index("def cmd_fewshot_rh"):src.index("PROTEINGYM_REF")]
    assert "accuracy_ci95" in body, "few-shot rows report a bare accuracy with no interval"
    assert "boot_ci" in body, "few-shot CI is not a bootstrap over the test queries"


def test_fewshot_saves_per_query_correctness():
    src = (ROOT / "late_interaction_eval.py").read_text()
    body = src[src.index("def cmd_fewshot_rh"):src.index("PROTEINGYM_REF")]
    assert "save_per_query" in body or "np.savez" in body, (
        "few-shot discards per-query correctness, so arms can never be compared paired"
    )
