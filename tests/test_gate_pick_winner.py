"""The gate makes an unattended multi-hour decision from a CSV, so its edge cases are pinned here.

The CSV is append-only and arms can be rescored, so duplicate rows for one model/step are normal.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VANILLA, PROTSENTV2 = "late-r2-esm2-150m", "late-r2-protsentv2-150m"
HEAD = "eligible_MAP,level,model\n"


def _pick(tmp_path, body: str) -> str:
    csv = tmp_path / "curve.csv"
    csv.write_text(HEAD + body)
    r = subprocess.run([sys.executable, str(ROOT / "gate_pick_winner.py"), str(csv)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_higher_score_at_the_shared_mark_wins(tmp_path):
    assert _pick(tmp_path, f"0.80,superfamily,{VANILLA}@4000\n0.70,superfamily,{PROTSENTV2}@4000\n") == VANILLA
    assert _pick(tmp_path, f"0.60,superfamily,{VANILLA}@4000\n0.70,superfamily,{PROTSENTV2}@4000\n") == PROTSENTV2


def test_a_missing_arm_falls_back_to_the_parsimonious_prior(tmp_path):
    """An absent measurement must never be what promotes the vanilla arm: every 35M comparison in
    this campaign favours the ProtSent-V2 base, so silence defaults to it."""
    assert _pick(tmp_path, f"0.99,superfamily,{VANILLA}@4000\n") == PROTSENTV2
    assert _pick(tmp_path, "") == PROTSENTV2


def test_a_tie_goes_to_the_prior(tmp_path):
    assert _pick(tmp_path, f"0.70,superfamily,{VANILLA}@4000\n0.70,superfamily,{PROTSENTV2}@4000\n") == PROTSENTV2


def test_a_rescored_arm_uses_its_latest_row(tmp_path):
    """Append-only CSV: a rescore adds a row rather than replacing one, so the last must win."""
    body = (f"0.10,superfamily,{VANILLA}@4000\n0.70,superfamily,{PROTSENTV2}@4000\n"
            f"0.90,superfamily,{VANILLA}@4000\n")
    assert _pick(tmp_path, body) == VANILLA


def test_only_the_shared_mark_counts(tmp_path):
    """@final and other steps must not decide a gate defined at 4000."""
    body = (f"0.99,superfamily,{VANILLA}@final\n0.99,superfamily,{VANILLA}@1000\n"
            f"0.70,superfamily,{PROTSENTV2}@4000\n")
    assert _pick(tmp_path, body) == PROTSENTV2


def test_the_level_matters(tmp_path):
    """fold/family rows must not be read as superfamily ones."""
    body = (f"0.99,fold,{VANILLA}@4000\n0.70,superfamily,{PROTSENTV2}@4000\n")
    assert _pick(tmp_path, body) == PROTSENTV2
