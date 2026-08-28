"""ProteinGym rows must record WHICH weights produced them, not just a model name.

A Hub repo can be retrained and re-pushed under the same name. ProteinGym is the one benchmark where
that silently changes the meaning of an existing row: ProtSent-V2 excludes dms_cosent.parquet and is
safe to score here, while V2.5 and ESM-C-300M-V2 train on it and are contaminated. If a same-name
push swaps one for the other, every stored row becomes unverifiable unless it pins a revision.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from late_interaction_eval import resolve_model_revision


def test_local_paths_have_no_revision():
    assert resolve_model_revision("/some/local/snapshot/step-10000") == ""


def test_hub_ids_resolve_to_a_commit_sha():
    sha = resolve_model_revision("GrimSqueaker/ProtSent-V2-35M")
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), sha


def test_unreachable_hub_id_degrades_to_empty_not_an_exception():
    assert resolve_model_revision("definitely/not-a-real-model-xyzzy-42") == ""
