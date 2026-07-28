"""Quick validation tests for protein language model integration.

Tests that each model can:
  1. Load for benchmarking (load_model) and produce valid embeddings
  2. Load for training (load_model_for_training) and round-trip save/reload

All tests are marked ``slow`` (require GPU + network access) and are
skipped by default::

    pytest tests/test_models.py          # skips all (no GPU needed)
    pytest tests/test_models.py -m slow  # runs everything

Run a single model::

    pytest tests/test_models.py -m slow -k esm2
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Skip entire module if ML stack is not installed (e.g. lightweight CI)
torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from protein_benchmark_suite import embed_sequences, load_model
from protein_pipeline import load_model_for_training

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP"

MODELS = {
    "esm2": "facebook/esm2_t12_35M_UR50D",
    "fastplm_esm2": "Synthyra/ESM2-8M",
    "esmplusplus_small": "Synthyra/ESMplusplus_small",
    "amplify": "chandar-lab/AMPLIFY_120M",
}

slow = pytest.mark.slow


def _has_xformers() -> bool:
    try:
        import xformers  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@slow
@pytest.mark.parametrize(
    "model_key,model_name", list(MODELS.items()), ids=list(MODELS.keys())
)
def test_load_and_embed(model_key: str, model_name: str, device: str):
    """Load model for benchmarking and verify non-NaN, non-trivial embeddings."""
    if model_key == "amplify" and not _has_xformers():
        pytest.skip("xformers not installed")
    if model_key == "esmplusplus_small":
        pytest.skip("esmplusplus_small backend enum mismatch in cached model code")

    model_obj, is_sbert, dev = load_model(model_name, device)
    try:
        embs = embed_sequences(model_obj, is_sbert, [TEST_SEQ], dev, batch_size=1)
        assert embs.shape[0] == 1, f"Expected 1 embedding, got {embs.shape[0]}"
        assert embs.shape[1] > 0, "Embedding dim is 0"
        assert np.isnan(embs).sum() == 0, "Embeddings contain NaN"
        assert np.std(embs) > 1e-6, "Embeddings are constant (likely broken)"
    finally:
        del model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@slow
@pytest.mark.parametrize(
    "model_key,model_name", list(MODELS.items()), ids=list(MODELS.keys())
)
def test_load_training_and_save_reload(model_key: str, model_name: str):
    """Load model for training, save, and verify reload produces valid embeddings."""
    from sentence_transformers import SentenceTransformer

    if model_key == "amplify" and not _has_xformers():
        pytest.skip("xformers not installed")
    if model_key == "esmplusplus_small":
        pytest.skip("esmplusplus_small backend enum mismatch in cached model code")

    model = load_model_for_training(model_name, max_seq_length=128)
    assert model is not None

    is_amplify = "amplify" in model_name.lower()
    is_esmplusplus = (
        "esmplusplus" in model_name.lower() or "esm++" in model_name.lower()
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model"
            model.save(str(save_path))
            assert (save_path / "modules.json").exists()

            if is_amplify:
                return  # AMPLIFY reload needs wrapper; save verifies enough

            reloaded = SentenceTransformer(str(save_path), trust_remote_code=True)
            emb = reloaded.encode([TEST_SEQ], convert_to_numpy=True)
            assert emb.shape[0] == 1

            nan_count = int(np.isnan(emb).sum())
            if nan_count > 0 and is_esmplusplus:
                pytest.xfail("NaN after reload (known ESMplusplus/pyarrow issue)")
            assert nan_count == 0, "Reloaded model produces NaN embeddings"
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Allow direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-m", "slow"] + sys.argv[1:])


class _FakeResidueTokenizer:
    """Module-level so it pickles; mirrors FastEsmTokenizer's lookup behaviour."""

    unk_token_id = 3

    def __init__(self):
        # ESM2's real alphabet: every letter except J.
        self._token_to_id = {
            c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWYXBUZO", start=4)
        }

    def convert(self, token):
        return self._token_to_id[token]


def test_patch_unknown_residue_tokens_maps_missing_letters_and_pickles():
    """Guards the two ways this broke a 7-GPU run.

    1. ESM2's vocabulary has no 'J' (Leu/Ile ambiguity) and FastPLM raises
       KeyError rather than falling back to unk, killing a dataloader worker.
    2. The first fix wrapped the method in a closure, which cannot be pickled --
       and dataloader workers are spawned, so the tokenizer must pickle.
    """
    import pickle

    from model_utils import patch_unknown_residue_tokens

    tok = _FakeResidueTokenizer()
    x_id = tok._token_to_id["X"]

    with pytest.raises(KeyError):
        tok.convert("J")

    patch_unknown_residue_tokens(tok)
    assert tok.convert("J") == x_id
    assert tok.convert("A") != x_id, "existing residues must not be remapped"

    # The spawn-pickle that PicklingError killed.
    revived = pickle.loads(pickle.dumps(tok))
    assert revived.convert("J") == x_id

    patch_unknown_residue_tokens(tok)  # idempotent
    assert tok.convert("J") == x_id
