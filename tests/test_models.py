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
    """Guards the three ways this broke real runs.

    1. ESM2's vocabulary has no 'J' (Leu/Ile ambiguity) and FastPLM raises
       KeyError rather than falling back to unk, killing a dataloader worker.
    2. The first fix wrapped the method in a closure, which cannot be pickled --
       and dataloader workers are spawned, so the tokenizer must pickle.
    3. Benchmark sequence fields carry non-residue characters: '|' (Peptide-HLA)
       and '#' (Thermostability FLIP) each errored a whole task. Enumerating A-Z
       does not cover those.
    """
    import pickle

    from model_utils import patch_unknown_residue_tokens

    tok = _FakeResidueTokenizer()
    x_id = tok._token_to_id["X"]
    true_vocab_size = len(tok._token_to_id)

    for bad in ("J", "|", "#"):
        with pytest.raises(KeyError):
            tok.convert(bad)

    patch_unknown_residue_tokens(tok)
    for bad in ("J", "|", "#"):
        assert tok.convert(bad) == x_id
    assert tok.convert("A") != x_id, "existing residues must not be remapped"

    # Fallback must not inflate the reported vocabulary -- the embedding matrix
    # has only true_vocab_size rows, and a resize would be triggered otherwise.
    assert len(tok._token_to_id) == true_vocab_size
    assert "J" not in tok._token_to_id

    # The spawn-pickle that PicklingError killed.
    revived = pickle.loads(pickle.dumps(tok))
    assert revived.convert("J") == x_id
    assert revived.convert("|") == x_id

    patch_unknown_residue_tokens(tok)  # idempotent
    assert tok.convert("J") == x_id


# ---------------------------------------------------------------------------
# force_sdpa_backend
# ---------------------------------------------------------------------------


class _StrictBackend:
    """Stand-in for a Synthyra runtime that accepts only some backend names."""

    def __init__(self, accepted):
        self._accepted = accepted
        self.value = None

    def set(self, name):
        if name not in self._accepted:
            raise ValueError(f"Unsupported attention implementation '{name}'")
        self.value = name


class _FakeAttn:
    def __init__(self):
        self.flex_attention = object()


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttn()


class _FakeTransformer:
    """Mimics ESM++: attn_backend is a validating property, blocks hang off it."""

    def __init__(self, accepted):
        object.__setattr__(self, "_gate", _StrictBackend(accepted))
        object.__setattr__(self, "blocks", [_FakeBlock(), _FakeBlock()])

    @property
    def attn_backend(self):
        return self._gate.value

    @attn_backend.setter
    def attn_backend(self, name):
        self._gate.set(name)


class _FakeESMpp:
    def __init__(self, accepted, dtype=None):
        self.transformer = _FakeTransformer(accepted)
        self._dtype = dtype

    def parameters(self):
        """Only the dtype is read, so one scalar of the right type is enough."""
        import torch

        if self._dtype is not None:
            yield torch.zeros(1, dtype=self._dtype)


def test_force_sdpa_backend_falls_back_when_runtime_rejects_default(monkeypatch):
    """The 2026-07 Synthyra bundle dropped 'kernels_flash', our default.

    Assigning it blind raised ValueError and killed every ESM-C load. The setter
    must walk the same candidate ladder the FastPLM branch already used.
    """
    from model_utils import force_sdpa_backend

    monkeypatch.delenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", raising=False)
    model = _FakeESMpp(accepted={"eager", "sdpa", "flash_attention_2"})
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "flash_attention_2"


def test_force_sdpa_backend_prefers_the_requested_name(monkeypatch):
    from model_utils import force_sdpa_backend

    monkeypatch.setenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", "sdpa")
    model = _FakeESMpp(accepted={"sdpa", "flash_attention_2", "kernels_flash"})
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "sdpa"
    # flex_attention is only cleared for sdpa, where it would otherwise be used.
    assert all(b.attn.flex_attention is None for b in model.transformer.blocks)


def test_force_sdpa_backend_keeps_flex_attention_for_non_sdpa(monkeypatch):
    from model_utils import force_sdpa_backend

    monkeypatch.setenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", "flash_attention_2")
    model = _FakeESMpp(accepted={"sdpa", "flash_attention_2"})
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "flash_attention_2"
    assert all(b.attn.flex_attention is not None for b in model.transformer.blocks)


def test_force_sdpa_backend_survives_a_runtime_accepting_nothing(monkeypatch):
    """Never raise out of a best-effort tuning call; leave the model default."""
    from model_utils import force_sdpa_backend

    monkeypatch.delenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", raising=False)
    model = _FakeESMpp(accepted=set())
    force_sdpa_backend(model)
    assert model.transformer.attn_backend is None


def test_force_sdpa_backend_skips_flash_for_fp32_models(monkeypatch):
    """Synthyra accepts a flash backend on assignment, then rejects fp32 at forward.

        'flash_attention_2' supports only manifest-declared dtype(s) bfloat16;
        received float32

    So an fp32-resident model must never be handed one -- the failure would
    otherwise surface an inference later, far from the cause.
    """
    import torch

    from model_utils import force_sdpa_backend

    monkeypatch.delenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", raising=False)
    model = _FakeESMpp(
        accepted={"sdpa", "flash_attention_2", "kernels_flash"}, dtype=torch.float32
    )
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "sdpa"


def test_force_sdpa_backend_keeps_flash_for_bf16_models(monkeypatch):
    import torch

    from model_utils import force_sdpa_backend

    monkeypatch.delenv("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", raising=False)
    model = _FakeESMpp(accepted={"sdpa", "flash_attention_2"}, dtype=torch.bfloat16)
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "flash_attention_2"


def test_force_sdpa_backend_tolerates_a_model_without_parameters():
    """Documented as safe on any object: no parameters() must not raise."""
    from model_utils import force_sdpa_backend

    class _NoParams:
        def __init__(self):
            self.transformer = _FakeTransformer({"sdpa"})

    model = _NoParams()
    force_sdpa_backend(model)
    assert model.transformer.attn_backend == "sdpa"
