"""Late-interaction helpers: masking, dense-view parity, SCOPe hierarchy metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import late_interaction as li  # noqa: E402

TINY = "facebook/esm2_t6_8M_UR50D"
SEQS = ["MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GSHMLEDPQ"]


@pytest.fixture(scope="module")
def tiny_models():
    try:
        return li.build_multivector_encoder(TINY, proj_dim=16, max_seq_length=64, device="cpu")
    except Exception as exc:  # no network / cache
        pytest.skip(f"cannot load {TINY}: {exc}")


def test_mask_drops_special_tokens_and_normalises(tiny_models):
    mve, _ = tiny_models
    embs = mve.encode_document(SEQS)
    assert [e.shape[0] for e in embs] == [len(s) for s in SEQS]  # no <cls>/<eos>
    assert embs[0].shape[1] == 16
    assert np.allclose(np.linalg.norm(embs[0].cpu().numpy(), axis=1), 1.0, atol=1e-4)
    # self MaxSim of an L2-normalised residue set is exactly L (each residue matches itself)
    sim = li.maxsim_matrix(mve, SEQS, chunk_elements=1_000_000)
    assert np.allclose(np.diag(sim), [len(s) for s in SEQS], atol=1e-2)


def test_zero_shot_control_uses_native_hidden_size():
    try:
        mve, st = li.build_multivector_encoder(TINY, proj_dim=0, max_seq_length=64, device="cpu")
    except Exception as exc:
        pytest.skip(f"cannot load {TINY}: {exc}")
    embs = mve.encode_document(SEQS[:1])
    assert embs[0].shape == (len(SEQS[0]), st.get_sentence_embedding_dimension())


def test_dense_view_parity_roundtrip(tiny_models, tmp_path):
    from sentence_transformers import SentenceTransformer

    mve, st = tiny_models
    ref = st.encode(SEQS, convert_to_numpy=True)
    late_dir, dense_dir = li.save_late_and_dense(mve, st[1], str(tmp_path))
    reloaded = SentenceTransformer(dense_dir, device="cpu")
    assert np.abs(reloaded.encode(SEQS, convert_to_numpy=True) - ref).max() < 1e-5
    mve2 = li.load_multivector_encoder(late_dir, device="cpu")
    a, b = mve.encode_document(SEQS[:1])[0], mve2.encode_document(SEQS[:1])[0]
    assert np.abs(a.cpu().numpy() - b.cpu().numpy()).max() < 1e-5
    backbone, proj = li.backbone_and_projection_params(mve)
    assert len(proj) == 1 and len(backbone) > 1


def test_scope_levels():
    assert li.scope_labels(["a.5.6.1"], "fold").tolist() == ["a.5"]
    assert li.scope_labels(["a.5.6.1"], "superfamily").tolist() == ["a.5.6"]
    assert li.scope_labels(["a.5.6.1"], "family").tolist() == ["a.5.6.1"]


def test_scope_rows_hierarchy_and_eligibility():
    # q0,q1: same family; q2: same superfamily as q0/q1 only; q3: singleton fold.
    fam = ["a.1.1.1", "a.1.1.1", "a.1.1.2", "b.1.1.1"]
    sim = np.array([[9, 3, 2, 0], [3, 9, 2, 0], [2, 3, 9, 0], [0, 0, 0, 9]], dtype=float)
    rows, pq = li.scope_rows(sim, fam, model="m", scoring="s", n_boot=0)
    by = {r["level"]: r for r in rows}
    assert by["family"]["n_eligible_queries"] == 2 and by["family"]["eligible_Recall@1"] == 1.0
    assert by["family"]["Recall@1"] == pytest.approx(0.5)  # singletons count as misses
    assert by["superfamily"]["n_eligible_queries"] == 3 and by["superfamily"]["MAP"] == pytest.approx(0.75)
    assert by["fold"]["n_eligible_queries"] == 3
    assert not pq["fold"]["eligible"][3]
    delta = li.paired_bootstrap(pq["family"], pq["family"], n_boot=10)
    assert delta["ap"]["delta"] == 0.0 and delta["ap"]["n"] == 2


def test_ranking_excludes_self():
    sim = np.random.default_rng(0).random((5, 5))
    r = li.ranking_from_similarity(sim)
    assert r.shape == (5, 4) and all(i not in r[i] for i in range(5))


def test_flash_attention_path_builds_the_same_module_stack(tiny_models):
    """The opt-in backend path must produce the same stack as the default one.

    Only the stack shape is asserted here: loading the flash kernel needs a GPU
    and kernels>=0.15.2, so the backend itself is exercised by the throughput
    probe rather than in a unit test.
    """
    mve, _ = tiny_models
    default = [type(m).__name__ for m in mve]
    from sentence_transformers.base.modules import Transformer

    rebuilt = [type(m).__name__ for m in li._late_modules(mve[0], proj_dim=16)]
    assert default == rebuilt
    assert isinstance(mve[0], Transformer)
    assert li.FLASH_ATTENTION.startswith("kernels-community/")
