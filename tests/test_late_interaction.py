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


def _fraction_of_params_moved(mve, lr):
    """One AdamW step at `lr` on the backbone; returns the fraction of ELEMENTS that changed.

    AdamW's first step moves every weight by very close to `lr` regardless of gradient
    magnitude (m_hat / sqrt(v_hat) is ~1 elementwise), which is what makes this a clean
    probe of whether an update of that size can be represented at all.

    Counting elements rather than tensors is the whole point. A tensor counts as "moved" if a
    single one of its millions of elements did, which on real 35M weights reads 24/24 tensors
    moved at 4.6% element movement -- a proxy that saturates exactly where the bug lives.
    """
    import torch

    backbone, _ = li.backbone_and_projection_params(mve)
    before = [p.detach().clone() for p in backbone]
    opt = torch.optim.AdamW(backbone, lr=lr)
    feats = mve.tokenize(SEQS)
    mve(feats)["token_embeddings"].float().pow(2).mean().backward()
    opt.step()
    moved = sum(int((b != p.detach()).sum()) for b, p in zip(before, backbone))
    return moved / sum(p.numel() for p in backbone)


def test_configured_learning_rate_actually_moves_the_backbone(tiny_models):
    """A step at the trainer's own backbone LR must change the weights it is applied to.

    bf16 carries an 8-bit mantissa, so representable values near magnitude 0.05 are spaced
    ~2**-8 * 0.05 = 2e-4 apart. train_late_interaction.py's default backbone LR is 1e-5, so
    an AdamW step asks each weight to move ~1/20th of that spacing. If the parameters
    themselves are bf16 rather than fp32 masters, the addition rounds to a no-op and the
    backbone cannot train at all. The spacing is a property of the bf16 format, not
    something this codebase computes.
    """
    mve, _ = tiny_models
    assert _fraction_of_params_moved(mve, lr=1e-5) > 0.9, "fp32 baseline must move"

    # Pinning a backend must not silently change the weight dtype. sdpa reaches the same
    # model_kwargs path as flash without needing a GPU; the regression this guards is a
    # dtype=bfloat16 that used to ride along with it.
    pinned_mve, _ = li.build_multivector_encoder(
        TINY, proj_dim=16, max_seq_length=64, device="cpu", attn_implementation="sdpa"
    )
    moved = _fraction_of_params_moved(pinned_mve, lr=1e-5)
    assert moved > 0.9, (
        f"pinning a backend cost the backbone its updates at lr 1e-5: only {moved:.1%} of "
        f"elements moved (half-precision weights cannot represent a step that small)"
    )
