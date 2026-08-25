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


def test_scoring_one_document_matches_the_all_vs_all_scorer(tiny_models):
    """The streamed one-document path must agree with `maxsim_matrix`, which it replaces.

    `maxsim_against_one` exists purely as a faster route to a number `maxsim_matrix` already
    computes, and its docstring claims they match. Nothing asserted that, so the claim was load
    bearing and unguarded: every ProteinGym score now comes from the fast path, and a silent
    divergence would move published results with no test failing.

    Substitution-shaped inputs on purpose -- same length as the document, differing in one residue
    -- because that is the regime the benchmark actually scores and the regime where the two paths
    are most likely to drift.
    """
    mve, _ = tiny_models
    wt = SEQS[1]
    mutants = [wt[:i] + ("A" if wt[i] != "A" else "G") + wt[i + 1:] for i in range(0, len(wt), 3)]

    fast = li.maxsim_against_one(mve, wt, mutants)
    reference = li.maxsim_matrix(mve, [wt], queries=mutants, chunk_elements=1_000_000)[:, 0]

    assert fast.shape == (len(mutants),)
    assert np.allclose(fast, reference, atol=1e-3), np.abs(fast - reference).max()
    # Ranking is what Spearman consumes, so agreement on order matters more than on magnitude.
    assert list(np.argsort(fast)) == list(np.argsort(reference))


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


def test_continuing_from_a_saved_model_keeps_its_trained_head(tiny_models, tmp_path):
    """Continuing training from a saved late model must not re-randomise its projection.

    The saved layout puts the backbone at the directory root (modules.json gives the Transformer
    `path: ""`) with the projection in `1_Dense/`. A loader that only reads the root therefore
    picks up a fully trained backbone and silently pairs it with a fresh random head — which
    looks like a working continuation and throws away everything the head learned.
    """
    import torch

    mve, _ = tiny_models
    with torch.no_grad():  # make the head distinctive so a fresh init cannot coincide
        mve[1].linear.weight.fill_(0.0123)
    saved = tmp_path / "late"
    mve.save_pretrained(str(saved))

    continued, _ = li.build_multivector_encoder(
        str(saved), proj_dim=16, max_seq_length=64, device="cpu"
    )
    assert torch.allclose(continued[1].linear.weight, mve[1].linear.weight), (
        "continuation discarded the trained projection head"
    )


def test_cosine_scores_are_not_divided_by_length():
    """Cosine is already length-invariant; dividing it by length corrupts the ranking.

    MaxSim sums over query residues, so it must be divided to become mean-MaxSim. Cosine is
    normalised, so the same division is not a normalisation but a bug — invisible on substitutions
    (equal lengths, constant divisor) and destructive on indels (varying lengths).
    """
    from late_interaction_eval import variant_scores

    sim = np.array([0.5, 0.9])                      # variant 1 is the better cosine match
    lens = np.array([10.0, 400.0])                  # an indel set: lengths differ
    assert np.array_equal(variant_scores(sim, lens, "cosine"), sim)
    assert np.allclose(variant_scores(sim, lens, "maxsim"), sim / lens)
    # dividing cosine by length flips which variant ranks first -- that is what broke Spearman
    assert list(np.argsort(-variant_scores(sim, lens, "cosine"))) == [1, 0]
    assert list(np.argsort(-(sim / lens))) == [0, 1]


def test_head_size_mismatch_refuses_to_silently_reinitialise(tiny_models, tmp_path):
    """Asking for a different proj_dim than the checkpoint saved must fail loudly.

    This warned and carried on, which is how `protsent_late_150m_prop` came to spend 30,000 steps
    training a fresh random 128-D head on a parent that had saved a 64-D one. The run looked like
    a continuation, its logs said "continuing", and its step-0 checkpoint -- the baseline every
    later checkpoint was compared against -- was a random projection. That made an ordinary
    regression read as a scale-dependent sign flip.

    A dimension mismatch is never a recoverable condition: the saved head cannot be loaded at the
    requested width by any means, so continuing can only mean discarding it. Say so before the
    GPUs are booked, not in a warning nobody reads eight hours later.
    """
    import torch

    mve, _ = tiny_models
    with torch.no_grad():
        mve[1].linear.weight.fill_(0.0123)
    saved = tmp_path / "late"
    mve.save_pretrained(str(saved))

    with pytest.raises(ValueError, match="16-D.*8-D|8-D.*16-D"):
        li.build_multivector_encoder(str(saved), proj_dim=8, max_seq_length=64, device="cpu")


def test_head_size_mismatch_is_allowed_when_asked_for_explicitly(tiny_models, tmp_path):
    """Retargeting a checkpoint to a new head width is a real experiment, just never an accident."""
    import torch

    mve, _ = tiny_models
    with torch.no_grad():
        mve[1].linear.weight.fill_(0.0123)
    saved = tmp_path / "late"
    mve.save_pretrained(str(saved))

    retargeted, _ = li.build_multivector_encoder(
        str(saved), proj_dim=8, max_seq_length=64, device="cpu", allow_head_reinit=True
    )
    assert retargeted[1].linear.weight.shape[0] == 8
    assert not torch.allclose(retargeted[1].linear.weight,
                              torch.full_like(retargeted[1].linear.weight, 0.0123))
