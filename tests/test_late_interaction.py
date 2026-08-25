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
    """The fast path must agree with `maxsim_matrix`, which it replaces.

    Substitution-shaped inputs: that is the regime ProteinGym scores.
    """
    mve, _ = tiny_models
    wt = SEQS[1]
    mutants = [wt[:i] + ("A" if wt[i] != "A" else "G") + wt[i + 1:] for i in range(0, len(wt), 3)]

    fast = li.maxsim_against_one(mve, wt, mutants)
    # maxsim_matrix returns the raw sum over query residues; maxsim_against_one returns the mean,
    # so the reference is divided by the residue count (the mask drops <cls>/<eos>, and ESM emits
    # one token per residue).
    raw = li.maxsim_matrix(mve, [wt], queries=mutants, chunk_elements=1_000_000)[:, 0]
    reference = raw / np.array([len(m) for m in mutants], dtype=float)

    assert fast.shape == (len(mutants),)      # allclose broadcasts; shape must be checked first
    assert np.allclose(fast, reference, atol=1e-3), np.abs(fast - reference).max()
    assert list(np.argsort(fast)) == list(np.argsort(reference))   # Spearman consumes the ranking


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


def test_scores_are_length_invariant_at_every_length(tiny_models):
    """A sequence scored against itself is 1.0, whatever its length.

    Divide by length twice and a self-score becomes 1/L; two very different lengths pin that.
    """
    mve, _ = tiny_models
    short, long = SEQS[2], SEQS[1]
    assert len(long) > 2 * len(short)
    for seq in (short, long):
        assert li.maxsim_against_one(mve, seq, [seq])[0] == pytest.approx(1.0, abs=1e-3)
def test_head_size_mismatch_refuses_to_silently_reinitialise(tiny_models, tmp_path):
    """Continuing at a different head width must fail loudly, not train a fresh random head.

    The 150M arm did exactly that: its parent had a 64-D head, the run asked for 128-D, and it
    spent 30,000 steps re-learning a head it should have inherited -- finishing below the model
    it continued from. Referenced by STATUS.md, so it needs to exist.
    """
    import pytest as _pytest

    mve, _ = tiny_models
    saved = tmp_path / "late"
    mve.save_pretrained(str(saved))

    with _pytest.raises(ValueError, match="projection"):
        li.build_multivector_encoder(str(saved), proj_dim=32, max_seq_length=64, device="cpu")

    # opting in is allowed, and must still work
    other, _ = li.build_multivector_encoder(
        str(saved), proj_dim=32, max_seq_length=64, device="cpu", allow_head_reinit=True
    )
    assert other[1].linear.weight.shape[0] == 32
def test_single_lr_mode_puts_every_parameter_in_one_group(tiny_models):
    """`proj_lr <= 0` must mean one param group at one LR — the simple config the clean runs use.

    The 10x head group was fine-tuning folklore, not sentence-transformers practice (ST and
    PyLate train backbone + fresh projection at a single LR); Adam's per-parameter adaptivity
    makes it redundant. It stays available for reproducing the pilot arms, but the default path
    should have one knob.
    """
    mve, _ = tiny_models
    groups = li.param_groups_for(mve, lr=5e-5, proj_lr=0.0)
    assert len(groups) == 1
    assert groups[0]["lr"] == 5e-5
    n_trainable = sum(1 for p in mve.parameters() if p.requires_grad)
    assert len(groups[0]["params"]) == n_trainable

    two = li.param_groups_for(mve, lr=1e-5, proj_lr=1e-4)
    assert [g["lr"] for g in two] == [1e-5, 1e-4]
    assert sum(len(g["params"]) for g in two) == n_trainable


def test_constant_schedule_reaches_the_optimizer_and_holds_peak():
    """--lr_scheduler constant_with_warmup: after warmup the LR is the peak, at any step.

    The point of the constant schedule is that checkpoints at different step counts stay
    comparable — a cosine run's checkpoint quality depends on where the anneal was, which is
    exactly the confound that muddied the phase-2 comparison.
    """
    import torch
    from transformers import get_scheduler

    from train_late_interaction import parse_args

    import sys as _sys

    argv = _sys.argv
    _sys.argv = ["x", "--model", "m", "--files", "f", "--output_dir", "o",
                 "--lr_scheduler", "constant_with_warmup"]
    try:
        args = parse_args()
    finally:
        _sys.argv = argv
    assert args.lr_scheduler == "constant_with_warmup"

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=5e-5)
    sched = get_scheduler(args.lr_scheduler, opt, num_warmup_steps=10, num_training_steps=1000)
    for _ in range(10):
        opt.step(); sched.step()
    lrs = []
    for _ in range(500):
        opt.step(); sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert all(abs(x - 5e-5) < 1e-12 for x in lrs), "LR drifted after warmup on a constant schedule"


def test_save_per_query_creates_its_own_output_dir(tmp_path):
    """cmd_watch_curve never mkdir's out_dir (cmd_scope does); the write must not depend on the
    caller having created it, or a live watcher silently drops every checkpoint's scores.

    This is exactly what happened during the vanilla35m_clean run 2026-08-25: the watcher ran
    for 30+ minutes across 3 checkpoints, encoding 2,207 SCOPe sequences each time, before the
    caught-and-logged FileNotFoundError was noticed -- pure GPU time with zero output.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import late_interaction_eval as lev

    out = tmp_path / "does" / "not" / "exist" / "yet"
    assert not out.exists()
    pq = {"fold": {"ap": np.array([0.5, 0.6])}}
    path = lev.save_per_query(out, "arm", pq)
    assert path.exists()
    assert out.exists()
