"""Late-interaction (ColBERT-style residue MaxSim) helpers for ProtSent.

Built on Sentence Transformers v6 ``MultiVectorEncoder``. The stack is

    Transformer (ESM-2 backbone, reused from protein_pipeline.load_model_for_training)
      -> Dense(hidden -> proj_dim, no bias)          [skipped when proj_dim == 0]
      -> MultiVectorMask (drops <cls>/<eos> on both query and document side)
      -> Normalize (per-residue L2)

``proj_dim == 0`` is the zero-shot control: native hidden states scored with MaxSim.
The Transformer module is shared with the ordinary mean-pooled SentenceTransformer,
so a "dense view" of any late-trained backbone is just that module + mean Pooling.
"""

from __future__ import annotations

import os

# MUST precede any third-party import: sentence_transformers pulls in datasets, which freezes its
# cache path at import time. Home is on a 96%-full volume and an ad-hoc call that forgets these
# writes GBs there. `or`, not setdefault, so an empty value is replaced too. Models use the shared
# hub cache; datasets are per-user and get their own space.
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or "/storage/models/hf_home"
os.environ["HF_DATASETS_CACHE"] = (os.environ.get("HF_DATASETS_CACHE")
                                   or "/storage/users/ddofer/hf_datasets")

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ci import boot_ci, per_query_metrics  # noqa: E402
from sentence_transformers import MultiVectorEncoder, SentenceTransformer  # noqa: E402
from sentence_transformers.base.modules import Dense, Normalize  # noqa: E402
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask  # noqa: E402

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ("<cls>", "<eos>")  # ESM-2 / FastPLM tokenizers; <pad> is already masked
SCOPE_LEVELS: Dict[str, int] = {"fold": 2, "superfamily": 3, "family": 4}


# --------------------------------------------------------------------------- models
# Flash-attention kernel that ships prebuilt via the `kernels` package, so no
# flash-attn build is needed. Measured on an A100 against sdpa, ProtSent-V2-35M,
# bs128/mini64, steady state over 20 post-warmup steps:
#     Pfam pairs    122.2 -> 240.9 pairs/s (1.97x), peak 10.67 -> 4.16 GB
#     STRING pairs   91.4 -> 135.6 pairs/s (1.48x), peak 11.88 -> 8.49 GB
# Most of that is input unpadding, which Sentence Transformers enables
# automatically whenever flash attention is active (1.30x of the STRING gain;
# the kernel itself is the other 1.14x). That is also why mini_batch_num_tokens
# does not help here: unpadding has already removed the padding work that token
# packing exists to reclaim.
FLASH_ATTENTION = "kernels-community/vllm-flash-attn3"


def build_multivector_encoder(
    model_name_or_path: str,
    *,
    proj_dim: int = 64,
    max_seq_length: int = 512,
    device: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    allow_head_reinit: bool = False,
) -> Tuple[MultiVectorEncoder, SentenceTransformer]:
    """Return ``(mve, dense_st)`` sharing one backbone Transformer module.

    ``dense_st`` is the ordinary ``[Transformer, mean Pooling]`` SentenceTransformer from
    ``protein_pipeline.load_model_for_training`` (max_seq_length enforced, ESM/FastPLM patches
    applied). ``mve`` reuses ``dense_st[0]`` in place, so training the multi-vector model trains
    the dense view too and the two can never drift apart.

    ``attn_implementation`` pins an attention backend, e.g. :data:`FLASH_ATTENTION` for the ~2x
    throughput measured above. Two conditions come with it, and both change more than speed:

    * Weights stay **fp32**. The kernel needs half-precision activations, which ``bf16=True``
      autocast supplies; transformers accepts pinned flash on an fp32 model. Loading in bf16 too
      was a serious bug: AdamW's params became bf16, and a 1e-5 update is ~1/24th of bf16's
      spacing near a typical weight, so 97.6% of backbone elements could not move (fp32: 93.4%).
      See test_configured_learning_rate_actually_moves_the_backbone.
    ``allow_head_reinit`` opts in to training a fresh head on a checkpoint that saved a different
    width. Without it that mismatch raises: it is the one condition under which "continuing" and
    "starting over" look identical in the logs but differ completely in what is being measured.

    * It reaches ``models.Transformer`` through ``load_model_for_training``'s native-checkpoint
      branch, which is the only branch that can honour it. :func:`resolve_attention` gates on the
      same families, and the loader raises rather than silently ignoring the request.
    """
    from protein_pipeline import load_model_for_training

    dense_st = load_model_for_training(
        model_name_or_path, max_seq_length=max_seq_length, device=device, pooling_mode="mean",
        model_kwargs=({"attn_implementation": attn_implementation}
                      if attn_implementation else None),
    )
    # One Transformer instance, two views of it: max_seq_length set on either reaches the same
    # module, so there is nothing to keep in sync.
    mve = MultiVectorEncoder(modules=_late_modules(dense_st[0], proj_dim), device=device)
    _restore_saved_projection(mve, model_name_or_path, allow_head_reinit=allow_head_reinit)
    return mve, dense_st


def _restore_saved_projection(mve: MultiVectorEncoder, path: str, *,
                              allow_head_reinit: bool = False) -> bool:
    """Reuse a saved projection head when continuing from a late checkpoint.

    A saved model keeps its backbone at the root and its projection in ``1_Dense/``; the loader
    above reads only the root. Without this, a continuation pairs a trained backbone with a fresh
    random head -- which runs, converges, and silently discards what the head had learned.
    """
    dense = next((m for m in mve if isinstance(m, Dense)), None)
    saved = Path(path) / "1_Dense"
    if dense is None or not saved.is_dir():
        return False
    try:
        loaded = Dense.load(str(saved))
    except Exception as exc:  # a malformed dir must not kill an 8-hour run at minute one
        logger.warning("could not read the saved projection at %s (%s); keeping the fresh head",
                       saved, exc)
        return False
    if loaded.linear.weight.shape != dense.linear.weight.shape:
        have, want = tuple(loaded.linear.weight.shape), tuple(dense.linear.weight.shape)
        if not allow_head_reinit:
            raise ValueError(
                f"{path} saved a {have} projection head but this run asks for {want}, so the "
                f"trained head cannot be carried over. Pass --proj_dim {have[0]} to continue it, "
                f"or --allow_head_reinit to train a fresh head on this backbone.")
        have, want = have[0], want[0]
        logger.warning("saved projection is %d-D but this run asks for %d-D; training a fresh head "
                       "on this backbone (--allow_head_reinit)", have, want)
        return False
    # Dense.load takes no device/dtype: it lands on CPU in whatever it was saved as. Match here.
    target = dense.linear.weight
    loaded = loaded.to(device=target.device, dtype=target.dtype)
    dense.load_state_dict(loaded.state_dict())
    logger.info("continuing from the saved %d-D projection head in %s",
                dense.linear.weight.shape[0], saved)
    return True


def _late_modules(transformer, proj_dim: int) -> List[torch.nn.Module]:
    """Transformer -> (Dense proj) -> MultiVectorMask -> Normalize."""
    modules: List[torch.nn.Module] = [transformer]
    if proj_dim:
        modules.append(
            Dense(
                transformer.get_embedding_dimension(),
                proj_dim,
                bias=False,
                activation_function=None,
                module_input_name="token_embeddings",
            )
        )
    return modules + [
        MultiVectorMask(skiplist_words=list(SPECIAL_TOKENS), skiplist_tasks=["query", "document"]),
        Normalize(module_input_name="token_embeddings"),
    ]


def resolve_attention(requested: Optional[str], model_name_or_path: str) -> Optional[str]:
    """Pick an attention backend: ``auto`` tries flash and falls back to sdpa.

    Returns the implementation string to pass to ``build_multivector_encoder``, or None
    for the standard fp32 + autocast path. A backend is only reported as available after
    the model actually loads under it, because a kernels version below the floor fails at
    load time rather than at import time.
    """
    if requested in (None, "", "sdpa"):
        return None
    if requested != "auto":
        return requested
    # Only the native-checkpoint branch can honour a pinned backend; the FastPLM branch swaps the
    # module's model out and would discard it. The loader raises, so this gate is what keeps a
    # Synthyra arm from erroring out -- and why it quietly trains on sdpa instead.
    from model_utils import detect_model_type

    family = detect_model_type(model_name_or_path)
    if family != "standard":
        logger.info("attention backend: sdpa (%s is %s; flash needs the native-ESM load path)",
                    model_name_or_path, family)
        return None
    try:
        import kernels
    except ImportError as exc:
        logger.warning("flash attention unavailable (%s); falling back to sdpa", exc)
        return None
    version = tuple(int(x) for x in kernels.__version__.split(".")[:2])
    if not ((0, 15) <= version < (0, 16)):
        logger.warning(
            "flash attention needs kernels >=0.15.2,<0.16 but %s is installed; falling back "
            "to sdpa. Run `uv sync` -- every queue job uses `uv run --no-sync`, so a corrected "
            "lock does not reach the venv on its own.", kernels.__version__)
        return None
    logger.info("attention backend: %s", FLASH_ATTENTION)
    return FLASH_ATTENTION


def load_multivector_encoder(path: str, device: Optional[str] = None) -> MultiVectorEncoder:
    """Load a saved late model (``make_loadable`` is applied first so FastPLM-saved
    checkpoints load as plain EsmModel without remote code)."""
    make_loadable(path)
    return MultiVectorEncoder(path, device=device)


def make_loadable(path: str) -> bool:
    """Rewrite FastPLM metadata -> vanilla ESM so stock transformers/ST load it."""
    from make_checkpoint_loadable import convert

    return convert(Path(path), backup=False)


def _sync_token_dropout_config(auto_model: torch.nn.Module) -> None:
    """Make ``config.token_dropout`` match what the model actually computes.

    ``model_utils.disable_esm2_token_dropout`` flips ``config.token_dropout`` at
    load time, but ``EsmEmbeddings`` caches the flag in ``__init__`` so the flip
    never changes the running model — it only poisons the *saved* config, and a
    reload then silently drops ESM's (1 - 0.15*0.8) embedding scaling (~0.09 max
    abs drift on ESM2-8M). Restore the runtime truth before every save.
    """
    embeddings = getattr(auto_model, "embeddings", None)
    config = getattr(auto_model, "config", None)
    if config is not None and hasattr(embeddings, "token_dropout"):
        config.token_dropout = bool(embeddings.token_dropout)


def save_late_and_dense(
    mve: MultiVectorEncoder, pooling: torch.nn.Module, out_dir: str
) -> Tuple[str, str]:
    """Save the multi-vector model and its mean-pooled "dense view" side by side.

    The dense view is ``[trained backbone, mean Pooling]`` -- an ordinary ProtSent model, which
    is what the pooled benchmarks consume, since they cannot score an ``[L x d]`` model.
    """
    late_dir, dense_dir = os.path.join(out_dir, "late"), os.path.join(out_dir, "dense_view")
    _sync_token_dropout_config(mve[0].auto_model)
    mve.save_pretrained(late_dir)
    make_loadable(late_dir)
    SentenceTransformer(modules=[mve[0], pooling], device=str(mve.device)).save_pretrained(dense_dir)
    make_loadable(dense_dir)
    return late_dir, dense_dir


def freeze_unused_heads(mve: MultiVectorEncoder) -> int:
    """Freeze heads that never feed token_embeddings (EsmModel pooler / contact
    head, FastPLM's dropped lm_head): DDP's reducer aborts on parameters that
    receive no grad. Confirmed unused by DDP's own no-grad report on both arms.
    """
    frozen = 0
    for name, param in mve.named_parameters():
        if ".pooler." in name or "contact_head" in name or "lm_head" in name:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def backbone_and_projection_params(mve: MultiVectorEncoder):
    """(backbone params, projection params) for two-LR optimisers (trainable only)."""
    proj = [p for m in mve if isinstance(m, Dense) for p in m.parameters() if p.requires_grad]
    proj_ids = {id(p) for p in proj}
    backbone = [p for p in mve.parameters() if p.requires_grad and id(p) not in proj_ids]
    return backbone, proj


def param_groups_for(mve: MultiVectorEncoder, *, lr: float, proj_lr: float) -> list:
    """Optimizer param groups. ``proj_lr <= 0`` means one group at ``lr`` — the default for the
    clean runs. The 10x head group was fine-tuning folklore rather than sentence-transformers
    practice (ST and PyLate train backbone + fresh projection at a single LR, and Adam adapts
    per-parameter anyway); it stays available for reproducing the pilot arms."""
    backbone, proj = backbone_and_projection_params(mve)
    if proj_lr <= 0 or not proj:
        return [{"params": backbone + proj, "lr": lr}]
    return [{"params": backbone, "lr": lr}, {"params": proj, "lr": proj_lr}]


# --------------------------------------------------------------------------- scoring
def maxsim_matrix(
    mve: MultiVectorEncoder,
    seqs: Sequence[str],
    *,
    batch_size: int = 64,
    chunk_elements: int = 50_000_000,
    queries: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Exact all-vs-all MaxSim scores (queries x seqs; queries default to seqs)."""
    # convert_to_numpy keeps the corpus on the host: ST's default leaves every sequence as a
    # separate CUDA allocation, which for CATH's 69,605-domain lookup set is 5.4 GB at 128-D
    # and 20-26 GB at native 480/640-D -- to produce a 40 MB score matrix. similarity(device=)
    # then moves one budget-sized document chunk at a time instead of the whole corpus.
    # Measured: bit-identical scores (max|diff| = 0.0), peak allocation 0.37 GB, and no slower.
    docs = mve.encode_document(list(seqs), batch_size=batch_size, show_progress_bar=False,
                               convert_to_numpy=True)
    qs = docs if queries is None else mve.encode_query(list(queries), batch_size=batch_size,
                                                       show_progress_bar=False, convert_to_numpy=True)
    device = str(mve.device)
    with torch.no_grad():
        scores = mve.similarity(qs, docs, chunk_elements=chunk_elements, device=device)
    return scores.float().cpu().numpy()


def maxsim_against_one(
    mve: MultiVectorEncoder,
    document: str,
    queries: Sequence[str],
    *,
    batch_size: int = 256,
    chunk_queries: int = 8192,
    chunk_elements: int = 25_000_000,
) -> np.ndarray:
    """Mean-MaxSim of every query against ONE document.

    ``maxsim_matrix`` is built for all-vs-all and converts the corpus to numpy, which is right for a
    69k-domain lookup set and wrong for "score 2.4M variants against a single wild type". Here the
    document is encoded once and every query batch is scored while it is still on the GPU.

    This is a thin wrapper over the library: ``encode_*`` keeps embeddings on the device (that is
    the ST 6 default), and ``similarity`` takes a bare 2-D left operand as a batch of one and bounds
    its own token-token intermediate via ``chunk_elements``. Hand-rolling those was both more code
    and, without chunking, an OOM at any interesting batch size.

    Returns **mean**-MaxSim (``length_normalize=True``), so the divisor is the real unmasked token
    count rather than an estimate from the string length, and a sequence scored against itself is
    1.0. Note this differs from ``maxsim_matrix``, which returns the raw sum.
    """
    mve.eval()
    use_bf16 = mve.device.type == "cuda"

    def encode(fn, texts: List[str]):
        # bf16 forward, fp32 scoring: measured 2.35x faster than fp32 throughout, for 0.0020 mean
        # (0.0051 max) on the per-assay Spearman we report -- which averages down across assays to
        # ~0.0002 on the headline, against effects of 0.014-0.093. Scoring stays fp32 because ST
        # upcasts only AFTER the token-token matmul, so bf16 there would quantize each per-residue
        # maximum before the sum.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            return fn(list(texts), convert_to_numpy=False, show_progress_bar=False)

    doc = encode(mve.encode_document, [document])[0].float()
    scores = []
    for i in range(0, len(queries), chunk_queries):
        qs = encode(
            lambda x, **kw: mve.encode_query(x, batch_size=batch_size, **kw),
            queries[i : i + chunk_queries],
        )
        with torch.no_grad():
            sim = mve.similarity(
                [q.float() for q in qs], [doc],
                chunk_elements=chunk_elements, length_normalize=True,
            )
        scores.append(sim[:, 0].float().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0)


def cosine_matrix(st: SentenceTransformer, seqs: Sequence[str], *, batch_size: int = 64,
                  queries: Optional[Sequence[str]] = None) -> np.ndarray:
    def encode(texts):
        return st.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    docs = encode(seqs)
    return (docs if queries is None else encode(queries)) @ docs.T


# --------------------------------------------------------------------------- SCOPe
def scope_labels(families: Iterable[str], level: str) -> np.ndarray:
    n = SCOPE_LEVELS[level]
    return np.array([".".join(str(f).split(".")[:n]) for f in families])


def ranking_from_similarity(sim: np.ndarray) -> np.ndarray:
    """Descending argsort of a square similarity matrix with self removed: (n, n-1)."""
    s = np.array(sim, dtype=np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind="stable")[:, :-1]


def scope_rows(
    sim: np.ndarray,
    families: Sequence[str],
    *,
    model: str,
    scoring: str,
    n_boot: int = 1000,
    seed: int = 0,
    runtime_s: float = float("nan"),
) -> Tuple[List[dict], Dict[str, Dict[str, np.ndarray]]]:
    """One result row per hierarchy level + per-query vectors (for paired bootstrap).

    Metric definitions follow bootstrap_ci.per_query_metrics / ProtBench:
    Recall@K over all queries (singletons count as misses), MAP over the full
    ranking, and ``eligible_*`` over queries with >=1 same-label gallery item.
    """
    ranking = ranking_from_similarity(sim)
    rows, per_query = [], {}
    for level in SCOPE_LEVELS:
        pq = per_query_metrics(ranking, scope_labels(families, level))
        rows.append({"model": model, "scoring": scoring, "level": level, "runtime_s": runtime_s,
                     **retrieval_row(pq, n_boot=n_boot, seed=seed)})
        per_query[level] = pq
    return rows, per_query


def retrieval_row(pq: Dict[str, np.ndarray], *, n_boot: int = 0, seed: int = 0) -> dict:
    """Per-query vectors -> the metric columns every retrieval table in this project uses.

    The ``eligible_*`` spelling is load-bearing: those are means over queries with at least one
    reachable positive, roughly 2 points above plain Recall over all queries. A table that
    reports one under the other's name joins wrongly against the rest. Having said that twice,
    in two places that computed it separately, is what this function replaces.

    Recall@k is read straight off ``per_query_metrics``; recomputing it here built three more
    n x (n-1) label matrices per model only to overwrite pq with identical values.
    """
    elig = pq["eligible"]
    row = {"n_queries": int(len(elig)), "n_eligible_queries": int(elig.sum())}
    for k in (1, 10, 30):
        hit = pq[f"hit{k}"]
        row[f"Recall@{k}"] = float(hit.mean())
        row[f"eligible_Recall@{k}"] = float(hit[elig].mean()) if elig.any() else float("nan")
    row["MAP"] = float(pq["ap"].mean())
    row["eligible_MAP"] = float(pq["ap"][elig].mean()) if elig.any() else float("nan")
    if n_boot and elig.any():
        for name in ("hit1", "hit10", "ap"):
            _, lo, hi = boot_ci(pq[name][elig], n_boot=n_boot, seed=seed)
            row[f"eligible_{name}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
    return row


def paired_bootstrap(
    pq_a: Dict[str, np.ndarray], pq_b: Dict[str, np.ndarray], *, metrics=("hit1", "hit10", "ap"),
    n_boot: int = 1000, seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """CI of (a - b) over the eligible queries shared by both (same gallery => same mask)."""
    elig = pq_a["eligible"] & pq_b["eligible"]
    out = {}
    for m in metrics:
        mean, lo, hi = boot_ci(pq_a[m][elig] - pq_b[m][elig], n_boot=n_boot, seed=seed)
        out[m] = {"delta": mean, "ci95_lo": lo, "ci95_hi": hi, "n": int(elig.sum())}
    return out


def load_scope40() -> Tuple[List[str], List[str]]:
    from datasets import load_dataset

    ds = load_dataset("tattabio/scope40_test", split="train")
    return list(ds["sequence"]), list(ds["family"])
