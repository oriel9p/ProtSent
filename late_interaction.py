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
) -> Tuple[MultiVectorEncoder, SentenceTransformer]:
    """Return ``(mve, dense_st)`` sharing one backbone Transformer module.

    ``dense_st`` is the ordinary ``[Transformer, mean Pooling]`` SentenceTransformer from
    ``protein_pipeline.load_model_for_training`` (max_seq_length enforced, ESM/FastPLM patches
    applied). ``mve`` reuses ``dense_st[0]`` in place, so training the multi-vector model trains
    the dense view too and the two can never drift apart.

    ``attn_implementation`` pins an attention backend, e.g. :data:`FLASH_ATTENTION` for the ~2x
    throughput measured above. Two conditions come with it, and both change more than speed:

    * Flash attention requires **bf16/fp16 weights**; transformers refuses it on an fp32 model.
      The default recipe loads fp32 and autocasts, keeping fp32 master weights for the
      optimizer, so pinning flash moves the optimizer to bf16 states. That is a numerics change.
    * It reaches ``models.Transformer`` through ``load_model_for_training``'s native-checkpoint
      branch, which is the only branch that can honour it. :func:`resolve_attention` gates on the
      same families, and the loader raises rather than silently ignoring the request.
    """
    from protein_pipeline import load_model_for_training

    dense_st = load_model_for_training(
        model_name_or_path, max_seq_length=max_seq_length, device=device, pooling_mode="mean",
        model_kwargs=({"attn_implementation": attn_implementation, "dtype": torch.bfloat16}
                      if attn_implementation else None),
    )
    # One Transformer instance, two views of it: max_seq_length set on either reaches the same
    # module, so there is nothing to keep in sync.
    mve = MultiVectorEncoder(modules=_late_modules(dense_st[0], proj_dim), device=device)
    return mve, dense_st


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
    # Family check first. Only load_model_for_training's native-checkpoint branch can honour a
    # pinned backend; the FastPLM branch (FastPLMESM2Wrapper, native tokenizer,
    # force_sdpa_backend) swaps the module's model out and would discard it. The loader raises
    # rather than ignoring, so this gate is what keeps a Synthyra arm from erroring out --
    # and it is why such an arm quietly trains on sdpa instead.
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
