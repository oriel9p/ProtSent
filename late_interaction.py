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
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ci import boot_ci, per_query_metrics  # noqa: E402
from sentence_transformers import MultiVectorEncoder, SentenceTransformer  # noqa: E402
from sentence_transformers.base.modules import Dense, Normalize  # noqa: E402
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask  # noqa: E402

SPECIAL_TOKENS = ("<cls>", "<eos>")  # ESM-2 / FastPLM tokenizers; <pad> is already masked
SCOPE_LEVELS: Dict[str, int] = {"fold": 2, "superfamily": 3, "family": 4}
KS: Tuple[int, ...] = (1, 10, 30)


# --------------------------------------------------------------------------- models
def build_multivector_encoder(
    model_name_or_path: str,
    *,
    proj_dim: int = 64,
    max_seq_length: int = 512,
    device: Optional[str] = None,
) -> Tuple[MultiVectorEncoder, SentenceTransformer]:
    """Return ``(mve, dense_st)`` sharing one backbone Transformer module.

    ``dense_st`` is the ordinary ``[Transformer, mean Pooling]`` SentenceTransformer
    from ``protein_pipeline.load_model_for_training`` (max_seq_length enforced,
    ESM/FastPLM patches applied). ``mve`` reuses ``dense_st[0]`` in place.
    """
    from protein_pipeline import load_model_for_training

    dense_st = load_model_for_training(
        model_name_or_path, max_seq_length=max_seq_length, device=device, pooling_mode="mean"
    )
    transformer = dense_st[0]
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
    modules += [
        MultiVectorMask(skiplist_words=list(SPECIAL_TOKENS), skiplist_tasks=["query", "document"]),
        Normalize(module_input_name="token_embeddings"),
    ]
    mve = MultiVectorEncoder(modules=modules, device=device)
    return mve, dense_st


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


def save_dense_view(mve: MultiVectorEncoder, pooling: torch.nn.Module, out_dir: str) -> str:
    """Export ``[mve[0] (trained backbone), mean Pooling]`` as a normal ProtSent model."""
    _sync_token_dropout_config(mve[0].auto_model)
    dense = SentenceTransformer(modules=[mve[0], pooling], device=str(mve.device))
    dense.save_pretrained(out_dir)
    make_loadable(out_dir)
    return out_dir


def save_late_and_dense(
    mve: MultiVectorEncoder, pooling: torch.nn.Module, out_dir: str
) -> Tuple[str, str]:
    late_dir, dense_dir = os.path.join(out_dir, "late"), os.path.join(out_dir, "dense_view")
    _sync_token_dropout_config(mve[0].auto_model)
    mve.save_pretrained(late_dir)
    make_loadable(late_dir)
    save_dense_view(mve, pooling, dense_dir)
    return late_dir, dense_dir


def backbone_and_projection_params(mve: MultiVectorEncoder):
    """(backbone params, projection params) for two-LR optimisers."""
    proj = [p for m in mve if isinstance(m, Dense) for p in m.parameters()]
    proj_ids = {id(p) for p in proj}
    backbone = [p for p in mve.parameters() if id(p) not in proj_ids]
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
    docs = mve.encode_document(list(seqs), batch_size=batch_size, show_progress_bar=False)
    qs = docs if queries is None else mve.encode_query(list(queries), batch_size=batch_size, show_progress_bar=False)
    with torch.no_grad():
        scores = mve.similarity(qs, docs, chunk_elements=chunk_elements)
    return scores.float().cpu().numpy()


def cosine_matrix(st: SentenceTransformer, seqs: Sequence[str], *, batch_size: int = 64,
                  queries: Optional[Sequence[str]] = None) -> np.ndarray:
    docs = st.encode(list(seqs), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    qs = docs if queries is None else st.encode(list(queries), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return qs @ docs.T


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
    ks: Tuple[int, ...] = KS,
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
        labels = scope_labels(families, level)
        pq = per_query_metrics(ranking, labels)
        elig = pq["eligible"]
        rel = labels[ranking] == labels[:, None]
        row = {"model": model, "scoring": scoring, "level": level,
               "n_queries": int(len(labels)), "n_eligible_queries": int(elig.sum()),
               "runtime_s": runtime_s}
        for k in ks:
            hit = rel[:, :k].any(axis=1)
            row[f"Recall@{k}"] = float(hit.mean())
            row[f"eligible_Recall@{k}"] = float(hit[elig].mean()) if elig.any() else float("nan")
            pq[f"hit{k}"] = hit.astype(float)
        row["MAP"] = float(pq["ap"].mean())
        row["eligible_MAP"] = float(pq["ap"][elig].mean()) if elig.any() else float("nan")
        if n_boot and elig.any():
            for name in ("hit1", "hit10", "ap"):
                _, lo, hi = boot_ci(pq[name][elig], n_boot=n_boot, seed=seed)
                row[f"eligible_{name}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
        per_query[level] = pq
    return rows, per_query


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


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
