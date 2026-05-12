"""Helpers for using StaticEmbedding models as CachedGIST guide encoders."""

from __future__ import annotations

import time
from pathlib import Path
from types import MethodType
from typing import Sequence, cast

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

DEFAULT_STATIC_GUIDE_SOURCE = "RaphaelMourad/Mistral-Prot-v1-417M"
DEFAULT_STATIC_GUIDE_DIR = "models/mistral_prot_static_guide"


def _load_embedding_source_model(
    source_model_name: str, device: str
) -> torch.nn.Module:
    """Load a model with accessible input embeddings using small fallback sequence."""
    errors: list[str] = []
    for loader in (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(
                source_model_name,
                trust_remote_code=True,
            )
            if device != "cpu":
                model = model.to(device)
            embedding_layer = model.get_input_embeddings()
            if embedding_layer is None:
                raise ValueError("model does not expose input embeddings")
            return model
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{loader.__name__}: {exc}")
    joined_errors = "; ".join(errors)
    raise RuntimeError(
        f"Could not load input embeddings for {source_model_name}: {joined_errors}"
    )


def patch_static_embedding_for_gist(
    model: SentenceTransformer,
) -> SentenceTransformer:
    """Patch a StaticEmbedding SentenceTransformer for CachedGIST guide use.

    Args:
        model: SentenceTransformer whose first module must be StaticEmbedding.

    Returns:
        The patched SentenceTransformer.

    Raises:
        ValueError: If the first module is not StaticEmbedding or no raw tokenizer
            backend is available.
    """
    module = model._first_module()
    if not isinstance(module, StaticEmbedding):
        raise ValueError(
            "The first SentenceTransformer module must be StaticEmbedding."
        )

    raw_tokenizer = getattr(module, "_gist_raw_tokenizer", None)
    if raw_tokenizer is None:
        if isinstance(module.tokenizer, Tokenizer):
            raw_tokenizer = module.tokenizer
        elif isinstance(module.tokenizer, PreTrainedTokenizerFast):
            raw_tokenizer = getattr(module.tokenizer, "_tokenizer", None)
        else:
            raw_tokenizer = None

    if not isinstance(raw_tokenizer, Tokenizer):
        raise ValueError(
            "StaticEmbedding guide patch requires a raw tokenizers.Tokenizer."
        )

    fast_tokenizer = module.tokenizer
    if not isinstance(fast_tokenizer, PreTrainedTokenizerFast):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    if not hasattr(fast_tokenizer, "vocab"):
        fast_tokenizer.vocab = fast_tokenizer.get_vocab()

    module._gist_raw_tokenizer = raw_tokenizer
    module.tokenizer = fast_tokenizer

    def tokenize(
        self: StaticEmbedding, texts: list[str], **_: object
    ) -> dict[str, torch.Tensor]:
        raw = cast(Tokenizer, getattr(self, "_gist_raw_tokenizer"))
        encodings = raw.encode_batch(list(texts), add_special_tokens=False)

        input_ids: list[int] = []
        offsets: list[int] = []
        for encoding in encodings:
            offsets.append(len(input_ids))
            input_ids.extend(encoding.ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
        }

    module.tokenize = MethodType(tokenize, module)
    return model


def build_static_embedding_model(
    source_model_name: str,
    device: str = "cpu",
) -> SentenceTransformer:
    """Build a StaticEmbedding guide model from a Hugging Face checkpoint.

    Args:
        source_model_name: Source model used for tokenizer and embedding weights.
        device: Device used while loading the source checkpoint.

    Returns:
        A patched StaticEmbedding SentenceTransformer guide model.

    Raises:
        ValueError: If a fast tokenizer backend is unavailable or the source model
            has no input embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if not isinstance(raw_tokenizer, Tokenizer):
        raise ValueError("Source tokenizer must provide a fast tokenizers.Tokenizer.")

    source_model = _load_embedding_source_model(source_model_name, device=device)
    embedding_layer = source_model.get_input_embeddings()
    if embedding_layer is None:
        raise ValueError(f"Source model {source_model_name} has no input embeddings.")
    embedding_weights = embedding_layer.weight.detach().cpu().clone()

    static_embedding = StaticEmbedding(
        raw_tokenizer,
        embedding_weights=embedding_weights,
        base_model=source_model_name,
    )
    model = SentenceTransformer(modules=[static_embedding], device="cpu")

    del source_model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return patch_static_embedding_for_gist(model)


def save_static_embedding_model(model: SentenceTransformer, output_path: str) -> str:
    """Save a StaticEmbedding guide model and return the resolved directory path.

    Args:
        model: Guide model to save.
        output_path: Directory where the SentenceTransformer model should be saved.

    Returns:
        Resolved output directory path.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    module = model._first_module()
    if isinstance(module, StaticEmbedding) and hasattr(module, "_gist_raw_tokenizer"):
        fast_tokenizer = module.tokenizer
        module.tokenizer = cast(Tokenizer, module._gist_raw_tokenizer)
        try:
            model.save(str(destination))
        finally:
            module.tokenizer = fast_tokenizer
    else:
        model.save(str(destination))

    return str(destination)


def load_static_guide_model(
    model_name_or_path: str,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load and patch a saved StaticEmbedding guide model.

    Args:
        model_name_or_path: Local path or model identifier to load.
        device: Device where the SentenceTransformer wrapper should run.

    Returns:
        A patched StaticEmbedding SentenceTransformer guide model.
    """
    model = SentenceTransformer(
        model_name_or_path,
        trust_remote_code=True,
        device=device,
    )
    return patch_static_embedding_for_gist(model)


def benchmark_guide_encode_speed(
    guide_model: SentenceTransformer,
    sequences: Sequence[str],
    repeats: int = 3,
) -> dict[str, float]:
    """Benchmark guide-model encode speed on a fixed sequence batch.

    Args:
        guide_model: Guide model to benchmark.
        sequences: Protein sequences to encode.
        repeats: Number of repeated encode runs.

    Returns:
        Timing summary with mean seconds, best seconds, and sequence count.

    Raises:
        ValueError: If no sequences are provided or repeats is less than 1.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    sequence_list = list(sequences)
    if not sequence_list:
        raise ValueError("sequences must not be empty")

    timings: list[float] = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        guide_model.encode(
            sequence_list,
            batch_size=len(sequence_list),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        timings.append(time.perf_counter() - start_time)

    return {
        "mean_seconds": sum(timings) / len(timings),
        "best_seconds": min(timings),
        "num_sequences": float(len(sequence_list)),
    }
