#!/usr/bin/env python3
"""
Protein-SBERT Pipeline
=====================================

A modular pipeline for downloading data and training Protein Sentence Transformers (Protein-SBERT)
using Contrastive Learning and the sentence-transformers library.

## Key Features
1.  **Hierarchical Hard Negatives**:
    -   Implements "Clade-Aware" sorting (Clan -> Family) for Pfam.
    -   Ensures structurally similar but functionally distinct proteins (Hard Negatives)
        appear in the same batch, forcing the model to learn fine-grained evolutionary signals.

2.  **Native Multi-Dataset Training**:
    -   Uses sentence-transformers' native `dict[str, Dataset]` multi-dataset support
        with `MultiDatasetBatchSamplers.ROUND_ROBIN` for balanced interleaving.
    -   Each parquet file becomes a separate Dataset.

3.  **Canonical `group_id` Column**:
    -   ETL (`data_prep.py`) adds a `group_id` column to all parquets, unifying
        pfam (`family_id`) and afdb (`cluster_id` = Foldseek structural cluster representative) schemas.
    -   Training code uses `group_id` for pair generation across all datasets.

4.  **Robust ESM2 & Multi-GPU Support**:
    -   **Automatic Tokenizer Shimming** (`_ensure_tokenizer_vocab_attr`):
        Fixes missing `.vocab` attribute on HuggingFace's `EsmTokenizer` by injecting
        `tokenizer.vocab = tokenizer.get_vocab()` before `CachedGISTEmbedLoss` initialization.
    -   **Rotary Embedding Autograd Safety** (`RotaryCacheSafetyCallback`):
        ESM2 rotary caches become inference-mode tensors when `model.encode()` runs
        under `torch.inference_mode()` (e.g., during evaluation). This causes
        `RuntimeError: Inference tensors cannot be saved for backward` on the next
        training step. The callback is available but **not enabled by default** since
        the training loop uses `torch.no_grad()` (safe) and no evaluator is configured.
        Enable it manually if adding mid-training evaluation.


## Discovered Best Practices (ESM2 150M)

### Stage 1: Triplet Loss (Hierarchical Hard Negatives)
- Batch size: 16 (per GPU)
- Gradient accumulation: 2 steps
- Max map rows: 800,000 pairs

- Key flags: `--loss_mode triplet --triplet_use_group_id --batch_sampler group_by_label`

### Stage 2: Cached GIST Fine-tuning (Guide Model Contrast)
- Batch size: 96 (per GPU)
- Mini-batch size: 32 (cached embeddings)
- Guide model: facebook/esm2_t6_8M_UR50D (default frozen transformer guide)
- Key flags: `--loss_mode cached_gist --batch_size 96 --mnrl_mini_batch_size 32`
- Guide path options: `--gist_guide_model` for the default transformer guide or
    `--gist_static_guide` for a local `StaticEmbedding` guide.
- One-time local guide conversion: `python tools/convert_mistral_prot_static.py`
    which writes to `models/mistral_prot_static_guide` by default.
- Static-guide caveats: only the frozen guide changes, training still requires
    CUDA, `--gist_static_guide_device cuda` is now the default for faster guide
    inference (set `cpu` only when you intentionally want offload), tiny smokes
    can be degenerate, and the
    static guide should be treated as a possibly faster ablation rather than a stronger
    teacher.

## Usage Examples

### 1. Data Preparation (ETL)
python data_prep.py --dataset pfam
python data_prep.py --dataset afdb --limit_gb 5

### 2. Training (Single or Multi-File)
python protein_pipeline.py train \\
    --files data/pfam_sorted.parquet \\
    --model facebook/esm2_t12_35M_UR50D \\
    --batch_size 64 \\
    --run_name sbert_pfam

# Multi-file: PFAM + AFDB together (round_robin interleaving)
python protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --loss_mode cached_mnrl \\
    --batch_size 256 \\
    --run_name sbert_pfam_afdb

### 3. Training (Multi-GPU / DDP) - RECOMMENDED
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision bf16 --num_processes 2 protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --batch_size 64 \\
    --run_name ddp_run_v1

### 4. Triplet Loss (PFAM, label-aware batching) - OPTIMIZED FOR ESM2 150M
python protein_pipeline.py train --files data/pfam_sorted.parquet \\
    --model facebook/esm2_t30_150M_UR50D --batch_size 16 \\
    --gradient_accumulation_steps 2 --loss_mode triplet \\
    --batch_sampler group_by_label --max_map_rows 800000 --run_name pfam_triplet

### 5. Cached GIST (Pair-Based, Fine-tuning) - OPTIMIZED FOR ESM2 150M
python protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --model facebook/esm2_t30_150M_UR50D \\
    --batch_size 96 --mnrl_mini_batch_size 32 --loss_mode cached_gist \\
    --gist_guide_model facebook/esm2_t6_8M_UR50D \\
    --max_map_rows 1000000 --run_name esm150_cached_gist

# Optional static-guide variant (guide only; trainable model still runs on CUDA)
python protein_pipeline.py train \
    --files data/pfam_sorted.parquet \
    --model facebook/esm2_t6_8M_UR50D \
    --batch_size 4 --mnrl_mini_batch_size 2 --loss_mode cached_gist \
    --gist_static_guide models/mistral_prot_static_guide \
    --gist_static_guide_device cuda \
    --max_map_rows 32 --max_steps 2 --max_seq_length 128 \
    --learning_rate 1e-5 --no_resume \
    --run_name smoke_cached_gist_static_guide

### 6. Quick Smoke Tests (fast)
python protein_pipeline.py train --files data/pfam_sorted.parquet \\
    --batch_size 64 --loss_mode triplet --batch_sampler group_by_label \\
    --max_map_rows 50000 --fast --no_resume --run_name smoke_pfam_triplet

python protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --batch_size 64 --loss_mode cached_mnrl \\
    --fast --no_resume --run_name smoke_multi_mnrl

### 7. Recommended Scalable Recipe (Two-Stage Training)
# Stage A: Triplet loss on PFAM with hierarchical hard negatives
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \\
    protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --model facebook/esm2_t30_150M_UR50D \\
    --batch_size 16 --gradient_accumulation_steps 2 \\
    --loss_mode triplet --triplet_use_group_id --batch_sampler group_by_label \\
    --max_map_rows 800000 --run_name esm150_stage1

# Stage B: Cached GIST fine-tuning with 8M guide model
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \\
    protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet \\
    --model models/esm150_stage1/final \\
    --batch_size 96 --mnrl_mini_batch_size 32 \\
    --loss_mode cached_gist --gist_guide_model facebook/esm2_t6_8M_UR50D \\
    --max_map_rows 1000000 --run_name esm150_stage2

### 8. STRING-DB PPI Pretraining
python protein_pipeline.py train \\
    --files data/stringdb/stringdb_train.parquet \\
    --model facebook/esm2_t30_150M_UR50D \\
    --loss_mode cached_mnrl \\
    --run_name ppi_sbert

# Multi-file: PPI + PFAM + AFDB together (round_robin interleaving)
python protein_pipeline.py train \\
    --files data/pfam_sorted.parquet data/afdb_sorted.parquet data/stringdb/stringdb_train.parquet \\
    --loss_mode cached_mnrl \\
    --batch_size 256 \\
    --run_name pfam_afdb_ppi_combined

## For detailed run instructions

See RUN_INSTRUCTIONS.md for copy-paste commands, system requirements, and benchmarks.
"""

import argparse
import faulthandler
import json
import logging
import multiprocessing
import os
import random
import signal
import sys
import time
from collections import Counter
from glob import glob
from itertools import combinations
from typing import Any, Iterable, List, Optional, Tuple, cast

# Python 3.14 changed the default multiprocessing start method from 'fork' to
# 'forkserver'. CUDA cannot reinitialize in forked subprocesses, causing
# "Cannot re-initialize CUDA in forked subprocess" when DataLoader workers > 0.
# Force 'spawn' start method before any CUDA initialization.
if sys.version_info >= (3, 14):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from datasets import Dataset, Features, Value
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer import modules as models
from sentence_transformers.sentence_transformer.data_collator import (
    SentenceTransformerDataCollator,
)

try:
    from sentence_transformers.base.sampler import DefaultBatchSampler
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    BatchSamplers: Optional[Any] = None
    DefaultBatchSampler = object
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    TrainerCallback,
)

# tempfile used only for ESMplusplus tokenizer conversion
import tempfile
from functools import partial
from torch.utils.data import BatchSampler

from model_utils import (
    AMPLIFYWrapper,
    DPLM2Wrapper,
    ESMplusplusWrapper,
    FastPLMESM2Wrapper,
    ProfluentE1Wrapper,
    apply_esmplusplus_compat_patch,
    detect_model_type,
    disable_esm2_token_dropout,
    force_sdpa_backend,
    patch_unknown_residue_tokens,
    from_pretrained_with_flash,
    get_torch_compile_settings,
    uses_hf_esm_rotary_embeddings,
)
from gor_loss import LossWithGOR, SubsampledLoss
from static_guide import DEFAULT_STATIC_GUIDE_DIR, load_static_guide_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_TRACEBACK_SIGNAL_HANDLE: Any | None = None

apply_esmplusplus_compat_patch()


def _pair_length(max_seq_length: int, *seqs: str | None) -> int:
    return min(max((len(seq or "") for seq in seqs), default=0), max_seq_length)


class LengthBucketBatchSampler(DefaultBatchSampler):
    """Shuffle examples within clipped-length buckets before forming batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        bucket_size: int = 64,
        length_column: str = "label",
    ) -> None:
        BatchSampler.__init__(self, dataset, batch_size=batch_size, drop_last=drop_last)
        self.valid_label_columns = valid_label_columns
        self.generator = generator
        self.seed = seed
        self.epoch = 0
        self.bucket_size = max(1, int(bucket_size))
        self.length_column = length_column
        if length_column not in dataset.column_names:
            raise ValueError(
                f"LengthBucketBatchSampler requires a '{length_column}' column"
            )

        # Fetch the entire column at once as a numpy array using C/Arrow optimizations,
        # which is 10000x faster than iterating ds[col] in python loops.
        import numpy as np
        
        lengths_arr = None
        if hasattr(dataset, "_indices") and dataset._indices is not None:
            try:
                indices = dataset._indices.column(0).to_numpy()
                physical_lengths = dataset.data.column(length_column).to_numpy()
                lengths_arr = physical_lengths[indices]
            except Exception as e:
                logger.warning(f"Could not use NumPy fast path for index mapping, falling back: {e}")
        
        if lengths_arr is None:
            try:
                lengths_arr = dataset.data.column(length_column).to_numpy()
            except Exception:
                try:
                    lengths_arr = dataset.select_columns([length_column]).to_pandas()[length_column].values
                except Exception as e:
                    raise ValueError(f"Could not retrieve length column '{length_column}' from dataset: {e}")

        if len(lengths_arr) == 0:
            buckets = {}
        else:
            buckets_np = lengths_arr // self.bucket_size
            sorted_indices = np.argsort(buckets_np)
            sorted_buckets = buckets_np[sorted_indices]
            unique_buckets, split_indices = np.unique(sorted_buckets, return_index=True)
            split_arrays = np.split(sorted_indices, split_indices[1:])
            # Keep as numpy arrays directly instead of converting to python list
            buckets = {
                int(b): idxs
                for b, idxs in zip(unique_buckets, split_arrays)
            }
        self.buckets = buckets
        self._len = 0
        for indices in buckets.values():
            full, remainder = divmod(len(indices), batch_size)
            self._len += full + (0 if drop_last or remainder == 0 else 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import numpy as np
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            
            num_batches = len(shuffled) // self.batch_size
            if num_batches > 0:
                complete_part = shuffled[:num_batches * self.batch_size]
                batches.extend(
                    complete_part.reshape(num_batches, self.batch_size).tolist()
                )
            if not self.drop_last and len(shuffled) % self.batch_size != 0:
                remainder_batch = shuffled[num_batches * self.batch_size:].tolist()
                batches.append(remainder_batch)
                
        # Now shuffle the batches order
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return self._len


def _register_signal_tracebacks(output_dir: str, local_rank: int) -> None:
    """Register SIGUSR1 traceback dumps to a per-rank debug file."""
    global _TRACEBACK_SIGNAL_HANDLE

    trace_dir = os.path.join(output_dir, "debug_traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"rank_{local_rank}_sigusr1.log")

    if _TRACEBACK_SIGNAL_HANDLE is not None:
        faulthandler.unregister(signal.SIGUSR1)
        _TRACEBACK_SIGNAL_HANDLE.close()

    _TRACEBACK_SIGNAL_HANDLE = open(trace_path, "a", encoding="utf-8")
    faulthandler.register(
        signal.SIGUSR1,
        file=_TRACEBACK_SIGNAL_HANDLE,
        all_threads=True,
    )
    logger.info("🪵 SIGUSR1 traceback file: %s", trace_path)


def _ensure_trainable_rotary_caches(model: nn.Module) -> None:
    """Clone ESM2 rotary caches out of inference mode for autograd safety.

    ESM2's ``RotaryEmbedding`` caches ``_cos_cached`` / ``_sin_cached`` tensors.
    When a forward pass runs under ``torch.inference_mode()`` (e.g., via
    ``SentenceTransformer.encode()``), these caches become inference-mode tensors.
    A subsequent training forward+backward then crashes with:
    ``RuntimeError: Inference tensors cannot be saved for backward``.

    ``torch.no_grad()`` does NOT cause this — only ``torch.inference_mode()``.
    This function is called at model load time (once) and optionally via
    ``RotaryCacheSafetyCallback`` when mid-training evaluation is used.
    """
    if not uses_hf_esm_rotary_embeddings(model):
        return

    patched = 0
    for module in model.modules():
        for attr in ("_cos_cached", "_sin_cached"):
            if not hasattr(module, attr):
                continue
            tensor = getattr(module, attr)
            if not isinstance(tensor, torch.Tensor):
                continue
            is_inference = bool(
                hasattr(tensor, "is_inference") and tensor.is_inference()
            )
            if is_inference:
                setattr(module, attr, tensor.clone())
                patched += 1
    if patched > 0:
        logger.info(
            "   🔧 Cloned %d rotary cache tensor(s) out of inference mode", patched
        )


def _is_distributed_launcher_active(world_size: int) -> bool:
    """Return True when launched via DDP tooling (accelerate/torchrun)."""
    if world_size <= 1:
        return False
    distributed_env_keys = (
        "LOCAL_RANK",
        "RANK",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_USE_DISTRIBUTED",
    )
    return any(key in os.environ for key in distributed_env_keys)


def _progress_bars_enabled(local_rank: Optional[int] = None) -> bool:
    """Return whether tqdm-style progress bars should be shown.

    Resolution order:
    1) `PROTEIN_PROGRESS_BARS=on|off` forces behavior.
    2) `auto` (default) enables bars on rank 0.
    """
    raw = os.environ.get("PROTEIN_PROGRESS_BARS", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    rank = local_rank
    if rank is None:
        try:
            rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            rank = 0
    if rank > 0:
        return False

    return True


def _configure_tqdm_defaults(progress_enabled: bool) -> float:
    """Configure tqdm update cadence to reduce log spam.

    Returns the effective min-interval in seconds.
    """
    if not progress_enabled:
        return 0.0

    raw = os.environ.get("PROTEIN_PROGRESS_MIN_INTERVAL", "5.0").strip()
    try:
        interval = float(raw)
    except ValueError:
        interval = 5.0
    interval = max(0.5, interval)

    # Apply sane defaults globally for tqdm-based progress emitters.
    os.environ["TQDM_MININTERVAL"] = f"{interval}"
    os.environ.setdefault("TQDM_MINITERS", "1")
    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "1")

    # Preserve in-place behavior on real terminals; no-op in piped logs.
    if sys.stderr.isatty() and os.environ.get("TERM", "") != "dumb":
        os.environ.setdefault("TQDM_POSITION", "0")

    return interval


def _compile_sentence_transformer_backbone(model: SentenceTransformer) -> bool:
    """Compile only the inner HF module to keep SentenceTransformer iterable."""
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this PyTorch build")
        return False

    if not hasattr(model, "_modules") or not model._modules:
        logger.warning("Skipping torch.compile: SentenceTransformer has no modules")
        return False

    first_module = list(model._modules.values())[0]
    backbone_attr = "model" if hasattr(first_module, "model") else "auto_model"
    backbone = getattr(first_module, backbone_attr, None)
    if backbone is None:
        logger.warning(
            "Skipping torch.compile: first SentenceTransformer module has no model/auto_model"
        )
        return False

    if isinstance(backbone, nn.DataParallel):
        logger.warning("Skipping torch.compile: DataParallel is not thread-safe")
        return False

    compile_kwargs, needs_unspec_int = get_torch_compile_settings(backbone)
    dynamic = bool(compile_kwargs.get("dynamic", False))
    backend = str(compile_kwargs.get("backend", "default") or "default")
    mode = str(compile_kwargs.get("mode", "default") or "default")

    if (
        needs_unspec_int
        and hasattr(torch, "_dynamo")
        and hasattr(torch._dynamo, "config")
    ):
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        logger.info(
            "⚙️  Enabled torch._dynamo.config.allow_unspec_int_on_nn_module for HF ESM rotary caches"
        )

    try:
        setattr(first_module, backbone_attr, torch.compile(backbone, **compile_kwargs))
    except Exception as e:
        logger.warning(
            "torch.compile failed (backend=%s, dynamic=%s, mode=%s): %s",
            backend,
            dynamic,
            mode,
            e,
        )
        return False

    logger.info(
        "⚡ torch.compile enabled (backend=%s, dynamic=%s, mode=%s)",
        backend,
        dynamic,
        mode,
    )
    return True


def _ensure_tokenizer_vocab_attr(st_model: SentenceTransformer) -> None:
    """Ensure `st_model.tokenizer.vocab` exists for losses that require it.

    Some HF tokenizers (e.g., EsmTokenizer) expose `get_vocab()` but no `.vocab`
    attribute. SentenceTransformers' CachedGISTEmbedLoss expects `.vocab`.
    """
    tokenizer = getattr(st_model, "tokenizer", None)
    if tokenizer is None:
        return
    if hasattr(tokenizer, "vocab"):
        return
    if hasattr(tokenizer, "get_vocab"):
        try:
            tokenizer.vocab = tokenizer.get_vocab()
            logger.info("   🔧 Added tokenizer.vocab compatibility shim")
        except Exception as exc:
            logger.warning("   ⚠️ Could not attach tokenizer.vocab shim: %s", exc)


def _patch_cached_gist_guide_preprocess(st_model: SentenceTransformer) -> None:
    """Drop ST 5.6 non-tensor preprocess metadata before CachedGIST moves tensors.

    sentence-transformers 5.6 may return fields such as ``modality`` from
    ``SentenceTransformer.preprocess``. ``CachedGISTEmbedLoss`` retokenizes with
    the guide model and then blindly calls ``.to(device)`` on every returned
    value, so string metadata crashes the cached GIST path. The transformer
    module defaults missing modality to text, so filtering non-tensors is safe
    for protein text guides.
    """
    if getattr(st_model, "_protsent_cached_gist_preprocess_patch", False):
        return

    original_preprocess = st_model.preprocess

    def preprocess_without_metadata(*args, **kwargs):
        features = original_preprocess(*args, **kwargs)
        if isinstance(features, dict):
            return {
                key: value for key, value in features.items() if isinstance(value, torch.Tensor)
            }
        return features

    st_model.preprocess = preprocess_without_metadata
    st_model._protsent_cached_gist_preprocess_patch = True


def _patch_cached_gist_embed_minibatch(loss_obj: nn.Module) -> None:
    """Allow CachedGISTEmbedLoss to retokenize ST 5.6 preprocess metadata.

    In sentence-transformers 5.6, ``SentenceTransformer.preprocess`` includes
    non-tensor fields such as ``modality``. ``CachedGISTEmbedLoss`` retokenizes
    with the guide model and then applies ``.to(self.guide.device)`` to every
    field, which crashes on strings. Patch the loss instance rather than
    site-packages so normal environments can keep using upstream behavior.
    """
    if getattr(loss_obj, "_protsent_cached_gist_embed_patch", False):
        return

    try:
        from contextlib import nullcontext
        from types import MethodType

        from sentence_transformers.sentence_transformer.losses.cached_gist_embed import (
            RandContext,
            _create_minibatch,
        )
    except Exception as exc:
        logger.warning("Could not patch CachedGISTEmbedLoss metadata handling: %s", exc)
        return

    def embed_minibatch(
        self,
        sentence_feature: dict[str, torch.Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state=None,
    ):
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        with random_state_context:
            with grad_context():
                if copy_random_state:
                    random_state = RandContext(
                        *[
                            value
                            for value in sentence_feature_minibatch.values()
                            if isinstance(value, torch.Tensor)
                        ]
                    )
                else:
                    random_state = None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
            with torch.no_grad():
                guide_features = sentence_feature_minibatch
                if self.must_retokenize:
                    decoded = self.tokenizer.batch_decode(
                        sentence_feature_minibatch["input_ids"],
                        skip_special_tokens=True,
                    )
                    guide_features = self.guide.preprocess(decoded)
                guide_features = {
                    key: value.to(self.guide.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in guide_features.items()
                }
                guide_reps = self.guide(guide_features)["sentence_embedding"]

        return reps, guide_reps, random_state

    loss_obj.embed_minibatch = MethodType(embed_minibatch, loss_obj)
    loss_obj._protsent_cached_gist_embed_patch = True


def _load_gist_guide_model(args: argparse.Namespace) -> SentenceTransformer:
    """Load the guide model used by CachedGISTEmbedLoss."""
    if args.gist_static_guide:
        logger.info(
            "🔍 Loading static GIST guide model: %s (device=%s)",
            args.gist_static_guide,
            args.gist_static_guide_device,
        )
        return load_static_guide_model(
            args.gist_static_guide,
            device=args.gist_static_guide_device,
        )

    logger.info("🔍 Loading GIST guide model: %s", args.gist_guide_model)
    return load_model_for_training(
        args.gist_guide_model,
        max_seq_length=args.max_seq_length,
    )


def _euclidean_similarity(
    embeddings1: torch.Tensor, embeddings2: torch.Tensor
) -> torch.Tensor:
    """Compute negative euclidean distance as similarity (higher = more similar).

    For use with CachedMultipleNegativesRankingLoss where higher values indicate
    greater similarity. Euclidean distance is inverted (negated) so that paired
    samples (anchor-positive) produce higher values than non-paired (anchor-negative).

    Args:
        embeddings1: (batch_size, dim) or (batch_size, 1, dim)
        embeddings2: (n_samples, dim) or (1, n_samples, dim)

    Returns:
        (batch_size, n_samples) similarity matrix, range (-inf, 0]
    """
    # Ensure 2D or 3D shapes for cdist
    if embeddings1.dim() == 3:
        embeddings1 = embeddings1.squeeze(1)  # (batch, dim)
    if embeddings2.dim() == 3:
        embeddings2 = embeddings2.squeeze(1)  # (n_samples, dim)

    # Compute euclidean distance and negate for similarity
    dist = torch.cdist(embeddings1, embeddings2, p=2.0)  # (batch, n_samples)
    return -dist  # Higher (less negative) = more similar


def _build_pooling_module(
    word_embedding_dimension: int,
    pooling_mode: str,
    pooling_activation: str,
) -> nn.Module:
    """Build the requested pooling module for sentence embeddings.

    Args:
        word_embedding_dimension: Hidden size of token embeddings.
        pooling_mode: Pooling variant to instantiate.
        pooling_activation: Activation for contextual attention pooling.

    Returns:
        A pooling module compatible with SentenceTransformer.

    Raises:
        ValueError: If the pooling mode is unsupported.
    """
    if pooling_mode == "mean":
        return models.Pooling(
            word_embedding_dimension,
            pooling_mode="mean",
        )

    if pooling_mode == "linear_attention":
        from attention_pooling import LinearAttentionPooling

        return LinearAttentionPooling(word_embedding_dimension)

    if pooling_mode == "contextual_attention":
        from attention_pooling import ContextualAttentionPooling

        return ContextualAttentionPooling(
            word_embedding_dimension,
            activation=pooling_activation,
        )

    if pooling_mode == "gem":
        from attention_pooling import GeneralizedMeanPooling

        return GeneralizedMeanPooling(word_embedding_dimension)

    raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")


def _attach_backbone(word_embedding_model, hf_model, tokenizer) -> None:
    """Swap the backbone and tokenizer into a SentenceTransformers Transformer.

    These modules are built from a small stand-in checkpoint and then have the
    real model grafted in. sentence-transformers >=5 turned both ``auto_model``
    and ``tokenizer`` into read-only properties backed by ``.model`` and
    ``.processor``.

    ``auto_model`` is the dangerous one: assigning to it does NOT raise, because
    ``nn.Module.__setattr__`` diverts any Module value into ``self._modules``,
    while every *read* still goes through the property and returns the stand-in.
    The result is a model that loads, encodes, and trains — as the 8M stand-in,
    at its hidden size, silently. Assign to the backing attributes instead, and
    verify the swap took.
    """
    if isinstance(getattr(type(word_embedding_model), "auto_model", None), property):
        word_embedding_model.model = hf_model
    else:
        word_embedding_model.auto_model = hf_model

    # 'J' (Leu/Ile ambiguity) is absent from the ESM2 vocabulary and FastPLM
    # raises KeyError rather than using unk_token, killing a dataloader worker
    # and with it the whole DDP run. Map unknown residues to 'X' here, the one
    # point every model branch routes its tokenizer through.
    patch_unknown_residue_tokens(tokenizer)

    if isinstance(getattr(type(word_embedding_model), "tokenizer", None), property):
        word_embedding_model.processor = cast(Any, tokenizer)
    else:
        word_embedding_model.tokenizer = cast(Any, tokenizer)

    if word_embedding_model.auto_model is not hf_model:
        raise RuntimeError(
            "Failed to graft the backbone into the SentenceTransformers Transformer "
            f"module: auto_model is still {type(word_embedding_model.auto_model).__name__}. "
            "sentence-transformers has changed its module layout again."
        )


def load_model_for_training(
    model_name: str,
    max_seq_length: int = 512,
    device: Optional[str] = None,
    pooling_mode: str = "mean",
    pooling_activation: str = "tanh",
) -> SentenceTransformer:
    """Load a model for training and enforce ``max_seq_length``.

    The enforcement is not redundant. Every branch below hands
    ``max_seq_length`` to ``models.Transformer``, but the custom-code backbones
    then replace that module's tokenizer with their own, and FastPLM's ships
    ``model_max_length`` = 1e24. sentence-transformers reads the truncation
    limit off the tokenizer, so the requested value was silently discarded and
    batches were padded to the longest sequence present — measured at 1,561
    tokens on a 512-pair batch, which is what turns a 6 GiB contrastive step
    into a 150 GiB one. Set it once here, after the model is fully assembled,
    so no branch can drop it.
    """
    model = _load_model_for_training(
        model_name,
        max_seq_length=max_seq_length,
        device=device,
        pooling_mode=pooling_mode,
        pooling_activation=pooling_activation,
    )
    _enforce_max_seq_length(model, max_seq_length)
    return model


def _enforce_max_seq_length(model: SentenceTransformer, max_seq_length: int) -> None:
    """Apply a truncation limit to an assembled SentenceTransformer.

    Every path that builds a model for *training* must call this, not just the
    fresh-load path: resuming from a checkpoint rebuilds the model with a bare
    ``SentenceTransformer(dir)``, which reads the limit off the checkpoint's own
    tokenizer and so reintroduces the bug for the resumed half of a run.

    Sets the requested value in both directions. Tighten-only was wrong: a
    checkpoint saved after this fix carries a 512 tokenizer, so resuming it at
    --max_seq_length 1024 would have silently stayed at 512.

    Deliberately unclamped. The obvious guard — clamp to the backbone's
    ``max_position_embeddings`` — is worthless for the models trained here: ESM-2
    is rotary, so its declared 1026 bounds nothing, while ESM++/ESM-C omit the
    field entirely and AMPLIFY spells it ``max_length``. A guard that only fires
    where the number is meaningless is worse than none. ST's own
    ``Transformer.max_seq_length`` already applies what bound it can.
    """
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    logger.info("   ✂️  max_seq_length enforced at %s", model.max_seq_length)


def _load_model_for_training(
    model_name: str,
    max_seq_length: int = 512,
    device: Optional[str] = None,
    pooling_mode: str = "mean",
    pooling_activation: str = "tanh",
) -> SentenceTransformer:
    """
    Robustly loads a model for SentenceTransformers training.

    Supports:
    - ESM++ / ESM-C (Synthyra/ESMplusplus_*) - special handling for model.tokenizer
    - FastPLM ESM2 (Synthyra/ESM2-*) - bug-fixed ESM2 re-implementation
    - ModernBERT (requires trust_remote_code)
    - ESM-2 (standard HF model, with token_dropout bug workaround)
    - BERT/RoBERTa (standard HF models)

    The resulting model is fully serializable via model.save().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🏗️  Loading model: {model_name} (device={device})")

    model_type = detect_model_type(model_name)

    if model_type == "amplify":
        logger.info("   ✨ Using AMPLIFY strategy")
        hf_model_raw = from_pretrained_with_flash(AutoModel, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        hf_model = AMPLIFYWrapper(hf_model_raw)
        hidden_size = hf_model_raw.config.hidden_size

        safe_name = "facebook/esm2_t6_8M_UR50D"
        word_embedding_model = models.Transformer(
            model_name_or_path=safe_name, max_seq_length=max_seq_length
        )
        _attach_backbone(word_embedding_model, hf_model, tokenizer)

        pooling_model = _build_pooling_module(
            hidden_size,
            pooling_mode,
            pooling_activation,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        logger.info(f"   ✅ AMPLIFY SentenceTransformer assembled (dim={hidden_size})")
        return model

    if model_type == "fastplm_esm2":
        logger.info("   ✨ Using Synthyra FastPLM ESM2 strategy")

        lm_model = from_pretrained_with_flash(AutoModelForMaskedLM, model_name)

        # Honour PROTSENT_ESMPLUSPLUS_ATTN_BACKEND here too (defaults to
        # kernels_flash). Hardcoding "sdpa" silently disabled flash attention for
        # these Synthyra models regardless of the environment variable.
        force_sdpa_backend(lm_model)

        if hasattr(lm_model, "tokenizer") and lm_model.tokenizer is not None:
            tokenizer = lm_model.tokenizer
            logger.info("   ✅ Using FastPLM ESM2 native tokenizer")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info("   ✅ Using AutoTokenizer for FastPLM ESM2")

        hf_model = FastPLMESM2Wrapper(lm_model)
        hidden_size = lm_model.config.hidden_size

        safe_name = "facebook/esm2_t6_8M_UR50D"
        word_embedding_model = models.Transformer(
            model_name_or_path=safe_name, max_seq_length=max_seq_length
        )
        _attach_backbone(word_embedding_model, hf_model, tokenizer)

        pooling_model = _build_pooling_module(
            hidden_size,
            pooling_mode,
            pooling_activation,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        logger.info(
            f"   ✅ FastPLM ESM2 SentenceTransformer assembled (dim={hidden_size})"
        )
        return model

    if model_type == "dplm2":
        logger.info("   ✨ Using Synthyra DPLM2 strategy")

        # DPLM2 uses AutoModel (not AutoModelForMaskedLM)
        lm_model = from_pretrained_with_flash(AutoModel, model_name)

        # Honour PROTSENT_ESMPLUSPLUS_ATTN_BACKEND here too (defaults to
        # kernels_flash). Hardcoding "sdpa" silently disabled flash attention for
        # these Synthyra models regardless of the environment variable.
        force_sdpa_backend(lm_model)

        if hasattr(lm_model, "tokenizer") and lm_model.tokenizer is not None:
            tokenizer = lm_model.tokenizer
            logger.info("   ✅ Using DPLM2 native tokenizer")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info("   ✅ Using AutoTokenizer for DPLM2")

        hf_model = DPLM2Wrapper(lm_model)
        hidden_size = lm_model.config.hidden_size

        safe_name = "facebook/esm2_t6_8M_UR50D"
        word_embedding_model = models.Transformer(
            model_name_or_path=safe_name, max_seq_length=max_seq_length
        )
        _attach_backbone(word_embedding_model, hf_model, tokenizer)

        pooling_model = _build_pooling_module(
            hidden_size,
            pooling_mode,
            pooling_activation,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        logger.info(f"   ✅ DPLM2 SentenceTransformer assembled (dim={hidden_size})")
        return model

    if model_type == "profluent_e1":
        logger.info("   ✨ Using Synthyra Profluent-E1 strategy")

        lm_model = from_pretrained_with_flash(AutoModelForMaskedLM, model_name)

        # Honour PROTSENT_ESMPLUSPLUS_ATTN_BACKEND here too (defaults to
        # kernels_flash). Hardcoding "sdpa" silently disabled flash attention for
        # these Synthyra models regardless of the environment variable.
        force_sdpa_backend(lm_model)

        # Profluent-E1 may have model.tokenizer or model.prep_tokens
        if hasattr(lm_model, "tokenizer") and lm_model.tokenizer is not None:
            tokenizer = lm_model.tokenizer
            logger.info("   ✅ Using Profluent-E1 native tokenizer")
        else:
            # Fallback to AutoTokenizer if no native tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info("   ✅ Using AutoTokenizer for Profluent-E1")

        hf_model = ProfluentE1Wrapper(lm_model)
        hidden_size = lm_model.config.hidden_size

        safe_name = "facebook/esm2_t6_8M_UR50D"
        word_embedding_model = models.Transformer(
            model_name_or_path=safe_name, max_seq_length=max_seq_length
        )
        _attach_backbone(word_embedding_model, hf_model, tokenizer)

        pooling_model = _build_pooling_module(
            hidden_size,
            pooling_mode,
            pooling_activation,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        logger.info(
            f"   ✅ Profluent-E1 SentenceTransformer assembled (dim={hidden_size})"
        )
        return model

    if model_type == "esmplusplus":
        logger.info("   ✨ Using ESMplusplus/ESM-C strategy")

        lm_model = from_pretrained_with_flash(AutoModelForMaskedLM, model_name)

        if hasattr(lm_model, "tokenizer"):
            native_tokenizer = lm_model.tokenizer
            with tempfile.TemporaryDirectory() as tmp_dir:
                native_tokenizer.save_pretrained(tmp_dir)
                cfg_path = os.path.join(tmp_dir, "tokenizer_config.json")
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                if "tokenizer_class" in cfg:
                    del cfg["tokenizer_class"]
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f)
                tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_dir)
            logger.info("   ✅ Extracted and converted ESMplusplus tokenizer")
        else:
            raise ValueError("ESMplusplus model missing tokenizer attribute")

        force_sdpa_backend(lm_model)

        hf_model = ESMplusplusWrapper(lm_model)
        hidden_size = lm_model.config.hidden_size

        safe_name = "facebook/esm2_t6_8M_UR50D"
        word_embedding_model = models.Transformer(
            model_name_or_path=safe_name, max_seq_length=max_seq_length
        )
        # sentence-transformers >=5.4 renamed Transformer.auto_model -> .model (and
        # forward() dispatches on self.model), and made .tokenizer a read-only
        # property backed by .processor. Assign the new attributes so the ESM-C
        # module is actually used (the old names silently no-op / raise).
        word_embedding_model.model = hf_model
        word_embedding_model.processor = tokenizer

        pooling_model = _build_pooling_module(
            hidden_size,
            pooling_mode,
            pooling_activation,
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        logger.info(f"   ✅ SentenceTransformer assembled (dim={hidden_size})")
        return model

    # Standard loading for ESM-2, ModernBERT, etc.
    logger.info("   ✨ Using standard SentenceTransformer loading")

    word_embedding_model = models.Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )

    hf_tokenizer = cast(Any, word_embedding_model.tokenizer)
    if hf_tokenizer.pad_token is None:
        if hf_tokenizer.eos_token is not None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
            logger.info("   ⚠️  Pad token was None, set to EOS token.")
        else:
            hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            word_embedding_model.auto_model.resize_token_embeddings(len(hf_tokenizer))
            logger.info("   ⚠️  Pad token was None, added '[PAD]' token.")

    pooling_dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = _build_pooling_module(
        pooling_dimension,
        pooling_mode,
        pooling_activation,
    )

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        trust_remote_code=True,
        device=device,
    )
    _ensure_trainable_rotary_caches(word_embedding_model.auto_model)

    # Fix ESM2 token_dropout bug (HuggingFace transformers >=5.x)
    disable_esm2_token_dropout(word_embedding_model.auto_model)

    # Validate tokenizer/model vocabulary consistency
    tokenizer = cast(Any, word_embedding_model.tokenizer)
    embedding_layer = word_embedding_model.auto_model.get_input_embeddings()
    model_vocab_size = embedding_layer.num_embeddings
    tokenizer_vocab_size = len(tokenizer)

    logger.info(
        f"   📊 Tokenizer vocab size: {tokenizer_vocab_size}, Model vocab size: {model_vocab_size}"
    )

    if tokenizer_vocab_size != model_vocab_size:
        logger.warning(
            f"   ⚠️  VOCAB MISMATCH DETECTED! Tokenizer: {tokenizer_vocab_size}, Model: {model_vocab_size}"
        )
        if tokenizer_vocab_size > model_vocab_size:
            logger.info(
                f"   🔧 Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}"
            )
            word_embedding_model.auto_model.resize_token_embeddings(
                tokenizer_vocab_size
            )
            model_vocab_size = tokenizer_vocab_size
        else:
            logger.error(
                "   ❌ Tokenizer vocab is smaller than model! This may indicate wrong tokenizer."
            )

    # Test tokenization
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN"
    try:
        test_tokens = tokenizer(test_seq, return_tensors="pt", truncation=True)
        test_ids = test_tokens["input_ids"][0]
        max_token_id = test_ids.max().item()
        min_token_id = test_ids.min().item()

        logger.info(
            f"   🧪 Test tokenization: min_id={min_token_id}, max_id={max_token_id}, model_vocab={model_vocab_size}"
        )

        if max_token_id >= model_vocab_size:
            raise ValueError(
                f"Tokenizer produced out-of-bounds token ID {max_token_id} >= vocab_size {model_vocab_size}."
            )
        if min_token_id < 0:
            raise ValueError(f"Tokenizer produced negative token ID {min_token_id}.")
        logger.info("   ✅ Tokenization test passed")
    except Exception as e:
        logger.error(f"   ❌ Tokenization test failed: {e}")
        raise

    logger.info(f"   ✅ Model loaded successfully (dim={pooling_dimension})")
    return model


# =============================================================================
# Helper: Time Limit Callback
# =============================================================================
class TimeLimitCallback(TrainerCallback):
    """Stops training after a set number of minutes, saving a checkpoint first.

    Uses only the standard HF Trainer ``control`` flags — no custom collectives
    or synchronisation.  All ranks share the same wall-clock, so they will each
    independently cross the threshold within one step of each other and the
    Trainer handles the rest.
    """

    def __init__(self, max_minutes: int):
        self.max_seconds = max_minutes * 60
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info(f"⏱️  Time limit set: {self.max_seconds / 60:.1f} minutes")

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
        if (time.time() - self.start_time) > self.max_seconds:
            logger.info("🛑 Time limit reached. Saving checkpoint and stopping.")
            control.should_save = True
            control.should_training_stop = True


class RotaryCacheSafetyCallback(TrainerCallback):
    """Periodically ensure ESM2 rotary caches are not inference-mode tensors.

    **When is this needed?**
    Only when an evaluator (e.g., TripletEvaluator) is used during training.
    Evaluators call ``model.encode()`` which is decorated with
    ``@torch.inference_mode()``. This contaminates ESM2's ``_cos_cached`` /
    ``_sin_cached`` rotary tensors, causing ``RuntimeError: Inference tensors
    cannot be saved for backward`` on the next training step.

    ``torch.no_grad()`` (used by CachedGISTEmbedLoss / CachedMNRL) does NOT
    cause this issue — only ``torch.inference_mode()`` does.

    **Not added by default.** Add this callback manually when using evaluators::

        callbacks.append(RotaryCacheSafetyCallback(every_n_steps=25))
    """

    def __init__(self, every_n_steps: int = 25):
        self.every_n_steps = max(1, int(every_n_steps))

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return
        model = kwargs.get("model")
        if model is not None:
            _ensure_trainable_rotary_caches(model)


# =============================================================================
# Noise Collator for SimCSE (mask-based augmentation)
# =============================================================================
class NoisySimCSECollator(SentenceTransformerDataCollator):
    """Applies token masking to sentence_1 after standard collation.

    Used for SimCSE (self-pair contrastive) batches only.
    Only amino acid tokens with attention_mask=1 are eligible.
    """

    def __init__(self, tokenize_fn, tokenizer, mask_rate: float = 0.15, **kwargs):
        super().__init__(tokenize_fn, **kwargs)
        self.mask_rate = mask_rate
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if self.mask_token_id is None:
            vocab = tokenizer.get_vocab()
            self.mask_token_id = vocab.get("<mask>", vocab.get("[MASK]", 32))
        # Detect amino acid token IDs dynamically
        vocab = tokenizer.get_vocab()
        self.aa_ids = frozenset(
            tid
            for token, tid in vocab.items()
            if len(token) == 1 and token.isalpha() and token.isupper()
        )
        logger.info(
            "NoisySimCSECollator: mask_rate=%.2f, mask_token_id=%d, %d AA token IDs",
            self.mask_rate,
            self.mask_token_id,
            len(self.aa_ids),
        )

    def __call__(self, features):
        # Strip the dataset marker before tokenization and only mask true SimCSE
        # batches so non-SimCSE tasks keep lexical inputs untouched.
        simcse_flags = [bool(feature.pop("is_simcse", False)) for feature in features]
        apply_mask = bool(simcse_flags) and all(simcse_flags)
        if any(simcse_flags) and not all(simcse_flags):
            logger.warning(
                "Mixed SimCSE/non-SimCSE samples in one batch; skipping token masking for safety."
            )

        batch = super().__call__(features)
        if self.mask_rate <= 0 or not apply_mask:
            return batch
        if self.mask_token_id is None:
            return batch
        mask_token_id = int(self.mask_token_id)

        # Apply masking to sentence_1_input_ids (the augmented view / positive)
        key = "sentence_1_input_ids"
        if key not in batch:
            return batch

        input_ids = batch[key]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
            batch[key] = input_ids
        attn_key = "sentence_1_attention_mask"
        attn_mask = batch.get(attn_key)
        if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
            attn_mask = torch.tensor(attn_mask)
            batch[attn_key] = attn_mask

        # Build eligibility mask: amino acid tokens with attention_mask=1
        aa_set = self.aa_ids
        eligible = torch.zeros_like(input_ids, dtype=torch.bool)
        for aa_id in aa_set:
            eligible |= input_ids == aa_id
        if attn_mask is not None:
            eligible &= attn_mask == 1

        # Bernoulli masking
        mask_prob = torch.full_like(input_ids, self.mask_rate, dtype=torch.float)
        mask_prob[~eligible] = 0.0
        mask_positions = torch.bernoulli(mask_prob).bool()
        input_ids[mask_positions] = mask_token_id

        return batch


def _resolve_dms_train_batch_size(
    base_batch_size: int,
    dms_batch_size: int,
    mnrl_mini_batch_size: int,
    train_dataset: dict[str, Dataset],
    world_size: int,
    sampler_mode: str,
    drop_last: bool,
) -> int:
    """Resolve per-device training batch size when DMS CoSENT is enabled.

    Uses ``dms_batch_size`` as the primary target. The training CLI resolves
    ``--dms_batch_size 0`` to ``--batch_size`` before calling this helper so
    non-cached contrastive and CoSENT datasets share the intended large batch.
    Direct helper calls retain the historical ``mnrl_mini_batch_size // 2``
    fallback when ``dms_batch_size <= 0``.

    For DDP + round-robin sampling, pick the smallest batch size >= target that
    yields a global round-robin batch count divisible by ``world_size``.
    """
    if base_batch_size <= 0:
        raise ValueError("base_batch_size must be > 0")

    fallback_target = max(1, mnrl_mini_batch_size // 2)
    requested_batch_size = dms_batch_size if dms_batch_size > 0 else fallback_target
    candidate_batch_size = min(base_batch_size, max(1, requested_batch_size))

    resolved_sampler = (sampler_mode or "round_robin").lower()
    if resolved_sampler == "auto":
        resolved_sampler = "round_robin"

    if world_size <= 1 or resolved_sampler != "round_robin":
        return candidate_batch_size

    for batch_size in range(candidate_batch_size, base_batch_size + 1):
        _, global_batches, _ = _estimate_multidataset_steps_per_epoch(
            train_dataset=train_dataset,
            per_device_batch_size=batch_size,
            world_size=world_size,
            sampler_mode=resolved_sampler,
            drop_last=drop_last,
        )
        if global_batches % world_size == 0:
            return batch_size

    return base_batch_size


# =============================================================================
# SimCSE and DMS dataset builders
# =============================================================================
def _build_simcse_dataset(
    file_paths: List[str],
    seq_col: str,
    max_rows: int = 600_000,
) -> Dataset:
    """Build self-pair dataset for SimCSE.

    Output rows are {"sentence_0": seq, "sentence_1": seq, "is_simcse": True}.

    Noise is applied at collator time. Streams from parquet to cap memory.
    """
    _BATCH = 200_000

    def _gen():
        total = 0
        for f in file_paths:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=_BATCH, columns=[seq_col]):
                for seq in batch.column(seq_col).to_pylist():
                    yield {"sentence_0": seq, "sentence_1": seq, "is_simcse": True}
                    total += 1
                    if max_rows > 0 and total >= max_rows:
                        return

    ds = Dataset.from_generator(
        _gen,
        features=Features(
            {
                "sentence_0": Value("string"),
                "sentence_1": Value("string"),
                "is_simcse": Value("bool"),
            }
        ),
    )
    logger.info("🔬 SimCSE self-pair dataset: %d sequences", len(ds))
    return ds


def _load_dms_dataset(file_path: str, max_rows: int = 0) -> Dataset:
    """Load DMS CoSENT parquet with an optional row cap for smoke runs."""

    expected = {"sentence_0", "sentence_1", "score"}
    parquet_file = pq.ParquetFile(file_path)
    missing = expected - set(parquet_file.schema.names)
    if missing:
        raise ValueError(f"DMS parquet missing columns: {missing}")

    if max_rows > 0:
        batch_size = min(50_000, max_rows)

        def _gen():
            total = 0
            batch_reader = pq.ParquetFile(file_path)
            for batch in batch_reader.iter_batches(
                batch_size=batch_size,
                columns=["sentence_0", "sentence_1", "score"],
            ):
                columns = batch.to_pydict()
                for sentence_0, sentence_1, score in zip(
                    columns["sentence_0"],
                    columns["sentence_1"],
                    columns["score"],
                ):
                    yield {
                        "sentence_0": sentence_0,
                        "sentence_1": sentence_1,
                        "score": float(score),
                    }
                    total += 1
                    if total >= max_rows:
                        return

        ds = Dataset.from_generator(
            _gen,
            features=Features(
                {
                    "sentence_0": Value("string"),
                    "sentence_1": Value("string"),
                    "score": Value("float64"),
                }
            ),
        )
    else:
        ds = Dataset.from_parquet(file_path)
        extra = [c for c in ds.column_names if c not in expected]
        if extra:
            ds = ds.remove_columns(extra)

    logger.info("🧬 DMS CoSENT dataset: %d pairs", len(ds))
    return ds


def _enable_esm_dropout(model: nn.Module, rate: float) -> None:
    """Re-enable dropout modules in ESM2 (ships with p=0.0)."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = rate
            count += 1
    logger.info("🎲 Enabled dropout (p=%.2f) on %d nn.Dropout modules", rate, count)


def _set_esm_dropout_rate(model: nn.Module, rate: float) -> int:
    """Set all dropout modules in a model to the same rate.

    Args:
        model: Model whose dropout modules should be updated.
        rate: Dropout probability to apply.

    Returns:
        Number of dropout modules updated.
    """
    updated_modules = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = rate
            updated_modules += 1
    return updated_modules


# =============================================================================
# File & Column Resolution
# =============================================================================
def _expand_paths(file_args: Iterable[str], data_dir: str) -> List[str]:
    resolved: List[str] = []
    unresolved: List[str] = []
    for raw in file_args:
        if os.path.isdir(raw):
            dir_matches = sorted(glob(os.path.join(raw, "*.parquet")))
            if dir_matches:
                resolved.extend(dir_matches)
            else:
                unresolved.append(f"{raw} (directory contains no .parquet files)")
            continue

        matches = sorted(glob(raw))
        if matches:
            for match in matches:
                if os.path.isdir(match):
                    dir_matches = sorted(glob(os.path.join(match, "*.parquet")))
                    if dir_matches:
                        resolved.extend(dir_matches)
                    else:
                        unresolved.append(
                            f"{match} (directory contains no .parquet files)"
                        )
                else:
                    resolved.append(match)
            continue

        alt = os.path.join(data_dir, raw)
        if os.path.isdir(alt):
            alt_matches = sorted(glob(os.path.join(alt, "*.parquet")))
            if alt_matches:
                resolved.extend(alt_matches)
            else:
                unresolved.append(
                    f"{raw} (resolved to '{alt}', but directory contains no .parquet files)"
                )
            continue

        if os.path.exists(raw):
            resolved.append(raw)
            continue

        if os.path.exists(alt):
            resolved.append(alt)
            continue

        unresolved.append(raw)

    if unresolved:
        details = "\n".join(f"  - {entry}" for entry in unresolved)
        raise FileNotFoundError(
            "Some --files entries could not be resolved:\n"
            f"{details}\n"
            f"Hint: paths are checked as provided, then relative to '{data_dir}/'."
        )

    return resolved


def _infer_columns(
    file_paths: List[str],
    seq_col: Optional[str],
    cluster_col: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str], bool]:
    """Detect sequence, group, and hierarchy columns across input files.

    Hierarchy is detected **per-file** so that mixing AFDB (no hierarchy)
    with PFAM (has family_id + clan_id) still enables hierarchy-aware
    training for the files that support it.

    Returns:
        (seq_col, group_col, clan_col, family_col, any_hierarchy)
        *clan_col* / *family_col* are set when **all** files have them.
        *any_hierarchy* is True when **at least one** file has both.
    """
    if not file_paths:
        raise ValueError("No input files provided.")

    def _has_col(path: str, col_name: str) -> bool:
        return col_name in pq.ParquetFile(path).schema.names

    def _missing_col(paths: List[str], col_name: str) -> List[str]:
        return [os.path.basename(p) for p in paths if not _has_col(p, col_name)]

    resolved_seq = seq_col or "sequence"
    missing_seq = _missing_col(file_paths, resolved_seq)
    if missing_seq:
        raise ValueError(
            f"Sequence column '{resolved_seq}' missing in: {', '.join(missing_seq)}"
        )

    resolved_group = cluster_col or "group_id"
    missing_group = _missing_col(file_paths, resolved_group)
    if missing_group:
        for fallback in ["cluster_id", "family_id"]:
            if not _missing_col(file_paths, fallback):
                logger.info("Using fallback group column: %s", fallback)
                resolved_group = fallback
                break
        else:
            raise ValueError(
                f"Group column '{resolved_group}' missing in: {', '.join(missing_group)}. "
                "Re-run data_prep.py to add canonical group_id column."
            )

    per_file_hierarchy = {
        os.path.basename(p): _has_col(p, "family_id") and _has_col(p, "clan_id")
        for p in file_paths
    }
    all_hierarchy = all(per_file_hierarchy.values())
    any_hierarchy = any(per_file_hierarchy.values())

    clan_col_name: Optional[str] = None
    family_col_name: Optional[str] = None
    if all_hierarchy:
        clan_col_name = "clan_id"
        family_col_name = "family_id"
        logger.info("Hierarchical columns (family_id, clan_id) detected in all files")
    elif any_hierarchy:
        hier_files = [f for f, h in per_file_hierarchy.items() if h]
        flat_files = [f for f, h in per_file_hierarchy.items() if not h]
        logger.info(
            "Mixed hierarchy: %s have family_id+clan_id; %s do not. "
            "Per-file label columns will be used in triplet training.",
            hier_files,
            flat_files,
        )

    return resolved_seq, resolved_group, clan_col_name, family_col_name, any_hierarchy


def _best_family_col_for_file(file_path: str) -> str:
    """Return the best label column available in a single parquet file.

    Preference order: family_id > group_id > cluster_id.
    """
    cols = set(pq.ParquetFile(file_path).schema.names)
    for candidate in ("family_id", "group_id", "cluster_id"):
        if candidate in cols:
            return candidate
    raise ValueError(
        f"No usable label column in {os.path.basename(file_path)}. "
        "Expected one of: family_id, group_id, cluster_id."
    )


MNRL_DIRECTIONS_DEFAULT = ("query_to_doc", "doc_to_query")


def _mnrl_directions(args) -> tuple:
    """Which InfoNCE interaction terms MNRL scores.

    Default is symmetric. sentence-transformers defaults to ``("query_to_doc",)``
    because its canonical task is asymmetric — a short query against a long
    document. Every contrastive corpus here is symmetric instead: Pfam and AFDB
    pairs are two members of one cluster, STRING pairs are two interacting
    proteins, and neither column is privileged. The reverse term costs no extra
    forward pass, the embeddings are already computed. V1/V2/V2.5 all trained
    one-directional.

    The fallback matches the argparse default rather than the library's, so a
    caller that builds args programmatically gets the same loss as the CLI.
    """
    return tuple(getattr(args, "mnrl_directions", None) or MNRL_DIRECTIONS_DEFAULT)


def _resolve_primary_loss(args) -> str:
    """The contrastive loss ``--loss_mode multi`` will actually build."""
    configured = getattr(args, "multi_primary_loss", "auto")
    if configured == "auto":
        return getattr(args, "multi_mnrl_loss", "mnrl")
    return configured


def _resolve_batch_sampler(batch_sampler: str, loss_mode: str) -> Optional[object]:
    """Map a resolved loss to its sampler. Takes ``effective_loss``, never "multi".

    The ST docs pair NO_DUPLICATES with (Cached)MNRL; V1, V2 and V2.5 all trained
    without it, because this used to be handed the raw mode and "multi" matched
    nothing here.
    """
    if batch_sampler == "none":
        return None
    if BatchSamplers is None:
        logger.warning(
            "BatchSamplers not available in this sentence-transformers version."
        )
        return None

    if batch_sampler == "auto":
        if loss_mode in {"mnrl", "cached_mnrl", "cached_gist", "gist"}:
            return BatchSamplers.NO_DUPLICATES
        if loss_mode == "triplet":
            return BatchSamplers.GROUP_BY_LABEL
        return None

    mapping = {
        "no_duplicates": BatchSamplers.NO_DUPLICATES,
        "group_by_label": BatchSamplers.GROUP_BY_LABEL,
    }
    return mapping.get(batch_sampler)


def _build_triplet_loss(
    model: SentenceTransformer, args: argparse.Namespace
) -> nn.Module:
    """Build the configured triplet loss variant for train or multi mode."""
    from sentence_transformers.losses import BatchHardTripletLossDistanceFunction

    distance_fn_map = {
        "cosine": BatchHardTripletLossDistanceFunction.cosine_distance,
        "euclidean": BatchHardTripletLossDistanceFunction.eucledian_distance,
    }
    distance_metric = distance_fn_map.get(
        args.triplet_distance_metric,
        BatchHardTripletLossDistanceFunction.cosine_distance,
    )

    if args.triplet_variant == "batch_hard_soft_margin":
        return losses.BatchHardSoftMarginTripletLoss(
            model,
            distance_metric=distance_metric,
        )
    if args.triplet_variant == "batch_all":
        return losses.BatchAllTripletLoss(model)
    if args.triplet_variant == "BatchSemiHardTripletLoss":
        return losses.BatchSemiHardTripletLoss(model)
    raise ValueError(f"Unsupported triplet variant: {args.triplet_variant}")


def _estimate_multidataset_steps_per_epoch(
    train_dataset: dict[str, Dataset],
    per_device_batch_size: int,
    world_size: int,
    sampler_mode: str,
    drop_last: bool,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    """Estimate per-rank steps/epoch for multi-dataset samplers.

    Returns:
        (steps_per_rank_epoch, global_batches_per_epoch, per_dataset_stats)
        where per_dataset_stats entries are (dataset_name, rows, batches).
    """
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be > 0")
    if not train_dataset:
        raise ValueError("train_dataset must not be empty")

    resolved_sampler = (sampler_mode or "round_robin").lower()
    if resolved_sampler == "auto":
        resolved_sampler = "round_robin"

    dataset_stats: list[tuple[str, int, int]] = []
    for name, ds in train_dataset.items():
        rows = len(ds)
        if drop_last:
            batches = rows // per_device_batch_size
        else:
            batches = (rows + per_device_batch_size - 1) // per_device_batch_size
        dataset_stats.append((name, rows, batches))

    if resolved_sampler == "round_robin":
        global_batches = min(batches for _, _, batches in dataset_stats) * len(
            dataset_stats
        )
    elif resolved_sampler in {"proportional", "none"}:
        global_batches = sum(batches for _, _, batches in dataset_stats)
    else:
        global_batches = sum(batches for _, _, batches in dataset_stats)

    per_rank_steps = (global_batches + max(1, world_size) - 1) // max(1, world_size)
    return max(1, per_rank_steps), global_batches, dataset_stats


# =============================================================================
# Dataset Builders (using HF datasets memory-mapping)
# =============================================================================
# NOTE: Future optimization potential (not implemented — large intermediate files):
# - Pre-compute pair/label datasets as parquet in data_prep.py to eliminate
#   training-time generation entirely (hundreds of GB disk trade-off).
# - Filter sequences > max_seq_length in generators to skip tokens that the
#   tokenizer will truncate anyway, reducing Arrow cache size.
def _is_ppi_parquet(file_path: str) -> bool:
    """Detect if a parquet file is a PPI (pre-built pair) dataset."""
    try:
        schema_names = pq.ParquetFile(file_path).schema.names
        return "seq1" in schema_names and "seq2" in schema_names
    except Exception:
        return False


def _load_ppi_pair_dataset(
    file_paths: List[str],
    max_pairs: int = 0,
    sample_seed: int = 40,
    length_labels: bool = False,
    max_seq_length: int = 1024,
) -> Dataset:
    """Load a PPI parquet directly as a sentence-pair Dataset.

    Fast path (no cap or cap >= total rows): uses Dataset.from_parquet()
    directly — no Python generator overhead, no row-by-row iteration.
    Only seq1/seq2 columns are written to the Arrow cache via features=.

    Slow path (max_pairs cap < total rows): samples shuffled parquet row groups
    to avoid training on a deterministic prefix when the source parquet is
    ordered (for example, STRING exports sorted by cluster-derived group IDs).
    """
    # Count total rows to decide which path to use
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in file_paths)
    use_fast_path = max_pairs <= 0 or max_pairs >= total_rows

    if use_fast_path:
        # Load directly from parquet into Arrow cache — no Python generator.
        # Drop any extra columns the parquet may contain before renaming.
        ds = Dataset.from_parquet(cast(Any, file_paths))
        extra = [c for c in ds.column_names if c not in ("seq1", "seq2")]
        if extra:
            ds = ds.remove_columns(extra)
        ds = ds.rename_columns({"seq1": "sentence_0", "seq2": "sentence_1"})
        if length_labels:
            ds = ds.map(
                lambda batch: {
                    "label": [
                        _pair_length(max_seq_length, a, b)
                        for a, b in zip(batch["sentence_0"], batch["sentence_1"])
                    ]
                },
                batched=True,
            )
        logger.info(
            "🔗  Loaded %d PPI pairs from %d file(s) (direct parquet)",
            len(ds),
            len(file_paths),
        )
        return ds

    # Generator path: max_pairs cap requires a randomized sample rather than
    # a deterministic prefix, which can badly bias ordered PPI parquets.
    row_groups: list[tuple[str, int, int]] = []
    for file_path in file_paths:
        parquet_file = pq.ParquetFile(file_path)
        for row_group_index in range(parquet_file.num_row_groups):
            row_groups.append(
                (
                    file_path,
                    row_group_index,
                    parquet_file.metadata.row_group(row_group_index).num_rows,
                )
            )

    rng = random.Random(sample_seed)
    rng.shuffle(row_groups)

    def _gen():
        total = 0
        current_file_path: str | None = None
        current_parquet: pq.ParquetFile | None = None
        for file_path, row_group_index, row_group_size in row_groups:
            remaining = max_pairs - total
            if remaining <= 0:
                return

            if current_file_path != file_path:
                current_parquet = pq.ParquetFile(file_path)
                current_file_path = file_path

            assert current_parquet is not None
            row_group = current_parquet.read_row_group(
                row_group_index,
                columns=["seq1", "seq2"],
            )
            if row_group_size > remaining:
                selected_rows = rng.sample(range(row_group_size), remaining)
            else:
                selected_rows = list(range(row_group_size))
                rng.shuffle(selected_rows)

            sampled_row_group = row_group.take(pa.array(selected_rows, type=pa.int32()))
            s0 = sampled_row_group.column("seq1").to_pylist()
            s1 = sampled_row_group.column("seq2").to_pylist()
            for a, b in zip(s0, s1):
                row = {"sentence_0": a, "sentence_1": b}
                if length_labels:
                    row["label"] = _pair_length(max_seq_length, a, b)
                yield row
                total += 1

    feat_dict = {"sentence_0": Value("string"), "sentence_1": Value("string")}
    if length_labels:
        feat_dict["label"] = Value("int64")

    ds = Dataset.from_generator(
        _gen,
        features=Features(feat_dict),
    )
    logger.info(
        "🔗  Loaded %d PPI pairs from %d file(s) (row-group sample, cap=%d, seed=%d)",
        len(ds),
        len(file_paths),
        max_pairs,
        sample_seed,
    )
    return ds


def _build_pair_dataset(
    file_paths: List[str],
    seq_col: str,
    group_col: str,
    max_pairs_per_cluster: int,
    max_pairs: int,
    hard_negatives: bool = False,
    length_labels: bool = False,
    max_seq_length: int = 1024,
) -> Dataset:
    """Build a pair Dataset using a generator over group-sorted data.

    Iterates parquet row groups (~200K rows at a time) so only a single
    batch lives in Python memory.  Groups are carried across row-group
    boundaries via a small carry buffer.  The output is written to a
    disk-backed HF Arrow cache by Dataset.from_generator().

    When ``hard_negatives=True`` and the parquet contains a
    ``hard_negative`` column, it is emitted as ``sentence_2`` for pairs where
    the anchor has valid negatives.  CachedMultipleNegativesRankingLoss
    and CachedGISTEmbedLoss treat extra sentence columns as explicit
    hard negatives automatically.

    Args:
        file_paths: Parquet file paths (must be pre-sorted by group_col).
        seq_col: Column name for sequences.
        group_col: Column name for group/family labels.
        max_pairs_per_cluster: Cap on sequences sampled per group.
        max_pairs: Global cap on emitted pairs (0 = no limit).
        hard_negatives: If True, include hard negative columns when present.

    Returns:
        HuggingFace Dataset with sentence_0, sentence_1,
        and optionally sentence_2.
    """
    _BATCH = 200_000

    # Detect hard negative columns in first file
    has_neg = False
    if hard_negatives and file_paths:
        schema_names = pq.ParquetFile(file_paths[0]).schema.names
        has_neg = "hard_negative" in schema_names
    if hard_negatives and not has_neg:
        logger.info(
            "--hard_negatives enabled but this parquet lacks "
            "a hard_negative column; "
            "continuing without explicit hard negatives for this file"
        )

    logger.info(
        "Reading source rows from %d file(s) (streaming, hard_neg=%s)...",
        len(file_paths),
        has_neg,
    )

    skipped_null_neg = 0

    def _pair_gen():
        """Yield pairs from consecutive same-group rows."""
        nonlocal skipped_null_neg
        _comb = combinations
        total_pairs = 0
        buf_seqs: List[str] = []
        buf_hard_neg: List[Optional[str]] = []
        prev_group = None

        columns = [seq_col, group_col]
        if has_neg:
            columns.append("hard_negative")

        def _flush_group():
            nonlocal total_pairs, skipped_null_neg
            if len(buf_seqs) < 2:
                return
            sample_idx = list(range(len(buf_seqs)))
            if len(sample_idx) > max_pairs_per_cluster:
                sample_idx = random.sample(sample_idx, max_pairs_per_cluster)
            for a_i, b_i in _comb(sample_idx, 2):
                # A null hard negative must not become an empty sentence_2:
                # MNRL pools every sentence_2 in the batch into the candidate
                # set, so a "" would be a degenerate zero-residue negative that
                # the loss pushes all proteins away from. Skip the pair instead.
                if has_neg and not buf_hard_neg[a_i]:
                    skipped_null_neg += 1
                    continue
                row: dict[str, str] = {
                    "sentence_0": buf_seqs[a_i],
                    "sentence_1": buf_seqs[b_i],
                }
                if has_neg:
                    row["sentence_2"] = buf_hard_neg[a_i]
                if length_labels:
                    row["label"] = _pair_length(
                        max_seq_length,
                        buf_seqs[a_i],
                        buf_seqs[b_i],
                        row.get("sentence_2"),
                    )
                yield row
                total_pairs += 1
                if max_pairs > 0 and total_pairs >= max_pairs:
                    return

        for f in file_paths:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=_BATCH, columns=columns):
                grps = batch.column(group_col).to_pylist()
                sqss = batch.column(seq_col).to_pylist()
                if has_neg:
                    surgs = batch.column("hard_negative").to_pylist()
                else:
                    surgs = [None] * len(grps)

                for grp, seq, sg in zip(grps, sqss, surgs):
                    if grp != prev_group:
                        yield from _flush_group()
                        if max_pairs > 0 and total_pairs >= max_pairs:
                            return
                        buf_seqs = []
                        buf_hard_neg = []
                        prev_group = grp
                    buf_seqs.append(seq)
                    buf_hard_neg.append(sg)

        # Flush last group
        yield from _flush_group()

    # Build Features schema
    feat_dict: dict[str, Value] = {
        "sentence_0": Value("string"),
        "sentence_1": Value("string"),
    }
    if has_neg:
        feat_dict["sentence_2"] = Value("string")
    if length_labels:
        feat_dict["label"] = Value("int64")

    pair_ds = Dataset.from_generator(
        _pair_gen,
        features=Features(feat_dict),
    )
    logger.info(
        "Built %d pairs (streaming, files: %s, hard_neg=%s)",
        len(pair_ds),
        [os.path.basename(f) for f in file_paths],
        has_neg,
    )
    if skipped_null_neg:
        logger.info(
            "Skipped %d pairs whose anchor had no hard negative "
            "(would previously have trained on an empty-string negative)",
            skipped_null_neg,
        )
    return pair_ds


def _build_label_dataset(
    file_paths: List[str],
    seq_col: str,
    family_col: str,
    max_rows: int,
    min_label_count: int,
    max_samples_per_label: int = 0,
    seed: int = 42,
) -> Dataset:
    """Build a Dataset with sentence + integer label for triplet losses.

    Two-pass streaming approach — avoids materializing rows that will be
    discarded by the rare-label filter or per-label cap:

      Pass 1 — Stream only the label column to build a frequency Counter.
               Memory usage: O(unique labels), not O(n rows).
      Pass 2 — Stream both columns, applying filter and cap in-generator
               so only surviving rows are written to the Arrow cache.

    The result is disk-backed Arrow via Dataset.from_generator().
    """
    _BATCH = 500_000

    # ── Pass 1: count label frequencies (label column only) ──────────────
    label_counts: Counter = Counter()
    total_scanned = 0
    for f in file_paths:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=_BATCH, columns=[family_col]):
            label_counts.update(batch.column(family_col).to_pylist())
            total_scanned += len(batch)
            if max_rows > 0 and total_scanned >= max_rows:
                break
        if max_rows > 0 and total_scanned >= max_rows:
            break

    # Build keep_set from pass-1 counts
    if min_label_count > 1:
        keep_set = frozenset(
            lbl for lbl, cnt in label_counts.items() if cnt >= min_label_count
        )
    else:
        keep_set = frozenset(label_counts.keys())

    if not keep_set:
        raise ValueError(
            "No labels with enough examples for triplet loss. "
            "Lower --min_label_count or provide more data."
        )

    n_dropped = len(label_counts) - len(keep_set)
    if n_dropped > 0:
        logger.info(
            "🏷️  Label pre-filter: keeping %d/%d labels (min_count=%d, dropped %d rare)",
            len(keep_set),
            len(label_counts),
            min_label_count,
            n_dropped,
        )

    # ── Pass 2: yield filtered rows (rare labels already excluded) ───────
    # Per-label cap is applied post-generator with shuffle to preserve
    # diversity (PFAM is sorted by clan→family, so in-order capping would
    # always pick the same sequences from each family).

    def _row_gen():
        total = 0
        for f in file_paths:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(
                batch_size=_BATCH, columns=[seq_col, family_col]
            ):
                seqs = batch.column(seq_col).to_pylist()
                labels = batch.column(family_col).to_pylist()
                for s, lbl in zip(seqs, labels):
                    if lbl not in keep_set:
                        continue
                    yield {seq_col: s, family_col: lbl}
                    total += 1
                    if max_rows > 0 and total >= max_rows:
                        return

    ds = Dataset.from_generator(
        _row_gen,
        features=Features({seq_col: Value("string"), family_col: Value("string")}),
    )

    if len(ds) == 0:
        raise ValueError("No samples loaded for label dataset.")

    # Cap samples per label — shuffle first for diversity, then select
    if max_samples_per_label > 0:
        ds = ds.shuffle(seed=seed)
        labels_list = ds[family_col]
        label_counts_cap: dict = {}
        keep_idx = []
        for i, lbl in enumerate(labels_list):
            cur = label_counts_cap.get(lbl, 0)
            if cur < max_samples_per_label:
                keep_idx.append(i)
                label_counts_cap[lbl] = cur + 1
        ds = ds.select(keep_idx)
        logger.info(
            "📍 Per-label cap applied (max=%d); dataset size reduced to %d",
            max_samples_per_label,
            len(ds),
        )

    # Shuffle final triplet dataset for label and within-label order diversity
    ds = ds.shuffle(seed=seed)

    # Rename and encode labels
    ds = ds.rename_columns({seq_col: "sentence", family_col: "label"})
    ds = ds.class_encode_column("label")

    n_labels = ds.features["label"].num_classes
    logger.info(
        "🏷️  Label dataset: %d samples, %d labels (min_count=%d)",
        len(ds),
        n_labels,
        min_label_count,
    )
    return ds


# =============================================================================
# Training
# =============================================================================
def run_training(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Training requires a GPU.")
    if getattr(args, "progress_bars", None):
        os.environ["PROTEIN_PROGRESS_BARS"] = args.progress_bars

    files = _expand_paths(args.files, data_dir="data") if args.files else []
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files and getattr(args, "loss_mode", None) != "multi":
        raise FileNotFoundError(f"Files not found: {args.files}")

    logger.info(
        f"📁 Resolved {len(files)} file(s): {[os.path.basename(f) for f in files]}"
    )

    # Separate PPI files early — they don't have sequence/group_id columns
    non_ppi_files = [f for f in files if not _is_ppi_parquet(f)]

    if non_ppi_files:
        seq_col, group_col, clan_col, family_col, any_hierarchy = _infer_columns(
            non_ppi_files, args.seq_col, args.cluster_col
        )
    else:
        seq_col, group_col, clan_col, family_col, any_hierarchy = (
            "sequence",
            "group_id",
            None,
            None,
            False,
        )
    hierarchical = clan_col is not None and family_col is not None

    logger.info(
        f"🔧 Config: seq_col='{seq_col}', group_col='{group_col}'"
        + (
            f", clan_col='{clan_col}', family_col='{family_col}'"
            if hierarchical
            else ""
        )
    )

    logger.info("⏳ Calculating size...")
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)

    loss_mode = args.loss_mode
    if loss_mode == "auto":
        loss_mode = "triplet" if (hierarchical or any_hierarchy) else "cached_mnrl"

    # "multi" is a container, not a loss. Expand it here, once, next to the "auto"
    # expansion above: every downstream test of the form `loss_mode in {...}` wants
    # the contrastive loss that actually gets built, and each one that read the raw
    # mode was silently dead for multi-task runs — the batch sampler resolved to
    # None, and the cached-loss split logging plus its DDP warning never fired.
    effective_loss = _resolve_primary_loss(args) if loss_mode == "multi" else loss_mode
    mnrl_directions = _mnrl_directions(args)

    effective_rows = total_rows
    if args.max_map_rows > 0:
        effective_rows = min(total_rows, args.max_map_rows)
    grad_accum = max(1, args.gradient_accumulation_steps)
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = args.batch_size * grad_accum * world_size
    max_steps = max(1, int((effective_rows / effective_batch) * args.epochs))
    if getattr(args, "max_steps", 0) and args.max_steps > 0:
        max_steps = args.max_steps
    elif args.fast:
        max_steps = 20
    logger.info(
        "📊 Dataset: total=%s rows, effective=%s rows -> ~%s steps (effective_batch=%s)",
        f"{total_rows:,}",
        f"{effective_rows:,}",
        max_steps,
        effective_batch,
    )

    if loss_mode == "triplet" and family_col is None and args.triplet_use_group_id:
        family_col = group_col
        clan_col = None
        logger.info(
            "🔁 Triplet mode: using '%s' as family labels (no clan hierarchy)",
            family_col,
        )

    if loss_mode == "triplet" and family_col is None:
        raise ValueError(
            "Triplet loss requires label columns, but neither family_id/clan_id hierarchy "
            "nor group_id fallback labels are available. "
            "Use PFAM hierarchy data, or re-enable --triplet_use_group_id."
        )

    # A triplet primary under multi-task keeps the historical None:
    # GroupByLabelBatchSampler raises when a dataset has no label column, and a
    # multi-task dict mixes labelled and unlabelled datasets.
    sampler_loss = "" if loss_mode == "multi" and effective_loss == "triplet" else effective_loss
    batch_sampler = _resolve_batch_sampler(args.batch_sampler, sampler_loss)
    if getattr(args, "length_bucketed_batches", False) and args.batch_sampler == "auto":
        # Explicit bucketing beats an auto-resolved sampler: they set the same
        # training_kwargs key, and the compatibility check below would otherwise
        # turn every bucketed multi-task run into a hard error.
        batch_sampler = None
    if (
        BatchSamplers is not None
        and batch_sampler == BatchSamplers.GROUP_BY_LABEL
        and args.batch_size % 2 != 0
    ):
        raise ValueError(
            "GroupByLabelBatchSampler requires an even batch size. "
            "Set --batch_size to an even number."
        )

    logger.info(
        "🧪 Loss config: mode=%s, batch_sampler=%s",
        loss_mode,
        str(batch_sampler) if batch_sampler is not None else "none",
    )
    if getattr(args, "length_bucketed_batches", False):
        if batch_sampler is not None and batch_sampler != BatchSamplers.BATCH_SAMPLER:
            raise ValueError(
                "--length_bucketed_batches is only compatible with the default batch sampler"
            )
        logger.info(
            "🪣 Length-bucketed batches enabled (bucket_size=%d)",
            args.length_bucket_size,
        )

    if effective_loss in {"cached_mnrl", "cached_gist"}:
        cache_splits = (args.batch_size + args.mnrl_mini_batch_size - 1) // max(
            1, args.mnrl_mini_batch_size
        )
        logger.info(
            "⚙️  Cached loss chunking: batch_size=%d, mini_batch=%d -> %d split(s)",
            args.batch_size,
            args.mnrl_mini_batch_size,
            cache_splits,
        )
        if world_size > 1 and cache_splits >= 8:
            logger.warning(
                "⚠️  High cached-loss split count under DDP (%d splits, WORLD_SIZE=%d). "
                "This can cause heavy synchronization overhead and very slow step time. "
                "Consider increasing --mnrl_mini_batch_size and/or lowering --batch_size.",
                cache_splits,
                world_size,
            )

    # Check for existing checkpoints to resume from
    output_dir = os.path.join(args.output_root, args.run_name)
    resume_from_checkpoint = None
    should_resume = args.resume and not args.no_resume

    if should_resume and os.path.exists(output_dir):
        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            resume_from_checkpoint = latest_checkpoint
            logger.info(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
        else:
            logger.info("📝 No existing checkpoints found. Starting fresh.")
    elif should_resume:
        logger.info("📝 No output directory found. Starting fresh.")
    else:
        logger.info("📝 Resume disabled. Starting fresh.")

    # Determine local rank early for device assignment and WandB configuration
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _register_signal_tracebacks(output_dir, local_rank)

    # Load model
    if resume_from_checkpoint:
        logger.info("⏳ Loading model from checkpoint...")
        model = SentenceTransformer(resume_from_checkpoint, trust_remote_code=True)
        logger.info("✅ Model loaded from checkpoint")
        # Checkpoints carry the backbone's own tokenizer, and FastPLM's declares
        # model_max_length = 1e24, so without this a resumed run trains untruncated
        # while the first half of the same run did not.
        _enforce_max_seq_length(model, args.max_seq_length)
        # Ensure rotary caches are trainable (ESM2 inference mode fix)
        _ensure_trainable_rotary_caches(model)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}"
        model = load_model_for_training(
            args.model,
            max_seq_length=args.max_seq_length,
            device=device,
            pooling_mode=args.pooling_mode,
            pooling_activation=args.pooling_activation,
        )

    # Enable ESM dropout globally for all training modes.
    if args.enable_esm_dropout > 0:
        _enable_esm_dropout(model, args.enable_esm_dropout)

    if args.compile:
        if world_size > 1 and not _is_distributed_launcher_active(world_size):
            logger.warning(
                "⚠️  Skipping torch.compile for WORLD_SIZE=%s without a distributed "
                "launcher (likely DataParallel path).",
                world_size,
            )
        elif _compile_sentence_transformer_backbone(model):
            logger.info(
                "⚡ Compiled training backbone with model-aware torch.compile settings"
            )

    callbacks = []
    if args.max_minutes > 0:
        callbacks.append(TimeLimitCallback(args.max_minutes))
    # Note: RotaryCacheSafetyCallback is NOT needed unless an evaluator is used
    # (evaluators call model.encode() which uses torch.inference_mode() and
    # contaminates ESM2 rotary caches). CachedGISTEmbedLoss/CachedMNRL use
    # torch.no_grad() which is safe. Uncomment if adding an evaluator:
    # callbacks.append(RotaryCacheSafetyCallback(every_n_steps=25))

    save_steps = 2 if args.fast else args.save_steps
    if args.report_to == "auto":
        resolved_report_to = "none" if world_size > 1 else "wandb"
    else:
        resolved_report_to = args.report_to

    if resolved_report_to == "wandb" and world_size > 1:
        report_to_value = "wandb" if local_rank == 0 else "none"
    else:
        report_to_value = resolved_report_to

    logger.info("📣 Reporting backend: %s", report_to_value)

    # Force-disable integrations when report backend is none. Some trainer
    # versions can still attempt callback registration when passed the string
    # "none".
    report_to_arg: list[str] | str = (
        [] if report_to_value == "none" else report_to_value
    )
    show_progress_bars = _progress_bars_enabled(local_rank=local_rank)
    progress_min_interval = _configure_tqdm_defaults(show_progress_bars)
    logger.info(
        "📊 Progress bars: %s (PROTEIN_PROGRESS_BARS=%s, min_interval=%.1fs)",
        "on" if show_progress_bars else "off",
        os.environ.get("PROTEIN_PROGRESS_BARS", "auto"),
        progress_min_interval,
    )

    # gradient_checkpointing adds a non-picklable input-require-grads hook to the
    # model; under the 'spawn' start method (forced on Python 3.14) the model-bound
    # DataLoader collate_fn then fails to pickle to worker processes. Fall back to
    # in-process data loading when checkpointing is on.
    _dl_workers = args.dataloader_num_workers
    if getattr(args, "gradient_checkpointing", False) and _dl_workers > 0:
        logger.warning(
            "⚠️  gradient_checkpointing on: forcing dataloader_num_workers=0 "
            "(spawn cannot pickle the model-bound collate with the checkpoint hook)."
        )
        _dl_workers = 0

    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": args.run_name,
        "num_train_epochs": args.epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
        if getattr(args, "gradient_checkpointing", False)
        else None,
        "logging_steps": 2 if args.fast else 20,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "warmup_steps": 2 if args.fast else args.warmup_steps,
        "save_total_limit": args.save_total_limit,
        "gradient_accumulation_steps": grad_accum,
        "ddp_timeout": 1800,
        "ddp_bucket_cap_mb": 50,
        "dataloader_drop_last": True,
        "ignore_data_skip": True,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": 2 if _dl_workers > 0 else None,
        "auto_find_batch_size": False,
        "report_to": report_to_arg,
        "dataloader_num_workers": _dl_workers,
        "dataloader_persistent_workers": _dl_workers > 0,
        "disable_tqdm": not show_progress_bars,
    }

    if getattr(args, "length_bucketed_batches", False):
        training_kwargs["batch_sampler"] = partial(
            LengthBucketBatchSampler,
            bucket_size=args.length_bucket_size,
            length_column="label",
        )

    # Multi-GPU DDP: use gather_across_devices=True for cross-device negatives.
    # Both CachedMNRL and MNRL support DDP natively via this parameter.
    _gather_across_devices = world_size > 1 and not getattr(
        args, "no_gather_across_devices", False
    )
    if _gather_across_devices:
        logger.info(
            "🌐 DDP (world_size=%d): enabling gather_across_devices for loss "
            "functions (cross-device in-batch negatives)",
            world_size,
        )
    elif world_size > 1:
        logger.info(
            "🌐 DDP (world_size=%d): gather_across_devices disabled; using "
            "per-rank in-batch negatives only",
            world_size,
        )

    lr_scheduler_kwargs = {}
    if args.lr_scheduler_type in {
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
    }:
        lr_scheduler_kwargs["num_cycles"] = args.lr_num_cycles
    if args.lr_scheduler_type in {"cosine_with_min_lr", "cosine_warmup_with_min_lr"}:
        lr_scheduler_kwargs["min_lr_rate"] = args.lr_min_lr_rate
    if args.lr_scheduler_kwargs:
        try:
            user_kwargs = json.loads(args.lr_scheduler_kwargs)
            if not isinstance(user_kwargs, dict):
                raise ValueError("--lr_scheduler_kwargs must parse to a JSON object")
            lr_scheduler_kwargs.update(user_kwargs)
        except Exception as e:
            raise ValueError(
                f"Invalid --lr_scheduler_kwargs JSON: {args.lr_scheduler_kwargs}"
            ) from e
    if lr_scheduler_kwargs:
        training_kwargs["lr_scheduler_kwargs"] = lr_scheduler_kwargs

    if loss_mode == "mnrl":
        training_kwargs["per_device_train_batch_size"] = max(
            1, args.mnrl_mini_batch_size
        )
        logger.info(
            "⚙️ MNRL mode: per_device_train_batch_size set from mnrl_mini_batch_size=%d",
            training_kwargs["per_device_train_batch_size"],
        )

    # Multi-dataset sampler (auto resolves to ROUND_ROBIN / interleaved)
    multi_dataset_sampler = None
    if len(files) > 1 and args.multi_dataset_sampler != "none":
        try:
            from sentence_transformers.sentence_transformer.training_args import (
                MultiDatasetBatchSamplers,
            )

            sampler_map = {
                "round_robin": "ROUND_ROBIN",
                "proportional": "PROPORTIONAL",
                "auto": "ROUND_ROBIN",
            }
            sampler_name = sampler_map.get(args.multi_dataset_sampler)
            if sampler_name:
                multi_dataset_sampler = getattr(
                    MultiDatasetBatchSamplers,
                    sampler_name,
                    MultiDatasetBatchSamplers.ROUND_ROBIN,
                )
            if multi_dataset_sampler is not None:
                logger.info(f"🎯 Multi-dataset sampler: {args.multi_dataset_sampler}")
                training_kwargs["multi_dataset_batch_sampler"] = multi_dataset_sampler
                # multi_dataset_batch_sampler controls cross-dataset interleaving
                # (ROUND_ROBIN / PROPORTIONAL) while batch_sampler controls
                # per-dataset grouping (GROUP_BY_LABEL / NO_DUPLICATES).
                # They compose as outer/inner samplers and are NOT mutually
                # exclusive — keep both active.
                if batch_sampler is not None:
                    logger.info(
                        "🎯 Per-dataset batch_sampler kept active: %s "
                        "(inner sampler within multi-dataset %s)",
                        batch_sampler,
                        args.multi_dataset_sampler,
                    )
        except ImportError:
            logger.warning(
                "MultiDatasetBatchSamplers not available; multi-file training will use sequential interleaving."
            )

    if batch_sampler is not None:
        training_kwargs["batch_sampler"] = batch_sampler

    training_args = SentenceTransformerTrainingArguments(**training_kwargs)

    logger.info(
        f"💾 Checkpoint saving: every {save_steps} steps (keeping last {training_args.save_total_limit})"
    )

    def _apply_matryoshka(
        loss_obj: nn.Module | dict[str, nn.Module],
    ) -> nn.Module | dict[str, nn.Module]:
        if not args.matryoshka:
            return loss_obj

        native_dim = model.get_sentence_embedding_dimension()
        dims_set = set(args.matryoshka_dims)
        dims_set.add(native_dim)
        dims = sorted([d for d in dims_set if d <= native_dim])

        logger.info(f"🪆 Applying MatryoshkaLoss with dimensions: {dims}")

        def _wrap(inner: nn.Module) -> nn.Module:
            # Matryoshka must sit *inside* the GOR wrapper, never outside it.
            # MatryoshkaLoss dispatches on the loss it is given: for a Cached*
            # loss it decorates calculate_loss and the backbone runs once, but
            # for anything else it decorates SentenceTransformer.forward with an
            # index-keyed cache. GOR's own forward pass then desynchronises that
            # index and CachedMNRL's backward hook dies with "inconsistent
            # tensor size, expected tensor [122880] and src [16384]".
            if isinstance(inner, LossWithGOR):
                inner.base_loss = losses.MatryoshkaLoss(
                    model, inner.base_loss, matryoshka_dims=dims
                )
                return inner
            return losses.MatryoshkaLoss(model, inner, matryoshka_dims=dims)

        if isinstance(loss_obj, dict):
            return {k: _wrap(v) for k, v in loss_obj.items()}
        return _wrap(loss_obj)

    gor_weight = float(getattr(args, "gor_weight", 0.0) or 0.0)
    if gor_weight < 0:
        raise ValueError("--gor_weight must be non-negative")
    if gor_weight > 0:
        logger.info("🧭 GOR enabled for contrastive losses (weight=%.4f)", gor_weight)

    def _apply_gor(loss_obj: nn.Module) -> nn.Module:
        if gor_weight <= 0:
            return loss_obj
        mini_batch_size = getattr(args, "mnrl_mini_batch_size", 32)
        return LossWithGOR(
            model,
            loss_obj,
            gor_weight=gor_weight,
            mini_batch_size=mini_batch_size,
            max_samples=args.gor_max_samples,
            mean_weight=args.gor_mean_weight,
        )

    # ── Build datasets & loss, then train ────────────────────────────────
    if loss_mode in {"mnrl", "cached_mnrl", "cached_gist"}:
        per_file_max = args.max_map_rows
        if len(files) > 1 and per_file_max > 0:
            per_file_max = max(1, per_file_max // len(files))

        ppi_files = [f for f in files if _is_ppi_parquet(f)]
        cluster_files = [f for f in files if not _is_ppi_parquet(f)]
        if ppi_files:
            logger.info(
                "🤝 Detected %d PPI file(s): %s",
                len(ppi_files),
                [os.path.basename(f) for f in ppi_files],
            )
        if cluster_files:
            logger.info(
                "🧬 Detected %d cluster-based file(s): %s",
                len(cluster_files),
                [os.path.basename(f) for f in cluster_files],
            )

        if len(files) > 1:
            train_dataset = {}
            for f in cluster_files:
                name = os.path.splitext(os.path.basename(f))[0]
                ds = _build_pair_dataset(
                    file_paths=[f],
                    seq_col=seq_col,
                    group_col=group_col,
                    max_pairs_per_cluster=args.max_pairs_per_cluster,
                    max_pairs=per_file_max,
                    hard_negatives=args.hard_negatives,
                    length_labels=args.length_bucketed_batches,
                    max_seq_length=args.max_seq_length,
                )
                if args.pair_dataset_shuffle:
                    ds = ds.shuffle(seed=args.pair_dataset_shuffle_seed)
                train_dataset[name] = ds
                logger.info("📦 %s: %d pairs", name, len(ds))
            for f in ppi_files:
                name = os.path.splitext(os.path.basename(f))[0]
                ds = _load_ppi_pair_dataset(
                    file_paths=[f],
                    max_pairs=per_file_max,
                    sample_seed=args.pair_dataset_shuffle_seed,
                    length_labels=args.length_bucketed_batches,
                    max_seq_length=args.max_seq_length,
                )
                if args.pair_dataset_shuffle:
                    ds = ds.shuffle(seed=args.pair_dataset_shuffle_seed)
                train_dataset[name] = ds
                logger.info("📦 %s: %d PPI pairs", name, len(ds))
        else:
            if ppi_files:
                train_dataset = _load_ppi_pair_dataset(
                    file_paths=ppi_files,
                    max_pairs=args.max_map_rows,
                    sample_seed=args.pair_dataset_shuffle_seed,
                    length_labels=args.length_bucketed_batches,
                    max_seq_length=args.max_seq_length,
                )
            else:
                train_dataset = _build_pair_dataset(
                    file_paths=files,
                    seq_col=seq_col,
                    group_col=group_col,
                    max_pairs_per_cluster=args.max_pairs_per_cluster,
                    max_pairs=args.max_map_rows,
                    hard_negatives=args.hard_negatives,
                    length_labels=args.length_bucketed_batches,
                    max_seq_length=args.max_seq_length,
                )
            if args.pair_dataset_shuffle:
                train_dataset = train_dataset.shuffle(
                    seed=args.pair_dataset_shuffle_seed
                )

        # Choose similarity function for MNRL based on distance metric
        if (
            hasattr(args, "mnrl_distance_metric")
            and args.mnrl_distance_metric == "euclidean"
        ):
            similarity_fct = _euclidean_similarity
            mnrl_scale = 1.0  # Euclidean distances are in [0, inf], scale differently
            logger.info("   📏 Using euclidean distance for MNRL/GIST")
        else:
            from sentence_transformers import util

            similarity_fct = util.cos_sim
            mnrl_scale = 20.0  # Default cosine scale
            logger.info("   📐 Using cosine similarity for MNRL/GIST")

        if loss_mode == "cached_gist":
            guide_model = _load_gist_guide_model(args)
            guide_model.eval()
            guide_model.requires_grad_(False)
            _ensure_tokenizer_vocab_attr(model)
            _ensure_tokenizer_vocab_attr(guide_model)
            logger.info(
                "   ✅ Guide model loaded and frozen (contrast_positives=%s, "
                "margin_strategy=%s, margin=%s)",
                args.gist_contrast_positives,
                args.gist_margin_strategy,
                args.gist_margin,
            )
            loss = losses.CachedGISTEmbedLoss(
                model,
                guide=guide_model,
                mini_batch_size=args.mnrl_mini_batch_size,
                margin_strategy=args.gist_margin_strategy,
                margin=args.gist_margin,
                contrast_anchors=True,
                contrast_positives=args.gist_contrast_positives,
                gather_across_devices=_gather_across_devices,
            )
        elif loss_mode == "cached_mnrl":
            loss = losses.CachedMultipleNegativesRankingLoss(
                model,
                scale=mnrl_scale,
                similarity_fct=similarity_fct,
                mini_batch_size=args.mnrl_mini_batch_size,
                gather_across_devices=_gather_across_devices,
                directions=mnrl_directions,
            )
        else:
            loss = losses.MultipleNegativesRankingLoss(
                model,
                scale=mnrl_scale,
                similarity_fct=similarity_fct,
                gather_across_devices=_gather_across_devices,
                directions=mnrl_directions,
            )

        loss = _apply_gor(loss)
        loss = _apply_matryoshka(loss)

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            callbacks=callbacks,
        )
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    elif loss_mode == "triplet":
        # Triplet loss with label-aware batching
        loss = _build_triplet_loss(model, args)

        assert family_col is not None or any_hierarchy or args.triplet_use_group_id

        # Build label dataset(s) — no windowing, full memory-mapped load
        if len(files) > 1 and multi_dataset_sampler is not None:
            per_file_max = args.max_map_rows
            if per_file_max > 0:
                per_file_max = max(1, per_file_max // len(files))

            train_dataset = {}
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                file_family_col = (
                    family_col
                    if family_col is not None
                    else _best_family_col_for_file(f)
                )
                ds = _build_label_dataset(
                    file_paths=[f],
                    seq_col=seq_col,
                    family_col=file_family_col,
                    max_rows=per_file_max,
                    min_label_count=args.min_label_count,
                    max_samples_per_label=args.triplet_max_samples_per_label,
                    seed=args.seed,
                )
                train_dataset[name] = ds
                logger.info("📦 %s: %d samples", name, len(ds))
        else:
            single_family_col = (
                family_col
                if family_col is not None
                else _best_family_col_for_file(files[0])
            )
            train_dataset = _build_label_dataset(
                file_paths=files,
                seq_col=seq_col,
                family_col=single_family_col,
                max_rows=args.max_map_rows,
                min_label_count=args.min_label_count,
                max_samples_per_label=args.triplet_max_samples_per_label,
                seed=args.seed,
            )

        # ── Recalculate max_steps from actual post-filter dataset size ────
        if not (getattr(args, "max_steps", 0) and args.max_steps > 0) and not args.fast:
            if isinstance(train_dataset, dict):
                actual_rows = sum(len(cast(Any, ds)) for ds in train_dataset.values())
            else:
                actual_rows = len(train_dataset)
            actual_max_steps = max(
                1, int((actual_rows / effective_batch) * args.epochs)
            )
            if actual_max_steps != max_steps:
                ratio = actual_rows / max(1, effective_rows)
                logger.info(
                    "Recalculated max_steps: %d -> %d "
                    "(post-filter rows=%s, %.1f%% of raw estimate)",
                    max_steps,
                    actual_max_steps,
                    f"{actual_rows:,}",
                    ratio * 100,
                )
                max_steps = actual_max_steps
                training_args.max_steps = actual_max_steps

        loss = _apply_matryoshka(loss)

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            callbacks=callbacks,
        )
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    elif loss_mode == "multi":
        # Multi-task: selectable primary loss + optional DMS CoSENT dataset.
        from sentence_transformers import util

        ppi_files = [f for f in files if _is_ppi_parquet(f)]
        cluster_files = [f for f in files if not _is_ppi_parquet(f)]

        def _pair_cap_for_file(file_path: str) -> int:
            """Per-file pair budget. 0 = no limit, matching --max_map_rows' help text.

            This previously returned the source row count, which is NOT "no limit":
            because --max_pairs_per_cluster emits C(min(n, k), 2) pairs per cluster
            (~55 pairs per source row at k=500), the budget was exhausted inside the
            first ~2% of a group-sorted corpus, so training only ever saw the
            lowest-sorted clans/clusters. Control the volume with
            --max_pairs_per_cluster (uniform across the corpus) or an explicit
            --max_map_rows, not with an accidental prefix truncation.
            """
            if args.max_map_rows > 0:
                return max(1, args.max_map_rows // max(1, len(files)))
            return 0

        primary_loss = _resolve_primary_loss(args)
        allowed_primary = {"mnrl", "cached_mnrl", "triplet", "gist", "cached_gist"}
        if primary_loss not in allowed_primary:
            raise ValueError(
                f"Unsupported --multi_primary_loss value: {primary_loss}. "
                f"Expected one of {sorted(allowed_primary)}"
            )

        logger.info(
            "🎛️ Multi-task primary loss resolved: %s (legacy multi_mnrl_loss=%s)",
            primary_loss,
            getattr(args, "multi_mnrl_loss", "mnrl"),
        )

        # Similarity function for MNRL variants.
        if args.mnrl_distance_metric == "euclidean":
            similarity_fct = _euclidean_similarity
            mnrl_scale = 1.0
            logger.info("   📏 Multi-task using euclidean distance for MNRL variants")
        else:
            similarity_fct = util.cos_sim
            mnrl_scale = 20.0
            logger.info("   📐 Multi-task using cosine similarity for MNRL variants")

        gist_temperature = 0.01
        guide_model: SentenceTransformer | None = None
        if primary_loss in {"gist", "cached_gist"}:
            guide_model = _load_gist_guide_model(args)
            guide_model.eval()
            guide_model.requires_grad_(False)
            _ensure_tokenizer_vocab_attr(model)
            _ensure_tokenizer_vocab_attr(guide_model)
            if primary_loss == "cached_gist":
                _patch_cached_gist_guide_preprocess(guide_model)
            logger.info(
                "   ✅ Multi-task guide model loaded and frozen for %s",
                primary_loss,
            )

        def _build_pair_primary_loss() -> nn.Module:
            if primary_loss == "cached_mnrl":
                return _apply_gor(
                    losses.CachedMultipleNegativesRankingLoss(
                        model,
                        scale=mnrl_scale,
                        similarity_fct=similarity_fct,
                        mini_batch_size=args.mnrl_mini_batch_size,
                        gather_across_devices=_gather_across_devices,
                        directions=mnrl_directions,
                    )
                )
            if primary_loss == "mnrl":
                return _apply_gor(
                    losses.MultipleNegativesRankingLoss(
                        model,
                        scale=mnrl_scale,
                        similarity_fct=similarity_fct,
                        gather_across_devices=_gather_across_devices,
                        directions=mnrl_directions,
                    )
                )
            if primary_loss == "cached_gist":
                assert guide_model is not None
                cached_gist_loss = losses.CachedGISTEmbedLoss(
                    model,
                    guide=guide_model,
                    temperature=gist_temperature,
                    mini_batch_size=args.mnrl_mini_batch_size,
                    margin_strategy=args.gist_margin_strategy,
                    margin=args.gist_margin,
                    contrast_anchors=True,
                    contrast_positives=args.gist_contrast_positives,
                    gather_across_devices=_gather_across_devices,
                )
                _patch_cached_gist_embed_minibatch(cached_gist_loss)
                return _apply_gor(cached_gist_loss)
            if primary_loss == "gist":
                assert guide_model is not None
                return _apply_gor(
                    losses.GISTEmbedLoss(
                        model,
                        guide=guide_model,
                        temperature=gist_temperature,
                        margin_strategy=args.gist_margin_strategy,
                        margin=args.gist_margin,
                        contrast_anchors=True,
                        contrast_positives=args.gist_contrast_positives,
                        gather_across_devices=_gather_across_devices,
                    )
                )
            raise ValueError(f"Pair primary loss is not supported for {primary_loss}")

        def _execute_dataset_and_loss_building():
            local_train_ds: dict[str, Dataset] = {}
            local_loss_dict: dict[str, nn.Module] = {}

            if primary_loss == "triplet":
                if ppi_files:
                    logger.warning(
                        "⚠️ Skipping %d PPI file(s) for triplet multi mode (pair-only schema)",
                        len(ppi_files),
                    )

                for f in cluster_files:
                    name = os.path.splitext(os.path.basename(f))[0]
                    file_family_col = (
                        family_col
                        if family_col is not None
                        else _best_family_col_for_file(f)
                    )
                    ds = _build_label_dataset(
                        file_paths=[f],
                        seq_col=seq_col,
                        family_col=file_family_col,
                        max_rows=per_file_max,
                        min_label_count=args.min_label_count,
                        max_samples_per_label=args.triplet_max_samples_per_label,
                        seed=args.seed,
                    )
                    local_train_ds[name] = ds
                    local_loss_dict[name] = _build_triplet_loss(model, args)
                    logger.info(
                        "📦 %s: %d labeled samples (Triplet)",
                        name,
                        len(ds),
                    )
            else:
                primary_label_map = {
                    "mnrl": "MNRL",
                    "cached_mnrl": "CachedMNRL",
                    "gist": "GIST",
                    "cached_gist": "CachedGIST",
                }
                primary_label = primary_label_map.get(primary_loss, primary_loss)

                for f in cluster_files:
                    name = os.path.splitext(os.path.basename(f))[0]
                    ds = _build_pair_dataset(
                        file_paths=[f],
                        seq_col=seq_col,
                        group_col=group_col,
                        max_pairs_per_cluster=args.max_pairs_per_cluster,
                        max_pairs=_pair_cap_for_file(f),
                        hard_negatives=args.hard_negatives,
                        length_labels=args.length_bucketed_batches,
                        max_seq_length=args.max_seq_length,
                    )
                    if args.pair_dataset_shuffle:
                        ds = ds.shuffle(seed=args.pair_dataset_shuffle_seed)
                    local_train_ds[name] = ds
                    local_loss_dict[name] = _build_pair_primary_loss()
                    logger.info("📦 %s: %d pairs (%s)", name, len(ds), primary_label)

                for f in ppi_files:
                    name = os.path.splitext(os.path.basename(f))[0]
                    ds = _load_ppi_pair_dataset(
                        file_paths=[f],
                        max_pairs=_pair_cap_for_file(f),
                        sample_seed=args.pair_dataset_shuffle_seed,
                        length_labels=args.length_bucketed_batches,
                        max_seq_length=args.max_seq_length,
                    )
                    if args.pair_dataset_shuffle:
                        ds = ds.shuffle(seed=args.pair_dataset_shuffle_seed)
                    local_train_ds[name] = ds
                    local_loss_dict[name] = _build_pair_primary_loss()
                    logger.info("📦 %s: %d PPI pairs (%s)", name, len(ds), primary_label)

            simcse_files_resolved = None
            if args.simcse_files:
                simcse_files_resolved = _expand_paths(args.simcse_files, data_dir="data")
            if simcse_files_resolved:
                if primary_loss == "triplet":
                    logger.warning(
                        "⚠️ SimCSE dataset skipped for multi_primary_loss=triplet"
                    )
                else:
                    simcse_seq_col = seq_col
                    simcse_ds = _build_simcse_dataset(
                        file_paths=simcse_files_resolved,
                        seq_col=simcse_seq_col,
                        max_rows=args.simcse_max_rows,
                    )
                    local_train_ds["simcse"] = simcse_ds
                    local_loss_dict["simcse"] = _build_pair_primary_loss()
                    logger.info(
                        "📦 simcse: %d self-pairs (%s, mini_bs=%d)",
                        len(simcse_ds),
                        primary_loss,
                        args.mnrl_mini_batch_size,
                    )

            # DMS always contributes a CoSENT objective when provided.
            if args.dms_file and os.path.exists(args.dms_file):
                dms_ds = _load_dms_dataset(args.dms_file, max_rows=args.dms_max_rows)
                local_train_ds["dms_cosent"] = dms_ds
                try:
                    cosent_loss: nn.Module = losses.CoSENTLoss(
                        model,
                        scale=mnrl_scale,
                        gather_across_devices=_gather_across_devices,
                    )
                except TypeError:
                    cosent_loss = losses.CoSENTLoss(model, scale=mnrl_scale)
                # CoSENT has no gradient cache: it embeds and backpropagates the
                # whole batch at once, and DMS sequences are the longest in the
                # corpus (median 448 residues against a 512-token truncation). At
                # the batch sizes CachedMNRL is run at, that alone exhausts a
                # 267 GiB B300. Cap it at the mini-batch that the contrastive path
                # is already tuned to fit in one grad-enabled forward.
                cosent_cap = max(1, args.mnrl_mini_batch_size)
                local_loss_dict["dms_cosent"] = SubsampledLoss(cosent_loss, cosent_cap)
                logger.info(
                    "📦 dms_cosent: %d pairs (CoSENTLoss, target_bs=%d, cap=%d, scale=%.1f)",
                    len(dms_ds),
                    args.dms_batch_size,
                    cosent_cap,
                    mnrl_scale,
                )
            elif args.dms_file:
                logger.warning("⚠️ DMS file not found: %s (skipping)", args.dms_file)

            return local_train_ds, local_loss_dict

        train_dataset: dict[str, Dataset] = {}
        loss_dict: dict[str, nn.Module] = {}

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_ddp = world_size > 1 and torch.distributed.is_initialized()

        if not is_ddp or local_rank == 0:
            train_dataset, loss_dict = _execute_dataset_and_loss_building()
            if is_ddp:
                logger.info("✅ Rank 0 completed dataset preparation. Releasing other ranks...")
                torch.distributed.barrier()
        else:
            logger.info("⏳ Other ranks waiting for rank 0 dataset preparation...")
            torch.distributed.barrier()
            logger.info("🚀 Rank 0 released. Proceeding to load cached datasets...")
            train_dataset, loss_dict = _execute_dataset_and_loss_building()
            logger.info("✅ Non-zero rank loaded datasets instantly from rank 0 cache!")

        if not train_dataset:
            raise ValueError("No datasets built for multi-task training.")

        logger.info(
            "🎯 Multi-task: %d datasets: %s",
            len(train_dataset),
            list(train_dataset.keys()),
        )

        base_multi_batch_size = max(1, args.batch_size)
        dms_target_batch_size = (
            args.dms_batch_size if args.dms_batch_size > 0 else base_multi_batch_size
        )
        training_kwargs["per_device_train_batch_size"] = base_multi_batch_size
        logger.info(
            "⚙️ Multi-task base batch size set from batch_size=%d",
            base_multi_batch_size,
        )

        if "dms_cosent" in train_dataset:
            resolved_multi_batch_size = _resolve_dms_train_batch_size(
                base_batch_size=base_multi_batch_size,
                dms_batch_size=dms_target_batch_size,
                mnrl_mini_batch_size=args.mnrl_mini_batch_size,
                train_dataset=train_dataset,
                world_size=world_size,
                sampler_mode=getattr(args, "multi_dataset_sampler", "round_robin"),
                drop_last=training_kwargs["dataloader_drop_last"],
            )
            training_kwargs["per_device_train_batch_size"] = resolved_multi_batch_size
            logger.info(
                "⚙️ DMS enabled: per_device_train_batch_size=%d (base=%d, dms_target=%d)",
                resolved_multi_batch_size,
                base_multi_batch_size,
                dms_target_batch_size,
            )

        # Multi-dataset sampler (respect CLI arg)
        try:
            from sentence_transformers.sentence_transformer.training_args import (
                MultiDatasetBatchSamplers,
            )

            sampler_map = {
                "round_robin": MultiDatasetBatchSamplers.ROUND_ROBIN,
                "proportional": MultiDatasetBatchSamplers.PROPORTIONAL,
                "auto": MultiDatasetBatchSamplers.ROUND_ROBIN,
            }
            chosen = sampler_map.get(
                getattr(args, "multi_dataset_sampler", "round_robin"),
                MultiDatasetBatchSamplers.ROUND_ROBIN,
            )
            training_kwargs["multi_dataset_batch_sampler"] = chosen
            logger.info("🎯 Multi-dataset sampler: %s", chosen)
        except ImportError:
            logger.warning("MultiDatasetBatchSamplers not available; using default.")

        # Recalculate max_steps from multi-dataset batch-sampler semantics.
        # For ROUND_ROBIN, epoch length is bounded by the smallest dataset's
        # batch count (not by total row count), then sharded across DDP ranks.
        if not (getattr(args, "max_steps", 0) and args.max_steps > 0) and not args.fast:
            sampler_mode = getattr(args, "multi_dataset_sampler", "round_robin")
            steps_per_epoch, global_batches, dataset_stats = (
                _estimate_multidataset_steps_per_epoch(
                    train_dataset=train_dataset,
                    per_device_batch_size=training_kwargs[
                        "per_device_train_batch_size"
                    ],
                    world_size=world_size,
                    sampler_mode=sampler_mode,
                    drop_last=training_kwargs["dataloader_drop_last"],
                )
            )
            # steps_per_epoch is in batch-level steps; divide by grad_accum
            # to get optimizer steps (which is what HF Trainer's max_steps expects)
            actual_max_steps = max(1, int(steps_per_epoch * args.epochs) // grad_accum)
            if actual_max_steps != max_steps:
                dataset_batch_summary = ", ".join(
                    f"{name}:{batches}b" for name, _, batches in dataset_stats
                )
                logger.info(
                    "Recalculated max_steps: %d -> %d "
                    "(sampler=%s, steps/epoch=%d, global_batches=%d, datasets=%s)",
                    max_steps,
                    actual_max_steps,
                    sampler_mode,
                    steps_per_epoch,
                    global_batches,
                    dataset_batch_summary,
                )
                max_steps = actual_max_steps
                training_kwargs["max_steps"] = actual_max_steps

            if resume_from_checkpoint:
                trainer_state_path = os.path.join(
                    resume_from_checkpoint,
                    "trainer_state.json",
                )
                if os.path.exists(trainer_state_path):
                    try:
                        with open(trainer_state_path, "r", encoding="utf-8") as f:
                            trainer_state = json.load(f)
                        checkpoint_max_steps = int(
                            trainer_state.get("max_steps", 0) or 0
                        )
                        if (
                            checkpoint_max_steps
                            and checkpoint_max_steps != actual_max_steps
                        ):
                            logger.warning(
                                "Checkpoint max_steps=%d differs from current max_steps=%d. "
                                "This resume may carry stale scheduler/step state; "
                                "prefer a fresh Stage 2 run (--no_resume).",
                                checkpoint_max_steps,
                                actual_max_steps,
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not read checkpoint trainer_state.json for max_steps validation: %s",
                            e,
                        )

        training_args = SentenceTransformerTrainingArguments(**training_kwargs)

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "loss": loss_dict,
            "callbacks": callbacks,
        }
        if args.mask_rate > 0 and "simcse" in train_dataset:
            tokenizer = model.tokenizer
            trainer_kwargs["data_collator"] = NoisySimCSECollator(
                tokenize_fn=model.tokenize,
                tokenizer=tokenizer,
                mask_rate=args.mask_rate,
            )
        elif args.mask_rate > 0:
            logger.info(
                "🧼 Mask noise requested but simcse dataset is absent; skipping noisy collator"
            )
        else:
            logger.info("🧼 Mask noise disabled (mask_rate=%.2f)", args.mask_rate)

        loss_dict = cast(dict[str, nn.Module], _apply_matryoshka(loss_dict))
        trainer_kwargs["loss"] = loss_dict

        trainer = SentenceTransformerTrainer(**trainer_kwargs)

        # DDP barrier: ensure all ranks have finished dataset construction
        # and model setup before any rank enters the training loop.  Without
        # this, a fast rank can start issuing NCCL collectives while a slow
        # rank is still loading data, leading to timeouts.
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
            logger.info("🔄 DDP barrier passed — all ranks ready")

        logger.info("🚂 Starting trainer.train()")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("✅ trainer.train() returned")

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    final_model_path = os.path.join(output_dir, "final")
    if args.enable_esm_dropout > 0:
        disabled_modules = _set_esm_dropout_rate(model, 0.0)
        logger.info(
            "🧯 Disabled dropout before final save (updated %d modules)",
            disabled_modules,
        )
    logger.info("💾 Saving final model via trainer.save_model(): %s", final_model_path)
    trainer.save_model(final_model_path)
    if training_args.should_save:
        logger.info("✅ Final model saved to: %s", final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep")
    prep.add_argument(
        "--dataset",
        choices=[
            "pfam",
            "nvidia",
            "afdb",
            "stringdb",
            "pfam_hard_negatives",
            "dms",
        ],
        required=True,
    )
    prep.add_argument("--limit_gb", type=int, default=30)
    prep.add_argument("--fast", action="store_true")
    prep.add_argument(
        "--min_combined_score",
        type=int,
        default=400,
        help="STRING-DB: minimum combined_score filter (default 400 = medium confidence)",
    )
    prep.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="STRING-DB: maximum interaction pairs (0 = no limit)",
    )
    prep.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="STRING-DB: minimum sequence length in amino acids (default 10)",
    )
    prep.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="STRING-DB: maximum sequence length in amino acids (default 1024)",
    )
    # ── pfam_hard_negatives args ─────────────────────────────────────────
    prep.add_argument(
        "--hard_negative_max_evalue",
        type=float,
        default=1.0,
        help="pfam_hard_negatives: accept a mutant as a negative once its E-value "
        "against its own family exceeds this (default 1.0)",
    )
    prep.add_argument(
        "--hard_negative_evalue_z",
        type=float,
        default=1e6,
        help="pfam_hard_negatives: effective database size for E-value calculation, "
        "pinned so acceptance is batch-independent (default 1e6)",
    )
    prep.add_argument(
        "--hard_negative_max_mut_frac",
        type=float,
        default=0.5,
        help="pfam_hard_negatives: cap on the fraction of aligned positions that may "
        "be mutated (default 0.5)",
    )
    prep.add_argument(
        "--min_aligned_positions",
        type=int,
        default=20,
        help="pfam_hard_negatives: skip anchors aligning to fewer match states "
        "(default 20)",
    )
    prep.add_argument(
        "--force",
        action="store_true",
        help="pfam_hard_negatives / dms: overwrite existing output if present",
    )
    prep.add_argument(
        "--max_total_rows",
        type=int,
        default=0,
        help="pfam_hard_negatives: global cap on selected rows before generation (0 = uncapped)",
    )
    prep.add_argument(
        "--max_seqs_per_family",
        type=int,
        default=100,
        help="pfam_hard_negatives: cap per-family rows before generation (default 100)",
    )
    prep.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pfam_hard_negatives: thread workers (0 = auto)",
    )

    train_cmd = subparsers.add_parser("train")
    train_cmd.add_argument("--files", nargs="+", default=[])
    train_cmd.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    train_cmd.add_argument("--batch_size", type=int, default=192)
    train_cmd.add_argument("--epochs", type=int, default=1)
    train_cmd.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Override computed training steps (0 = auto from dataset size and epochs)",
    )
    train_cmd.add_argument("--max_minutes", type=int, default=0)
    train_cmd.add_argument(
        "--learning_rate",
        type=float,
        default=8e-5,
        help="Initial learning rate (default: 8e-5)",
    )
    train_cmd.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    train_cmd.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer (default: adamw_torch; try adamw_torch_fused for speed)",
    )
    train_cmd.add_argument(
        "--warmup_steps",
        type=int,
        default=800,
        help="Warmup steps for LR scheduler (default: 800)",
    )
    train_cmd.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_min_lr",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "cosine_with_min_lr",
            "cosine_warmup_with_min_lr",
            "constant",
            "constant_with_warmup",
            "reduce_on_plateau",
            "warmup_stable_decay",
        ],
        help="Learning rate scheduler type (default: cosine_with_min_lr)",
    )
    train_cmd.add_argument(
        "--lr_num_cycles",
        type=float,
        default=3.0,
        help="Number of cycles for cosine-family schedulers (default: 3.0)",
    )
    train_cmd.add_argument(
        "--lr_min_lr_rate",
        type=float,
        default=0.05,
        help="Minimum LR as fraction of base LR for min-lr cosine schedulers (default: 0.05)",
    )
    train_cmd.add_argument(
        "--lr_scheduler_kwargs",
        type=str,
        default="",
        help="Optional JSON object of scheduler kwargs to merge with defaults",
    )
    train_cmd.add_argument(
        "--report_to",
        choices=["auto", "wandb", "none"],
        default="none",
        help="Training reporter backend. auto uses wandb on single-GPU and none on multi-GPU.",
    )
    train_cmd.add_argument("--run_name", default="prot_sbert")
    train_cmd.add_argument("--fast", action="store_true")

    train_cmd.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for tokenizer truncation",
    )
    train_cmd.add_argument(
        "--output_root",
        type=str,
        default="models",
        help="Root directory for checkpoints and final models (default: models)",
    )
    train_cmd.add_argument("--seq_col", default=None)
    train_cmd.add_argument("--cluster_col", default=None)
    train_cmd.add_argument(
        "--loss_mode",
        choices=["auto", "mnrl", "cached_mnrl", "cached_gist", "triplet", "multi"],
        default="auto",
        help="Loss selection: auto picks triplet for PFAM hierarchy, else cached_mnrl. "
        "mnrl enables standard MultipleNegativesRankingLoss. "
        "multi: multi-task with selectable primary loss (MNRL/CachedMNRL/Triplet/GIST/CachedGIST) + optional DMS CoSENT.",
    )
    train_cmd.add_argument(
        "--triplet_variant",
        choices=["batch_hard_soft_margin", "batch_all", "BatchSemiHardTripletLoss"],
        default="batch_hard_soft_margin",
        help="Triplet loss variant for triplet loss mode",
    )
    train_cmd.add_argument(
        "--triplet_use_group_id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use group_id as fallback family labels when family_id/clan_id are missing "
        "(default: True; set --no-triplet_use_group_id to require hierarchy).",
    )
    train_cmd.add_argument(
        "--triplet_max_samples_per_label",
        type=int,
        default=25,
        help="Cap samples per family label for triplet loss (0=no cap). Lower values increase families per batch.",
    )
    train_cmd.add_argument(
        "--triplet_distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for BatchHardSoftMarginTripletLoss",
    )
    train_cmd.add_argument(
        "--batch_sampler",
        choices=["auto", "no_duplicates", "group_by_label", "none"],
        default="auto",
        help="Batch sampler selection (map mode only)",
    )
    train_cmd.add_argument(
        "--multi_dataset_sampler",
        choices=["auto", "proportional", "round_robin", "none"],
        default="round_robin",
        help="Multi-dataset sampler: round_robin (alternates dataset per batch, balanced), "
        "proportional (samples by size), auto (picks round_robin), none (sequential interleaving)",
    )
    train_cmd.add_argument(
        "--length_bucketed_batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bucket pair batches by clipped sequence length to reduce padding waste.",
    )
    train_cmd.add_argument(
        "--length_bucket_size",
        type=int,
        default=64,
        help="Length bucket width used by --length_bucketed_batches.",
    )
    train_cmd.add_argument(
        "--max_map_rows",
        type=int,
        default=50_000_000,
        help="Max rows/pairs for map-mode datasets (0 = no limit). Default 50M.",
    )
    train_cmd.add_argument(
        "--min_label_count",
        type=int,
        default=6,
        help="Minimum samples per label for triplet loss",
    )
    train_cmd.add_argument(
        "--mnrl_mini_batch_size",
        type=int,
        default=256,
        help="Mini-batch size for CachedMultipleNegativesRankingLoss / CachedGISTEmbedLoss. "
        "Ignored by non-cached MNRL/GIST except as a legacy DMS fallback in helper tests.",
    )
    train_cmd.add_argument(
        "--multi_mnrl_loss",
        choices=["mnrl", "cached_mnrl"],
        default="mnrl",
        help="For --loss_mode multi, choose contrastive loss for pair/simcse datasets. "
        "mnrl avoids cached backward hooks and is DDP-safe with mixed losses.",
    )
    train_cmd.add_argument(
        "--multi_primary_loss",
        choices=["auto", "mnrl", "cached_mnrl", "triplet", "gist", "cached_gist"],
        default="auto",
        help="Primary multi-task loss for non-DMS datasets. "
        "auto preserves legacy --multi_mnrl_loss behavior. "
        "Use triplet for triplet+CoSENT and gist/cached_gist for GIST+CoSENT.",
    )
    train_cmd.add_argument(
        "--mnrl_distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for CachedMultipleNegativesRankingLoss / CachedGISTEmbedLoss. "
        "cosine: scaled dot-product (default), euclidean: negative L2 distance",
    )
    # ── CachedGISTEmbedLoss arguments ────────────────────────────────────
    train_cmd.add_argument(
        "--gist_guide_model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Guide model for CachedGISTEmbedLoss. A fast, frozen protein LM. "
        "Default: facebook/esm2_t6_8M_UR50D (8M params, minimal compute overhead).",
    )
    train_cmd.add_argument(
        "--gist_static_guide",
        type=str,
        default="",
        help="Optional locally saved StaticEmbedding guide model for CachedGISTEmbedLoss. "
        f"Example local path: {DEFAULT_STATIC_GUIDE_DIR}. When set, this overrides "
        "--gist_guide_model.",
    )
    train_cmd.add_argument(
        "--gist_static_guide_device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for a locally saved StaticEmbedding guide model. Defaults to "
        "'cuda' so GIST guide inference runs on GPU. Set to 'cpu' only if you "
        "explicitly want to offload the frozen guide. This only affects the frozen "
        "guide path and does not make training CPU-compatible.",
    )
    train_cmd.add_argument(
        "--gist_margin_strategy",
        choices=["absolute", "relative"],
        default="absolute",
        help="False-negative filtering strategy for CachedGISTEmbedLoss. "
        "'absolute': discard negatives with sim >= positive_score - margin. "
        "'relative': discard negatives with sim >= positive_score * (1 - margin). "
        "Default: absolute.",
    )
    train_cmd.add_argument(
        "--gist_margin",
        type=float,
        default=0.05,
        help="Margin for false-negative filtering in CachedGISTEmbedLoss. "
        "With 'absolute' strategy and margin=0.0, only removes negatives that are "
        "more similar to the query than the positive. Default: 0.05.",
    )
    train_cmd.add_argument(
        "--gist_contrast_positives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include positive-positive pairs in loss (push positives apart). "
        "Default: False — recommended when many positives share the same family/label, "
        "as pushing them apart can hurt performance.",
    )
    train_cmd.add_argument(
        "--mnrl_directions",
        nargs="+",
        default=list(MNRL_DIRECTIONS_DEFAULT),
        choices=["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
        help="InfoNCE interaction terms for (Cached)MNRL. Default is symmetric, which "
        "suits this data: both columns of a pair are proteins drawn the same way. "
        "Pass just query_to_doc to reproduce V1/V2/V2.5.",
    )
    train_cmd.add_argument(
        "--gor_weight",
        type=float,
        default=0.0,
        help="Outer multiplier on GlobalOrthogonalRegularizationLoss for contrastive "
        "losses (requires sentence-transformers>=5.3.0). 0 disables. The GOR paper "
        "(1708.06320) uses 1.0; V2.5 ran 0.1 and moved geometry but no task metric.",
    )
    train_cmd.add_argument(
        "--gor_max_samples",
        type=int,
        default=192,
        help="Rows per column GOR estimates from. The second-moment term is a tail "
        "statistic and converges slowly, so this trades accuracy for an extra "
        "grad-enabled forward pass. 0 uses the whole batch.",
    )
    train_cmd.add_argument(
        "--gor_mean_weight",
        type=float,
        default=1.0,
        help="Weight on GOR's mean term. EmbeddingGemma (2509.20354) sets this to 0, "
        "keeping only the second moment.",
    )
    train_cmd.add_argument("--max_files", type=int, default=0)
    train_cmd.add_argument("--max_pairs_per_cluster", type=int, default=30)
    train_cmd.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Trade ~30%% step speed for ~10x less activation memory. Lets the "
        "grad-enabled DMS/CoSENT step (the batch-size cap) hold larger batches at "
        "1024 tokens. Drop the flag if the ESM++ wrapper lacks "
        "gradient_checkpointing_enable.",
    )
    train_cmd.add_argument(
        "--no_gather_across_devices",
        action="store_true",
        default=False,
        help="Disable cross-device contrastive gather under DDP. Faster but fewer negatives per step.",
    )
    train_cmd.add_argument(
        "--hard_negatives",
        action="store_true",
        default=False,
        help="Use hard_negative as an explicit negative for CachedMNRL/CachedGIST",
    )
    train_cmd.add_argument(
        "--pair_dataset_shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle pair datasets before training (default: True)",
    )
    train_cmd.add_argument(
        "--pair_dataset_shuffle_seed",
        type=int,
        default=40,
        help="Seed for pair dataset shuffle (default: 40)",
    )
    train_cmd.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Global random seed for triplet dataset shuffling and per-label capping (default: 41)",
    )
    train_cmd.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint if exists (default: True)",
    )
    train_cmd.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resuming from checkpoint",
    )
    train_cmd.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps (default: 200)",
    )
    train_cmd.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Number of checkpoints to keep (default: 1)",
    )
    train_cmd.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=32,
        help="Number of dataloader workers (default: 32)",
    )
    train_cmd.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps to increase effective batch size",
    )
    train_cmd.add_argument(
        "--progress_bars",
        choices=["auto", "on", "off"],
        default=None,
        help="Progress bars for training/embedding. Default: honor PROTEIN_PROGRESS_BARS env (or auto if unset).",
    )
    train_cmd.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile for training (default: True). Uses model-aware defaults (backend=inductor, dynamic=False by default) and only enables extra Dynamo workarounds when the loaded backbone actually uses HF ESM rotary caches.",
    )
    # ── Multi-task (--loss_mode multi) arguments ──────────────────────────
    train_cmd.add_argument(
        "--simcse_files",
        nargs="+",
        default=None,
        help="Files for SimCSE self-pair dataset (used with --loss_mode multi)",
    )
    train_cmd.add_argument(
        "--dms_file",
        type=str,
        default=None,
        help="DMS CoSENT parquet file (used with --loss_mode multi)",
    )
    train_cmd.add_argument(
        "--dms_max_rows",
        type=int,
        default=0,
        help="Optional cap for DMS CoSENT rows (0 = full dataset)",
    )
    train_cmd.add_argument(
        "--dms_batch_size",
        type=int,
        default=0,
        help="Target per-device batch size when DMS CoSENT is enabled (0 = inherit --batch_size).",
    )
    train_cmd.add_argument(
        "--mask_rate",
        type=float,
        default=0.05,
        help="Token masking rate for NoisySimCSECollator (default: 0.05)",
    )
    train_cmd.add_argument(
        "--enable_esm_dropout",
        type=float,
        default=0.1,
        help="Re-enable ESM2 dropout at this rate (default: 0.1 = On)",
    )
    train_cmd.add_argument(
        "--simcse_max_rows",
        type=int,
        default=5_000_000,
        help="Max sequences for SimCSE self-pair dataset (default: 5M)",
    )
    train_cmd.add_argument(
        "--pooling_mode",
        choices=["mean", "linear_attention", "contextual_attention", "gem"],
        default="mean",
        help="Sentence embedding pooling strategy (default: mean)",
    )
    train_cmd.add_argument(
        "--pooling_activation",
        choices=["tanh", "silu", "gelu"],
        default="tanh",
        help="Activation for contextual_attention pooling (default: tanh)",
    )
    # ── Matryoshka Loss Arguments ─────────────────────────────────────────
    train_cmd.add_argument(
        "--matryoshka",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Matryoshka Representation Learning loss wrapper (default: True)",
    )
    train_cmd.add_argument(
        "--matryoshka_dims",
        type=int,
        nargs="+",
        default=[32, 64, 256],
        help="Prefix dimensions for Matryoshka loss. The model's native dimension is automatically appended. (default: 32 64 256)",
    )
    args = parser.parse_args()
    if args.command == "prep":
        from data_prep import DataPrep

        dp = DataPrep()
        if args.dataset == "nvidia":
            dp.prep_nvidia(args.limit_gb)
        elif args.dataset == "pfam":
            dp.prep_pfam_full(fast=args.fast)
        elif args.dataset == "afdb":
            dp.prep_afdb(args.limit_gb)
        elif args.dataset == "stringdb":
            dp.prep_stringdb(
                min_combined_score=args.min_combined_score,
                max_rows=args.max_rows,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
            )
        elif args.dataset == "pfam_hard_negatives":
            dp.prep_pfam_hard_negatives(
                max_evalue=args.hard_negative_max_evalue,
                evalue_z=args.hard_negative_evalue_z,
                max_mutation_fraction=args.hard_negative_max_mut_frac,
                min_aligned_positions=args.min_aligned_positions,
                force=args.force,
                max_total_rows=args.max_total_rows,
                max_seqs_per_family=args.max_seqs_per_family,
                workers=args.workers,
            )
        elif args.dataset == "dms":
            dp.prep_dms(force=args.force)
    elif args.command == "train":
        run_training(args)
