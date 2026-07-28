"""
Shared utilities for non-standard protein language model support.

Handles model-type detection, compatibility patches, and wrapper modules
for models that don't follow vanilla HuggingFace conventions:
  - AMPLIFY (chandar-lab): additive attention mask, xformers stride alignment
  - ESMplusplus (Synthyra): custom tokenizer, SentenceTransformer assembly
  - FastPLM ESM2 (Synthyra): bug-fixed ESM2 re-implementation
  - DPLM2 (Synthyra): protein diffusion language model
  - Profluent-E1 (Synthyra): retrieval-augmented protein encoder

Used by both protein_pipeline.py (training) and protein_benchmark_suite.py (benchmarking).
"""

import importlib
import importlib.util
import json
import logging
import os
from string import ascii_uppercase
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)

ModelType = Literal[
    "amplify", "esmplusplus", "fastplm_esm2", "dplm2", "profluent_e1", "standard"
]

# Flash attention availability (cached at import time)
try:
    HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
except Exception:
    HAS_FLASH_ATTN = False

# =============================================================================
# Model type detection
# =============================================================================


def detect_model_type(model_name: str) -> ModelType:
    """Detect model family from name/path. Checks HF name and local config.json."""
    name_lower = model_name.lower()

    if "amplify" in name_lower:
        return "amplify"

    if "esmplusplus" in name_lower or "esm++" in name_lower or "esm-c" in name_lower:
        return "esmplusplus"

    # Synthyra FastPLM ESM2 re-implementation (e.g. Synthyra/ESM2-150M)
    # Matches: "Synthyra/ESM2-*", names containing "fastplm"
    if "synthyra/esm2" in name_lower or "fastplm" in name_lower:
        return "fastplm_esm2"

    # Synthyra DPLM2 (e.g. Synthyra/DPLM2-150M, Synthyra/DPLM2-650M)
    if "dplm2" in name_lower:
        return "dplm2"

    # Synthyra Profluent-E1 (e.g. Synthyra/Profluent-E1-150M)
    if ("profluent" in name_lower and "synthyra" in name_lower) or "e1-" in name_lower:
        return "profluent_e1"

    # Check local config.json for non-obvious model paths
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("model_type") == "AMPLIFY":
                return "amplify"
            archs = cfg.get("architectures", [])
            auto_map = cfg.get("auto_map", {})
            all_vals = [*archs, *auto_map.values()]
            if any("ESMplusplus" in str(v) for v in all_vals):
                return "esmplusplus"
            # FastPLM ESM2: custom architectures from Synthyra ESM2 repos
            if any("FastESM" in str(v) or "FastEsmModel" in str(v) for v in all_vals):
                return "fastplm_esm2"
            # DPLM2: check for DPLM2 architectures
            if any("DPLM2" in str(v) or "Dplm2" in str(v) for v in all_vals):
                return "dplm2"
            # Profluent-E1: check for E1 architectures
            if any("E1" in str(v) or "ProfluentE1" in str(v) for v in all_vals):
                return "profluent_e1"
        except Exception:
            pass

    return "standard"


# =============================================================================
# Transformers compatibility patch (ESMplusplus)
# =============================================================================


def apply_esmplusplus_compat_patch():
    """Patch transformers to handle ESMplusplus models lacking 'all_tied_weights_keys'.

    Also ensures 'entrypoint_setup' (a side-effect config module shipped with the
    ESMplusplus Hub repo) is importable from the HF modules cache.  transformers'
    check_imports() treats any top-level import as a PyPI dependency; this module
    doesn't exist on PyPI so we copy it into the modules cache on demand.

    Safe to call multiple times — applies only once via a sentinel attribute.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel

        if getattr(PreTrainedModel, "_esmplusplus_patch_applied", False):
            return
        if hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
            _orig = PreTrainedModel.mark_tied_weights_as_initialized

            def _patched(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = []
                return _orig(self, *args, **kwargs)

            PreTrainedModel.mark_tied_weights_as_initialized = _patched
        PreTrainedModel._esmplusplus_patch_applied = True
    except ImportError:
        pass

    # ── entrypoint_setup: make importable ─────────────────────────────────────
    # Older ESMplusplus Hub code did `import entrypoint_setup` at the top level.
    # We search ALL Synthyra snapshot dirs (not just ESMplusplus_small) and add
    # them to sys.path so the import resolves.  Best-effort; no-op if not found.
    try:
        import sys
        from pathlib import Path as _P

        hub_dir = _P.home() / ".cache" / "huggingface" / "hub"
        if hub_dir.exists():
            for model_dir in hub_dir.iterdir():
                if not model_dir.name.startswith("models--Synthyra"):
                    continue
                snap_base = model_dir / "snapshots"
                if not snap_base.exists():
                    continue
                for snap_dir in snap_base.iterdir():
                    if (snap_dir / "entrypoint_setup.py").exists():
                        snap_str = str(snap_dir)
                        if snap_str not in sys.path:
                            sys.path.insert(0, snap_str)
                            logger.debug(
                                "Added Synthyra snapshot to sys.path: %s", snap_str
                            )
    except Exception:
        pass  # best-effort; failure handled at load time


def apply_esm_rotary_autograd_patch() -> None:
    """Patch ESM RotaryEmbedding.forward to sanitize inference-mode caches.

    During some SentenceTransformer cached-loss code paths, ESM rotary caches
    (``_cos_cached``/``_sin_cached``) can be refreshed under inference mode,
    then reused in a grad-enabled pass, triggering:
    ``RuntimeError: Inference tensors cannot be saved for backward``.

    This patch is idempotent and safe: in grad-enabled forwards, if a cached
    rotary tensor is inference-mode, it is cloned back to a normal tensor before
    continuing.
    """
    try:
        from transformers.models.esm.modeling_esm import RotaryEmbedding
    except Exception:
        return

    if getattr(RotaryEmbedding, "_rotary_autograd_patch_applied", False):
        return

    original_forward = RotaryEmbedding.forward

    def _patched_forward(self, q, k):
        if torch.is_grad_enabled():
            for attr in ("_cos_cached", "_sin_cached"):
                tensor = getattr(self, attr, None)
                if not isinstance(tensor, torch.Tensor):
                    continue
                is_inference = bool(
                    hasattr(tensor, "is_inference") and tensor.is_inference()
                )
                if is_inference:
                    setattr(self, attr, tensor.clone())
        return original_forward(self, q, k)

    RotaryEmbedding.forward = _patched_forward
    RotaryEmbedding._rotary_autograd_patch_applied = True


# =============================================================================
# Shared model-loading helpers
# =============================================================================


def from_pretrained_with_flash(model_cls, model_name: str, **extra_kwargs):
    """Load a model with flash attention if available, falling back to default.

    Args:
        model_cls: Model class to instantiate (AutoModel, AutoModelForMaskedLM, etc.)
        model_name: Model name or path
        **extra_kwargs: Additional kwargs to pass to from_pretrained (for example dtype)
    """
    # transformers >=5.3 warns that torch_dtype is deprecated in favor of dtype.
    if "torch_dtype" in extra_kwargs and "dtype" not in extra_kwargs:
        extra_kwargs["dtype"] = extra_kwargs.pop("torch_dtype")

    kwargs = {"trust_remote_code": True, **extra_kwargs}
    uses_custom_attention_backend = detect_model_type(model_name) == "esmplusplus"
    # Prefer flash_attention_2 if available, otherwise eagerly enforce PyTorch native SDPA
    # native SDPA prevents standard custom attention masks bugs and operates faster than eager.
    if uses_custom_attention_backend:
        # ESM++/ESM-C uses its own transformer.attn_backend switch; avoid passing
        # HuggingFace's attention implementation flag into the remote code path.
        pass
    elif HAS_FLASH_ATTN:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        # Simplest stable solution for avoiding custom eager-mode logic leaks
        kwargs["attn_implementation"] = "sdpa"

    try:
        return model_cls.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        if "attn_implementation" in kwargs:
            kwargs.pop("attn_implementation", None)
            return model_cls.from_pretrained(model_name, **kwargs)
        raise


def disable_esm2_token_dropout(model) -> bool:
    """Disable ESM2 token_dropout to work around HuggingFace transformers bug.

    HuggingFace transformers >=5.x broke ESM2's token_dropout: the attention_mask
    is no longer passed to the embeddings layer, causing incorrect scaling of
    embeddings during both training and inference.  See:
    https://github.com/huggingface/transformers/issues/44162

    This sets config.token_dropout = False on the model (and any wrapped inner
    model) so the broken code path is never entered.  Safe to call on any model;
    returns True if a fix was applied.
    """
    fixed = False
    if not needs_esm2_token_dropout_workaround(model):
        return False

    for obj in _iter_model_wrappers(model):
        if obj is None:
            continue
        cfg = getattr(obj, "config", None)
        if (
            cfg is not None
            and getattr(cfg, "model_type", None) == "esm"
            and getattr(cfg, "token_dropout", False)
        ):
            cfg.token_dropout = False
            fixed = True
            logger.info(
                "   Disabled ESM2 token_dropout (HF transformers bug workaround)"
            )
    return fixed


def _iter_model_wrappers(model):
    """Yield model plus common wrapped inner models exactly once."""
    stack = [model]
    seen: set[int] = set()

    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj

        for attr in ("model", "auto_model"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                stack.append(inner)


def is_fastplm_runtime_model(model) -> bool:
    """Return True when a loaded model originates from Synthyra FastPLM code."""
    for obj in _iter_model_wrappers(model):
        cls = obj.__class__
        module_name = getattr(cls, "__module__", "").lower()
        class_name = getattr(cls, "__name__", "").lower()
        if "modeling_fastesm" in module_name or "fastesm" in class_name:
            return True
    return False


def needs_esm2_token_dropout_workaround(model) -> bool:
    """Return True only for the stock HF ESM path affected by the token_dropout bug."""
    if is_fastplm_runtime_model(model):
        return False

    return any(
        getattr(getattr(obj, "config", None), "model_type", None) == "esm"
        and bool(getattr(getattr(obj, "config", None), "token_dropout", False))
        for obj in _iter_model_wrappers(model)
    )


def uses_hf_esm_rotary_embeddings(model) -> bool:
    """Return True when the model contains HF ESM RotaryEmbedding modules.

    This is capability-based rather than family-name-based, so it also catches
    wrappers/reimplementations that directly reuse HF's RotaryEmbedding class.
    """
    for obj in _iter_model_wrappers(model):
        if not isinstance(obj, nn.Module):
            continue
        for module in obj.modules():
            cls = module.__class__
            if (
                getattr(cls, "__name__", "") == "RotaryEmbedding"
                and getattr(cls, "__module__", "")
                == "transformers.models.esm.modeling_esm"
                and hasattr(module, "_seq_len_cached")
            ):
                return True
    return False


def get_torch_compile_settings(model) -> tuple[dict[str, object], bool]:
    """Return model-aware torch.compile kwargs and whether unspec-int should be enabled."""
    raw_dynamic = os.environ.get("PROTEIN_COMPILE_DYNAMIC", "0").strip().lower()
    dynamic = raw_dynamic not in {"0", "false", "no", "off"}
    backend = os.environ.get("PROTEIN_COMPILE_BACKEND", "inductor").strip()
    mode = os.environ.get("PROTEIN_COMPILE_MODE", "default").strip()

    compile_kwargs: dict[str, object] = {"dynamic": dynamic}
    if backend:
        compile_kwargs["backend"] = backend
    if mode and mode != "default":
        compile_kwargs["mode"] = mode

    return compile_kwargs, uses_hf_esm_rotary_embeddings(model)


# =============================================================================
# AMPLIFY helpers
# =============================================================================


def fix_amplify_meta_tensors(model):
    """Recompute freqs_cis if stuck on meta device (happens with from_pretrained)."""
    if hasattr(model, "freqs_cis") and model.freqs_cis.is_meta:
        mod = importlib.import_module(model.__class__.__module__)
        model.freqs_cis = mod.precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.max_length,
        )


def _pad_to_multiple(tensor: torch.Tensor, multiple: int, value=0):
    """Pad last dimension to a multiple of `multiple`. Returns (padded, pad_len)."""
    remainder = tensor.shape[-1] % multiple
    if remainder == 0:
        return tensor, 0
    pad_len = multiple - remainder
    return nn.functional.pad(tensor, (0, pad_len), value=value), pad_len


def _prepare_amplify_inputs(input_ids, attention_mask, device=None):
    """Pad inputs and build additive mask for AMPLIFY/xformers.

    Returns (input_ids, additive_mask, orig_len, pad_len).
    The caller should slice hidden states back to [:, :orig_len, :] after inference.
    """
    orig_len = input_ids.shape[1]

    # xformers cutlass requires stride(-2) % 4 == 0 → pad to multiple of 8
    input_ids, pad_len = _pad_to_multiple(input_ids, 8, value=0)
    if attention_mask is not None:
        attention_mask, _ = _pad_to_multiple(attention_mask, 8, value=0)
    else:
        attention_mask = torch.ones(
            input_ids.shape[0], orig_len, device=input_ids.device
        )
        attention_mask, _ = _pad_to_multiple(attention_mask, 8, value=0)

    # Additive mask: 0.0 for real tokens, -inf for padding
    additive_mask = torch.where(attention_mask.bool(), 0.0, float("-inf"))

    # Match compute dtype (bf16 under autocast, or model param dtype)
    if torch.is_autocast_enabled():
        additive_mask = additive_mask.to(torch.get_autocast_dtype("cuda"))
    if device is not None:
        additive_mask = additive_mask.to(device)

    return input_ids, additive_mask, orig_len, pad_len


# =============================================================================
# ESMplusplus attention backend
# =============================================================================


class _FallbackVocab(dict):
    """Vocabulary dict that returns a fallback id for any unknown token.

    Defined at module level, with ``__reduce__``, so it survives pickling into
    spawned dataloader workers.

    ``__missing__`` fires only on ``[]`` lookup, so ``in``, ``len()`` and
    ``get_vocab()`` still report the true vocabulary -- nothing downstream sees
    a larger vocab than the embedding matrix has rows.
    """

    def __init__(self, mapping, fallback_id):
        super().__init__(mapping)
        self.fallback_id = fallback_id

    def __missing__(self, key):
        return self.fallback_id

    def __reduce__(self):
        return (_FallbackVocab, (dict(self), self.fallback_id))


def patch_unknown_residue_tokens(tokenizer):
    """Map any token missing from the vocabulary onto 'X' instead of raising.

    FastPLM's ``_convert_token_to_id``
    (``fastplms/models/esm2/modeling_fastesm.py:198``) raises ``KeyError`` for an
    out-of-vocabulary token rather than falling back to ``unk_token``. Stock
    ``EsmTokenizer`` does not -- so this only bites checkpoints saved with
    FastPLM's tokenizer identity.

    Two families of offender show up in practice:

    * **residue codes**: 'J' (IUPAC ambiguity code for Leu/Ile) is the one letter
      absent from the ESM2 vocabulary. A single 'J' killed a dataloader worker at
      step 134 of a 4,244-step run, which took down the rank and then the DDP job.
    * **non-residue characters** carried in benchmark sequence fields: '|' in
      Peptide-HLA and '#' in Thermostability (FLIP) each errored a whole task in
      the ESM-2 35M benchmark arm.

    Enumerating A-Z covers the first family only, so the fallback is installed for
    every unknown key instead.

    'X' is used rather than ``<unk>`` deliberately: ESM2 was pretrained with 'X'
    as the unknown-residue symbol and has essentially never seen ``<unk>`` in a
    sequence, so 'X' keeps the input inside the pretraining distribution.

    Idempotent.
    """
    table = getattr(tokenizer, "_token_to_id", None)
    if not isinstance(table, dict) or isinstance(table, _FallbackVocab):
        return
    fallback_id = table.get("X", tokenizer.unk_token_id)
    if fallback_id is None:
        return
    tokenizer._token_to_id = _FallbackVocab(table, fallback_id)
    logger.info(
        "Tokenizer will map out-of-vocabulary tokens (e.g. %s) to 'X' (id %d)",
        ", ".join(repr(c) for c in ascii_uppercase if c not in table) or "'|', '#'",
        fallback_id,
    )


def force_sdpa_backend(model):
    """Set the ESMplusplus / Synthyra attention backend.

    Defaults to ``kernels_flash`` for ESM-C on B300/Hopper-class GPUs. Set
    ``PROTSENT_ESMPLUSPLUS_ATTN_BACKEND=sdpa`` to fall back without editing code.
    Safe to call on any model — no-op when transformer.attn_backend is absent.
    """
    backend = (
        os.environ.get("PROTSENT_ESMPLUSPLUS_ATTN_BACKEND", "kernels_flash").strip()
        or "kernels_flash"
    )
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        # FastPLM ESM2 exposes attn_backend directly on the model, but routes it to
        # HuggingFace's attention interface, which spells flash differently than
        # ESM++ does ("flash_attention_2", not "kernels_flash"). Try the requested
        # name first, then equivalents, so one env var works across both families.
        if hasattr(model, "attn_backend"):
            candidates = {
                "kernels_flash": ("kernels_flash", "flash_attention_2", "sdpa"),
                "flash_attention_2": ("flash_attention_2", "kernels_flash", "sdpa"),
                "flex": ("flex", "flex_attention", "sdpa"),
            }.get(backend, (backend, "sdpa"))
            for candidate in candidates:
                try:
                    model.attn_backend = candidate
                except (ValueError, AssertionError, KeyError, NotImplementedError):
                    continue
                if candidate != backend:
                    logger.info(
                        "   FastPLM does not accept '%s'; using '%s'", backend, candidate
                    )
                logger.info("   Forced FastPLM attention backend: %s", candidate)
                return
            logger.warning(
                "   Could not set any attention backend from %s; leaving model default",
                candidates,
            )
        return
    if hasattr(transformer, "attn_backend"):
        # The ESM++ transformer's attn_backend setter resolves the string to an
        # AttentionBackend enum and propagates it to every block's attention. Do NOT
        # also assign the raw string per block: block.attn._attn compares against the
        # enum, so a leaked string raises "Unsupported resolved backend: <str>".
        transformer.attn_backend = backend
    for block in getattr(transformer, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and backend == "sdpa" and hasattr(attn, "flex_attention"):
            attn.flex_attention = None
    flex_note = " (flex_attention disabled)" if backend == "sdpa" else ""
    logger.info("   Forced ESM++ attention backend: %s%s", backend, flex_note)


# =============================================================================
# Model wrappers
# =============================================================================


class _PLMWrapperBase(nn.Module):
    """Base wrapper for PLMs that need hidden_states extraction.

    Subclasses override _prepare_inputs() to handle model-specific quirks.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def _prepare_inputs(self, input_ids, attention_mask):
        """Override to transform inputs before model call. Returns (kwargs, orig_len)."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }, input_ids.shape[1]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        embedding = outputs.hidden_states[-1][:, :orig_len, :]
        # Must match ESMplusplusWrapper: sentence-transformers >=5.4 reads
        # `last_hidden_state` off the module output, so a bare tuple fails.
        return BaseModelOutput(last_hidden_state=embedding)

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class ESMplusplusWrapper(_PLMWrapperBase):
    """ESMplusplus thin wrapper: extracts last_hidden_state for SentenceTransformer.

    ESM++ (Feb 2026) defaults to SDPA and exposes output.last_hidden_state from
    standard forward(). We simply use that path — no _embed() hack needed.
    force_sdpa_backend() is still called at load time as a safety net.
    """

    def _prepare_inputs(self, input_ids, attention_mask):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }, input_ids.shape[1]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        # last_hidden_state is available directly; fall back to hidden_states tuple
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        else:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        # sentence-transformers >=5.4 Transformer.forward reads `last_hidden_state`
        # from the model output (attribute/index); a bare tuple no longer works.
        # BaseModelOutput also indexes as [0]==last_hidden_state, so older callers
        # remain compatible.
        return BaseModelOutput(last_hidden_state=embedding)


class FastPLMESM2Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's FastPLM ESM2 re-implementation.

    FastPLM ESM2 (Synthyra/ESM2-*) is a bug-fixed ESM2 that correctly handles
    attention_mask in embeddings (unlike HuggingFace transformers >=5.x).
    Identical to ESMplusplusWrapper but raises on missing hidden states.

    Runs the bare encoder (``model.esm``) rather than the ForMaskedLM head, and
    does not ask for ``output_hidden_states``. Only the last layer is ever used
    (see below), so requesting all 13 pinned every intermediate activation for
    the backward pass, and the ``lm_head`` vocabulary projection ran on every
    forward while receiving no gradient. Neither is needed to embed.
    """

    def __init__(self, model):
        super().__init__(model)
        # Keep self.model as the full model so save_pretrained/get_input_embeddings
        # still round-trip the lm_head; only the forward path skips it.
        self._encoder = getattr(model, "esm", None) or model

    def _prepare_inputs(self, input_ids, attention_mask):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, input_ids.shape[1]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self._encoder(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("FastPLM ESM2 model returned no hidden states")
        # Must match ESMplusplusWrapper: sentence-transformers >=5.4 reads
        # `last_hidden_state` off the module output, so a bare tuple fails.
        return BaseModelOutput(last_hidden_state=embedding)


class DPLM2Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's DPLM2 (protein diffusion language model).

    DPLM2 (Synthyra/DPLM2-150M, Synthyra/DPLM2-650M) returns last_hidden_state
    from standard forward(). Uses AutoModel (not AutoModelForMaskedLM) and has
    model.tokenizer for standard HF tokenization.
    """

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("DPLM2 model returned no hidden states")
        # Must match ESMplusplusWrapper: sentence-transformers >=5.4 reads
        # `last_hidden_state` off the module output, so a bare tuple fails.
        return BaseModelOutput(last_hidden_state=embedding)


class ProfluentE1Wrapper(ESMplusplusWrapper):
    """Wrapper for Synthyra's Profluent-E1 (retrieval-augmented protein encoder).

    Profluent-E1 (Synthyra/Profluent-E1-150M, -300M, -600M) returns last_hidden_state
    from standard forward(). Uses AutoModelForMaskedLM and may have model.tokenizer
    or model.prep_tokens for tokenization.
    """

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embedding = outputs.last_hidden_state[:, :orig_len, :]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            embedding = outputs.hidden_states[-1][:, :orig_len, :]
        else:
            raise RuntimeError("Profluent-E1 model returned no hidden states")
        # Must match ESMplusplusWrapper: sentence-transformers >=5.4 reads
        # `last_hidden_state` off the module output, so a bare tuple fails.
        return BaseModelOutput(last_hidden_state=embedding)


class AMPLIFYWrapper(_PLMWrapperBase):
    """AMPLIFY: pad to multiple of 8 + additive attention mask for xformers.

    Also applies the final layer_norm_2 which AMPLIFY omits from
    hidden_states (it is only applied internally before the decoder head).
    """

    def __init__(self, model):
        super().__init__(model)
        fix_amplify_meta_tensors(model)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        model_kwargs, orig_len = self._prepare_inputs(input_ids, attention_mask)
        outputs = self.model(**model_kwargs)
        embedding = outputs.hidden_states[-1][:, :orig_len, :]
        # Apply final layer norm — AMPLIFY excludes it from hidden_states
        if hasattr(self.model, "layer_norm_2"):
            embedding = self.model.layer_norm_2(embedding)
        # Must match ESMplusplusWrapper: sentence-transformers >=5.4 reads
        # `last_hidden_state` off the module output, so a bare tuple fails.
        return BaseModelOutput(last_hidden_state=embedding)

    def _prepare_inputs(self, input_ids, attention_mask):
        input_ids, additive_mask, orig_len, _ = _prepare_amplify_inputs(
            input_ids, attention_mask
        )
        return {
            "input_ids": input_ids,
            "attention_mask": additive_mask,
            "output_hidden_states": True,
        }, orig_len
