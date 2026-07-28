#!/usr/bin/env python3
"""Test script to quickly load and validate Synthyra models.

Tests ESMplusplus_small, DPLM2-150M, and Profluent-E1-150M to determine
which one works best for overnight training.
"""

import logging
import sys
import traceback

import torch

from protein_pipeline import load_model_for_training

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test sequence
TEST_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN"

MODELS_TO_TEST = [
    ("Synthyra/ESMplusplus_small", 300, 960),  # (model_name, params_M, hidden_size)
    ("Synthyra/DPLM2-150M", 150, 768),
    ("Synthyra/Profluent-E1-150M", 150, 768),
]


def test_model(model_name: str, expected_params_m: int, expected_hidden_size: int):
    """Test a single model: load, tokenize, forward pass, check embedding shape."""
    logger.info("=" * 80)
    logger.info(f"Testing: {model_name}")
    logger.info(
        f"Expected: {expected_params_m}M params, hidden_size={expected_hidden_size}"
    )
    logger.info("=" * 80)

    try:
        # 1. Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model to {device}...")
        model = load_model_for_training(model_name, max_seq_length=512, device=device)
        logger.info("✅ Model loaded successfully")

        # 2. Check embedding dimension
        first_module = list(model._modules.values())[0]
        actual_hidden_size = first_module.get_word_embedding_dimension()
        if actual_hidden_size != expected_hidden_size:
            logger.warning(
                f"⚠️  Hidden size mismatch: expected {expected_hidden_size}, got {actual_hidden_size}"
            )
        else:
            logger.info(f"✅ Hidden size correct: {actual_hidden_size}")

        # 3. Encode test sequence
        logger.info("Testing encoding on sample sequence...")
        with torch.no_grad():
            embeddings = model.encode(
                [TEST_SEQUENCE], convert_to_tensor=True, show_progress_bar=False
            )

        # 4. Check output shape
        expected_shape = (1, actual_hidden_size)
        actual_shape = tuple(embeddings.shape)
        if actual_shape != expected_shape:
            logger.error(
                f"❌ Output shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
            return False

        logger.info(f"✅ Embedding shape correct: {actual_shape}")
        logger.info(
            f"✅ Embedding stats: mean={embeddings.mean().item():.6f}, std={embeddings.std().item():.6f}"
        )

        # 5. Test batch encoding
        logger.info("Testing batch encoding (2 sequences)...")
        with torch.no_grad():
            batch_embeddings = model.encode(
                [TEST_SEQUENCE, TEST_SEQUENCE[:50]],
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        expected_batch_shape = (2, actual_hidden_size)
        actual_batch_shape = tuple(batch_embeddings.shape)
        if actual_batch_shape != expected_batch_shape:
            logger.error(
                f"❌ Batch shape mismatch: expected {expected_batch_shape}, got {actual_batch_shape}"
            )
            return False

        logger.info(f"✅ Batch embedding shape correct: {actual_batch_shape}")

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✅✅✅ {model_name} PASSED ALL TESTS ✅✅✅")
        logger.info("=" * 80)
        logger.info("")
        return True

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"❌❌❌ {model_name} FAILED ❌❌❌")
        logger.error(f"Error: {type(e).__name__}: {e}")
        logger.error("=" * 80)
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error("")
        return False


def main():
    """Test all models and report results."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Synthyra Model Test Suite")
    logger.info("=" * 80)
    logger.info("")

    results = {}
    for model_name, params_m, hidden_size in MODELS_TO_TEST:
        success = test_model(model_name, params_m, hidden_size)
        results[model_name] = success

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {model_name}")

    passed = sum(results.values())
    total = len(results)
    logger.info("")
    logger.info(f"Total: {passed}/{total} models passed")
    logger.info("=" * 80)
    logger.info("")

    if passed == 0:
        logger.error("❌ No models passed! Cannot proceed with training.")
        sys.exit(1)
    elif passed < total:
        logger.warning(
            f"⚠️  Only {passed}/{total} models passed. Proceeding with working models."
        )
        # Pick the first working model
        working_model = next((m for m, s in results.items() if s), None)
        logger.info(f"✅ Recommended model for training: {working_model}")
    else:
        logger.info(
            "✅ All models passed! ESMplusplus_small recommended (largest, most tested)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
