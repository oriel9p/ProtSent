"""Validate DMS dataset variants."""

import polars as pl
from pathlib import Path


def validate_dms_variant(name: str) -> dict:
    """Validate a DMS variant parquet file."""
    cosent_path = Path("data/dms_cosent.parquet")
    simcse_path = Path("data/dms_sequences.parquet")

    print(f"\n{'=' * 60}")
    print(f"VALIDATING: {name}")
    print(f"{'=' * 60}\n")

    # Load parquets
    df = pl.read_parquet(cosent_path)
    simcse_df = pl.read_parquet(simcse_path)

    # Basic stats
    total_rows = len(df)
    unique_sentence_0 = df["sentence_0"].n_unique()
    unique_sentence_1 = df["sentence_1"].n_unique()
    simcse_rows = len(simcse_df)
    simcse_unique = simcse_df["sequence"].n_unique()

    # Schema validation
    expected_cols = {"sentence_0", "sentence_1", "score"}
    actual_cols = set(df.columns)
    schema_valid = expected_cols == actual_cols

    # Null check
    null_counts = df.null_count()
    has_nulls = any(null_counts.row(0))

    # Score range validation
    min_score = df["score"].min()
    max_score = df["score"].max()
    mean_score = df["score"].mean()
    std_score = df["score"].std()
    scores_valid = min_score >= 0.0 and max_score <= 1.0

    # Print results
    print("CoSENT Parquet Stats:")
    print(f"  Total pairs: {total_rows:,}")
    print(f"  Unique sentence_0 (mutated): {unique_sentence_0:,}")
    print(f"  Unique sentence_1 (target): {unique_sentence_1:,}")
    print(f"  Schema valid: {schema_valid} (expected {expected_cols})")
    print(f"  Has nulls: {has_nulls}")
    print(f"  Score range: [{min_score:.3f}, {max_score:.3f}]")
    print(f"  Score stats: mean={mean_score:.3f}, std={std_score:.3f}")
    print(f"  Scores in [0,1]: {scores_valid}")

    print("\nSimCSE Parquet Stats:")
    print(f"  Total rows: {simcse_rows:,}")
    print(f"  Unique sequences: {simcse_unique:,}")
    print(f"  All unique: {simcse_rows == simcse_unique}")

    # Check that SimCSE contains union of sentence_0 and sentence_1
    all_sequences = pl.concat(
        [
            df.select(pl.col("sentence_0").alias("sequence")),
            df.select(pl.col("sentence_1").alias("sequence")),
        ]
    ).unique()

    expected_simcse = len(all_sequences)
    simcse_coverage = simcse_rows >= expected_simcse
    print(f"  Expected sequences: {expected_simcse:,}")
    print(f"  Coverage valid: {simcse_coverage}")

    # File sizes
    cosent_size_mb = cosent_path.stat().st_size / (1024 * 1024)
    simcse_size_mb = simcse_path.stat().st_size / (1024 * 1024)
    print("\nFile Sizes:")
    print(f"  dms_cosent.parquet: {cosent_size_mb:.1f} MB")
    print(f"  dms_sequences.parquet: {simcse_size_mb:.1f} MB")

    # Overall validation
    all_valid = (
        schema_valid
        and not has_nulls
        and scores_valid
        and simcse_coverage
        and (simcse_rows == simcse_unique)
    )
    print(f"\n{'=' * 60}")
    print(f"OVERALL VALIDATION: {'✓ PASS' if all_valid else '✗ FAIL'}")
    print(f"{'=' * 60}\n")

    return {
        "name": name,
        "total_pairs": total_rows,
        "unique_sentence_0": unique_sentence_0,
        "unique_sentence_1": unique_sentence_1,
        "simcse_sequences": simcse_rows,
        "schema_valid": schema_valid,
        "no_nulls": not has_nulls,
        "scores_valid": scores_valid,
        "simcse_valid": simcse_coverage and (simcse_rows == simcse_unique),
        "all_valid": all_valid,
    }


if __name__ == "__main__":
    import sys

    variant_name = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    stats = validate_dms_variant(variant_name)

    # Exit with error if validation failed
    sys.exit(0 if stats["all_valid"] else 1)
