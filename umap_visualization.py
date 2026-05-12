"""
UMAP visualization of protein embeddings from baseline vs trained model.

Loads SCOPe-40 sequences (held-out evaluation data), encodes with both models,
runs UMAP, and saves colored scatter plots (by structural superfamily).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer


def load_scope_sample(n_families: int = 50, per_family: int = 30, seed: int = 42,
                      level: str = "fold"):
    """Load a diverse sample of SCOPe-40 sequences grouped by structural hierarchy.

    Args:
        level: SCOPe hierarchy level to group by.
            'superfamily' — e.g. a.1.1  (finest, original 'family' column)
            'fold'        — e.g. a.1    (recommended for UMAP)
            'class'       — e.g. a      (coarsest: all-alpha, all-beta, etc.)
    """
    from datasets import load_dataset

    print("Loading SCOPe-40 dataset from HuggingFace (tattabio/scope40_test)...")
    ds = load_dataset("tattabio/scope40_test", split="train")
    df = ds.to_pandas()

    seq_col = "sequence"

    # Extract the desired hierarchy level from the superfamily label (e.g. "a.1.1.2")
    # class = 1 part (a), fold = 2 parts (a.1), superfamily = 3 parts (a.1.1)
    n_parts = {"class": 1, "fold": 2, "superfamily": 3}
    if level not in n_parts:
        raise ValueError(f"Unknown level '{level}'. Choose from: {list(n_parts.keys())}")

    k = n_parts[level]
    df["group_label"] = df["family"].apply(lambda x: ".".join(x.split(".")[:k]))
    group_col = "group_label"

    print(f"SCOPe-40: {len(df)} proteins, {df[group_col].nunique()} unique {level}-level groups")

    # Get top N groups by count
    top_groups = df[group_col].value_counts().head(n_families).index.tolist()
    df_filtered = df[df[group_col].isin(top_groups)]

    # Sample up to per_family sequences from each group
    rng = np.random.default_rng(seed)
    pieces = []
    for grp in top_groups:
        subset = df_filtered[df_filtered[group_col] == grp]
        n = min(len(subset), per_family)
        pieces.append(subset.sample(n=n, random_state=int(rng.integers(1e9))))
    sampled = pd.concat(pieces, ignore_index=True)

    print(f"Sampled {len(sampled)} sequences from {sampled[group_col].nunique()} {level}-level groups")

    return sampled[seq_col].tolist(), sampled[group_col].tolist()


def load_parquet_sample(parquet_path: str, n_families: int = 100, per_family: int = 50, seed: int = 42):
    """Load a diverse sample from a local parquet file grouped by family."""
    df = pd.read_parquet(parquet_path)

    # Identify the family column
    family_col = None
    for col in ["family_id", "family", "Family", "pfam_family", "clan", "clan_id", "label", "accession"]:
        if col in df.columns:
            family_col = col
            break
    if family_col is None:
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() > 10:
                family_col = col
                break
    if family_col is None:
        raise ValueError(f"Cannot find family column. Columns: {list(df.columns)}")

    print(f"Using family column: '{family_col}' ({df[family_col].nunique()} unique families)")

    top_families = df[family_col].value_counts().head(n_families).index.tolist()
    df_filtered = df[df[family_col].isin(top_families)]

    rng = np.random.default_rng(seed)
    pieces = []
    for fam in top_families:
        subset = df_filtered[df_filtered[family_col] == fam]
        n = min(len(subset), per_family)
        pieces.append(subset.sample(n=n, random_state=int(rng.integers(1e9))))
    sampled = pd.concat(pieces, ignore_index=True)

    # Identify the sequence column
    seq_col = None
    for col in ["sequence", "Sequence", "seq", "protein_sequence", "aa_seq"]:
        if col in sampled.columns:
            seq_col = col
            break
    if seq_col is None:
        raise ValueError(f"Cannot find sequence column. Columns: {list(sampled.columns)}")

    print(f"Sampled {len(sampled)} sequences from {sampled[family_col].nunique()} families")
    return sampled[seq_col].tolist(), sampled[family_col].tolist()


def encode_sequences(model_name_or_path: str, sequences: list, batch_size: int = 64):
    """Encode sequences with a SentenceTransformer model."""
    print(f"Loading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
    model.max_seq_length = 512

    print(f"Encoding {len(sequences)} sequences...")
    embeddings = model.encode(sequences, batch_size=batch_size, show_progress_bar=True, device="cuda")
    return embeddings


def run_umap(embeddings: np.ndarray, n_components: int = 2, seed: int = 42):
    """Run UMAP dimensionality reduction."""
    print(f"Running UMAP on {embeddings.shape} embeddings...")
    reducer = umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=15, min_dist=0.1)
    projection = reducer.fit_transform(embeddings)
    return projection


def plot_umap(projection: np.ndarray, families: list, title: str, output_path: str, top_k_colored: int = 10):
    """Plot UMAP with top families colored distinctly, rest in gray."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    families_arr = np.array(families)
    unique_families, counts = np.unique(families_arr, return_counts=True)
    top_families = unique_families[np.argsort(-counts)][:top_k_colored]

    cmap = plt.cm.get_cmap("tab10", top_k_colored)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot gray points first (non-top families)
    mask_other = ~np.isin(families_arr, top_families)
    if mask_other.any():
        ax.scatter(
            projection[mask_other, 0], projection[mask_other, 1],
            c="lightgray", s=5, alpha=0.3, label="Other", rasterized=True
        )

    # Plot top families with distinct colors
    for i, fam in enumerate(top_families):
        mask = families_arr == fam
        ax.scatter(
            projection[mask, 0], projection[mask, 1],
            c=[cmap(i)], s=12, alpha=0.7, label=fam, rasterized=True
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=7, markerscale=2, ncol=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(proj_baseline, proj_trained, families, output_path,
                    baseline_title="Baseline (ESM-2)", trained_title="Trained (ProtSentBERT)",
                    top_k_colored=10):
    """Side-by-side comparison plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    families_arr = np.array(families)
    unique_families, counts = np.unique(families_arr, return_counts=True)
    top_families = unique_families[np.argsort(-counts)][:top_k_colored]

    cmap = plt.cm.get_cmap("tab10", top_k_colored)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, proj, title in zip(axes, [proj_baseline, proj_trained], [baseline_title, trained_title]):
        mask_other = ~np.isin(families_arr, top_families)
        if mask_other.any():
            ax.scatter(
                proj[mask_other, 0], proj[mask_other, 1],
                c="lightgray", s=5, alpha=0.3, label="Other", rasterized=True
            )

        for i, fam in enumerate(top_families):
            mask = families_arr == fam
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=[cmap(i)], s=12, alpha=0.7, label=fam, rasterized=True
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # Single legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, markerscale=2, bbox_to_anchor=(0.99, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of protein embeddings")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model name or path")
    parser.add_argument("--trained", type=str, required=True, help="Trained model path")
    parser.add_argument("--data", type=str, default="scope",
                        help="Data source: 'scope' for SCOPe-40 from HuggingFace (default), "
                             "or path to a local parquet file")
    parser.add_argument("--output_dir", type=str, default="results/umap", help="Output directory")
    parser.add_argument("--level", type=str, default="fold", choices=["class", "fold", "superfamily"],
                        help="SCOPe hierarchy level for grouping (default: fold)")
    parser.add_argument("--n_families", type=int, default=50, help="Number of groups to sample")
    parser.add_argument("--per_family", type=int, default=30, help="Max sequences per group")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.data == "scope":
        sequences, families = load_scope_sample(
            n_families=args.n_families, per_family=args.per_family, seed=args.seed,
            level=args.level
        )
        data_label = f"SCOPe-40 {args.level}"
    else:
        sequences, families = load_parquet_sample(
            args.data, n_families=args.n_families, per_family=args.per_family, seed=args.seed
        )
        data_label = Path(args.data).stem

    # Extract model short names for titles
    baseline_name = Path(args.baseline).name if "/" not in args.baseline or "facebook" not in args.baseline \
        else args.baseline.split("/")[-1].replace("esm2_", "ESM-2 ").replace("_UR50D", "")
    trained_name = "ProtSentBERT"

    # Encode with both models
    emb_baseline = encode_sequences(args.baseline, sequences, batch_size=args.batch_size)
    emb_trained = encode_sequences(args.trained, sequences, batch_size=args.batch_size)

    # Run UMAP
    proj_baseline = run_umap(emb_baseline, seed=args.seed)
    proj_trained = run_umap(emb_trained, seed=args.seed)

    # Plot individual
    plot_umap(
        proj_baseline, families,
        title=f"UMAP — {baseline_name} ({data_label})",
        output_path=os.path.join(args.output_dir, "umap_baseline.png")
    )
    plot_umap(
        proj_trained, families,
        title=f"UMAP — {trained_name} ({data_label})",
        output_path=os.path.join(args.output_dir, "umap_trained.png")
    )

    # Plot comparison
    plot_comparison(
        proj_baseline, proj_trained, families,
        output_path=os.path.join(args.output_dir, "umap_comparison.png"),
        baseline_title=f"{baseline_name} ({data_label})",
        trained_title=f"{trained_name} ({data_label})",
    )

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
