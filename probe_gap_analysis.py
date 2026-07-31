"""Why contrastive fine-tuning helps k-NN retrieval but not a trained linear probe.

Four measurements, one code path per arm, all label-free except the probes:

  spectrum  Anisotropy / effective rank of mean-pooled final-layer embeddings.
            Tests the "contrastive causes dimensional collapse" hypothesis.
  metric    Remote-homology 3-NN under euclidean (the benchmark default) vs
            cosine. Tests whether the k-NN win is a distance-metric artifact.
  whiten    Remote-homology 3-NN after PCA-whitening -- an invertible linear map
            fitted WITHOUT labels. A linear probe is invariant to such maps;
            k-NN is not. Tests whether contrastive FT moved information or added
            it.
  scope     The same whitening control on SCOPe-40 retrieval, the paper's
            headline task, scored through the suite's own evaluate_retrieval.

Analysis written up in rebuttal/ANALYSIS_probe_gap.md.

    python probe_gap_analysis.py --out results/benchmarks/probe_gap_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

ARMS = {
    "ESM-2-35M": "/storage/models/ESM2-35M",
    "ProtSent-V2-35M": "models/protsent_esm2_35m_v3/final",
    "ESM-2-150M": "Synthyra/ESM2-150M",
    "ProtSent-V2-150M": "models/protsent_esm2_150m_v2/final",
}


def load(path):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path, device="cuda", trust_remote_code=True)


def encode(model, seqs):
    return np.asarray(
        model.encode(seqs, batch_size=64, convert_to_numpy=True), dtype=np.float64
    )


def whiten(fit, *apply, eps=1e-6, keep=None):
    """PCA-whitening fitted on `fit`, applied to each of `apply`.

    Invertible linear map (up to the eps floor and any truncation), so a linear
    probe's hypothesis class is unchanged by it. k-NN's is not.
    """
    mu = fit.mean(0)
    _, s, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    n = keep or len(s)
    V, scale = Vt[:n].T, np.sqrt(len(fit) - 1) / (s[:n] + eps * s[0])
    return [(X - mu) @ V * scale for X in apply]


# ── 1. spectrum ──────────────────────────────────────────────────────────────
def spectrum_stats(X):
    n, d = X.shape
    out = {"n": n, "d": d}
    norms = np.linalg.norm(X, axis=1)
    out["norm_mean"] = float(norms.mean())
    out["norm_cv"] = float(norms.std() / norms.mean())

    U = X / norms[:, None]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = i != j
    out["mean_cos_random_pair"] = float((U[i[keep]] * U[j[keep]]).sum(1).mean())

    # Matryoshka front-loading: variance in the leading COORDINATES (not the
    # leading singular directions) vs a random block of the same width.
    v = X.var(0)
    rng2 = np.random.default_rng(1)
    rand64 = np.mean([v[rng2.permutation(d)[:64]].sum() for _ in range(200)])
    out["var_frac_first64"] = float(v[:64].sum() / v.sum())
    out["var_frac_random64"] = float(rand64 / v.sum())
    out["front_load_ratio"] = out["var_frac_first64"] / out["var_frac_random64"]

    for tag, M in (
        ("centered", X - X.mean(0)),
        ("standardized", (X - X.mean(0)) / X.std(0)),  # what StandardScaler hands the probe
    ):
        s = np.linalg.svd(M, compute_uv=False)
        lam = s**2
        p = lam / lam.sum()
        p = p[p > 0]
        out[f"{tag}_eff_rank_entropy"] = float(np.exp(-(p * np.log(p)).sum()))
        out[f"{tag}_participation_ratio"] = float(lam.sum() ** 2 / (lam**2).sum())
        out[f"{tag}_var_top1"] = float(p[0])
        out[f"{tag}_var_top10"] = float(p[:10].sum())
        out[f"{tag}_n_dims_for_95pct"] = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
        out[f"{tag}_cond_1_to_50"] = float(s[0] / s[min(49, len(s) - 1)])
    return out


# ── 2 & 3. remote homology: metric control + whitening control ───────────────
def homology_probes(A, B, ytr, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def sc(pred):
        return {
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        }

    def knn(a, b, metric="cosine"):
        return sc(KNeighborsClassifier(3, metric=metric, n_jobs=-1).fit(a, ytr).predict(b))

    ln = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    r = {
        "knn_euclidean_raw": knn(A, B, "minkowski"),  # what the benchmark actually ran
        "knn_cosine_raw": knn(A, B),
        "knn_euclidean_l2normed": knn(ln(A), ln(B), "minkowski"),
    }
    Aw, Bw = whiten(A, A, B)
    r["knn_whitened_full"] = knn(Aw, Bw)
    for k in (64, 128, 256):
        if k < A.shape[1]:
            Ak, Bk = whiten(A, A, B, keep=k)
            r[f"knn_whitened_top{k}"] = knn(Ak, Bk)
    mu, sd = A.mean(0), A.std(0)  # diagonal rescale only: no rotation
    r["knn_standardized"] = knn((A - mu) / sd, (B - mu) / sd)
    r["linear"] = sc(
        OneVsRestClassifier(
            make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear")),
            n_jobs=-1,
        )
        .fit(A, ytr)
        .predict(B)
    )
    return r


# ── 4. SCOPe-40 retrieval under the same whitening control ──────────────────
def scope_retrieval(X, labels):
    from protein_benchmark_suite import evaluate_retrieval

    keep = ["eligible_Recall@1", "eligible_Recall@10", "eligible_MAP", "Recall@1", "MAP"]
    out = {}
    variants = [("raw", X), ("whitened_full", whiten(X, X)[0])]
    variants += [(f"whitened_top{k}", whiten(X, X, keep=k)[0]) for k in (64, 128)]
    for tag, E in variants:
        m = evaluate_retrieval(E, labels, k_list=[1, 10, 30])
        out[tag] = {k: round(m[k], 4) for k in keep}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmarks/probe_gap_analysis.json")
    args = ap.parse_args()

    from datasets import load_dataset

    scope = load_dataset("tattabio/scope40_test", split="train")
    scope_seqs, scope_labels = list(scope["sequence"]), list(scope["family"])
    tr = load_dataset("biomap-research/fold_prediction", split="train")
    te = load_dataset("biomap-research/fold_prediction", split="test")
    tr_s, ytr = list(tr["seq"]), np.array(tr["label"])
    te_s, yte = list(te["seq"]), np.array(te["label"])
    print(
        f"SCOPe-40 {len(scope_seqs)} domains / {len(set(scope_labels))} families; "
        f"remote homology {len(ytr)} train / {len(yte)} test / "
        f"{len(set(ytr.tolist()) | set(yte.tolist()))} classes",
        flush=True,
    )

    res = {}
    for name, path in ARMS.items():
        print(f"\n=== {name} ===", flush=True)
        model = load(path)
        S = encode(model, scope_seqs)
        A, B = encode(model, tr_s), encode(model, te_s)
        del model

        res[name] = {
            "spectrum_scope40": spectrum_stats(S),
            "remote_homology": homology_probes(A, B, ytr, yte),
            "scope40_retrieval": scope_retrieval(S, scope_labels),
        }
        sp, rh = res[name]["spectrum_scope40"], res[name]["remote_homology"]
        print(
            f"  mean random-pair cosine {sp['mean_cos_random_pair']:.3f}   "
            f"eff.rank {sp['centered_eff_rank_entropy']:.1f}   "
            f"PR {sp['centered_participation_ratio']:.1f}",
            flush=True,
        )
        print(
            f"  remote homology acc: 3-NN raw {rh['knn_cosine_raw']['accuracy']:.4f} "
            f"-> whitened {rh['knn_whitened_full']['accuracy']:.4f} "
            f"| linear {rh['linear']['accuracy']:.4f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
