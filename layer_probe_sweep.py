#!/usr/bin/env python
"""Does the linear-probe result depend on which layer is pooled?

Reviewer HNXd asked for a linear-classifier baseline, and across the benchmark the
linear probe favours stock ESM-2 while the 3-NN probe favours ProtSent. Before
concluding anything from that, one confound has to be ruled out: both probes are
run on the mean-pooled FINAL layer, and it is well established that the last
layer of a masked-LM is not where linearly decodable property information is
richest -- the objective pushes the top of the stack toward token reconstruction.

Contrastive fine-tuning acts on the final-layer geometry, because that is what the
similarity objective sees. So "final-layer linear probe" is the measurement most
favourable to a model whose last layer was never reorganised, and least
informative about what ProtSent changed.

This sweeps the pooled layer for both models on two tasks with different shapes
(one regression, one many-class classification) and reports the probe score per
layer. It answers three things at once:

  * whether the ESM-2 linear-probe advantage is a final-layer artifact
  * where each model's linearly decodable information actually peaks
  * whether ProtSent's reorganisation costs information or only moves it

Usage:
    python layer_probe_sweep.py --tasks stability remote_homology
    python layer_probe_sweep.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

MODELS = {
    "ESM-2-35M": "/storage/models/ESM2-35M",
    "ProtSent-V2": "models/protsent_esm2_35m_v3/final",
}


def mean_pool(hidden: "np.ndarray", mask: "np.ndarray") -> np.ndarray:
    """Mean over real tokens only. Padding must not dilute the average."""
    m = mask[..., None].astype(hidden.dtype)
    return (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)


def embed_layers(model_path: str, seqs: list[str], layers: list[int], batch_size: int = 32):
    """Mean-pooled embeddings at each requested layer. Returns {layer: array}."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from model_utils import patch_unknown_residue_tokens

    patch_unknown_residue_tokens(tok)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()

    out = {l: [] for l in layers}
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            enc = tok(seqs[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=512)
            enc = {k: v.cuda() for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].cpu().numpy()
            for l in layers:
                h = hs[l].float().cpu().numpy()
                out[l].append(mean_pool(h, mask))
    return {l: np.concatenate(v) for l, v in out.items()}


def score(train_x, train_y, test_x, test_y, problem: str) -> float:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(train_x)
    train_x, test_x = sc.transform(train_x), sc.transform(test_x)
    if problem == "regression":
        from scipy.stats import spearmanr
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(train_x, train_y)
        return float(spearmanr(m.predict(test_x), test_y).statistic)
    m = LogisticRegression(max_iter=2000, n_jobs=-1).fit(train_x, train_y)
    return float((m.predict(test_x) == test_y).mean())


def run_task(task: str, max_train: int, max_test: int, batch_size: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_tasks import TASKS as BENCHMARK_TASKS
    from datasets import load_dataset

    cfg = BENCHMARK_TASKS[task]
    ds = load_dataset(cfg.dataset)
    seq_col = list(cfg.input_map.values())[0]
    test_key = "test" if "test" in ds else "valid"

    def take(split, n):
        d = ds[split]
        if len(d) > n:
            d = d.shuffle(seed=0).select(range(n))
        return list(d[seq_col]), np.array(d[cfg.label_col])

    tr_seq, tr_y = take("train", max_train)
    te_seq, te_y = take(test_key, max_test)
    problem = "regression" if cfg.problem_type == "regression" else "classification"
    print(f"\n=== {task} ({problem}) train {len(tr_seq)} test {len(te_seq)} ===", flush=True)

    res = {"task": task, "problem": problem, "n_train": len(tr_seq),
           "n_test": len(te_seq), "by_model": {}}
    for name, path in MODELS.items():
        import json as _json
        cfg_path = Path(path) / "config.json"
        if cfg_path.exists():
            n_layers = _json.loads(cfg_path.read_text())["num_hidden_layers"]
        else:
            # Hub ids have no local config.json; ask transformers rather than guess,
            # because a wrong depth silently probes the wrong layers.
            from transformers import AutoConfig
            n_layers = AutoConfig.from_pretrained(path, trust_remote_code=True).num_hidden_layers
        # hidden_states has n_layers+1 entries (embeddings + each block).
        layers = sorted({n_layers // 3, n_layers // 2, (2 * n_layers) // 3,
                         n_layers - 2, n_layers})
        embs = embed_layers(path, tr_seq + te_seq, layers, batch_size)
        res["by_model"][name] = {}
        for l in layers:
            e = embs[l]
            s = score(e[:len(tr_seq)], tr_y, e[len(tr_seq):], te_y, problem)
            res["by_model"][name][str(l)] = s
            print(f"  {name:12s} layer {l:2d}/{n_layers}: {s:.4f}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", default=["stability", "remote_homology"])
    ap.add_argument("--max_train", type=int, default=8000)
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="results/benchmarks/layer_probe_sweep.json")
    ap.add_argument("--models", nargs="+", default=None, metavar="NAME=PATH",
                    help="override the model set, e.g. ESM-2-150M=Synthyra/ESM2-150M")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    if args.models:
        MODELS.clear()
        MODELS.update(dict(m.split("=", 1) for m in args.models))

    report = [run_task(t, args.max_train, args.max_test, args.batch_size) for t in args.tasks]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    # Mean pooling must ignore padded positions entirely.
    h = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    m = np.array([[1, 1, 0]])
    assert np.allclose(mean_pool(h, m), [[2.0, 2.0]]), mean_pool(h, m)

    # A feature that linearly determines the target must score near 1; noise near 0.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x[:, 0] * 3.0
    assert score(x[:300], y[:300], x[300:], y[300:], "regression") > 0.95
    yn = rng.normal(size=400)
    assert abs(score(x[:300], yn[:300], x[300:], yn[300:], "regression")) < 0.3

    lab = (x[:, 0] > 0).astype(int)
    assert score(x[:300], lab[:300], x[300:], lab[300:], "classification") > 0.9
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
