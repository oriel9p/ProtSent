#!/usr/bin/env python
"""Does the identity-vs-gain null survive controlling for baseline headroom?

`scope_identity_correlation.py` reports a null-to-negative correlation between a
query's maximum identity to the pretraining corpus and that query's retrieval
gain, and argues this is evidence against memorization.

A blind reader raised the obvious confound: high-identity queries are already
well solved by the baseline, so they have less headroom, and regression to the
mean alone would produce a flat-or-negative slope with no memorization story
either way. The gain and the baseline score are not independent -- gain is
bounded above by (1 - baseline).

Two controls, both on the same per-query vectors:

  partial correlation   Spearman between identity and gain after residualising
                        both on the baseline score (rank-transform, then remove
                        the linear component of each on baseline rank). If the
                        null is an artifact of headroom, removing baseline
                        should reveal a positive residual slope.

  matched strata        Spearman between identity and gain computed WITHIN each
                        baseline-score quartile, where headroom is roughly
                        constant. A memorization effect should show up as a
                        positive within-stratum correlation.

Also reports normalised gain, gain / (1 - baseline), which is the fraction of
the available headroom that was captured -- the quantity that is comparable
across queries with different baselines.

Usage:
    python scope_identity_partial.py --json results/benchmarks/scope_identity_correlation_v2.json
    python scope_identity_partial.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman between x and y, controlling for z. Returns (r, p)."""
    from scipy.stats import rankdata, spearmanr

    rx, ry, rz = (rankdata(v) for v in (x, y, z))

    def resid(a, b):
        b = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ coef

    return spearmanr(resid(rx, rz), resid(ry, rz))


def analyse(ident: np.ndarray, base: np.ndarray, gain: np.ndarray, label: str) -> dict:
    from scipy.stats import spearmanr

    raw_r, raw_p = spearmanr(ident, gain)
    par_r, par_p = partial_spearman(ident, gain, base)

    # Headroom-normalised gain. Queries the baseline already solves perfectly have
    # no headroom and are undefined here, so they are excluded rather than clipped.
    head = 1.0 - base
    ok = head > 1e-9
    norm_r, norm_p = spearmanr(ident[ok], (gain[ok] / head[ok]))

    out = {
        "metric": label,
        "raw": {"spearman_r": float(raw_r), "p": float(raw_p)},
        "partial_controlling_for_baseline": {"spearman_r": float(par_r), "p": float(par_p)},
        "headroom_normalised": {"spearman_r": float(norm_r), "p": float(norm_p),
                                "n": int(ok.sum())},
        "strata": [],
    }
    print(f"\n{label}")
    print(f"  raw                       spearman {raw_r:+.4f}  p={raw_p:.3g}")
    print(f"  partial (control base)    spearman {par_r:+.4f}  p={par_p:.3g}")
    print(f"  headroom-normalised gain  spearman {norm_r:+.4f}  p={norm_p:.3g}  (n={int(ok.sum())})")

    q = np.quantile(base, [0.25, 0.5, 0.75])
    edges = [(-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]
    print(f"  {'baseline quartile':22s} {'n':>5s} {'mean gain':>10s} {'spearman':>9s} {'p':>9s}")
    for lo, hi in edges:
        sel = (base >= lo) & (base < hi)
        if sel.sum() < 20:
            continue
        r, p = spearmanr(ident[sel], gain[sel])
        out["strata"].append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                              "mean_gain": float(gain[sel].mean()),
                              "spearman_r": float(r), "p": float(p)})
        print(f"  [{lo:6.3f}, {hi:6.3f})      {int(sel.sum()):5d} {gain[sel].mean():10.4f} "
              f"{r:+9.4f} {p:9.3g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs=2, metavar=("BASELINE", "PROTSENT"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/storage/models/hf_home")
    sys.path.insert(0, str(Path(__file__).parent))
    from scope_identity_correlation import compute_per_query, embed, load_identities, load_scope

    seqs, labels = load_scope()
    ident = load_identities(seqs)
    base = compute_per_query(embed(args.models[0], seqs, args.batch_size), labels)
    ps = compute_per_query(embed(args.models[1], seqs, args.batch_size), labels)

    el = base["eligible"]
    print(f"{int(el.sum())} eligible queries; identity median {np.median(ident[el]):.3f}")

    report = {"baseline": args.models[0], "protsent": args.models[1],
              "n_eligible": int(el.sum()), "metrics": []}
    for metric in ("hit10", "ap"):
        report["metrics"].append(
            analyse(ident[el], base[metric][el], (ps[metric] - base[metric])[el], metric)
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    n = 4000

    # Case 1: pure headroom artifact. Gain depends ONLY on headroom; identity is
    # correlated with baseline but carries no independent signal. The raw
    # correlation must be negative and the partial must collapse toward zero.
    base = rng.uniform(0, 0.95, n)
    ident = np.clip(base + rng.normal(0, 0.1, n), 0, 1)
    gain = 0.3 * (1 - base) + rng.normal(0, 0.01, n)
    raw = analyse(ident, base, gain, "selfcheck: headroom-only")
    assert raw["raw"]["spearman_r"] < -0.3, raw["raw"]
    assert abs(raw["partial_controlling_for_baseline"]["spearman_r"]) < 0.15, raw

    # Case 2: real memorization. Gain rises with identity ON TOP of headroom, so
    # the partial correlation must stay clearly positive.
    gain2 = 0.3 * (1 - base) + 0.4 * ident + rng.normal(0, 0.01, n)
    mem = analyse(ident, base, gain2, "selfcheck: memorization present")
    assert mem["partial_controlling_for_baseline"]["spearman_r"] > 0.3, mem

    print("\nselfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
