#!/usr/bin/env python
"""Identity-stratified SCOPe-40 retrieval for the late-interaction arms.

SCOPe-40 cannot be decontaminated at the corpus level (median max identity of a
domain to the pretraining corpus is 0.91), so the honest control is to restrict
to queries whose maximum identity to that corpus is low and check the ranking
still holds. Reads the per-query vectors written by ``late_interaction_eval.py
scope`` — no model is loaded and no GPU is used.

    uv run --no-sync python analyze_late_identity_strata.py --selfcheck
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ci import N_BOOT, boot_ci  # noqa: E402
from scope_identity_correlation import load_identities  # noqa: E402

THRESHOLDS = (0.4, 0.5, 0.7, 1.01)
METRICS = {"hit1": "R@1", "hit10": "R@10", "ap": "MAP"}


def rows_for(npz_dir: Path, level: str, identities: np.ndarray, n_boot: int, seed: int) -> list[dict]:
    out = []
    for path in sorted(npz_dir.glob("per_query_*.npz")):
        arm = path.stem.replace("per_query_", "")
        data = np.load(path)
        elig = data[f"{level}_eligible"].astype(bool)
        for thr in THRESHOLDS:
            sel = elig & (identities < thr)
            if sel.sum() < 10:
                continue
            row = {"arm": arm, "level": level, "max_identity_below": thr,
                   "n_eligible_queries": int(sel.sum())}
            for key, label in METRICS.items():
                mean, lo, hi = boot_ci(data[f"{level}_{key}"][sel], n_boot=n_boot, seed=seed)
                row[label] = round(mean, 4)
                row[f"{label}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_dir", default="results/late_interaction/pilot_35m/scope")
    ap.add_argument("--out", default="results/late_interaction/pilot_35m/scope/scope_identity_strata.csv")
    ap.add_argument("--levels", nargs="+", default=["family", "superfamily", "fold"])
    ap.add_argument("--n_boot", type=int, default=N_BOOT,
                    help="Resamples; defaults to bootstrap_ci.N_BOOT so this table matches "
                         "scope_clean_subset.py's identity strata rather than resampling differently")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        # identity thresholds must be nested, and a threshold above every identity
        # must reproduce the unrestricted eligible count.
        ident = np.array([0.1, 0.45, 0.6, 0.95])
        counts = [int((ident < t).sum()) for t in THRESHOLDS]
        assert counts == sorted(counts), counts
        assert counts[-1] == len(ident)
        print("selfcheck ok")
        return 0

    from late_interaction import load_scope40

    seqs, _ = load_scope40()
    identities = load_identities(seqs)
    print(f"{len(seqs)} queries; median max identity to the pretraining corpus "
          f"{np.median(identities):.3f}")

    rows: list[dict] = []
    for level in args.levels:
        rows += rows_for(Path(args.npz_dir), level, identities, args.n_boot, args.seed)
    if not rows:
        raise SystemExit(f"no per_query_*.npz found under {args.npz_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
