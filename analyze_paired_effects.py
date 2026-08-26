#!/usr/bin/env python
"""Paired effect sizes for the late-interaction section, from saved per-query vectors.

Every headline number in that section is a difference between two models scored on the SAME
SCOPe-40 queries, so the marginal CIs printed in scope_hierarchy.csv are the wrong test: they
ignore the pairing and badly understate power. Two arms whose marginal CIs overlap can still
differ significantly on a paired bootstrap, and several of ours do.

Reads the per_query_*.npz files written by late_interaction_eval.py (they are the interchange
format) and prints one row per contrast. No GPU, no model loading -- pure re-analysis, so the
table can be regenerated from the repo at any time.

    python analyze_paired_effects.py [--level superfamily] [--n_boot 5000]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import late_interaction as li  # noqa: E402

# (a, b, label) -> reported effect is a - b.
CONTRASTS = [
    ("protsent_v2_zeroshot", "esm2_zeroshot",
     "ProtSent-V2 pretraining (35M, raw MaxSim)"),
    ("protsent_v2_150m_zeroshot", "esm2_150m_zeroshot",
     "ProtSent-V2 pretraining (150M, raw MaxSim)"),
    ("protsent_v2_150m_zeroshot", "protsent_v2_zeroshot",
     "size 35M->150M (V2, raw MaxSim)"),
    ("esm2_zeroshot", "esm2_dense",
     "MaxSim - cosine, same ESM2-35M weights"),
    ("protsent_v2_150m_zeroshot", "protsent_v2_150m_dense",
     "MaxSim - cosine, same V2-150M weights"),
    ("late-r2-protsentv2-150m@10000", "protsent_v2_150m_zeroshot",
     "late-trained 128d - untrained raw 640d (V2-150M)"),
    ("late-r2-esm2-150m@8000", "protsent_v2_150m_zeroshot",
     "late-trained 128d ESM2 init - untrained raw 640d V2"),
    ("late-r2-protsentv2-150m@0", "protsent_v2_150m_zeroshot",
     "cost of an untrained 128d head (V2-150M)"),
    ("late-r2-esm2-150m@8000", "late-r2-protsentv2-150m@10000",
     "init: ESM2 - ProtSent-V2 (150M, best vs best)"),
    ("late-r2-esm2-150m@8000", "late-r2-esm2-150m@1000",
     "more training: ESM2-150M 8000 - 1000"),
    ("vanilla35m_clean@8000", "vanilla35m_clean@1000",
     "more training: ESM2-35M 8000 - 1000"),
    ("late-r2-protsentv2-150m@10000", "late-r2-protsentv2-150m@1000",
     "more training: V2-150M 10000 - 1000"),
    ("late-r2-esm2-150m@8000", "vanilla35m_clean@8000",
     "size: 150M - 35M (ESM2 init, step 8000)"),
    # Filled in by close_gaps.sh; absent until then and reported as MISSING rather than skipped.
    ("esm2_150m_zeroshot", "esm2_zeroshot",
     "size 35M->150M (ESM2, raw MaxSim) [mechanism prediction: ~0]"),
]


def load_per_query(roots):
    out = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "per_query_*.npz")):
            name = os.path.basename(f)[len("per_query_"):-len(".npz")]
            out[name] = np.load(f, allow_pickle=True)
    return out


def level_view(z, level):
    return {m: z[f"{level}_{m}"] for m in ("ap", "hit1", "hit10", "eligible")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="superfamily", choices=list(li.SCOPE_LEVELS))
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--roots", nargs="*", default=[
        "results/late_interaction/pilot_35m/scope",
        "results/late_interaction/clean_35m/scope",
        "results/late_interaction/clean_150m/scope",
    ])
    args = ap.parse_args()

    D = load_per_query(args.roots)
    print(f"level={args.level}  n_boot={args.n_boot}  models_found={len(D)}\n")
    print(f"{'effect':<52}{'dAP':>9}{'CI95':>22}{'sig':>6}{'n':>7}")
    print("-" * 96)
    missing = []
    for a, b, label in CONTRASTS:
        if a not in D or b not in D:
            missing.append((label, a if a not in D else b))
            print(f"{label:<52}{'MISSING':>9}")
            continue
        r = li.paired_bootstrap(level_view(D[a], args.level), level_view(D[b], args.level),
                                n_boot=args.n_boot, seed=args.seed)["ap"]
        sig = "YES" if (r["ci95_lo"] > 0 or r["ci95_hi"] < 0) else "no"
        print(f"{label:<52}{r['delta']:>+9.4f}  [{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]"
              f"{sig:>6}{r['n']:>7}")
    for label, who in missing:
        print(f"\nMISSING: {label!r} needs per_query_{who}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
