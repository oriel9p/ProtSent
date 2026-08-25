#!/usr/bin/env python
"""Emit the ProteinGym results tables as markdown, straight from the CSV + per-group npz.

Regenerate after any rerun so tables can never drift from the data:
    PG_DIR=<dir> uv run --no-sync python report_proteingym.py > <dir>/PROTEINGYM_TABLES.md

Defaults to the quarantined PARTIAL run (500-variant cap, 512 truncation) — see that
directory's README. Point PG_DIR at the benchmarks dir once a full-coverage rerun lands.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ci import boot_ci  # noqa: E402
from late_interaction_eval import corrected_average  # noqa: E402

_B = Path(__file__).resolve().parent / "results/late_interaction/pilot_35m/benchmarks"
# Which directory is authoritative must be decided the SAME way build_late_results.py decides it.
# Keying on Path.exists() disagreed with it the moment a partial rerun dropped a CSV in the
# benchmarks dir: one script called that full coverage, the other still called it partial.
from build_late_results import _proteingym_is_full  # noqa: E402

D = Path(os.environ["PG_DIR"]) if os.environ.get("PG_DIR") else (
    _B if _proteingym_is_full(_B / "proteingym_maxsim.csv") else _B / "proteingym_partial")
VARIANTS = ("dms_substitutions", "dms_indels", "clinical_substitutions", "clinical_indels")
METRIC = {"dms_substitutions": "Spearman", "dms_indels": "Spearman",
          "clinical_substitutions": "AUC", "clinical_indels": "AUC"}

# name -> (display, size, backbone-trained, scoring) — presentation order
ARMS = [
    ("esm2_35m_cosine",             "ESM-2 35M (vanilla)",       "35M",  "pooled cosine"),
    ("esm2_35m_dense",              "ESM-2 35M (vanilla)",       "35M",  "pooled cosine"),
    ("esm2_35m_zeroshot",           "ESM-2 35M (vanilla)",       "35M",  "MaxSim (untrained, 480-D)"),
    ("protsent_v2_zeroshot",        "ProtSent-V2 35M",           "35M",  "MaxSim (untrained, 480-D)"),
    ("esm2_35m_maxsim_untrained",   "ESM-2 35M",                 "35M",  "MaxSim (untrained, 480-D)"),
    ("esm2_late_35m_maxsim",        "ESM-2 35M + late train",    "35M",  "MaxSim (trained, 64-D)"),
    ("protsent_v2_dense",           "ProtSent-V2 35M",           "35M",  "pooled cosine"),
    ("v2_35m_maxsim_untrained",     "ProtSent-V2 35M",           "35M",  "MaxSim (untrained, 480-D)"),
    ("v2p5_35m_cosine",             "ProtSent-V2.5 35M",         "35M",  "pooled cosine"),
    ("proj128_late",                "ProtSent-V2 35M + late 4k", "35M",  "MaxSim (trained, 128-D)"),
    ("protsent_late_35m_prop_dense","ProtSent-V2 35M + late 31k","35M",  "pooled cosine"),
    ("protsent_late_35m_prop_late", "ProtSent-V2 35M + late 31k","35M",  "MaxSim (trained, 128-D)"),
    ("esm2_150m_cosine",            "ESM-2 150M",                "150M", "pooled cosine"),
    ("esm2_150m_maxsim_untrained",  "ESM-2 150M",                "150M", "MaxSim (untrained, 640-D)"),
    ("esm2_late_150m_maxsim",       "ESM-2 150M + late train",   "150M", "MaxSim (trained, 64-D)"),
    ("v2_150m_cosine",              "ProtSent-V2 150M",          "150M", "pooled cosine"),
    ("v2_150m_maxsim_untrained",    "ProtSent-V2 150M",          "150M", "MaxSim (untrained, 640-D)"),
    ("protsent_late_150m_maxsim",   "ProtSent-V2 150M + late 5k","150M", "MaxSim (trained, 64-D)"),
    ("protsent_late_150m_prop_late","ProtSent-V2 150M + late 30k","150M","MaxSim (trained, 128-D)"),
    ("protsent_late_150m_prop_dense","ProtSent-V2 150M + late 30k","150M","pooled cosine"),
]
NAMES = {n: (d, s, sc) for n, d, s, sc in ARMS}


def scored_models(variant):
    """Models with a per-assay npz for this variant, which is written per model as it finishes.

    The CSV only appends when an entire invocation completes, so it under-reports whenever a run
    was interrupted. The npz files are the source of truth; everything below is derived from them.
    """
    pre = f"proteingym_{variant}_"
    return sorted(f.name[len(pre):-4] for f in D.glob(f"{pre}*.npz"))


def pq(variant, model):
    p = D / f"proteingym_{variant}_{model.replace('/', '_')}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    key = "score" if "score" in z else "spearman"
    return dict(zip(z["assay"], z[key]))


def summary(variant, model):
    g = pq(variant, model)
    if not g:
        return None
    v = np.array(list(g.values()), dtype=float)
    mean, lo, hi = boot_ci(v, n_boot=2000, seed=42)
    corr = corrected_average(g) if variant == "dms_substitutions" else float("nan")
    return float(v.mean()), lo, hi, len(v), corr


def paired(variant, a, b):
    x, y = pq(variant, a), pq(variant, b)
    if not x or not y:
        return None
    common = sorted(set(x) & set(y))
    if len(common) < 2:
        return None
    d = np.array([x[k] - y[k] for k in common])
    mean, lo, hi = boot_ci(d, n_boot=2000, seed=42)
    return mean, lo, hi, len(common)


def fmt(delta):
    if delta is None:
        return "—"
    m, lo, hi, n = delta
    star = " \\*" if not (lo < 0 < hi) else ""
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"


print("<!-- generated by report_proteingym.py; regenerate, do not hand-edit -->\n")
if D.name == "proteingym_partial":
    print("> **PARTIAL — quarantined.** 500-variant cap (~4.3% coverage), 512-residue truncation.\n> Absolute values are for ranking our own arms only; never place them beside a published\n> ProteinGym score. Paired deltas are sound. See `README.md` in this directory.\n")
for v in VARIANTS:
    models = scored_models(v)
    if not models:
        continue
    known = [(n, d, s, sc) for n, d, s, sc in ARMS if n in models]
    unknown = [m for m in models if m not in NAMES]
    n_groups = max((summary(v, m) or [0, 0, 0, 0, 0])[3] for m in models)
    print(f"### ProteinGym {v.replace('_', ' ')} — mean {METRIC[v]} across {n_groups} "
          f"{'assays' if 'dms' in v else 'protein groups'}\n")
    corr_col = " ProteinGym-corrected |" if v == "dms_substitutions" else ""
    print("| Model | Size | Scoring | " + METRIC[v] + " | 95% CI |" + corr_col)
    print("|---|---|---|---|---|" + ("---|" if corr_col else ""))
    for name, disp, size, scoring in known:
        s = summary(v, name)
        if not s:
            continue
        mean, lo, hi, n, corr = s
        extra = f" {corr:.4f} |" if corr_col else ""
        print(f"| {disp} | {size} | {scoring} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |{extra}")
    for name in unknown:
        s = summary(v, name)
        if s:
            mean, lo, hi, n, corr = s
            extra = f" {corr:.4f} |" if corr_col else ""
            print(f"| `{name}` (unlabelled) | ? | ? | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |{extra}")
    print()
    deltas = [
        ("esm2_35m_maxsim_untrained", "esm2_35m_cosine",  "MaxSim vs cosine, vanilla ESM-2 35M (same weights)"),
        ("v2_35m_maxsim_untrained",  "protsent_v2_dense", "MaxSim vs cosine, ProtSent-V2 35M (same weights)"),
        ("protsent_late_35m_prop_late", "protsent_late_35m_prop_dense", "MaxSim vs cosine, late-31k 35M (same weights)"),
        ("proj128_late", "v2_35m_maxsim_untrained",       "Trained 128-D head vs untrained 480-D MaxSim (ProtSent-V2 35M)"),
        ("v2_35m_maxsim_untrained", "esm2_35m_maxsim_untrained", "ProtSent-V2 vs vanilla ESM-2, both untrained MaxSim"),
        ("esm2_late_35m_maxsim", "esm2_35m_maxsim_untrained", "ESM-2 late-trained vs untrained MaxSim"),
        ("protsent_late_35m_prop_late", "proj128_late",
         "phase-2 proportional recipe vs phase-1 round-robin arm (5 confounds, NOT step count)"),
        ("v2_150m_maxsim_untrained", "esm2_150m_maxsim_untrained", "ProtSent-V2 vs vanilla ESM-2, untrained MaxSim, 150M"),
        ("protsent_late_150m_prop_late", "v2_150m_maxsim_untrained", "150M late-30k vs untrained MaxSim"),
    ]
    have = [(a, b, lab) for a, b, lab in deltas if a in models and b in models]
    if have:
        print("Paired deltas (identical group sets, bootstrap over groups; \\* = 95% CI excludes zero):\n")
        print("| Comparison | Δ " + METRIC[v] + " |")
        print("|---|---|")
        for a, b, lab in have:
            print(f"| {lab} | {fmt(paired(v, a, b))} |")
        print()
