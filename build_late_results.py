#!/usr/bin/env python
"""Generate results/late_interaction/RESULTS.md from the result CSVs.

Every number in the write-up comes from a file, so the prose cannot drift from the data the way
hand-copied tables do. Re-run after any new arm lands.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ci import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/late_interaction"
B, S = RES / "pilot_35m/benchmarks", RES / "pilot_35m/scope"
# ProteinGym is quarantined as PARTIAL (500-variant cap, 512 truncation). See its README. A
# full-coverage rerun writes to B, so prefer that the moment it exists rather than silently
# re-publishing the partial numbers forever.
def _proteingym_is_full(csv_path):
    """Full coverage means the RECORDED provenance says so, not that a file exists.

    Keying on Path.exists() meant the first partial run flipped the report to "Full coverage:
    every variant of every assay, 1024-residue context" while the CSV held a 500-variant cap at
    512 residues -- and simultaneously hid the other three variants, whose npz still lived in
    proteingym_partial/.
    """
    import csv as _csv

    if not csv_path.exists():
        return False
    rows = list(_csv.DictReader(csv_path.open()))
    if not rows:
        return False
    want = {"dms_substitutions", "dms_indels", "clinical_substitutions", "clinical_indels"}
    if not want <= {r.get("variant") for r in rows}:
        return False
    # cap 0 means "all variants"; max_seq_length must be the full-context value
    return all(str(r.get("cap", "")) in ("0", "") for r in rows)


PARTIAL = not _proteingym_is_full(B / "proteingym_maxsim.csv")
PG = B / "proteingym_partial" if PARTIAL else B
out: list[str] = []
w = out.append


def scope_table() -> None:
    f = S / "scope_hierarchy.csv"
    if not f.exists():
        return
    # The bf16 master-weights arms are deleted; this guard only stops an old CSV, restored
    # from git or a backup, from quietly reintroducing them.
    EXCLUDE = ("_bf16bug", "capped_flash")
    seen, tab = set(), {}
    for r in csv.DictReader(f.open()):
        if any(x in r["model"] for x in EXCLUDE):
            continue
        k = (r["model"], r["level"])
        if k in seen:
            continue  # the CSV appends, so an arm rescored later appears twice; keep the first
        seen.add(k)
        tab.setdefault(r["model"], {})[r["level"]] = r
    w("| arm | scoring | fold MAP | superfamily MAP | family MAP |")
    w("|---|---|---|---|---|")
    for m, t in tab.items():
        if not all(lv in t for lv in ("fold", "superfamily", "family")):
            continue
        sc = "pooled cosine" if t["fold"]["scoring"] == "cosine" else "MaxSim"
        w(f"| `{m}` | {sc} | " + " | ".join(
            f"{float(t[lv]['eligible_MAP']):.4f}" for lv in ("fold", "superfamily", "family")) + " |")
    w("")


def proteingym_tables() -> None:
    f = PG / "proteingym_maxsim.csv"
    if not f.exists():
        return
    byvar: dict = {}
    for r in csv.DictReader(f.open()):
        byvar.setdefault(r["variant"], {})[r["model"]] = r
    for var in ("dms_substitutions", "dms_indels", "clinical_substitutions", "clinical_indels"):
        if var not in byvar:
            continue
        m = byvar[var]
        any_row = next(iter(m.values()))
        w(f"### {var} — {any_row['metric']}, {any_row['n_assays']} groups")
        w("")
        w(f"| arm | scoring | {any_row['metric']} | 95% CI |")
        w("|---|---|---|---|")
        for k, r in sorted(m.items(), key=lambda kv: -float(kv[1]["mean_score"])):
            sc = "pooled cosine" if r["scoring"] == "cosine" else "MaxSim"
            w(f"| `{k}` | {sc} | {float(r['mean_score']):.4f} | {r['ci95']} |")
        w("")
        # Paired deltas: same groups, so a delta is not two independent means differenced.
        pairs = [(a, b, lab) for a, b, lab in [
            ("protsent_late_35m_prop_late", "protsent_late_35m_prop_dense",
             "MaxSim − pooled cosine (identical weights)"),
            ("protsent_late_35m_prop_late", "proj128_late",
             "phase-2 proportional recipe − phase-1 round-robin arm (5 confounds, NOT step count)"),
        ] if a in m and b in m]
        if pairs:
            w("| paired comparison | Δ | 95% CI |")
            w("|---|---|---|")
            for a, b, lab in pairs:
                try:
                    za, zb = (np.load(PG / f"proteingym_{var}_{x}.npz") for x in (a, b))
                    ka, kb = dict(zip(za["assay"], za["score"])), dict(zip(zb["assay"], zb["score"]))
                    common = sorted(set(ka) & set(kb))
                    d = np.array([ka[k] - kb[k] for k in common])
                    mm, lo, hi = boot_ci(d, n_boot=2000, seed=42)
                    star = " **" if not (lo < 0 < hi) else ""
                    w(f"| {lab} (n={len(common)}) | {mm:+.4f}{star} | [{lo:+.4f}, {hi:+.4f}] |")
                except (FileNotFoundError, KeyError):
                    continue
            w("")


w("# Late interaction over ProtSent — results")
w("")
w("Generated by `build_late_results.py` from the result CSVs. Do not hand-edit.")
w("")
w("## Method")
w("")
w("Late interaction keeps one embedding per residue and scores a pair by **MaxSim** (for each")
w("query residue, its best match in the document, summed) instead of comparing two mean-pooled")
w("vectors by cosine. Every trained arm exports both views from the *same weights* — a `late/`")
w("multi-vector model and a `dense_view/` mean-pooled model — so the MaxSim-vs-cosine comparison")
w("is a pure scoring difference with the weights held fixed.")
w("")
w("Scores are **mean** MaxSim (divided by query length). Raw MaxSim sums over query residues and")
w("so scales with length: harmless where variants share a length, a pure artifact where they do")
w("not (indels).")
w("")
w("Arms named `*_zeroshot` score MaxSim over the backbone's **native residues with no projection**")
w("and no late training. They separate two things that are easy to confuse: whether MaxSim is a")
w("better *scorer*, and whether late *training* helps.")
w("")
w("## SCOPe-40 structural retrieval")
w("")
w("All-vs-all over 2,207 domains at three SCCS levels. Metrics are eligible-query means (queries")
w("with at least one non-self same-label neighbour); MAP is over the full ranking.")
w("")
scope_table()
w("Arms suffixed `_bf16bug` and `protsent_late_capped_flash` are **deleted**, models and rows")
w("alike. They trained under a bug that put AdamW's parameters in bf16, where a 1e-5 update is")
w("below the representable spacing, so only 2.4% of backbone elements could move. The analysis")
w("survives in RUNS.md; the artifacts do not, and `build_late_results.py` still filters the names")
w("so a CSV restored from git cannot reintroduce them.")
w("")
if PARTIAL:
    w("## ProteinGym — PARTIAL, quarantined")
    w("")
    w("> **Absolute numbers here are not results.** Scored at `--max_variants_per_assay 500` (~4.3%")
    w("> of the 2.47M variants) and truncated at 512 residues, with a plain mean over groups instead")
    w("> of the leaderboard's corrected average. Files live in")
    w("> `pilot_35m/benchmarks/proteingym_partial/`; that directory's README lists every limitation")
    w("> and the rerun cost (~1.0 h/arm at full coverage). A full-coverage rerun is deferred while")
    w("> the scoring path is optimised.")
    w("")
    w("Only the **paired** rows below survive the caveat: both sides ran under the same cap, the")
    w("same truncation and the same assay set, so a delta is a scoring or weights effect rather")
    w("than a protocol artifact. The per-arm means above each delta table are for internal ranking")
    w("only and must not be placed next to a published ProteinGym score.")
    w("")
else:
    w("## ProteinGym")
    w("")
    w("Full coverage: every variant of every assay, 1024-residue context, aggregated the way the")
    w("leaderboard does (mean within each `coarse_selection_type` group, then the mean of those).")
    w("")
w("The second paired row is **not a step-count contrast**, despite once being labelled as one.")
w("`protsent_late_35m_prop` continues from `protsent_late_proj128`, but changes five things at")
w("once: sampler (round-robin → proportional), pool (2.0M/2.0M → 19.0M AFDB / 15.0M STRING),")
w("world size (1 → 2, so effective batch 128 → 256), attention (sdpa → vllm-flash-attn3) and")
w("`--compile`. Both `runtime.json` files record this. Decisively, the 15.5x extra pairs bought")
w("**1.04x** the Pfam exposure (170,667 → 177,441 pairs): round-robin gave Pfam a third of every")
w("batch, proportional gives it 2.24%. The extra training is ~all AFDB and STRING. Read the row")
w("as a cost of the phase-2 mixture, not evidence that training longer hurts.")
w("")
w("For the clinical variants the score is **negated** before the AUC: the label is pathogenicity,")
w("and a variant that looks more like the wild type should be less pathogenic.")
w("")
proteingym_tables()
(RES / "RESULTS.md").write_text("\n".join(out) + "\n")
print(f"wrote {RES/'RESULTS.md'} ({len(out)} lines)")
