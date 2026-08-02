#!/usr/bin/env python
"""Convert ISM-C-300M's raw ESM-C state dict into a loadable HF model directory.

`jozhang97/ismc-300m-2024-12` (ISM: Distilling Structural Representations into
Protein Sequence Models) ships three files -- README, .gitattributes, and a bare
1.33 GB `ismc_300m_2024_12_v0.pth`. There is no config.json, no architectures, no
model_type, no auto_map, no tokenizer and no safetensors, so
`AutoModel.from_pretrained` fails at config resolution; `trust_remote_code` is
irrelevant because there is no remote code to trust. The published interface is
`esm.models.esmc.ESMC`, and PyPI `esm` declares `requires_python <3.13` against
this repo's Python 3.14 venv.

The weights are still usable. `Synthyra/ESMplusplus_small` is a name-for-name HF
port of upstream ESM-C at exactly the ISM-C size (hidden 960, 30 layers, 15 heads,
vocab 64), down to the parameter paths:

    embed.weight
    transformer.blocks.{i}.attn.{layernorm_qkv.{0,1},out_proj,q_ln,k_ln}.*
    transformer.blocks.{i}.ffn.{0,1,3}.*
    transformer.norm.weight
    sequence_head.{0,2,3}.*

Both implementations register rotary `inv_freq` with persistent=False, so neither
contributes a state-dict key. The conversion is therefore a strict load with no
key remapping -- and `strict=True` is the point, not an optimisation: a rename
upstream would silently leave randomly-initialised layers in place and produce a
model that benchmarks as a mediocre ESM-C rather than as ISM-C.

We copy the whole ESMplusplus_small snapshot and overwrite model.safetensors
rather than calling save_pretrained into a fresh directory. auto_map points at
`modeling_fastplms.ESMplusplusModel`, whose siblings under `fastplms/` are not
auto_map entries and so would not be copied.

Downstream, `detect_model_type` sees "ESMplusplus" in the copied config's
auto_map and routes to the existing esmplusplus branch in both
protein_benchmark_suite.py and protein_pipeline.py. No loader changes anywhere.

Gates, each of which aborts:

    G1  strict load: every ISM-C key lands on a real parameter, nothing missing
    G2  the result differs from vanilla ESM-C (else we re-saved the control)
    G3  forward pass is sane and is not vanilla's output

Usage:
    python convert_ismc_to_hf.py [--out DIR] [--force]
    python convert_ismc_to_hf.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/storage/models/hf_home")

ISM_REPO = "jozhang97/ismc-300m-2024-12"
ISM_WEIGHTS = "ismc_300m_2024_12_v0.pth"
ESMC_REPO = "Synthyra/ESMplusplus_small"
DEFAULT_OUT = Path("/storage/models/ISM-C-300M")

# Eight SCOPe-40-style sequences for the forward-pass gate. Real domains, varied
# fold and length, so a model that has collapsed shows up as uniform similarity.
PROBE_SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
    "GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEKMLADIGRVLDDDVKRLNTLNAETIVTQIQTLKDEIQ",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "AEQCGRQAGGKLCPNNLCCSQWGWCGSTDEYCSPDHNCQSNCKGGSGGGSGGGSGSGGSDPNAVKTFDGRVSLPAT",
    "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQED",
    "PIVQNLQGQMVHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEE",
    "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFDKATLKEIAEQFGCTPQAVSNWKRRFGKDGQVKAQEVEVK",
    "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
]


def _resolve(repo: str, filename: str | None = None) -> Path:
    """Path to a cached repo snapshot (or one file in it). Downloads if absent."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename is not None:
        return Path(hf_hub_download(repo, filename))
    return Path(snapshot_download(repo))


def load_ism_state_dict(pth: Path):
    """Read the raw ISM-C checkpoint, unwrapping a nested state dict if present."""
    import torch

    try:
        obj = torch.load(pth, map_location="cpu", weights_only=True)
    except Exception as exc:  # pickled more than tensors
        print(f"note: weights_only load failed ({exc}); retrying unrestricted")
        obj = torch.load(pth, map_location="cpu", weights_only=False)

    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj:
        raise SystemExit(f"G1 FAILED: {pth} did not yield a state dict (got {type(obj)})")
    # Some releases prefix everything with "model."; ours does not, but check.
    if all(k.startswith("model.") for k in obj):
        obj = {k[len("model."):]: v for k, v in obj.items()}
    return obj


def gate1_strict_load(model, sd) -> None:
    """Every ISM-C tensor must land on a real parameter, and none may be left over."""
    have = model.state_dict()
    missing = sorted(set(have) - set(sd))
    unexpected = sorted(set(sd) - set(have))
    mismatched = [
        (k, tuple(have[k].shape), tuple(sd[k].shape))
        for k in sorted(set(have) & set(sd))
        if tuple(have[k].shape) != tuple(sd[k].shape)
    ]
    if missing or unexpected or mismatched:
        print("G1 FAILED: ISM-C state dict does not match ESMplusplus_small.", file=sys.stderr)
        print(f"  target keys {len(have)}, checkpoint keys {len(sd)}", file=sys.stderr)
        for label, items in (("MISSING", missing), ("UNEXPECTED", unexpected)):
            if items:
                print(f"  {label} ({len(items)}):", file=sys.stderr)
                for k in items[:20]:
                    print(f"    {k}", file=sys.stderr)
                if len(items) > 20:
                    print(f"    ... {len(items) - 20} more", file=sys.stderr)
        for k, a, b in mismatched[:20]:
            print(f"  SHAPE {k}: target {a} != checkpoint {b}", file=sys.stderr)
        raise SystemExit(1)

    model.load_state_dict(sd, strict=True)
    print(f"G1 ok: {len(sd)} tensors loaded strict")


def gate2_differs(vanilla_sd, ism_sd) -> None:
    """The converted weights must not be the vanilla weights we started from."""
    import torch

    diffs = {
        k: (vanilla_sd[k].float() - ism_sd[k].float()).abs().max().item()
        for k in vanilla_sd
        if torch.is_floating_point(vanilla_sd[k])
    }
    changed = sum(1 for v in diffs.values() if v > 0)
    worst = max(diffs.items(), key=lambda kv: kv[1])
    if changed == 0:
        raise SystemExit(
            "G2 FAILED: converted weights are identical to vanilla ESM-C. "
            "The checkpoint was not applied."
        )
    print(
        f"G2 ok: {changed}/{len(diffs)} tensors differ from vanilla ESM-C "
        f"(largest {worst[0]} {worst[1]:.4f})"
    )


def _mean_pool(model, tokenizer, seqs, device):
    """Attention-masked mean pool, matching the benchmark suite's generic path."""
    import torch

    enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, return_dict=True)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out.hidden_states[-1]
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()


def gate3_forward(ism_model, vanilla_model, tokenizer, device) -> None:
    """Embeddings must be finite, correctly shaped, and not vanilla's."""
    import torch

    ism = _mean_pool(ism_model, tokenizer, PROBE_SEQS, device)
    van = _mean_pool(vanilla_model, tokenizer, PROBE_SEQS, device)

    if not torch.isfinite(ism).all():
        raise SystemExit("G3 FAILED: ISM-C embeddings contain NaN or Inf")
    if ism.shape != (len(PROBE_SEQS), 960):
        raise SystemExit(f"G3 FAILED: expected (8, 960) embeddings, got {tuple(ism.shape)}")

    cos = torch.nn.functional.cosine_similarity(ism, van, dim=1)
    if (cos >= 0.999).all():
        raise SystemExit(
            f"G3 FAILED: ISM-C output is indistinguishable from vanilla ESM-C "
            f"(cosine {cos.min():.5f}-{cos.max():.5f})"
        )
    if (cos <= 0).any():
        raise SystemExit(
            f"G3 FAILED: ISM-C output is unrelated to vanilla ESM-C "
            f"(min cosine {cos.min():.5f}) -- suggests scrambled weights"
        )

    # Anisotropy on the probe set, for the record. Vanilla ESM-2 sits at 0.85-0.90
    # (see results/benchmarks/embedding_geometry.json); a collapsed conversion
    # would read ~1.0.
    def _aniso(x):
        xn = torch.nn.functional.normalize(x, dim=1)
        sim = xn @ xn.T
        off = ~torch.eye(len(x), dtype=torch.bool)
        return sim[off].mean().item()

    print(
        f"G3 ok: finite, {tuple(ism.shape)}, cosine to vanilla "
        f"{cos.min():.4f}-{cos.max():.4f}; mean pairwise cosine "
        f"ISM-C {_aniso(ism):.4f} vs vanilla {_aniso(van):.4f}"
    )


def convert(out_dir: Path, force: bool = False, device: str = "cuda") -> Path:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from model_utils import apply_esmplusplus_compat_patch

    apply_esmplusplus_compat_patch()

    if out_dir.exists():
        if not force:
            raise SystemExit(f"{out_dir} already exists; pass --force to overwrite")
        shutil.rmtree(out_dir)

    pth = _resolve(ISM_REPO, ISM_WEIGHTS)
    snapshot = _resolve(ESMC_REPO)
    print(f"ISM-C weights   {pth}")
    print(f"ESM-C snapshot  {snapshot}")

    # symlinks=False so the copy resolves the cache blobs into real files; the
    # result must survive `hf cache prune` and be movable.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, out_dir, symlinks=False)
    print(f"copied scaffold {out_dir}")

    model = AutoModelForMaskedLM.from_pretrained(out_dir, trust_remote_code=True)
    vanilla_sd = {k: v.clone() for k, v in model.state_dict().items()}

    sd = load_ism_state_dict(pth)
    gate1_strict_load(model, sd)
    gate2_differs(vanilla_sd, model.state_dict())

    if torch.cuda.is_available() and device.startswith("cuda"):
        vanilla = AutoModelForMaskedLM.from_pretrained(out_dir, trust_remote_code=True)
        tokenizer = getattr(model, "tokenizer", None) or AutoTokenizer.from_pretrained(
            out_dir, trust_remote_code=True
        )
        gate3_forward(model.to(device).eval(), vanilla.to(device).eval(), tokenizer, device)
        model = model.cpu()
        del vanilla
        torch.cuda.empty_cache()
    else:
        print("G3 skipped: no CUDA device")

    weights = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {out_dir / 'model.safetensors'} ({len(weights)} tensors)")

    # Record provenance next to the weights so the directory is self-describing.
    (out_dir / "PROVENANCE.md").write_text(
        "# ISM-C-300M (converted)\n\n"
        f"Weights: `{ISM_REPO}` / `{ISM_WEIGHTS}`\n"
        f"Scaffold (config, tokenizer, modeling code): `{ESMC_REPO}`\n\n"
        "Produced by `convert_ismc_to_hf.py`, which loads the upstream ESM-C state\n"
        "dict into `ESMplusplusForMaskedLM` with `strict=True`. Weights are\n"
        "unmodified; only the container changed.\n\n"
        "ISM: Ouyang-Zhang et al., *Distilling Structural Representations into\n"
        "Protein Sequence Models*, biorxiv 2024.11.08.622579.\n\n"
        "Local artefact. Not for redistribution -- these are jozhang97's weights.\n"
    )
    return out_dir


def _selfcheck() -> None:
    """Exercise the gates on tiny synthetic state dicts. No downloads, no GPU."""
    import torch

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 3)
            self.out = torch.nn.Linear(3, 3)

    m = _Tiny()
    good = {k: torch.randn_like(v) for k, v in m.state_dict().items()}
    gate1_strict_load(m, good)
    assert torch.equal(m.state_dict()["out.weight"], good["out.weight"])

    # G1 rejects a renamed key rather than silently leaving it random.
    renamed = {("mlp.weight" if k == "out.weight" else k): v for k, v in good.items()}
    try:
        gate1_strict_load(_Tiny(), renamed)
    except SystemExit:
        pass
    else:
        raise AssertionError("G1 accepted a renamed key")

    # G2 rejects an unchanged checkpoint and accepts a changed one.
    try:
        gate2_differs(good, good)
    except SystemExit:
        pass
    else:
        raise AssertionError("G2 accepted identical weights")
    gate2_differs(good, {k: v + 1.0 for k, v in good.items()})

    # load_ism_state_dict unwraps the common nesting and prefixing conventions.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "w.pth"
        torch.save({"state_dict": {"model.a": torch.zeros(2)}}, p)
        assert list(load_ism_state_dict(p)) == ["a"]

    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        raise SystemExit(0)

    dest = convert(args.out, force=args.force, device=args.device)
    print(f"\nready: {dest}")
