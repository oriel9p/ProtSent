#!/usr/bin/env python
"""Make a FastPLM-saved SentenceTransformer checkpoint loadable by stock transformers.

Training saves ESM2 checkpoints with FastPLM's custom-code identity:

    config.json           model_type: "fast_esm", architectures: ["FastEsmForMaskedLM"],
                          auto_map -> modeling_fastplms.*
    tokenizer_config.json tokenizer_class: "FastEsmTokenizer"

`SentenceTransformer(path)` then fails with:

    ValueError: Unrecognized processing class in <path>. Can't instantiate a
    processor, a tokenizer, an image processor, ...

because "fast_esm" is not a registered model type and no auto_map entry exists for
AutoTokenizer. The published paper checkpoint (oriel9p/protsent-esm2-35M) does not
have this problem: it was saved as plain ESM (model_type "esm", tokenizer_class
"EsmTokenizer", no auto_map). This rewrites a checkpoint into that same form.

The rewrite touches metadata only; weights are untouched. Loading the result as a
vanilla EsmModel reports:

    MISSING     pooler.dense.{weight,bias}   -- EsmModel's optional pooler
    UNEXPECTED  lm_head.*                    -- the MLM head

Both are correct. ProtSent embeds by mean-pooling token embeddings through the
1_Pooling module, so the pooler is never called -- verified empirically by scaling
pooler.dense.weight by 1000x and adding 5 to its bias, which moves the output
embeddings by exactly 0.0. The MLM head is not used to embed. All 215 encoder
tensors load.

Usage:
    python make_checkpoint_loadable.py <checkpoint_dir> [more dirs...]
    python make_checkpoint_loadable.py --selfcheck
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

VANILLA = {"model_type": "esm", "architectures": ["EsmModel"]}


def convert(path: Path, backup: bool = True) -> bool:
    """Rewrite one checkpoint in place. Returns True if anything changed."""
    cfg_path = path / "config.json"
    tok_path = path / "tokenizer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {path}")

    cfg = json.loads(cfg_path.read_text())
    tok = json.loads(tok_path.read_text()) if tok_path.exists() else None

    needs = (
        cfg.get("model_type") != "esm"
        or "auto_map" in cfg
        or (tok is not None and tok.get("tokenizer_class") != "EsmTokenizer")
    )
    if not needs:
        return False

    if backup and not (path / "config.json.fastplm").exists():
        shutil.copy2(cfg_path, path / "config.json.fastplm")
        if tok_path.exists():
            shutil.copy2(tok_path, path / "tokenizer_config.json.fastplm")

    cfg.pop("auto_map", None)
    cfg.update(VANILLA)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    if tok is not None:
        tok["tokenizer_class"] = "EsmTokenizer"
        tok_path.write_text(json.dumps(tok, indent=2))
    return True


def _selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "config.json").write_text(json.dumps({
            "model_type": "fast_esm",
            "architectures": ["FastEsmForMaskedLM"],
            "auto_map": {"AutoModel": "modeling_fastplms.FastEsmModel"},
            "hidden_size": 480,
        }))
        (p / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "FastEsmTokenizer"})
        )

        assert convert(p) is True
        cfg = json.loads((p / "config.json").read_text())
        tok = json.loads((p / "tokenizer_config.json").read_text())
        assert cfg["model_type"] == "esm", cfg
        assert cfg["architectures"] == ["EsmModel"], cfg
        assert "auto_map" not in cfg, cfg
        assert tok["tokenizer_class"] == "EsmTokenizer", tok
        assert cfg["hidden_size"] == 480, "unrelated config keys must survive"
        # originals preserved
        assert json.loads((p / "config.json.fastplm").read_text())["model_type"] == "fast_esm"
        # idempotent: a second pass is a no-op
        assert convert(p) is False
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selfcheck" in args:
        _selfcheck()
        raise SystemExit(0)
    if not args:
        raise SystemExit(__doc__)
    for a in args:
        changed = convert(Path(a))
        print(f"{'converted' if changed else 'already vanilla'}: {a}")
