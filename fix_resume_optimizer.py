#!/usr/bin/env python
"""Make a SentenceTransformer checkpoint's optimizer state resumable.

The pipeline builds the model for a FRESH run with `load_model_for_training(args.model)`,
which loads a MaskedLM backbone and keeps its 6-tensor LM head. On RESUME, the
SentenceTransformer trainer rebuilds the model from the checkpoint directory, and
that round-trip drops the head. So the saved optimizer state has 491 parameters
while the rebuilt model has 485, and resume dies with:

    ValueError: loaded state dict contains a parameter group that doesn't match
    the size of optimizer's group

The 6 missing tensors are the LM head: dense.{weight,bias}, layer_norm.{weight,bias},
decoder.weight and bias. HuggingFace splits parameters into a weight-decay group and
a no-decay group (biases and LayerNorm), so the head contributes 2 to the first and
4 to the second -- which is exactly the observed 184->182 and 307->303 = 485.

This drops those six entries from optimizer.pt and renumbers the remaining state so
the file matches the rebuilt model. The LM head is never used to produce embeddings
(the model mean-pools encoder output), so its Adam moments are genuinely irrelevant.

SAFETY: the script refuses to write unless EVERY surviving state entry's exp_avg
shape matches, position for position, the corresponding parameter of the actually
rebuilt model. A wrong mapping would pair Adam moments with the wrong parameters and
silently corrupt training, so the check is a hard gate rather than a warning.

Usage:
    python fix_resume_optimizer.py models/protsent_esm2_150m_v2/checkpoint-2500
    python fix_resume_optimizer.py --selfcheck
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch


def rebuilt_param_shapes(ckpt: Path) -> list[tuple]:
    """Shapes of the trainable parameters, in the order the Trainer will group them.

    Mirrors transformers' grouping: decay group first (everything that is not a bias
    or a LayerNorm weight), then the no-decay group, each in model parameter order.
    """
    from sentence_transformers import SentenceTransformer
    from transformers.trainer_pt_utils import get_parameter_names
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

    model = SentenceTransformer(str(ckpt), trust_remote_code=True, device="cpu")
    decay = set(n for n in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in n)
    named = dict(model.named_parameters())
    g0 = [n for n, p in model.named_parameters() if p.requires_grad and n in decay]
    g1 = [n for n, p in model.named_parameters() if p.requires_grad and n not in decay]
    # Per GROUP, because the Trainer builds one param group per decay class and the
    # sizes must match group for group -- a global sequence match can satisfy the
    # total while still putting the dropped entries in the wrong group.
    return [[tuple(named[n].shape) for n in g] for g in (g0, g1)], [g0, g1]


def fix(ckpt: Path, apply: bool = True) -> bool:
    opt_path = ckpt / "optimizer.pt"
    o = torch.load(opt_path, map_location="cpu", weights_only=False)
    groups = o["param_groups"]
    state = o["state"]
    total = sum(len(g["params"]) for g in groups)

    per_group_shapes, per_group_names = rebuilt_param_shapes(ckpt)
    target_shapes = [s for g in per_group_shapes for s in g]
    target_names = [n for g in per_group_names for n in g]
    print(f"optimizer groups {[len(g['params']) for g in groups]} (total {total}); "
          f"rebuilt model groups {[len(g) for g in per_group_shapes]} "
          f"(total {len(target_shapes)})")
    if total == len(target_shapes):
        print("already consistent; nothing to do")
        return False
    if len(groups) != len(per_group_shapes):
        print(f"REFUSING: {len(groups)} optimizer groups vs {len(per_group_shapes)} model groups")
        return False

    # Align WITHIN each group: walk the optimizer's entries against that group's
    # target shapes and skip where they diverge. Doing this globally can satisfy the
    # total while leaving the per-group counts wrong, which is the failure that made
    # the first attempt at this still die on resume.
    keep: list[int] = []
    for gi, (g, want) in enumerate(zip(groups, per_group_shapes)):
        ti = 0
        for idx in g["params"]:
            shp = tuple(state[idx]["exp_avg"].shape) if idx in state and "exp_avg" in state[idx] else None
            if ti < len(want) and (shp is None or shp == want[ti]):
                keep.append(idx)
                ti += 1
            else:
                print(f"  dropping optimizer index {idx} (group {gi}, shape {shp})")
        if ti != len(want):
            print(f"REFUSING: group {gi} matched {ti} of {len(want)} target params")
            return False

    # Rebuild groups and renumber state to 0..n-1 in the kept order.
    keep_set = set(keep)
    new_groups = []
    remap: dict[int, int] = {}
    counter = 0
    for g in groups:
        params = [i for i in g["params"] if i in keep_set]
        ng = {k: v for k, v in g.items() if k != "params"}
        ng["params"] = []
        for i in params:
            remap[i] = counter
            ng["params"].append(counter)
            counter += 1
        new_groups.append(ng)
    new_state = {remap[i]: state[i] for i in state if i in remap}

    # HARD GATE: every surviving moment must line up with the rebuilt model.
    for new_i, shp in enumerate(target_shapes):
        if new_i in new_state and "exp_avg" in new_state[new_i]:
            got = tuple(new_state[new_i]["exp_avg"].shape)
            if got != shp:
                print(f"REFUSING: position {new_i} shape {got} != model {shp} ({target_names[new_i]})")
                return False
    print(f"verified: {len(new_state)} moment tensors align positionally with all "
          f"{len(target_shapes)} model parameters")

    if not apply:
        return True
    backup = ckpt / "optimizer.pt.orig"
    if not backup.exists():
        shutil.copy2(opt_path, backup)
    o["param_groups"] = new_groups
    o["state"] = new_state
    torch.save(o, opt_path)
    print(f"wrote {opt_path} (original kept as {backup.name})")
    return True


def _selfcheck() -> None:
    """A synthetic optimizer with two extra entries must be trimmed and verified."""
    import tempfile

    shapes = [(4, 4), (4,), (3, 3), (3,)]
    extra = (7, 7)
    state = {}
    order = [(4, 4), (3, 3), extra, (4,), (3,)]  # decay group then no-decay, with one extra each
    for i, s in enumerate(order):
        state[i] = {"exp_avg": torch.zeros(s), "exp_avg_sq": torch.zeros(s)}
    groups = [{"params": [0, 1, 2], "lr": 1e-4}, {"params": [3, 4], "lr": 1e-4}]

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        torch.save({"param_groups": groups, "state": state}, p / "optimizer.pt")

        # Stub out the model rebuild with the known target ordering.
        import fix_resume_optimizer as mod
        mod.rebuilt_param_shapes = lambda ck: ([[(4, 4), (3, 3)], [(4,), (3,)]],
                                               [["a", "b"], ["c", "d"]])
        assert mod.fix(p, apply=True) is True
        o = torch.load(p / "optimizer.pt", map_location="cpu", weights_only=False)
        assert [len(g["params"]) for g in o["param_groups"]] == [2, 2], o["param_groups"]
        assert sorted(o["state"]) == [0, 1, 2, 3], sorted(o["state"])
        assert tuple(o["state"][0]["exp_avg"].shape) == (4, 4)
        assert tuple(o["state"][1]["exp_avg"].shape) == (3, 3)
        assert (p / "optimizer.pt.orig").exists()

        # A target the optimizer cannot satisfy must be refused, not guessed.
        mod.rebuilt_param_shapes = lambda ck: ([[(9, 9)] * 2, [(9, 9)] * 2], [["x"] * 2] * 2)
        torch.save({"param_groups": groups, "state": state}, p / "optimizer.pt")
        assert mod.fix(p, apply=False) is False
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif len(sys.argv) == 2:
        raise SystemExit(0 if fix(Path(sys.argv[1])) else 1)
    else:
        raise SystemExit(__doc__)
