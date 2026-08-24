#!/usr/bin/env python
"""Is this run actually training, or is its backbone frozen?

Three independent checks, cheapest first. Written because a run whose backbone cannot move
still produces a falling-ish loss, a rising SCOPe curve on a vanilla backbone, and a perfectly
healthy-looking progress bar -- the bf16 bug survived hours of all three.

    uv run --no-sync python validate_run.py models/late_interaction/protsent_late_long
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--base", default=None, help="Backbone the run started from (default: runtime.json/argv)")
    ap.add_argument("--n_seqs", type=int, default=300)
    ap.add_argument("--max_cos", type=float, default=0.995,
                    help="Dense view this close to the base model means the backbone never moved")
    ap.add_argument("--max_loss", type=float, default=3.0,
                    help="bf16-frozen arms plateaued at 3.87-3.90; healthy fp32 arms reach 1.4-2.0")
    args = ap.parse_args()

    run = Path(args.run_dir)
    fails, checks = [], {}

    # 1. dtype. Free, and it is the actual bug rather than a symptom of it.
    import torch
    from safetensors import safe_open

    late = run / "late"
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1]))
    model_dir = late if late.exists() else (ckpts[-1] if ckpts else None)
    if model_dir is None:
        print("no exported model or checkpoint yet")
        return 0
    st = next(model_dir.rglob("model.safetensors"), None)
    if st is not None:
        with safe_open(st, framework="pt") as f:
            dtypes = {f.get_slice(k).get_dtype() for k in f.keys()}
        print(f"backbone dtypes: {sorted(dtypes)}")
        checks["dtype"] = "PASS"
        if any("BF16" in d or "F16" in d for d in dtypes):
            checks["dtype"] = "FAIL"
            fails.append("backbone weights are half precision: AdamW cannot represent a 1e-5 update")

    # 2. loss. Config-independent, and it separated the frozen arms from the healthy ones cleanly.
    log = run / "train_log.csv"
    if log.exists():
        import csv
        rows = [r for r in csv.DictReader(log.open()) if r.get("loss")]
        if rows:
            last = float(rows[-1]["loss"])
            print(f"last logged loss: {last:.3f}")
            checks["loss"] = "PASS"
            if last > args.max_loss:
                checks["loss"] = "FAIL"
                fails.append(f"loss {last:.3f} > {args.max_loss}: consistent with a frozen backbone")

    # 3. drift. The decisive one: a frozen backbone produces a dense view identical to its base.
    # step0/ is this run's own untrained export, so it is both local and the exactly right
    # reference: "how far has THIS run moved its own starting point". runtime.json only exists
    # once the run finishes, which is too late to be useful for a live check.
    base = args.base
    if base is None and (run / "step0" / "dense_view").exists():
        base = str(run / "step0" / "dense_view")
    if base is None and (run / "runtime.json").exists():
        base = json.loads((run / "runtime.json").read_text()).get("model")
    if base is None:
        print("no step0/ export and no runtime.json: pass --base to enable the drift check")
    if base:
        import late_interaction as li
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Pooling

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        seqs = list(load_dataset("tattabio/scope40_test", split="train")["sequence"])[: args.n_seqs]

        def emb(model):
            model.max_seq_length = 512
            e = model.encode(seqs, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            del model
            torch.cuda.empty_cache()
            return e

        # Mean-pool whatever backbone this checkpoint holds, rather than waiting for the run to
        # export a dense_view -- a live run has not exported one yet.
        mve = li.load_multivector_encoder(str(model_dir), device=dev)
        trained = emb(SentenceTransformer(
            modules=[mve[0], Pooling(mve[0].get_embedding_dimension(), pooling_mode="mean")],
            device=dev))
        cos = float((trained * emb(SentenceTransformer(base, device=dev, trust_remote_code=True))).sum(1).mean())
        print(f"dense-view cosine vs {base}: {cos:.5f}  (1.000 = backbone never moved)")
        checks["drift"] = "PASS"
        if cos > args.max_cos:
            checks["drift"] = "FAIL"
            fails.append(f"cosine {cos:.5f} > {args.max_cos}: backbone is still the base model")

    for name in ("dtype", "loss", "drift"):
        print(f"  {name:6s} {checks.get(name, 'SKIPPED')}")
    for f in fails:
        print(f"FAIL: {f}")
    if fails:
        print("NOT TRAINING PROPERLY")
        return 1
    if checks.get("drift") != "PASS":
        # dtype alone is necessary, not sufficient; drift is the check that actually observes
        # the backbone having moved. Saying OK without it is the false confidence this exists to stop.
        print("INCONCLUSIVE: the drift check did not run, so nothing here observed the backbone move")
        return 2
    print("OK: run is training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
