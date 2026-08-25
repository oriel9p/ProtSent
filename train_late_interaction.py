#!/usr/bin/env python
"""Late-interaction contrastive continuation (ColBERT MaxSim) for ProtSent.

Short GradCache-MNRL run over the same three V2 pair sources (Pfam, AFDB,
STRING), round-robin, no DMS / hard negatives / GOR / Matryoshka. Reuses
protein_pipeline's model loading and pair builders. Single GPU or DDP via:

    accelerate launch --num_processes 2 --mixed_precision bf16 \
        train_late_interaction.py --model GrimSqueaker/ProtSent-V2-35M \
        --files /storage/users/ddofer/data/protsent-data-dc40/pfam_sorted.parquet ... \
        --output_dir models/late_interaction/protsent_late --max_steps 2000

Outputs under --output_dir:
    step0/{late,dense_view}     pre-training export (dense-view parity control)
    checkpoint-*/               trainer checkpoints (MVE format)
    late/, dense_view/          final model + its mean-pooled view
    runtime.json, train_log.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import late_interaction as li  # noqa: E402
from protein_pipeline import (  # noqa: E402
    TimeLimitCallback,
    _build_pair_dataset,
    _infer_columns,
    _is_ppi_parquet,
    _load_ppi_pair_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("late_interaction_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Backbone: HF id or local ST/HF dir")
    p.add_argument("--files", nargs="+", required=True, help="Pair parquet files (cluster or seq1/seq2 schema)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--allow_head_reinit", action="store_true",
                   help="Continue from a checkpoint whose projection head is a different width, "
                        "training a fresh head on its backbone. Without this the mismatch is an "
                        "error, because the run would otherwise look like a continuation while "
                        "measuring a from-scratch head.")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32, help="Per-device contrastive batch")
    p.add_argument("--mini_batch_size", type=int, default=16, help="GradCache embedding chunk")
    p.add_argument("--mini_batch_num_tokens", type=int, default=0,
                   help="Token-packed GradCache mini-batches (ST>=5.7); overrides --mini_batch_size when >0")
    p.add_argument("--score_mini_batch_size", type=int, default=8, help="MaxSim scoring chunk")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_minutes", type=int, default=165, help="Hard wall-clock stop (<2h45m default)")
    p.add_argument("--lr", type=float, default=1e-5, help="Backbone learning rate")
    p.add_argument("--proj_lr", type=float, default=1e-4, help="Projection learning rate")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--max_pairs_per_cluster", type=int, default=8)
    p.add_argument("--max_pairs_per_file", type=int, default=2_000_000,
                   help="Cap pairs built per cluster file (0 = no cap); a short run never sees more")
    p.add_argument("--string_max_pairs", type=int, default=2_000_000, help="Cap on STRING pairs (0 = all)")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", default="protsent_late")
    p.add_argument("--gather_across_devices", action="store_true")
    p.add_argument("--skip_step0_export", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Resume from the latest checkpoint-* in --output_dir if present")
    p.add_argument("--save_total_limit", type=int, default=1,
                   help="Checkpoints kept on disk (1 = crash-resume only; curve sampling reads them live)")
    p.add_argument("--attn_implementation", default="auto",
                   help="auto (try flash, fall back to sdpa), sdpa, or an explicit kernel repo id. "
                        "Flash measured 1.48-1.97x faster with peak VRAM 11.9->9.0 GB, and a paired quality "
                        "A/B found no cost at the scale of this pilot's effects (see report)")
    p.add_argument("--multi_dataset_sampler", default="round_robin", choices=["round_robin", "proportional"],
                   help="round_robin cycles sources until the smallest is exhausted; proportional samples in "
                        "proportion to source size, as ProtSent-V2's own run did")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model. 1.14x faster under 2-way DDP with flash "
                        "(0.586 vs 0.516 steps/s over 400 steps, both arms equally contended), "
                        "~250 s warmup. Worth it for runs of a few thousand steps or more")
    return p.parse_args()


def build_datasets(args, world_size: int):
    """{file_stem: Dataset(sentence_0, sentence_1)}, rank-0 builds first (shared Arrow cache)."""
    from datasets import Dataset

    def _build() -> dict:
        # _build_pair_dataset samples clusters with the global `random` module, so without
        # this the same flags produce a different pair set on every cold cache and no run is
        # reproducible - which also means a control and the run it controls for could differ.
        import random as _random

        _random.seed(args.seed)
        out: dict[str, Dataset] = {}
        for f in args.files:
            name = os.path.splitext(os.path.basename(f))[0]
            if _is_ppi_parquet(f):
                ds = _load_ppi_pair_dataset([f], max_pairs=args.string_max_pairs, sample_seed=args.seed)
            else:
                seq_col, group_col, *_ = _infer_columns([f], None, None)
                ds = _build_pair_dataset(
                    file_paths=[f],
                    seq_col=seq_col,
                    group_col=group_col,
                    max_pairs_per_cluster=args.max_pairs_per_cluster,
                    max_pairs=args.max_pairs_per_file,
                    max_seq_length=args.max_seq_length,
                )
            out[name] = ds.shuffle(seed=args.seed)
            logger.info("dataset %s: %d pairs", name, len(ds))
        return out

    is_ddp = world_size > 1 and torch.distributed.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not is_ddp or local_rank == 0:
        datasets = _build()
        if is_ddp:
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        datasets = _build()  # instant reload from rank 0's Arrow cache
    return datasets


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import (
        MultiVectorEncoderTrainer,
        MultiVectorEncoderTrainingArguments,
    )
    from sentence_transformers.base.sampler import BatchSamplers, MultiDatasetBatchSamplers
    from sentence_transformers.multi_vector_encoder.losses import (
        CachedMultiVectorMultipleNegativesRankingLoss,
    )

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else None

    # TF32 lowers mantissa precision inside a matmul only; stored weights stay fp32, so optimizer
    # updates stay exact. The safe half of what pinning bf16 weights was reaching for.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Creating TrainingArguments first initializes the accelerate/torch.distributed
    # state, which build_datasets relies on for its rank-0-first barrier.
    train_args = MultiVectorEncoderTrainingArguments(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        bf16=True,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        batch_sampler=BatchSamplers.BATCH_SAMPLER,  # NO_DUPLICATES hangs on cluster-sorted corpora
        multi_dataset_batch_sampler=(MultiDatasetBatchSamplers.PROPORTIONAL
                                     if args.multi_dataset_sampler == "proportional"
                                     else MultiDatasetBatchSamplers.ROUND_ROBIN),
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        logging_steps=10,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    # The head is built before Trainer.__init__ runs set_seed, so without this --seed never
    # reached it and every arm's step0 head was a different draw.
    from transformers import set_seed

    set_seed(args.seed)

    attn = li.resolve_attention(args.attn_implementation, args.model)
    mve, dense_st = li.build_multivector_encoder(
        args.model, proj_dim=args.proj_dim, max_seq_length=args.max_seq_length, device=device,
        allow_head_reinit=args.allow_head_reinit,
        attn_implementation=attn,
    )
    if args.compile:
        mve.compile(dynamic=True)
        logger.info("torch.compile enabled (dynamic=True); expect ~250 s of warmup")
    pooling = dense_st[1]
    frozen = li.freeze_unused_heads(mve)
    if frozen:
        logger.info("froze %d unused head params (pooler/contact_head) for DDP", frozen)

    if is_main and not args.skip_step0_export and not (out / "step0" / "late").exists():
        li.save_late_and_dense(mve, pooling, str(out / "step0"))
        logger.info("step-0 late + dense_view exported (parity control)")

    train_dataset = build_datasets(args, world_size)

    loss = CachedMultiVectorMultipleNegativesRankingLoss(
        mve,
        mini_batch_size=args.mini_batch_size,
        mini_batch_num_tokens=args.mini_batch_num_tokens or None,
        score_mini_batch_size=args.score_mini_batch_size,
        gather_across_devices=args.gather_across_devices,
    )
    backbone, proj = li.backbone_and_projection_params(mve)
    param_groups = [{"params": backbone, "lr": args.lr}]
    if proj:
        param_groups.append({"params": proj, "lr": args.proj_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)

    trainer = MultiVectorEncoderTrainer(
        model=mve,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        optimizers=(optimizer, None),
        callbacks=[TimeLimitCallback(args.max_minutes)] if args.max_minutes > 0 else [],
    )

    # resumed_from is the step this process starts at, so throughput is reported over the work
    # this process actually did rather than over every step the run has ever taken.
    # Capture HEAD now, not when runtime.json is written at the end: the repo gets edited during a
    # 10-hour run, so an end-of-run capture names a commit that never trained the model.
    git_commit_at_start = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=Path(__file__).parent).stdout.strip()

    steps_on_disk = sorted(int(c.name.split("-")[1]) for c in out.glob("checkpoint-*"))
    resuming = bool(args.resume and steps_on_disk)
    resumed_from = steps_on_disk[-1] if resuming else 0
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resuming)
    wall = time.time() - t0

    if is_main:
        li.save_late_and_dense(mve, pooling, str(out))
        # Crash-resume state, not artifacts; snapshot_checkpoints.sh keeps weights-only copies.
        import shutil

        for ckpt in sorted(out.glob("checkpoint-*")):
            shutil.rmtree(ckpt)
        logger.info("checkpoints cleaned up (final late/ + dense_view/ kept)")

        import sentence_transformers as st_pkg
        import transformers

        steps_done = int(trainer.state.global_step)
        steps_this_process = steps_done - resumed_from
        runtime = {
            "model": args.model,
            "proj_dim": args.proj_dim,
            "head_reinit": args.allow_head_reinit,
            "max_seq_length": args.max_seq_length,
            "per_device_batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "mini_batch_num_tokens": args.mini_batch_num_tokens or None,
            "score_mini_batch_size": args.score_mini_batch_size,
            "world_size": world_size,
            "steps": steps_done,
            "pairs_seen": steps_done * args.batch_size * world_size,
            "wall_time_s": round(wall, 1),
            # Rates use steps completed in THIS process, not cumulative global_step: a resumed
            # run divides all prior steps by only the resumed wall time and reports a throughput
            # it never achieved.
            "steps_this_process": steps_this_process,
            "steps_per_s": round(steps_this_process / wall, 4) if wall else None,
            "pairs_per_s": round(steps_this_process * args.batch_size * world_size / wall, 2) if wall else None,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "seed": args.seed,
            "lr": args.lr,
            "proj_lr": args.proj_lr,
            "warmup_steps": args.warmup_steps,
            "sampler": args.multi_dataset_sampler,
            "attn_implementation": attn or "sdpa",
            "compiled": bool(args.compile),
            "datasets": {k: len(v) for k, v in train_dataset.items()},
            "sentence_transformers": st_pkg.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "git_commit": git_commit_at_start,
            "argv": sys.argv[1:],
        }
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
        import csv

        history = [h for h in trainer.state.log_history if "loss" in h or "train_loss" in h]
        if history:
            keys = sorted({k for h in history for k in h})
            with (out / "train_log.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(history)
        logger.info("done: %d steps in %.1f min; outputs in %s", steps_done, wall / 60, out)


if __name__ == "__main__":
    main()
