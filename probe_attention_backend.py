#!/usr/bin/env python
"""Late-interaction training throughput: attention backend x token packing x compile.

Standalone (no ProtSent imports) so it runs in an isolated venv while the main
campaign holds the shared one. Every rate is steady-state: warmup steps are timed
but excluded, and a recompile inside the measured window is flagged rather than
averaged away.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/opt/hpc/ddofer/ProtSent")
from throughput_probe import steady_state_rate  # the tested harness

from datasets import Dataset
from sentence_transformers import (MultiVectorEncoder, MultiVectorEncoderTrainer,
                                   MultiVectorEncoderTrainingArguments)
from sentence_transformers.base.modules import Dense, Normalize, Transformer
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
from transformers import TrainerCallback


class StepTimer(TrainerCallback):
    def __init__(self): self.times, self._t = [], None
    def on_step_begin(self, *a, **k): torch.cuda.synchronize(); self._t = time.perf_counter()
    def on_step_end(self, *a, **k):
        torch.cuda.synchronize(); self.times.append(time.perf_counter() - self._t)


def build(model_name, attn, proj_dim, max_len, compile_backbone, fp16=False, unpad="auto"):
    kw = {"dtype": torch.float16 if fp16 else torch.bfloat16}
    if attn != "none":
        kw["attn_implementation"] = attn
    tr = Transformer(model_name, model_kwargs=kw)
    if unpad != "auto":
        tr.unpad_inputs = unpad == "on"
    dim = tr.get_embedding_dimension()
    mods = [tr, Dense(dim, proj_dim, bias=False, activation_function=None,
                      module_input_name="token_embeddings"),
            MultiVectorMask(skiplist_words=["<cls>", "<eos>"], skiplist_tasks=["query", "document"]),
            Normalize(module_input_name="token_embeddings")]
    mve = MultiVectorEncoder(modules=mods, device="cuda")
    mve.max_seq_length = max_len
    for n, p in mve.named_parameters():
        if ".pooler." in n or "contact_head" in n or "lm_head" in n:
            p.requires_grad_(False)
    if compile_backbone:
        # The documented API. dynamic=True so a compiled graph handles variable
        # sequence lengths: with flash-attention unpadding the flattened length
        # changes every step, and a static compile would re-trace continually.
        mve.compile(dynamic=True)
    return mve


DATA = "/storage/users/ddofer/data/protsent-data-dc40"


def load_pairs(source, n, seed=0):
    """Sample pairs from RANDOM row groups.

    These parquets are sorted by group_id, so reading the file prefix yields a
    severely length-biased sample (mean 34 tokens for pfam, against 168 sampled
    across the file). Padding waste is the whole subject of this probe, so the
    length distribution has to be the real one.
    """
    rng = np.random.default_rng(seed)
    if source == "string":
        f = pq.ParquetFile(f"{DATA}/stringdb_train_15M.parquet")
        picks = sorted(rng.choice(f.metadata.num_row_groups, 12, replace=False).tolist())
        pairs = []
        for g in picks:
            d = f.read_row_group(g, columns=["seq1", "seq2"]).to_pydict()
            pairs += [{"sentence_0": a, "sentence_1": b} for a, b in zip(d["seq1"], d["seq2"])]
            if len(pairs) >= n:
                break
    else:
        f = pq.ParquetFile(f"{DATA}/{source}_sorted.parquet")
        picks = sorted(rng.choice(f.metadata.num_row_groups, 12, replace=False).tolist())
        pairs = []
        for g in picks:
            d = f.read_row_group(g, columns=["sequence", "group_id"]).to_pydict()
            s, gr = d["sequence"], d["group_id"]
            pairs += [{"sentence_0": s[i], "sentence_1": s[i + 1]}
                      for i in range(len(s) - 1) if gr[i] == gr[i + 1]]
            if len(pairs) >= n:
                break
    rng.shuffle(pairs)
    return Dataset.from_list(pairs[:n])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GrimSqueaker/ProtSent-V2-35M")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--mini_batch_size", type=int, default=64)
    p.add_argument("--mini_batch_num_tokens", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fp16", action="store_true", help="load fp16 instead of bf16")
    p.add_argument("--unpad", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measured", type=int, default=20)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--source", default="pfam", choices=["pfam", "afdb", "string"])
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="throughput_matrix.jsonl")
    a = p.parse_args()

    steps = a.warmup + a.measured
    ds = load_pairs(a.source, steps * a.batch_size + 256)
    mve = build(a.model, a.attn, 64, a.max_seq_length, a.compile, a.fp16, a.unpad)
    loss = CachedMultiVectorMultipleNegativesRankingLoss(
        mve, mini_batch_size=a.mini_batch_size,
        mini_batch_num_tokens=a.mini_batch_num_tokens or None, score_mini_batch_size=32)
    timer = StepTimer()
    args = MultiVectorEncoderTrainingArguments(
        output_dir=f"/tmp/tp_{a.tag}", max_steps=steps, per_device_train_batch_size=a.batch_size,
        bf16=not a.fp16, fp16=a.fp16, learning_rate=1e-5, warmup_steps=0, save_strategy="no", logging_strategy="no",
        report_to=[], batch_sampler=BatchSamplers.BATCH_SAMPLER, dataloader_num_workers=4,
        dataloader_drop_last=True, seed=42)
    MultiVectorEncoderTrainer(model=mve, args=args, train_dataset=ds, loss=loss,
                              callbacks=[timer]).train()

    res = steady_state_rate(timer.times, a.batch_size, a.warmup)
    res |= {"tag": a.tag, "attn": a.attn, "mini_batch_size": a.mini_batch_size,
            "mini_batch_num_tokens": a.mini_batch_num_tokens, "compile": a.compile, "fp16": a.fp16, "unpad": a.unpad,
            "batch_size": a.batch_size, "model": a.model, "source": a.source,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    with open(a.out, "a") as fh: fh.write(json.dumps(res) + "\n")
    print(f"RESULT {a.tag}: {res['pairs_per_s']:.1f} pairs/s (blended {res['blended_pairs_per_s']:.1f}), "
          f"steady={res['steady']} outliers={res['outlier_steps']} peak={res['peak_vram_gb']}GB "
          f"warmup={res['warmup_s']:.1f}s")


if __name__ == "__main__":
    main()
