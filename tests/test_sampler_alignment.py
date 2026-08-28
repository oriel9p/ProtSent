"""Every DDP rank must draw the same dataset on a given step.

Guards the fix in `_align_proportional_sampler_to_world_size`. Without it,
accelerate hands rank i batches i, i+N, ... so ranks run different datasets on
one step; when one of those datasets uses CoSENT and the rest use CachedMNRL the
ranks emit different numbers of gradient allreduces and NCCL aborts the job.
Measured five times on ESM-C 300M, always at step 11-20.

Run: uv run --no-sync python tests/test_sampler_alignment.py
"""

import pathlib
import sys

import torch
from torch.utils.data import BatchSampler, SequentialSampler

from accelerate.data_loader import BatchSamplerShard
from sentence_transformers.base.sampler import ProportionalBatchSampler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from protein_pipeline import _align_proportional_sampler_to_world_size  # noqa: E402

# The real corpus, from the run's "Recalculated max_steps" line.
BATCH_COUNTS = {"pfam": 600, "afdb": 13234, "string": 7324, "dms_cosent": 488}
BATCH_SIZE = 2048


class _Cat:
    def __init__(self, sizes):
        self.datasets = [range(s) for s in sizes]


def _mixed_step_fraction(world_size, seed=42):
    """Fraction of steps where ranks disagree on dataset, dms vs the rest."""
    names = list(BATCH_COUNTS)
    sizes = [BATCH_COUNTS[n] * BATCH_SIZE for n in names]
    samplers = [
        BatchSampler(SequentialSampler(range(s)), batch_size=BATCH_SIZE, drop_last=True)
        for s in sizes
    ]
    gen = torch.Generator()
    gen.manual_seed(seed)
    multi = ProportionalBatchSampler(_Cat(sizes), samplers, generator=gen, seed=seed)

    bounds, acc = [], 0
    for s in sizes:
        bounds.append((acc, acc + s))
        acc += s

    def which(batch):
        return next(names[k] for k, (lo, hi) in enumerate(bounds) if lo <= batch[0] < hi)

    shards = [
        list(BatchSamplerShard(multi, num_processes=world_size, process_index=r,
                               split_batches=False, even_batches=False))
        for r in range(world_size)
    ]
    steps = min(len(s) for s in shards)
    mixed = sum(
        1
        for t in range(steps)
        if len({which(shards[r][t]) for r in range(world_size)}) > 1
        and "dms_cosent" in {which(shards[r][t]) for r in range(world_size)}
    )
    return mixed / steps, steps


def main():
    # Unpatched, the hazard is real and shows up within the first few steps.
    for ws in (2, 3, 4):
        frac, _ = _mixed_step_fraction(ws)
        assert frac > 0.02, f"expected unpatched mixing at ws={ws}, got {frac:.3%}"
    print("unpatched: mixing present at 2, 3, 4 ranks (as expected)")

    for ws in (2, 3, 4):
        assert _align_proportional_sampler_to_world_size(ws)
        frac, steps = _mixed_step_fraction(ws)
        assert frac == 0.0, f"ws={ws} still mixes on {frac:.2%} of steps"
        print(f"patched ws={ws}: 0 mixed steps over {steps:,}")

    # world_size 1 has nothing to align.
    assert not _align_proportional_sampler_to_world_size(1)
    print("OK")


if __name__ == "__main__":
    main()
