"""Loss wrappers that bound peak activation memory by subsampling the batch."""

from __future__ import annotations

import torch
import torch.nn as nn

from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (  # noqa: E501
    _create_minibatch,
    _get_batch_size,
)


class SubsampledLoss(nn.Module):
    """Run ``base_loss`` on at most ``max_samples`` rows of each batch.

    For losses with no gradient cache — CoSENTLoss above all — peak memory is set
    by the training batch size, because the whole batch is embedded with grad and
    backpropagated at once. sentence-transformers uses a single
    ``per_device_train_batch_size`` for every dataset in a multi-task dict, so the
    only way to keep a large contrastive batch *and* an auxiliary CoSENT target on
    one GPU is to cap what the auxiliary loss actually consumes.

    The cap takes the leading rows of the batch, so it is only an unbiased sample
    when the underlying dataset is itself unordered. ``dms_cosent.parquet`` is
    (rows are interleaved across assays); a corpus sorted by cluster is not.
    """

    def __init__(self, base_loss: nn.Module, max_samples: int):
        super().__init__()
        self.base_loss = base_loss
        self.max_samples = int(max_samples)

    def forward(self, sentence_features, labels: torch.Tensor | None = None):
        features = list(sentence_features)
        if self.max_samples > 0 and features:
            n = min(_get_batch_size(features[0]), self.max_samples)
            features = [_create_minibatch(f, 0, n) for f in features]
            if labels is not None:
                labels = labels[:n]
        return self.base_loss(features, labels)


class LossWithGOR(nn.Module):
    """Primary sentence loss plus Global Orthogonal Regularization.

    ``GlobalOrthogonalRegularizationLoss`` was added in sentence-transformers 5.3.
    Import it lazily so ordinary training paths still import cleanly when GOR is
    disabled, while opt-in runs fail with a clear dependency message.
    """

    def __init__(
        self,
        model,
        base_loss: nn.Module,
        gor_weight: float = 1.0,
        mini_batch_size: int = 48,
        max_samples: int = 192,
        mean_weight: float = 1.0,
    ):
        super().__init__()
        if gor_weight < 0:
            raise ValueError("gor_weight must be non-negative")

        self.base_loss = base_loss
        self.gor_weight = float(gor_weight)
        self.gor: nn.Module | None = None
        if self.gor_weight > 0:
            try:
                from sentence_transformers.sentence_transformer import losses

                GlobalOrthogonalRegularizationLoss = (
                    losses.GlobalOrthogonalRegularizationLoss
                )
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "GlobalOrthogonalRegularizationLoss requires "
                    "sentence-transformers>=5.3.0"
                ) from exc

            # Only the mean term is exposed. The objective is
            # gor_weight * (mean_weight * mean + second_moment_weight * second_moment),
            # so (gor_weight, mean_weight) already spans every ratio worth setting --
            # a second_moment_weight knob, and an aggregation knob that is exactly a
            # factor of the column count, would just be two more ways to write one run.
            self.gor = GlobalOrthogonalRegularizationLoss(model, mean_weight=mean_weight)

            # Replace GlobalOrthogonalRegularizationLoss.forward, which embeds the
            # whole batch at once with grad enabled. GOR is a moment-matching
            # regularizer, so a subsample estimates it fine, and capping at
            # ``max_samples`` is what keeps peak memory bounded: the base loss here
            # is CachedMNRL, whose whole point is that a large batch never has all
            # of its activations live at once.
            #
            # Slice with sentence-transformers' own helpers rather than indexing
            # tensors by their first dimension. Under DataCollatorWithFlattening a
            # batch arrives packed into one token axis with `cu_seq_lens_q`
            # metadata, so a naive `v[:n]` matches nothing, silently embeds the
            # whole batch and OOMs.
            def patched_forward(sentence_features, labels=None):
                features_list = list(sentence_features)
                if not features_list:
                    return {}

                embeddings = []
                for feature in features_list:
                    full_size = _get_batch_size(feature)
                    n = min(full_size, max_samples) if max_samples > 0 else full_size
                    col_embs = []
                    for start in range(0, n, mini_batch_size):
                        end = min(start + mini_batch_size, n)
                        mb_emb = model(_create_minibatch(feature, start, end))
                        col_embs.append(mb_emb["sentence_embedding"])
                    embeddings.append(torch.cat(col_embs, dim=0))
                return self.gor.compute_loss_from_embeddings(embeddings)

            self.gor.forward = patched_forward

    @staticmethod
    def _reduce(value):
        # sentence-transformers >=5.6 losses may return a dict of named loss
        # terms (GlobalOrthogonalRegularizationLoss returns
        # {"gor_mean": ..., "gor_second_moment": ...}); sum them to a scalar.
        if isinstance(value, dict):
            return sum(value.values())
        return value

    def forward(self, sentence_features, labels: torch.Tensor | None = None):
        loss = self._reduce(self.base_loss(sentence_features, labels))
        if self.gor is None:
            return loss
        gor = self._reduce(self.gor(sentence_features, labels))
        return loss + self.gor_weight * gor
