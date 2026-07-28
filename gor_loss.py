"""Global Orthogonal Regularization wrapper for SentenceTransformer losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class LossWithGOR(nn.Module):
    """Primary sentence loss plus Global Orthogonal Regularization.

    ``GlobalOrthogonalRegularizationLoss`` was added in sentence-transformers 5.3.
    Import it lazily so ordinary training paths still import cleanly when GOR is
    disabled, while opt-in runs fail with a clear dependency message.
    """

    def __init__(self, model, base_loss: nn.Module, gor_weight: float = 0.1, mini_batch_size: int = 32):
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

            self.gor = GlobalOrthogonalRegularizationLoss(model)

            # Override the forward pass of GlobalOrthogonalRegularizationLoss to be memory efficient.
            # This prevents CUDA OOM on large batch sizes by executing the forward pass in mini-batches.
            def patched_forward(sentence_features, labels=None):
                features_list = list(sentence_features)
                if not features_list:
                    return {}
                
                first_feature = features_list[0]
                first_tensor = next(iter(first_feature.values()))
                batch_size = first_tensor.size(0)
                
                embeddings = []
                for feature in features_list:
                    col_embs = []
                    for start in range(0, batch_size, mini_batch_size):
                        end = min(start + mini_batch_size, batch_size)
                        sliced = {
                            k: v[start:end] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size
                            else v
                            for k, v in feature.items()
                        }
                        mb_emb = model(sliced)["sentence_embedding"]
                        col_embs.append(mb_emb)
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
