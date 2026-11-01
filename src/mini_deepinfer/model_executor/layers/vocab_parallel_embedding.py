"""Token embedding and the LM head's weight holder (S1.2).

Mirrors deepinfer/model_executor/layers/vocab_parallel_embedding.py. Despite
the vLLM-era names nothing is split across GPUs. The embedding is a row
gather. ``ParallelLMHead`` only holds a weight: the sampler multiplies the
selected hidden states by it (``sampler_utils.get_logits``), so calling the
module is an error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_deepinfer.model_executor.utils import reconstruct_param


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight is None:
            weight = torch.empty(num_embeddings, embedding_dim)
        if tuple(weight.shape) != (num_embeddings, embedding_dim):
            raise ValueError(f"weight shape {tuple(weight.shape)} != ({num_embeddings}, {embedding_dim})")
        self.weight = reconstruct_param(weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, input_: torch.Tensor) -> None:
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
