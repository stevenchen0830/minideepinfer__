"""What one forward pass needs to know besides the weights (S1.2).

Mirrors ``InputMetadata`` in deepinfer/model_executor/input_metadata/
input_metadata.py: an explicit dataclass handed to every layer, with no global
forward context. All tokens of all sequences are flattened into one dimension
of length ``total_q``; there is no batch dimension and no padding.

This first version carries only what a prompt-only forward without a KV cache
needs. DeepInfer's has many more fields (slot mappings, block tables,
``cu_seqlens_k``, context lengths, ...); they arrive with the KV cache in S1.3
and the paged cache in phase 2. DeepInfer keeps ``cu_seqlens_q`` on the GPU;
here it is a host list because the attention stand-in loops over sequences in
Python.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class InputMetadata:
    is_prompt: bool
    input_ids: torch.Tensor  # [total_q] int64
    positions: torch.Tensor  # [total_q] int64, the position of each token in its sequence
    cu_seqlens_q: list[int]  # [num_reqs + 1], start offset of each sequence, then total_q

    @property
    def num_reqs(self) -> int:
        return len(self.cu_seqlens_q) - 1

    @property
    def total_q(self) -> int:
        return self.cu_seqlens_q[-1]

    @classmethod
    def for_prompts(cls, prompts: Sequence[Sequence[int]], device: torch.device | str) -> InputMetadata:
        """Whole prompts, each starting at position 0, packed back to back."""
        lens = [len(p) for p in prompts]
        if any(n == 0 for n in lens):
            raise ValueError("empty prompt")
        ids = [int(t) for p in prompts for t in p]
        positions = [i for n in lens for i in range(n)]
        return cls(
            is_prompt=True,
            input_ids=torch.tensor(ids, dtype=torch.long, device=device),
            positions=torch.tensor(positions, dtype=torch.long, device=device),
            cu_seqlens_q=[0, *itertools.accumulate(lens)],
        )

    def last_token_indices(self) -> list[int]:
        """Row of each sequence's last token: where its next-token logits come from."""
        return [end - 1 for end in self.cu_seqlens_q[1:]]
