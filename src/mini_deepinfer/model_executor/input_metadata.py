"""What one forward pass needs to know besides the weights (S1.2, S1.3).

Mirrors ``InputMetadata`` in deepinfer/model_executor/input_metadata/
input_metadata.py: an explicit dataclass handed to every layer, with no global
forward context. All tokens of all sequences of one step are flattened into
one dimension of length ``total_q``; there is no batch dimension and no
padding. A step is either all prefill or all decode (``is_prompt``); DeepInfer
rejects mixed steps.

Everything is derived from two numbers per sequence, as in DeepInfer's C++
``SequenceMetadata``:
  ``seqlen_q``  query tokens this step (decode: 1);
  ``seqlen_k``  keys visible to the last query token, including this step's
                tokens, so the step covers positions ``[seqlen_k - seqlen_q, seqlen_k)``.
A prefill with ``seqlen_k > seqlen_q`` continues a sequence whose first
``seqlen_k - seqlen_q`` tokens already have KV in the cache (chunked prefill).

Device tensors (int32, as in DeepInfer):
  ``input_ids``, ``positions``  [total_q]
  ``slot_mapping``              [total_q] cache slot that each token's K/V is
                                written to, ``block_id * block_size + offset``;
                                -1 means "do not store". DeepInfer's dense-layout
                                equivalent is ``slot_mapping_store_ca["1"]``.
  ``block_tables``              [num_reqs, max_num_blocks] block ids of each
                                sequence; unused entries are 0.

Differences from DeepInfer:
- ``seqlens_q``/``seqlens_k`` are kept as host lists, and ``cu_seqlens_q``/
  ``cu_seqlens_k`` are computed from them on the host, because the PyTorch
  attention stand-ins loop over sequences in Python. DeepInfer keeps only
  device tensors (``cu_seqlens_q/k`` for prefill, ``context_lens`` per query
  token for decode).
- ``block_tables`` has one row per sequence and is used by prefill too.
  DeepInfer's decode has one row per query token (draft tokens of speculative
  decoding get their own rows), and its prefill gathers cached KV through
  ``load_segments`` instead.
- ``slot_mapping`` and ``block_tables`` are None for a forward without a KV
  cache (S1.2's prompt-only check).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class InputMetadata:
    is_prompt: bool
    input_ids: torch.Tensor  # [total_q] int32
    positions: torch.Tensor  # [total_q] int32
    seqlens_q: list[int]  # per sequence, host
    seqlens_k: list[int]  # per sequence, host
    slot_mapping: torch.Tensor | None = None  # [total_q] int32
    block_tables: torch.Tensor | None = None  # [num_reqs, max_num_blocks] int32

    @property
    def num_reqs(self) -> int:
        return len(self.seqlens_q)

    @property
    def total_q(self) -> int:
        return sum(self.seqlens_q)

    @property
    def cu_seqlens_q(self) -> list[int]:
        """Start of each sequence's query tokens in the flat dimension, then total_q."""
        return [0, *itertools.accumulate(self.seqlens_q)]

    @property
    def cu_seqlens_k(self) -> list[int]:
        return [0, *itertools.accumulate(self.seqlens_k)]

    @classmethod
    def for_prompts(cls, prompts: Sequence[Sequence[int]], device: torch.device | str) -> InputMetadata:
        """Whole prompts from position 0, packed back to back, no KV cache."""
        lens = [len(p) for p in prompts]
        if any(n == 0 for n in lens):
            raise ValueError("empty prompt")
        ids = [int(t) for p in prompts for t in p]
        positions = [i for n in lens for i in range(n)]
        return cls(
            is_prompt=True,
            input_ids=torch.tensor(ids, dtype=torch.int32, device=device),
            positions=torch.tensor(positions, dtype=torch.int32, device=device),
            seqlens_q=lens,
            seqlens_k=list(lens),
        )

    def last_token_indices(self) -> list[int]:
        """Row of each sequence's last query token: where its next-token logits come from."""
        return [end - 1 for end in self.cu_seqlens_q[1:]]
