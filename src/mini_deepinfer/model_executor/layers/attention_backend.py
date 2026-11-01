"""Attention kernels as free functions (S1.2).

Mirrors the shape of deepinfer/model_executor/layers/attention_backend.py:
kernels are plain functions over flat token tensors
(``query: [total_q, num_heads, head_dim]``). DeepInfer's are FlashMLA sparse
kernels: prefill attends over a flat KV buffer that the caller first gathers
from the paged cache, picking keys by top-k indices; decode reads the paged
cache directly. Both return ``(output, lse)`` and call the scale
``softmax_scale``; DeepInfer's callers drop the lse, so ours returns only the
output.

This is the simplest dense version: causal attention within each sequence,
with no KV cache, so queries, keys and values cover the same tokens. A KV
cache arrives in S1.3 and the paged one in phase 2.

The sequences of one call are packed back to back; ``cu_seqlens_q`` holds the
start offsets plus the total (``[0, len0, len0 + len1, ...]``), so sequence
``i`` is ``query[cu[i]:cu[i + 1]]``. No token attends across a sequence
boundary.

``enable_gqa=True`` lets each group of ``num_heads / num_kv_heads`` query
heads read one shared K/V head without copying K/V first.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def prefill_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Sequence[int],
    scale: float,
) -> torch.Tensor:
    """query ``[T, H, D]``, key ``[T, H_kv, D]``, value ``[T, H_kv, Dv]`` -> output ``[T, H, Dv]``."""
    if key.shape[0] != query.shape[0] or value.shape[0] != query.shape[0]:
        raise ValueError(
            f"query, key and value must cover the same tokens: {query.shape[0]}, {key.shape[0]}, {value.shape[0]}"
        )
    if cu_seqlens_q[0] != 0 or cu_seqlens_q[-1] != query.shape[0]:
        raise ValueError(f"cu_seqlens_q must run from 0 to {query.shape[0]}, got {list(cu_seqlens_q)}")
    out = query.new_empty(*query.shape[:-1], value.shape[-1])
    for start, end in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:]):
        if end <= start:
            continue
        # SDPA wants [batch, heads, seq, dim].
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        out[start:end] = o[0].transpose(0, 1)
    return out
