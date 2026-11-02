"""Attention and KV-cache kernels as free functions (S1.2, S1.3).

Mirrors the shape of deepinfer/model_executor/layers/attention_backend.py and
tilelang_ops/{store_kv_cache,load_paged_kv_cache}.py: plain functions over
flat token tensors (``query: [total_q, num_heads, head_dim]``).

The KV cache of one layer is a K tensor and a V tensor, each
``[num_blocks, block_size, num_kv_heads, head_dim]``: the shape DeepInfer's
store and load kernels list for a plain BF16 cache. A token's slot is
``block_id * block_size + offset``; the block ids of a sequence are its block
table, so position ``p`` lives in slot
``block_table[p // block_size] * block_size + p % block_size``. With one block
per sequence and ``block_size`` = the longest sequence, the cache is
contiguous per sequence (S1.3); phase 2 keeps this layout and slot formula
with smaller blocks.

DeepInfer's own attention is MLA-only: FlashMLA sparse kernels that read one
shared K=V latent head through per-query index lists, with no causal flag.
Both its kernels return ``(output, lse)`` and call the scale
``softmax_scale``; its callers drop the lse, so ours return only the output.
There is no dense kernel in DeepInfer, so these are PyTorch stand-ins built on
``scaled_dot_product_attention``. They loop over sequences in Python, using the
host lists in ``InputMetadata``; S2.4 batches decode without the loop, and the
optional phase 7 writes these kernels in Triton.

``enable_gqa=True`` lets each group of ``num_heads / num_kv_heads`` query
heads read one shared K/V head without copying K/V first.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def store_kv_cache(kv: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Write ``kv`` ``[n, H_kv, D]`` into ``kv_cache`` ``[num_blocks, block_size, H_kv, D]``.

    ``slot_mapping`` ``[n]`` int32; a slot of -1 is not stored. As in DeepInfer's
    PyTorch reference (``_masked_cache_store``), -1 is sent to slot 0 and slot 0
    is put back afterwards.
    """
    assert slot_mapping.ndim == 1 and slot_mapping.dtype == torch.int32
    assert kv.shape[0] == slot_mapping.shape[0] and kv.shape[1:] == kv_cache.shape[2:]
    if slot_mapping.numel() == 0:
        return
    flat = kv_cache.view(-1, *kv_cache.shape[2:])
    old_slot_zero = flat[0].clone()
    flat[slot_mapping.clamp_min(0).long()] = kv.to(kv_cache.dtype)
    is_zero = slot_mapping == 0
    first_zero = is_zero.to(torch.int32).argmax().reshape(1)  # first token that really targets slot 0
    row = kv.index_select(0, first_zero)[0].to(kv_cache.dtype)
    flat[0] = torch.where(is_zero.any(), row, old_slot_zero)


def _slots(block_table: torch.Tensor, length: int, block_size: int) -> torch.Tensor:
    """Cache slots of positions ``0 .. length-1`` of one sequence."""
    pos = torch.arange(length, device=block_table.device)
    return block_table[pos // block_size].long() * block_size + pos % block_size


def load_paged_kv_cache(
    kv_cache: torch.Tensor, block_tables: torch.Tensor, seqlens_k: Sequence[int]
) -> torch.Tensor:
    """Gather positions ``[0, seqlen_k)`` of every sequence into one flat
    ``[sum(seqlens_k), H_kv, D]`` buffer, sequence after sequence.

    DeepInfer's version takes precomputed ``load_segments`` rows
    ``[src_block, src_begin, src_end, dst_offset]`` instead of block tables.
    """
    block_size = kv_cache.shape[1]
    flat = kv_cache.view(-1, *kv_cache.shape[2:])
    parts = [flat[_slots(block_tables[i], n, block_size)] for i, n in enumerate(seqlens_k)]
    return torch.cat(parts) if parts else flat[:0]


def prefill_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Sequence[int],
    scale: float,
    cu_seqlens_k: Sequence[int] | None = None,
) -> torch.Tensor:
    """Causal attention for packed sequences.

    query ``[total_q, H, D]``, key ``[total_k, H_kv, D]``, value
    ``[total_k, H_kv, Dv]`` -> ``[total_q, H, Dv]``. Sequence ``i`` has queries
    ``cu_seqlens_q[i]:cu_seqlens_q[i+1]`` and keys
    ``cu_seqlens_k[i]:cu_seqlens_k[i+1]`` (the same as the queries when
    ``cu_seqlens_k`` is None). When a sequence has more keys than queries, its
    queries are its last tokens.
    """
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    if len(cu_seqlens_q) != len(cu_seqlens_k):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same sequences")
    if cu_seqlens_q[0] != 0 or cu_seqlens_q[-1] != query.shape[0]:
        raise ValueError(f"cu_seqlens_q must run from 0 to {query.shape[0]}, got {list(cu_seqlens_q)}")
    if cu_seqlens_k[0] != 0 or cu_seqlens_k[-1] != key.shape[0] or value.shape[0] != key.shape[0]:
        raise ValueError(f"cu_seqlens_k must run from 0 to {key.shape[0]}, got {list(cu_seqlens_k)}")
    out = query.new_empty(*query.shape[:-1], value.shape[-1])
    for qs, qe, ks, ke in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:]):
        lq, lk = qe - qs, ke - ks
        if lq == 0:
            continue
        if lq > lk:
            raise ValueError(f"a sequence has {lq} queries but only {lk} keys")
        # SDPA wants [batch, heads, seq, dim].
        q = query[qs:qe].transpose(0, 1).unsqueeze(0)
        k = key[ks:ke].transpose(0, 1).unsqueeze(0)
        v = value[ks:ke].transpose(0, 1).unsqueeze(0)
        if lq == lk:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        else:
            i = torch.arange(lq, device=query.device)[:, None]
            j = torch.arange(lk, device=query.device)[None, :]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=j <= i + (lk - lq), scale=scale, enable_gqa=True)
        out[qs:qe] = o[0].transpose(0, 1)
    return out


def decode_attn(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens_k: Sequence[int],
    scale: float,
) -> torch.Tensor:
    """One query token per sequence against all its cached keys.

    query ``[num_reqs, H, D]``; sequence ``i`` reads positions
    ``[0, seqlens_k[i])`` through ``block_tables[i]``. The current token's K/V
    must already be stored.
    """
    if query.shape[0] != len(seqlens_k):
        raise ValueError(f"decode wants one query per sequence: {query.shape[0]} queries, {len(seqlens_k)} sequences")
    block_size = k_cache.shape[1]
    k_flat = k_cache.view(-1, *k_cache.shape[2:])
    v_flat = v_cache.view(-1, *v_cache.shape[2:])
    out = query.new_empty(*query.shape[:-1], v_cache.shape[-1])
    for i, n in enumerate(seqlens_k):
        slots = _slots(block_tables[i], n, block_size)
        q = query[i : i + 1].transpose(0, 1).unsqueeze(0)  # [1, H, 1, D]
        k = k_flat[slots].transpose(0, 1).unsqueeze(0)  # [1, H_kv, n, D]
        v = v_flat[slots].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
        out[i] = o[0, :, 0]
    return out
