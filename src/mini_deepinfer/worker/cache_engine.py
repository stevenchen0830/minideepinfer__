"""KV-cache memory (S1.3).

Mirrors the interface of deepinfer/worker/cache_engine.py: a ``KVCache``
dataclass per layer, a list-like ``gpu_cache`` so that the model can take
``kv_caches[layer_idx]``, and ``get_cache_block_size`` for the memory budget.

The allocation is our own. DeepInfer's cache only serves DeepSeek-V4
attention (one shared, FP8-compressed latent head; it refuses plain
full-attention layers), and it packs all layers into one block-major uint8
buffer. Qwen3 needs a separate K and V for each layer, so each layer gets two
tensors of shape ``[num_blocks, block_size, num_kv_heads, head_dim]``, the
shape DeepInfer's store and load kernels list for a plain BF16 cache (DeepInfer
itself never allocates one).

A block holds ``block_size`` consecutive positions of one sequence. S1.3 uses
one block per sequence with ``block_size`` = the longest allowed sequence,
which makes each sequence's cache contiguous; phase 2 uses small blocks
(DeepInfer: 128) handed out by a block manager. The memory is not
initialized, as in DeepInfer: correct code only reads slots it wrote.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_deepinfer.hugging_face.config import HFModelConfig


@dataclass
class KVCache:
    k_cache: torch.Tensor  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor


class CacheEngine:
    def __init__(
        self,
        cfg: HFModelConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError(f"num_blocks and block_size must be positive, got {num_blocks}, {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype or cfg.dtype
        shape = (num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        self.gpu_cache: list[KVCache] = [
            KVCache(
                k_cache=torch.empty(shape, dtype=self.dtype, device=device),
                v_cache=torch.empty(shape, dtype=self.dtype, device=device),
            )
            for _ in range(cfg.num_hidden_layers)
        ]

    def clone_blocks(self, clone_blocks: list[tuple[int, int]]) -> list[int]:
        """Copy-on-write copies for forked sequences (S2.2).

        The block part of DeepInfer's ``clone_and_zero_states_and_blocks``
        (csrc/engine/scheduler.hpp): each ``(src, dst)`` pair copies block
        ``src`` into block ``dst`` in every layer's K and V; ``dst == 0`` means
        "no copy, only release ``src``". Returns every ``src``: the caller
        unpins them once the copies are done (DeepInfer: after a CUDA event,
        in the same step's ``pre_sampler_sync``).

        DeepInfer issues one ``cudaMemcpyAsync`` per pair over its block-major
        buffer, in order. Here one gather and one scatter per tensor do all
        pairs at once. A block that is both a source and a destination, or a
        destination twice, is refused; the scheduler never produces either.
        """
        srcs = [src for src, _ in clone_blocks]
        copies = [(src, dst) for src, dst in clone_blocks if dst]
        if copies:
            dsts = [dst for _, dst in copies]
            if len(set(dsts)) != len(dsts) or set(dsts) & {src for src, _ in copies}:
                raise ValueError(f"overlapping clone pairs: {copies}")
            device = self.gpu_cache[0].k_cache.device
            index = torch.tensor(copies, dtype=torch.long)
            if device.type == "cuda":
                index = index.pin_memory()
            index = index.to(device, non_blocking=True)
            src_idx, dst_idx = index[:, 0], index[:, 1]
            for layer in self.gpu_cache:
                for cache in (layer.k_cache, layer.v_cache):
                    cache.index_copy_(0, dst_idx, cache.index_select(0, src_idx))
        return srcs

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.gpu_cache[layer_idx]

    def __len__(self) -> int:
        return len(self.gpu_cache)

    @staticmethod
    def get_cache_block_size(block_size: int, cfg: HFModelConfig, dtype: torch.dtype | None = None) -> int:
        """Bytes of one block across all layers, K and V."""
        dtype = dtype or cfg.dtype
        per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * dtype.itemsize
        return block_size * per_token
