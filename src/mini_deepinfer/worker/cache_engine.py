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
