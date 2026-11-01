"""Rotary position embedding (S1.2).

Mirrors deepinfer/model_executor/layers/rotary_embedding.py:
- the cos/sin cache is float32, shape ``[max_position, rotary_dim]``: the
  first half of each row holds cos, the second half sin. It is built with
  ``torch.polar``, exactly as DeepInfer does;
- ``forward(positions, query, key)`` rotates ``query`` and ``key`` in place
  and returns None. Tokens are flat: ``query`` is
  ``[num_tokens, num_heads, rotary_dim]`` and ``positions`` is ``[num_tokens]``,
  so every token carries its own position;
- ``get_rope`` is memoized, as in DeepInfer.

``apply_rotary`` below stands in for ``tile_kernels.transform.apply_rotary``,
an external package that is not in the DeepInfer repo, so its internal
precision could not be checked. This version computes in float32 and casts
once. Hugging Face instead casts cos and sin to bfloat16 first and then does
the arithmetic in bfloat16. Like DeepInfer's kernel wrapper
(``tilelang_ops/fused_transform.py``), ``forward`` refuses a cache that is not
float32 or not on the query's device.

Neox style (Qwen, Llama) rotates the two halves of each head
``(x[:d/2], x[d/2:])``; interleaved style (DeepSeek-V4) rotates the pairs
``(x[0], x[1]), (x[2], x[3]), ...``. DeepInfer passes
``interleaved = not is_neox_style``. ``conjugate=True`` rotates backwards; V4
uses it on its attention output.

Differences: no YaRN scaling (it returns with the DeepSeek pieces in phase 5);
the cache device is a parameter instead of a hard-coded ``"cuda"``, and the
memo key includes it; a memo entry whose cache was cast or moved is rebuilt;
``model_loader.load_model`` gives each load its own memo (``rope_memo_scope``),
because unlike DeepInfer one process here often holds several models.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def _build_cos_sin_cache(t: torch.Tensor, inv_freq: torch.Tensor, mscale: float = 1.0) -> torch.Tensor:
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cos = freqs_cis.real * mscale
    sin = freqs_cis.imag * mscale
    return torch.cat((cos, sin), dim=-1)


def apply_rotary(
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key: torch.Tensor | None = None,
    *,
    positions: torch.Tensor,
    interleaved: bool = False,
    conjugate: bool = False,
) -> None:
    half = cos_sin_cache.shape[-1] // 2
    cos_sin = cos_sin_cache[positions]  # [num_tokens, rotary_dim]
    cos = cos_sin[:, None, :half]  # broadcast over heads
    sin = cos_sin[:, None, half:]
    if conjugate:
        sin = -sin
    for x in (query, key):
        if x is None:
            continue
        xf = x.float()
        if interleaved:
            x1, x2 = xf[..., 0::2], xf[..., 1::2]
            out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
        else:
            x1, x2 = xf[..., :half], xf[..., half:]
            out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        x.copy_(out)


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    mscale: float = 1.0

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        cache = self._compute_cos_sin_cache(device)
        assert cache.dtype == torch.float
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, device) -> torch.Tensor:
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim
        return 1.0 / (self.base**exponent)

    def _compute_cos_sin_cache(self, device) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(device)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)
        return _build_cos_sin_cache(t, inv_freq, mscale=self.mscale)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        conjugate: bool = False,
    ) -> None:
        assert positions.shape[0] == query.shape[0], f"{positions.shape=}, {query.shape=}"
        assert query.shape[-1] == self.rotary_dim
        if key is not None:
            assert key.shape[-1] == self.rotary_dim
        assert self.cos_sin_cache.dtype == torch.float32, f"cos_sin_cache is {self.cos_sin_cache.dtype}"
        assert self.cos_sin_cache.device == query.device, f"{self.cos_sin_cache.device=}, {query.device=}"
        apply_rotary(
            query,
            self.cos_sin_cache,
            key,
            positions=positions,
            interleaved=not self.is_neox_style,
            conjugate=conjugate,
        )

    def extra_repr(self) -> str:
        return f"rotary_dim={self.rotary_dim}, max_position={self.max_position_embeddings}, base={self.base}, neox={self.is_neox_style}"


_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


def _resolve(device: torch.device | str | None) -> torch.device:
    device = torch.device(device) if device is not None else torch.get_default_device()
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
) -> RotaryEmbedding:
    if rope_scaling is not None:
        raise ValueError(f"RoPE scaling is not supported yet: {rope_scaling}")
    device = _resolve(device)
    key = (head_size, rotary_dim, max_position, base, is_neox_style, str(device))
    rope = _ROPE_DICT.get(key)
    if rope is None or rope.cos_sin_cache.dtype != torch.float32 or rope.cos_sin_cache.device != device:
        rope = _ROPE_DICT[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style, device=device)
    return rope


@contextmanager
def rope_memo_scope():
    """Use a fresh, empty memo inside the block; restore the old one after."""
    global _ROPE_DICT
    saved, _ROPE_DICT = _ROPE_DICT, {}
    try:
        yield
    finally:
        _ROPE_DICT = saved
