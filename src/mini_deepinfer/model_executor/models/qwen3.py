"""Qwen3 dense model (S1.2; KV cache since S1.3).

DeepInfer ships only DeepSeek-V4, so this file follows the structure of its
model code (models/deepseek.py, models/decoder_layer.py) on Qwen3's
architecture; vLLM's qwen3.py is the other reference.

One forward pass, ``model(kv_caches, input_metadata)``, with ``T`` = total
query tokens of all sequences in the step:

    input_ids [T] -> embed_tokens -> hidden [T, H]
    each layer: (hidden, residual) -> add + RMSNorm -> attention (reads and
                writes this layer's kv_caches[i]) -> add + RMSNorm -> MLP
    hidden += residual -> final RMSNorm -> hidden [T, H]

Attention first stores this step's K and V at ``slot_mapping``, then:
- prefill without cached history: attends over this step's own K/V;
- prefill continuing a sequence (chunked): gathers all its K/V from the cache;
- decode: each sequence's single query reads all its K/V from the cache.

(In DeepInfer each "add + RMSNorm" is one kernel; here it is two torch ops.)

Logits are not computed here: as in DeepInfer, the caller picks the rows it
needs and calls ``sampler_utils.get_logits(rows, lm_head.weight, vocab)``.

Qwen3 attention, per token:

    qkv = x W_qkv^T, split into q [H_q, d], k [H_kv, d], v [H_kv, d]
    q, k = RMSNorm over d for each head (q_norm, k_norm), then RoPE
    o = causal softmax(q k^T / sqrt(d)) v, each K/V head shared by H_q / H_kv query heads
    out = o W_o^T

The per-head q/k norm is Qwen3's; Qwen2 and Llama have none.

Parameter names equal the Hugging Face names except two merged weights:
``self_attn.qkv_proj`` = cat(q_proj, k_proj, v_proj) and
``mlp.gate_up_proj`` = cat(gate_proj, up_proj), concatenated along dim 0.
DeepInfer names its norms ``attn_layernorm``/``ffn_layernorm``/``output_norm``;
the HF names are kept here so the name map stays small.

The RoPE cache has ``max_position_embeddings`` rows (from config.json).
DeepInfer sizes it by the engine's ``runtime_max_model_len`` instead, a serving
limit (``max_model_len``, default 4096, plus some slack for speculative
decoding); that engine setting arrives with the scheduler steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from mini_deepinfer.hugging_face.config import HFModelConfig
from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.activation import silu_and_mul
from mini_deepinfer.model_executor.layers.attention_backend import (
    decode_attn,
    load_paged_kv_cache,
    prefill_attn_varlen,
    store_kv_cache,
)
from mini_deepinfer.model_executor.layers.layernorm import RMSNorm
from mini_deepinfer.model_executor.layers.linear import ReplicatedLinear
from mini_deepinfer.model_executor.layers.rotary_embedding import get_rope
from mini_deepinfer.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

if TYPE_CHECKING:
    from mini_deepinfer.worker.cache_engine import KVCache


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: HFModelConfig, layer_idx: int, *, device=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = cfg.q_size
        self.kv_size = cfg.kv_size
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ReplicatedLinear(cfg.hidden_size, self.q_size + 2 * self.kv_size)
        self.o_proj = ReplicatedLinear(self.q_size, cfg.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=cfg.max_position_embeddings,
            base=cfg.rope_theta,
            is_neox_style=True,
            device=device,
        )

    def forward(
        self, hidden_states: torch.Tensor, kv_cache: KVCache | None, input_metadata: InputMetadata
    ) -> torch.Tensor:
        md = input_metadata
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # The norms return new tensors, so RoPE below never writes into qkv.
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(num_tokens, self.num_kv_heads, self.head_dim))
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        self.rotary_emb(md.positions, q, k)

        if kv_cache is not None:
            store_kv_cache(k, kv_cache.k_cache, md.slot_mapping)
            store_kv_cache(v, kv_cache.v_cache, md.slot_mapping)
        if not md.is_prompt:
            o = decode_attn(q, kv_cache.k_cache, kv_cache.v_cache, md.block_tables, md.seqlens_k, self.scaling)
        elif md.seqlens_k == md.seqlens_q:
            # No cached history: this step's K/V are all the keys.
            o = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, self.scaling)
        else:
            # Chunked prefill: earlier tokens' K/V come back from the cache.
            k_all = load_paged_kv_cache(kv_cache.k_cache, md.block_tables, md.seqlens_k)
            v_all = load_paged_kv_cache(kv_cache.v_cache, md.block_tables, md.seqlens_k)
            o = prefill_attn_varlen(q, k_all, v_all, md.cu_seqlens_q, self.scaling, md.cu_seqlens_k)
        return self.o_proj(o.reshape(num_tokens, self.q_size))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: HFModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = ReplicatedLinear(cfg.hidden_size, 2 * cfg.intermediate_size)
        self.down_proj = ReplicatedLinear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: HFModelConfig, layer_idx: int, *, device=None) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, layer_idx, device=device)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache | None,
        input_metadata: InputMetadata,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes and returns a (hidden_states, residual) pair, as DeepInfer's
        decoder layer does; the first layer gets ``residual=None``."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, kv_cache, input_metadata)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(self, cfg: HFModelConfig, *, device=None) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, i, device=device) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, kv_caches: Sequence[KVCache] | None, input_metadata: InputMetadata) -> torch.Tensor:
        if input_metadata.slot_mapping is None and kv_caches is not None:
            raise ValueError("kv_caches given but input_metadata has no slot_mapping")
        if kv_caches is None and (not input_metadata.is_prompt or input_metadata.seqlens_k != input_metadata.seqlens_q):
            raise ValueError("decode and chunked prefill need kv_caches")
        hidden_states = self.embed_tokens(input_metadata.input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, residual = layer(hidden_states, kv_cache, input_metadata, residual)
        if residual is not None:
            hidden_states = hidden_states + residual
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: HFModelConfig, *, device=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg, device=device)
        tied = self.model.embed_tokens.weight if cfg.tie_word_embeddings else None
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, weight=tied)

    def forward(self, kv_caches: Sequence[KVCache] | None, input_metadata: InputMetadata) -> torch.Tensor:
        """Final hidden states ``[total_q, hidden]``, after the last norm.

        ``kv_caches[i]`` is layer ``i``'s cache. With ``kv_caches=None`` the step
        must be a prefill without cached history, and nothing is stored (S1.2).
        """
        return self.model(kv_caches, input_metadata)
