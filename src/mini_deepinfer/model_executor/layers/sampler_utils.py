"""Logits from hidden states (S1.2).

Mirrors ``get_logits`` in deepinfer/model_executor/layers/sampler_utils.py.
The model's forward ends at the final norm. The sampler picks the rows it
needs and multiplies them by the LM head weight here, into a float32 tensor.
The sampler itself arrives in phase 3.
"""

from __future__ import annotations

import torch

from mini_deepinfer.model_executor.layers.linear import bf16_gemm_nt


def get_logits(hidden_states: torch.Tensor, embedding: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """``[n, hidden]`` x ``[vocab, hidden]`` -> float32 logits ``[n, vocab]``."""
    logits = torch.empty(hidden_states.size(0), embedding.size(0), device=hidden_states.device, dtype=torch.float)
    bf16_gemm_nt(hidden_states, embedding, logits)
    assert logits.shape[-1] == vocab_size
    return logits
