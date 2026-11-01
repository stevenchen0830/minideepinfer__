"""Tiny random Qwen3 checkpoints written by Hugging Face transformers.

They run on the CPU in a second and need no download, so the loader and the
model can be compared with Hugging Face exactly (float32) in every test run.
"""

import pytest
import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from mini_deepinfer.model_executor.utils import init_torch

TINY = dict(
    vocab_size=97,
    hidden_size=48,  # != num_heads * head_dim, so q_proj is not square
    intermediate_size=80,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=256,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
)


def make_hf_model(*, seed: int = 0, dtype: torch.dtype = torch.float32, **overrides) -> HFQwen3ForCausalLM:
    """A random HF Qwen3 whose norm weights are not all ones.

    HF initializes norm weights to 1, which would hide a missing or misplaced
    norm weight, so every weight is redrawn here.
    """
    cfg = Qwen3Config(**{**TINY, "tie_word_embeddings": True, **overrides})
    torch.manual_seed(seed)
    model = HFQwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.3 * torch.randn_like(p))
            elif "embed_tokens" in name or "lm_head" in name:
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
    model.config._attn_implementation = "eager"  # reference independent of SDPA
    return model.to(dtype)


def save_hf_model(model, path, *, shards: bool) -> str:
    # A small max_shard_size forces an index file and many shards.
    model.save_pretrained(path, max_shard_size="8KB" if shards else "1GB")
    return str(path)


@pytest.fixture(scope="session", autouse=True)
def deepinfer_torch_settings():
    """The process settings DeepInfer's worker applies (matters for GPU tests)."""
    init_torch()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Factory: (hf_model, checkpoint_dir) for the given overrides, cached per session."""
    cache = {}

    def make(*, shards: bool = False, dtype: torch.dtype = torch.float32, **overrides):
        key = (shards, dtype, tuple(sorted(overrides.items())))
        if key not in cache:
            model = make_hf_model(dtype=dtype, **overrides)
            path = tmp_path_factory.mktemp("tiny-qwen3")
            cache[key] = (model, save_hf_model(model, path, shards=shards))
        return cache[key]

    return make
