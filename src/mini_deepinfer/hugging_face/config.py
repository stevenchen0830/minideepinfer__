"""Read a Hugging Face config.json into a small, checked config object (S1.2).

Mirrors deepinfer/hugging_face/config.py: plain ``json.loads`` (no
``transformers.AutoConfig``), then hard checks that refuse anything the model
code does not implement, so an unsupported checkpoint fails at load time
instead of producing wrong numbers. Keys the model needs are required, not
guessed: a missing ``head_dim`` or ``num_key_value_heads`` is refused.

``HFModelConfig`` corresponds to DeepInfer's ``model_cfg`` (an ``AttrDict``
holding ``architecture``, ``dtype`` and the model fields under ``.gpt``),
flattened into a frozen dataclass. DeepInfer's class named ``ModelConfig``
(deepinfer/config.py) is a different object, the engine-level config that
holds ``model_cfg`` together with serving limits such as ``max_model_len``.

Deliberate differences from DeepInfer:
- DeepInfer accepts only ``DeepseekV41ForCausalLM``; this accepts only
  ``Qwen3ForCausalLM``.
- DeepInfer reads the dtype from the ``dtype`` key and requires bfloat16.
  The released Qwen3 checkpoints were written by transformers 4.51 and use
  ``torch_dtype`` and ``rope_theta``; transformers 5 writes ``dtype`` and
  ``rope_parameters``. Both spellings are accepted (``rope_parameters`` wins,
  as in transformers 5; two different values are refused).
- float32 is accepted as well, for the exact tests on tiny random checkpoints.
- DeepInfer's ``get_rope`` supports YaRN scaling. The small Qwen3 models use no
  scaling, so any RoPE scaling is refused here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SUPPORTED_ARCHITECTURES = ("Qwen3ForCausalLM",)
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class HFModelConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def load_hf_model_config(path: str | Path) -> HFModelConfig:
    """Read ``config.json`` (the file, or the directory that holds it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    hf = json.loads(path.read_text(encoding="utf-8"))
    _assert_supported(hf)
    return HFModelConfig(
        architecture=hf["architectures"][0],
        vocab_size=hf["vocab_size"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"],
        head_dim=hf["head_dim"],
        rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
        rope_theta=_rope_theta(hf),
        max_position_embeddings=hf["max_position_embeddings"],
        # transformers 5's Qwen3Config defaults to False when the key is absent.
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        dtype=_DTYPES[_dtype_name(hf)],
    )


def _dtype_name(hf: dict[str, Any]) -> str:
    return hf.get("dtype") or hf.get("torch_dtype") or ""


def _rope_theta(hf: dict[str, Any]) -> float:
    params = hf.get("rope_parameters") or {}
    return params["rope_theta"] if "rope_theta" in params else hf["rope_theta"]


def _assert_supported(hf: dict[str, Any]) -> None:
    def refuse(msg: str):
        raise ValueError(f"unsupported checkpoint config: {msg}")

    archs = hf.get("architectures") or []
    if len(archs) != 1 or archs[0] not in SUPPORTED_ARCHITECTURES:
        refuse(f"architectures must be one of {SUPPORTED_ARCHITECTURES}, got {archs}")
    if hf.get("model_type") != "qwen3":
        refuse(f"model_type must be 'qwen3', got {hf.get('model_type')!r}")
    if _dtype_name(hf) not in _DTYPES:
        refuse(f"dtype must be one of {sorted(_DTYPES)}, got {_dtype_name(hf)!r}")
    if hf.get("hidden_act", "silu") != "silu":
        refuse(f"hidden_act must be 'silu', got {hf['hidden_act']!r}")
    if hf.get("attention_bias", False):
        refuse("attention_bias is not supported")
    if hf.get("use_sliding_window", False) or any(t != "full_attention" for t in hf.get("layer_types") or []):
        refuse("sliding-window attention is not supported")
    if hf.get("rope_scaling") is not None:
        refuse("rope_scaling is not supported")
    rope_params = hf.get("rope_parameters") or {}
    if rope_params.get("rope_type", "default") != "default":
        refuse(f"rope_type must be 'default', got {rope_params['rope_type']!r}")
    if "rope_theta" not in hf and "rope_theta" not in rope_params:
        refuse("no rope_theta")
    if "rope_theta" in hf and "rope_theta" in rope_params and hf["rope_theta"] != rope_params["rope_theta"]:
        refuse(f"rope_theta {hf['rope_theta']} and rope_parameters.rope_theta {rope_params['rope_theta']} differ")
    for key in ("head_dim", "num_key_value_heads"):
        if hf.get(key) is None:
            refuse(f"no {key}")
    num_heads, num_kv_heads = hf["num_attention_heads"], hf["num_key_value_heads"]
    if num_heads % num_kv_heads:
        refuse(f"num_attention_heads ({num_heads}) must be a multiple of num_key_value_heads ({num_kv_heads})")
    if hf["head_dim"] % 2:
        refuse(f"head_dim must be even for RoPE, got {hf['head_dim']}")
