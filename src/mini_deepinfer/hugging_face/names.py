"""Map Hugging Face checkpoint names to mini-DeepInfer parameter names (S1.2).

Mirrors deepinfer/hugging_face/names.py. The routing is a pure function plus
small tables: the loader, not the modules, decides where each tensor goes.
There are no per-parameter ``weight_loader`` hooks; that is vLLM's design, not
DeepInfer's.

Our parameter names keep the Hugging Face names except for the two merged
projections. DeepInfer's own parameter names differ more (for example its
``attn_norm.weight`` becomes ``norm1.weight``).

A merged parameter is the dim-0 concatenation of several checkpoint tensors.
DeepInfer's loader concatenates them in sorted HF-name order (``w1`` before
``w3``); that is its only load-time merge (see model_loader.py for where its
attention projections are merged). For Qwen3 sorted order would give k, q, v,
so the order is explicit here.
"""

from __future__ import annotations

import re

from mini_deepinfer.hugging_face.config import HFModelConfig

# Merged parameter suffix -> checkpoint suffixes, in concatenation order.
MERGED_SUFFIXES: dict[str, tuple[str, ...]] = {
    "self_attn.qkv_proj.weight": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ),
    "mlp.gate_up_proj.weight": (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ),
}
_PART_OF = {part: (merged, i) for merged, parts in MERGED_SUFFIXES.items() for i, part in enumerate(parts)}
_LAYER_NAME = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def hf_to_model(hf_name: str) -> str:
    """Return the model parameter name that ``hf_name`` feeds.

    Unknown layer suffixes pass through unchanged, as in DeepInfer: whether
    such a parameter exists is decided later, when the names meet the real
    model. Unlike DeepInfer, which returns None for an unknown top-level name,
    every other name passes through too (our parameter names are the HF
    names), so ``load_hf_checkpoint`` can list it as unused.
    """
    m = _LAYER_NAME.match(hf_name)
    if m is None:
        return hf_name
    layer, suffix = m.groups()
    if suffix in _PART_OF:
        suffix = _PART_OF[suffix][0]
    return f"model.layers.{layer}.{suffix}"


def merged_parts(model_name: str) -> tuple[str, ...] | None:
    """The checkpoint names a merged parameter needs, or None if not merged."""
    m = _LAYER_NAME.match(model_name)
    if m is None or m.group(2) not in MERGED_SUFFIXES:
        return None
    layer, suffix = m.groups()
    return tuple(f"model.layers.{layer}.{part}" for part in MERGED_SUFFIXES[suffix])


def concat_index(hf_name: str) -> int:
    """Position of ``hf_name`` inside its merged parameter (0 if not merged)."""
    m = _LAYER_NAME.match(hf_name)
    if m is None or m.group(2) not in _PART_OF:
        return 0
    return _PART_OF[m.group(2)][1]


def tied_aliases(cfg: HFModelConfig) -> dict[str, str]:
    """Tied parameter -> the parameter whose storage it shares.

    Taken from ``tie_word_embeddings``. DeepInfer parses that key but never
    uses it; its only ties are the MTP layers' copies of the embedding and
    head, found from the model's parameter names.
    """
    if cfg.tie_word_embeddings:
        return {"lm_head.weight": "model.embed_tokens.weight"}
    return {}
