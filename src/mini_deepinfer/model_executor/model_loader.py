"""Build a model and load a Hugging Face checkpoint into it (S1.2).

Mirrors deepinfer/model_executor/model_loader.py (``get_model``,
``load_hf_checkpoint``) and the order of ``DeepseekForCausalLM.load_model``,
the classmethod that DeepInfer's ``get_model(model_config, skip_load=False)``
calls under a default-dtype and CUDA device context:

1. read config.json and pick the model class from the registry;
2. check the checkpoint files and build the load plan;
3. build the model on the ``meta`` device (parameters with a shape and a
   dtype but no memory);
4. add the tied aliases to the plan;
5. materialize: give every parameter real, uninitialized memory on the target
   device; a tied parameter instead shares its source's storage;
6. check that every parameter is covered by the plan, check shapes and
   dtypes from the file headers, then read and copy the tensors.

DeepInfer builds an HF-shaped parameter container, loads the checkpoint into
it, and then constructs separate runtime modules from it (``from_model_cfg``).
Its loader concatenates only the MoE ``w1``/``w3`` pairs (our gate/up analog).
Its fused attention projections (for example ``q_a_kv_swa_proj`` =
``Wq.A`` then ``Wkv.A``, our QKV analog) are concatenated later, in an explicit
order, by ``merge_linear_weights`` while the runtime modules are built. Here
the runtime modules are built directly, so the loader concatenates both of our
merged weights.

Checkpoint tensors that feed no parameter are not loaded; DeepInfer drops
them silently, while ``load_hf_checkpoint`` here lists them in its report.
After loading, the CUDA caching allocator's leftover staging memory is
released, as DeepInfer's model constructor does (``torch.cuda.empty_cache``).

Differences in the entry points: ``get_model``/``load_model`` take a
checkpoint directory instead of DeepInfer's engine ``ModelConfig``, accept a
``device`` and an optional ``dtype``, and return a ``LoadReport``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from mini_deepinfer.hugging_face.config import load_hf_model_config
from mini_deepinfer.hugging_face.loading import cover_tied_parameters, prepare_hf_coverage, validate_hf_shapes
from mini_deepinfer.hugging_face.names import tied_aliases
from mini_deepinfer.hugging_face.parameters import model_params
from mini_deepinfer.hugging_face.weights import HFCoverage, load_covered_params
from mini_deepinfer.model_executor.layers.rotary_embedding import rope_memo_scope
from mini_deepinfer.model_executor.models import ModelRegistry


@dataclass
class LoadReport:
    params: int  # parameters filled from the checkpoint
    tensors: int  # checkpoint tensors read
    bytes: int
    aliases: dict[str, str]  # tied parameter -> source
    ignored: list[str]  # checkpoint copies of tied parameters, left out on purpose
    unused: list[str]  # checkpoint tensors that match no parameter


@contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _materialize(model: nn.Module, device: torch.device, aliases: dict[str, str]) -> None:
    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        for name, p in list(module.named_parameters(recurse=False)):
            if prefix + name in aliases:
                continue
            new = torch.empty(p.shape, dtype=p.dtype, device=device)
            setattr(module, name, nn.Parameter(new, requires_grad=False))
    params = model_params(model)
    for target, source in aliases.items():
        mod_name, _, name = target.rpartition(".")
        module = model.get_submodule(mod_name)
        setattr(module, name, nn.Parameter(params[source].data, requires_grad=False))


def load_hf_checkpoint(model: nn.Module, *, hf_dir: str | Path, hf_coverage: HFCoverage, cast: bool = False) -> LoadReport:
    params = model_params(model)
    missing = sorted(set(params) - hf_coverage.model_names())
    if missing:
        raise RuntimeError(f"{len(missing)} model parameters are not in the checkpoint, e.g. {missing[:5]}")
    validate_hf_shapes(params, hf_coverage, hf_dir=hf_dir, cast=cast)
    stats = load_covered_params(model, hf_coverage, hf_dir=hf_dir, cast=cast)
    unused = sorted(hf for name, parts in hf_coverage.tensors.items() if name not in params for hf in parts)
    return LoadReport(
        params=stats["params"],
        tensors=stats["tensors"],
        bytes=stats["bytes"],
        aliases=dict(hf_coverage.aliases),
        ignored=list(hf_coverage.ignored),
        unused=unused,
    )


def load_model(
    model_path: str | Path,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, LoadReport]:
    """Build the model for the checkpoint in ``model_path`` and load it.

    With ``dtype`` given, the model is built in that dtype and every weight is
    cast to it after reading. Without it, the config's dtype is used and any
    tensor in another dtype is an error. DeepInfer has no such option: it
    always runs in bfloat16 and never casts.
    """
    hf_dir = Path(model_path)
    device = torch.device(device)
    cfg = load_hf_model_config(hf_dir / "config.json")
    cast = dtype is not None
    if dtype is not None:
        cfg = replace(cfg, dtype=dtype)
    model_cls = ModelRegistry.load_model_cls(cfg.architecture)

    coverage = prepare_hf_coverage(hf_dir=hf_dir)
    with rope_memo_scope(), _set_default_torch_dtype(cfg.dtype), torch.device("meta"):
        model = model_cls(cfg, device=device)
    coverage = cover_tied_parameters(coverage, tied_aliases(cfg))
    _materialize(model, device, coverage.aliases)
    report = load_hf_checkpoint(model, hf_dir=hf_dir, hf_coverage=coverage, cast=cast)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, t in [*model.named_parameters(), *model.named_buffers()]:
        on_device = t.device.type == device.type and device.index in (None, t.device.index)
        if not on_device:
            raise RuntimeError(f"{name} is on {t.device}, expected {device}")
    return model.eval().requires_grad_(False), report


def get_model(model_path: str | Path, *, device: torch.device | str = "cuda", dtype: torch.dtype | None = None) -> nn.Module:
    return load_model(model_path, device=device, dtype=dtype)[0]
