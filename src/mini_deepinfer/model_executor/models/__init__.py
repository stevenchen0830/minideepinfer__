"""Model registry. Mirrors deepinfer/model_executor/models/__init__.py.

The ``architectures`` entry of the checkpoint's config.json selects the class;
the module is imported only when it is needed.
"""

from __future__ import annotations

import importlib

from torch import nn

# architecture name -> (module in this package, class name)
_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
}


class ModelRegistry:
    @staticmethod
    def load_model_cls(model_arch: str) -> type[nn.Module]:
        if model_arch not in _MODELS:
            raise ValueError(f"model architecture {model_arch!r} is not supported; supported: {sorted(_MODELS)}")
        module_name, class_name = _MODELS[model_arch]
        module = importlib.import_module(f"mini_deepinfer.model_executor.models.{module_name}")
        return getattr(module, class_name)

    @staticmethod
    def get_supported_archs() -> list[str]:
        return list(_MODELS)
