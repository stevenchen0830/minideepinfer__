"""Name every tensor the loader must fill (S1.2).

Mirrors ``model_params`` in deepinfer/hugging_face/parameters.py.
"""

from __future__ import annotations

import torch
from torch import nn


def model_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Dotted name -> parameter, plus persistent buffers.

    Walks each module's own parameters, so a parameter registered under two
    names (a tied weight) appears under both. ``model.named_parameters()``
    would drop the second name. Non-persistent buffers are left out.
    """
    params: dict[str, torch.Tensor] = {}
    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        for name, p in module.named_parameters(recurse=False):
            params[prefix + name] = p
        for name, b in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                params[prefix + name] = b
    return params
