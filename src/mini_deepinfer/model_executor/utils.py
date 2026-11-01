"""Small helpers for the model executor. Mirrors deepinfer/model_executor/utils.py."""

from __future__ import annotations

import torch
from torch import nn


def reconstruct_param(param: torch.Tensor) -> nn.Parameter:
    """A new inference-only Parameter over the same storage.

    DeepInfer builds runtime layers from tensors that already exist (loaded,
    or owned by another module). Wrapping the data in a fresh Parameter keeps
    the two modules' parameter objects apart while sharing the memory.
    """
    data = param.data if isinstance(param, nn.Parameter) else param
    return nn.Parameter(data, requires_grad=False)


def init_torch() -> None:
    """Process-wide numerics settings, as DeepInfer's ``Worker._init_torch``
    (deepinfer/worker/base.py) applies them before any model runs: no
    reduced-precision reductions inside cuBLAS FP16/BF16 matrix multiplies.

    The flags are global, so they also change Hugging Face's results when HF
    runs in the same process. DeepInfer's allocator and thread settings are
    left out.
    """
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
