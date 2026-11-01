"""RMSNorm with an optional residual add (S1.2).

Mirrors deepinfer/model_executor/layers/layernorm.py, whose ``RMSNorm`` is a
thin wrapper around a TileLang kernel (``tilelang_ops/norm.py``) that does the
add and the norm in one kernel. This file is a torch stand-in with the same
contract; here the add and the norm are separate torch operations:

- ``norm(x)`` returns the normalized tensor;
- ``norm(x, residual)`` first adds ``residual`` to ``x`` (in the input dtype),
  then returns ``(normalized sum, sum)``. Both are new tensors; the inputs are
  not changed (vLLM's ``fused_add_rms_norm`` works in place instead).

Numerics follow the CUDA kernel: square, mean, rsqrt and the multiply by the
weight all happen in float32, and the result is cast to the input dtype once.
Hugging Face's ``Qwen3RMSNorm`` casts back to bfloat16 *before* multiplying by
the weight, so in bfloat16 the two can differ by one rounding step.

Left out: the FP8 output (``use_fp8``, ``out_scale``), ``return_bf16`` and the
preallocated ``out`` buffer.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_deepinfer.model_executor.utils import reconstruct_param


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight is None:
            weight = torch.ones(hidden_size)
        if tuple(weight.shape) != (hidden_size,):
            raise ValueError(f"RMSNorm weight shape {tuple(weight.shape)} != ({hidden_size},)")
        self.weight = reconstruct_param(weight)
        self.variance_epsilon = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.variance_epsilon)
        x = x + residual
        return rms_norm(x, self.weight, self.variance_epsilon), x

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
