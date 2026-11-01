"""Linear layer and the BF16 GEMM it runs on (S1.2).

Mirrors deepinfer/model_executor/layers/linear.py, which has a single class,
``ReplicatedLinear``: no column- or row-parallel layers, no ``weight_loader``
hooks. A layer receives its weight tensor; merged projections (QKV,
gate/up) are just a weight made by concatenating the parts along dim 0.

``bf16_gemm_nt`` stands in for ``deep_gemm.bf16_gemm_nt``: ``d = a @ b.T``
(plus ``c``), accumulated in float32 and stored in ``d``'s dtype. With a
float32 ``d`` and bfloat16 inputs there is no bfloat16 rounding of the result;
the sampler uses this for the logits.

DeepGEMM keeps one float32 accumulator per output. cuBLAS, which runs
``F.linear`` and ``torch.mm`` here, does the same only after
``model_executor.utils.init_torch()`` has applied DeepInfer's process
settings; PyTorch's defaults allow it more roundings in bfloat16.

Differences: ``weight`` may be omitted (an uninitialized tensor is made, for
the loader to fill); ``torch.nn.functional.linear`` replaces DeepGEMM; no FP8
path; no padded output stride; ``forward`` has no ``out_acc`` argument
(DeepInfer can add the product into an existing output through DeepGEMM's
``c``, which its MoE shared expert uses; it returns in phase 5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_deepinfer.model_executor.utils import reconstruct_param


def bf16_gemm_nt(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, c: torch.Tensor | None = None) -> None:
    """Write ``a @ b.T (+ c)`` into ``d``. ``a``: [m, k], ``b``: [n, k], ``d``: [m, n]."""
    if d.dtype == a.dtype:
        out = torch.mm(a, b.t())
    elif a.is_cuda:
        # cuBLAS reads bf16, accumulates in fp32 and writes fp32 directly.
        out = torch.mm(a, b.t(), out_dtype=d.dtype)
    else:
        # The CPU has no mixed-dtype mm; upcasting first gives the same math.
        out = torch.mm(a.to(d.dtype), b.to(d.dtype).t())
    if c is not None:
        out = out + c
    d.copy_(out)


class ReplicatedLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_fp32_output: bool = False,
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if weight is None:
            weight = torch.empty(output_size, input_size)
        if tuple(weight.shape) != (output_size, input_size):
            raise ValueError(f"weight shape {tuple(weight.shape)} != ({output_size}, {input_size})")
        if bias is not None and tuple(bias.shape) != (output_size,):
            raise ValueError(f"bias shape {tuple(bias.shape)} != ({output_size},)")
        self.input_size = input_size
        self.output_size = output_size
        self.use_fp32_output = use_fp32_output
        self.weight = reconstruct_param(weight)
        self.bias = reconstruct_param(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp32_output:
            out = F.linear(x, self.weight)
        else:
            out = torch.empty(*x.shape[:-1], self.output_size, dtype=torch.float32, device=x.device)
            bf16_gemm_nt(x.reshape(-1, self.input_size), self.weight, out.view(-1, self.output_size))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in={self.input_size}, out={self.output_size}, bias={self.bias is not None}"
