"""SwiGLU activation (S1.2).

DeepInfer has no activation layer file: its MLPs call
``tile_kernels.quant.swiglu_forward_and_per_token_cast``, from an external
package that is not in the repo. This is a torch stand-in that follows vLLM's
``SiluAndMul`` layout: the first half of the last dimension is the gate, the
second half is the up projection, as produced by a merged ``gate_up_proj``
whose weight is ``cat([gate_proj, up_proj])``.

It computes in float32 and casts once. Hugging Face's ``Qwen3MLP`` rounds to
bfloat16 twice: after ``silu(gate)`` and after the multiply.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` with ``d = x.shape[-1] // 2``."""
    d = x.shape[-1] // 2
    gate, up = x[..., :d].float(), x[..., d:].float()
    return (F.silu(gate) * up).to(x.dtype)
