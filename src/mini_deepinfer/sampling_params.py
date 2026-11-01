"""Per-request sampling settings (S1.1).

Mirrors DeepInfer's `deepinfer/sampling_params.py` (itself derived from vLLM):
same field names, defaults, validation messages, and the same statement order
in `__init__`, because the order changes behaviour.

Kept for now: temperature, top_p, top_k, ignore_eos, max_tokens,
first_n_token_no_eos, eos_id, golden_token_ids, seq_seed.
Left out until a later step needs them: logprobs (DeepInfer accepts and drops
it), prompt_logprobs and top_logprobs (logprob outputs), so_args (structured
output, S6.3).

Divergences from DeepInfer:
  - Arguments are keyword-only. DeepInfer allows positional arguments, but the
    positions would not line up here because some parameters are left out.
  - Integer limits come from plain Python constants instead of `torch.iinfo`,
    so this module does not import torch.
  - A NaN temperature is rejected; DeepInfer's `< 0.0` check lets it through.
  - `_make_seed` raises TypeError with a message; DeepInfer's has none.
Like DeepInfer, a float top_k or max_tokens (even 2.0) passes here and is
rejected only when the Sequence is built (see sequence.py).
"""

from __future__ import annotations

import random

# Temperatures below this are treated as greedy (see __init__).
SAMPLING_EPS = 1e-5

INT32_MAX = 2**31 - 1
UINT64_MASK = 0xFFFF_FFFF_FFFF_FFFF


class SamplingParams:
    """Sampling settings for one request.

    Args:
        temperature: Randomness of sampling. Values below SAMPLING_EPS
            (including 0) mean greedy: the temperature is raised to
            SAMPLING_EPS and top-p / top-k are switched off. There is no
            separate argmax path: the logits are multiplied by
            1 / SAMPLING_EPS = 1e5, so the softmax is almost always one-hot.
        top_p: Nucleus threshold in (0, 1]. 1.0 keeps every token.
        top_k: Keep the k most likely tokens. -1 keeps every token.
        ignore_eos: Keep generating after the EOS token.
        max_tokens: Maximum number of generated tokens. 0 becomes 1.
        first_n_token_no_eos: When 1, EOS is masked out for the first
            generated token (DeepInfer acts only on the value 1).
        eos_id: The token masked by first_n_token_no_eos. The engine
            overwrites it with the tokenizer's EOS id before building the
            Sequence; the default 0 is a placeholder.
        golden_token_ids: Tokens forced as outputs, for teacher-forced
            comparisons in tests. Not validated.
        seq_seed: Per-request seed for reproducible sampling, as an unsigned
            64-bit integer. None draws a random one.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        first_n_token_no_eos: int = 0,
        eos_id: int = 0,
        golden_token_ids: list[int] | None = None,
        seq_seed: int | None = None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        if self.max_tokens == 0:
            self.max_tokens = 1
        self.seq_seed = self._make_seed(seq_seed)
        self._verify_args()
        if self.temperature < SAMPLING_EPS:
            self.temperature = SAMPLING_EPS
            self.top_p = 1.0
            self.top_k = -1
        # Like DeepInfer, these are stored without validation.
        self.first_n_token_no_eos = first_n_token_no_eos
        self.eos_id = eos_id
        self.golden_token_ids = golden_token_ids

    @staticmethod
    def _make_seed(seed: int | None) -> int:
        """Return a seed in [0, 2**64).

        None draws a random seed. Any int is masked to 64 bits, so -1 becomes
        2**64 - 1, and a bool counts as an int (both as in DeepInfer).
        """
        if seed is None:
            return random.getrandbits(64)
        if isinstance(seed, int):
            return seed & UINT64_MASK
        raise TypeError(f"seq_seed must be an int or None, got {type(seed).__name__}.")

    def _verify_args(self) -> None:
        # Written as `not >=` so NaN fails too: NaN compares False with
        # everything. DeepInfer writes `< 0.0` and lets NaN through.
        if not self.temperature >= 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0 or self.top_k > INT32_MAX:
            raise ValueError(
                f"top_k must be -1 (disable), or between 1 and int32 max, got {self.top_k}."
            )
        if self.max_tokens < 1 or self.max_tokens > INT32_MAX:
            raise ValueError(
                f"max_tokens must be >= 1 and in range of int32, got {self.max_tokens}."
            )
        # Unreachable for ints after _make_seed; kept to match DeepInfer.
        if self.seq_seed < 0 or self.seq_seed >= 2**64:
            raise ValueError(f"seq_seed must be in range of uint64_t, got {self.seq_seed}.")

    def __repr__(self) -> str:
        return (
            f"SamplingParams(temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"seq_seed={self.seq_seed})"
        )
