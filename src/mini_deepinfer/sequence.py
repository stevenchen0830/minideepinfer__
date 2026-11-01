"""Per-request state (S1.1).

DeepInfer keeps this state in C++ (`csrc/engine/sequence.hpp`, bound to Python
in `csrc/engine/pybind_sequence.cpp`; `deepinfer/sequence.py` only re-exports
it). This module is a pure-Python version of the core of that file, with the
same names:

  SequenceStatus    the 10-value status enum, in the same order
  SequenceData      prompt and output tokens in one list
  KvCacheState      how many tokens already have KV in the cache, the block table
  Sequence          one request: ids, tokens, sampling params, status, cache state

Left out until a later step needs them: speculative-decoding state
(`MtpState`, including golden tokens), in-flight token counts for overlapped
scheduling (ZOS, S4.3), structured-output state (S6.3), multimodal input, the
compressed token stream, sliding-window and state-slot fields, prefix-cache hit
counts, forking, chunked-prefill checkpoints, and the constructor arguments
`compressed_prompt_token_ids`, `multi_modal_data`, `user_id` and
`golden_token_ids`.

Who changes what: the Sequence only stores state. The scheduler (S2.5) moves
the status and fills and clears `cache.block_table`, using block ids from the
block manager (S2.1). After each model step, the scheduler's process_outputs
appends the sampled tokens, updates the KV count and runs the stop checks; the
engine loop (S2.6) calls it. (DeepInfer: `Scheduler::process_outputs` and
`_check_stop` in `csrc/engine/scheduler.hpp`, called from
`LLMEngine._process_model_outputs`.)

Deliberate differences from DeepInfer:
  - Prompts may be any iterable of Python or NumPy ints. DeepInfer's binding
    takes only a 1-D C-contiguous int32 NumPy array and raises TypeError for a
    list, an int64 array or a strided view.
  - block_size must be a positive integer. DeepInfer does not check it.
  - SequenceStatus is an IntEnum, so `status >= FINISHED_STOPPED` works in
    Python. DeepInfer does that comparison only in C++; its Python enum has no
    ordering and never equals a plain int (use int(status) there).
  - The constructor is keyword-only. DeepInfer's binding takes ordinary named
    arguments, but the engine always passes them by keyword; positions would
    not line up here because some arguments are left out.
"""

from __future__ import annotations

import enum
import operator
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_deepinfer.sampling_params import SamplingParams


class SequenceStatus(enum.IntEnum):
    """Lifecycle of a sequence. The order matters: every value from
    FINISHED_STOPPED on is terminal, which is what is_finished() checks."""

    WAITING = 0
    RUNNING = 1
    SWAPPED = 2  # In DeepInfer's enum but never used: there is no swap path.
    FINISHED_STOPPED = 3  # Sampled the EOS token.
    FINISHED_LENGTH_CAPPED = 4  # Hit max_tokens or the model's max length.
    FINISHED_ABORTED = 5  # Aborted by the user.
    FINISHED_IGNORED = 6  # Prompt too long to ever run.
    FINISHED_PREEMPTED = 7  # Dropped for lack of memory (not recompute).
    FINISHED_SO_FAILED = 8  # Structured-output failure (S6.3).
    FINISHED_VE_FAILED = 9  # Vision-encoder failure (not used here).


_FINISHED_REASONS = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH_CAPPED: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
    SequenceStatus.FINISHED_IGNORED: "prompt_length_exceeded",
    SequenceStatus.FINISHED_PREEMPTED: "insufficient_system_resource",
    SequenceStatus.FINISHED_SO_FAILED: "xgrammar runtime errors or timeout",
    SequenceStatus.FINISHED_VE_FAILED: "ve_error",
}


def is_finished(status: SequenceStatus) -> bool:
    return status >= SequenceStatus.FINISHED_STOPPED


def get_finished_reason(status: SequenceStatus) -> str:
    """The finish reason reported to the user; "" while not finished."""
    return _FINISHED_REASONS.get(status, "")


class SequenceData:
    """Token ids of one sequence: the prompt followed by the generated tokens.

    One list holds both. prompt_len never changes after construction. Add
    tokens only through append_token_id.
    """

    def __init__(self, prompt_token_ids: Iterable[int]) -> None:
        # operator.index accepts Python and NumPy integers and rejects floats
        # and strings with a TypeError. No int32 range check here: DeepInfer
        # checks token ids against the vocabulary in add_request (S2.6).
        # Copying into a new list means later changes to the caller's list
        # cannot reach this sequence.
        self.token_ids: list[int] = [operator.index(t) for t in prompt_token_ids]
        self.prompt_len: int = len(self.token_ids)

    def append_token_id(self, token_id: int) -> None:
        self.token_ids.append(operator.index(token_id))

    def get_len(self) -> int:
        return len(self.token_ids)

    def get_prompt_len(self) -> int:
        return self.prompt_len

    def get_output_len(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def get_token_ids(self) -> list[int]:
        # A new list, as DeepInfer's binding returns: changing it does not
        # change the sequence. For the newest token use get_last_token_id().
        return list(self.token_ids)

    def get_last_token_id(self) -> int:
        # Raises IndexError on an empty sequence (undefined behaviour in C++).
        return self.token_ids[-1]

    def __repr__(self) -> str:
        return f"SequenceData(prompt_len={self.prompt_len}, len={self.get_len()})"


class KvCacheState:
    """KV-cache bookkeeping for one sequence.

    num_tokens counts the tokens whose K and V are already in the cache.
    DeepInfer's process_outputs moves it like this after every model step:

      advance_optimistically(q)   on every step: num_tokens += q, where q is
                                  the number of query tokens the step
                                  processed. The old value is remembered.
      commit_processed_tokens(n)  on decode steps only: num_tokens = old + n,
                                  where n is the number of tokens appended to
                                  the sequence in this step.

    Speculative decoding (S6.1) is why the second move exists.
    """

    def __init__(self, block_size: int) -> None:
        # Integer check: DeepInfer stores block_size as a C++ int, so a float
        # raises TypeError there too. The positivity check is extra; DeepInfer
        # has none, and get_num_blocks would divide by zero.
        block_size = operator.index(block_size)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.block_size = block_size
        self.num_tokens = 0
        self.previous_num_tokens = 0
        # Physical block ids. The scheduler (S2.5) fills it with ids from the
        # block manager (S2.1) and clears it when it frees the sequence.
        self.block_table: list[int] = []

    def set_num_tokens(self, n: int) -> None:
        """Set the count from outside (DeepInfer: an injected prefix-cache
        hit); the remembered old value moves with it."""
        self.num_tokens = n
        self.previous_num_tokens = n

    def advance_optimistically(self, n: int) -> None:
        self.previous_num_tokens = self.num_tokens
        self.num_tokens += n

    def commit_processed_tokens(self, n: int) -> None:
        self.num_tokens = self.previous_num_tokens + n

    def reset_for_recompute(self) -> None:
        """Forget all cached KV. The block table is left alone: the scheduler
        frees the blocks and clears it separately."""
        self.num_tokens = 0
        self.previous_num_tokens = 0


# SamplingParams fields that DeepInfer's Sequence constructor copies into C++
# ints (SamplingScalars::capture, StoppingCriteria::capture).
_INT_SAMPLING_FIELDS = ("top_k", "max_tokens", "first_n_token_no_eos", "eos_id")


class Sequence:
    """One request moving through the engine.

    The status starts at WAITING. Only the scheduler changes it afterwards,
    including through the stop checks in its process_outputs.

    DeepInfer copies the scalar sampling fields into C++ when the Sequence is
    built, so later changes to the Python SamplingParams never reach the
    engine. Here the object is shared, so the engine must finish setting it
    up (eos_id) before building the Sequence and leave it alone afterwards.
    """

    def __init__(
        self,
        *,
        seq_id: int,
        prompt_token_ids: Iterable[int],
        block_size: int,
        request_id: str,
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        # DeepInfer's copy into C++ uses pybind's int cast, which rejects any
        # float, even 2.0. We only check; nothing is copied.
        for name in _INT_SAMPLING_FIELDS:
            try:
                operator.index(getattr(sampling_params, name))
            except TypeError:
                raise TypeError(
                    f"sampling_params.{name} must be an int, "
                    f"got {getattr(sampling_params, name)!r}."
                ) from None
        self.seq_id = seq_id
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.data = SequenceData(prompt_token_ids)
        self.cache = KvCacheState(block_size)
        self.status: SequenceStatus = SequenceStatus.WAITING
        self._sampling_params = sampling_params

    @property
    def sampling_params(self) -> SamplingParams:
        # Read-only, as in DeepInfer.
        return self._sampling_params

    # Python-side names that DeepInfer's bindings give to cache fields.
    # block_size is fixed at construction here (DeepInfer's binding also
    # allows writes, but nothing uses them).
    @property
    def block_size(self) -> int:
        return self.cache.block_size

    @property
    def num_cached_tokens(self) -> int:
        """Number of tokens with KV cache ready."""
        return self.cache.num_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, n: int) -> None:
        self.cache.set_num_tokens(n)

    @property
    def old_cached_tokens(self) -> int:
        return self.cache.previous_num_tokens

    # Token accessors, delegated to SequenceData.
    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> list[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def append_token_id(self, token_id: int) -> None:
        self.data.append_token_id(token_id)

    def get_num_blocks(self) -> int:
        """Blocks needed to hold KV for every token, rounded up.

        DeepInfer also counts in-flight tokens (ZOS) and draft tokens (MTP)
        here; this version adds them in S4.3 and S6.1.
        """
        return -(-self.get_len() // self.cache.block_size)

    def is_finished(self) -> bool:
        return is_finished(self.status)

    def reset_for_recompute(self) -> None:
        """Prepare a preempted sequence to be prefilled again from scratch.

        Only the KV counters go back to zero. The tokens, the status and the
        block table are left alone.
        """
        self.cache.reset_for_recompute()

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, request_id={self.request_id!r}, "
            f"status={self.status.name}, prompt_len={self.get_prompt_len()}, "
            f"output_len={self.get_output_len()}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"blocks={len(self.cache.block_table)})"
        )
