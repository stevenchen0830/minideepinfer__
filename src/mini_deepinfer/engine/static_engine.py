"""A static-batch generation loop (S1.3; block manager since S2.1).

A small stand-in for DeepInfer's ``LLMEngine`` (engine/llm_engine.py) and its
C++ ``Scheduler`` (csrc/engine/scheduler.hpp), enough to run requests end to
end before the real scheduler arrives in S2.5. "Static": all requests start
together, are prefilled in one step, then decoded together until each one
stops; no request joins later and nothing is preempted, so when blocks run
out the step fails instead of waiting.

What it keeps from DeepInfer:
- ``add_requests`` sets every request's ``SamplingParams.eos_id`` to the
  engine's ``eos_token_id`` and refuses an empty prompt (both
  ``LLMEngine.add_request``). Every request starts WAITING;
- a step is all prefill or all decode;
- prefill, as in ``_schedule_prefill``: a request with an empty block table
  gets blocks for its whole prompt (``can_allocate_nolock``, then
  ``allocate_seq``), even if the prompt is prefilled in several chunks; it
  becomes RUNNING when the step that completes its prompt is scheduled, and
  stays WAITING between chunks;
- decode: ``_append_slot`` adds at most one block, or trims blocks beyond
  ``get_num_blocks()``;
- ``process_outputs`` updates a sequence after a step the way
  ``Scheduler::process_outputs`` does: ``advance_optimistically(seqlen_q)``
  on every step; the sampled token is appended only once the prompt is fully
  cached or on a decode step; ``commit_processed_tokens(1)`` only on decode;
- ``check_stop`` in DeepInfer's order: EOS (unless ``ignore_eos``) ->
  FINISHED_STOPPED; ``get_len() > max_model_len`` -> FINISHED_LENGTH_CAPPED;
  ``get_output_len() == max_tokens`` -> FINISHED_LENGTH_CAPPED;
- ``get_gpu_cache_usage`` as in ``Scheduler::get_gpu_cache_usage``.

Differences:
- A prompt that can never run (longer than ``max_model_len``, or
  ``can_allocate_nolock`` answers NEVER) is marked FINISHED_IGNORED by
  ``add_requests``. DeepInfer decides this in ``_schedule_prefill``, when the
  request reaches the head of the waiting queue.
- Blocks of a finished sequence are freed at the end of the step in which it
  finished. DeepInfer frees them later (``LLMEngine.finalize_seq`` ->
  ``Scheduler::free_seq``, after the next schedule), so they still count for
  that schedule.
- A decode step asks only for the blocks it needs, so it can run with no free
  block; DeepInfer's ``_schedule_decode`` wants one free block before every
  ``_append_slot``.
- Only the one EOS id stops a sequence; Qwen3's second end token
  (``<|endoftext|>``, 151643) is not checked, as DeepInfer has one EOS id.
- Failures: a prefill step that finds too few blocks takes nothing and leaves
  its sequences WAITING. When a decode step runs out of blocks, or the model
  call of any step fails, the step's sequences are aborted and their blocks
  freed, so the engine stays usable.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceList

from mini_deepinfer.engine.block_manager import AllocStatus, BlockManager, BlockManagerMutexAcquire
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import Sequence, SequenceStatus
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner, SamplerOutput


def check_stop(seq: Sequence, max_model_len: int) -> None:
    sp = seq.sampling_params
    if not sp.ignore_eos and seq.get_last_token_id() == sp.eos_id:
        seq.status = SequenceStatus.FINISHED_STOPPED
    elif seq.get_len() > max_model_len:
        seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
    elif seq.get_output_len() == sp.max_tokens:
        seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED


def process_outputs(
    seqs: SequenceList[Sequence],
    token_ids: SequenceList[int],
    is_prompt: bool,
    seqlens_q: SequenceList[int],
    max_model_len: int,
) -> None:
    for seq, token, q in zip(seqs, token_ids, seqlens_q, strict=True):
        seq.cache.advance_optimistically(q)
        if is_prompt:
            if seq.cache.num_tokens < seq.get_len():
                continue  # the prompt is not fully cached yet: the sampled token is dropped
            seq.append_token_id(token)
        else:
            seq.append_token_id(token)
            seq.cache.commit_processed_tokens(1)
        check_stop(seq, max_model_len)


class StaticEngine:
    def __init__(self, runner: ModelRunner, cache_engine: CacheEngine, *, eos_token_id: int) -> None:
        if cache_engine.block_size != runner.block_size:
            raise ValueError("runner and cache engine disagree on block_size")
        self.runner = runner
        self.cache_engine = cache_engine
        self.eos_token_id = eos_token_id
        self.max_model_len = runner.max_model_len
        # The block manager counts every block of the cache, including the
        # reserved block 0.
        self.block_manager = BlockManager(cache_engine.num_blocks)
        self._next_id = 0

    def add_requests(
        self,
        prompts: SequenceList[SequenceList[int]],
        sampling_params: SamplingParams | SequenceList[SamplingParams],
    ) -> list[Sequence]:
        """One Sequence per prompt; ``sampling_params`` is shared or one per prompt."""
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("one SamplingParams per prompt")
        if any(len(p) == 0 for p in prompts):
            raise ValueError("prompt must not be empty")
        seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            sp.eos_id = self.eos_token_id  # before the Sequence is built, as DeepInfer does
            seq = Sequence(
                seq_id=self._next_id,
                prompt_token_ids=prompt,
                block_size=self.cache_engine.block_size,
                request_id=str(self._next_id),
                sampling_params=sp,
                arrival_time=0.0,
            )
            self._next_id += 1
            with BlockManagerMutexAcquire(self.block_manager):
                never = self.block_manager.can_allocate_nolock(seq.get_num_blocks()) == AllocStatus.NEVER
            if seq.get_prompt_len() > self.max_model_len or never:
                seq.status = SequenceStatus.FINISHED_IGNORED
            seqs.append(seq)
        return seqs

    def _append_slot(self, seq: Sequence) -> None:
        """Add at most one block, or trim blocks beyond ``get_num_blocks()``
        (DeepInfer ``Scheduler::_append_slot``). Call with the lock held."""
        num_blocks = seq.get_num_blocks()
        table = seq.cache.block_table
        if len(table) < num_blocks:
            table.append(self.block_manager.allocate_single_nolock())
        elif len(table) > num_blocks:
            self.block_manager.free_blocks_nolock(table[num_blocks:])
            del table[num_blocks:]
        if len(table) != num_blocks:
            raise RuntimeError(f"len(block_table) = {len(table)}, but should be {num_blocks}")

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free_blocks(seq.cache.block_table)
        seq.cache.block_table.clear()

    def get_gpu_cache_usage(self, num_gpu_blocks_to_use: int = 0) -> float:
        total = self.block_manager.num_blocks
        used = total - self.block_manager.get_num_free_blocks() + num_gpu_blocks_to_use
        return used / total

    def _allocate_prefill(self, seqs: SequenceList[Sequence], seqlens_q: SequenceList[int] | None) -> None:
        """Blocks for the requests that have none, all or nothing; RUNNING for
        those whose prompt this step completes."""
        with BlockManagerMutexAcquire(self.block_manager) as acquire:
            new = [s for s in seqs if not s.cache.block_table]
            need = sum(s.get_num_blocks() for s in new)
            if self.block_manager.can_allocate_nolock(need) != AllocStatus.OK:
                raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
            for seq in new:
                acquire.allocate_seq(seq)
        qs = seqlens_q if seqlens_q is not None else [s.get_len() - s.num_cached_tokens for s in seqs]
        for seq, q in zip(seqs, qs):
            if seq.num_cached_tokens + q == seq.get_len():
                seq.status = SequenceStatus.RUNNING

    def _allocate_decode(self, seqs: SequenceList[Sequence]) -> None:
        with BlockManagerMutexAcquire(self.block_manager):
            need = sum(len(s.cache.block_table) < s.get_num_blocks() for s in seqs)
            if self.block_manager.can_allocate_nolock(need) != AllocStatus.OK:
                raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
            for seq in seqs:
                self._append_slot(seq)

    def step(
        self,
        seqs: SequenceList[Sequence],
        *,
        is_prompt: bool,
        seqlens_q: SequenceList[int] | None = None,
    ) -> SamplerOutput:
        """One model step for ``seqs``: WAITING ones for prefill, RUNNING ones
        whose prompt is fully cached for decode."""
        for seq in seqs:
            if is_prompt and seq.status != SequenceStatus.WAITING:
                raise ValueError(f"seq {seq.seq_id} is {seq.status.name}; a prefill step wants WAITING")
            if not is_prompt and (seq.status != SequenceStatus.RUNNING or seq.num_cached_tokens != seq.get_len() - 1):
                raise ValueError(f"seq {seq.seq_id} ({seq.status.name}) is not ready for a decode step")
        if is_prompt:
            self._allocate_prefill(seqs, seqlens_q)  # on failure nothing is taken
        try:
            if not is_prompt:
                self._allocate_decode(seqs)
            md = self.runner.prepare_input_tensors(seqs, is_prompt, seqlens_q)
            hidden = self.runner.execute_main_model(md, self.cache_engine.gpu_cache)
            out = self.runner.execute_sample(seqs, md, hidden)
            process_outputs(seqs, out.token_ids, is_prompt, md.seqlens_q, self.max_model_len)
        except BaseException:
            for seq in seqs:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.free_seq(seq)
            raise
        for seq in seqs:
            if seq.is_finished():
                self.free_seq(seq)
        return out

    def generate(
        self,
        prompts: SequenceList[SequenceList[int]],
        sampling_params: SamplingParams | SequenceList[SamplingParams],
    ) -> list[Sequence]:
        """Prefill all prompts in one step, then decode until every one stops."""
        seqs = self.add_requests(prompts, sampling_params)
        running = [s for s in seqs if s.status == SequenceStatus.WAITING]
        if running:
            self.step(running, is_prompt=True)
        while running := [s for s in running if not s.is_finished()]:
            self.step(running, is_prompt=False)
        return seqs
