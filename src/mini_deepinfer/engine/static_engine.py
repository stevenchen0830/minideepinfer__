"""A static-batch generation loop (S1.3).

A small stand-in for DeepInfer's ``LLMEngine`` (engine/llm_engine.py) and its
C++ ``Scheduler`` (csrc/engine/scheduler.hpp), enough to run requests end to
end before phase 2 builds the real ones. "Static": all requests start
together, are prefilled in one step, then decoded together until each one
stops; no request joins later and nothing is preempted.

What it keeps from DeepInfer:
- ``add_requests`` sets every request's ``SamplingParams.eos_id`` to the
  engine's ``eos_token_id`` (``LLMEngine.add_request``), refuses an empty
  prompt, and marks a prompt that can never run FINISHED_IGNORED: longer than
  ``max_model_len``, or needing more blocks than exist (AllocStatus::NEVER);
- a step is all prefill or all decode;
- before a step each sequence gets ``get_num_blocks()`` blocks, for its whole
  token list (DeepInfer: a new prompt takes them all in
  ``BlockManagerMutexAcquire::allocate_seq``; a decode step adds at most one
  in ``Scheduler::_append_slot``); a finished sequence gives them back;
- ``process_outputs`` updates a sequence after a step the way
  ``Scheduler::process_outputs`` does: ``advance_optimistically(seqlen_q)``
  on every step; the sampled token is appended only once the prompt is fully
  cached or on a decode step; ``commit_processed_tokens(1)`` only on decode;
- ``check_stop`` in DeepInfer's order: EOS (unless ``ignore_eos``) ->
  FINISHED_STOPPED; ``get_len() > max_model_len`` -> FINISHED_LENGTH_CAPPED;
  ``get_output_len() == max_tokens`` -> FINISHED_LENGTH_CAPPED.

``BlockAllocator`` is a plain FIFO free list of block ids in which block 0 is
an ordinary block. DeepInfer's block manager is also FIFO (a freed table goes
to the tail, last block first), with reference counts and OK/LATER/NEVER
answers, and it never hands out block 0: block 0 is a dummy that CUDA-graph
padding rows write to and that its block copies use to mean "no block". That
is S2.1. Only the one EOS id stops a sequence; Qwen3's second end token
(``<|endoftext|>``, 151643) is not checked, as DeepInfer has one EOS id.

When a step fails, its sequences are aborted and their blocks freed, so the
engine stays usable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence as SequenceList

from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import Sequence, SequenceStatus
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner, SamplerOutput


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
        return self._free.popleft()

    def free(self, block_ids: SequenceList[int]) -> None:
        self._free.extend(block_ids)


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
        self.allocator = BlockAllocator(cache_engine.num_blocks)
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
            never_fits = seq.get_num_blocks() > self.allocator.num_blocks
            if seq.get_prompt_len() > self.max_model_len or never_fits:
                seq.status = SequenceStatus.FINISHED_IGNORED
            else:
                seq.status = SequenceStatus.RUNNING
            seqs.append(seq)
        return seqs

    def _append_slots(self, seq: Sequence) -> None:
        while len(seq.cache.block_table) < seq.get_num_blocks():
            seq.cache.block_table.append(self.allocator.allocate())

    def free_seq(self, seq: Sequence) -> None:
        self.allocator.free(seq.cache.block_table)
        seq.cache.block_table.clear()

    def step(
        self,
        seqs: SequenceList[Sequence],
        *,
        is_prompt: bool,
        seqlens_q: SequenceList[int] | None = None,
    ) -> SamplerOutput:
        """One model step for ``seqs``."""
        for seq in seqs:
            if seq.status != SequenceStatus.RUNNING:
                raise ValueError(f"seq {seq.seq_id} is {seq.status.name}, not RUNNING")
        need = sum(max(0, s.get_num_blocks() - len(s.cache.block_table)) for s in seqs)
        if need > self.allocator.num_free:
            raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
        try:
            for seq in seqs:
                self._append_slots(seq)
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
        running = [s for s in seqs if s.status == SequenceStatus.RUNNING]
        if running:
            self.step(running, is_prompt=True)
        while running := [s for s in running if not s.is_finished()]:
            self.step(running, is_prompt=False)
        return seqs
