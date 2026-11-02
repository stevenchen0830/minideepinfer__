"""A static-batch generation loop (S1.3; block manager since S2.1; block
sharing since S2.2).

A small stand-in for DeepInfer's ``LLMEngine`` (engine/llm_engine.py) and its
C++ ``Scheduler`` (csrc/engine/scheduler.hpp), enough to run requests end to
end before the real scheduler arrives in S2.5. "Static": the caller says which
sequences go into each step, and nothing is preempted, so when blocks run out
the step fails instead of waiting.

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

Fork with copy-on-write (S2.2), as in DeepInfer's scheduler:
- ``fork_seq`` (``Scheduler::fork_seq``) builds a new sequence from the
  parent's tokens (the last one may be replaced), all of which become its
  prompt. It keeps the parent's ``request_id``, starts WAITING with
  ``min(parent's cached tokens, len - 1)`` cached tokens and a copy of the
  parent's block table, and pins every block of that table.
- A fork whose parent has every token but the newest cached
  (``cached >= len - 1``; DeepInfer: not ``prompt_incomplete``) waits
  (DeepInfer: the ``prompted`` queue) until a decode step takes it. That step
  allocates one block, copies the block holding the last cached token into it
  (even when that block is full), releases the table's blocks past it, and
  sets RUNNING (``_schedule_decode``'s admission of ``prompted``). A "_" on
  such a fork is kept and has no effect, as in DeepInfer.
- A fork of a parent with more tokens uncached (``cached < len - 1``: a
  chunked prefill before its last chunk) must have a branch id starting with
  "_". Its next prefill step keeps the full shared blocks, copies a partly
  filled tail block into a new block, releases the rest, allocates the blocks
  the prompt still needs, and strips the "_" (``_schedule_prefill``).
- Copies run before the step's forward (``CacheEngine.clone_blocks``, as
  ``clone_and_zero_states_and_blocks``). Every copied or released source
  block is unpinned in the same step, after the forward, once the copies are
  done (DeepInfer: a CUDA event recorded after the copies, then
  ``unpin_blocks`` in ``pre_sampler_sync``).

Prefix cache (S2.2, ours; see prefix_cache.py), with
``enable_prefix_caching=True``:
- a request with an empty block table first looks up the chain hashes of its
  first ``(len - 1) // block_size`` full blocks. The hit blocks start its
  table and its cached count (``hit_tokens``, 0 on a miss); only the rest is
  allocated and prefilled. A ``seqlens_q`` chunk counts from there, which
  ``num_cached_after_lookup`` tells the caller;
- after every successful step, the blocks that became full of committed
  tokens are registered, before finished sequences are freed (a freed block
  keeps its hash until it is handed out again). vLLM registers a block when
  it allocates it, before the forward; see prefix_cache.py.

Differences:
- A prompt that can never run (longer than ``max_model_len``, or
  ``can_allocate_nolock`` answers NEVER) is marked FINISHED_IGNORED by
  ``add_requests``. DeepInfer decides this in ``_schedule_prefill``, when the
  request reaches the head of the waiting queue.
- Blocks of a finished sequence are freed at the end of the step in which it
  finished. DeepInfer frees them later: the next decode schedule takes it out
  of ``running``, and ``LLMEngine.finalize_seq`` -> ``Scheduler::free_seq``
  frees it after that step's forward. So its blocks still count for that
  schedule, and it can still be forked until then. Here its table is already
  empty and ``fork_seq`` raises: fork before the parent's last step.
- A decode step asks only for the blocks it needs, so it can run with no free
  block; DeepInfer's ``_schedule_decode`` wants one free block before every
  ``_append_slot``.
- Only the one EOS id stops a sequence; Qwen3's second end token
  (``<|endoftext|>``, 151643) is not checked, as DeepInfer has one EOS id.
- Failures: a prefill step that finds too few blocks, or whose ``seqlens_q``
  does not fit (one chunk per sequence, from 1 to the tokens left after the
  prefix-cache lookup), takes nothing and leaves its sequences WAITING. When a
  decode step runs out of blocks, or the model call of any step fails, the
  step's sequences are aborted and their blocks freed (copy sources after
  the copies are done), so the engine stays usable. If registering blocks in
  the prefix cache fails after a step, the step stands, finished sequences
  are still freed, and the error is raised afterwards.
- ``fork_seq`` takes no ``so_matcher`` (S6.3), ``seq_id`` or
  ``arrival_time`` (the engine numbers sequences), and no ``sp_token``.
  DeepInfer forces ``sp_token`` as the first output token only of a fork
  whose parent has every token but the newest cached, and never reads a
  fork's own ``SamplingParams.golden_token_ids``; our runner forces
  ``SamplingParams.golden_token_ids`` for every sequence. To match DeepInfer,
  give such a fork ``golden_token_ids=[sp_token]`` (nothing when sp_token is
  -1) and a "_" fork nothing. check_sharing.py forces longer lists, a
  test-only aid.
- ``fork_seq``'s ``blocks``, ``target_branch_id`` and
  ``replace_last_token_id`` have defaults, so ``sampling_params`` is passed
  by keyword; DeepInfer's binding takes all nine arguments, required. It
  sets the fork's ``eos_id`` as ``add_requests`` does. It checks every block
  before pinning any, that the table can hold the cached tokens (DeepInfer
  checks only when the fork is admitted), that given ``blocks`` hold no id
  twice and, with the prefix cache on, that a given block the cache knows
  holds the fork's own tokens. It refuses a fork with ``cached < len - 1``
  and no "_" (DeepInfer does not check). A fork that is already finished
  (longer than ``max_model_len``) is freed at once; DeepInfer still copies
  its block at admission and frees it afterwards.
- A waiting fork is admitted by the first decode step that includes it, not
  at the start of every decode schedule, and all the blocks a decode step
  needs (copies and new slots) are checked at once.
- DeepInfer's ``pre_sampler_sync`` runs after the sampler is enqueued and
  before the CPU waits for the sampled tokens. Our ``execute_sample`` waits
  (``.tolist()``), so the copy sources are unpinned just before it.
- ``num_cached_after_lookup`` is ours: DeepInfer's scheduler sizes prefill
  chunks itself, after the lookup.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence as SequenceList

import torch

from mini_deepinfer.engine.block_manager import AllocStatus, BlockManager, BlockManagerMutexAcquire
from mini_deepinfer.engine.prefix_cache import NONE_HASH, PrefixCachingBlockManager, hash_block_tokens, hash_blocks
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import Sequence, SequenceStatus
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner, SamplerOutput

CloneList = list[tuple[int, int]]


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


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class StaticEngine:
    def __init__(
        self,
        runner: ModelRunner,
        cache_engine: CacheEngine,
        *,
        eos_token_id: int,
        enable_prefix_caching: bool = False,
    ) -> None:
        if cache_engine.block_size != runner.block_size:
            raise ValueError("runner and cache engine disagree on block_size")
        self.runner = runner
        self.cache_engine = cache_engine
        self.eos_token_id = eos_token_id
        self.max_model_len = runner.max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        # The block manager counts every block of the cache, including the
        # reserved block 0.
        manager = PrefixCachingBlockManager if enable_prefix_caching else BlockManager
        self.block_manager = manager(cache_engine.num_blocks)
        # Forks whose parent had every token but the newest cached, waiting for
        # the decode step that copies their last cached block (DeepInfer: the
        # ``prompted`` queue).
        self.prompted: list[Sequence] = []
        self._next_id = 0

    def _new_seq(self, prompt: SequenceList[int], sp: SamplingParams, request_id: str | None = None) -> Sequence:
        sp.eos_id = self.eos_token_id  # before the Sequence is built, as DeepInfer does
        seq = Sequence(
            seq_id=self._next_id,
            prompt_token_ids=prompt,
            block_size=self.cache_engine.block_size,
            request_id=str(self._next_id) if request_id is None else request_id,
            sampling_params=sp,
            arrival_time=0.0,
        )
        self._next_id += 1
        return seq

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
            seq = self._new_seq(prompt, sp)
            with BlockManagerMutexAcquire(self.block_manager):
                never = self.block_manager.can_allocate_nolock(seq.get_num_blocks()) == AllocStatus.NEVER
            if seq.get_prompt_len() > self.max_model_len or never:
                seq.status = SequenceStatus.FINISHED_IGNORED
            seqs.append(seq)
        return seqs

    def fork_seq(
        self,
        seq: Sequence,
        blocks: SequenceList[int] | None = None,
        target_branch_id: str = "default",
        *,
        sampling_params: SamplingParams,
        replace_last_token_id: int = -1,
    ) -> Sequence:
        """A new sequence that shares ``seq``'s cached KV (DeepInfer
        ``Scheduler::fork_seq``). ``blocks`` replaces the parent's block table:
        blocks someone already holds, with the fork's own KV for its cached
        tokens. A branch id starting with "_" is required when the parent has
        fewer than ``len - 1`` tokens cached (DeepInfer's
        ``prompt_incomplete``)."""
        if seq.num_cached_tokens <= 0:
            raise ValueError("Cannot fork a sequence without a computed prefix")
        prompt_incomplete = seq.num_cached_tokens < seq.get_len() - 1
        if prompt_incomplete and not target_branch_id.startswith("_"):
            raise ValueError(
                f"a fork of a sequence with {seq.num_cached_tokens} of {seq.get_len()} tokens cached "
                "(fewer than len - 1) needs a branch id starting with '_'"
            )
        table = list(seq.cache.block_table) if blocks is None else [operator.index(b) for b in blocks]
        if not table:
            raise ValueError("Cannot fork a sequence without allocated KV blocks")
        if len(set(table)) != len(table):
            raise ValueError(f"a block appears twice in {table}")
        tokens = seq.get_token_ids()
        if replace_last_token_id >= 0:
            tokens[-1] = replace_last_token_id
        num_cached = min(seq.num_cached_tokens, len(tokens) - 1)
        if len(table) < _ceil_div(num_cached, seq.block_size):
            raise ValueError(f"{len(table)} blocks cannot hold {num_cached} cached tokens")
        child = self._new_seq(tokens, sampling_params, request_id=seq.request_id)
        bm = self.block_manager
        with BlockManagerMutexAcquire(bm):
            for block_id in table:  # check them all before pinning any
                bm.blocks._check_id(block_id)
                if bm.blocks.ref_count[block_id] <= 0:
                    raise RuntimeError(f"try to pin free block {block_id}")
            if blocks is not None and self.enable_prefix_caching:
                # A given block that the cache knows must hold the fork's own tokens.
                for block_id, h in zip(table, hash_blocks(tokens, seq.block_size, num_cached // seq.block_size)):
                    if bm.block_hash[block_id] not in (None, h):
                        raise ValueError(f"block {block_id} holds other tokens than the fork's")
            bm.pin_blocks_nolock(table)
        child.branch_id = target_branch_id
        child.cache.hit_tokens = seq.cache.hit_tokens
        child.cache.block_table = table
        child.cache.set_num_tokens(num_cached)
        if blocks is None:  # the same tokens in the same blocks: the same hashes
            child.cache.block_hashes = seq.cache.block_hashes[: num_cached // child.block_size]
        if child.get_len() > self.max_model_len:
            child.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            self.free_seq(child)
        elif not prompt_incomplete:
            self.prompted.append(child)
        return child

    def _is_prompted(self, seq: Sequence) -> bool:
        return any(seq is p for p in self.prompted)

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
        self.prompted = [p for p in self.prompted if p is not seq]

    def get_gpu_cache_usage(self, num_gpu_blocks_to_use: int = 0) -> float:
        total = self.block_manager.num_blocks
        used = total - self.block_manager.get_num_free_blocks() + num_gpu_blocks_to_use
        return used / total

    def num_cached_after_lookup(self, seq: Sequence) -> int:
        """The cached count the next prefill of ``seq`` starts from, which its
        ``seqlens_q`` chunk counts from: a request without blocks may hit the
        prefix cache."""
        if seq.cache.block_table:
            return seq.num_cached_tokens
        with BlockManagerMutexAcquire(self.block_manager):
            return len(self._lookup_nolock(seq)[1]) * seq.block_size

    def _lookup_nolock(self, seq: Sequence) -> tuple[list[bytes], list[int]]:
        """Chain hashes of the blocks a new request may take from the prefix
        cache, and the cached blocks of the longest leading run."""
        if not self.enable_prefix_caching:
            return [], []
        bs = seq.block_size
        hashes = hash_blocks(seq.get_token_ids(), bs, (seq.get_len() - 1) // bs)
        hits = self.block_manager.find_cached_blocks_nolock(hashes)
        return hashes[: len(hits)], hits

    @staticmethod
    def _cow_prefill_fork(seq: Sequence, new_blocks: list[int]) -> CloneList:
        """DeepInfer ``_schedule_prefill`` for a branch id starting with "_"."""
        bs = seq.block_size
        reserve = seq.num_cached_tokens // bs
        has_tail = seq.num_cached_tokens % bs != 0
        table = seq.cache.block_table
        clones = [(table[reserve], new_blocks[0])] if has_tail else []
        clones += [(block_id, 0) for block_id in table[reserve + has_tail :]]
        table[reserve:] = new_blocks
        seq.branch_id = seq.branch_id[1:]
        return clones

    def _allocate_prefill(self, seqs: SequenceList[Sequence], seqlens_q: SequenceList[int] | None) -> CloneList:
        """Blocks for the requests that have none (prefix-cache hits first) and
        copy-on-write for "_" forks, all or nothing; RUNNING for those whose
        prompt this step completes. Returns the block copies to run."""
        bm = self.block_manager
        clones: CloneList = []
        with BlockManagerMutexAcquire(bm) as acquire:
            fresh = [s for s in seqs if not s.cache.block_table]
            forks = [s for s in seqs if s.cache.block_table and s.branch_id.startswith("_")]
            lookups = [self._lookup_nolock(s) for s in fresh]
            if seqlens_q is not None:  # a chunk counts from the hit: check before taking anything
                if len(seqlens_q) != len(seqs):
                    raise ValueError("seqlens_q chooses prefill chunk sizes: one per sequence")
                starts = {id(s): len(hits) * s.block_size for s, (_, hits) in zip(fresh, lookups)}
                for seq, q in zip(seqs, seqlens_q):
                    left = seq.get_len() - starts.get(id(seq), seq.num_cached_tokens)
                    if not 0 < q <= left:
                        raise ValueError(f"seq {seq.seq_id}: a chunk of {q} tokens, but {left} are left to prefill")
            need = sum(s.get_num_blocks() - len(hits) for s, (_, hits) in zip(fresh, lookups))
            need += sum(s.get_num_blocks() - s.num_cached_tokens // s.block_size for s in forks)
            if self.enable_prefix_caching:
                need += bm.num_free_cached_nolock(b for _, hits in lookups for b in hits)
            if bm.can_allocate_nolock(need) != AllocStatus.OK:
                raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
            # Take every hit before allocating anything.
            for _, hits in lookups:
                if hits:
                    bm.take_cached_blocks_nolock(hits)
            for seq, (hashes, hits) in zip(fresh, lookups):
                seq.cache.hit_tokens = len(hits) * seq.block_size  # this admission's hit, 0 on a miss
                if not hits:
                    acquire.allocate_seq(seq)
                    continue
                seq.cache.block_table = hits + bm.allocate_nolock(seq.get_num_blocks() - len(hits))
                seq.cache.set_num_tokens(len(hits) * seq.block_size)
                seq.cache.block_hashes = hashes
            for seq in forks:
                new_blocks = bm.allocate_nolock(seq.get_num_blocks() - seq.num_cached_tokens // seq.block_size)
                clones += self._cow_prefill_fork(seq, new_blocks)
        qs = seqlens_q if seqlens_q is not None else [s.get_len() - s.num_cached_tokens for s in seqs]
        for seq, q in zip(seqs, qs):
            if seq.num_cached_tokens + q == seq.get_len():
                seq.status = SequenceStatus.RUNNING
        return clones

    def _cow_decode_fork(self, seq: Sequence) -> CloneList:
        """DeepInfer ``_schedule_decode``'s admission of a forked sequence from
        ``prompted``. Call with the lock held."""
        num_blocks = _ceil_div(seq.num_cached_tokens, seq.block_size)
        table = seq.cache.block_table
        new_block = self.block_manager.allocate_single_nolock()
        clones = [(table[num_blocks - 1], new_block)]
        clones += [(block_id, 0) for block_id in table[num_blocks:]]
        table[num_blocks - 1] = new_block
        del table[num_blocks:]
        seq.status = SequenceStatus.RUNNING
        self.prompted = [p for p in self.prompted if p is not seq]
        return clones

    def _allocate_decode(self, seqs: SequenceList[Sequence]) -> CloneList:
        bm = self.block_manager
        with BlockManagerMutexAcquire(bm):
            forks = [s for s in seqs if self._is_prompted(s)]
            need = 0
            for seq in seqs:
                if self._is_prompted(seq):  # the copy, plus a slot if the copied block is full
                    need += 1 + (seq.get_num_blocks() > _ceil_div(seq.num_cached_tokens, seq.block_size))
                else:
                    need += len(seq.cache.block_table) < seq.get_num_blocks()
            if bm.can_allocate_nolock(need) != AllocStatus.OK:
                raise RuntimeError("out of KV-cache blocks (no preemption in the static engine)")
            clones: CloneList = []
            try:
                for seq in forks:
                    clones += self._cow_decode_fork(seq)
                for seq in seqs:
                    self._append_slot(seq)
            except BaseException:
                # An admitted fork's table no longer lists its copy sources:
                # drop those pins here; step's failure path frees the tables.
                bm.free_blocks_nolock([src for src, _ in clones])
                raise
        return clones

    def _run_clones(self, clones: CloneList) -> torch.cuda.Event | None:
        """Enqueue the copies; on CUDA, return an event that marks them done."""
        if not clones:
            return None
        self.cache_engine.clone_blocks(clones)
        if self.cache_engine.gpu_cache[0].k_cache.device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    def _release_clone_sources(self, sources: list[int], event: torch.cuda.Event | None, *, failed: bool = False) -> None:
        if event is not None and sources:
            try:
                event.synchronize()
            except Exception:
                if not failed:  # after a failed step a broken CUDA context cannot be waited on
                    raise
        if sources:
            self.block_manager.unpin_blocks(sources)

    def _cache_full_blocks(self, seqs: SequenceList[Sequence]) -> None:
        """Register every block that is full of committed tokens and has no
        hash in its sequence's chain yet."""
        bm = self.block_manager
        with BlockManagerMutexAcquire(bm):
            for seq in seqs:
                bs = seq.block_size
                hashes = seq.cache.block_hashes
                tokens = seq.data.token_ids
                if not len(hashes) * bs <= seq.num_cached_tokens <= len(tokens):
                    raise RuntimeError(
                        f"seq {seq.seq_id}: {len(hashes)} registered blocks, "
                        f"{seq.num_cached_tokens} cached tokens, {len(tokens)} tokens"
                    )
                for i in range(len(hashes), seq.num_cached_tokens // bs):
                    h = hash_block_tokens(hashes[-1] if hashes else NONE_HASH, tokens[i * bs : (i + 1) * bs])
                    bm.register_nolock(seq.cache.block_table[i], h)
                    hashes.append(h)  # only once the cache accepted it

    def step(
        self,
        seqs: SequenceList[Sequence],
        *,
        is_prompt: bool,
        seqlens_q: SequenceList[int] | None = None,
    ) -> SamplerOutput:
        """One model step for ``seqs``: WAITING ones for prefill; for decode,
        RUNNING ones and waiting forks, each with every token but the newest
        cached. ``seqlens_q`` (prefill only) gives each sequence's chunk,
        counted from the cached count the step starts it from (after a
        prefix-cache hit: ``num_cached_after_lookup``)."""
        if len({id(s) for s in seqs}) != len(seqs):
            raise ValueError("a sequence appears twice in the step")
        if not is_prompt and seqlens_q is not None:
            raise ValueError("seqlens_q chooses prefill chunk sizes; a decode step takes one token per sequence")
        for seq in seqs:
            if is_prompt and self._is_prompted(seq):
                raise ValueError(f"seq {seq.seq_id} is a fork waiting in engine.prompted: it goes to a decode step")
            if is_prompt and seq.status != SequenceStatus.WAITING:
                raise ValueError(f"seq {seq.seq_id} is {seq.status.name}; a prefill step wants WAITING")
            if not is_prompt and (
                not (seq.status == SequenceStatus.RUNNING or self._is_prompted(seq))
                or seq.num_cached_tokens != seq.get_len() - 1
                or len(seq.cache.block_table) < _ceil_div(seq.num_cached_tokens, seq.block_size)
            ):
                raise ValueError(f"seq {seq.seq_id} ({seq.status.name}) is not ready for a decode step")
        clones = self._allocate_prefill(seqs, seqlens_q) if is_prompt else []  # on failure nothing is taken
        sources = [src for src, _ in clones]
        event = None
        try:
            if not is_prompt:
                clones = self._allocate_decode(seqs)
                sources = [src for src, _ in clones]
            event = self._run_clones(clones)
            md = self.runner.prepare_input_tensors(seqs, is_prompt, seqlens_q)
            hidden = self.runner.execute_main_model(md, self.cache_engine.gpu_cache)
            sources, to_release = [], sources
            # DeepInfer: pre_sampler_sync, after the sampler is enqueued; ours
            # samples synchronously, so just before it.
            self._release_clone_sources(to_release, event)
            out = self.runner.execute_sample(seqs, md, hidden)
            process_outputs(seqs, out.token_ids, is_prompt, md.seqlens_q, self.max_model_len)
        except BaseException:
            self._release_clone_sources(sources, event, failed=True)
            for seq in seqs:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.free_seq(seq)
            raise
        try:
            if self.enable_prefix_caching:
                self._cache_full_blocks(seqs)
        finally:  # the step has committed: finished sequences are freed whatever happens
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
