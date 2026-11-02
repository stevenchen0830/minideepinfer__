"""Block sharing in the engine (S2.2): fork with copy-on-write and the prefix
cache, on tiny random checkpoints in float32 on the CPU.

Every step is checked the S1.3 way: each row of logits must equal a full
recompute of that sequence's tokens without a cache, and the cache starts
full of NaN. On top of that, every slot a step writes is checked against the
other holders of its block and against the prefix cache (``check_writes``).
"""

import random

import pytest
import torch

from mini_deepinfer.engine.prefix_cache import hash_blocks
from mini_deepinfer.sequence import SequenceStatus
from test_engine import full_logits, greedy, make_engine, random_prompts

BS = 4


def engine_for(tiny_checkpoint, **kw):
    _, path = tiny_checkpoint()
    kw.setdefault("block_size", BS)
    return make_engine(path, **kw)


def holders(seqs):
    """Sequences that hold blocks."""
    return [s for s in seqs if s.cache.block_table]


def check_writes(engine, everyone, step_seqs, md):
    bm = engine.block_manager
    bs = engine.cache_engine.block_size
    slots = md.slot_mapping.tolist()
    start = 0
    for seq, q in zip(step_seqs, md.seqlens_q):
        for slot in slots[start : start + q]:
            block, offset = divmod(slot, bs)
            for other in holders(everyone):
                if other is seq:
                    continue
                for j, b in enumerate(other.cache.block_table):
                    if b == block:
                        assert j * bs + offset >= other.num_cached_tokens, (
                            f"seq {seq.seq_id} writes position {j * bs + offset} of block {block}, cached by seq {other.seq_id}"
                        )
            if engine.enable_prefix_caching:
                assert bm.block_hash[block] is None, f"write into cached block {block}"
        start += q


def run_step(engine, seqs, *, is_prompt, seqlens_q=None, everyone=None):
    """One engine step; checks every row against a full recompute and every
    write against the other holders. Returns the step's InputMetadata."""
    runner = engine.runner
    seen = {}

    def prepare(step_seqs, prompt, q=None):
        md = type(runner).prepare_input_tensors(runner, step_seqs, prompt, q)
        seen["md"], seen["tokens"] = md, [s.get_token_ids() for s in step_seqs]
        check_writes(engine, everyone if everyone is not None else step_seqs, step_seqs, md)
        return md

    runner.prepare_input_tensors = prepare
    try:
        out = engine.step(seqs, is_prompt=is_prompt, seqlens_q=seqlens_q)
    finally:
        del runner.prepare_input_tensors
    md = seen["md"]
    for row, tokens, k in zip(out.logits, seen["tokens"], md.seqlens_k):
        torch.testing.assert_close(row, full_logits(runner.model, tokens[:k]), rtol=1e-4, atol=1e-4)
    return md


def check_ref_counts(engine, everyone):
    bm = engine.block_manager
    for b in range(1, bm.num_blocks):
        expected = sum(s.cache.block_table.count(b) for s in holders(everyone))
        assert bm.blocks.ref_count[b] == expected, f"block {b}: count {bm.blocks.ref_count[b]}, held {expected} times"
    assert bm.blocks.ref_count[0] == 0 and all(0 not in s.cache.block_table for s in everyone)


# --- CacheEngine.clone_blocks -------------------------------------------------


def test_clone_blocks_copies_every_layer(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=6)
    cache = engine.cache_engine
    for layer in cache.gpu_cache:
        layer.k_cache.copy_(torch.randn_like(layer.k_cache))
        layer.v_cache.copy_(torch.randn_like(layer.v_cache))
    before = [(layer.k_cache.clone(), layer.v_cache.clone()) for layer in cache.gpu_cache]
    srcs = cache.clone_blocks([(1, 3), (2, 0), (1, 4)])
    assert srcs == [1, 2, 1]  # every source, the release-only one too
    for layer, (k, v) in zip(cache.gpu_cache, before):
        for new, old in ((layer.k_cache, k), (layer.v_cache, v)):
            assert torch.equal(new[3], old[1]) and torch.equal(new[4], old[1])
            assert torch.equal(new[[0, 1, 2, 5]], old[[0, 1, 2, 5]])  # nothing else changed
    with pytest.raises(ValueError):
        cache.clone_blocks([(1, 3), (3, 4)])  # 3 is a source and a destination
    with pytest.raises(ValueError):
        cache.clone_blocks([(1, 3), (2, 3)])  # two copies into 3


# --- fork after the prompt (decode path) --------------------------------------


@pytest.mark.parametrize("prompt_len", [4, 5, 7, 8, 9])  # cached tokens % 4 = 0, 1, 3, 0, 1
@pytest.mark.parametrize("replace", [True, False])
def test_fork_after_prefill(tiny_checkpoint, prompt_len, replace):
    engine = engine_for(tiny_checkpoint)
    vocab = engine.runner.model.cfg.vocab_size
    (prompt,) = random_prompts([prompt_len], vocab, seed=prompt_len)
    (parent,) = engine.add_requests([prompt], greedy(max_tokens=6))
    run_step(engine, [parent], is_prompt=True)
    p = parent.num_cached_tokens
    last = (parent.get_last_token_id() + 1) % vocab if replace else -1
    child = engine.fork_seq(parent, sampling_params=greedy(max_tokens=5), replace_last_token_id=last)
    everyone = [parent, child]
    assert child.status == SequenceStatus.WAITING and engine.prompted == [child]
    assert child.get_prompt_len() == child.get_len() == parent.get_len() and child.get_output_len() == 0
    assert child.num_cached_tokens == p and child.request_id == parent.request_id
    assert child.cache.block_table == parent.cache.block_table and child.cache.block_table is not parent.cache.block_table
    check_ref_counts(engine, everyone)
    shared = parent.cache.block_table[: -(-p // BS) - 1]  # the full blocks before the last cached one
    run_step(engine, [parent, child], is_prompt=False, everyone=everyone)
    assert child.status == SequenceStatus.RUNNING and engine.prompted == []
    assert child.cache.block_table[: len(shared)] == shared
    assert child.cache.block_table[len(shared)] != parent.cache.block_table[len(shared)]  # its own copy
    check_ref_counts(engine, everyone)
    while not (parent.is_finished() and child.is_finished()):
        run_step(engine, [s for s in everyone if not s.is_finished()], is_prompt=False, everyone=everyone)
        check_ref_counts(engine, everyone)
    assert child.get_output_len() == 5 and parent.get_output_len() == 6
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


def test_fork_at_a_block_boundary(tiny_checkpoint):
    """A fork whose cached tokens fill whole blocks."""
    engine = engine_for(tiny_checkpoint, num_blocks=6)  # 5 usable
    (parent,) = engine.add_requests([[1] * 8], greedy())
    engine.step([parent], is_prompt=True)
    child = engine.fork_seq(parent, sampling_params=greedy())
    free = engine.block_manager.get_num_free_blocks()
    run_step(engine, [child], is_prompt=False, everyone=[parent, child])
    assert child.cache.block_table[0] == parent.cache.block_table[0]
    assert len(child.cache.block_table) == 3
    assert engine.block_manager.get_num_free_blocks() == free - 2 == 1
    other = engine.fork_seq(parent, sampling_params=greedy())
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([other], is_prompt=False)
    check_ref_counts(engine, [parent, child, other])


def test_parent_writes_while_the_fork_waits(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    vocab = engine.runner.model.cfg.vocab_size
    (parent,) = engine.add_requests(random_prompts([6], vocab, seed=11), greedy(max_tokens=8))
    run_step(engine, [parent], is_prompt=True)
    child = engine.fork_seq(parent, sampling_params=greedy(max_tokens=3), replace_last_token_id=5)
    for _ in range(3):
        run_step(engine, [parent], is_prompt=False, everyone=[parent, child])
    run_step(engine, [child], is_prompt=False, everyone=[parent, child])
    run_step(engine, [child, parent], is_prompt=False, everyone=[parent, child])


def test_fork_outlives_its_parent_and_forks_of_forks(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    vocab = engine.runner.model.cfg.vocab_size
    (parent,) = engine.add_requests(random_prompts([9], vocab, seed=12), greedy(max_tokens=8))
    run_step(engine, [parent], is_prompt=True)
    a = engine.fork_seq(parent, sampling_params=greedy(max_tokens=4), replace_last_token_id=1)
    b = engine.fork_seq(a, sampling_params=greedy(max_tokens=4), replace_last_token_id=2)  # before a is admitted
    everyone = [parent, a, b]
    check_ref_counts(engine, everyone)
    parent.status = SequenceStatus.FINISHED_ABORTED
    engine.free_seq(parent)  # the forks keep the blocks alive
    check_ref_counts(engine, everyone)
    run_step(engine, [a, b], is_prompt=False, everyone=everyone)
    c = engine.fork_seq(b, sampling_params=greedy(max_tokens=2))  # a fork of a running fork
    everyone.append(c)
    while any(not s.is_finished() for s in (a, b, c)):
        run_step(engine, [s for s in (a, b, c) if not s.is_finished()], is_prompt=False, everyone=everyone)
        check_ref_counts(engine, everyone)
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


@pytest.mark.parametrize("prompt_len", [5, 9, 10])
@pytest.mark.parametrize("branch", ["fork", "_fork"])
def test_fork_one_prompt_token_short(tiny_checkpoint, prompt_len, branch):
    """A parent with every token but its last prompt token cached: the fork
    goes to the decode path, "_" or not."""
    engine = engine_for(tiny_checkpoint)
    vocab = engine.runner.model.cfg.vocab_size
    (prompt,) = random_prompts([prompt_len], vocab, seed=prompt_len)
    (parent,) = engine.add_requests([prompt], greedy(max_tokens=3))
    run_step(engine, [parent], is_prompt=True, seqlens_q=[prompt_len - 1])
    child = engine.fork_seq(parent, None, branch, sampling_params=greedy(max_tokens=3))
    everyone = [parent, child]
    assert engine.prompted == [child] and parent.status == SequenceStatus.WAITING
    with pytest.raises(ValueError, match="decode step"):
        engine.step([child], is_prompt=True)
    table = list(child.cache.block_table)
    cache, pairs = engine.cache_engine, []

    def clone(clone_pairs):
        pairs.extend(clone_pairs)
        return type(cache).clone_blocks(cache, clone_pairs)

    cache.clone_blocks = clone
    run_step(engine, [child], is_prompt=False, everyone=everyone)
    del cache.clone_blocks
    num_blocks = -(-(prompt_len - 1) // BS)
    assert pairs[0][0] == table[num_blocks - 1] and pairs[1:] == [(b, 0) for b in table[num_blocks:]]
    assert child.branch_id == branch and child.status == SequenceStatus.RUNNING
    check_ref_counts(engine, everyone)
    run_step(engine, [parent], is_prompt=True, everyone=everyone)
    while not (parent.is_finished() and child.is_finished()):
        run_step(engine, [s for s in everyone if not s.is_finished()], is_prompt=False, everyone=everyone)
        check_ref_counts(engine, everyone)
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


# --- fork during a chunked prefill (prefill path, "_" branch) ------------------


@pytest.mark.parametrize("cached", [4, 5, 7])
def test_fork_mid_prefill(tiny_checkpoint, cached):
    engine = engine_for(tiny_checkpoint)
    vocab = engine.runner.model.cfg.vocab_size
    (prompt,) = random_prompts([11], vocab, seed=cached)
    (parent,) = engine.add_requests([prompt], greedy(max_tokens=3))
    run_step(engine, [parent], is_prompt=True, seqlens_q=[cached])
    with pytest.raises(ValueError, match="'_'"):
        engine.fork_seq(parent, sampling_params=greedy())
    child = engine.fork_seq(parent, None, "_branch", sampling_params=greedy(max_tokens=3))
    everyone = [parent, child]
    assert child.status == SequenceStatus.WAITING and engine.prompted == []
    assert child.num_cached_tokens == cached and child.get_len() == 11
    parent_table = list(parent.cache.block_table)
    cache, pairs = engine.cache_engine, []

    def clone(clone_pairs):
        pairs.extend(clone_pairs)
        return type(cache).clone_blocks(cache, clone_pairs)

    cache.clone_blocks = clone
    run_step(engine, [child, parent], is_prompt=True, everyone=everyone)
    del cache.clone_blocks
    reserve = cached // BS
    copies = [(src, dst) for src, dst in pairs if dst]
    assert [src for src, _ in pairs] == parent_table[reserve:]
    assert [src for src, _ in copies] == ([parent_table[reserve]] if cached % BS else [])
    assert child.branch_id == "branch"
    assert child.cache.block_table[:reserve] == parent_table[:reserve]
    assert not set(child.cache.block_table[reserve:]) & set(parent_table)
    assert child.status == parent.status == SequenceStatus.RUNNING
    check_ref_counts(engine, everyone)
    while not (parent.is_finished() and child.is_finished()):
        run_step(engine, [s for s in everyone if not s.is_finished()], is_prompt=False, everyone=everyone)
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


# --- rules and failures ----------------------------------------------------------


def test_fork_rules(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=8)
    (parent,) = engine.add_requests([[1, 2, 3, 4, 5, 6]], greedy())
    with pytest.raises(ValueError, match="computed prefix"):
        engine.fork_seq(parent, sampling_params=greedy())
    engine.step([parent], is_prompt=True)
    free = engine.block_manager.get_num_free_blocks()
    with pytest.raises(RuntimeError, match="free block"):
        engine.fork_seq(parent, [parent.cache.block_table[0], 7], sampling_params=greedy())  # 7 is free
    with pytest.raises(ValueError, match="cannot hold"):
        engine.fork_seq(parent, parent.cache.block_table[:1], sampling_params=greedy())  # 6 cached tokens, 1 block
    assert engine.block_manager.get_num_free_blocks() == free
    assert engine.block_manager.blocks.ref_count[parent.cache.block_table[0]] == 1  # nothing was pinned
    child = engine.fork_seq(parent, sampling_params=greedy())
    with pytest.raises(ValueError):
        engine.step([child], is_prompt=True)  # a fork of a cached prompt goes to decode
    with pytest.raises(ValueError, match="twice"):
        engine.step([parent, parent], is_prompt=False)
    engine.max_model_len = 6  # the next fork is longer than this
    capped = engine.fork_seq(parent, sampling_params=greedy())
    assert capped.status == SequenceStatus.FINISHED_LENGTH_CAPPED and capped.cache.block_table == []
    child.status = SequenceStatus.FINISHED_ABORTED
    engine.free_seq(child)  # a waiting fork can be freed: it leaves the waiting list too
    assert engine.prompted == []
    check_ref_counts(engine, [parent, child, capped])


def test_fork_rules_for_given_blocks(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    vocab = engine.runner.model.cfg.vocab_size
    a, b = engine.add_requests(random_prompts([8, 8], vocab, seed=71), greedy())
    engine.step([a, b], is_prompt=True)  # both register their two full blocks
    counts = list(engine.block_manager.blocks.ref_count)
    with pytest.raises(ValueError, match="other tokens"):
        engine.fork_seq(a, b.cache.block_table, sampling_params=greedy())  # b's blocks hold b's tokens
    with pytest.raises(ValueError, match="twice"):
        engine.fork_seq(a, [a.cache.block_table[0]] * 2, sampling_params=greedy())
    assert engine.block_manager.blocks.ref_count == counts  # nothing was pinned
    child = engine.fork_seq(a, list(a.cache.block_table), sampling_params=greedy())  # its own blocks: fine
    run_step(engine, [child], is_prompt=False, everyone=[a, b, child])


def test_a_finished_parent_cannot_be_forked(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    (parent,) = engine.add_requests([[1, 2, 3, 4, 5]], greedy(max_tokens=1))
    engine.step([parent], is_prompt=True)
    assert parent.is_finished() and parent.num_cached_tokens == 5
    with pytest.raises(ValueError, match="allocated KV blocks"):
        engine.fork_seq(parent, sampling_params=greedy())  # its blocks were freed with its last step


def test_decode_refuses_a_table_too_short(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    (parent,) = engine.add_requests([list(range(10, 19))], greedy())
    engine.step([parent], is_prompt=True)
    child = engine.fork_seq(parent, sampling_params=greedy())
    last = parent.cache.block_table.pop()  # misuse: 9 cached tokens, 2 blocks
    with pytest.raises(ValueError, match="not ready"):
        engine.step([child, parent], is_prompt=False)
    assert engine.prompted == [child] and child.status == SequenceStatus.WAITING  # nothing happened
    parent.cache.block_table.append(last)
    run_step(engine, [child, parent], is_prompt=False, everyone=[parent, child])
    check_ref_counts(engine, [parent, child])


def test_seqlens_q_is_checked_before_anything_is_taken(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    vocab = engine.runner.model.cfg.vocab_size
    (prompt,) = random_prompts([9], vocab, seed=81)
    engine.generate([prompt], greedy(max_tokens=1))  # 2 full blocks cached
    (seq,) = engine.add_requests([prompt[:8] + [1, 2, 3]], greedy())
    free = engine.block_manager.get_num_free_blocks()
    assert engine.num_cached_after_lookup(seq) == 8
    for bad in ([11], [0], [3, 3]):
        with pytest.raises(ValueError):
            engine.step([seq], is_prompt=True, seqlens_q=bad)
        assert seq.status == SequenceStatus.WAITING and seq.cache.block_table == []
        assert engine.block_manager.get_num_free_blocks() == free
    md = run_step(engine, [seq], is_prompt=True, seqlens_q=[2])
    assert md.seqlens_q == [2] and seq.num_cached_tokens == 10 and seq.status == SequenceStatus.WAITING
    with pytest.raises(ValueError, match="decode step"):
        engine.step([seq], is_prompt=False, seqlens_q=[1])


def test_recompute_after_preemption_hits_again(tiny_checkpoint):
    """S2.5 will preempt by recompute: reset the counters, free the blocks,
    prefill again. With the prefix cache the new prefill may hit."""
    engine = engine_for(tiny_checkpoint, num_blocks=7, enable_prefix_caching=True)  # 6 usable
    vocab = engine.runner.model.cfg.vocab_size
    (seq,) = engine.add_requests(random_prompts([9], vocab, seed=91), greedy(max_tokens=8))
    run_step(engine, [seq], is_prompt=True)
    for _ in range(4):
        run_step(engine, [seq], is_prompt=False)

    def preempt():
        seq.reset_for_recompute()
        seq.status = SequenceStatus.WAITING
        engine.free_seq(seq)

    preempt()
    run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 12  # 14 tokens: the decode-written blocks count too
    preempt()
    engine.generate(random_prompts([24], vocab, seed=92), greedy(max_tokens=1))  # takes every block
    run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 0


def test_a_failed_registration_still_frees_finished_sequences(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    (seq,) = engine.add_requests([[1, 2, 3, 4, 5]], greedy(max_tokens=1))

    def fail(seqs):
        raise RuntimeError("registration failed")

    engine._cache_full_blocks = fail
    with pytest.raises(RuntimeError, match="registration failed"):
        engine.step([seq], is_prompt=True)
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED and seq.get_output_len() == 1  # the step stands
    assert seq.cache.block_table == [] and engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


def test_decode_fork_out_of_blocks_aborts_and_frees(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=3)  # 2 usable
    (parent,) = engine.add_requests([[1, 2, 3, 4, 5, 6]], greedy())
    engine.step([parent], is_prompt=True)  # both blocks
    child = engine.fork_seq(parent, sampling_params=greedy())
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([child], is_prompt=False)  # the copy needs a block
    assert child.status == SequenceStatus.FINISHED_ABORTED and child.cache.block_table == []
    assert engine.prompted == []
    check_ref_counts(engine, [parent, child])


def test_sources_are_unpinned_before_sampling_and_after_a_failure(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    runner = engine.runner
    bm = engine.block_manager
    (parent,) = engine.add_requests([[1, 2, 3, 4, 5, 6]], greedy())
    engine.step([parent], is_prompt=True)
    tail = parent.cache.block_table[1]
    child = engine.fork_seq(parent, sampling_params=greedy())
    assert bm.blocks.ref_count[tail] == 2
    seen = []
    cache = engine.cache_engine

    def clone(pairs):
        seen.append(("copy", pairs[0][0] == tail, bm.blocks.ref_count[tail]))
        return type(cache).clone_blocks(cache, pairs)

    def sample(*args):
        seen.append(("sample", bm.blocks.ref_count[tail]))
        return type(runner).execute_sample(runner, *args)

    cache.clone_blocks, runner.execute_sample = clone, sample
    engine.step([child], is_prompt=False)
    del cache.clone_blocks, runner.execute_sample
    # Still pinned while it is copied; released after the forward, before our
    # synchronous sampler (DeepInfer: in pre_sampler_sync, after enqueuing the
    # sampler and before waiting for it).
    assert seen == [("copy", True, 2), ("sample", 1)]
    other = engine.fork_seq(parent, sampling_params=greedy())
    assert bm.blocks.ref_count[tail] == 2

    def fail(*args):
        raise RuntimeError("model failure")

    runner.execute_main_model = fail
    with pytest.raises(RuntimeError, match="model failure"):
        engine.step([other], is_prompt=False)
    del runner.execute_main_model
    assert bm.blocks.ref_count[tail] == 1 and other.cache.block_table == []
    check_ref_counts(engine, [parent, child, other])


# --- prefix cache in the engine ---------------------------------------------------


def test_prefix_cache_off_by_default(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint)
    engine.generate([[1] * 9], greedy(max_tokens=2))
    (seq,) = engine.add_requests([[1] * 9], greedy())
    md = run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 0 and md.seqlens_q == [9]


def test_second_request_hits_the_cache(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    vocab = engine.runner.model.cfg.vocab_size
    (prompt,) = random_prompts([11], vocab, seed=21)
    (first,) = engine.generate([prompt], greedy(max_tokens=3))  # 13 tokens cached: 3 full blocks registered
    assert engine.block_manager.num_cached_blocks() == 3
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1  # cached blocks are free
    tokens = first.get_token_ids()
    (seq,) = engine.add_requests([tokens[:9] + [5, 6]], greedy(max_tokens=2))
    md = run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 8 and md.seqlens_q == [3]  # two blocks hit; the rest is computed
    assert seq.cache.block_hashes == hash_blocks(tokens, BS, 2)
    run_step(engine, [seq], is_prompt=False)


def test_same_prompt_again(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    engine.generate([[3] * 8], greedy(max_tokens=1))
    (seq,) = engine.add_requests([[3] * 8], greedy(max_tokens=1))
    md = run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 4 and md.seqlens_q == [4]


def test_hit_on_blocks_in_use_and_same_batch_duplicates(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    bm = engine.block_manager
    a, b = engine.add_requests([[7] * 9, [7] * 9], greedy(max_tokens=4))
    run_step(engine, [a, b], is_prompt=True)  # same prompt in one batch: no hits, two copies
    assert a.cached_hit_tokens == b.cached_hit_tokens == 0
    assert bm.num_cached_blocks() == 2  # the first registration of each hash wins
    assert [bm.block_hash[x] is not None for x in b.cache.block_table[:2]] == [False, False]
    (c,) = engine.add_requests([[7] * 9], greedy(max_tokens=4))
    run_step(engine, [c], is_prompt=True, everyone=[a, b, c])
    assert c.cache.block_table[:2] == a.cache.block_table[:2]  # a's blocks, still in use, are pinned
    check_ref_counts(engine, [a, b, c])
    while not all(s.is_finished() for s in (a, b, c)):
        run_step(engine, [s for s in (a, b, c) if not s.is_finished()], is_prompt=False, everyone=[a, b, c])
    assert bm.get_num_free_blocks() == bm.num_blocks - 1


def test_hits_are_taken_before_new_blocks(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=7, enable_prefix_caching=True)  # blocks 1-6
    bm = engine.block_manager
    vocab = engine.runner.model.cfg.vocab_size
    pa, pd = random_prompts([12, 8], vocab, seed=31)
    (a,) = engine.add_requests([pa], greedy(max_tokens=1))
    run_step(engine, [a], is_prompt=True)  # blocks 1-3, all registered, then freed
    (e,) = engine.add_requests([[2] * 3], greedy(max_tokens=9))
    run_step(engine, [e], is_prompt=True)  # takes block 4 and keeps it
    (d,) = engine.add_requests([pd], greedy(max_tokens=1))
    run_step(engine, [d], is_prompt=True)  # blocks 5, 6, then freed
    assert e.cache.block_table == [4]
    (b,) = engine.add_requests([pa + [1]], greedy(max_tokens=2))  # hits 1-3, needs one new block
    run_step(engine, [b], is_prompt=True, everyone=[b, e])
    assert b.cache.block_table[:3] == [1, 2, 3] and b.cached_hit_tokens == 12
    assert b.cache.block_table[3] in (5, 6)


def test_admission_with_free_hits(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=6, enable_prefix_caching=True)  # blocks 1-5
    bm = engine.block_manager
    vocab = engine.runner.model.cfg.vocab_size
    (pa,) = random_prompts([12], vocab, seed=61)
    engine.generate([pa], greedy(max_tokens=1))  # blocks 1-3 cached and free
    (x,) = engine.add_requests([[5]], greedy(max_tokens=9))
    run_step(engine, [x], is_prompt=True)  # holds block 4: 4 free, 3 of them cached
    (b,) = engine.add_requests([pa + [1] * 5], greedy())  # 3 hits + 2 new
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([b], is_prompt=True)
    assert b.cache.block_table == [] and b.status == SequenceStatus.WAITING  # nothing taken
    assert bm.get_num_free_blocks() == 4 and bm.num_cached_blocks() == 3
    (c,) = engine.add_requests([pa + [1] * 4], greedy())  # 3 hits + 1 new
    run_step(engine, [c], is_prompt=True, everyone=[x, c])
    assert c.cached_hit_tokens == 12 and bm.get_num_free_blocks() == 0


def test_lru_eviction_order(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, num_blocks=6, enable_prefix_caching=True)  # blocks 1-5
    bm = engine.block_manager
    vocab = engine.runner.model.cfg.vocab_size
    old, new = random_prompts([12, 12], vocab, seed=41)
    engine.generate([old], greedy(max_tokens=1))  # 3 cached blocks
    engine.generate([new], greedy(max_tokens=1))
    (seq,) = engine.add_requests([old + [9]], greedy(max_tokens=1))
    run_step(engine, [seq], is_prompt=True)
    assert seq.cached_hit_tokens == 8


@pytest.mark.parametrize("prefix", [False, True], ids=["no-prefix-cache", "prefix-cache"])
def test_same_tokens_with_and_without_the_cache(tiny_checkpoint, prefix):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=prefix)
    prompts = random_prompts([7, 10, 13], engine.runner.model.cfg.vocab_size, seed=51)
    first = [s.get_token_ids() for s in engine.generate(prompts, greedy(max_tokens=6))]
    again = engine.generate(prompts, greedy(max_tokens=6))
    assert [s.get_token_ids() for s in again] == first
    assert [s.cached_hit_tokens for s in again] == ([4, 8, 12] if prefix else [0, 0, 0])


def test_fork_children_inherit_the_chain(tiny_checkpoint):
    engine = engine_for(tiny_checkpoint, enable_prefix_caching=True)
    bm = engine.block_manager
    (parent,) = engine.add_requests([[4] * 10], greedy(max_tokens=6))
    run_step(engine, [parent], is_prompt=True)
    other = (parent.get_last_token_id() + 1) % engine.runner.model.cfg.vocab_size
    child = engine.fork_seq(parent, sampling_params=greedy(max_tokens=6), replace_last_token_id=other)
    assert child.cache.block_hashes == parent.cache.block_hashes == hash_blocks([4] * 10, BS, 2)
    everyone = [parent, child]
    for _ in range(4):
        run_step(engine, everyone, is_prompt=False, everyone=everyone)
    for seq in everyone:
        tokens = seq.get_token_ids()
        assert seq.cache.block_hashes == hash_blocks(tokens, BS, seq.num_cached_tokens // BS)
    assert parent.cache.block_hashes[2] != child.cache.block_hashes[2]  # their third blocks differ
    assert bm.num_cached_blocks() == 4


# --- randomized ---------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [False, True], ids=["no-prefix-cache", "prefix-cache"])
@pytest.mark.parametrize("seed", range(10))
def test_randomized_sharing(tiny_checkpoint, seed, prefix):
    """Random requests, chunked prefills, decodes, forks of every kind and
    frees; every step checked against a full recompute and the write rule,
    and the reference counts checked after every operation."""
    rng = random.Random(seed)
    engine = engine_for(tiny_checkpoint, num_blocks=rng.choice([12, 24, 60]), enable_prefix_caching=prefix)
    bm = engine.block_manager
    vocab = engine.runner.model.cfg.vocab_size
    seqs, history = [], []

    def tokens(n):
        return [rng.randrange(vocab) for _ in range(n)]

    cached_after_lookup = engine.num_cached_after_lookup

    for _ in range(90):
        live = [s for s in seqs if not s.is_finished()]
        op = rng.random()
        if op < 0.2 or not live:
            n = rng.randrange(1, 15)
            prompt = tokens(n)
            if history and rng.random() < 0.7:  # share a prefix with something seen before
                base = rng.choice(history + [s.get_token_ids() for s in seqs])
                m = min(len(base), rng.choice([4, 8, 12, rng.randrange(0, len(base) + 1)]))
                prompt = base[:m] + prompt[: max(n - m, 1)]
            (seq,) = engine.add_requests([prompt], greedy(max_tokens=rng.randrange(1, 10)))
            seqs.append(seq)
            history.append(prompt)
        elif op < 0.45:
            ready = [s for s in live if s.status == SequenceStatus.WAITING and not engine._is_prompted(s)]
            batch = rng.sample(ready, rng.randrange(1, len(ready) + 1)) if ready else []
            if batch:
                chunks = None
                if rng.random() < 0.6:
                    chunks = [rng.randrange(1, s.get_len() - cached_after_lookup(s) + 1) for s in batch]
                try:
                    run_step(engine, batch, is_prompt=True, seqlens_q=chunks, everyone=seqs)
                except RuntimeError as e:
                    assert "out of KV-cache blocks" in str(e)
                    assert all(s.status == SequenceStatus.WAITING for s in batch)
        elif op < 0.75:
            ready = [
                s
                for s in live
                if (s.status == SequenceStatus.RUNNING or engine._is_prompted(s)) and s.num_cached_tokens == s.get_len() - 1
            ]
            batch = rng.sample(ready, rng.randrange(1, len(ready) + 1)) if ready else []
            if batch:
                try:
                    run_step(engine, batch, is_prompt=False, everyone=seqs)
                except RuntimeError as e:
                    assert "out of KV-cache blocks" in str(e)
                    assert all(s.status == SequenceStatus.FINISHED_ABORTED and not s.cache.block_table for s in batch)
        elif op < 0.92:
            parents = [s for s in live if s.num_cached_tokens > 0 and s.cache.block_table]
            if parents:
                parent = rng.choice(parents)
                incomplete = parent.num_cached_tokens < parent.get_len() - 1
                child = engine.fork_seq(
                    parent,
                    None,
                    "_fork" if incomplete else "fork",
                    sampling_params=greedy(max_tokens=rng.randrange(1, 8)),
                    replace_last_token_id=rng.choice([-1, rng.randrange(vocab)]),
                )
                seqs.append(child)
                history.append(child.get_token_ids())
        else:
            victim = rng.choice(live)
            victim.status = SequenceStatus.FINISHED_ABORTED
            engine.free_seq(victim)
        check_ref_counts(engine, seqs)
        if prefix:
            for h, b in bm.cached_blocks.items():
                assert bm.block_hash[b] == h
            for s in holders(seqs):
                n = len(s.cache.block_hashes)
                assert n <= s.num_cached_tokens // BS
                assert s.cache.block_hashes == hash_blocks(s.get_token_ids(), BS, n)
    for s in holders(seqs):
        s.status = SequenceStatus.FINISHED_ABORTED
        engine.free_seq(s)
    assert engine.prompted == []
    assert bm.get_num_free_blocks() == bm.num_blocks - 1
    bm.reset()
