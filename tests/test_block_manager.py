"""Block manager (S2.1): unit tests for each rule, then a randomized test
against a reference model in plain Python (DV style: the same stimulus goes to
the design and to the model, and every observable output is compared)."""

import random
from collections import deque

import pytest

from mini_deepinfer.engine.block_manager import AllocStatus, BlockManager, BlockManagerMutexAcquire, Blocks
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import Sequence, SequenceStatus


def free_list(blocks: Blocks) -> list[int]:
    """Walk the free list from head to tail, checking the links on the way."""
    out, i, prev = [], blocks.head, 0
    while i:
        assert blocks.prev[i] == prev, f"block {i}: prev link {blocks.prev[i]}, expected {prev}"
        out.append(i)
        prev, i = i, blocks.next[i]
        assert len(out) <= len(blocks.ref_count), "cycle in the free list"
    assert blocks.tail == prev
    return out


def test_initial_state():
    bm = BlockManager(5)
    assert bm.get_num_free_blocks() == 4
    assert free_list(bm.blocks) == [1, 2, 3, 4]  # block 0 is never on the list
    with pytest.raises(RuntimeError):
        BlockManager(1)


def test_fifo_and_reverse_free():
    bm = BlockManager(8)
    a = bm.allocate_nolock(3)
    b = bm.allocate_nolock(2)
    assert (a, b) == ([1, 2, 3], [4, 5])
    bm.free_blocks(a)
    assert free_list(bm.blocks) == [6, 7, 3, 2, 1]  # appended at the tail, last block first
    assert bm.allocate_nolock(4) == [6, 7, 3, 2]


def test_can_allocate_boundaries():
    bm = BlockManager(5)  # 4 usable blocks
    assert bm.can_allocate_nolock(4) == AllocStatus.OK
    assert bm.can_allocate_nolock(5) == AllocStatus.NEVER
    bm.allocate_nolock(2)
    assert bm.can_allocate_nolock(2) == AllocStatus.OK
    assert bm.can_allocate_nolock(3) == AllocStatus.LATER
    assert bm.can_allocate_nolock(4) == AllocStatus.LATER
    assert bm.can_allocate_single_nolock()
    bm.allocate_nolock(1)
    assert bm.can_allocate_single_nolock()  # exactly one free
    bm.allocate_nolock(1)
    assert not bm.can_allocate_single_nolock()
    with pytest.raises(RuntimeError):
        bm.allocate_single_nolock()


def test_pin_keeps_a_block_until_every_owner_frees_it():
    bm = BlockManager(4)
    table = bm.allocate_nolock(2)
    bm.pin_blocks(table)  # a second owner
    assert bm.blocks.ref_count[1:3] == [2, 2]
    bm.unpin_blocks(table)
    assert bm.get_num_free_blocks() == 1  # still held by the first owner
    bm.free_blocks(table)
    assert bm.get_num_free_blocks() == 3
    bm.reset()  # passes: everything is free again


def test_misuse_is_an_error():
    bm = BlockManager(4)
    with pytest.raises(RuntimeError):
        bm.pin_block(1)  # free block
    table = bm.allocate_nolock(1)
    bm.free_blocks(table)
    before = (list(bm.blocks.ref_count), list(bm.blocks.prev), list(bm.blocks.next), bm.get_num_free_blocks())
    with pytest.raises(RuntimeError):
        bm.free_blocks(table)  # double free
    assert (list(bm.blocks.ref_count), list(bm.blocks.prev), list(bm.blocks.next), bm.get_num_free_blocks()) == before
    with pytest.raises(RuntimeError):
        bm.free_blocks([0])  # the reserved block
    with pytest.raises(RuntimeError):
        bm.pin_block(4)  # out of range
    bm.allocate_nolock(1)
    with pytest.raises(RuntimeError):
        bm.reset()  # a block is still in use


def make_seq(n_tokens, block_size=4):
    sp = SamplingParams(temperature=0.0)
    return Sequence(seq_id=0, prompt_token_ids=[1] * n_tokens, block_size=block_size, request_id="r", sampling_params=sp, arrival_time=0.0)


def test_try_allocate_seq():
    bm = BlockManager(5)  # 4 usable
    seq = make_seq(9)  # 3 blocks of 4
    with BlockManagerMutexAcquire(bm) as acquire:
        assert acquire.try_allocate_seq(seq)
        assert seq.cache.block_table == [1, 2, 3]
        other = make_seq(5)  # 2 blocks, 1 free
        assert not acquire.try_allocate_seq(other)  # LATER: nothing taken
        assert other.cache.block_table == [] and bm.get_num_free_blocks() == 1
        with pytest.raises(RuntimeError):
            acquire.try_allocate_seq(make_seq(17))  # NEVER
        with pytest.raises(RuntimeError):
            acquire.try_allocate_seq(seq)  # table not empty
        for status in (SequenceStatus.RUNNING, SequenceStatus.FINISHED_IGNORED):
            other = make_seq(1)
            other.status = status
            with pytest.raises(RuntimeError):
                acquire.try_allocate_seq(other)  # not WAITING
        assert bm.get_num_free_blocks() == 1


def test_mutex_acquire_holds_the_lock():
    bm = BlockManager(3)
    with BlockManagerMutexAcquire(bm):
        assert not bm._lock.acquire(blocking=False)
    assert bm._lock.acquire(blocking=False)
    bm._lock.release()
    with pytest.raises(RuntimeError), BlockManagerMutexAcquire(bm) as acquire:
        acquire.try_allocate_seq(make_seq(100))  # an error inside the with
    assert bm._lock.acquire(blocking=False)  # released anyway
    bm._lock.release()


def test_lock_misuse_raises_instead_of_hanging():
    bm = BlockManager(3)
    table = bm.allocate_nolock(1)
    with BlockManagerMutexAcquire(bm):
        with pytest.raises(RuntimeError, match="already holds"):
            bm.free_blocks(table)  # a locking call while holding the lock
    with pytest.raises(RuntimeError, match="does not hold"):
        bm.unlock()


class ReferenceModel:
    """What the block manager should do, without any linked list."""

    def __init__(self, n):
        self.free = deque(range(1, n))
        self.ref = {}

    def allocate(self, k):
        out = [self.free.popleft() for _ in range(k)]
        for b in out:
            self.ref[b] = 1
        return out

    def pin(self, table):
        for b in table:
            self.ref[b] += 1

    def free_table(self, table):
        for b in reversed(table):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                del self.ref[b]
                self.free.append(b)

    def status(self, k, n):
        if k >= n:
            return AllocStatus.NEVER
        return AllocStatus.OK if len(self.free) >= k else AllocStatus.LATER


@pytest.mark.parametrize("seed", range(20))
def test_randomized_against_reference_model(seed):
    rng = random.Random(seed)
    n = rng.choice([2, 3, 8, 33, 129])
    bm, ref = BlockManager(n), ReferenceModel(n)
    tables = []  # each entry: one owner's list of block ids
    for _ in range(400):
        op = rng.random()
        if op < 0.4:
            k = rng.randrange(0, n + 2)
            assert bm.can_allocate_nolock(k) == ref.status(k, n)
            if ref.status(k, n) == AllocStatus.OK:
                table = bm.allocate_nolock(k)
                assert table == ref.allocate(k)  # same ids, same order
                tables.append(table)
        elif op < 0.55 and tables:
            table = rng.choice(tables)  # a fork shares an owner's blocks
            bm.pin_blocks(table)
            ref.pin(table)
            tables.append(list(table))
        elif tables:
            table = tables.pop(rng.randrange(len(tables)))
            bm.free_blocks(table)
            ref.free_table(table)
        # Invariants, checked after every operation.
        blocks = bm.blocks
        assert free_list(blocks) == list(ref.free)
        assert bm.get_num_free_blocks() == len(ref.free)
        assert blocks.ref_count[0] == 0 and 0 not in ref.ref
        for b in range(1, n):
            assert blocks.ref_count[b] == ref.ref.get(b, 0)
            in_list = b in ref.free
            assert in_list == (blocks.ref_count[b] == 0)  # free <=> on the list, exactly once
            if not in_list:
                assert blocks.prev[b] == blocks.next[b] == 0
        owners = sum(len(t) for t in tables)
        assert owners == sum(blocks.ref_count)  # every reference belongs to some owner
    for table in tables:
        bm.free_blocks(table)
        ref.free_table(table)
    bm.reset()
    assert free_list(bm.blocks) == list(ref.free) and bm.get_num_free_blocks() == n - 1
