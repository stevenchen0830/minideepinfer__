"""Prefix cache at the block level (S2.2): the hash chain, registration, hits
and eviction, then a randomized test against a reference model."""

import hashlib
import random
from array import array
from collections import deque

import pytest

from mini_deepinfer.engine.prefix_cache import NONE_HASH, PrefixCachingBlockManager, hash_block_tokens, hash_blocks


def free_list(blocks):
    out, i, prev = [], blocks.head, 0
    while i:
        assert blocks.prev[i] == prev
        out.append(i)
        prev, i = i, blocks.next[i]
        assert len(out) <= len(blocks.ref_count), "cycle in the free list"
    assert blocks.tail == prev
    return out


def H(*names):
    """Distinct fake hashes for block-level tests."""
    return [hashlib.sha256(str(n).encode()).digest() for n in names]


def test_hash_chain():
    tokens = list(range(10))
    hashes = hash_blocks(tokens, 4, 2)
    assert hashes[0] == hashlib.sha256(NONE_HASH + array("i", [0, 1, 2, 3]).tobytes()).digest()
    assert hashes[1] == hash_block_tokens(hashes[0], [4, 5, 6, 7])
    assert hash_blocks(tokens + [99], 4, 2) == hashes  # tokens past the full blocks do not matter
    changed = hash_blocks([7] + tokens[1:], 4, 2)
    assert changed[0] != hashes[0] and changed[1] != hashes[1]  # an early change changes every later hash
    assert hash_block_tokens(H("x")[0], [4, 5, 6, 7]) != hashes[1]  # same tokens, other prefix
    with pytest.raises(ValueError):
        hash_blocks(tokens, 4, 3)  # 10 tokens hold only 2 full blocks


def test_register_find_and_take_from_the_middle():
    bm = PrefixCachingBlockManager(8)  # blocks 1-7
    a = bm.allocate_nolock(3)
    h = H(0, 1, 2)
    bm.register_nolock(a[0], h[0])
    bm.register_nolock(a[1], h[1])
    assert bm.find_cached_blocks_nolock(h) == a[:2]  # stops at the first hash that is not cached
    assert bm.find_cached_blocks_nolock([h[1]]) == [a[1]]  # a lookup does not check the chain itself
    assert bm.find_cached_blocks_nolock([h[0], h[2], h[1]]) == [a[0]]  # nothing after the first miss
    bm.free_blocks(a)  # freed last block first: 3, 2, 1 at the tail
    assert free_list(bm.blocks) == [4, 5, 6, 7, 3, 2, 1]
    assert bm.find_cached_blocks_nolock(h) == [1, 2]  # free, but still cached
    assert bm.num_free_cached_nolock([1, 2, 2]) == 2
    bm.take_cached_blocks_nolock([1, 2])
    assert free_list(bm.blocks) == [4, 5, 6, 7, 3]  # taken from the middle and the tail
    assert bm.blocks.ref_count[1:3] == [1, 1]
    assert bm.block_hash[1] == h[0] and bm.block_hash[2] == h[1]  # a hit keeps the hash
    bm.take_cached_blocks_nolock([1])  # in use now: pinned
    assert bm.blocks.ref_count[1] == 2


def test_eviction_when_a_cached_block_is_handed_out_again():
    bm = PrefixCachingBlockManager(4)  # blocks 1-3
    a = bm.allocate_nolock(2)
    h = H(0, 1)
    for b, x in zip(a, h):
        bm.register_nolock(b, x)
    bm.free_blocks(a)
    assert free_list(bm.blocks) == [3, 2, 1]
    assert bm.allocate_nolock(2) == [3, 2]  # allocation evicts the hashes of the blocks it takes
    assert bm.block_hash[2] is None and bm.num_cached_blocks() == 1
    assert bm.find_cached_blocks_nolock(h) == [1]
    bm.allocate_single_nolock()  # the single-block path evicts too
    assert bm.num_cached_blocks() == 0 and bm.find_cached_blocks_nolock(h) == []


def test_first_registration_wins():
    bm = PrefixCachingBlockManager(5)
    a, b = bm.allocate_nolock(1), bm.allocate_nolock(1)
    (x,) = H(0)
    bm.register_nolock(a[0], x)
    bm.register_nolock(b[0], x)  # same content in another block: left unregistered
    assert bm.cached_blocks == {x: a[0]} and bm.block_hash[b[0]] is None
    bm.register_nolock(a[0], x)  # again with the same hash: nothing changes
    with pytest.raises(RuntimeError):
        bm.register_nolock(a[0], H(1)[0])  # a cached block never changes content
    bm.free_blocks(a)
    bm.allocate_nolock(3)  # takes 3, 4, then evicts block 1 (a)
    assert bm.find_cached_blocks_nolock([x]) == []  # b holds the content but was never registered


def test_reset_keeps_the_hashes():
    bm = PrefixCachingBlockManager(4)
    a = bm.allocate_nolock(1)
    bm.register_nolock(a[0], H(0)[0])
    bm.free_blocks(a)
    bm.reset()  # only checks that every block is free, as DeepInfer's does
    assert bm.num_cached_blocks() == 1 and bm.find_cached_blocks_nolock(H(0)) == a


def test_misuse_is_an_error():
    bm = PrefixCachingBlockManager(4)
    with pytest.raises(RuntimeError):
        bm.register_nolock(1, H(0)[0])  # a free block
    a = bm.allocate_nolock(1)
    with pytest.raises(RuntimeError):
        bm.blocks.take(a[0])  # in use, not on the free list
    with pytest.raises(RuntimeError):
        bm.blocks.take(0)  # the reserved block
    bm.free_blocks(a)
    bm.reset()  # everything is free again (cached blocks count as free)


class ReferenceModel:
    """The cache without linked lists: a deque for the free list, dicts for
    the counts and the hashes."""

    def __init__(self, n):
        self.free = deque(range(1, n))
        self.ref = {}
        self.cached = {}  # hash -> block
        self.hash_of = {}  # block -> hash

    def allocate(self, k):
        out = []
        for _ in range(k):
            b = self.free.popleft()
            if b in self.hash_of:
                del self.cached[self.hash_of.pop(b)]
            self.ref[b] = 1
            out.append(b)
        return out

    def find(self, hashes):
        out = []
        for h in hashes:
            if h not in self.cached:
                break
            out.append(self.cached[h])
        return out

    def take(self, blocks):
        for b in blocks:
            if self.ref.get(b, 0):
                self.ref[b] += 1
            else:
                self.free.remove(b)
                self.ref[b] = 1

    def register(self, b, h):
        if b in self.hash_of or h in self.cached:
            return
        self.cached[h] = b
        self.hash_of[b] = h

    def free_table(self, table):
        for b in reversed(table):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                del self.ref[b]
                self.free.append(b)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_against_reference_model(seed):
    rng = random.Random(seed)
    n = rng.choice([3, 8, 33])
    bm, ref = PrefixCachingBlockManager(n), ReferenceModel(n)
    # Each owner: (table, chain) where chain[i] is the hash its block i holds,
    # or None while the block is not "full" yet.
    owners = []
    names = iter(range(10**9))

    def chain_hashes(length):
        """A chain that shares a random prefix with an existing owner, so hits happen."""
        donors = [c for _, c in owners if c and c[0] is not None]
        prefix = []
        if donors and rng.random() < 0.7:
            donor = rng.choice(donors)
            for h in donor:
                if h is None or rng.random() < 0.2:
                    break
                prefix.append(h)
        return (prefix + H(*(next(names) for _ in range(length))))[:length]

    for _ in range(300):
        op = rng.random()
        if op < 0.35:  # a new request: hits first, then new blocks
            length = rng.randrange(1, max(2, n // 2))
            hashes = chain_hashes(length)
            hits = bm.find_cached_blocks_nolock(hashes)
            assert hits == ref.find(hashes)
            need = length - len(hits) + bm.num_free_cached_nolock(hits)
            if need <= bm.get_num_free_blocks():
                bm.take_cached_blocks_nolock(hits)
                ref.take(hits)
                new = bm.allocate_nolock(length - len(hits))
                assert new == ref.allocate(length - len(hits))  # same ids, same evictions
                owners.append((hits + new, hashes[: len(hits)] + [None] * len(new)))
        elif op < 0.6 and owners:  # a step fills some blocks: register them in order
            table, chain = rng.choice(owners)
            if None in chain:
                i = chain.index(None)
                full = H(next(names)) if rng.random() < 0.5 else [chain_hashes(i + 1)[i]]
                chain[i] = full[0]
                bm.register_nolock(table[i], chain[i])
                ref.register(table[i], chain[i])
        elif op < 0.7 and owners:  # a fork pins an owner's blocks
            table, chain = rng.choice(owners)
            bm.pin_blocks(table)
            ref.take(table)
            owners.append((list(table), chain))  # same blocks, same content: one chain for both
        elif owners:
            table, _ = owners.pop(rng.randrange(len(owners)))
            bm.free_blocks(table)
            ref.free_table(table)
        # Invariants after every operation.
        assert free_list(bm.blocks) == list(ref.free)
        assert bm.get_num_free_blocks() == len(ref.free)
        for b in range(1, n):
            assert bm.blocks.ref_count[b] == ref.ref.get(b, 0)
            assert bm.block_hash[b] == ref.hash_of.get(b)
        assert bm.cached_blocks == ref.cached
        assert all(bm.block_hash[b] == h for h, b in bm.cached_blocks.items())  # the two maps agree
        assert bm.block_hash[0] is None and bm.blocks.ref_count[0] == 0
        assert sum(len(t) for t, _ in owners) == sum(bm.blocks.ref_count)
    for table, _ in owners:
        bm.free_blocks(table)
    bm.reset()
