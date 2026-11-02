"""Hash-chain prefix cache (S2.2). Our own component.

DeepInfer keeps its prefix cache outside the engine and has not released it
(doc/inference_deployment_qa.md, "KV cache storage"): the entrypoint matches a
new prompt against a KV store with "a hash-chain prefix match over metadata
only", the matched 128-token blocks are fetched from the 3FS cluster into host
memory, and a thread in the engine copies them into fresh GPU blocks. Inside
the engine a freed block's content is dead. This module keeps the same hash
chain but stores nothing outside the GPU: a freed block keeps its KV and its
hash until the free list hands it out again. That is how vLLM's automatic
prefix caching works (read at v0.11.0: vllm/v1/core/block_pool.py,
kv_cache_utils.py).

The chain
    The hash of block i is ``sha256(hash of block i-1 || token ids of block
    i)``, with a fixed seed as the parent of block 0. Only full blocks get a
    hash. Two sequences with the same first i blocks of tokens therefore get
    the same i hashes, and one hash names a whole prefix, not just a block.

What is kept
    ``hash -> block id`` and, for each block, its hash or None. A block is
    registered after the step that fills it (``register_nolock``). It stays
    registered while it is in use and after it is freed. It loses its hash
    only when ``allocate`` takes it off the free list for new content
    (eviction). Which block is evicted therefore depends on the order of the
    free list (S2.1).

A hit
    ``find_cached_blocks_nolock(hashes)`` returns the blocks of the longest
    run of leading hashes that are registered. ``take_cached_blocks_nolock``
    then gives the new owner a reference to each: a block in use is pinned; a
    free block is taken off the free list wherever it is (this is what the
    ``prev`` links of S2.1 are for) and gets count 1.

Differences from vLLM (v0.11.0):
- A block is registered after the step that wrote it. vLLM registers it in
  ``allocate_slots``, before the forward, as soon as the step will fill it,
  so a later request in the same schedule can hit it. Here two identical
  prompts in one step both compute everything.
- One block per hash: the first block registered wins. A later block with the
  same content stays unregistered and is never hit, and blocks its holder
  registers later chain from the first block's hash; once that block is
  evicted they cannot be hit until it is computed again. vLLM keeps every
  block under its hash. A decode fork's copy of a full block is the same case
  (it keeps its parent's hash).
- The hash covers the tokens only (no LoRA, multimodal or salt keys).
- Every freed block goes to the tail, as in vLLM v0.11.0; newer vLLM puts a
  freed block with no hash at the head, so it is reused before cached ones.
- There is no ``reset_prefix_cache``: ``reset()`` (inherited) only checks
  that every block is free, as DeepInfer's does, and every hash survives it.
  A new engine starts with an empty cache.
- Lookup rules left out until the features exist: vLLM skips the lookup for a
  request with prompt logprobs (hit tokens get no logits) and drops the last
  hit block for EAGLE/MTP drafting heads. DeepInfer's published hit policy
  (doc/inference_deployment_qa.md) also leaves room for speculative tokens.

``BlockManager`` (block_manager.py) is unchanged: the cache subclasses it and
its ``Blocks``.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Iterable, Sequence

from mini_deepinfer.engine.block_manager import BlockManager, Blocks

# Parent of the first block. Fixed, so the hashes are the same in every
# process (vLLM v0.11.0's seed is random unless PYTHONHASHSEED is set; newer
# versions use a fixed seed for sha256).
NONE_HASH = hashlib.sha256(b"mini-deepinfer prefix cache").digest()


def hash_block_tokens(parent_hash: bytes, token_ids: Sequence[int]) -> bytes:
    """Chain hash of one full block: sha256(parent hash || int32 token ids)."""
    return hashlib.sha256(parent_hash + array("i", token_ids).tobytes()).digest()


def hash_blocks(token_ids: Sequence[int], block_size: int, num_blocks: int, parent_hash: bytes = NONE_HASH) -> list[bytes]:
    """Chain hashes of the first ``num_blocks`` full blocks of ``token_ids``."""
    if num_blocks * block_size > len(token_ids):
        raise ValueError(f"{len(token_ids)} tokens hold fewer than {num_blocks} full blocks of {block_size}")
    out = []
    for i in range(num_blocks):
        parent_hash = hash_block_tokens(parent_hash, token_ids[i * block_size : (i + 1) * block_size])
        out.append(parent_hash)
    return out


class CachingBlocks(Blocks):
    """``Blocks`` plus the two operations the cache needs: take one given free
    block, and forget a block's hash when it is handed out for new content."""

    def __init__(self, n: int, cache: PrefixCachingBlockManager) -> None:
        super().__init__(n)
        self._cache = cache

    def allocate(self) -> int:
        i = super().allocate()
        self._cache._evict(i)
        return i

    def take(self, block_id: int) -> None:
        """Take a free block off the list, wherever it is, with count 1."""
        self._check_id(block_id)
        if self.ref_count[block_id] != 0 or (self.head != block_id and self.prev[block_id] == 0):
            raise RuntimeError(f"block {block_id} is not on the free list")
        self._erase(block_id)
        self.ref_count[block_id] = 1


class PrefixCachingBlockManager(BlockManager):
    def __init__(self, num_blocks: int) -> None:
        super().__init__(num_blocks)
        self.blocks = CachingBlocks(num_blocks, self)
        self.cached_blocks: dict[bytes, int] = {}  # hash -> block id
        self.block_hash: list[bytes | None] = [None] * num_blocks  # block id -> hash

    def _evict(self, block_id: int) -> None:
        h = self.block_hash[block_id]
        if h is not None:
            del self.cached_blocks[h]
            self.block_hash[block_id] = None

    def find_cached_blocks_nolock(self, hashes: Iterable[bytes]) -> list[int]:
        """Blocks of the longest run of leading hashes that are cached."""
        out = []
        for h in hashes:
            block_id = self.cached_blocks.get(h)
            if block_id is None:
                break
            out.append(block_id)
        return out

    def num_free_cached_nolock(self, block_ids: Iterable[int]) -> int:
        """How many of these (distinct) blocks are on the free list."""
        return sum(self.blocks.ref_count[b] == 0 for b in set(block_ids))

    def take_cached_blocks_nolock(self, block_ids: Iterable[int]) -> None:
        """One more reference to each block: pin it if in use, else take it
        off the free list."""
        for block_id in block_ids:
            if self.blocks.ref_count[block_id] > 0:
                self.blocks.pin_block(block_id)
            else:
                self.blocks.take(block_id)

    def register_nolock(self, block_id: int, block_hash: bytes) -> None:
        """Record that ``block_id`` holds the full block named ``block_hash``.
        A block registered with this hash already, or a hash that another
        block holds, is left as it is (the first block wins)."""
        if self.blocks.ref_count[block_id] <= 0:
            raise RuntimeError(f"registering free block {block_id}")
        old = self.block_hash[block_id]
        if old is not None:
            if old != block_hash:
                raise RuntimeError(f"block {block_id} is already cached under another hash")
            return
        if block_hash in self.cached_blocks:
            return
        self.cached_blocks[block_hash] = block_id
        self.block_hash[block_id] = block_hash

    def num_cached_blocks(self) -> int:
        return len(self.cached_blocks)
