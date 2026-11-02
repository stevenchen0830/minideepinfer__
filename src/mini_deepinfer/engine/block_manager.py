"""KV-cache block manager (S2.1).

Mirrors csrc/engine/block_manager.hpp, which DeepInfer binds to Python as
``deepinfer._C.engine.BlockManager``:

``Blocks``
    The free list and the reference counts. Block ids run from 0 to
    ``num_blocks - 1``, but block 0 is never handed out: it is a dummy block
    (CUDA-graph padding rows write their K/V there), and 0 doubles as the
    "no block" value in the list links. The free list is a doubly linked list
    kept in the ``prev``/``next`` arrays; it starts as 1, 2, ..., n-1.
    ``allocate`` takes the head and sets its count to 1; ``free`` lowers the
    count and puts the block back at the tail when it reaches 0; ``pin_block``
    raises the count of a block that is already in use.

``BlockManager``
    ``can_allocate_nolock(k)`` answers NEVER (``k >= num_blocks``), OK (enough
    free blocks) or LATER. ``free_blocks_nolock`` frees a block table last
    block first. ``pin_blocks``/``unpin_blocks`` let a second owner share
    blocks (fork, S2.2). The ``_nolock`` methods assume the caller holds the
    lock; the others take it themselves.

``BlockManagerMutexAcquire``
    Holds the lock for as long as it lives (DeepInfer: one whole ``_schedule``
    pass, and ``free_seq``). Through it a sequence gets its blocks:
    ``allocate_seq`` (the caller has checked there is room) and
    ``try_allocate_seq`` (checks first; False means LATER, an error means
    NEVER). DeepInfer's scheduler uses ``allocate_seq`` after its own check;
    ``try_allocate_seq`` is bound for Python callers.

Differences from DeepInfer:
- The lock is a ``threading.Lock`` that behaves like DeepInfer's
  error-checking pthread mutex: locking it again from the thread that holds
  it raises at once, waiting longer than 20 s raises, and unlocking from a
  thread that does not hold it raises. ``BlockManagerMutexAcquire`` is a
  context manager instead of an RAII guard.
- Errors are RuntimeError (DeepInfer: ``TORCH_CHECK``).
- No ``state_manager``: DeepInfer gives DeepSeek-V4 sequences a second
  manager for their sliding-window state slots; Qwen3 has none.
- The scheduler-side helpers that use it (``_append_slot``, ``free_seq``,
  ``get_gpu_cache_usage``) live in ``static_engine.py`` until the scheduler
  arrives in S2.5.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Iterable

from mini_deepinfer.sequence import Sequence, SequenceStatus


class AllocStatus(enum.Enum):
    OK = 0
    LATER = 1
    NEVER = 2


class Blocks:
    def __init__(self, n: int) -> None:
        if n <= 1:
            raise RuntimeError(f"too few GPU blocks: {n}")
        self.ref_count = [0] * n
        self.prev = [0] * n
        self.next = [0] * n
        for i in range(1, n):
            self.prev[i] = i - 1
            self.next[i] = i + 1
        self.next[n - 1] = 0
        self.head, self.tail, self.size = 1, n - 1, n - 1

    def reset(self) -> None:
        """Check that every block is free again (DeepInfer's reset only checks)."""
        if self.size != len(self.ref_count) - 1:
            raise RuntimeError("still some blocks in use when reset")

    def _check_id(self, i: int) -> None:
        if not 0 < i < len(self.ref_count):
            raise RuntimeError(f"block id {i} out of range 1..{len(self.ref_count) - 1}")

    def _push_back(self, i: int) -> None:
        self._check_id(i)
        if self.next[i] != 0 or self.prev[i] != 0:
            raise RuntimeError(f"block {i} is already linked")
        if self.ref_count[i] != 0:
            raise RuntimeError("only a free block can be pushed back")
        self.prev[i] = self.tail
        if self.tail:
            self.next[self.tail] = i
        else:
            self.head = i
        self.tail = i
        self.size += 1

    def _erase(self, i: int) -> None:
        self._check_id(i)
        if self.ref_count[i] != 0:
            raise RuntimeError("free block list corrupted: ref_count != 0")
        if self.next[i]:
            self.prev[self.next[i]] = self.prev[i]
        else:
            self.tail = self.prev[i]
        if self.prev[i]:
            self.next[self.prev[i]] = self.next[i]
        else:
            self.head = self.next[i]
        self.prev[i] = self.next[i] = 0
        self.size -= 1

    def _pop_front(self) -> int:
        if self.head == 0:
            raise RuntimeError("free list is empty, but pop_front is called")
        i = self.head
        self._erase(i)
        return i

    def num_free_blocks(self) -> int:
        return self.size

    def allocate(self) -> int:
        i = self._pop_front()
        self.ref_count[i] += 1
        return i

    def pin_block(self, block_id: int) -> None:
        self._check_id(block_id)
        if self.ref_count[block_id] <= 0:
            raise RuntimeError(f"try to pin free block {block_id}")
        self.ref_count[block_id] += 1

    def free(self, block_id: int) -> None:
        self._check_id(block_id)
        if self.ref_count[block_id] <= 0:
            raise RuntimeError(f"ref_count corrupted: freeing free block {block_id}")
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self._push_back(block_id)


class BlockManager:
    LOCK_TIMEOUT_S = 20

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.blocks = Blocks(num_blocks)
        self._lock = threading.Lock()
        self._owner: int | None = None

    def lock(self) -> None:
        me = threading.get_ident()
        if self._owner == me:
            raise RuntimeError("failed to lock mutex: this thread already holds it")
        if not self._lock.acquire(timeout=self.LOCK_TIMEOUT_S):
            raise RuntimeError(f"failed to lock mutex: timed out after {self.LOCK_TIMEOUT_S} s")
        self._owner = me

    def unlock(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("failed to unlock mutex: this thread does not hold it")
        self._owner = None
        self._lock.release()

    def __enter__(self) -> BlockManager:
        self.lock()
        return self

    def __exit__(self, *exc) -> None:
        self.unlock()

    def reset(self) -> None:
        with self:
            self.blocks.reset()

    def can_allocate_nolock(self, to_allocate_blocks: int) -> AllocStatus:
        if to_allocate_blocks >= self.num_blocks:
            return AllocStatus.NEVER
        if self.blocks.num_free_blocks() >= to_allocate_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate_nolock(self, to_allocate_blocks: int) -> list[int]:
        return [self.blocks.allocate() for _ in range(to_allocate_blocks)]

    def free_blocks_nolock(self, block_ids: Iterable[int]) -> None:
        for block_id in reversed(list(block_ids)):
            self.blocks.free(block_id)

    def free_blocks(self, block_ids: Iterable[int]) -> None:
        with self:
            self.free_blocks_nolock(block_ids)

    unpin_blocks = free_blocks  # the Python binding's name

    def pin_blocks_nolock(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self.blocks.pin_block(block_id)

    def pin_blocks(self, block_ids: Iterable[int]) -> None:
        with self:
            self.pin_blocks_nolock(block_ids)

    def pin_block(self, block_id: int) -> None:
        with self:
            self.blocks.pin_block(block_id)

    def can_allocate_single_nolock(self) -> bool:
        return self.blocks.num_free_blocks() >= 1

    def allocate_single_nolock(self) -> int:
        return self.blocks.allocate()

    def get_num_free_blocks(self) -> int:
        return self.blocks.num_free_blocks()


class BlockManagerMutexAcquire:
    """``with BlockManagerMutexAcquire(block_manager) as acquire: ...``"""

    def __init__(self, block_manager: BlockManager) -> None:
        self.block_manager = block_manager

    def __enter__(self) -> BlockManagerMutexAcquire:
        self.block_manager.lock()
        return self

    def __exit__(self, *exc) -> None:
        self.block_manager.unlock()

    def allocate_seq(self, seq: Sequence) -> None:
        """Give ``seq`` blocks for all its tokens. The caller has checked there is room."""
        seq.cache.block_table = self.block_manager.allocate_nolock(seq.get_num_blocks())

    def try_allocate_seq(self, seq: Sequence) -> bool:
        if seq.status != SequenceStatus.WAITING:
            raise RuntimeError(f"seq {seq.seq_id} is {seq.status.name}, not WAITING")
        if seq.cache.block_table:
            raise RuntimeError(f"seq {seq.seq_id}: block table not empty")
        num_blocks = seq.get_num_blocks()
        status = self.block_manager.can_allocate_nolock(num_blocks)
        if status != AllocStatus.OK:
            if status == AllocStatus.NEVER:
                raise RuntimeError(f"can never allocate {num_blocks} blocks")
            return False
        self.allocate_seq(seq)
        return True
