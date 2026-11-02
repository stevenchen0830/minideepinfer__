"""KV-cache kernels (S1.3): store, gather, prefill with history, decode."""

import pytest
import torch

from mini_deepinfer.model_executor.layers.attention_backend import (
    decode_attn,
    load_paged_kv_cache,
    prefill_attn_varlen,
    store_kv_cache,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def slots_of(block_table, positions, block_size):
    return torch.tensor([block_table[p // block_size] * block_size + p % block_size for p in positions], dtype=torch.int32)


def test_store_writes_exactly_the_given_slots():
    cache = torch.full((3, 4, 2, 8), float("nan"))
    kv = torch.randn(5, 2, 8)
    slots = torch.tensor([0, 5, 11, 6, 2], dtype=torch.int32)
    store_kv_cache(kv, cache, slots)
    flat = cache.view(-1, 2, 8)
    assert torch.equal(flat[slots.long()], kv)
    untouched = torch.ones(12, dtype=torch.bool)
    untouched[slots.long()] = False
    assert flat[untouched].isnan().all()


@pytest.mark.parametrize("slots, slot0_from", [([-1, 3, -1], None), ([-1, 0, 7, -1], 1), ([0, -1], 0)])
def test_store_skips_minus_one(slots, slot0_from):
    cache = torch.full((2, 4, 1, 4), 7.0)
    kv = torch.randn(len(slots), 1, 4)
    store_kv_cache(kv, cache, torch.tensor(slots, dtype=torch.int32))
    flat = cache.view(-1, 1, 4)
    expected0 = torch.full((1, 4), 7.0) if slot0_from is None else kv[slot0_from]
    assert torch.equal(flat[0], expected0)
    for i, s in enumerate(slots):
        if s > 0:
            assert torch.equal(flat[s], kv[i])
    touched = {s for s in slots if s >= 0} | {0}
    assert all(torch.equal(flat[s], torch.full((1, 4), 7.0)) for s in range(8) if s not in touched)


def test_store_wants_int32_slots():
    with pytest.raises(AssertionError):
        store_kv_cache(torch.randn(1, 1, 4), torch.zeros(1, 4, 1, 4), torch.tensor([0]))


def test_load_follows_the_block_table():
    bs, h, d = 4, 2, 8
    cache = torch.full((5, bs, h, d), float("nan"))
    tables = [[2, 0, 3], [4]]  # blocks need not be in order
    lens = [10, 3]
    kvs = [torch.randn(n, h, d) for n in lens]
    for table, n, kv in zip(tables, lens, kvs):
        store_kv_cache(kv, cache, slots_of(table, range(n), bs))
    block_tables = torch.tensor([[2, 0, 3], [4, 0, 0]], dtype=torch.int32)
    loaded = load_paged_kv_cache(cache, block_tables, lens)
    assert torch.equal(loaded, torch.cat(kvs))


def test_prefill_with_history_equals_the_last_rows_of_a_full_prefill():
    # Two sequences: 9 keys of which the last 3 are queries, and 5 keys all queries.
    lk, lq = [9, 5], [3, 5]
    q_full = [torch.randn(n, 4, 16) for n in lk]
    k = [torch.randn(n, 2, 16) for n in lk]
    v = [torch.randn(n, 2, 16) for n in lk]
    full = [prefill_attn_varlen(qq, kk, vv, [0, n], 0.25) for qq, kk, vv, n in zip(q_full, k, v, lk)]
    q_tail = torch.cat([qq[n - m :] for qq, n, m in zip(q_full, lk, lq)])
    out = prefill_attn_varlen(q_tail, torch.cat(k), torch.cat(v), [0, 3, 8], 0.25, [0, 9, 14])
    torch.testing.assert_close(out[:3], full[0][6:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[3:], full[1], rtol=1e-5, atol=1e-5)


def test_prefill_refuses_more_queries_than_keys():
    q, kv = torch.randn(4, 2, 8), torch.randn(3, 2, 8)
    with pytest.raises(ValueError):
        prefill_attn_varlen(q, kv, kv, [0, 4], 0.3, [0, 3])


def reference_attention(q, k, v, scale):
    """Plain math for one sequence whose queries are its last len(q) tokens."""
    lq, lk = q.shape[0], k.shape[0]
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    i, j = torch.arange(lq)[:, None], torch.arange(lk)[None, :]
    scores = scores.masked_fill(j > i + (lk - lq), float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def test_kernels_match_explicit_softmax():
    scale = 0.3  # not 1/sqrt(head_dim)
    lk, lq = [9, 5], [3, 5]
    q = [torch.randn(n, 4, 16) for n in lq]
    k = [torch.randn(n, 2, 16) for n in lk]
    v = [torch.randn(n, 2, 16) for n in lk]
    out = prefill_attn_varlen(torch.cat(q), torch.cat(k), torch.cat(v), [0, 3, 8], scale, [0, 9, 14])
    torch.testing.assert_close(out[:3], reference_attention(q[0], k[0], v[0], scale), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[3:], reference_attention(q[1], k[1], v[1], scale), rtol=1e-5, atol=1e-5)
    cache_k, cache_v = torch.zeros(3, 4, 2, 16), torch.zeros(3, 4, 2, 16)
    slots = slots_of([2, 0, 1], range(9), 4)
    store_kv_cache(k[0], cache_k, slots)
    store_kv_cache(v[0], cache_v, slots)
    out = decode_attn(q[0][-1:], cache_k, cache_v, torch.tensor([[2, 0, 1]], dtype=torch.int32), [9], scale)
    torch.testing.assert_close(out, reference_attention(q[0][-1:], k[0], v[0], scale), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_store_kv_cache_on_cuda():
    cache = torch.zeros(2, 4, 1, 8, device="cuda")
    kv = torch.randn(4, 1, 8, device="cuda")
    slots = torch.tensor([3, -1, 0, 5], dtype=torch.int32, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        store_kv_cache(kv, cache, slots)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    flat = cache.view(-1, 1, 8)
    assert torch.equal(flat[3], kv[0]) and torch.equal(flat[0], kv[2]) and torch.equal(flat[5], kv[3])


@pytest.mark.parametrize("block_size", [4, 16])
def test_decode_equals_the_last_row_of_a_full_prefill(block_size):
    lens = [1, 7, 12]
    cache_k = torch.full((8, block_size, 2, 16), float("nan"))  # unwritten slots are NaN
    cache_v = torch.full((8, block_size, 2, 16), float("nan"))
    free = list(range(8))[::-1]
    tables, queries, expected = [], [], []
    for n in lens:
        table = [free.pop() for _ in range(-(-n // block_size))]
        q, k, v = torch.randn(n, 4, 16), torch.randn(n, 2, 16), torch.randn(n, 2, 16)
        slots = slots_of(table, range(n), block_size)
        store_kv_cache(k, cache_k, slots)
        store_kv_cache(v, cache_v, slots)
        tables.append(table + [0] * (3 - len(table)))
        queries.append(q[-1])
        expected.append(prefill_attn_varlen(q, k, v, [0, n], 0.25)[-1])
    out = decode_attn(torch.stack(queries), cache_k, cache_v, torch.tensor(tables, dtype=torch.int32), lens, 0.25)
    assert out.isfinite().all()  # no NaN slot was read
    torch.testing.assert_close(out, torch.stack(expected), rtol=1e-5, atol=1e-5)
