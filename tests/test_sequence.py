"""Tests for SequenceStatus, SequenceData, KvCacheState and Sequence (S1.1)."""

import random
import struct

import numpy as np
import pytest

from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import (
    KvCacheState,
    Sequence,
    SequenceData,
    SequenceStatus,
    get_finished_reason,
    is_finished,
)


def make_seq(prompt=(1, 2, 3), block_size=16, **kwargs):
    # Same settings as DeepInfer's tests/engine/test_sequence.py helper.
    sp = SamplingParams(max_tokens=4, temperature=0.7, top_k=8, ignore_eos=True)
    sp.eos_id = 2
    args = dict(
        seq_id=7,
        prompt_token_ids=prompt,
        block_size=block_size,
        request_id="request",
        sampling_params=sp,
        arrival_time=1.5,
    )
    args.update(kwargs)
    return Sequence(**args)


# --- SequenceStatus ---------------------------------------------------------


def test_status_values_match_deepinfer():
    names = [s.name for s in sorted(SequenceStatus)]
    assert names == [
        "WAITING",
        "RUNNING",
        "SWAPPED",
        "FINISHED_STOPPED",
        "FINISHED_LENGTH_CAPPED",
        "FINISHED_ABORTED",
        "FINISHED_IGNORED",
        "FINISHED_PREEMPTED",
        "FINISHED_SO_FAILED",
        "FINISHED_VE_FAILED",
    ]
    assert [int(s) for s in sorted(SequenceStatus)] == list(range(10))


@pytest.mark.parametrize("status", list(SequenceStatus))
def test_is_finished_boundary(status):
    assert is_finished(status) == (status.value >= 3)


def test_finished_reasons():
    S = SequenceStatus
    assert {s: get_finished_reason(s) for s in S} == {
        S.WAITING: "",
        S.RUNNING: "",
        S.SWAPPED: "",
        S.FINISHED_STOPPED: "stop",
        S.FINISHED_LENGTH_CAPPED: "length",
        S.FINISHED_ABORTED: "abort",
        S.FINISHED_IGNORED: "prompt_length_exceeded",
        S.FINISHED_PREEMPTED: "insufficient_system_resource",
        S.FINISHED_SO_FAILED: "xgrammar runtime errors or timeout",
        S.FINISHED_VE_FAILED: "ve_error",
    }
    # Every terminal status has a non-empty reason.
    for status in S:
        assert (get_finished_reason(status) != "") == is_finished(status)


# --- SequenceData -----------------------------------------------------------


def test_token_round_trip_from_list():
    data = SequenceData([1, 2, 3])
    assert data.get_token_ids() == [1, 2, 3]
    assert data.get_prompt_len() == 3
    assert data.get_output_len() == 0
    assert data.get_last_token_id() == 3


def test_token_round_trip_from_int32_array():
    data = SequenceData(np.asarray([1, 2, 3], dtype=np.int32))
    assert data.get_token_ids() == [1, 2, 3]
    assert all(type(t) is int for t in data.get_token_ids())


def test_token_round_trip_from_buffer_view():
    view = np.frombuffer(struct.pack("3i", 1, 2, 3), dtype=np.int32)
    assert SequenceData(view).get_token_ids() == [1, 2, 3]


def test_empty_prompt():
    data = SequenceData(np.empty(0, dtype=np.int32))
    assert data.get_prompt_len() == 0
    assert data.get_token_ids() == []
    with pytest.raises(IndexError):
        data.get_last_token_id()


@pytest.mark.parametrize("bad", [[1, 2.0, 3], ["1"], [None]])
def test_non_integer_tokens_raise(bad):
    with pytest.raises(TypeError):
        SequenceData(bad)


def test_prompt_is_copied():
    prompt = [5, 6]
    data = SequenceData(prompt)
    prompt.append(7)
    assert data.get_token_ids() == [5, 6]


def test_get_token_ids_returns_a_copy():
    data = SequenceData([1, 2, 3])
    ids = data.get_token_ids()
    ids.append(99)
    ids.clear()
    assert data.get_token_ids() == [1, 2, 3]
    assert data.get_output_len() == 0
    assert data.get_token_ids() is not data.get_token_ids()


def test_append_counts_as_output():
    data = SequenceData([1, 2])
    data.append_token_id(9)
    data.append_token_id(np.int64(10))
    assert data.get_token_ids() == [1, 2, 9, 10]
    assert data.get_prompt_len() == 2
    assert data.get_output_len() == 2
    assert data.get_last_token_id() == 10
    with pytest.raises(TypeError):
        data.append_token_id(1.0)


# --- KvCacheState -----------------------------------------------------------


def test_block_size_must_be_positive():
    for bad in (0, -16):
        with pytest.raises(ValueError):
            KvCacheState(bad)


@pytest.mark.parametrize("bad", [16.0, 1.5, "16"])
def test_block_size_must_be_an_integer(bad):
    with pytest.raises(TypeError):
        KvCacheState(bad)


def test_numpy_block_size_becomes_int():
    seq = make_seq(prompt=range(17), block_size=np.int64(16))
    assert type(seq.block_size) is int
    assert type(seq.get_num_blocks()) is int
    assert seq.get_num_blocks() == 2


def test_advance_and_commit_arithmetic():
    cache = KvCacheState(16)
    cache.advance_optimistically(5)
    assert (cache.previous_num_tokens, cache.num_tokens) == (0, 5)
    cache.advance_optimistically(4)
    assert (cache.previous_num_tokens, cache.num_tokens) == (5, 9)
    cache.commit_processed_tokens(2)
    assert (cache.previous_num_tokens, cache.num_tokens) == (5, 7)
    cache.commit_processed_tokens(4)
    assert (cache.previous_num_tokens, cache.num_tokens) == (5, 9)


def test_set_num_tokens_moves_baseline():
    cache = KvCacheState(16)
    cache.set_num_tokens(32)
    assert (cache.previous_num_tokens, cache.num_tokens) == (32, 32)
    cache.commit_processed_tokens(1)
    assert cache.num_tokens == 33


def test_cache_reset_keeps_block_table():
    cache = KvCacheState(16)
    cache.block_table = [4, 9]
    cache.advance_optimistically(20)
    cache.reset_for_recompute()
    assert (cache.previous_num_tokens, cache.num_tokens) == (0, 0)
    assert cache.block_table == [4, 9]


# --- Sequence ---------------------------------------------------------------


def test_new_sequence():
    seq = make_seq()
    assert seq.seq_id == 7
    assert seq.request_id == "request"
    assert seq.arrival_time == 1.5
    assert seq.status == SequenceStatus.WAITING
    assert not seq.is_finished()
    assert seq.get_token_ids() == [1, 2, 3]
    assert seq.get_len() == seq.get_prompt_len() == 3
    assert seq.get_output_len() == 0
    assert seq.block_size == 16
    assert seq.num_cached_tokens == 0
    assert seq.cache.block_table == []
    assert seq.sampling_params.eos_id == 2


def test_constructor_is_keyword_only():
    with pytest.raises(TypeError):
        Sequence(7, [1], 16, "r", SamplingParams(), 0.0)  # type: ignore[misc]


@pytest.mark.parametrize("kwargs", [{"max_tokens": 1.5}, {"max_tokens": 2.0}, {"top_k": 8.5}])
def test_float_counts_rejected_when_building_sequence(kwargs):
    sp = SamplingParams(**kwargs)
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        make_seq(sampling_params=sp)


def test_greedy_forcing_clears_float_top_k():
    # temperature=0 resets top_k to -1 before the Sequence checks it.
    sp = SamplingParams(temperature=0.0, top_k=8.5)
    assert make_seq(sampling_params=sp).sampling_params.top_k == -1


def test_sampling_params_is_read_only():
    seq = make_seq()
    with pytest.raises(AttributeError):
        seq.sampling_params = SamplingParams()  # type: ignore[misc]


def test_num_cached_tokens_alias():
    seq = make_seq()
    seq.num_cached_tokens = 2
    assert seq.cache.num_tokens == 2
    assert seq.old_cached_tokens == 2


@pytest.mark.parametrize(
    "length, blocks",
    [(1, 1), (15, 1), (16, 1), (17, 2), (32, 2), (33, 3)],
)
def test_get_num_blocks(length, blocks):
    seq = make_seq(prompt=range(length))
    assert seq.get_num_blocks() == blocks


def test_empty_sequence_needs_no_blocks():
    assert make_seq(prompt=[]).get_num_blocks() == 0


def test_status_drives_is_finished():
    seq = make_seq()
    seq.status = SequenceStatus.RUNNING
    assert not seq.is_finished()
    seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
    assert seq.is_finished()


def test_reset_for_recompute():
    seq = make_seq()
    seq.status = SequenceStatus.RUNNING
    seq.cache.block_table = [3]
    seq.cache.advance_optimistically(3)
    seq.append_token_id(42)
    seq.cache.advance_optimistically(1)
    seq.reset_for_recompute()
    assert seq.num_cached_tokens == 0
    assert seq.old_cached_tokens == 0
    # Tokens, status and blocks are left for the scheduler to handle.
    assert seq.get_token_ids() == [1, 2, 3, 42]
    assert seq.get_output_len() == 1
    assert seq.status == SequenceStatus.RUNNING
    assert seq.cache.block_table == [3]


def test_randomized_against_plain_lists():
    """DV-style check: drive a Sequence with random operations and compare
    every observable value against a plain-Python model of what it should be."""
    rng = random.Random(2026)
    for _ in range(200):
        block_size = rng.choice([1, 2, 16, 128])
        prompt = [rng.randrange(151936) for _ in range(rng.randrange(0, 300))]
        seq = make_seq(prompt=prompt, block_size=block_size)
        tokens = list(prompt)
        cached = 0
        for _ in range(rng.randrange(0, 60)):
            op = rng.random()
            if op < 0.4:
                t = rng.randrange(151936)
                seq.append_token_id(t)
                tokens.append(t)
            elif op < 0.8:
                # One model step over every token that has no KV yet. Without
                # draft tokens, committing the same count changes nothing.
                q = len(tokens) - cached
                seq.cache.advance_optimistically(q)
                if rng.random() < 0.5:
                    seq.cache.commit_processed_tokens(q)
                cached = len(tokens)
            elif op < 0.9:
                seq.reset_for_recompute()
                cached = 0
            else:
                seq.status = rng.choice(list(SequenceStatus))
            # Invariants checked after every operation.
            assert seq.block_size == block_size
            assert seq.get_token_ids() == tokens
            assert seq.get_len() == len(tokens)
            assert seq.get_prompt_len() == len(prompt)
            assert seq.get_output_len() == len(tokens) - len(prompt)
            assert seq.num_cached_tokens == cached
            assert 0 <= seq.num_cached_tokens <= seq.get_len()
            assert seq.get_num_blocks() == (len(tokens) + block_size - 1) // block_size
            assert (seq.get_num_blocks() - 1) * block_size < max(seq.get_len(), 1)
            if tokens:
                assert seq.get_last_token_id() == tokens[-1]
            assert seq.is_finished() == (seq.status >= SequenceStatus.FINISHED_STOPPED)
