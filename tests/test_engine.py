"""Prefill and decode with the KV cache, end to end (S1.3), on tiny random
checkpoints in float32 on the CPU.

The central property: decoding with the cache gives the same logits as
recomputing the whole sequence without a cache (S1.2's checked path), at every
step, for every sequence of a batch, whatever the block size.
"""

import math

import pytest
import torch

from mini_deepinfer.engine.block_manager import BlockManagerMutexAcquire
from mini_deepinfer.engine.static_engine import StaticEngine, check_stop, process_outputs
from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.sequence import Sequence, SequenceStatus
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner

MAX_LEN = 64
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_engine(path, *, block_size=MAX_LEN, num_blocks=None, max_model_len=MAX_LEN, eos_token_id=0, device="cpu"):
    runner = ModelRunner(path, device=device, block_size=block_size, max_model_len=max_model_len)
    runner.load_model()
    if num_blocks is None:
        num_blocks = 8 * math.ceil(max_model_len / block_size) + 1  # + the reserved block 0
    cache = CacheEngine(runner.model.cfg, num_blocks=num_blocks, block_size=block_size, device=device)
    for layer in cache.gpu_cache:  # any read of a slot that was never written turns the output into NaN
        layer.k_cache.fill_(float("nan"))
        layer.v_cache.fill_(float("nan"))
    return StaticEngine(runner, cache, eos_token_id=eos_token_id)


def greedy(max_tokens=8, **kw):
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, **kw)


@torch.inference_mode()
def full_logits(model, tokens):
    """Next-token logits after the last token, recomputed without a cache."""
    md = InputMetadata.for_prompts([tokens], device=model.lm_head.weight.device)
    hidden = model(None, md)
    return get_logits(hidden[-1:], model.lm_head.weight, model.cfg.vocab_size)[0].cpu()


def random_prompts(lengths, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.mark.parametrize("block_size", [MAX_LEN, 4], ids=["contiguous", "paged-4"])
@pytest.mark.parametrize("overrides", [{}, {"num_key_value_heads": 1}], ids=["gqa2", "mqa"])
def test_decode_with_cache_matches_full_recompute(tiny_checkpoint, block_size, overrides):
    _, path = tiny_checkpoint(**overrides)
    engine = make_engine(path, block_size=block_size)
    model = engine.runner.model
    prompts = random_prompts([5, 1, 12], model.cfg.vocab_size)
    golden = random_prompts([10, 10, 10], model.cfg.vocab_size, seed=1)  # tokens fed back instead of sampled ones
    seqs = engine.add_requests(prompts, [greedy(max_tokens=10, golden_token_ids=g) for g in golden])
    out = engine.step(seqs, is_prompt=True)
    for prompt, row in zip(prompts, out.logits):
        torch.testing.assert_close(row, full_logits(model, prompt), rtol=1e-4, atol=1e-4)
    for t in range(1, 10):
        batch = seqs if t % 2 else seqs[::-1]  # the batch order must not matter
        before = [s.get_token_ids() for s in batch]
        out = engine.step(batch, is_prompt=False)
        assert out.logits.isfinite().all()
        assert all(0 not in s.cache.block_table for s in seqs)
        for tokens, row in zip(before, out.logits):
            torch.testing.assert_close(row, full_logits(model, tokens), rtol=1e-4, atol=1e-4)
    assert [s.get_token_ids()[s.get_prompt_len() :] for s in seqs] == golden


def test_greedy_matches_hf_generate(tiny_checkpoint):
    hf, path = tiny_checkpoint()
    engine = make_engine(path)
    prompts = random_prompts([3, 9, 17], hf.config.vocab_size, seed=2)
    ours = [s.get_token_ids()[s.get_prompt_len() :] for s in engine.generate(prompts, greedy(max_tokens=12))]
    for prompt, tokens in zip(prompts, ours):
        with torch.inference_mode():
            ref = hf.generate(torch.tensor([prompt]), do_sample=False, max_new_tokens=12, min_new_tokens=12)
        assert tokens == ref[0, len(prompt) :].tolist()
    # The same prompts one at a time give the same tokens.
    for prompt, tokens in zip(prompts, ours):
        (seq,) = engine.generate([prompt], greedy(max_tokens=12))
        assert seq.get_token_ids()[len(prompt) :] == tokens


def test_golden_tokens_replace_sampled_ones(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path)
    vocab = engine.runner.model.cfg.vocab_size
    (free,) = engine.generate([[3, 4, 5]], greedy(max_tokens=6))
    forced = [(t + 1) % vocab for t in free.get_token_ids()[3:5]]  # not what greedy picks
    (seq,) = engine.generate([[3, 4, 5]], greedy(max_tokens=6, golden_token_ids=forced))
    out = seq.get_token_ids()[3:]
    assert out[:2] == forced and len(out) == 6  # forced first, then greedy again
    assert out[2] == full_logits(engine.runner.model, [3, 4, 5, *forced]).argmax().item()


@pytest.mark.parametrize("chunks", [[4, 5, 3], [1, 11], [12], [11, 1]])
def test_chunked_prefill_matches_whole_prefill(tiny_checkpoint, chunks):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4)
    model = engine.runner.model
    (prompt,) = random_prompts([12], model.cfg.vocab_size, seed=3)
    (seq,) = engine.add_requests([prompt], greedy())
    for i, n in enumerate(chunks):
        out = engine.step([seq], is_prompt=True, seqlens_q=[n])
        last = i == len(chunks) - 1
        assert seq.num_cached_tokens == sum(chunks[: i + 1])
        assert seq.get_len() == len(prompt) + (1 if last else 0)
        assert seq.status == (SequenceStatus.RUNNING if last else SequenceStatus.WAITING)
        assert len(seq.cache.block_table) == 3  # the whole prompt's blocks from the first chunk on
    torch.testing.assert_close(out.logits[0], full_logits(model, prompt), rtol=1e-4, atol=1e-4)
    out = engine.step([seq], is_prompt=False)
    torch.testing.assert_close(out.logits[0], full_logits(model, seq.get_token_ids()[:-1]), rtol=1e-4, atol=1e-4)


def test_prefill_without_seqlens_q_takes_the_rest(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4)
    model = engine.runner.model
    (prompt,) = random_prompts([12], model.cfg.vocab_size, seed=6)
    (seq,) = engine.add_requests([prompt], greedy())
    engine.step([seq], is_prompt=True, seqlens_q=[5])
    out = engine.step([seq], is_prompt=True)
    assert (seq.num_cached_tokens, seq.get_len()) == (12, 13)
    torch.testing.assert_close(out.logits[0], full_logits(model, prompt), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("continued_first", [True, False])
def test_prefill_batch_of_new_and_continued_sequences(tiny_checkpoint, continued_first):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4)
    model = engine.runner.model
    pa, pb = random_prompts([12, 7], model.cfg.vocab_size, seed=5)
    a, b = engine.add_requests([pa, pb], greedy())
    engine.step([a], is_prompt=True, seqlens_q=[5])  # a has 5 tokens cached, b none
    order = [a, b] if continued_first else [b, a]
    out = engine.step(order, is_prompt=True, seqlens_q=[7, 7])
    for seq, row in zip(order, out.logits):
        prompt = pa if seq is a else pb
        torch.testing.assert_close(row, full_logits(model, prompt), rtol=1e-4, atol=1e-4)


def test_cache_counters_and_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=10)
    (seq,) = engine.add_requests([[1, 2, 3, 4, 5, 6]], greedy(max_tokens=7))
    engine.step([seq], is_prompt=True)
    assert (seq.num_cached_tokens, seq.get_len(), len(seq.cache.block_table)) == (6, 7, 2)
    while not seq.is_finished():
        engine.step([seq], is_prompt=False)
        assert seq.num_cached_tokens == seq.get_len() - 1
        assert len(seq.cache.block_table) in (math.ceil(seq.num_cached_tokens / 4), 0)
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED and seq.get_output_len() == 7
    assert seq.cache.block_table == [] and engine.block_manager.get_num_free_blocks() == 9  # blocks returned


def test_status_and_usage(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=10)
    assert engine.get_gpu_cache_usage() == pytest.approx(1 / 10)
    (seq,) = engine.add_requests([[1, 2, 3, 4, 5]], greedy(max_tokens=3))
    assert seq.status == SequenceStatus.WAITING and seq.cache.block_table == []
    engine.step([seq], is_prompt=True)
    assert seq.status == SequenceStatus.RUNNING and len(seq.cache.block_table) == 2
    assert engine.get_gpu_cache_usage() == pytest.approx(3 / 10)
    assert engine.get_gpu_cache_usage(num_gpu_blocks_to_use=2) == pytest.approx(5 / 10)


def test_append_slot_grows_by_one_or_trims(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=10)
    (seq,) = engine.add_requests([[1, 2, 3, 4]], greedy())
    with BlockManagerMutexAcquire(engine.block_manager) as acquire:
        acquire.allocate_seq(seq)
        assert seq.cache.block_table == [1]
        seq.append_token_id(5)  # 5 tokens need 2 blocks
        engine._append_slot(seq)
        assert seq.cache.block_table == [1, 2]
        seq.cache.block_table.append(engine.block_manager.allocate_single_nolock())  # one too many
        engine._append_slot(seq)
        assert seq.cache.block_table == [1, 2]
    assert engine.block_manager.get_num_free_blocks() == 7  # the trimmed block went back
    for _ in range(8):
        seq.append_token_id(5)  # 13 tokens need 4 blocks; the table has 2
    with BlockManagerMutexAcquire(engine.block_manager), pytest.raises(RuntimeError):
        engine._append_slot(seq)  # one block per call, never more


def test_reused_blocks_give_the_same_tokens(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=12)
    prompts = random_prompts([7, 10], engine.runner.model.cfg.vocab_size, seed=4)
    first = [s.get_token_ids() for s in engine.generate(prompts, greedy(max_tokens=9))]
    second = [s.get_token_ids() for s in engine.generate(prompts[::-1], greedy(max_tokens=9))][::-1]
    assert first == second  # the second run reuses blocks full of the first run's K/V


def test_stop_on_eos_and_lengths(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, max_model_len=10, block_size=10, eos_token_id=42)

    def sp(golden):
        return SamplingParams(temperature=0.0, max_tokens=5, golden_token_ids=golden)

    a, b, c, d = engine.add_requests([[1, 2], [3, 4], [5] * 10, [6] * 11], [sp([7, 9, 9, 9, 9]), sp([42]), sp([8]), sp(None)])
    assert all(s.sampling_params.eos_id == 42 for s in (a, b, c, d))  # set by the engine
    assert d.status == SequenceStatus.FINISHED_IGNORED  # longer than max_model_len
    engine.step([a, b, c], is_prompt=True)
    assert b.status == SequenceStatus.FINISHED_STOPPED  # EOS
    assert c.status == SequenceStatus.FINISHED_LENGTH_CAPPED and c.get_len() == 11  # past max_model_len
    assert a.status == SequenceStatus.RUNNING
    for _ in range(4):
        engine.step([a], is_prompt=False)
    assert a.status == SequenceStatus.FINISHED_LENGTH_CAPPED and a.get_output_len() == 5  # max_tokens
    assert engine.block_manager.get_num_free_blocks() == engine.block_manager.num_blocks - 1


def make_seq(prompt_len, sp):
    return Sequence(seq_id=0, prompt_token_ids=[1] * prompt_len, block_size=4, request_id="r", sampling_params=sp, arrival_time=0.0)


def test_check_stop_order():
    # EOS on the step that also reaches max_tokens counts as a stop, not a length cap.
    seq = make_seq(1, SamplingParams(temperature=0.0, max_tokens=1, eos_id=3))
    seq.append_token_id(3)
    check_stop(seq, max_model_len=100)
    assert seq.status == SequenceStatus.FINISHED_STOPPED
    seq = make_seq(1, SamplingParams(temperature=0.0, max_tokens=1, eos_id=3, ignore_eos=True))
    seq.append_token_id(3)
    check_stop(seq, max_model_len=100)
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED


def test_check_stop_at_max_model_len():
    sp = SamplingParams(temperature=0.0, max_tokens=5, eos_id=3)
    seq = make_seq(9, sp)
    seq.append_token_id(7)  # length 10 == max_model_len
    check_stop(seq, max_model_len=10)
    assert not seq.is_finished()
    seq.append_token_id(7)  # length 11
    check_stop(seq, max_model_len=10)
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED
    seq = make_seq(10, sp)
    seq.append_token_id(3)  # EOS at length 11
    check_stop(seq, max_model_len=10)
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_prepare_input_tensors(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4)
    runner = engine.runner
    a, b = engine.add_requests([[10, 11, 12, 13, 14, 15], [20]], greedy())
    a.cache.block_table[:] = [5, 2]
    b.cache.block_table[:] = [7]
    md = runner.prepare_input_tensors([a, b], is_prompt=True)
    assert md.input_ids.dtype == md.positions.dtype == md.slot_mapping.dtype == torch.int32
    assert md.input_ids.tolist() == [10, 11, 12, 13, 14, 15, 20]
    assert md.positions.tolist() == [0, 1, 2, 3, 4, 5, 0]
    assert md.slot_mapping.tolist() == [20, 21, 22, 23, 8, 9, 28]
    assert (md.seqlens_q, md.seqlens_k, md.cu_seqlens_q) == ([6, 1], [6, 1], [0, 6, 7])
    assert md.block_tables.shape == (2, runner.max_num_blocks) and md.block_tables[1, :2].tolist() == [7, 0]
    with pytest.raises(ValueError):
        runner.prepare_input_tensors([a], is_prompt=False)  # nothing cached yet: not a decode
    a.cache.block_table[:] = [5]
    with pytest.raises(ValueError):
        runner.prepare_input_tensors([a], is_prompt=True)  # 6 tokens do not fit one block of 4


def test_admission_and_out_of_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=4)  # blocks 1-3 usable
    with pytest.raises(ValueError):
        engine.add_requests([[]], greedy())
    big, fits = engine.add_requests([[1] * 13, [1] * 12], greedy())
    assert big.status == SequenceStatus.FINISHED_IGNORED  # 4 blocks of 4: can never run
    assert fits.status == SequenceStatus.WAITING  # 3 blocks: all the usable ones
    a, b = engine.add_requests([[1] * 9, [2] * 9], greedy())  # 3 blocks each: one fits, not both
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([a, b], is_prompt=True)
    assert engine.block_manager.get_num_free_blocks() == 3 and a.cache.block_table == []  # nothing was taken
    assert a.status == b.status == SequenceStatus.WAITING
    assert engine.block_manager._lock.acquire(blocking=False)  # the failure released the lock
    engine.block_manager._lock.release()


def test_prefill_that_must_wait_takes_nothing(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=6)  # 5 usable
    (x,) = engine.add_requests([[3] * 8], greedy())
    engine.step([x], is_prompt=True)  # x holds 2 blocks, 3 free
    a, b = engine.add_requests([[1] * 8, [2] * 8], greedy())  # 2 blocks each: 4 > 3
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([a, b], is_prompt=True)
    assert a.cache.block_table == b.cache.block_table == []
    assert a.status == b.status == SequenceStatus.WAITING
    assert engine.block_manager.get_num_free_blocks() == 3


def test_continued_chunk_needs_no_new_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=6)  # 5 usable
    a, b = engine.add_requests([[1] * 12, [2] * 8], greedy())
    engine.step([a], is_prompt=True, seqlens_q=[4])  # a takes all 3 blocks now
    engine.step([a, b], is_prompt=True, seqlens_q=[8, 8])  # only b needs blocks: 2 of the 2 left
    assert (len(a.cache.block_table), len(b.cache.block_table)) == (3, 2)
    assert engine.block_manager.get_num_free_blocks() == 0


def test_decode_at_exact_capacity(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=4)  # 3 usable
    a, b = engine.add_requests([[1] * 4, [2] * 2], greedy())
    engine.step([a, b], is_prompt=True)  # a: 4 tokens -> 1 block (+1 sampled), b: 1 block
    engine.step([a, b], is_prompt=False)  # a's sampled token needs a 2nd block; b's fits
    assert (len(a.cache.block_table), len(b.cache.block_table)) == (2, 1)
    assert engine.block_manager.get_num_free_blocks() == 0


def test_decode_out_of_blocks_aborts_and_frees(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=3)  # 2 usable
    a, b = engine.add_requests([[1] * 4, [2] * 4], greedy())
    engine.step([a, b], is_prompt=True)
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.step([a, b], is_prompt=False)  # both need a 2nd block, none is free
    assert a.status == b.status == SequenceStatus.FINISHED_ABORTED
    assert a.cache.block_table == b.cache.block_table == []
    assert engine.block_manager.get_num_free_blocks() == 2
    with pytest.raises(RuntimeError, match="out of KV-cache blocks"):
        engine.generate([[1] * 4, [2] * 4], greedy())  # generate() gives nothing back, but frees
    assert engine.block_manager.get_num_free_blocks() == 2


def test_decode_refuses_a_half_prefilled_sequence(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4)
    a, b = engine.add_requests([[1] * 7, [2] * 3], greedy())
    engine.step([a], is_prompt=True, seqlens_q=[3])
    engine.step([b], is_prompt=True)
    with pytest.raises(ValueError):
        engine.step([a, b], is_prompt=False)
    assert a.status == SequenceStatus.WAITING and b.status == SequenceStatus.RUNNING  # nothing changed


def test_failed_step_frees_its_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=8)
    with pytest.raises(NotImplementedError):
        engine.generate([[1, 2, 3]], SamplingParams(temperature=0.7))
    with pytest.raises(NotImplementedError):
        engine.generate([[1, 2, 3]], SamplingParams(temperature=0.0, first_n_token_no_eos=1))
    assert engine.block_manager.get_num_free_blocks() == 7
    (seq,) = engine.generate([[1, 2, 3]], greedy(max_tokens=2))  # the engine still works
    assert seq.get_output_len() == 2


def test_cache_engine_shapes(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, num_blocks=3)
    cfg = engine.runner.model.cfg
    cache = engine.cache_engine
    assert len(cache) == cfg.num_hidden_layers
    assert cache[0].k_cache.shape == (3, 4, cfg.num_key_value_heads, cfg.head_dim)
    per_block = sum(t.numel() * t.element_size() for layer in cache.gpu_cache for t in (layer.k_cache, layer.v_cache)) // 3
    assert CacheEngine.get_cache_block_size(4, cfg) == per_block


@cuda
def test_forward_on_cuda(tiny_checkpoint):
    _, path = tiny_checkpoint()
    engine = make_engine(path, block_size=4, device="cuda")
    runner = engine.runner
    (seq,) = engine.add_requests(random_prompts([9], runner.model.cfg.vocab_size, seed=7), greedy())
    for is_prompt in (True, False):
        if is_prompt:
            engine._allocate_prefill([seq], None)
        else:
            engine._allocate_decode([seq])
        md = runner.prepare_input_tensors([seq], is_prompt)
        torch.cuda.set_sync_debug_mode("error")
        try:
            hidden = runner.execute_main_model(md, engine.cache_engine.gpu_cache)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        out = runner.execute_sample([seq], md, hidden)
        process_outputs([seq], out.token_ids, is_prompt, md.seqlens_q, engine.max_model_len)
    torch.testing.assert_close(out.logits[0].cpu(), full_logits(runner.model, seq.get_token_ids()[:-1]), rtol=1e-4, atol=1e-4)
