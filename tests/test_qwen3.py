"""The whole Qwen3 model (S1.2) against Hugging Face, in float32 on the CPU.

In float32 the two implementations differ only by the order of floating-point
operations, so their logits agree to about 1e-5 (compared at 1e-4). A real bug
(wrong head grouping, RoPE on the wrong half, a missing norm weight, swapped
q/k/v) moves them by O(1). The bfloat16 check against the saved golden logits
of the real models is ``scripts/check_model.py``.
"""

import pytest
import torch

from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.model_executor.model_loader import load_model


def our_logits(model, prompts):
    """Next-token logits at every position of every prompt, one packed forward."""
    md = InputMetadata.for_prompts(prompts, device="cpu")
    hidden = model(None, md)
    logits = get_logits(hidden, model.lm_head.weight, model.cfg.vocab_size)
    return [logits[s:e] for s, e in zip(md.cu_seqlens_q[:-1], md.cu_seqlens_q[1:])]


@torch.inference_mode()
def hf_logits(hf, prompt):
    return hf(torch.tensor([prompt])).logits[0].float()


def random_prompts(lengths, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # GQA, 2 query heads per K/V head, tied
        {"num_key_value_heads": 4},  # MHA
        {"num_key_value_heads": 1},  # MQA
        {"tie_word_embeddings": False},
        {"num_attention_heads": 6, "num_key_value_heads": 3, "head_dim": 8},
        {"rms_norm_eps": 1e-3},  # the config's eps must reach every norm
    ],
    ids=["gqa2-tied", "mha", "mqa", "untied", "6-heads-dim8", "eps1e-3"],
)
@torch.inference_mode()
def test_logits_match_hf_fp32(tiny_checkpoint, overrides):
    hf, path = tiny_checkpoint(**overrides)
    model, _ = load_model(path, device="cpu")
    prompts = random_prompts([1, 2, 7, 40, 200], hf.config.vocab_size)
    for prompt, ours in zip(prompts, our_logits(model, prompts)):
        ref = hf_logits(hf, prompt)
        assert ours.shape == ref.shape == (len(prompt), hf.config.vocab_size)
        torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
        assert torch.equal(ours.argmax(-1), ref.argmax(-1))


@torch.inference_mode()
def test_packing_does_not_change_results(tiny_checkpoint):
    _, path = tiny_checkpoint()
    model, _ = load_model(path, device="cpu")
    prompts = random_prompts([3, 17, 1, 64], model.cfg.vocab_size, seed=1)
    packed = our_logits(model, prompts)
    for prompt, together in zip(prompts, packed):
        (alone,) = our_logits(model, [prompt])
        torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_prefix_logits_do_not_depend_on_later_tokens(tiny_checkpoint):
    # Causality through the whole stack: the logits at position j only see
    # tokens 0..j, so appending tokens leaves earlier rows unchanged.
    _, path = tiny_checkpoint()
    model, _ = load_model(path, device="cpu")
    (prompt,) = random_prompts([30], model.cfg.vocab_size, seed=2)
    (short,) = our_logits(model, [prompt[:12]])
    (full,) = our_logits(model, [prompt])
    torch.testing.assert_close(full[:12], short, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_positions_matter(tiny_checkpoint):
    # Same tokens at shifted positions must give different results (RoPE is
    # applied), but only through relative distances: shifting every position
    # by the same amount changes almost nothing.
    _, path = tiny_checkpoint()
    model, _ = load_model(path, device="cpu")
    (prompt,) = random_prompts([10], model.cfg.vocab_size, seed=3)
    md = InputMetadata.for_prompts([prompt], device="cpu")
    base = get_logits(model(None, md), model.lm_head.weight, model.cfg.vocab_size)
    md.positions = md.positions + 100
    shifted = get_logits(model(None, md), model.lm_head.weight, model.cfg.vocab_size)
    torch.testing.assert_close(shifted, base, rtol=1e-3, atol=1e-3)
    md.positions = torch.tensor([0, 1, 2, 3, 4, 9, 6, 7, 8, 5])  # swap two positions
    swapped = get_logits(model(None, md), model.lm_head.weight, model.cfg.vocab_size)
    assert (swapped - base).abs().max() > 1e-2


def test_bf16_model_runs_and_stays_close(tiny_checkpoint):
    hf32, path32 = tiny_checkpoint()
    _, path16 = tiny_checkpoint(dtype=torch.bfloat16)
    model, _ = load_model(path16, device="cpu")
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    (prompt,) = random_prompts([25], model.cfg.vocab_size, seed=4)
    with torch.inference_mode():
        (ours,) = our_logits(model, [prompt])
        ref = hf_logits(hf32, prompt)
    assert ours.dtype == torch.float32  # logits are fp32 even for a bf16 model
    # bf16 weights and activations: a few percent of the logit scale.
    assert (ours - ref).abs().max() < 0.05 * ref.abs().max()
