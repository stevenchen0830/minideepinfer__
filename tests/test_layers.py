"""Layer tests (S1.2): each layer against Hugging Face's own module or a
plain-math reference, plus properties that must hold whatever the weights."""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.activation import silu_and_mul
from mini_deepinfer.model_executor.layers.attention_backend import prefill_attn_varlen
from mini_deepinfer.model_executor.layers.layernorm import RMSNorm, rms_norm
from mini_deepinfer.model_executor.layers.linear import ReplicatedLinear, bf16_gemm_nt
from mini_deepinfer.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope, rope_memo_scope
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# --- RMSNorm ----------------------------------------------------------------


def test_rmsnorm_matches_hf_in_fp32():
    w = 1 + 0.3 * torch.randn(64)
    x = torch.randn(5, 64) * 3
    hf = Qwen3RMSNorm(64, eps=1e-6)
    hf.weight.data.copy_(w)
    torch.testing.assert_close(RMSNorm(64, 1e-6, weight=w)(x), hf(x), rtol=1e-6, atol=1e-6)


def test_rmsnorm_bf16_rounds_once():
    # Our rule: everything in fp32, one cast at the end. Check it bit for bit.
    w = (1 + 0.3 * torch.randn(64)).bfloat16()
    x = (torch.randn(7, 64) * 3).bfloat16()
    ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()
    out = RMSNorm(64, 1e-6, weight=w)(x)
    assert out.dtype == torch.bfloat16
    # One rounding to bf16 (8 significant bits) has relative error <= 2^-9;
    # 2^-8 leaves room for the fp32 arithmetic before it.
    assert ((out.double() - ref).abs() <= ref.abs() * 2**-8).all()
    # HF rounds twice (before and after the weight); the two agree within
    # two bf16 steps of the larger value.
    hf = Qwen3RMSNorm(64, eps=1e-6).bfloat16()
    hf.weight.data.copy_(w)
    hf_out = hf(x).float()
    big = torch.maximum(out.float().abs(), hf_out.abs()).clamp_min(1e-30)
    step = 2.0 ** (torch.floor(torch.log2(big)) - 7)
    assert ((out.float() - hf_out).abs() <= 2 * step).all()
    assert not torch.equal(out, hf(x))  # the extra rounding is visible somewhere


def test_rmsnorm_fused_residual():
    norm = RMSNorm(32, 1e-6, weight=1 + torch.randn(32))
    x, r = torch.randn(4, 32).bfloat16(), torch.randn(4, 32).bfloat16()
    x0, r0 = x.clone(), r.clone()
    out, new_r = norm(x, r)
    total = x + r  # the add happens in the input dtype
    assert torch.equal(new_r, total)
    assert torch.equal(out, norm(total))
    assert torch.equal(x, x0) and torch.equal(r, r0)  # inputs untouched


def test_rmsnorm_over_heads_normalizes_each_head():
    # q_norm/k_norm: the last dim is head_dim, every (token, head) row separately.
    norm = RMSNorm(16, 1e-6)
    x = torch.randn(3, 4, 16) * torch.arange(1, 13).view(3, 4, 1)
    out = norm(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.pow(2).mean(-1), torch.ones(3, 4), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out, rms_norm(x.reshape(12, 16), norm.weight, 1e-6).view(3, 4, 16))


def test_rmsnorm_weight_shape_checked():
    with pytest.raises(ValueError):
        RMSNorm(16, weight=torch.ones(15))


# --- RoPE -------------------------------------------------------------------


def hf_cos_sin(positions, dim, base):
    # The same math as transformers' Qwen3RotaryEmbedding (default rope type).
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = positions[:, None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def test_rope_matches_hf_in_fp32():
    rope = RotaryEmbedding(64, 64, 512, 1_000_000.0, is_neox_style=True)
    positions = torch.tensor([0, 1, 2, 3, 511, 37, 37, 100])  # any order, repeats allowed
    q, k = torch.randn(8, 4, 64), torch.randn(8, 2, 64)
    cos, sin = hf_cos_sin(positions, 64, 1_000_000.0)
    # HF layout [batch, heads, seq, dim]; cos/sin [batch, seq, dim].
    q_ref, k_ref = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[None], sin[None])
    ret = rope(positions, q, k)
    assert ret is None  # in place
    torch.testing.assert_close(q, q_ref[0].transpose(0, 1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k, k_ref[0].transpose(0, 1), rtol=1e-5, atol=1e-5)


def test_rope_cache_layout():
    rope = RotaryEmbedding(16, 16, 100, 10000.0, is_neox_style=True)
    cache = rope.cos_sin_cache
    assert cache.dtype == torch.float32 and cache.shape == (100, 16)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 16, 2).float() / 16))
    angle = 7 * inv_freq
    torch.testing.assert_close(cache[7, :8], angle.cos())
    torch.testing.assert_close(cache[7, 8:], angle.sin())
    assert "cos_sin_cache" not in rope.state_dict()


def test_rope_cache_ignores_default_dtype():
    # load_model builds modules under the checkpoint's default dtype (bf16);
    # the cache must come out exactly as under fp32.
    ref = RotaryEmbedding(128, 128, 4096, 1_000_000.0, is_neox_style=True).cos_sin_cache
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        cache = RotaryEmbedding(128, 128, 4096, 1_000_000.0, is_neox_style=True).cos_sin_cache
    finally:
        torch.set_default_dtype(old)
    assert cache.dtype == torch.float32
    assert torch.equal(cache, ref)


def test_rope_refuses_a_cast_cache():
    rope = RotaryEmbedding(16, 16, 32, 10000.0, is_neox_style=True)
    rope.half()
    with pytest.raises(AssertionError):
        rope(torch.tensor([1]), torch.randn(1, 2, 16))


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(32, 32, 16, 10000.0, is_neox_style=True)
    q = torch.randn(3, 2, 32)
    q0 = q.clone()
    rope(torch.zeros(3, dtype=torch.long), q)
    torch.testing.assert_close(q, q0)


@pytest.mark.parametrize("neox", [True, False])
def test_rope_relative_position(neox):
    # The point of RoPE: <rope(q, m), rope(k, n)> depends only on m - n.
    rope = RotaryEmbedding(32, 32, 4096, 10000.0, is_neox_style=neox)
    q, k = torch.randn(1, 1, 32), torch.randn(1, 1, 32)

    def rotated(x, pos):
        y = x.clone()
        rope(torch.tensor([pos]), y)
        return y

    def score(m, n):
        return (rotated(q, m) * rotated(k, n)).sum().item()

    assert score(10, 3) == pytest.approx(score(1007, 1000), abs=2e-2)
    assert score(5, 5) == pytest.approx(score(2000, 2000), abs=2e-2)
    assert score(10, 3) != pytest.approx(score(10, 4), abs=2e-2)


def test_rope_interleaved_matches_complex_multiply():
    rope = RotaryEmbedding(8, 8, 64, 10000.0, is_neox_style=False)
    positions = torch.tensor([0, 5, 63])
    q = torch.randn(3, 2, 8)
    ref = torch.view_as_complex(q.reshape(3, 2, 4, 2).contiguous())
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    ref = ref * torch.polar(torch.ones(3, 1, 4), positions[:, None, None].float() * inv_freq)
    rope(positions, q)
    torch.testing.assert_close(q, torch.view_as_real(ref).reshape(3, 2, 8))


@pytest.mark.parametrize("neox", [True, False])
def test_rope_conjugate_undoes_rotation(neox):
    rope = RotaryEmbedding(16, 16, 128, 10000.0, is_neox_style=neox)
    positions = torch.randint(0, 128, (6,))
    q = torch.randn(6, 3, 16)
    q0 = q.clone()
    rope(positions, q)
    assert not torch.allclose(q, q0)
    rope(positions, q, conjugate=True)
    torch.testing.assert_close(q, q0, rtol=1e-5, atol=1e-5)


def test_rope_bf16_input_stays_bf16():
    rope = RotaryEmbedding(16, 16, 32, 10000.0, is_neox_style=True)
    q = torch.randn(2, 2, 16).bfloat16()
    ref = q.float().clone()
    rope(torch.tensor([3, 9]), q)
    rope(torch.tensor([3, 9]), ref)
    assert q.dtype == torch.bfloat16
    assert torch.equal(q, ref.bfloat16())  # computed in fp32, rounded once


def test_get_rope_memoizes_and_refuses_scaling():
    with rope_memo_scope():
        a = get_rope(16, 16, 64, 10000.0, device="cpu")
        assert get_rope(16, 16, 64, 10000.0, device="cpu") is a
        assert get_rope(16, 16, 64, 500000.0, device="cpu") is not a
        a.half()  # a cast entry is rebuilt, not handed out again
        b = get_rope(16, 16, 64, 10000.0, device="cpu")
        assert b is not a and b.cos_sin_cache.dtype == torch.float32
    with pytest.raises(ValueError):
        get_rope(16, 16, 64, 10000.0, rope_scaling={"type": "yarn", "factor": 4.0})


# --- Linear, embedding, logits, activation -----------------------------------


def test_replicated_linear():
    w, b = torch.randn(6, 4), torch.randn(6)
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(ReplicatedLinear(4, 6, weight=w)(x), F.linear(x, w))
    torch.testing.assert_close(ReplicatedLinear(4, 6, weight=w, bias=b)(x), F.linear(x, w, b))
    with pytest.raises(ValueError):
        ReplicatedLinear(4, 6, weight=torch.randn(4, 6))


def test_merged_linear_equals_parts():
    # A merged projection is just the concatenated weight.
    wq, wk, wv = torch.randn(8, 5), torch.randn(4, 5), torch.randn(4, 5)
    x = torch.randn(3, 5)
    out = ReplicatedLinear(5, 16, weight=torch.cat([wq, wk, wv]))(x)
    q, k, v = out.split([8, 4, 4], dim=-1)
    torch.testing.assert_close(q, x @ wq.T)
    torch.testing.assert_close(k, x @ wk.T)
    torch.testing.assert_close(v, x @ wv.T)


def test_embedding_and_lm_head():
    w = torch.randn(10, 4)
    emb = VocabParallelEmbedding(10, 4, weight=w)
    assert torch.equal(emb(torch.tensor([3, 0, 3])), w[[3, 0, 3]])
    head = ParallelLMHead(10, 4, weight=emb.weight)
    assert head.weight.data_ptr() == emb.weight.data_ptr()  # shares storage
    assert head.weight is not emb.weight  # but is its own Parameter
    with pytest.raises(RuntimeError):
        head(torch.tensor([1]))


def test_get_logits_fp32():
    h, w = torch.randn(3, 8), torch.randn(11, 8)
    logits = get_logits(h, w, 11)
    assert logits.dtype == torch.float32 and logits.shape == (3, 11)
    torch.testing.assert_close(logits, h @ w.T)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_get_logits_bf16_inputs(device):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(4, 256, generator=g).bfloat16().to(device)
    w = torch.randn(1000, 256, generator=g).bfloat16().to(device)
    exact = (h.double() @ w.double().T).cpu()
    logits = get_logits(h, w, 1000).cpu()
    assert logits.dtype == torch.float32
    assert (logits.double() - exact).abs().max() < 1e-3  # close to the exact float64 product


@cuda
def test_bf16_gemm_fp32_output_is_more_accurate():
    a = torch.randn(16, 1024, device="cuda").bfloat16()
    b = torch.randn(4096, 1024, device="cuda").bfloat16()
    exact = a.double() @ b.double().T
    d = torch.empty(16, 4096, device="cuda")
    bf16_gemm_nt(a, b, d)
    err_fp32 = (d.double() - exact).abs().max()
    err_bf16 = ((a @ b.T).double() - exact).abs().max()
    assert err_fp32 < err_bf16 / 50


def test_silu_and_mul():
    g, u = torch.randn(4, 6), torch.randn(4, 6)
    torch.testing.assert_close(silu_and_mul(torch.cat([g, u], -1)), F.silu(g) * u)
    x = torch.randn(4, 12).bfloat16()
    ref = (F.silu(x[:, :6].float()) * x[:, 6:].float()).bfloat16()
    assert torch.equal(silu_and_mul(x), ref)


# --- attention ---------------------------------------------------------------


def naive_attention(q, k, v, scale):
    """One sequence, plain math: causal softmax(q k^T * scale) v with GQA."""
    t, h, d = q.shape
    group = h // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    mask = torch.ones(t, t, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@pytest.mark.parametrize("heads, kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_attention_matches_naive_per_sequence(heads, kv_heads):
    lens = [1, 5, 9]
    md = InputMetadata.for_prompts([[0] * n for n in lens], device="cpu")
    total = md.total_q
    q, k, v = torch.randn(total, heads, 16), torch.randn(total, kv_heads, 16), torch.randn(total, kv_heads, 16)
    out = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, 0.25)
    for s, e in zip(md.cu_seqlens_q[:-1], md.cu_seqlens_q[1:]):
        torch.testing.assert_close(out[s:e], naive_attention(q[s:e], k[s:e], v[s:e], 0.25), rtol=1e-5, atol=1e-5)


def test_attention_value_head_dim_may_differ():
    md = InputMetadata.for_prompts([[0] * 3, [0] * 4], device="cpu")
    q, k, v = torch.randn(7, 4, 16), torch.randn(7, 2, 16), torch.randn(7, 2, 8)
    out = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, 0.25)
    assert out.shape == (7, 4, 8)
    torch.testing.assert_close(out[3:], naive_attention(q[3:], k[3:], v[3:], 0.25), rtol=1e-5, atol=1e-5)


def test_attention_refuses_unequal_lengths():
    q, kv = torch.randn(1, 2, 8), torch.randn(5, 2, 8)
    with pytest.raises(ValueError):
        prefill_attn_varlen(q, kv, kv, [0, 1], 0.3)


def test_attention_sequences_do_not_mix():
    md = InputMetadata.for_prompts([[0] * 4, [0] * 6], device="cpu")
    q, k, v = torch.randn(10, 2, 8), torch.randn(10, 1, 8), torch.randn(10, 1, 8)
    out = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, 0.3)
    k2, v2 = k.clone(), v.clone()
    k2[4:], v2[4:] = torch.randn(6, 1, 8), torch.randn(6, 1, 8)  # change only sequence 1
    out2 = prefill_attn_varlen(q, k2, v2, md.cu_seqlens_q, 0.3)
    assert torch.equal(out[:4], out2[:4])
    assert not torch.allclose(out[4:], out2[4:])


def test_attention_is_causal():
    # Changing token j must not change the output of any token before j.
    md = InputMetadata.for_prompts([[0] * 8], device="cpu")
    q, k, v = torch.randn(8, 2, 8), torch.randn(8, 2, 8), torch.randn(8, 2, 8)
    out = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, 0.3)
    k[5], v[5] = torch.randn(2, 8), torch.randn(2, 8)
    out2 = prefill_attn_varlen(q, k, v, md.cu_seqlens_q, 0.3)
    assert torch.equal(out[:5], out2[:5])
    assert not torch.allclose(out[5:], out2[5:])


def test_input_metadata_for_prompts():
    md = InputMetadata.for_prompts([[5, 6, 7], [8], [9, 10]], device="cpu")
    assert md.input_ids.tolist() == [5, 6, 7, 8, 9, 10]
    assert md.positions.tolist() == [0, 1, 2, 0, 0, 1]
    assert md.cu_seqlens_q == [0, 3, 4, 6]
    assert (md.num_reqs, md.total_q, md.is_prompt) == (3, 6, True)
    assert md.last_token_indices() == [2, 3, 5]
    with pytest.raises(ValueError):
        InputMetadata.for_prompts([[1], []], device="cpu")
