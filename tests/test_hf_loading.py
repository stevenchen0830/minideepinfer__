"""Config, name map and weight loading (S1.2), against checkpoints written by
Hugging Face transformers."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mini_deepinfer.hugging_face.config import load_hf_model_config
from mini_deepinfer.hugging_face.loading import cover_tied_parameters, prepare_hf_coverage
from mini_deepinfer.hugging_face.names import concat_index, hf_to_model, tied_aliases
from mini_deepinfer.hugging_face.weights import (
    INDEX_FILE,
    SINGLE_FILE,
    SafetensorsReader,
    load_weight_map,
    read_safetensors_header,
)
from mini_deepinfer.model_executor.layers.rotary_embedding import RotaryEmbedding
from mini_deepinfer.model_executor.model_loader import load_model

# --- config ------------------------------------------------------------------

QWEN3_06B_CONFIG = {  # the released file (transformers 4.51 spelling)
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def write_config(tmp_path, **changes):
    cfg = {**QWEN3_06B_CONFIG, **changes}
    cfg = {k: v for k, v in cfg.items() if v is not ...}  # `...` deletes a key
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_config_released_qwen3(tmp_path):
    cfg = load_hf_model_config(write_config(tmp_path))
    assert cfg.architecture == "Qwen3ForCausalLM"
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (1024, 16, 8, 128)
    assert (cfg.q_size, cfg.kv_size) == (2048, 1024)  # q_size != hidden_size
    assert cfg.rope_theta == 1_000_000 and cfg.tie_word_embeddings and cfg.dtype == torch.bfloat16


def test_config_transformers5_spelling(tmp_path):
    path = write_config(
        tmp_path,
        rope_theta=...,
        torch_dtype=...,
        rope_scaling=...,
        dtype="float32",
        rope_parameters={"rope_theta": 5e5, "rope_type": "default"},
    )
    cfg = load_hf_model_config(path)
    assert cfg.rope_theta == 5e5 and cfg.dtype == torch.float32


def test_config_optional_keys(tmp_path):
    cfg = load_hf_model_config(write_config(tmp_path, tie_word_embeddings=..., rms_norm_eps=1e-5))
    assert cfg.tie_word_embeddings is False and cfg.rms_norm_eps == 1e-5
    both = load_hf_model_config(write_config(tmp_path, rope_parameters={"rope_theta": 1000000, "rope_type": "default"}))
    assert both.rope_theta == 1_000_000


@pytest.mark.parametrize(
    "changes",
    [
        {"architectures": ["Qwen2ForCausalLM"]},
        {"architectures": ["Qwen3ForCausalLM", "Qwen3ForCausalLM"]},
        {"model_type": "qwen2"},
        {"torch_dtype": "float16"},
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"use_sliding_window": True},
        {"layer_types": ["full_attention", "sliding_attention"]},
        {"rope_scaling": {"type": "yarn", "factor": 4.0}},
        {"rope_theta": ..., "rope_parameters": {"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0}},
        {"rope_parameters": {"rope_theta": 5e5, "rope_type": "default"}},  # disagrees with rope_theta
        {"num_key_value_heads": 5},
        {"num_key_value_heads": ...},
        {"head_dim": ...},
        {"rope_theta": ...},
    ],
)
def test_config_refuses_unsupported(tmp_path, changes):
    with pytest.raises(ValueError):
        load_hf_model_config(write_config(tmp_path, **changes))


# --- name map ----------------------------------------------------------------


def test_hf_to_model_names():
    assert hf_to_model("model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert hf_to_model("model.norm.weight") == "model.norm.weight"
    assert hf_to_model("lm_head.weight") == "lm_head.weight"
    assert hf_to_model("model.layers.3.self_attn.q_norm.weight") == "model.layers.3.self_attn.q_norm.weight"
    for part in ("q", "k", "v"):
        assert hf_to_model(f"model.layers.12.self_attn.{part}_proj.weight") == "model.layers.12.self_attn.qkv_proj.weight"
    for part in ("gate", "up"):
        assert hf_to_model(f"model.layers.0.mlp.{part}_proj.weight") == "model.layers.0.mlp.gate_up_proj.weight"
    assert hf_to_model("model.layers.0.mlp.down_proj.weight") == "model.layers.0.mlp.down_proj.weight"
    assert hf_to_model("something.else") == "something.else"  # unknown names pass through


def test_concat_order_is_explicit_not_alphabetical():
    names = [f"model.layers.0.self_attn.{p}_proj.weight" for p in ("v", "k", "q")]
    assert sorted(names, key=concat_index) == [f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v")]
    assert sorted(names) != sorted(names, key=concat_index)  # sorted() alone gives k, q, v


def test_tied_aliases(tmp_path):
    tied = load_hf_model_config(write_config(tmp_path))
    assert tied_aliases(tied) == {"lm_head.weight": "model.embed_tokens.weight"}
    untied = load_hf_model_config(write_config(tmp_path, tie_word_embeddings=False))
    assert tied_aliases(untied) == {}


# --- safetensors reader --------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reader_matches_safe_open(tiny_checkpoint, dtype):
    _, path = tiny_checkpoint(shards=True, dtype=dtype)
    weight_map = load_weight_map(path)
    with SafetensorsReader(path, weight_map) as reader:
        for name, file in weight_map.items():
            with safe_open(Path(path) / file, "pt") as f:
                expected = f.get_tensor(name)
            got = reader.read(name)
            assert got.dtype == expected.dtype == dtype
            assert torch.equal(got, expected), name


def test_reader_handles_odd_tensors(tmp_path):
    tensors = {
        "empty": torch.empty(0, 3),
        "scalar": torch.tensor(2.5),
        "ints": torch.arange(7, dtype=torch.int64),
        "bf16": torch.randn(3, 5).bfloat16(),
    }
    save_file(tensors, str(tmp_path / SINGLE_FILE), metadata={"format": "pt"})
    header = read_safetensors_header(tmp_path / SINGLE_FILE)
    assert set(header) == set(tensors)  # __metadata__ skipped
    with SafetensorsReader(tmp_path, load_weight_map(tmp_path)) as reader:
        for name, t in tensors.items():
            assert torch.equal(reader.read(name), t), name


def test_weight_map_single_file_and_index(tiny_checkpoint):
    _, single = tiny_checkpoint(shards=False)
    _, sharded = tiny_checkpoint(shards=True)
    assert not (Path(single) / INDEX_FILE).exists()
    assert (Path(sharded) / INDEX_FILE).exists()
    a, b = load_weight_map(single), load_weight_map(sharded)
    assert set(a) == set(b)
    assert set(a.values()) == {SINGLE_FILE} and len(set(b.values())) > 3


# --- load plan -----------------------------------------------------------------


def test_coverage_merges_in_order(tiny_checkpoint):
    _, path = tiny_checkpoint(shards=True)
    cov = prepare_hf_coverage(hf_dir=path)
    assert cov.tensors["model.layers.1.self_attn.qkv_proj.weight"] == tuple(
        f"model.layers.1.self_attn.{p}_proj.weight" for p in ("q", "k", "v")
    )
    assert cov.tensors["model.layers.2.mlp.gate_up_proj.weight"] == (
        "model.layers.2.mlp.gate_proj.weight",
        "model.layers.2.mlp.up_proj.weight",
    )


def test_tied_copy_in_checkpoint_is_ignored(tiny_checkpoint):
    # save_pretrained omits a tied lm_head, so use an untied checkpoint to get a copy.
    _, path = tiny_checkpoint(tie_word_embeddings=False)
    cov = prepare_hf_coverage(hf_dir=path)
    tied = cover_tied_parameters(cov, {"lm_head.weight": "model.embed_tokens.weight"})
    assert "lm_head.weight" not in tied.tensors
    assert tied.ignored == ("lm_head.weight",)
    assert tied.aliases == {"lm_head.weight": "model.embed_tokens.weight"}
    with pytest.raises(RuntimeError):
        cover_tied_parameters(cov, {"lm_head.weight": "model.not_there.weight"})


# --- full load -------------------------------------------------------------------


@pytest.mark.parametrize("shards", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_loaded_weights_equal_hf(tiny_checkpoint, shards, tie):
    hf, path = tiny_checkpoint(shards=shards, tie_word_embeddings=tie)
    model, report = load_model(path, device="cpu")
    ours = dict(model.named_parameters(remove_duplicate=False))
    ref = dict(hf.named_parameters(remove_duplicate=False))
    for name, p in ours.items():
        if name.endswith("qkv_proj.weight"):
            pre = name.removesuffix("qkv_proj.weight")
            expected = torch.cat([ref[pre + f"{x}_proj.weight"] for x in "qkv"])
        elif name.endswith("gate_up_proj.weight"):
            pre = name.removesuffix("gate_up_proj.weight")
            expected = torch.cat([ref[pre + "gate_proj.weight"], ref[pre + "up_proj.weight"]])
        else:
            expected = ref[name]
        assert torch.equal(p, expected), name
        assert not p.requires_grad and p.device.type == "cpu"
    assert report.tensors == len(ref) - len(report.aliases)
    assert report.params == len(ours) - len(report.aliases)
    assert report.unused == [] and report.ignored == []
    if tie:
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert report.aliases == {"lm_head.weight": "model.embed_tokens.weight"}
    else:
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def copy_checkpoint(src, dst):
    shutil.copytree(src, dst)
    return dst


def rewrite_single_file(path, change):
    """Load model.safetensors, let ``change`` edit the dict, write it back."""
    file = Path(path) / SINGLE_FILE
    with safe_open(file, "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    change(tensors)
    save_file(tensors, str(file), metadata={"format": "pt"})


def test_missing_weight_is_an_error(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint()
    path = copy_checkpoint(src, tmp_path / "ckpt")
    rewrite_single_file(path, lambda t: t.pop("model.layers.1.post_attention_layernorm.weight"))
    with pytest.raises(RuntimeError, match="post_attention_layernorm"):
        load_model(path, device="cpu")


def test_missing_part_of_merged_weight_is_an_error(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint()
    path = copy_checkpoint(src, tmp_path / "ckpt")
    rewrite_single_file(path, lambda t: t.pop("model.layers.1.self_attn.k_proj.weight"))
    with pytest.raises(ValueError, match="k_proj"):
        load_model(path, device="cpu")


def test_wrong_shape_is_an_error(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint()
    path = copy_checkpoint(src, tmp_path / "ckpt")
    rewrite_single_file(path, lambda t: t.update({"model.layers.0.mlp.up_proj.weight": torch.zeros(81, 48)}))
    with pytest.raises(ValueError, match="gate_up_proj"):
        load_model(path, device="cpu")


def test_wrong_dtype_is_an_error_unless_cast(tiny_checkpoint, tmp_path):
    hf, src = tiny_checkpoint()
    path = copy_checkpoint(src, tmp_path / "ckpt")
    rewrite_single_file(path, lambda t: t.update({"model.norm.weight": t["model.norm.weight"].double()}))
    with pytest.raises(ValueError, match="model.norm.weight"):
        load_model(path, device="cpu")
    model, _ = load_model(path, device="cpu", dtype=torch.float32)  # explicit dtype allows the cast
    assert torch.equal(model.model.norm.weight, hf.model.norm.weight)


def test_extra_tensor_is_reported_not_loaded(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint()
    path = copy_checkpoint(src, tmp_path / "ckpt")
    rewrite_single_file(
        path,
        lambda t: t.update({"model.layers.99.mlp.down_proj.weight": torch.zeros(2), "model.rotary_emb.inv_freq": torch.zeros(8)}),
    )
    _, report = load_model(path, device="cpu")
    assert report.unused == ["model.layers.99.mlp.down_proj.weight", "model.rotary_emb.inv_freq"]


def test_truncated_shard_fails_before_building(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint(shards=True)
    path = copy_checkpoint(src, tmp_path / "ckpt")
    shard = sorted(Path(path).glob("model-*.safetensors"))[2]
    data = shard.read_bytes()
    shard.write_bytes(data[:-1])  # a download cut short by one byte
    with pytest.raises(ValueError, match="outside the data section"):
        prepare_hf_coverage(hf_dir=path)


@pytest.mark.parametrize("offsets", [[-8, 0], [0, 10**9]])
def test_header_offsets_must_lie_in_the_file(tmp_path, offsets):
    header = json.dumps({"t": {"dtype": "U8", "shape": [offsets[1] - offsets[0]], "data_offsets": offsets}}).encode()
    (tmp_path / SINGLE_FILE).write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 16)
    with pytest.raises(ValueError, match="outside the data section"):
        read_safetensors_header(tmp_path / SINGLE_FILE)


def test_header_length_is_checked(tmp_path):
    (tmp_path / SINGLE_FILE).write_bytes((10**6).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="header length"):
        read_safetensors_header(tmp_path / SINGLE_FILE)


def test_index_naming_a_missing_shard(tiny_checkpoint, tmp_path):
    _, src = tiny_checkpoint(shards=True)
    path = copy_checkpoint(src, tmp_path / "ckpt")
    shard = sorted(Path(path).glob("model-*.safetensors"))[1]
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        load_model(path, device="cpu")


def test_bf16_checkpoint_loads_as_bf16(tiny_checkpoint):
    hf, path = tiny_checkpoint(dtype=torch.bfloat16)
    model, _ = load_model(path, device="cpu")
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    rope = model.model.layers[0].self_attn.rotary_emb
    assert rope.cos_sin_cache.dtype == torch.float32
    # Built under the bf16 default dtype, yet identical to an fp32 build.
    assert torch.equal(rope.cos_sin_cache, RotaryEmbedding(16, 16, 256, 1_000_000.0, True).cos_sin_cache)
    assert torch.equal(model.model.norm.weight, hf.model.norm.weight)


def test_each_load_has_its_own_rope_cache(tiny_checkpoint):
    _, path = tiny_checkpoint()
    a, _ = load_model(path, device="cpu")
    b, _ = load_model(path, device="cpu")
    rope_a = a.model.layers[0].self_attn.rotary_emb
    rope_b = b.model.layers[0].self_attn.rotary_emb
    assert rope_a is not rope_b
    a.half()  # casting one model must not touch another
    assert rope_b.cos_sin_cache.dtype == torch.float32
    c, _ = load_model(path, device="cpu")
    assert c.model.layers[0].self_attn.rotary_emb.cos_sin_cache.dtype == torch.float32
