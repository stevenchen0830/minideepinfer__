"""Save golden outputs from Hugging Face transformers for mini-DeepInfer (S0.3).

Usage:
    uv run scripts/make_golden.py --model Qwen/Qwen3-0.6B
    uv run scripts/make_golden.py --model Qwen/Qwen3-1.7B

For one model this script:
  1. runs greedy decoding on a fixed prompt list, one prompt at a time (batch
     size 1), and saves prompt ids, generated ids, full-vocabulary logits for
     the first few generated positions, top-k logits and the logsumexp for
     every position;
  2. reruns the same prompts as one left-padded batch and compares (QC2);
  3. measures GPU memory (QC3) and records where the model files are (QC4);
  4. times decode speed at batch size 1 (QC5).

Tensors go to golden/<model>.safetensors, the measurements to golden/<model>.json.
"""

import argparse
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Raw text, no chat template: the model sees exactly these strings.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In digital hardware design, a FIFO is used to",
    "1, 1, 2, 3, 5, 8, 13,",
    "人工智能的发展历史可以追溯到",
    "请用一句话解释什么是缓存：",
    (
        "Large language model inference has two phases. During prefill, the model "
        "processes the whole prompt at once; during decode, it produces one token at "
        "a time and must read every weight from GPU memory for each token. Because "
        "of this, decode speed is usually limited by"
    ),
    (
        "在芯片验证中，工程师通常先根据设计规格编写验证计划，然后搭建测试平台，"
        "用随机激励和功能覆盖率来衡量验证是否充分。一个好的参考模型应该"
    ),
]

MAX_NEW_TOKENS = 32
FULL_LOGIT_POSITIONS = 8  # full-vocabulary logits for the first K generated positions
TOP_K = 5  # top-k logits for every generated position
TIMING_PROMPT = 6  # index of the prompt used for the speed measurement
TIMING_NEW_TOKENS = 64
TIMING_REPEATS = 5

EOS_RULE = (
    "Generation stops at the first EOS token. The EOS token is kept as the last "
    "generated id, so a prompt can have fewer than max_new_tokens generated ids. "
    "Logits exist only for positions that were generated."
)
ROW_MEANING = (
    "Row j of logits_first, topk_* and lse holds the next-token logits after "
    "prompt_ids + gen_ids[:j]. Row 0 comes from prefill (last prompt position); "
    "rows >= 1 come from decode steps with the KV cache."
)

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def load(repo_id: str):
    """Download (or reuse) the snapshot, then load the tokenizer and a BF16 model on the GPU."""
    path = Path(snapshot_download(repo_id))
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    return path, tok, model


@torch.inference_mode()
def greedy(model, input_ids, attention_mask, max_new_tokens, min_new_tokens=0, with_logits=True):
    # Qwen3's generation_config.json asks for sampling (temperature, top-p, top-k).
    # do_sample=False overrides it: those settings are only used when sampling.
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        return_dict_in_generate=True,
        output_logits=with_logits,
    )


def run_single(model, tok):
    """Batch size 1, one prompt at a time: this is the golden reference."""
    records = []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt").to("cuda")
        out = greedy(model, enc.input_ids, enc.attention_mask, MAX_NEW_TOKENS)
        n_prompt = enc.input_ids.shape[1]
        gen = out.sequences[0, n_prompt:]
        logits = torch.stack(out.logits)[:, 0, :]  # [generated, vocab], float32
        # Greedy means every generated token is the argmax of the raw logits.
        if not torch.equal(logits.argmax(dim=-1), gen):
            raise RuntimeError("generated tokens are not the argmax: generation is not greedy")
        # Keep results on the CPU so they do not count in later GPU memory peaks.
        records.append({"prompt_ids": enc.input_ids[0].cpu(), "gen_ids": gen.cpu(), "logits": logits.cpu()})
        del out
    return records


def run_batch(model, tok, records):
    """All prompts as one left-padded batch, compared with the batch-size-1 results (QC2)."""
    # Decoder-only: new tokens are appended on the right, so the padding goes on the left.
    enc = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    out = greedy(model, enc.input_ids, enc.attention_mask, MAX_NEW_TOKENS)
    padded_len = enc.input_ids.shape[1]
    seqs = out.sequences[:, padded_len:].cpu()
    logits_b = torch.stack(out.logits).cpu()  # [generated, batch, vocab]
    del out
    results, batch_ids = [], []
    for i, ref in enumerate(records):
        ref_gen, ref_logits = ref["gen_ids"], ref["logits"]
        gen_b = seqs[i, : len(ref_gen)]
        batch_ids.append(gen_b)
        n = min(len(ref_gen), len(gen_b))
        mismatches = (gen_b[:n] != ref_gen[:n]).nonzero().flatten().tolist()
        first_bad = mismatches[0] if mismatches else None
        # Logits at position j depend only on tokens before j, so they stay
        # comparable up to and including the first token that differs.
        last = n if first_bad is None else first_bad + 1
        diff = (logits_b[:last, i, :] - ref_logits[:last]).abs()
        top5 = ref_logits[:last].topk(5, dim=-1)
        margins = top5.values[:, 0] - top5.values[:, 1]
        entry = {
            "prompt": i,
            "tokens_equal": first_bad is None and len(gen_b) >= len(ref_gen),
            "first_mismatch": first_bad,
            "max_abs_logit_diff": float(diff.max()),
            "max_abs_diff_on_ref_top5": float(diff.gather(1, top5.indices).max()),
            "bitwise_equal_logits": bool((diff == 0).all()),
            "max_abs_diff_per_position": [round(float(x), 4) for x in diff.amax(dim=-1)],
            "ref_top2_margin_per_position": [round(float(x), 4) for x in margins],
        }
        if first_bad is not None:
            entry["top2_margin_at_mismatch"] = float(margins[first_bad])
        results.append(entry)
    return results, batch_ids


def peak_bytes(model, enc):
    """Peak GPU memory of one generate() call that keeps no logits (QC3)."""
    torch.cuda.reset_peak_memory_stats()
    greedy(model, enc.input_ids, enc.attention_mask, MAX_NEW_TOKENS, with_logits=False)
    return torch.cuda.max_memory_allocated()


def decode_speed(model, tok):
    """Decode tokens per second at batch size 1 (QC5): (N - 1) / (time(N) - time(1))."""
    enc = tok(PROMPTS[TIMING_PROMPT], return_tensors="pt").to("cuda")

    def run(n):
        torch.cuda.synchronize()  # wait for earlier GPU work before starting the clock
        t0 = time.perf_counter()
        greedy(model, enc.input_ids, enc.attention_mask, n, min_new_tokens=n, with_logits=False)
        torch.cuda.synchronize()  # wait for this run's GPU work before stopping the clock
        return time.perf_counter() - t0

    run(TIMING_NEW_TOKENS)  # warm-up: kernel selection, allocator growth
    # Interleave the short and long runs so clock or temperature drift hits both.
    pairs = [(run(1), run(TIMING_NEW_TOKENS)) for _ in range(TIMING_REPEATS)]
    rates = [(TIMING_NEW_TOKENS - 1) / (t_all - t_first) for t_first, t_all in pairs]
    return {
        "prompt_tokens": int(enc.input_ids.shape[1]),
        "new_tokens": TIMING_NEW_TOKENS,
        "decode_tokens_per_s": statistics.median(rates),
        "decode_tokens_per_s_all_repeats": [round(r, 2) for r in rates],
        "median_s_prefill_plus_first_token": statistics.median(p[0] for p in pairs),
    }


def save_golden(stem: str, records, batch_ids, meta) -> tuple[Path, list[bool]]:
    tensors = {}
    lossless = []
    for i, r in enumerate(records):
        logits = r["logits"]
        first = logits[:FULL_LOGIT_POSITIONS]
        # The model computes logits in BF16 and generate() upcasts them to FP32,
        # so BF16 storage should lose nothing. Check instead of assuming.
        exact = torch.equal(first.to(torch.bfloat16).float(), first)
        lossless.append(exact)
        top_vals, top_ids = logits.topk(TOP_K, dim=-1)
        tensors[f"p{i}.prompt_ids"] = r["prompt_ids"].to(torch.int32)
        tensors[f"p{i}.gen_ids"] = r["gen_ids"].to(torch.int32)
        tensors[f"p{i}.gen_ids_batch8"] = batch_ids[i].to(torch.int32)
        tensors[f"p{i}.logits_first"] = (first.to(torch.bfloat16) if exact else first).contiguous()
        tensors[f"p{i}.topk_vals"] = top_vals.contiguous()
        tensors[f"p{i}.topk_ids"] = top_ids.to(torch.int32).contiguous()
        tensors[f"p{i}.lse"] = torch.logsumexp(logits, dim=-1)
    path = GOLDEN_DIR / f"{stem}.safetensors"
    save_file(tensors, str(path), metadata={"meta": json.dumps(meta, ensure_ascii=False)})
    return path, lossless


def files_info(path: Path):
    """Where the model lives in the Hugging Face cache and how big it is (QC4)."""
    files = [p for p in path.iterdir() if p.is_file()]
    weights = [p for p in files if p.suffix == ".safetensors"]
    return {
        "snapshot_dir": str(path),
        "repo_cache_dir": str(path.parents[1]),
        "weight_files_resolve_to": [str(p.resolve()) for p in weights],
        "weight_bytes": sum(p.resolve().stat().st_size for p in weights),
        "all_file_bytes": sum(p.resolve().stat().st_size for p in files),
        "files": sorted(p.name for p in files),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    GOLDEN_DIR.mkdir(exist_ok=True)
    stem = args.model.split("/")[-1]

    path, tok, model = load(args.model)
    after_load = torch.cuda.memory_allocated()

    # QC3 first, while nothing else of ours sits on the GPU.
    longest = max(PROMPTS, key=lambda p: len(tok(p).input_ids))
    peak_single = peak_bytes(model, tok(longest, return_tensors="pt").to("cuda"))
    peak_batch = peak_bytes(model, tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left").to("cuda"))

    records = run_single(model, tok)
    batch, batch_ids = run_batch(model, tok, records)
    speed = decode_speed(model, tok)

    meta = {
        "model": args.model,
        "commit": path.name,  # the snapshot folder name is the commit hash
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "dtype": "bfloat16",
        "attn_implementation": model.config._attn_implementation,
        "prompts": PROMPTS,
        "chat_template": "none: raw text prompts",
        "tokenizer": {"bos_added": tok.add_bos_token if hasattr(tok, "add_bos_token") else None,
                      "pad_token_id": tok.pad_token_id, "batch_padding_side": "left"},
        "decoding": "greedy, batch size 1, one prompt at a time",
        "max_new_tokens": MAX_NEW_TOKENS,
        "full_logit_positions": FULL_LOGIT_POSITIONS,
        "top_k": TOP_K,
        "eos_token_id": model.generation_config.eos_token_id,
        "hub_generation_config": model.generation_config.to_dict(),
        "eos_rule": EOS_RULE,
        "row_meaning": ROW_MEANING,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    golden_path, lossless = save_golden(stem, records, batch_ids, meta)

    files = files_info(path)
    report = {
        "golden_file": str(golden_path),
        "golden_file_bytes": golden_path.stat().st_size,
        "generated_lengths": [len(r["gen_ids"]) for r in records],
        "logits_first_stored_as_bf16_losslessly": lossless,
        "memory": {
            "weights_on_disk_bytes": files["weight_bytes"],
            "allocated_after_load_bytes": after_load,
            "peak_longest_prompt_batch1_bytes": peak_single,
            "peak_all_prompts_batch8_bytes": peak_batch,
        },
        "batch_vs_single": batch,
        "decode_speed_batch1": speed,
        "model_files": files,
        "meta": {k: v for k, v in meta.items() if k != "prompts"},
    }
    report_path = GOLDEN_DIR / f"{stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    gib = 1024**3
    print(f"model {args.model} @ {meta['commit'][:12]}  ({meta['gpu']}, torch {meta['torch']}, transformers {meta['transformers']})")
    for i, r in enumerate(records):
        text = tok.decode(r["gen_ids"], skip_special_tokens=False).replace("\n", "\\n")
        print(f"  p{i} prompt {len(r['prompt_ids']):3d} tok, generated {len(r['gen_ids']):2d}: {text[:70]}")
    print(f"golden file: {golden_path} ({golden_path.stat().st_size / 1e6:.1f} MB), BF16 lossless: {all(lossless)}")
    print(f"weights on disk {files['weight_bytes'] / gib:.2f} GiB | allocated after load {after_load / gib:.2f} GiB | "
          f"peak longest prompt {peak_single / gib:.2f} GiB | peak batch 8 {peak_batch / gib:.2f} GiB")
    same = sum(b["tokens_equal"] for b in batch)
    exact = sum(b["bitwise_equal_logits"] for b in batch)
    worst = max(b["max_abs_logit_diff"] for b in batch)
    worst_top5 = max(b["max_abs_diff_on_ref_top5"] for b in batch)
    print(f"batch 8 vs batch 1: tokens equal {same}/{len(batch)}, logits bitwise equal {exact}/{len(batch)}, "
          f"max |diff| {worst:.4g} (on the top-5 tokens {worst_top5:.4g})")
    for b in batch:
        if b["first_mismatch"] is not None:
            print(f"  p{b['prompt']}: first token mismatch at position {b['first_mismatch']}, "
                  f"top-2 margin there {b['top2_margin_at_mismatch']:.4g}")
    print(f"decode speed (batch 1, {speed['prompt_tokens']}-token prompt): {speed['decode_tokens_per_s']:.1f} tokens/s "
          f"(repeats {speed['decode_tokens_per_s_all_repeats']})")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
