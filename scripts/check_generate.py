"""Generate with the KV cache on the real weights and check against S0.3 (S1.3).

Usage:
    uv run scripts/check_generate.py --model Qwen/Qwen3-0.6B
    uv run scripts/check_generate.py --model Qwen/Qwen3-1.7B
    uv run scripts/check_generate.py --model Qwen/Qwen3-0.6B --block-size 16

Our side uses the static engine: one prefill step, then one decode step per
token, bfloat16, with DeepInfer's process settings (``init_torch``), which
also apply to the HF runs below. The golden files were made by HF
``generate`` one prompt at a time.

A. Free-running greedy, one prompt at a time: our tokens vs the golden
   tokens. Passes if every prompt either matches all the way or first differs
   at a near-tie (golden top-2 margin <= NEAR_TIE).
B. Teacher-forced decode, one prompt at a time: the golden tokens are fed back
   instead of our own (``SamplingParams.golden_token_ids``), so every row has
   the same history as the golden row. Compared with
     (a) the golden logits, and
     (b) HF's own decode path run now: prefill the prompt, then one token at a
         time with its KV cache, fed the same golden tokens.
   As in check_model.py, HF's (b) against (a) is the noise floor. A prompt
   passes when ours vs (b) stays within OURS_VS_HF_TOL, ours vs (a) within
   twice the floor (or 0.5 top-5 / 1.0 all logits when the floor is smaller),
   and every greedy disagreement sits at a near-tie. Also reported: our
   decode path vs our own prefill path (S1.2's full forward).
C. All prompts in one batch. Teacher-forced, every prompt must pass B's golden
   rule (within twice HF's noise floor, greedy disagreements only at
   near-ties); its distance from the batch-1 logits is reported. Free-running,
   the number of prompts with the same tokens as A is reported.

Also reported: decode speed at batch 1 and with the whole batch, next to the
HF ``generate`` speed measured in S0.3. Chunked prefill and the EOS and
max_model_len stop rules are covered by the tests (tests/test_engine.py),
not here.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from check_model import FULL_FLOOR, GOLDEN_DIR, NEAR_TIE, OURS_VS_HF_TOL, TOP5_FLOOR, load_golden
from mini_deepinfer.engine.static_engine import StaticEngine
from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.model_executor.utils import init_torch
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner


def run(engine, prompts, n_tokens, golden=None):
    """Prefill + (n_tokens - 1) decode steps. Returns tokens [B][n], logits
    [B, n, V] on the CPU, and the wall time of each decode step."""
    golden = golden or [None] * len(prompts)
    sps = [SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True, golden_token_ids=g) for g in golden]
    seqs = engine.add_requests(prompts, sps)
    logits, decode_times = [], []
    out = engine.step(seqs, is_prompt=True)
    logits.append(out.logits.cpu())
    for _ in range(1, n_tokens):
        t0 = time.perf_counter()
        out = engine.step(seqs, is_prompt=False)
        decode_times.append(time.perf_counter() - t0)  # the sampler's .tolist() waits for the GPU
        logits.append(out.logits.cpu())
    tokens = [s.get_token_ids()[s.get_prompt_len() :] for s in seqs]
    return tokens, torch.stack(logits, dim=1), decode_times


@torch.inference_mode()
def prefill_path_logits(model, prompt, gen):
    """S1.2's path: the whole sequence in one forward without a cache."""
    seq = prompt + gen[:-1]
    md = InputMetadata.for_prompts([seq], device=model.lm_head.weight.device)
    hidden = model(None, md)
    rows = list(range(len(prompt) - 1, len(seq)))
    return get_logits(hidden[rows], model.lm_head.weight, model.cfg.vocab_size).cpu()


@torch.inference_mode()
def hf_decode_logits(hf, prompt, gen, device):
    """HF's decode path fed the golden tokens: prefill, then one token at a time."""
    out = hf(input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)  # as generate() does
    rows, cache = [out.logits[0, -1].float()], out.past_key_values
    for token in gen[:-1]:
        out = hf(input_ids=torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True)
        rows.append(out.logits[0, -1].float())
        cache = out.past_key_values
    return torch.stack(rows).cpu()


def vs_golden(logits, g):
    """Max |diff| on the golden top-5 (all rows) and on all logits (first 8 rows)."""
    top5 = (logits.gather(1, g["topk_ids"].long()) - g["topk_vals"]).abs().amax(1)
    n = g["logits_first"].shape[0]
    full = (logits[:n] - g["logits_first"].float()).abs().max().item()
    return top5.max().item(), int(top5.argmax()), full


def margins(g):
    return g["topk_vals"][:, 0] - g["topk_vals"][:, 1]


def mismatches(logits, g):
    """Rows where our greedy token differs from the golden one, with the golden top-2 margin."""
    rows = (logits.argmax(-1) != g["gen_ids"].long()).nonzero().flatten().tolist()
    return [(j, round(margins(g)[j].item(), 4)) for j in rows]


def within_floor(logits, g, hf_top5, hf_full):
    """B's golden rule: within twice HF's own distance from golden, greedy differences only at near-ties."""
    top5, _, full = vs_golden(logits, g)
    near_ties = all(m <= NEAR_TIE for _, m in mismatches(logits, g))
    return top5 <= max(2 * hf_top5, TOP5_FLOOR) and full <= max(2 * hf_full, FULL_FLOOR) and near_ties


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=None, help="default: max-model-len (one contiguous block per sequence)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    init_torch()

    stem = args.model.split("/")[-1]
    _, golden = load_golden(stem)
    try:
        model_dir = snapshot_download(args.model, local_files_only=True)
    except Exception:
        model_dir = snapshot_download(args.model)
    prompts = [g["prompt_ids"].tolist() for g in golden]
    gens = [g["gen_ids"].tolist() for g in golden]
    n_tokens = len(gens[0])
    if any(len(x) != n_tokens for x in gens):
        sys.exit("golden generations have different lengths (an EOS); this script expects equal lengths")
    if max(len(p) for p in prompts) + n_tokens > args.max_model_len:
        sys.exit("--max-model-len too small for the golden prompts")

    # --- our engine ---------------------------------------------------------
    block_size = args.block_size or args.max_model_len
    runner = ModelRunner(model_dir, block_size=block_size, max_model_len=args.max_model_len, device=args.device)
    runner.load_model()
    blocks_per_seq = math.ceil(args.max_model_len / block_size)
    cache = CacheEngine(runner.model.cfg, num_blocks=len(prompts) * blocks_per_seq, block_size=block_size, device=args.device)
    eos_token_id = json.loads((Path(model_dir) / "config.json").read_text())["eos_token_id"]
    engine = StaticEngine(runner, cache, eos_token_id=eos_token_id)
    kv_mib = CacheEngine.get_cache_block_size(block_size, runner.model.cfg) * cache.num_blocks / 2**20
    print(f"{args.model}: {len(prompts)} prompts x {n_tokens} tokens, block_size {block_size}, "
          f"{cache.num_blocks} blocks ({kv_mib:.0f} MiB of KV cache), device {args.device}")
    run(engine, prompts[:1], 4)  # warm-up

    free_tokens, times1 = [], []
    for prompt in prompts:
        tokens, _, dt = run(engine, [prompt], n_tokens)
        free_tokens.append(tokens[0])
        times1.append(dt)
    forced = [run(engine, [p], n_tokens, golden=[gen])[1][0] for p, gen in zip(prompts, gens)]
    prefill_path = [prefill_path_logits(runner.model, p, gen) for p, gen in zip(prompts, gens)]
    batch_tokens, _, times8 = run(engine, prompts, n_tokens)
    batch_forced = run(engine, prompts, n_tokens, golden=gens)[1]
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # --- HF's decode path, fed the same golden tokens -------------------------
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(args.device).eval()
    hf_rows = [hf_decode_logits(hf, p, gen, args.device) for p, gen in zip(prompts, gens)]
    del hf
    gc.collect()

    ok = True
    report = dict(model=args.model, block_size=block_size, max_model_len=args.max_model_len, A=[], B=[], C={})

    print("\nA. free-running greedy, batch 1: tokens vs golden (first difference: row, golden top-2 margin)")
    for i, (tokens, g) in enumerate(zip(free_tokens, golden)):
        diff = next((j for j, (a, b) in enumerate(zip(tokens, gens[i])) if a != b), None)
        m = None if diff is None else round(margins(g)[diff].item(), 4)
        passed = diff is None or m <= NEAR_TIE
        ok &= passed
        print(f"p{i:<3} {'all 32 equal' if diff is None else f'first difference at row {diff}, margin {m}'}{'' if passed else '  <-- FAIL'}")
        report["A"].append(dict(prompt=i, first_diff=diff, margin=m, passed=passed))

    print("\nB. teacher-forced decode, batch 1")
    print(f"{'':4} {'ours vs golden':>22} | {'HF decode vs golden':>22} | {'ours vs HF':>10} | {'ours decode':>11} | greedy diffs")
    print(f"{'':4} {'top-5 (row)':>13} {'full':>8} | {'top-5 (row)':>13} {'full':>8} | {'full':>10} | {'vs prefill':>11} | (row, margin)")
    floors = []
    for i, (o, h, p, g) in enumerate(zip(forced, hf_rows, prefill_path, golden)):
        o5, orow, of = vs_golden(o, g)
        h5, hrow, hf_full = vs_golden(h, g)
        floors.append((h5, hf_full))
        oh = (o - h).abs().max().item()
        op = (o - p).abs().max().item()
        mm = mismatches(o, g)
        passed = oh <= OURS_VS_HF_TOL and within_floor(o, g, h5, hf_full)
        ok &= passed
        print(f"p{i:<3} {o5:7.3f} {f'({orow})':>5} {of:8.3f} | {h5:7.3f} {f'({hrow})':>5} {hf_full:8.3f} | {oh:10.3f} | {op:11.3f} | {mm}"
              f"{'' if passed else '  <-- FAIL'}")
        report["B"].append(dict(prompt=i, ours_top5=o5, ours_top5_row=orow, ours_full=of, hf_top5=h5, hf_top5_row=hrow,
                                hf_full=hf_full, ours_vs_hf=oh, decode_vs_prefill=op, mismatches=mm, passed=passed))

    same = [b == s for b, s in zip(batch_tokens, free_tokens)]
    firsts = [next((j for j, (a, b) in enumerate(zip(bt, st)) if a != b), None) for bt, st in zip(batch_tokens, free_tokens)]
    batch_diff = [(batch_forced[i] - forced[i]).abs().max().item() for i in range(len(prompts))]
    batch_ok = [within_floor(batch_forced[i], g, *floors[i]) for i, g in enumerate(golden)]
    c_passed = all(batch_ok)
    ok &= c_passed
    print(f"\nC. all {len(prompts)} prompts in one batch, teacher-forced: golden rule passed in "
          f"{sum(batch_ok)}/{len(batch_ok)} prompts{'' if c_passed else '  <-- FAIL'}; "
          f"max |diff| from batch 1 per prompt {[round(d, 3) for d in batch_diff]}")
    print(f"   free-running: same tokens as batch 1 in {sum(same)}/{len(same)} prompts; "
          f"first differences (prompt, row): {[(i, j) for i, j in enumerate(firsts) if j is not None]}")
    report["C"] = dict(batch_vs_single=batch_diff, passed=c_passed, same=sum(same), first_differences=firsts)

    speed1 = statistics.median((n_tokens - 1) / sum(dt) for dt in times1)
    speed_all = len(prompts) * (n_tokens - 1) / sum(times8)
    hf_speed = json.loads((GOLDEN_DIR / f"{stem}.json").read_text())["decode_speed_batch1"]["decode_tokens_per_s"]
    print(f"decode speed: batch 1 {speed1:.1f} tokens/s (HF generate in S0.3: {hf_speed:.1f}); "
          f"batch {len(prompts)} {speed_all:.1f} tokens/s in total")
    report["speed"] = dict(batch1=speed1, batch_all=speed_all, hf_batch1=hf_speed)

    out = GOLDEN_DIR / f"{stem}.check-generate-bs{block_size}.json"
    out.write_text(json.dumps(dict(passed=ok, **report), indent=2))
    print(f"{'PASS' if ok else 'FAIL'}  (report: {out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
