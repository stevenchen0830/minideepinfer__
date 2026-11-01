"""Check the hand-written Qwen3 against Hugging Face on the real weights (S1.2).

Usage:
    uv run scripts/check_model.py --model Qwen/Qwen3-0.6B
    uv run scripts/check_model.py --model Qwen/Qwen3-1.7B
    uv run scripts/check_model.py --model Qwen/Qwen3-0.6B --fp32
    uv run scripts/check_model.py --model Qwen/Qwen3-1.7B --fp32 --device cpu

Default (bfloat16, the way the model is served):
  For every golden prompt from S0.3, the model reads the prompt plus all but
  the last generated token in one forward pass (no KV cache yet) and the
  next-token logits are taken at every generated position (row j predicts
  generated token j). They are compared with
    (a) the golden logits saved by HF ``generate`` in S0.3, and
    (b) HF run live on the same full sequence (one forward, no cache).
  A prompt passes when
    - ours vs (b), all logits, stays within OURS_VS_HF_TOL (an empirical
      bound: the measured maximum was about 0.8);
    - ours vs (a) stays within twice HF's own (b)-vs-(a) difference, or within
      0.5 (top-5) / 1.0 (all logits) when that is smaller. HF's (b)-vs-(a) is
      the noise floor: the same model, weights and dtype as the golden run,
      but not run the same way;
    - every greedy mismatch against the golden tokens sits at a near-tie
      (golden top-2 margin <= NEAR_TIE, one bfloat16 step for logits below 64).

``--fp32``:
  Both models are cast to float32 and compared with each other. Measured
  max |ours - HF| is 2e-5 to 6e-5, and up to 3.5e-4 on one 1.7B prompt.
  FP32_TOL is a coarse line between that noise and a real bug, which moves
  logits by 0.1 or more; the fine-grained float32 check is
  tests/test_qwen3.py at 1e-4.

Both modes first apply DeepInfer's process settings
(``model_executor.utils.init_torch``), for HF too.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.model_executor.model_loader import load_model
from mini_deepinfer.model_executor.utils import init_torch

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
TOP5_FLOOR = 0.5  # pass bound on golden top-5 logits when HF's own noise is smaller
FULL_FLOOR = 1.0  # same for the whole vocabulary
OURS_VS_HF_TOL = 1.5  # ours vs HF live, same input and shapes (measured max about 0.8)
NEAR_TIE = 0.25  # largest golden top-2 margin at which a greedy mismatch is accepted
FP32_TOL = 1e-2  # measured noise <= 3.5e-4; a bug moves logits by 0.1 or more


def load_golden(stem):
    path = GOLDEN_DIR / f"{stem}.safetensors"
    if not path.exists():
        sys.exit(f"{path} not found: run scripts/make_golden.py --model ... first (S0.3)")
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["meta"])
        n = len(meta["prompts"])
        prompts = [{k: f.get_tensor(f"p{i}.{k}") for k in ("prompt_ids", "gen_ids", "logits_first", "topk_vals", "topk_ids", "lse")} for i in range(n)]
    return meta, prompts


def full_sequences(golden):
    """Prompt + generated tokens except the last, and the rows that predict each generated token."""
    seqs = []
    for g in golden:
        p, gen = g["prompt_ids"].tolist(), g["gen_ids"].tolist()
        seq = p + gen[:-1]
        rows = list(range(len(p) - 1, len(seq)))  # row j predicts gen[j]
        seqs.append((seq, rows))
    return seqs


@torch.inference_mode()
def ours_logits(model_dir, seqs, device, dtype):
    model, report = load_model(model_dir, device=device, dtype=dtype)
    cfg = model.cfg
    single = []
    for seq, rows in seqs:
        md = InputMetadata.for_prompts([seq], device=device)
        hidden = model(md)
        single.append(get_logits(hidden[rows], model.lm_head.weight, cfg.vocab_size).cpu())
    # All prompts packed into one forward pass (only used for the report).
    md = InputMetadata.for_prompts([s for s, _ in seqs], device=device)
    hidden = model(md)
    packed = []
    for (seq, rows), start in zip(seqs, md.cu_seqlens_q):
        idx = [start + r for r in rows]
        packed.append(get_logits(hidden[idx], model.lm_head.weight, cfg.vocab_size).cpu())
    del model, hidden
    free(device)
    return single, packed, report


@torch.inference_mode()
def hf_logits(model_dir, seqs, device, dtype):
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    out = []
    for seq, rows in seqs:
        logits = hf(input_ids=torch.tensor([seq], device=device), use_cache=False).logits[0]
        out.append(logits[rows].float().cpu())
    attn = hf.config._attn_implementation
    del hf
    free(device)
    return out, attn


def free(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def compare_to_golden(logits, g):
    """Max |diff| on the full rows (first 8), on the golden top-5 (all rows), on
    the logsumexp, and the row with the largest top-5 difference."""
    n_full = g["logits_first"].shape[0]
    full = (logits[:n_full] - g["logits_first"].float()).abs().max().item()
    top5_rows = (logits.gather(1, g["topk_ids"].long()) - g["topk_vals"]).abs().amax(dim=1)
    lse = (torch.logsumexp(logits, dim=-1) - g["lse"]).abs().max().item()
    return full, top5_rows.max().item(), lse, int(top5_rows.argmax())


def greedy_mismatches(logits, g):
    gen = g["gen_ids"].long()
    margins = g["topk_vals"][:, 0] - g["topk_vals"][:, 1]
    bad = (logits.argmax(-1) != gen).nonzero().flatten().tolist()
    return [(j, round(margins[j].item(), 4)) for j in bad]


def check_bf16(model_dir, golden, seqs, device):
    ours, packed, report = ours_logits(model_dir, seqs, device, None)
    hf, attn = hf_logits(model_dir, seqs, device, torch.bfloat16)
    rows, ok = [], True
    print(f"{'':4} {'ours vs golden':>36} | {'HF live vs golden':>21} | {'ours vs HF live':>15} | greedy mismatches (row, golden top-2 margin)")
    print(f"{'':4} {'full':>8} {'top-5':>8} {'(row)':>7} {'lse':>9} | {'full':>10} {'top-5':>10} | {'full':>15} |")
    for i, (o, h, g) in enumerate(zip(ours, hf, golden)):
        of, o5, ol, worst = compare_to_golden(o, g)
        hf_full, h5, _, _ = compare_to_golden(h, g)
        oh = (o - h).abs().max().item()
        mm = greedy_mismatches(o, g)
        hf_mm = greedy_mismatches(h, g)
        passed = (
            oh <= OURS_VS_HF_TOL
            and o5 <= max(2 * h5, TOP5_FLOOR)
            and of <= max(2 * hf_full, FULL_FLOOR)
            and all(margin <= NEAR_TIE for _, margin in mm)
        )
        ok &= passed
        print(f"p{i:<3} {of:8.4f} {o5:8.4f} {f'({worst})':>7} {ol:9.5f} | {hf_full:10.4f} {h5:10.4f} | {oh:15.4f} | "
              f"{mm} {'' if passed else '  <-- FAIL'}")
        rows.append(dict(prompt=i, ours_full=of, ours_top5=o5, ours_top5_worst_row=worst, ours_lse=ol, hf_full=hf_full,
                         hf_top5=h5, ours_vs_hf=oh, ours_mismatches=mm, hf_live_mismatches=hf_mm, passed=passed))
    pk = max((p - s).abs().max().item() for p, s in zip(packed, ours))
    same = sum(torch.equal(p.argmax(-1), s.argmax(-1)) for p, s in zip(packed, ours))
    print(f"packed (all prompts in one forward) vs one prompt per forward: max |diff| {pk:.4f}, "
          f"same greedy tokens in {same}/{len(ours)} prompts")
    print(f"load: {report.params} parameters from {report.tensors} tensors ({report.bytes / 2**30:.2f} GiB), "
          f"tied {report.aliases}, ignored {report.ignored}, unused {report.unused}")
    print(f"HF attention implementation: {attn}")
    return ok, dict(rows=rows, packed_vs_single_max_diff=pk, packed_same_greedy=same,
                    load=dict(params=report.params, tensors=report.tensors, bytes=report.bytes,
                              aliases=report.aliases, ignored=report.ignored, unused=report.unused))


def check_fp32(model_dir, golden, seqs, device):
    ours, _, _ = ours_logits(model_dir, seqs, device, torch.float32)
    hf, attn = hf_logits(model_dir, seqs, device, torch.float32)
    rows, ok = [], True
    for i, (o, h, g) in enumerate(zip(ours, hf, golden)):
        d = (o - h).abs().max().item()
        same = torch.equal(o.argmax(-1), h.argmax(-1))
        ok &= d <= FP32_TOL and same
        print(f"p{i}: max |ours - HF| {d:.2e} over {o.shape[0]} rows x {o.shape[1]} logits, same greedy tokens: {same}")
        rows.append(dict(prompt=i, max_abs_diff=d, same_greedy=same))
    print(f"HF attention implementation: {attn}")
    return ok, dict(rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--fp32", action="store_true", help="compare ours and HF, both in float32")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    init_torch()

    stem = args.model.split("/")[-1]
    meta, golden = load_golden(stem)
    try:
        model_dir = snapshot_download(args.model, local_files_only=True)
    except Exception:
        model_dir = snapshot_download(args.model)
    if Path(model_dir).name != meta["commit"]:
        print(f"warning: snapshot {Path(model_dir).name[:12]} differs from the golden file's {meta['commit'][:12]}")
    seqs = full_sequences(golden)
    print(f"{args.model}: {len(seqs)} prompts, {sum(len(r) for _, r in seqs)} positions, "
          f"device {args.device}, {'float32' if args.fp32 else 'bfloat16'}")

    if args.fp32:
        ok, result = check_fp32(model_dir, golden, seqs, args.device)
    else:
        ok, result = check_bf16(model_dir, golden, seqs, args.device)
    out = GOLDEN_DIR / f"{stem}.check-{'fp32' if args.fp32 else 'bf16'}.json"
    out.write_text(json.dumps(dict(model=args.model, device=args.device, passed=ok, **result), indent=2))
    print(f"{'PASS' if ok else 'FAIL'}  (report: {out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
