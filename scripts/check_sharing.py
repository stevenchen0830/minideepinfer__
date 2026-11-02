"""Block sharing on the real weights (S2.2): fork with copy-on-write and the prefix cache.

Usage:
    uv run scripts/check_sharing.py --model Qwen/Qwen3-0.6B
    uv run scripts/check_sharing.py --model Qwen/Qwen3-1.7B
    uv run scripts/check_sharing.py --model Qwen/Qwen3-0.6B --block-size 128

Teacher-forced rows are judged by check_generate.py's B rule: within twice
HF's own distance from the golden logits (HF's decode path, run now, on the
same rows), and greedy disagreements only at near-ties. Every step holds the 8
prompts of one group (parents, or one kind of fork), the batch size that rule
was calibrated on in S1.3. The default block size is 16, so that sequences
share partly filled blocks; with 128 every golden prompt fits in one block and
the prefix cache (F) cannot hit.

D. Fork after the prompt. Every prompt is prefilled and decoded for FORK_AT
   tokens, teacher-forced; then each sequence gets two forks. From then on
   each decode round steps the parents first (they write into the tail block
   they still share with the waiting forks), then the forks:
   - parents: must pass the golden rule on every row, so no fork wrote into
     their blocks;
   - "same" forks: no token replaced, teacher-forced on the rest of the golden
     tokens. They read the parent's KV through the shared blocks and their own
     copy of the tail block, with the same batch shapes as the parents, so
     their logits must equal the parents' bit for bit;
   - "other" forks: the last token replaced by golden's second choice at that
     row, then free-running greedy. Compared with HF's decode path fed the
     same tokens: within OURS_VS_HF_TOL, or twice HF's own distance between
     its decode path and one full forward when that is larger; and wherever
     HF's greedy token differs from ours, HF's top-2 margin is a near-tie.
E. Fork during a chunked prefill. Every prompt is prefilled up to half its
   length and forked with a "_" branch. The parents finish the prompt first,
   writing into blocks the forks still hold; then the forks copy their tail
   and finish theirs; both decode, parents first in each round. The parents
   must pass the golden rule, and the forks must equal them bit for bit.
F. Prefix cache. The prompts run once, which leaves their full blocks cached;
   then each prompt plus its first FORK_AT golden tokens comes back as a new
   request. It hits floor((len - 1) / block_size) blocks, prefills only the
   rest, and its rows (golden rows FORK_AT on) must pass the golden rule.
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from check_generate import hf_decode_logits, mismatches
from check_model import FULL_FLOOR, GOLDEN_DIR, NEAR_TIE, OURS_VS_HF_TOL, TOP5_FLOOR, load_golden
from mini_deepinfer.engine.static_engine import StaticEngine
from mini_deepinfer.model_executor.utils import init_torch
from mini_deepinfer.sampling_params import SamplingParams
from mini_deepinfer.worker.cache_engine import CacheEngine
from mini_deepinfer.worker.model_runner import ModelRunner

FORK_AT = 4  # decode steps before forking; F's requests add this many golden tokens


def sp(n, golden=None):
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, golden_token_ids=golden)


def rows_from(g, start):
    """The golden file restricted to rows start..end."""
    return {k: g[k][start:] for k in ("topk_ids", "topk_vals", "gen_ids", "logits_first")}


def vs_golden(logits, g):
    """Max |diff| on the golden top-5 (all rows) and on all logits (rows with full golden logits)."""
    top5 = (logits.gather(1, g["topk_ids"].long()) - g["topk_vals"]).abs().amax(1)
    n = g["logits_first"].shape[0]
    full = (logits[:n] - g["logits_first"].float()).abs().max().item() if n else 0.0
    return top5.max().item(), full


def golden_rule(ours, hf, g):
    """(passed, ours top-5, ours full, HF top-5, HF full, greedy diffs)."""
    o5, of = vs_golden(ours, g)
    h5, hf_full = vs_golden(hf, g)
    mm = mismatches(ours, g)
    passed = o5 <= max(2 * h5, TOP5_FLOOR) and of <= max(2 * hf_full, FULL_FLOOR) and all(m <= NEAR_TIE for _, m in mm)
    return passed, o5, of, h5, hf_full, mm


@torch.inference_mode()
def hf_full_logits(hf, prompt, gen, device):
    """HF's other path: the whole sequence in one forward, the same rows as hf_decode_logits."""
    seq = prompt + gen[:-1]
    logits = hf(input_ids=torch.tensor([seq], device=device)).logits[0].float()
    return logits[len(prompt) - 1 :].cpu()


class Recorder:
    """Steps the engine and keeps each sequence's logits rows (float32, CPU)."""

    def __init__(self, engine):
        self.engine = engine
        self.rows = {}

    def step(self, seqs, *, is_prompt, seqlens_q=None, keep=True):
        out = self.engine.step(seqs, is_prompt=is_prompt, seqlens_q=seqlens_q)
        if keep:
            for seq, row in zip(seqs, out.logits):
                self.rows.setdefault(seq.seq_id, []).append(row.float().cpu())

    def of(self, seq):
        return torch.stack(self.rows[seq.seq_id])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
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
    n = len(gens[0])
    if any(len(x) != n for x in gens) or max(len(p) for p in prompts) + n > args.max_model_len:
        sys.exit("golden generations of different lengths, or --max-model-len too small")
    bs = args.block_size
    eos_token_id = json.loads((Path(model_dir) / "config.json").read_text())["eos_token_id"]

    runner = ModelRunner(model_dir, block_size=bs, max_model_len=args.max_model_len, device=args.device)
    runner.load_model()
    num_blocks = 3 * len(prompts) * math.ceil(args.max_model_len / bs) + 1  # parents + two forks each, + block 0
    cache = CacheEngine(runner.model.cfg, num_blocks=num_blocks, block_size=bs, device=args.device)
    print(f"{args.model}: {len(prompts)} prompts x {n} tokens, block_size {bs}, {num_blocks} blocks, device {args.device}")

    # --- D: fork after the prompt ----------------------------------------------
    engine = StaticEngine(runner, cache, eos_token_id=eos_token_id)
    rec = Recorder(engine)
    parents = engine.add_requests(prompts, [sp(n, g) for g in gens])
    rec.step(parents, is_prompt=True)
    for _ in range(FORK_AT):
        rec.step(parents, is_prompt=False)
    alts = [int(g["topk_ids"][FORK_AT][1]) for g in golden]  # golden's second choice at the fork row
    rest = n - FORK_AT - 1
    same = [engine.fork_seq(p, sampling_params=sp(rest, gen[FORK_AT + 1 :])) for p, gen in zip(parents, gens)]
    other = [engine.fork_seq(p, sampling_params=sp(rest), replace_last_token_id=a) for p, a in zip(parents, alts)]
    for _ in range(rest):
        for group in (parents, same, other):
            rec.step(group, is_prompt=False)
    d_parent = [rec.of(s) for s in parents]
    d_same = [rec.of(s) for s in same]
    d_other = [rec.of(s) for s in other]
    other_prompts = [s.get_token_ids()[: s.get_prompt_len()] for s in other]
    other_gens = [s.get_token_ids()[s.get_prompt_len() :] for s in other]
    assert engine.block_manager.get_num_free_blocks() == num_blocks - 1, "blocks not all returned after D"
    print(f"D: {len(prompts)} parents, {2 * len(prompts)} forks at row {FORK_AT}, all blocks returned")

    # --- E: fork during a chunked prefill ---------------------------------------
    engine = StaticEngine(runner, cache, eos_token_id=eos_token_id)
    rec = Recorder(engine)
    parents_e = engine.add_requests(prompts, [sp(n, g) for g in gens])
    halves = [len(p) // 2 for p in prompts]
    rec.step(parents_e, is_prompt=True, seqlens_q=halves, keep=False)  # these rows are not golden rows
    forks_e = [engine.fork_seq(p, None, "_half", sampling_params=sp(n, g)) for p, g in zip(parents_e, gens)]
    tails = sum(h % bs != 0 for h in halves)
    rec.step(parents_e, is_prompt=True)
    rec.step(forks_e, is_prompt=True)
    for _ in range(n - 1):
        rec.step(parents_e, is_prompt=False)
        rec.step(forks_e, is_prompt=False)
    e_parent = [rec.of(s) for s in parents_e]
    e_fork = [rec.of(s) for s in forks_e]
    assert engine.block_manager.get_num_free_blocks() == num_blocks - 1, "blocks not all returned after E"
    print(f"E: forked at half of each prompt; {tails}/{len(prompts)} forks copied a partly filled tail block")

    # --- F: prefix cache ----------------------------------------------------------
    engine = StaticEngine(runner, cache, eos_token_id=eos_token_id, enable_prefix_caching=True)
    rec = Recorder(engine)
    first = engine.add_requests(prompts, [sp(n, g) for g in gens])
    rec.step(first, is_prompt=True, keep=False)
    for _ in range(n - 1):
        rec.step(first, is_prompt=False, keep=False)
    cached_blocks = engine.block_manager.num_cached_blocks()
    second = engine.add_requests([p + g[:FORK_AT] for p, g in zip(prompts, gens)], [sp(n - FORK_AT, g[FORK_AT:]) for g in gens])
    rec.step(second, is_prompt=True)
    hits = [s.cached_hit_tokens for s in second]
    for _ in range(n - FORK_AT - 1):
        rec.step(second, is_prompt=False)
    f_rows = [rec.of(s) for s in second]
    expected_hits = [(len(p) + FORK_AT - 1) // bs * bs for p in prompts]
    print(f"F: {cached_blocks} blocks cached after the first run; hit tokens {hits} (expected {expected_hits})")

    del engine, rec, cache, runner
    gc.collect()
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # --- HF's decode path ----------------------------------------------------------
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(args.device).eval()
    hf_rows = [hf_decode_logits(hf, p, gen, args.device) for p, gen in zip(prompts, gens)]
    hf_other = [hf_decode_logits(hf, p, gen, args.device) for p, gen in zip(other_prompts, other_gens)]
    hf_other_full = [hf_full_logits(hf, p, gen, args.device) for p, gen in zip(other_prompts, other_gens)]
    del hf
    gc.collect()

    ok = hits == expected_hits
    report = dict(model=args.model, block_size=bs, fork_at=FORK_AT, D=[], E=[], F=dict(hits=hits, expected=expected_hits, rows=[]))

    def show(label, i, res):
        passed, o5, of, h5, hf_full, mm = res
        print(f"  p{i:<3} {label:7} top-5 {o5:6.3f} (HF {h5:6.3f})  full {of:6.3f} (HF {hf_full:6.3f})  "
              f"greedy diffs {mm}{'' if passed else '  <-- FAIL'}")
        return dict(prompt=i, part=label, passed=passed, top5=o5, full=of, hf_top5=h5, hf_full=hf_full, mismatches=mm)

    def bitwise(label, i, fork_rows, parent_rows):
        diff = (fork_rows - parent_rows).abs().max().item()
        passed = diff == 0
        print(f"  p{i:<3} {label:7} vs parent, same rows: max |diff| {diff:.3g}{'' if passed else '  <-- FAIL (not bit-identical)'}")
        return passed, dict(prompt=i, part=label, passed=passed, vs_parent=diff)

    start = FORK_AT + 1
    print(f"\nD. fork after row {FORK_AT}: parents by the golden rule; 'same' forks equal to their parents (rows {start} on); "
          "'other' forks vs HF decode")
    for i, g in enumerate(golden):
        res = golden_rule(d_parent[i], hf_rows[i], g)
        ok &= res[0]
        report["D"].append(show("parent", i, res))
        passed, entry = bitwise("same", i, d_same[i], d_parent[i][start:])
        ok &= passed
        report["D"].append(entry)
        diff = (d_other[i] - hf_other[i]).abs().max().item()
        floor = (hf_other[i] - hf_other_full[i]).abs().max().item()
        top2 = hf_other[i].topk(2, dim=-1).values
        hf_margin = top2[:, 0] - top2[:, 1]
        disagree = (hf_other[i].argmax(-1) != torch.tensor(other_gens[i])).nonzero().flatten().tolist()
        dis = [(j, round(hf_margin[j].item(), 4)) for j in disagree]
        passed = diff <= max(OURS_VS_HF_TOL, 2 * floor) and all(m <= NEAR_TIE for _, m in dis)
        ok &= passed
        print(f"  p{i:<3} other   vs HF decode max |diff| {diff:6.3f} (HF decode vs HF full forward {floor:6.3f}); "
              f"HF greedy differs at (row, HF margin) {dis}{'' if passed else '  <-- FAIL'}")
        report["D"].append(dict(prompt=i, part="other", passed=passed, vs_hf=diff, hf_floor=floor, disagreements=dis))

    print("\nE. fork at half of the prompt ('_' branch): parents by the golden rule; forks equal to their parents")
    for i, g in enumerate(golden):
        res = golden_rule(e_parent[i], hf_rows[i], g)
        ok &= res[0]
        report["E"].append(show("parent", i, res))
        passed, entry = bitwise("fork", i, e_fork[i], e_parent[i])
        ok &= passed
        report["E"].append(entry)

    print(f"\nF. prefix-cache hit, then rows {FORK_AT} on: golden rule"
          f"{'' if hits == expected_hits else '  <-- FAIL: unexpected hit counts'}")
    for i, g in enumerate(golden):
        res = golden_rule(f_rows[i], hf_rows[i][FORK_AT:], rows_from(g, FORK_AT))
        ok &= res[0]
        report["F"]["rows"].append(show("hit", i, res))

    out = GOLDEN_DIR / f"{stem}.check-sharing-bs{bs}.json"
    out.write_text(json.dumps(dict(passed=ok, **report), indent=2))
    print(f"{'PASS' if ok else 'FAIL'}  (report: {out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
