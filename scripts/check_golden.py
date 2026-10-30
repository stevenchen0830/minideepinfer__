"""Regenerate a golden file's outputs with Hugging Face transformers and compare (S0.3).

Usage:
    uv run scripts/check_golden.py golden/Qwen3-0.6B.safetensors

The prompts are fed as the saved token ids, so the check isolates the model
from the tokenizer; tokenizer drift is reported separately. Exit code 1 if any
generated token differs.
"""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_golden(path: Path):
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["meta"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return meta, tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("golden", type=Path, help="a golden/<model>.safetensors file")
    args = ap.parse_args()
    meta, tensors = read_golden(args.golden)

    now = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    for key, value in now.items():
        if meta.get(key) != value:
            print(f"note: {key} differs: golden {meta.get(key)!r}, now {value!r}")

    # Load exactly the commit and attention backend the golden file was made with.
    path = snapshot_download(meta["model"], revision=meta["commit"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=meta["attn_implementation"]).to("cuda").eval()

    k_full = meta["full_logit_positions"]
    failures, drift = 0, 0
    print(f"prompt  tokenizer  tokens  max|dlogit| first {k_full}  max|dlogit| top-k")
    for i, prompt in enumerate(meta["prompts"]):
        prompt_ids = tensors[f"p{i}.prompt_ids"].long()
        gold_gen = tensors[f"p{i}.gen_ids"].long()
        gold_first = tensors[f"p{i}.logits_first"].float()
        gold_ids = tensors[f"p{i}.topk_ids"].long()
        gold_vals = tensors[f"p{i}.topk_vals"]

        same_tok = torch.equal(tok(prompt, return_tensors="pt").input_ids[0], prompt_ids)
        drift += not same_tok
        ids = prompt_ids[None].cuda()
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=meta["max_new_tokens"], return_dict_in_generate=True, output_logits=True)
        gen = out.sequences[0, ids.shape[1]:].cpu()
        logits = torch.stack(out.logits)[:, 0, :].cpu()

        # Compare logits only while both runs have the same token history:
        # up to and including the first position where the tokens differ.
        n = min(len(gen), len(gold_gen))
        bad = (gen[:n] != gold_gen[:n]).nonzero().flatten().tolist()
        last = bad[0] + 1 if bad else n
        tokens_ok = not bad and len(gen) == len(gold_gen)
        rows = min(last, len(gold_first))
        dmax_first = float((logits[:rows] - gold_first[:rows]).abs().max())
        # Compare values at the golden top-k token ids. Comparing index order would
        # fail on ties, which are common because the logits are BF16 values.
        dmax_topk = float((logits[:last].gather(1, gold_ids[:last]) - gold_vals[:last]).abs().max())

        if not tokens_ok:
            failures += 1
        print(f"p{i:<5}  {'same' if same_tok else 'DRIFT':<9}  {'same' if tokens_ok else 'DIFFER':<6}  "
              f"{dmax_first:<20.4g}  {dmax_topk:.4g}")
        if bad:
            j = bad[0]
            margin = float(gold_vals[j, 0] - gold_vals[j, 1])
            print(f"        first mismatch at position {j}, golden top-2 margin there {margin:.4g}")

    print(f"tokenizer drift on {drift}/{len(meta['prompts'])} prompts")
    print("OK: all generated tokens identical" if failures == 0 else f"FAIL: {failures} prompt(s) differ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
