"""Checks that run before any weight bytes are read (S1.2).

Mirrors deepinfer/hugging_face/loading.py:
- ``prepare_hf_coverage``: every file the index names exists and has a
  readable header (a truncated download fails here), every indexed tensor
  really is in its file, then the load plan is built.
- ``cover_tied_parameters``: a tied parameter shares its source's storage,
  so it is never loaded. A checkpoint copy of it is taken out of the plan and
  reported as ``ignored``; a missing source raises here. Here the tie comes
  from ``tie_word_embeddings`` (``lm_head.weight`` -> ``model.embed_tokens.weight``).
  DeepInfer's version, ``(coverage, param_names, *, n_layer)``, finds its ties
  (only the MTP layers' copies) from the model's parameter names; their
  checkpoint copies never enter the plan, and a missing source surfaces later
  as an uncovered parameter.
- ``validate_hf_shapes``: shapes and dtypes are checked from the file headers
  only, so a wrong checkpoint fails before gigabytes are read.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from mini_deepinfer.hugging_face.names import merged_parts
from mini_deepinfer.hugging_face.weights import (
    HFCoverage,
    build_coverage,
    load_weight_map,
    read_safetensors_header,
)


def prepare_hf_coverage(*, hf_dir: str | Path) -> HFCoverage:
    hf_dir = Path(hf_dir)
    weight_map = load_weight_map(hf_dir)
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, file in weight_map.items():
        by_file[file].append(name)
    for file, names in sorted(by_file.items()):
        path = hf_dir / file
        if not path.exists():
            raise FileNotFoundError(f"checkpoint shard {path} is listed in the index but missing")
        present = read_safetensors_header(path)
        absent = sorted(set(names) - set(present))
        if absent:
            raise ValueError(f"{file}: indexed tensors not present in the file: {absent[:5]}")
    return build_coverage(weight_map)


def cover_tied_parameters(coverage: HFCoverage, aliases: dict[str, str]) -> HFCoverage:
    """Add the tied aliases; drop any checkpoint copy of a tied parameter."""
    tensors = dict(coverage.tensors)
    ignored = list(coverage.ignored)
    for target, source in aliases.items():
        if source not in tensors:
            raise RuntimeError(f"tied parameter {target} needs {source}, which the checkpoint does not have")
        ignored.extend(tensors.pop(target, ()))
    return replace(coverage, tensors=tensors, aliases={**coverage.aliases, **aliases}, ignored=tuple(ignored))


def validate_hf_shapes(
    params: dict[str, torch.Tensor],
    coverage: HFCoverage,
    *,
    hf_dir: str | Path,
    cast: bool = False,
) -> None:
    hf_dir = Path(hf_dir)
    headers = {}

    def info(hf_name):
        file = coverage.weight_map[hf_name]
        if file not in headers:
            headers[file] = read_safetensors_header(hf_dir / file)
        return headers[file][hf_name]

    for name, parts in sorted(coverage.tensors.items()):
        if name not in params:
            continue
        param = params[name]
        needed = merged_parts(name)
        if needed is not None and set(parts) != set(needed):
            missing = sorted(set(needed) - set(parts))
            raise ValueError(f"{name}: checkpoint is missing {missing}")
        infos = [info(p) for p in parts]
        for p, i in zip(parts, infos):
            if i.shape[1:] != tuple(param.shape[1:]):
                raise ValueError(f"{name}: part {p} has shape {i.shape}, parameter is {tuple(param.shape)}")
            if i.dtype != param.dtype and not cast:
                raise ValueError(f"{name}: part {p} has dtype {i.dtype}, parameter is {param.dtype}")
        rows = sum(i.shape[0] if i.shape else 1 for i in infos)
        expected = tuple(param.shape)
        got = (rows, *infos[0].shape[1:]) if len(parts) > 1 else infos[0].shape
        if got != expected:
            raise ValueError(f"{name}: checkpoint gives shape {got} from {list(parts)}, parameter is {expected}")
    for target, source in coverage.aliases.items():
        a, b = params[target], params[source]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"tied {target} {tuple(a.shape)} {a.dtype} != {source} {tuple(b.shape)} {b.dtype}")
