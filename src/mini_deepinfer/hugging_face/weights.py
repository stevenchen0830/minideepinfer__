"""Read safetensors checkpoints and copy them into model parameters (S1.2).

Mirrors deepinfer/hugging_face/weights.py:
- ``HFCoverage`` is the load plan: for each model parameter, the checkpoint
  tensors that fill it (two or three of them for a merged parameter).
- ``SafetensorsReader`` (DeepInfer: ``_TensorReader``) parses the file format
  by hand instead of using ``safetensors.safe_open``. The format is small:
  an 8-byte little-endian header length, a JSON header that gives each
  tensor's dtype, shape and byte range, then the raw bytes. The header
  checks are DeepInfer's: the header length and every byte range must lie
  inside the file. Each tensor is read into CPU memory and then copied to the
  target device.
- ``load_covered_params`` reads every planned tensor (concatenating merged
  ones along dim 0), checks its shape and copies it into the parameter.

Deliberate differences from DeepInfer:
- DeepInfer finds the files only through ``model.safetensors.index.json``.
  Qwen3-0.6B ships a single ``model.safetensors`` with no index, so a
  single-file checkpoint is accepted too.
- DeepInfer never casts: the file dtype must equal the parameter dtype. Here
  ``cast=True`` allows a cast after reading; ``load_model(dtype=...)`` uses
  it for the float32 check of real bfloat16 weights.
- A merged parameter with a missing part (say no ``k_proj``) stays in the
  plan, and ``validate_hf_shapes`` raises ValueError naming the missing
  tensor. DeepInfer's ``build_coverage`` drops an incomplete group, so its
  coverage check raises RuntimeError for the whole parameter instead.
- No FP8 scales, routed experts or Engram tables.
"""

from __future__ import annotations

import json
import math
import os
import struct
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from mini_deepinfer.hugging_face.names import concat_index, hf_to_model
from mini_deepinfer.hugging_face.parameters import model_params

MAX_HEADER_BYTES = 100_000_000  # the same cap as DeepInfer and the safetensors library

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class TensorInfo:
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int  # absolute byte offset in the file
    nbytes: int


def read_safetensors_header(path: str | Path) -> dict[str, TensorInfo]:
    """Parse the header of one safetensors file. Reads no tensor data."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size < 8:
            raise ValueError(f"{path}: {size} bytes is too short for a safetensors file")
        (header_len,) = struct.unpack("<Q", f.read(8))
        if not 0 < header_len <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError(f"{path}: invalid header length {header_len} for a file of {size} bytes")
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    infos = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] not in DTYPES:
            raise ValueError(f"{path}: tensor {name} has unsupported dtype {entry['dtype']}")
        dtype = DTYPES[entry["dtype"]]
        shape = tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        if not 0 <= begin <= end <= size - data_start:
            raise ValueError(f"{path}: tensor {name} bytes [{begin}, {end}) lie outside the data section")
        nbytes = math.prod(shape) * dtype.itemsize
        if end - begin != nbytes:
            raise ValueError(f"{path}: tensor {name} has {end - begin} bytes, expected {nbytes} for {shape} {dtype}")
        infos[name] = TensorInfo(dtype, shape, data_start + begin, nbytes)
    return infos


def load_weight_map(hf_dir: str | Path) -> dict[str, str]:
    """Checkpoint tensor name -> file name, from the index or the single file."""
    hf_dir = Path(hf_dir)
    index = hf_dir / INDEX_FILE
    if index.exists():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    single = hf_dir / SINGLE_FILE
    if single.exists():
        return {name: SINGLE_FILE for name in read_safetensors_header(single)}
    raise FileNotFoundError(f"{hf_dir} has neither {INDEX_FILE} nor {SINGLE_FILE}")


@dataclass(frozen=True)
class HFCoverage:
    """The load plan. All names on the left are model parameter names."""

    weight_map: dict[str, str]  # checkpoint name -> file name
    tensors: dict[str, tuple[str, ...]]  # parameter -> checkpoint names, in concat order
    aliases: dict[str, str] = field(default_factory=dict)  # tied parameter -> source parameter
    ignored: tuple[str, ...] = ()  # checkpoint names left out on purpose (tied copies)

    def model_names(self) -> set[str]:
        return set(self.tensors) | set(self.aliases)


def build_coverage(weight_map: dict[str, str]) -> HFCoverage:
    groups: dict[str, list[str]] = {}
    for hf_name in sorted(weight_map):
        groups.setdefault(hf_to_model(hf_name), []).append(hf_name)
    tensors = {name: tuple(sorted(parts, key=concat_index)) for name, parts in groups.items()}
    return HFCoverage(weight_map=dict(weight_map), tensors=tensors)


class SafetensorsReader(ExitStack):
    """Reads tensors by name; keeps one open file handle per shard."""

    def __init__(self, hf_dir: str | Path, weight_map: dict[str, str]):
        super().__init__()
        self.hf_dir = Path(hf_dir)
        self.weight_map = weight_map
        self._files = {}
        self._headers = {}

    def info(self, name: str) -> TensorInfo:
        file = self.weight_map[name]
        if file not in self._headers:
            self._headers[file] = read_safetensors_header(self.hf_dir / file)
        return self._headers[file][name]

    def read(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        info = self.info(name)
        file = self.weight_map[name]
        if file not in self._files:
            self._files[file] = self.enter_context(open(self.hf_dir / file, "rb"))
        f = self._files[file]
        data = bytearray(info.nbytes)
        f.seek(info.start)
        if f.readinto(data) != info.nbytes:
            raise ValueError(f"{file}: short read for tensor {name}")
        if info.nbytes == 0:
            tensor = torch.empty(info.shape, dtype=info.dtype)
        else:
            tensor = torch.frombuffer(data, dtype=info.dtype).reshape(info.shape)
        return tensor.to(device=device)

    def read_concat(self, names: tuple[str, ...], device: torch.device | str = "cpu") -> torch.Tensor:
        parts = [self.read(name, device) for name in names]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def load_covered_params(
    model: nn.Module,
    coverage: HFCoverage,
    *,
    hf_dir: str | Path,
    cast: bool = False,
) -> dict[str, int]:
    """Copy every planned checkpoint tensor into its parameter.

    Planned names that match no parameter are skipped here;
    ``model_loader.load_hf_checkpoint`` lists them in its report.
    """
    params = model_params(model)
    todo = {name: parts for name, parts in coverage.tensors.items() if name in params}
    stats = {"params": 0, "tensors": 0, "bytes": 0}
    with SafetensorsReader(hf_dir, coverage.weight_map) as reader:
        for name, parts in sorted(todo.items()):
            param = params[name]
            weight = reader.read_concat(parts, param.device)
            if tuple(weight.shape) != tuple(param.shape):
                raise ValueError(f"{name}: checkpoint shape {tuple(weight.shape)} != parameter shape {tuple(param.shape)}")
            if weight.dtype != param.dtype and not cast:
                raise ValueError(f"{name}: checkpoint dtype {weight.dtype} != parameter dtype {param.dtype}")
            param.data.copy_(weight)
            stats["params"] += 1
            stats["tensors"] += len(parts)
            stats["bytes"] += weight.numel() * weight.element_size()
    return stats
