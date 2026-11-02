"""Run one model step: prepare inputs, forward, sample (S1.3).

Mirrors deepinfer/worker/model_runner.py. There the worker makes two calls
per step: ``execute_main_model(seq_metadata_list, kv_caches)`` builds the
``InputMetadata`` itself (``_prepare_input_tensors``, with the index math in
C++, pinned CPU tensors, then ``.cuda(non_blocking=True)``), runs the model and
returns both; ``execute_sample`` then samples. Here input preparation is its
own public call, so tests can look at the metadata:
  ``prepare_input_tensors``  sequences -> ``InputMetadata``; tensors are built
                             in pinned CPU memory and copied to the GPU with
                             ``non_blocking=True``.
  ``execute_main_model``     ``model(kv_caches, input_metadata)`` -> hidden
                             states; the model computes no logits.
  ``execute_sample``         gathers the rows to sample from (the last query
                             token of each sequence; DeepInfer's
                             ``selected_token_indices``), computes float32
                             logits with ``get_logits`` and picks the tokens.

For every sequence the step covers positions ``[seqlen_k - seqlen_q,
seqlen_k)`` with ``seqlen_k = num_cached_tokens + seqlen_q``, as DeepInfer's
scheduler sets them (without its in-flight tokens, S4.3). Prefill takes every
token without KV unless ``seqlens_q`` limits the chunk; decode takes the one
newest token. (In DeepInfer the scheduler computes these numbers and the
runner only turns them into tensors; the scheduler arrives in S2.5.)

Temporary: the sampler is greedy ``argmax``. It raises NotImplementedError for
a temperature above 0 and for ``first_n_token_no_eos``. DeepInfer has no
argmax; its greedy is temperature 1e-5 in the full sampler, which arrives in
phase 3. As in DeepInfer's sampler, ``SamplingParams.golden_token_ids``
replaces the sampled token while the output is shorter than that list
(teacher forcing). Also left out: CUDA graphs (phase 4), overlapped
scheduling (ZOS), speculative decoding, and DeepInfer's padding of
``max_num_blocks`` to a multiple of 64.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as SequenceList
from dataclasses import dataclass
from pathlib import Path

import torch

from mini_deepinfer.model_executor.input_metadata import InputMetadata
from mini_deepinfer.model_executor.layers.sampler_utils import get_logits
from mini_deepinfer.model_executor.model_loader import load_model
from mini_deepinfer.sampling_params import SAMPLING_EPS
from mini_deepinfer.sequence import Sequence


@dataclass
class SamplerOutput:
    logits: torch.Tensor  # [num_reqs, vocab] float32, on the model's device
    token_ids: list[int]  # one sampled token per sequence


class ModelRunner:
    def __init__(
        self,
        model_path: str | Path,
        *,
        block_size: int,
        max_model_len: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_path = model_path
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.max_num_blocks = math.ceil(max_model_len / block_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = None
        self.load_report = None

    def load_model(self) -> None:
        self.model, self.load_report = load_model(self.model_path, device=self.device, dtype=self.dtype)
        if self.max_model_len > self.model.cfg.max_position_embeddings:
            raise ValueError(
                f"max_model_len {self.max_model_len} > max_position_embeddings {self.model.cfg.max_position_embeddings}"
            )

    def _to_device(self, data: list, dtype=torch.int32) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype)
        if self.device.type == "cuda":
            t = t.pin_memory()
        return t.to(self.device, non_blocking=True)

    def prepare_input_tensors(
        self,
        seqs: SequenceList[Sequence],
        is_prompt: bool,
        seqlens_q: SequenceList[int] | None = None,
    ) -> InputMetadata:
        if not seqs:
            raise ValueError("no sequences")
        if seqlens_q is not None and (not is_prompt or len(seqlens_q) != len(seqs)):
            raise ValueError("seqlens_q chooses prefill chunk sizes: one per sequence, prefill only")
        input_ids, positions, slot_mapping, block_tables = [], [], [], []
        sq, sk = [], []
        bs = self.block_size
        for i, seq in enumerate(seqs):
            cached, length = seq.num_cached_tokens, seq.get_len()
            if is_prompt:
                q = seqlens_q[i] if seqlens_q is not None else length - cached
            else:
                if cached != length - 1:
                    raise ValueError(f"seq {seq.seq_id}: decode needs every token but the newest cached ({cached} of {length})")
                q = 1
            k = cached + q
            if q <= 0 or k > length:
                raise ValueError(f"seq {seq.seq_id}: cannot process {q} tokens with {cached} of {length} cached")
            table = seq.cache.block_table
            if len(table) * bs < k:
                raise ValueError(f"seq {seq.seq_id}: {len(table)} blocks hold {len(table) * bs} tokens, the step needs {k}")
            if len(table) > self.max_num_blocks:
                raise ValueError(f"seq {seq.seq_id}: {len(table)} blocks, more than max_num_blocks {self.max_num_blocks}")
            tokens = seq.get_token_ids()
            for p in range(k - q, k):
                input_ids.append(tokens[p])
                positions.append(p)
                slot_mapping.append(table[p // bs] * bs + p % bs)
            block_tables.append(table + [0] * (self.max_num_blocks - len(table)))
            sq.append(q)
            sk.append(k)
        return InputMetadata(
            is_prompt=is_prompt,
            input_ids=self._to_device(input_ids),
            positions=self._to_device(positions),
            seqlens_q=sq,
            seqlens_k=sk,
            slot_mapping=self._to_device(slot_mapping),
            block_tables=self._to_device(block_tables),
        )

    @torch.inference_mode()
    def execute_main_model(self, input_metadata: InputMetadata, kv_caches) -> torch.Tensor:
        return self.model(kv_caches=kv_caches, input_metadata=input_metadata)

    @torch.inference_mode()
    def execute_sample(
        self, seqs: SequenceList[Sequence], input_metadata: InputMetadata, hidden_states: torch.Tensor
    ) -> SamplerOutput:
        for seq in seqs:
            sp = seq.sampling_params
            if sp.temperature != SAMPLING_EPS or sp.first_n_token_no_eos == 1:
                raise NotImplementedError("only plain greedy sampling until the sampler (phase 3)")
        selected = self._to_device(input_metadata.last_token_indices())
        rows = hidden_states.index_select(0, selected)
        logits = get_logits(rows, self.model.lm_head.weight, self.model.cfg.vocab_size)
        token_ids = logits.argmax(dim=-1).tolist()
        for i, seq in enumerate(seqs):
            golden = seq.sampling_params.golden_token_ids
            n = seq.get_output_len()
            if golden is not None and n < len(golden):
                token_ids[i] = int(golden[n])
        return SamplerOutput(logits=logits, token_ids=token_ids)
