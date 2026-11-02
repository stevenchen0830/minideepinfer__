# mini-DeepInfer

A small, educational rebuild of [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer) inference engine on one consumer GPU. The package `mini_deepinfer` copies DeepInfer's file layout and conventions, but runs Qwen3-0.6B / Qwen3-1.7B with plain PyTorch, and every piece is checked against Hugging Face transformers.
一个用于学习的 [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) 推理引擎小型复刻，跑在一张消费级显卡上。`mini_deepinfer` 照搬 DeepInfer 的目录结构和约定，但用纯 PyTorch 跑 Qwen3-0.6B / Qwen3-1.7B，每一部分都与 Hugging Face transformers 对比验证。

## What is implemented / 已完成的部分

| Part / 部分 | Files / 文件 | Checked by / 检查方式 |
|---|---|---|
| Environment: uv project, Python 3.12, PyTorch, Triton / 环境 | `pyproject.toml`, `uv.lock` | PyTorch sees the GPU; a Triton kernel compiles / 能识别显卡，Triton 算子能编译 |
| Golden references from HF `generate` (greedy tokens, logits) / 金标准 | `scripts/make_golden.py`, `scripts/check_golden.py` | The check script reproduces them / 检查脚本能复现 |
| `SamplingParams`, `Sequence` (status, token data, KV-cache counters) / 请求参数与序列状态 | `src/mini_deepinfer/{sampling_params,sequence}.py` | Unit tests and a randomized test against plain lists / 单元测试与随机对照测试 |
| Qwen3 model: RMSNorm, RoPE, GQA attention with q/k norm, SwiGLU / Qwen3 模型 | `src/mini_deepinfer/model_executor/` | Logits vs HF in float32 on tiny random checkpoints / 用随机小检查点在 float32 下与 HF 对比 |
| Hugging Face weight loading: config reader, name map, safetensors reader, load plan / 权重加载 | `src/mini_deepinfer/hugging_face/` | Failure cases: missing, wrong shape or dtype, truncated files / 各种加载失败情形 |
| Real-weight check / 真实权重检查 | `scripts/check_model.py` | BF16 vs the golden files and live HF; float32 vs HF / BF16 对比金标准与现场 HF；float32 对比 HF |
| KV cache (block-based, contiguous as one block per sequence), prefill and decode, model runner, a static-batch engine / KV cache、prefill 与 decode、模型运行器、静态批引擎 | `src/mini_deepinfer/worker/`, `src/mini_deepinfer/engine/` | Decode with the cache equals a full recompute (float32); greedy tokens equal HF `generate` / 带缓存的 decode 与整段重算一致；greedy 结果与 HF `generate` 一致 |
| Block manager: free list with reference counts, OK / LATER / NEVER, reserved block 0 / 块管理器：带引用计数的空闲链表、OK / LATER / NEVER、保留块 0 | `src/mini_deepinfer/engine/block_manager.py` | Randomized test against a reference model, with invariants checked after every operation / 与参考模型对比的随机测试，每步检查不变量 |
| Generation check / 生成检查 | `scripts/check_generate.py` | Free-running and teacher-forced greedy vs the golden files and HF's own decode path / 自由生成与 teacher forcing 对比金标准和 HF 自己的 decode |

Results on an RTX 4060 Laptop (8 GB): both Qwen3-0.6B and Qwen3-1.7B pass. In float32 the logits agree with HF to within about 6e-5 over the whole vocabulary. In BF16 the difference from the golden logits is the same size as HF's own difference from them when HF is run now as one full-sequence forward (`scripts/check_model.py`), and greedy tokens differ only at near-ties. Generating with the KV cache reproduces the golden 32-token outputs exactly for 4 of 8 prompts (0.6B) and 6 of 8 (1.7B); every other prompt first differs where the top two golden logits are equal.
在 RTX 4060 Laptop（8 GB）上的结果：Qwen3-0.6B 和 Qwen3-1.7B 都通过。float32 下 logits 与 HF 在整个词表上相差约 6e-5 以内。BF16 下与金标准的差距，和现在用一次整段前向运行的 HF 与金标准的差距同一量级（`scripts/check_model.py`），greedy token 只在前两名几乎相等处不同。用 KV cache 生成时，0.6B 有 4/8、1.7B 有 6/8 个 prompt 的 32 个 token 与金标准完全一致，其余 prompt 第一次不同都出现在金标准前两名 logits 相等的位置。

Docstrings refer to steps by number (S1.2 = phase 1, step 2). Each module docstring names the DeepInfer file it mirrors and lists every deliberate difference. Where DeepInfer calls a GPU kernel (TileLang, DeepGEMM, FlashMLA), this code uses a short PyTorch stand-in.
代码注释里用编号指代步骤（S1.2 = 第 1 阶段第 2 步）。每个模块开头写明对应 DeepInfer 的哪个文件，并列出每一处有意的不同。DeepInfer 调用 GPU 算子（TileLang、DeepGEMM、FlashMLA）的地方，这里用几行 PyTorch 代替。

## Run / 运行

Tested on WSL2 Ubuntu with [uv](https://docs.astral.sh/uv/), Python 3.12, torch 2.15 (CUDA 13.2 wheels) and transformers 5.22.
在 WSL2 Ubuntu 上测试，使用 uv、Python 3.12、torch 2.15（CUDA 13.2）和 transformers 5.22。

```bash
uv sync
```

```bash
uv run pytest -q
```

The tests run on the CPU in a few seconds and need no download. The real-weight check downloads the model once and needs the golden files first:
测试在 CPU 上几秒跑完，不需要下载。真实权重检查会下载一次模型，并且需要先生成金标准：

```bash
uv run scripts/make_golden.py --model Qwen/Qwen3-0.6B
```

```bash
uv run scripts/check_model.py --model Qwen/Qwen3-0.6B
```

```bash
uv run scripts/check_model.py --model Qwen/Qwen3-0.6B --fp32
```

```bash
uv run scripts/check_generate.py --model Qwen/Qwen3-0.6B
```

## Next / 后续

Block sharing (fork, prefix cache), a scheduler with continuous batching, the sampler, CUDA graphs, and DeepSeek-style model pieces (MLA, MoE).
接下来：块共享（fork、前缀缓存）、带连续批处理的调度器、采样器、CUDA graph，以及 DeepSeek 风格的模型组件（MLA、MoE）。
