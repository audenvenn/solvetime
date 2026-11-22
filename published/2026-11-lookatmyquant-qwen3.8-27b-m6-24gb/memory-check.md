# What LookAtMyQuant's 7.2 GB Qwen3.8-27B build holds in memory

**How it was measured.** On 2026-11-22 at 15:26, after round 4, on the same Mac, under the GPU lock.
- **The script.** [`memory_check.py`](memory_check.py) loads the build as the round's launcher did:
  `runtime_mlx.load` with its defaults.
- **The environment.** The round's venv: mlx 0.32.4, mlx-lm 0.32.2, LookAtMyQuant 0.3.0 at 51069235.
- **The readings.** MLX's own counters.
  - **Active memory** is the live arrays.
  - **The cache** is the allocator's freed buffers.
  - **The peak** is the peak of active memory.
- **The prefill.** 10,634 tokens of synthetic text, the round's longest prompt length, in the 2,048-token chunks of
  mlx-lm's server. It ran twice, each time with a fresh KV cache:
  - **first, as mlx-lm 0.32.2's server does,** clearing MLX's cache after each chunk;
  - **then without clearing,** as the first version of this check did at 14:34. That run gave the same readings.

| Point | Active | Cache | Peak active |
|---|---|---|---|
| After loading (6.0 s) | 7.57 GB | 0.06 GB | 7.57 GB |
| After the prefill as the server does it (62.4 s), its KV cache held | 9.14 GB | 0.28 GB | 15.93 GB |
| After the prefill without clearing (62.5 s), its KV cache held | 9.14 GB | 8.97 GB | 15.93 GB |
| After dropping the KV cache and clearing | 7.57 GB | 0.00 GB | 15.93 GB |

**What it shows:**
- **The model.** The loaded model takes 7.57 GB, about its file's size.
- **The prefill.** A prefill of this length needs up to 15.93 GB of live arrays, 8.4 GB more than the model, with or
  without the cache cleared. The KV cache of the 10,634 tokens is 1.57 GB of that.
- **The cache.** The server's clearing leaves 0.28 GB of freed buffers. Without it, 8.97 GB stay.

**What it doesn't show:**
- **The server's memory.** It doesn't break down the server's own memory. During the agent run the engine's process
  footprint peaked at 18.6 GB.
  - **The "17G" readings.** macOS `top` showed the engine at 17G in 62 of the watchdog's 66 snapshots during the
    run. The watchdog took those only at memory-pressure warnings, which favours the moments of a prefill.
  - **The prompts.** The run's prefills were up to 10,315 new tokens, after the 319 the server took from its cache.

**What the loaded model holds.** This split counts the arrays reachable from these objects, to a depth of 4, each
array once:
- **0.21 GB** of parameters in the model's module tree, all float32: the norms and other uncoded tensors. There is
  no vision tower among them.
- **6.91 GB** in LookAtMyQuant's own structures: the coded matrices with their tables.
- **The other 0.45 GB** was not attributed.
