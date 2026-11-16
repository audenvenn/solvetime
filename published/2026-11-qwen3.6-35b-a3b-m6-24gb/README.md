# Qwen3.6-35B-A3B on a 24 GB Mac mini (M6), November 2026

Machine: Apple M6, 24 GB, macOS 27.2.
Runs from 2026-11-15 20:59:43 to 2026-11-16 10:28:16. Agent: mini-swe-agent 2.4.6; limits: 50 steps and 900 s per attempt. Sampling sent with every request: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0, thinking on. Tables made with solvetime 0.1.0.dev0 (f98ef16).

## About these runs

- **Machine.** Mac mini (2026), Apple M6 with a 12-core GPU, 24 GB, macOS 27.2, on AC power. The usual
  background was open throughout (a few terminal sessions, two browsers, an IDE: about 5 to 6 GB), and
  nothing else heavy ran during a measurement, except where a row says otherwise.
- **Builds.**
  - raceline: main at 3de235d, recorded as `raceline 0.7.1 (v0.7.1-6-g3de235d)` and released unchanged as
    0.8.0 (0.8.0 adds only the version number and the changelog). The model file is
    `qwen3.6-35b-a3b.rlm` (FP4 with its MTP head), the file `raceline pull qwen3.6-35b-a3b` downloads
    (sha256 `025728fd540c7aea1236cd63cbd5c2f3ee0b9e3ba9295b5936477018a3e8b8ea`). raceline is the
    author's own engine.
  - kapitsa 0.2.36 (fe6a17a), the Homebrew release. `kapitsa serve --model 35b` resolves to `35b:16gb` on
    a 24 GB Mac: unsloth's UD-IQ2_XXS experts. Its 4-bit pack does not fit, by its own `kapitsa doctor`.
  - kapitsa main at 7445a6c5, built from source. It carries the fix for the dropped tool-call tag
    ([nace-ai/kapitsa#1277](https://github.com/nace-ai/kapitsa/issues/1277), fixed in PR #1279 the day
    after our report); same pack. Its version string still says 0.2.36, with the commit.
  - llama.cpp 0.9.0 (build 12110, Homebrew) on `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` from
    unsloth/Qwen3.6-35B-A3B-GGUF at revision a483e9e6 (sha256
    `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef`), plain decoding, a 65536-token
    window.
- **Tasks.** The core set (8 tasks) and the extended set (8 harder ones), two attempts each, round-robin.
- **Harness.** solvetime from 2026-11-15 (PR #1) to 2026-11-16 (#6) ran the agent tasks, and #10/#11 ran
  the speed tests. The agent (mini-swe-agent 2.4.6), its prompts and limits, the sampling and the checks
  did not change across those versions. Runs before 22:00 on 2026-11-15 did not yet keep the raw replies.
- **raceline's extended run** used `--expert-cache 8`: 8 GB of experts in memory instead of the 11.5 GB it
  picks itself. During the night, the dense 27B had pushed memory pressure to critical, and the smaller
  cache leaves the Mac more room. The cache changes where experts are read from, not what the model
  computes. Tasks solved compare with the other runs; times do not, and the Time table says so.
- **Left out.**
  - One kapitsa 0.2.36 extended run, interrupted at 09:59 on 2026-11-16. Memory pressure turned critical
    because of a process the author started by mistake. The row here is a full rerun.
  - Speed tests whose "continue" prompt let the model stop after 9 to 26 tokens (fixed in #10).
  - Speed tests that overlapped a large file read by the author.
- **Not run.** MTPLX: its own catalog rates every 35B pack insufficient_memory on 24 GB. mlx-lm: 20.4 GB
  of 4-bit weights, more than Metal lets the GPU use here.
- **The dropped tool-call tag.** At about 2 bits, Qwen3.6-35B-A3B often writes a tool call without its
  opening `<tool_call>` token. A server that parses strictly then returns the call as plain text, and the
  agent gets no command. The minimal repro, with raw replies, is in `evidence/`.

## Agent tasks, each engine build pooled over its task sets

Solved with a 95% Wilson interval. "Ended by malformed output": the attempt stopped after three replies in a row that the agent could not use (no tool call). "Replies without a tool call" counts every such reply, recovered or not, out of all replies.

| Model | Engine | Build | Settings | Bits per weight | Tasks | Solved | Ended by malformed output | Replies without a tool call | Replies cut off at the token limit | Median time, solved attempts | Other failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | qwen3.6-35b-a3b (FP4, 4.0 bits, with its MTP head) | defaults; --expert-cache 8 | 4.06 | 16 (core, extended) | 30/32 (94%, 95% CI 80-98%) | 0/32 | 5 of 373 | 1 | 62 s | wrong 2 |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 16 (core, extended) | 19/32 (59%, 95% CI 42-74%) | 13/32 | 42 of 270 | 0 | 42 s | none |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf | defaults | 2.42 | 16 (core, extended) | 23/32 (72%, 95% CI 55-84%) | 5/32 | 19 of 358 | 0 | 61 s | wrong 4 |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 16 (core, extended) | 28/32 (88%, 95% CI 72-95%) | 0/32 | 0 of 415 | 0 | 48 s | wrong 4 |
| qwen3.6-35b-a3b | mtplx 2.12.3 | Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed (the smallest of its 35B packs) | |  | | not run: does not fit (MTPLX's own verdict) | | | | | MTPLX's catalog rates every Qwen3.6-35B-A3B pack insufficient_memory on this Mac (24 GiB): the smallest, Optimized Speed, is 21.0 GB with a measured peak of 28.0 GiB. MTPLX has no way to keep experts on disk. |
| qwen3.6-35b-a3b | mlx-lm 0.32.2 | mlx-community/Qwen3.6-35B-A3B-4bit | |  | | not run: does not fit (computed, not run) | | | | | Its weights take 20.40 GB, and Metal lets the GPU use 19.07 GB on this Mac (recommendedMaxWorkingSetSize). mlx-lm keeps all of a model's weights in GPU memory. |


## Pairwise differences

Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.

| Model | A | B | Solved, A vs B | p | Ended by malformed output, A vs B | p |
|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | kapitsa 0.2.36 (fe6a17a) | 30/32 vs 19/32 | 0.002 | 0/32 vs 13/32 | < 0.001 |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | 30/32 vs 23/32 | 0.043 | 0/32 vs 5/32 | 0.053 |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | kapitsa 0.2.36 (7445a6c) | 30/32 vs 28/32 | 0.672 | 0/32 vs 0/32 | 1.000 |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | 19/32 vs 23/32 | 0.430 | 13/32 vs 5/32 | 0.050 |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | kapitsa 0.2.36 (7445a6c) | 19/32 vs 28/32 | 0.022 | 13/32 vs 0/32 | < 0.001 |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | kapitsa 0.2.36 (7445a6c) | 23/32 vs 28/32 | 0.213 | 5/32 vs 0/32 | 0.053 |


## Time

Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over tasks solved. Time in the engine: the requests themselves. Decode and prefill during the run: from each engine's own log, medians over the requests that wrote at least 64 tokens and that read at least 256 new tokens. A run with settings beyond the defaults has times that compare with nothing else.

| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | Tokens written | Time in the engine | Decode during the run (median) | Prefill during the run (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | core | defaults | 16 | 14 | 19.6 min | 84 s | 187 | 47682 | 16.3 min | 61 tok/s (159 requests) | 420 tok/s (60 requests) |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | core | defaults | 16 | 7 | 6.6 min | 56 s | 118 | 18109 | 5.2 min | 67 tok/s (99 requests) | 975 tok/s (36 requests) |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | core | defaults | 16 | 10 | 12.7 min | 76 s | 183 | 29867 | 9.9 min | 60 tok/s (142 requests) | 1005 tok/s (40 requests) |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | core | defaults | 16 | 14 | 13.5 min | 58 s | 211 | 37586 | 10.3 min | 66 tok/s (166 requests) | 925 tok/s (44 requests) |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | extended | defaults | 16 | 14 | 13.5 min | 58 s | 204 | 38488 | 10.8 min | 66 tok/s (163 requests) | 946 tok/s (35 requests) |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | extended | --expert-cache 8 (times not comparable) | 16 | 16 | 25.7 min | 96 s | 186 | 41407 | 24.1 min | 35 tok/s (139 requests) | 306 tok/s (43 requests) |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | extended | defaults | 16 | 12 | 7.8 min | 39 s | 152 | 23342 | 6.5 min | 67 tok/s (103 requests) | 961 tok/s (20 requests) |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | extended | defaults | 16 | 13 | 14.3 min | 66 s | 175 | 33366 | 11.0 min | 61 tok/s (152 requests) | 1017 tok/s (36 requests) |


## Each run

Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it understates an engine that maps its weights from the file. To compare memory across engines, use the speed section's memory taken from the system.

| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, solved attempts | Total time per solved task | Total time | Tokens written (median) | Requests (median) | Failures | Paged out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | qwen3.6-35b-a3b (FP4, 4.0 bits, with its MTP head) | defaults | 4.06 | 18.0 GB | 17.5 GB | 4 s | 14/16 | 44 s | 84 s | 19.6 min | 1906 | 10 | wrong 2 | 0.2 GB |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 12.7 GB | 13.9 GB | 10 s | 7/16 | 58 s | 56 s | 6.6 min | 364 | 3 | malformed output 7, loop 2 | 0.0 GB |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf | defaults | 2.42 | 10.8 GB | 12.7 GB | 8 s | 10/16 | 56 s | 76 s | 12.7 min | 2268 | 11 | malformed output 4, wrong 2 | 0.0 GB |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 12.7 GB | 13.9 GB | 12 s | 14/16 | 49 s | 58 s | 13.5 min | 2314 | 11 | wrong 2 | none |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 12.7 GB | 14.3 GB | 4 s | 14/16 | 47 s | 58 s | 13.5 min | 2130 | 13 | wrong 2 | 0.0 GB |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | qwen3.6-35b-a3b (FP4, 4.0 bits, with its MTP head) | --expert-cache 8 | 4.06 | 18.0 GB | 13.6 GB | 4 s | 16/16 | 106 s | 96 s | 25.7 min | 2760 | 12 | none | 0.1 GB |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | kapitsa 35b: qwen3.6-35b-a3b-16gb | defaults | 2.48 | 12.7 GB | 13.8 GB | 4 s | 12/16 | 32 s | 39 s | 7.8 min | 1176 | 6 | malformed output 4 | 0.0 GB |
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf | defaults | 2.42 | 10.8 GB | 12.2 GB | 3 s | 13/16 | 67 s | 66 s | 14.3 min | 2366 | 12 | wrong 2, malformed output 1 | 0.0 GB |


## Speed

`solvetime speed`: thinking off, after one uncounted warm-up; prefill on a 4K-token prompt, decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what the system had available before the engine started, minus the least it had while the engine ran (free, inactive, speculative and purgeable pages), on an otherwise quiet machine.

| Model | Engine | Bits per weight | Prefill 4K | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 | Memory taken (system) | RSS | Footprint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | llama.cpp 0.9.0 (build 12110, commit cd959bae9) | 2.42 | 1121 (1116-1122) | 61.4 (61.4-61.8) | 60.3 (60.0-61.1) | 60.6 (60.3-60.7) | 59.8 (59.7-60.1) | 10.6 GB | 15.1 GB | 4.5 GB |  |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (7445a6c) | 2.48 | 1091 (1091-1091) | 67.9 (67.9-67.9) | 68.1 (67.6-68.1) | 67.8 (67.7-67.9) | 68.0 (68.0-68.0) | 8.2 GB | 11.4 GB | 12.5 GB |  |
| qwen3.6-35b-a3b | kapitsa 0.2.36 (fe6a17a) | 2.48 | 1053 (1053-1054) | 64.0 (64.0-64.2) | 68.1 (65.8-68.2) | 67.9 (67.9-67.9) | 68.1 (68.0-68.1) | 10.2 GB | 11.4 GB | 12.5 GB |  |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | 4.06 | 1268 (1152-1285) | 80.0 (78.9-82.8) | 84.8 (78.6-86.2) | 71.8 (68.9-75.1) | 80.7 (80.6-82.5) | 12.7 GB | 12.0 GB | 16.6 GB | experts evicted from the file cache just before (20 GB of other files read) |
| qwen3.6-35b-a3b | raceline (ours) 0.7.1 (v0.7.1-6-g3de235d) | 4.06 | 1333 (1285-1434) | 78.5 (77.4-79.2) | 84.5 (81.3-92.0) | 65.5 (61.8-70.7) | 81.6 (79.8-83.0) | 14.3 GB | 12.1 GB | 16.6 GB | experts cached (right after the run above) |


## Not run

- qwen3.6-35b-a3b, mtplx 2.12.3 (Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed (the smallest of its 35B packs)): does not fit (MTPLX's own verdict). MTPLX's catalog rates every Qwen3.6-35B-A3B pack insufficient_memory on this Mac (24 GiB): the smallest, Optimized Speed, is 21.0 GB with a measured peak of 28.0 GiB. MTPLX has no way to keep experts on disk. How: ~/venvs/mtplx/bin/python -c "from mtplx import model_catalog as mc; m = [x for x in mc.OFFICIAL_CATALOG if x.hf_model_id.endswith('35B-A3B-MTPLX-Optimized-Speed')][0]; print(mc.evaluate_feasibility(m, chip_tier=mc.MODERN_TIER, ram_gib=24.0))"
- qwen3.6-35b-a3b, mlx-lm 0.32.2 (mlx-community/Qwen3.6-35B-A3B-4bit): does not fit (computed, not run). Its weights take 20.40 GB, and Metal lets the GPU use 19.07 GB on this Mac (recommendedMaxWorkingSetSize). mlx-lm keeps all of a model's weights in GPU memory. How: huggingface_hub.HfApi().model_info('mlx-community/Qwen3.6-35B-A3B-4bit', files_metadata=True): the sum of the .safetensors sizes; mlx.core.device_info()['max_recommended_working_set_size']
