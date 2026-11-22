# Inkling 2 Mini on a 24 GB Mac mini (M6), November 2026

Machine: Apple M6, 24 GB, macOS 27.2.
Runs from 2026-11-21 12:44:18 to 2026-11-21 15:09:44. Agent: mini-swe-agent 2.4.6; limits: 50 steps, and a 900 s budget the agent checks before each query (a query in flight and its retries can run past it). Sampling sent with every request: temperature 1, top-p 1, top-k 0, min-p 0. Tables made with solvetime 0.1.0.dev0 (d632180-dirty).

## About these runs

- **What the tables compare.** Each scored configuration ran the 8 core tasks twice, with the model's default
  thinking. The two scored rows differ in engine, model build (and its bits per weight), settings, and the memory
  free at each start. So the tables compare these setups as they ran; they don't say how much any single factor
  contributed.
- **The comparison.** Both configurations ran the same eight tasks twice. llama.cpp solved all eight both times.
  oMLX solved six both times, pathlib once and slugify neither time. The pooled Fisher test (p = 0.226) and the
  confidence intervals treat the 16 attempts as independent, and ignore that the same tasks were repeated.
- **The model.** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) at
  ed40e7d, Apache-2.0: 26.4B parameters, 26.07B of them in the text model.
  - **A looped mixture of experts.** Each layer has 128 experts, 8 active. The middle 12 of the 24 layers run three
    times per token, so a token goes through 48 layer applications made from 24 layers of weights. That is 3.4B
    active parameters, or 5.4B counting the passes.
  - **Attention.** 20 of the 24 layers attend over a 1,024-token window.
  - **Size.** The release is about 20 GB, in NVFP4 and MXFP8. Its card asks for Apple silicon with 32 GB or more,
    so on 24 GB only a smaller build fits.
- **Thinking and sampling.**
  - **Thinking depth.** The model's template sets it with `reasoning_effort` and ignores `enable_thinking`. The
    agent runs sent no effort, so the template's default, "high" (0.9), applied.
  - **Sampling in the agent runs.** The card's temperature 1.0, with top_p 1.0, top_k 0 and min_p 0 sent so that
    the engines' own defaults do not differ.
  - **The speed tests** asked for the lowest effort: "none", which the template turns into 0. Both kept runs got
    it, as a template variable (solvetime #56).
    - **The first llama.cpp speed test** also sent the field at the top level. llama.cpp gives that precedence and
      renders the default, so that test was rerun, and the first is left out.
  - **The model kept reasoning even so.** Each of the 24 decode replies reached the 512-token cap, with 510 counted
    tokens, all of them reasoning and none of them answer text. They were timed, and they are counted
    (solvetime #52).
  - **The thinking-switch probe** reports "ignored: always thinks" on both engines. The model writes a short
    thinking block even at "none": on llama.cpp, 144 characters before the answer to a one-line question. The
    agent runs asked for thinking, so that is what they got.
- **Builds.**
  - **llama.cpp b12600** (29b52d398), from GitHub: the first release with Inkling 2 Mini (#32694, 2026-11-20).
    Homebrew's 1.0.0 predates it.
    - **The model build.** InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF, `Inkling-2-Mini-IQ3_XXS.gguf`: 11.75 GB,
      3.60 bits per weight for the text model, revision 279557c, SHA-256
      `32fcaba1b08f3d23c646d4c628441125adfe5953a99c70572214d98ae3fef8f7`. That is fewer bits per weight than
      oMLX's build.
    - **Why not unsloth's,** the usual source of GGUFs. At b12600 llama.cpp's Metal backend does not support
      multiplying NVFP4 weights, and the two unsloth `Mainline_Support` GGUFs we checked keep the checkpoint's NVFP4
      experts (below).
    - **The other candidates.** ggml-org's NVFP4-free 4-bit build is 17.6 GB and was not tried, as too big for the
      headroom here. AtomicChat's NVFP4-free files declare the `inkling2` architecture rather than the looped
      `inkling2_loop`, so they were not used.
  - **oMLX 0.8.1** (Homebrew; released on 2026-11-21) on mlx-community/Inkling-2-Mini-oQ3e-mtp, revision 5552e32:
    oMLX's own mixed quantization, at 3.81 bits per weight for the text model. The 13.70 GB package also holds the
    vision encoder and a 0.64 GB DSpark drafter.
    - **Speculative decoding.** oMLX turned the DSpark drafter on for this build by default. By oMLX's own notes,
      with DSpark on a cache hit can change the output after some tokens; with it off the output matches bit for
      bit. It was not run with DSpark off.
- **Settings.**
  - **llama.cpp.** A 32,768-token window: solvetime's default is 64K, and 32K leaves memory headroom; the largest
    context in the run reached 17,149 tokens, and none was cut.
    - **The RAM prompt cache, off** (`--cache-ram 0`). We turned it off after a 27B run reached critical memory
      pressure with the default 8 GiB cache on (2026-11-20).
    - **Slots.** Its default of 4 slots on one KV cache.
  - **oMLX.** Memory guard "aggressive", which is not its default, by the maintainer's decision. At its default
    ("balanced") neither launch reached a scored attempt. oMLX's log writes GiB as "GB":
    - **The check at 12:07** refused to load the model. Its ceiling was 11.10 GiB for a model it put at 14.06 GiB,
      and system-wide anonymous memory was 8.5 GB.
    - **The round's launch at 13:45** loaded the model, then rejected its 18-token warm-up: "Prefill would require
      ~14.86 GB peak ... but the prefill limit is 14.58 GB". That refusal is the row with 0 attempts scored.
    - **The aggressive run** started at 13:47, with 2.63 GB more available than at 13:45.
- **Machine and memory.** Mac mini (2026), Apple M6, 24 GB, macOS 27.2.
  - **Other work on the Mac.** Other sessions ran on it, among them a CPU-bound search from another project. They
    were asked to keep memory low while the GPU was in use; CPU contention was not measured.
  - **Readings before each launch,** taken after the GPU calibration's memory was released (solvetime #53):

  | Run | Estimated available before launch | System-wide anonymous memory |
  |---|---|---|
  | llama.cpp (12:44) | 19.0 GB | 2.8 GB |
  | oMLX, default guard; warm-up refused, no attempt scored (13:45) | 16.6 GB | 3.3 GB |
  | oMLX, guard "aggressive" (13:47) | 19.2 GB | 3.7 GB |

- **oMLX's aggressive run.**
  - **Request aborts.** Its log records 80 request aborts under memory pressure, each one request aborted while
    the model stayed loaded, and no model evictions.
  - **The failures.** Its three failed attempts (slugify twice, pathlib once) each ended when a request failed
    with an API error and an empty message.
  - **Tokens.** The tables count them from its 118 replies to the agent: 43,628 tokens in all.
    - **Why from the replies.** solvetime's proxy lost oMLX's replies, which oMLX sends chunked; fixed after the
      round.
    - **A floor.** Tokens of aborted requests and of replies the agent did not keep are not in it.
- **llama.cpp and NVFP4.**
  - **The cause.** In `ggml-metal-device.m` at b12600, the Metal backend's matrix-multiplication support check
    returns false for `GGML_TYPE_NVFP4`, so those multiplications fall back to the CPU.
  - **unsloth's GGUFs.** Its `Mainline_Support` files keep the checkpoint's NVFP4 experts. In UD-Q3_K_XL that is
    the down projections of 20 layers and the gate-up projections of 12; UD-IQ3_XXS has the 20 down projections.
  - **The measurement.** llama-bench b12600 ([llama-bench.md](llama-bench.md)), with offload of all layers
    requested, flash attention on and two repetitions:

  | GGUF | Prompt (pp512) | Generation (tg128) |
  |---|---|---|
  | unsloth UD-Q3_K_XL (13.63 GB, NVFP4 experts) | 28.6 tok/s | 15.5 tok/s |
  | InterpreticAI IQ3_XXS (11.75 GB, no NVFP4) | 319.2 tok/s | 33.6 tok/s |

  - **Reading the gap.** The two files differ in more than their NVFP4 tensors, and no per-operation timing was
    taken, so not all of the gap can be put down to the CPU.
  - **The agent check on unsloth's file.** Its 30-minute limit ended before the agent got a reply. The server's last
    cumulative rates were 7.43 and 5.35 tok/s, for two overlapping requests.
  - **Threads (added on 2026-11-22).** Every llama.cpp run here used 2 CPU threads. No thread count was set, and
    on Apple chips llama.cpp takes its default from the first tier of cores (`hw.perflevel0`), which on the M6
    is its 2 Super cores, of 12 ([llama.cpp #29782](https://github.com/ggml-org/llama.cpp/issues/29782)). So
    unsloth's NVFP4 multiplications ran on 2 threads, in llama-bench and in the agent check; more threads could
    narrow the gap, but that was not measured. Whether the thread count affected the scored run, which computes
    on the GPU, was not measured either. The records stay as measured.
- **Harness.** solvetime at ee9e319 for the agent runs and oMLX's speed test, and 0c8e140 for llama.cpp's speed
  test:
  - #51: the model entry and its thinking switch;
  - #52: reasoning counted in the speed test;
  - #53: memory read after the calibration's release;
  - #56: the switch sent as a template variable only, and validate checks that a run got the thinking it asked for.

  oMLX's token counts and bits per weight were corrected in its records after the round, from its replies and its
  files, and the records keep the values first recorded beside them.

## Agent tasks, each engine build pooled over its task sets

Solved: the attempt's final workspace passed the task's check, work done before an error included, with a 95% Wilson interval. "Failed checks after a repeated-format exit": attempts the agent stopped after three replies in a row it could not use, whose check then failed. "Requests": every request the agent sent, the ones that failed included. "Agent tool-call error notices" and "Agent output-limit notices": the agent's own feedback messages, for a reply without a usable tool call and for a reply cut off at the output limit. "Estimated available before launch": the memory macOS could hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages), measured after the harness's GPU calibration; runs made before 2026-11-18 did not record it. A run whose engine never served an attempt is listed with how it stopped, and no score.

| Model | Engine | Build | Settings | Bits per weight | Estimated available before launch | Tasks | Solved | Failed checks after a repeated-format exit | Requests | Agent tool-call error notices | Agent output-limit notices | Median time, solved attempts | Other failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inkling-2-mini | llama.cpp b12600 (29b52d398) | InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF IQ3_XXS (279557c) | context 32768, prompt cache off | 3.6 | 19.0 GB | 8 (core) | 16/16 (100%, 95% CI 81-100%) | 0/16 | 162 | 0 | 0 | 156 s | none |
| inkling-2-mini | oMLX 0.8.1 | mlx-community/Inkling-2-Mini-oQ3e-mtp (5552e32) | memory guard aggressive | 3.81 | 19.2 GB | 8 (core) | 13/16 (81%, 95% CI 57-93%) | 0/16 | 198 | 0 | 0 | 76 s | error 3 |
| inkling-2-mini | oMLX 0.8.1 | mlx-community/Inkling-2-Mini-oQ3e-mtp (5552e32) | defaults |  | 16.6 GB | | 0 attempts scored — server started, warm-up refused: its first request failed: HTTP 400: oMLX prefill memory guard rejected this prompt: Prefill would require ~14.86 GB peak (current 14.86 GB + KV+SDPA 7.78 MB) but the prefill limit is 14.58 GB (dynamic ceiling 15.84 GB - safety margin 1.27 GB). | | | | | | |
| inkling-2-mini | llama.cpp b12600 | unsloth/Inkling-2-Mini-GGUF Mainline_Support UD-Q3_K_XL (ea0c79b, 13.63 GB) | |  | | | not run: no scored run (benchmarked and checked) | | | | | | At b12600 llama.cpp's Metal backend does not support multiplying NVFP4 weights, and this file keeps the checkpoint's NVFP4 experts, so those multiplications fall back to the CPU. With offload of all layers requested and llama.cpp's default of 2 CPU threads, llama-bench gave pp512 28.6 and tg128 15.5 tok/s, against 319.2 and 33.6 for the IQ3_XXS build; the two files also differ in other ways. In an agent check, the 30-minute limit ended before the agent got a reply. Not scored, by the maintainer's decision. |
| inkling-2-mini | Ollama 0.40.17-rc4 (pre-release, checked; installed here: 0.40.16) | inkling-2-mini (latest, 26b-a5b, 26b-a5b-nvfp4: about 20 GB each) | |  | | | not run: not tried: builds of about 20 GB | | | | | | Its library's only builds are the original checkpoint, about 20 GB in NVFP4 and MXFP8, with its drafter. We expected too little headroom on a 24 GB Mac, as the model's card asks for 32 GB or more, and did not load it. Support needs 0.40.17, a pre-release on 2026-11-21. |
| inkling-2-mini | LM Studio llmster 0.0.27-1 | none in its catalog | |  | | | not run: no support found | | | | | | We found no lmstudio-community build, and its installed runtimes predate support: its llama.cpp runtime (2.73.0) predates llama.cpp's Inkling 2 Mini support (b12600, 2026-11-20), and its MLX runtime relies on mlx-lm, whose support is a draft pull request. We checked the catalog on Hugging Face and the installed runtimes; no load was attempted. |
| inkling-2-mini | mlx-lm 0.32.2 | mlx-community/Inkling-2-Mini-4bit, 17.3 GB, made with the draft pull request | |  | | | not run: no support in a release | | | | | | Inkling 2 Mini support is a draft pull request (ml-explore/mlx-lm#2086), not reviewed on 2026-11-21, and the release we have (0.32.2) has none. mlx-community's 4-bit build is made with that branch and was not tried. |
| inkling-2-mini | kapitsa 0.2.39 (checked; installed here: 0.2.36) | none in its catalog | |  | | | not run: no support found | | | | | | Inkling 2 is not mentioned in its releases, issues or pull requests on 2026-11-21. No load was attempted. |
| inkling-2-mini | MTPLX 2.12.3 | no pack | |  | | | not run: no support found | | | | | | We found no Inkling 2 support or pack. |
| inkling-2-mini | raceline (ours) 0.8.1 | no build | |  | | | not run: no support | | | | | | Inkling 2 Mini's looped architecture would need a new implementation in raceline, which is in maintenance mode: a cache per pass, 128-expert MoE, and window and global attention with relative positions. |


## Pairwise differences

Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.

| Model | A | B | Solved, A vs B | p | Failed checks after a repeated-format exit, A vs B | p |
|---|---|---|---|---|---|---|
| inkling-2-mini | llama.cpp b12600 (29b52d398) (context 32768, prompt cache off) | oMLX 0.8.1 (memory guard aggressive) | 16/16 vs 13/16 | 0.226 | 0/16 vs 0/16 | 1.000 |


## Time

Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over tasks solved. Summed proxy request duration: the requests' durations added up; with retries it can exceed the elapsed time. Decode and prefill during the run: from each engine's own log, medians over the requests that wrote at least 64 tokens and that read at least 256 new tokens; an engine whose log has no timing of its own for these phases says so. Elapsed totals describe each listed configuration, and include the time spent on failed attempts.

| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | Tokens written | Summed proxy request duration (can exceed elapsed attempt time) | Decode during the run (median) | Prefill during the run (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inkling-2-mini | llama.cpp b12600 (29b52d398) | core | context 32768, prompt cache off (times not comparable) | 16 | 16 | 51.1 min | 192 s | 162 | 76209 | 51.0 min | 29 tok/s (150 requests) | 161 tok/s (79 requests) |
| inkling-2-mini | oMLX 0.8.1 | core | memory guard aggressive (times not comparable) | 16 | 13 | 82.4 min | 380 s | 198 | 43628 | 55.4 min | no comparable timing logged | no comparable timing logged |


## Each run

Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it understates an engine that maps its weights from the file. To compare memory across engines, use the speed section's memory taken from the system.

| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, solved attempts | Total time per solved task | Total time | Median tokens written, among attempts reporting usage | Requests (median) | Failures | Paged out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inkling-2-mini | llama.cpp b12600 (29b52d398) | InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF IQ3_XXS (279557c) | context 32768, prompt cache off | 3.6 | 11.7 GB | 7.4 GB | 3 s | 16/16 | 156 s | 192 s | 51.1 min | 3884 | 10.5 | none | 0.1 GB |
| inkling-2-mini | oMLX 0.8.1 | mlx-community/Inkling-2-Mini-oQ3e-mtp (5552e32) | defaults | | | | | 0 attempts scored — server started, warm-up refused: its first request failed: HTTP 400: oMLX prefill memory guard rejected this prompt: Prefill would require ~14.86 GB peak (current 14.86 GB + KV+SDPA 7.78 MB) but the prefill limit is 14.58 GB (dynamic ceiling 15.84 GB - safety margin 1.27 GB). | | | | | | | |
| inkling-2-mini | oMLX 0.8.1 | mlx-community/Inkling-2-Mini-oQ3e-mtp (5552e32) | memory guard aggressive | 3.81 | 13.7 GB | 19.6 GB | 13 s | 13/16 | 76 s | 380 s | 82.4 min | 2276 | 11.5 | error 3 | 0.1 GB |


## Speed

`solvetime speed`: thinking off, by each model's own switch, after one uncounted warm-up; prefill on a prompt of Python standard-library source (its size in the model's own tokens is in the cell), decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what the system had available before the engine started, minus the least it had while the decode requests ran, which is when it is sampled (free, inactive, speculative and purgeable pages); other processes can change it.

| Model | Engine | Settings | Bits per weight | Prefill | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 | Memory taken (system) | RSS | Footprint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inkling-2-mini | oMLX 0.8.1 | memory guard aggressive | 3.81 | 451 (437-471), 5.0K tokens | 30.7 (28.8-31.1) | 32.5 (31.0-32.7) | 33.1 (31.5-34.4) | 31.4 (31.2-32.1) | 15.4 GB | 2.3 GB | 16.2 GB |  |
| inkling-2-mini | llama.cpp b12600 (29b52d398) | context 32768, prompt cache off | 3.6 | 145 (131-190), 5.0K tokens | 26.3 (26.3-26.4) | 26.3 (26.3-27.2) | 26.9 (26.9-27.0) | 26.9 (26.9-26.9) | 13.6 GB | 13.7 GB | 3.4 GB |  |


## Not run

- inkling-2-mini, llama.cpp b12600 (unsloth/Inkling-2-Mini-GGUF Mainline_Support UD-Q3_K_XL (ea0c79b, 13.63 GB)): no scored run (benchmarked and checked). At b12600 llama.cpp's Metal backend does not support multiplying NVFP4 weights, and this file keeps the checkpoint's NVFP4 experts, so those multiplications fall back to the CPU. With offload of all layers requested and llama.cpp's default of 2 CPU threads, llama-bench gave pp512 28.6 and tg128 15.5 tok/s, against 319.2 and 33.6 for the IQ3_XXS build; the two files also differ in other ways. In an agent check, the 30-minute limit ended before the agent got a reply. Not scored, by the maintainer's decision.
- inkling-2-mini, Ollama 0.40.17-rc4 (pre-release, checked; installed here: 0.40.16) (inkling-2-mini (latest, 26b-a5b, 26b-a5b-nvfp4: about 20 GB each)): not tried: builds of about 20 GB. Its library's only builds are the original checkpoint, about 20 GB in NVFP4 and MXFP8, with its drafter. We expected too little headroom on a 24 GB Mac, as the model's card asks for 32 GB or more, and did not load it. Support needs 0.40.17, a pre-release on 2026-11-21.
- inkling-2-mini, LM Studio llmster 0.0.27-1 (none in its catalog): no support found. We found no lmstudio-community build, and its installed runtimes predate support: its llama.cpp runtime (2.73.0) predates llama.cpp's Inkling 2 Mini support (b12600, 2026-11-20), and its MLX runtime relies on mlx-lm, whose support is a draft pull request. We checked the catalog on Hugging Face and the installed runtimes; no load was attempted.
- inkling-2-mini, mlx-lm 0.32.2 (mlx-community/Inkling-2-Mini-4bit, 17.3 GB, made with the draft pull request): no support in a release. Inkling 2 Mini support is a draft pull request (ml-explore/mlx-lm#2086), not reviewed on 2026-11-21, and the release we have (0.32.2) has none. mlx-community's 4-bit build is made with that branch and was not tried.
- inkling-2-mini, kapitsa 0.2.39 (checked; installed here: 0.2.36) (none in its catalog): no support found. Inkling 2 is not mentioned in its releases, issues or pull requests on 2026-11-21. No load was attempted.
- inkling-2-mini, MTPLX 2.12.3 (no pack): no support found. We found no Inkling 2 support or pack.
- inkling-2-mini, raceline (ours) 0.8.1 (no build): no support. Inkling 2 Mini's looped architecture would need a new implementation in raceline, which is in maintenance mode: a cache per pass, 128-expert MoE, and window and global attention with relative positions.
