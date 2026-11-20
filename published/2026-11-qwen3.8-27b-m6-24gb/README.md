# Qwen3.8-27B on a 24 GB Mac mini (M6), November 2026

Machine: Apple M6, 24 GB, macOS 27.2.
Runs from 2026-11-18 15:11:17 to 2026-11-19 22:28:00. Agent: mini-swe-agent 2.4.6; limits: 50 steps and 900 s per attempt. Sampling sent with every request: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0, thinking on. Tables made with solvetime 0.1.0.dev0 (10bda79-dirty).

## About these runs

- **Machine.** Mac mini (2026), Apple M6 with a 12-core GPU, 24 GB, macOS 27.2, on AC power. Several terminal
  sessions and an editor stayed open throughout. For most of the measurements the browsers and System Settings
  were closed; the one row where apps were opened during the run says so. The memory left for the engine still
  varied from run to run, so every run records it at its start, right after the GPU calibration:

  | Row | Available at start | Of which other apps |
  |---|---|---|
  | raceline (2026-11-18 15:11) | 11.9 GB | 5.1 GB |
  | kapitsa 0.2.36 (11-18 16:02) | 15.6 GB | 2.6 GB |
  | kapitsa main (11-18 17:14) | 16.4 GB | 2.5 GB |
  | mlx-lm, default (11-19 09:23) | 10.4 GB | 6.3 GB |
  | mlx-lm, 16384-token replies (11-19 11:28) | 14.1 GB | 3.6 GB |
  | LM Studio, guardrails off (11-19 14:22) | 15.2 GB | 4.2 GB |
  | oMLX, guard "aggressive" (11-19 15:54) | 15.6 GB | 3.1 GB |
  | MTPLX (11-19 21:16) | 13.3 GB | 5.4 GB |

- **The model.** Qwen3.8-27B, a dense model (no experts) with 64 layers: 48 linear-attention (Gated DeltaNet)
  layers and 16 full-attention layers, one in every four. raceline, kapitsa, LM Studio and MTPLX run the build
  each picks or recommends for a 24 GB Mac. mlx-lm and oMLX recommend none, so they run the mlx-community MLX 4-bit
  build. All of them run at their default settings. A row whose settings are not the engine's
  defaults says so in its name and its settings column; each of those was the user's decision, after the default
  failed to load or to run.
- **Builds and windows.**
  - raceline 0.8.1: its `qwen3.8-27b` file (FP4, 4.0 bits, with its MTP head), with a 32,768-token window set by
    solvetime's adapter for this model. raceline is the author's own engine.
  - kapitsa 0.2.36 (fe6a17a), the Homebrew release, and kapitsa main at c1c42e6 (21 commits later), built from
    source. Both ran on the same pack: kapitsa's q4 (4.43 bits per weight), its MTP drafter, and the
    16,384-token window kapitsa sizes for 24 GB. Main carries two fixes:
    - [PR #1279](https://github.com/nace-ai/kapitsa/pull/1279) accepts a Qwen tool call that is missing its
      `<tool_call>` tag ([issue #1277](https://github.com/nace-ai/kapitsa/issues/1277));
    - [PR #1302](https://github.com/nace-ai/kapitsa/pull/1302) makes the memory fit count the copy that a growing
      KV cache needs ([issue #1300](https://github.com/nace-ai/kapitsa/issues/1300)).

    kapitsa 0.2.37 contains c1c42e6. Both runs started with kapitsa's SSD prefix cache cleared.
  - mlx-lm 0.32.2 on mlx-community/Qwen3.8-27B-4bit at b935172 (4.43 bits per weight), plain decoding, no
    fixed window. The local copy matches that revision file for file, except tokenizer.json. That file is a
    re-serialization, and it encodes and decodes identically on 915 texts (150,291 tokens) from the round.
    mlx_lm.server's default reply limit is 512 tokens (`--max-tokens`). solvetime sends no limit, so the default
    row runs with it, and a second, non-default row allows 16,384.
  - LM Studio (llmster 0.0.27-1, MLX runtime 1.14.0): `qwen/qwen3.8-27b@4bit` (lmstudio-community MLX 4-bit,
    4.43 bits per weight), with a 16,384-token window. At its default resource guardrail ("relaxed") it refuses to
    load the model, estimating 20.97 GiB with "Confidence: LOW". The non-default row turns the guardrail off.
  - oMLX 0.8.0 on the same mlx-community MLX 4-bit weights as mlx-lm. At its default memory guard ("balanced") it refuses the
    model, because its ceiling follows free memory: 13.2 GiB at 09:20, and 15.5 GiB at the best reading (11:15),
    against the model's 15.70. The non-default row uses its "aggressive" guard. oMLX's paged SSD KV cache is on
    by default and may use up to half the free disk; it wrote 16 GB during the agent run. Each oMLX run had its
    own fresh settings folder (`--base-path`).
  - MTPLX 2.12.3: `Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed` (4.5 bits per weight), its "tight_fit" pack for
    24 GB.
- **Names in the tables.** Each run's `run.json` keeps the version it recorded:
  - `raceline 0.8.1 (v0.8.1)` is shown as raceline 0.8.1 (ours);
  - `kapitsa 0.2.36 (fe6a17a)` as kapitsa 0.2.36 (release, fe6a17a);
  - `kapitsa 0.2.36 (c1c42e6)` as kapitsa main c1c42e6 (fixes #1279, #1302);
  - LM Studio's runtime list as LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0).
- **Tasks.** The core set (8 tasks), two attempts each, round-robin. Thinking on, the model card's sampling; 50
  steps and 900 s per attempt.
- **Harness.** The rows ran on solvetime d0df79f (2026-11-18), 984f250, 9f793fc, 5dba5ea and e98dbd7
  (2026-11-19). The agent (mini-swe-agent 2.4.6), its prompts and limits, the sampling, the requests, the
  checks, the sandbox and the speed test did not change between them. The later commits changed only records,
  pre-run download checks and settings that act only when set. One exception: from 5dba5ea every oMLX run starts
  from a fresh settings folder. The oMLX default refusal (984f250) ran with oMLX's default guard tier, so it is
  equivalent. Since d0df79f the sandbox ends an agent's whole-disk search at once, and attempts can't list each
  other's directories.
- **What happened in each row,** beyond the tables:
  - **mlx-lm, default:** all 7 failures come from the 512-token reply limit. 43 of 128 replies ended at it, and
    each failed attempt ended on three such replies in a row.
  - **mlx-lm, 16,384-token replies:**
    - 3 attempts ran out of time on long thinking at about 9.4 tok/s and failed. Two more ran out of time after
      their fix was already in place, and pass the check.
    - At 14:11, during an ~8.6K-token prompt, the server's generation thread died ("[METAL] Command buffer
      execution failed: Insufficient Memory"). The server kept running and answered 404 to every later request,
      so the last 4 attempts were lost to the crash, not to the model.
  - **LM Studio, guardrails off:** in attempt 3 its MLX engine died with the same Metal out-of-memory error. The
    engine process stayed alive holding 19 GB, and every later request got "No models loaded", so 13 attempts
    were lost to the crash. Its default guardrail's refusal was, in effect, prudent. No request was truncated (no
    "TruncateMiddle" in its log).
  - **oMLX, guard "aggressive":** no crash. At run time its guard rejected 8 prompts and aborted 70 requests while
    keeping the model loaded. The agent's retries absorbed most of them, at a cost in time. 3 failures are
    requests that stayed aborted, and 1 is a timeout.
  - **MTPLX:** its per-request admission guard refused every prompt (483 refusals), because the other apps left
    1.3–2.0 GiB free against its floors of 2.0–2.4 GiB. No crash. Other apps held 5.4 GB at its start, the most of
    the round. Chrome, Xcode and Pages were opened during the run (21:32–21:43), after it was already refusing.
    By the user's decision it is recorded as is, without a rerun.
- **Speed tests:** thinking off; a 4K-token prompt for prefill; up to 512 tokens of decode on two prompts, three
  runs each.
  - LM Studio and MTPLX: not measured. The prefill request asked for one token, and got none back:
    - LM Studio's MLX runtime returns max_tokens − 1 tokens
      ([lmstudio-bug-tracker#2602](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2602));
    - MTPLX warns that one token "leaves no room to answer".

    solvetime fixed this after the round.
  - oMLX "aggressive": prefill from run 1 only (317 tok/s), by the user's decision. Runs 2 and 3 overlapped 14 s of
    unrelated unit tests on the same Mac and are excluded (`speed-exclusions.json`). Decode is from all runs.
- **Left out** (in `results/_excluded`, with notes):
  - the 27B runs from 2026-11-17: the harness held about 3.5 GB of memory during every run (the calibration's MLX
    cache, fixed in #27) and let agents scan the whole disk;
  - the run from the morning of 2026-11-18: still the whole-disk scan, which #29 removed;
  - an oMLX run at 14:15 on 2026-11-19, stopped after 2 minutes, because it started with the "aggressive" tier
    left behind by an earlier probe;
  - two oMLX speed files from 09:21–09:22, where the speed test stopped on the 507 refusal instead of recording it
    (fixed in #32);
  - LM Studio's 11:25 default record, superseded by the 14:21 one, which names the reason;
  - two later oMLX speed reruns: one ended on critical memory pressure, and the other was refused by its guard;
  - LM Studio's two speed tries with the guardrail off: one while the crashed engine still held 19 GB, and one at
    20:44 whose 1-token prefill request got nothing back;
  - a run recorded while LM Studio's download had been aborted.
- **Not run:** Ollama 0.40.16. Memory pressure turned critical while it loaded at a 64K window, and again in the
  first attempt at a 16K window, where its process grew past its own memory estimate (see `not-run.json` and
  [ollama/ollama#19757](https://github.com/ollama/ollama/issues/19757)).

## Agent tasks, each engine build pooled over its task sets

Solved with a 95% Wilson interval. "Ended by malformed output": the attempt stopped after three replies in a row that the agent could not use (no tool call). "Replies without a tool call" counts every such reply, recovered or not, out of all replies. "Available at start": the memory macOS could hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages); the rest was in use by macOS and the other open programs. On a 24 GB Mac it decides whether a 15-19 GB build fits. Runs made before 2026-11-18 did not record it.

| Model | Engine | Build | Settings | Bits per weight | Available at start | Tasks | Solved | Ended by malformed output | Replies without a tool call | Replies cut off at the token limit | Median time, solved attempts | Other failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | qwen3.8-27b (FP4, 4.0 bits, with its MTP head) | defaults | 4.04 | 11.9 GB | 8 (core) | 16/16 (100%, 95% CI 81-100%) | 0/16 | 0 of 171 | 3 | 119 s | none |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 15.6 GB | 8 (core) | 14/16 (88%, 95% CI 64-97%) | 1/16 | 0 of 147 | 7 | 162 s | wrong 1 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 16.4 GB | 8 (core) | 16/16 (100%, 95% CI 81-100%) | 0/16 | 0 of 159 | 7 | 154 s | none |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | 4.43 | 10.4 GB | 8 (core) | 9/16 (56%, 95% CI 33-77%) | 7/16 | 0 of 128 | 41 | 422 s | none |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 14.1 GB | 8 (core) | 9/16 (56%, 95% CI 33-77%) | 0/16 | 0 of 109 | 0 | 552 s | error 4, timeout 3 |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b (lmstudio-community/Qwen3.8-27B-MLX-4bit) | context 16384, guardrails off | 4.43 | 15.2 GB | 8 (core) | 2/16 (12%, 95% CI 3-36%) | 0/16 | 0 of 171 | 0 | 451 s | error 14 |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | memory guard aggressive | 4.43 | 15.6 GB | 8 (core) | 12/16 (75%, 95% CI 51-90%) | 0/16 | 0 of 208 | 0 | 601 s | error 3, loop 1 |
| qwen3.8-27b | MTPLX 2.12.3 | Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed | defaults | 4.5 | 13.3 GB | 8 (core) | 0/16 (0%, 95% CI 0-19%) | 0/16 | 0 of 484 | 0 |  | error 16 |
| qwen3.8-27b | Ollama 0.40.16 | qwen3.8:27b (q4_K_M, 17.74 GB) | |  | | | not run: does not fit | | | | | Memory pressure turned critical twice on this 24 GB Mac: while loading at a 64K window (Ollama's own estimate: 20.6 GB of GPU memory, 18.2 GB free) and in the first attempt at a 16K window (llama-server reached 21 GB against its 17.3 GB estimate; Ollama disables mmap here). Its own default window here is 4K, which would cut an agent's conversation. |
| qwen3.8-27b | LM Studio llmster 0.0.27-1, mlx-llm 1.14.0 | qwen/qwen3.8-27b@4bit (lmstudio-community MLX 4-bit, 4.43 bits per weight), 16K window, guardrails off (non-default) | |  | | | not run: speed not measured | | | | | First try (2026-11-19 15:48): memory pressure turned critical while the model loaded, because the engine of the agent run before it had crashed (Metal out of memory) but kept its 19 GB until stopped, and the reload overlapped its teardown. Retry from a settled state (20:44): the model loaded, but the prefill test got no token back: LM Studio's MLX runtime returns max_tokens - 1 tokens (asked for 1, it returns none; lmstudio-ai/lmstudio-bug-tracker#2602), and it ignores chat_template_kwargs.enable_thinking (only reasoning_effort "none" turns its thinking off), so the speed test's requests did not fit it. Found with qwen3.5-4b at 21:0x; the speed test is fixed after the round. The 27B was removed after the retry, so it was not measured again. The user's choice for a second failure (#44): record it as not measured. |
| qwen3.8-27b | MTPLX 2.12.3 | Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed (4.5 bpw, its tight fit for 24 GB) | |  | | | not run: speed not measured | | | | | The speed test (speed-20261119-222857, 22:28) loaded it (cold start 9.5 s), and its warm-up requests decoded at 19.5-23.8 tok/s by its own log. The prefill request asked for 1 token and got none back; MTPLX's log warns "max_tokens leaves no room to answer". The speed test asks for 8 since solvetime #37, but the pack had been removed by then, and the user's #47 kept the MTPLX row as measured. |


## Pairwise differences

Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.

| Model | A | B | Solved, A vs B | p | Ended by malformed output, A vs B | p |
|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | kapitsa 0.2.36 (release, fe6a17a) | 16/16 vs 14/16 | 0.484 | 0/16 vs 1/16 | 1.000 |
| qwen3.8-27b | raceline 0.8.1 (ours) | kapitsa main c1c42e6 (fixes #1279, #1302) | 16/16 vs 16/16 | 1.000 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | raceline 0.8.1 (ours) | mlx-lm 0.32.2 | 16/16 vs 9/16 | 0.007 | 0/16 vs 7/16 | 0.007 |
| qwen3.8-27b | raceline 0.8.1 (ours) | mlx-lm 0.32.2 (max tokens 16384) | 16/16 vs 9/16 | 0.007 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | raceline 0.8.1 (ours) | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | 16/16 vs 2/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | raceline 0.8.1 (ours) | oMLX 0.8.0 (memory guard aggressive) | 16/16 vs 12/16 | 0.101 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | raceline 0.8.1 (ours) | MTPLX 2.12.3 | 16/16 vs 0/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | kapitsa main c1c42e6 (fixes #1279, #1302) | 14/16 vs 16/16 | 0.484 | 1/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | mlx-lm 0.32.2 | 14/16 vs 9/16 | 0.113 | 1/16 vs 7/16 | 0.037 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | mlx-lm 0.32.2 (max tokens 16384) | 14/16 vs 9/16 | 0.113 | 1/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | 14/16 vs 2/16 | < 0.001 | 1/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | oMLX 0.8.0 (memory guard aggressive) | 14/16 vs 12/16 | 0.654 | 1/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | MTPLX 2.12.3 | 14/16 vs 0/16 | < 0.001 | 1/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | mlx-lm 0.32.2 | 16/16 vs 9/16 | 0.007 | 0/16 vs 7/16 | 0.007 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | mlx-lm 0.32.2 (max tokens 16384) | 16/16 vs 9/16 | 0.007 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | 16/16 vs 2/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | oMLX 0.8.0 (memory guard aggressive) | 16/16 vs 12/16 | 0.101 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | MTPLX 2.12.3 | 16/16 vs 0/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-lm 0.32.2 (max tokens 16384) | 9/16 vs 9/16 | 1.000 | 7/16 vs 0/16 | 0.007 |
| qwen3.8-27b | mlx-lm 0.32.2 | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | 9/16 vs 2/16 | 0.023 | 7/16 vs 0/16 | 0.007 |
| qwen3.8-27b | mlx-lm 0.32.2 | oMLX 0.8.0 (memory guard aggressive) | 9/16 vs 12/16 | 0.458 | 7/16 vs 0/16 | 0.007 |
| qwen3.8-27b | mlx-lm 0.32.2 | MTPLX 2.12.3 | 9/16 vs 0/16 | < 0.001 | 7/16 vs 0/16 | 0.007 |
| qwen3.8-27b | mlx-lm 0.32.2 (max tokens 16384) | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | 9/16 vs 2/16 | 0.023 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | mlx-lm 0.32.2 (max tokens 16384) | oMLX 0.8.0 (memory guard aggressive) | 9/16 vs 12/16 | 0.458 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | mlx-lm 0.32.2 (max tokens 16384) | MTPLX 2.12.3 | 9/16 vs 0/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | oMLX 0.8.0 (memory guard aggressive) | 2/16 vs 12/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) (context 16384, guardrails off) | MTPLX 2.12.3 | 2/16 vs 0/16 | 0.484 | 0/16 vs 0/16 | 1.000 |
| qwen3.8-27b | oMLX 0.8.0 (memory guard aggressive) | MTPLX 2.12.3 | 12/16 vs 0/16 | < 0.001 | 0/16 vs 0/16 | 1.000 |


## Time

Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over tasks solved. Time in the engine: the requests themselves. Decode and prefill during the run: from each engine's own log, medians over the requests that wrote at least 64 tokens and that read at least 256 new tokens. A run with settings beyond the defaults has times that compare with nothing else.

| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | Tokens written | Time in the engine | Decode during the run (median) | Prefill during the run (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | core | defaults | 16 | 16 | 45.8 min | 172 s | 171 | 92734 | 45.7 min | 40 tok/s (170 requests) | 267 tok/s (70 requests) |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | core | defaults | 16 | 14 | 65.8 min | 282 s | 147 | 85648 | 65.7 min | 25 tok/s (145 requests) | 287 tok/s (65 requests) |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | core | defaults | 16 | 16 | 66.4 min | 249 s | 159 | 92140 | 66.3 min | 27 tok/s (155 requests) | 287 tok/s (70 requests) |
| qwen3.8-27b | mlx-lm 0.32.2 | core | defaults | 16 | 9 | 98.0 min | 653 s | 128 | 44082 | 97.9 min | not logged | not logged |
| qwen3.8-27b | mlx-lm 0.32.2 | core | max tokens 16384 (times not comparable) | 16 | 9 | 162.7 min | 1085 s | 109 | 76960 | 162.9 min | not logged | not logged |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | core | context 16384, guardrails off (times not comparable) | 16 | 2 | 85.0 min | 2549 s | 171 | 14026 | 27.0 min | not logged | not logged |
| qwen3.8-27b | oMLX 0.8.0 | core | memory guard aggressive (times not comparable) | 16 | 12 | 276.1 min | 1380 s | 208 | not reported | 247.8 min | not logged | not logged |
| qwen3.8-27b | MTPLX 2.12.3 | core | defaults | 16 | 0 | 71.4 min |  | 484 | 138 | 1.6 min | not logged | not logged |


## Each run

Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it understates an engine that maps its weights from the file. To compare memory across engines, use the speed section's memory taken from the system.

| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, solved attempts | Total time per solved task | Total time | Tokens written (median) | Requests (median) | Failures | Paged out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | qwen3.8-27b (FP4, 4.0 bits, with its MTP head) | defaults | 4.04 | 13.8 GB | 7.2 GB | 4 s | 16/16 | 119 s | 172 s | 45.8 min | 4330 | 10 | none | 0.0 GB |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 20.2 GB | 18.5 GB | 5 s | 14/16 | 162 s | 282 s | 65.8 min | 3752 | 10 | malformed output 1, wrong 1 | 0.0 GB |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 20.2 GB | 18.5 GB | 7 s | 16/16 | 154 s | 249 s | 66.4 min | 3891 | 9 | none | 0.0 GB |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | | | | | does not start: its first request failed: HTTP 507: {"error":{"message":"Model 'Qwen3.8-27B-4bit' (15.70GB) does not fit under the dynamic memory ceiling (13.18GB). Close other apps to free RAM (macOS and other apps leave oMLX 13.18GB of the 21.00GB static cap, and oMLX itself uses 208.84MB of that), raise memory_guard_tier (safe → balanced → aggressive), or use a smaller model.","type":"server_error","param":null,"code":null}} | | | | | | | |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | 4.43 | 16.1 GB | 19.4 GB | 6 s | 9/16 | 422 s | 653 s | 98.0 min | 2569 | 8 | malformed output 4, loop 3 | 0.0 GB |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 16.1 GB | 19.5 GB | 5 s | 9/16 | 552 s | 1085 s | 162.7 min | 6034 | 8 | error 4, timeout 3 | 0.0 GB |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b | context 16384 | | | | | does not start: exited with code -15: Error: Model loading was stopped due to insufficient system resources. Continuing to load the model would likely overload your system and cause it to freeze. If you think this is incorrect, you can adjust the model loading guardrails in settings. | | | | | | | |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b (lmstudio-community/Qwen3.8-27B-MLX-4bit) | context 16384, guardrails off | 4.43 | 16.1 GB | not measured | 17 s | 2/16 | 451 s | 2549 s | 85.0 min | 3940 | 10 | error 14 | 0.1 GB |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | memory guard aggressive | 4.43 | 16.1 GB | 19.6 GB | 20 s | 12/16 | 601 s | 1380 s | 276.1 min | not reported | 11 | error 3, loop 1 | 0.0 GB |
| qwen3.8-27b | MTPLX 2.12.3 | Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed | defaults | 4.5 | 16.3 GB | 19.1 GB | 10 s | 0/16 |  | nothing solved | 71.4 min | 138 | 30 | error 16 | 0.0 GB |


## Speed

`solvetime speed`: thinking off, after one uncounted warm-up; prefill on a 4K-token prompt, decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what the system had available before the engine started, minus the least it had while the engine ran (free, inactive, speculative and purgeable pages), on an otherwise quiet machine.

Runs left out of a cell are named in its row's notes, and listed with the reason in speed-exclusions.json.

| Model | Engine | Bits per weight | Prefill 4K | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 | Memory taken (system) | RSS | Footprint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | 4.04 | 292 (279-296) | 35.4 (33.3-38.5) | 31.7 (30.8-31.9) | 40.6 (39.8-41.0) | 40.5 (40.2-40.5) | 14.9 GB | 1.9 GB | 6.1 GB |  |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | 4.43 | 384 (377-387) | 26.3 (24.1-27.6) | 27.6 (27.6-27.8) | 30.8 (30.4-30.9) | 30.8 (30.7-30.8) | 14.6 GB | 16.5 GB | 18.3 GB |  |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | 4.43 | 385 (379-386) | 24.9 (23.7-26.9) | 27.7 (27.3-28.2) | 30.5 (29.9-30.7) | 30.7 (30.7-30.8) | 14.7 GB | 16.5 GB | 18.3 GB |  |
| qwen3.8-27b | mlx-lm 0.32.2 | 4.43 | 289 (286-291) | 9.3 (9.2-9.4) | 9.4 (9.3-9.4) | 9.5 (9.4-9.5) | 9.5 (9.5-9.6) | 13.0 GB | 14.2 GB | 16.9 GB |  |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | | | | | | | | | | does not start: exited with code -15: Error: Model loading was stopped due to insufficient system resources. Continuing to load the model would likely overload your system and cause it to freeze. If you think this is incorrect, you can adjust the model loading guardrails in settings. |
| qwen3.8-27b | oMLX 0.8.0 | | | | | | | | | | does not start: its first request failed: HTTP 507: {"error":{"message":"Cannot load Qwen3.8-27B-4bit: projected memory 15.90GB would exceed the dynamic memory ceiling 15.85GB (current: 208.78MB, model: 15.70GB). Close other apps to free RAM (macOS and other apps leave oMLX 15.84GB of the 21.00GB static cap, and oMLX itself uses 208.78MB of that), raise memory_guard_tier (safe → balanced → aggressive), or unload another model.","type":"server_e |
| qwen3.8-27b | oMLX 0.8.0 | 4.43 | 317 (317-317) | 9.9 (9.4-9.9) | 9.9 (9.9-9.9) | 9.9 (9.9-9.9) | 9.9 (9.9-9.9) | 13.0 GB | 1.1 GB | 17.8 GB | prefill runs 2, 3 left out: they overlapped 14 s of unit tests run on the same Mac; the user's #51 kept run 1 only |


## Not run

- qwen3.8-27b, Ollama 0.40.16 (qwen3.8:27b (q4_K_M, 17.74 GB)): does not fit. Memory pressure turned critical twice on this 24 GB Mac: while loading at a 64K window (Ollama's own estimate: 20.6 GB of GPU memory, 18.2 GB free) and in the first attempt at a 16K window (llama-server reached 21 GB against its 17.3 GB estimate; Ollama disables mmap here). Its own default window here is 4K, which would cut an agent's conversation.
- qwen3.8-27b, LM Studio llmster 0.0.27-1, mlx-llm 1.14.0 (qwen/qwen3.8-27b@4bit (lmstudio-community MLX 4-bit, 4.43 bits per weight), 16K window, guardrails off (non-default)): speed not measured. First try (2026-11-19 15:48): memory pressure turned critical while the model loaded, because the engine of the agent run before it had crashed (Metal out of memory) but kept its 19 GB until stopped, and the reload overlapped its teardown. Retry from a settled state (20:44): the model loaded, but the prefill test got no token back: LM Studio's MLX runtime returns max_tokens - 1 tokens (asked for 1, it returns none; lmstudio-ai/lmstudio-bug-tracker#2602), and it ignores chat_template_kwargs.enable_thinking (only reasoning_effort "none" turns its thinking off), so the speed test's requests did not fit it. Found with qwen3.5-4b at 21:0x; the speed test is fixed after the round. The 27B was removed after the retry, so it was not measured again. The user's choice for a second failure (#44): record it as not measured.
- qwen3.8-27b, MTPLX 2.12.3 (Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed (4.5 bpw, its tight fit for 24 GB)): speed not measured. The speed test (speed-20261119-222857, 22:28) loaded it (cold start 9.5 s), and its warm-up requests decoded at 19.5-23.8 tok/s by its own log. The prefill request asked for 1 token and got none back; MTPLX's log warns "max_tokens leaves no room to answer". The speed test asks for 8 since solvetime #37, but the pack had been removed by then, and the user's #47 kept the MTPLX row as measured.
