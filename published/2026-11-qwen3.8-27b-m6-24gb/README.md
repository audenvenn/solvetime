# Qwen3.8-27B on a 24 GB Mac mini (M6), November 2026

Machine: Apple M6, 24 GB, macOS 27.2.
Runs from 2026-11-18 15:11:17 to 2026-11-19 22:28:00. Agent: mini-swe-agent 2.4.6; limits: 50 steps, and a 900 s budget the agent checks before each query (a query in flight and its retries can run past it). Sampling sent with every request: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0, thinking on. Tables made with solvetime 0.1.0.dev0 (ce79f07).

## About these runs

- **What the tables compare.** Each scored configuration ran the 8 core tasks twice. The tables compare the
  listed engine, model build and settings on this Mac. Builds, windows, decoding methods and the memory free at
  each start all differed, so the scores don't say how much any single factor contributed.
- **Machine.** Mac mini (2026), Apple M6 with a 12-core GPU, 24 GB, macOS 27.2, on AC power. Several terminal
  sessions and an editor stayed open throughout. For most of the measurements the browsers and System Settings
  were closed; the one row where apps were opened during the run says so. Each run takes two separate readings
  just before it launches the engine, and each engine then applied its own guard as memory changed:
  - the estimated available memory, from four `vm_stat` counters;
  - the system-wide anonymous memory: all apps together, not attributed to any one app.

  | Run | Estimated available before launch | System-wide anonymous memory |
  |---|---|---|
  | raceline (2026-11-18 15:11) | 11.9 GB | 5.1 GB |
  | kapitsa 0.2.36 (11-18 16:02) | 15.6 GB | 2.6 GB |
  | kapitsa main (11-18 17:14) | 16.4 GB | 2.5 GB |
  | oMLX, default guard; warm-up refused, no attempt scored (11-19 09:20) | 11.1 GB | 6.2 GB |
  | mlx-lm, default (11-19 09:23) | 10.4 GB | 6.3 GB |
  | mlx-lm, replies up to 16,384 tokens (11-19 11:28) | 14.1 GB | 3.6 GB |
  | LM Studio, default guardrail; load refused, no attempt scored (11-19 14:21) | 15.1 GB | 4.0 GB |
  | LM Studio, guardrails off (11-19 14:22) | 15.2 GB | 4.2 GB |
  | oMLX, guard "aggressive" (11-19 15:54) | 15.6 GB | 3.1 GB |
  | MTPLX (11-19 21:16) | 13.3 GB | 5.4 GB |

  **Corrected on 2026-11-21.** The estimated available memory reads low, here and in the tables below. Both
  readings were taken the moment solvetime's GPU calibration process had ended, before macOS released the
  memory that process had used. That memory stays wired for about a second: 2.1 GB in a test on 2026-11-21.
  - **The size of the error.** At these launches wired memory read 5.3–6.5 GB, against about 2.4–2.8 GB at idle.
    So the estimate reads low by roughly 2.5–4 GB.
  - **The speed table.** "Memory taken (system)" counts from the same reading, so it reads low by about as much.
  - **Not affected.** The anonymous-memory readings leave wired memory out. The engines started seconds later,
    after the release.

  The records stay as measured; solvetime #53 reads the memory after the release.

- **The model.** Qwen3.8-27B, a dense model (no experts) with 64 layers: 48 linear-attention (Gated DeltaNet)
  layers and 16 full-attention layers, one in every four. raceline, kapitsa, LM Studio and MTPLX run the build
  each picks or recommends for a 24 GB Mac. mlx-lm and oMLX recommend none, so they run the mlx-community MLX 4-bit
  build.
- **Builds and windows.**
  - raceline 0.8.1: its `qwen3.8-27b` file (FP4, 4.0 bits, with its MTP head). solvetime's adapter sets a
    32,768-token window for this model, which is also raceline's default here. raceline is the author's own engine.
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
    mlx_lm.server caps a reply at 512 tokens (`--max-tokens`) unless the request asks for more. solvetime asks
    for no limit, so the default row runs with that cap. The second row, tried after the default run's cut-off
    replies, allows 16,384.
  - LM Studio (llmster 0.0.27-1, MLX runtime 1.14.0): `qwen/qwen3.8-27b@4bit` (lmstudio-community MLX 4-bit,
    4.43 bits per weight), pinned because `lms get` picks a variant by hardware, with an explicit 16,384-token
    window. Its default resource guardrail ("relaxed") declined to load the model, estimating 20.97 GiB with
    "Confidence: LOW". The scored row turns the guardrail off.
  - oMLX 0.8.0 on the same mlx-community MLX 4-bit weights as mlx-lm. Its default memory guard ("balanced")
    declined the model, because its ceiling follows free memory. By oMLX's own figures:
    - at 09:20, a 13.2 GB ceiling against the model's 15.70 GB;
    - at the best morning probe (11:15), 15.5;
    - in the afternoon speed test, a 15.85 GB ceiling against a projected 15.90 GB.

    The scored row uses the "aggressive" guard and a fresh settings folder. The earlier balanced run used the
    existing settings folder, and got a balanced-tier memory refusal. A scan during the aggressive agent run found
    about 16 GB of oMLX's SSD cache files, which are on by default.
  - MTPLX 2.12.3: `Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed` (4.5 bits per weight), its "tight_fit" pack for
    24 GB. Its startup log says the model does not fit this Mac, so it set its 4,096-token floor as the window.
- **Names in the tables.** Each run's `run.json` keeps the version it recorded:
  - `raceline 0.8.1 (v0.8.1)` is shown as raceline 0.8.1 (ours);
  - `kapitsa 0.2.36 (fe6a17a)` as kapitsa 0.2.36 (release, fe6a17a);
  - `kapitsa 0.2.36 (c1c42e6)` as kapitsa main c1c42e6 (fixes #1279, #1302);
  - LM Studio's runtime list as LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0).
- **Tasks and limits.** The core set (8 tasks), two attempts each, round-robin. Thinking on, with the model
  card's sampling. 50 steps. The agent checks a 900-second budget before it starts each query, so a query
  already in flight, and its retries, can run past it: the longest attempt lasted 47.2 minutes. A check scores
  the final workspace, so an attempt passes if its fix was in place, even when the agent then ended on an error.
- **Corrected on 2026-11-22: the sampling.** "The model card's sampling" above is not accurate for Qwen3.8.
  - **What was sent:** temperature 0.6, top_p 0.95, top_k 20, min_p 0 and presence_penalty 0. That is the Qwen3.5 and
    3.6 cards' setting for precise coding tasks, which solvetime applied to Qwen3.8 too.
  - **What Qwen3.8's card recommends:** temperature 1.0 in thinking mode.
  - **The comparisons stand.** Every engine on this page got the same settings, so its comparisons are not affected.
    The records stay as measured.
  - **Later runs.** After round 4 (2026-11-22), solvetime runs Qwen3.8 at its card's 1.0, so later 27B results with
    thinking are not directly comparable with this page's.
- **Harness.** The rows ran on solvetime d0df79f (2026-11-18), 984f250, 9f793fc, 5dba5ea and e98dbd7
  (2026-11-19). The agent (mini-swe-agent 2.4.6), its prompts and limits, the sampling, the requests, the
  checks, the sandbox and the speed test did not change between them. The later commits changed only records,
  pre-run download checks and settings that act only when set. One exception: from 5dba5ea every oMLX run starts
  from a fresh settings folder. Since d0df79f the sandbox ends an agent's whole-disk search at once, and attempts
  can't list each other's directories.
- **What happened in each row,** beyond the tables:
  - **kapitsa 0.2.36:** one attempt ended after three tool calls that stopped at the length limit before their
    closing tags; the other failed its check. Main passed all 16. This comparison doesn't show which of main's 21
    changes, if any, made the difference.
  - **mlx-lm, default:** 43 of its 128 replies reached the 512-token cap, 41 of them without a tool call. All 7
    failed attempts ended after three cut-off replies in a row.
  - **mlx-lm, replies up to 16,384 tokens:**
    - 3 attempts ran out of time on long thinking and failed. Two more ran out of time after their fix was
      already in place, and pass the check.
    - The server's generation thread died during attempt 12, which still passed its check ("[METAL] Command
      buffer execution failed: Insufficient Memory"). The server kept running and answered 404, so attempts 13–16
      failed at once.
    - Matched by task and attempt against the default row, two failures became passes (docstrings 1, bisect 1)
      and two passes became failures: pathlib 1 ran out of time, and config-defaults 2 came after the crash.
  - **LM Studio, guardrails off:** the first two attempts passed. Its MLX engine ran out of memory during attempt
    3, and kept its 19 GB while attempts 4–16 spent about 54 minutes on requests answered with "No models
    loaded". No request was truncated (no "TruncateMiddle" in its log).
  - **oMLX, guard "aggressive":** the server stayed up through 64 requests aborted under hard memory pressure, 8
    prompts rejected by its prefill guard, and 4 model evictions, each followed by a reload (16:57, 17:44, 19:22
    and 20:11). 3 failures are requests that stayed aborted after retries, and 1 is a timeout.
  - **MTPLX:** its per-request memory guard refused 483 of its 484 requests, because the other apps left 1.3–2.0
    GiB free against its floors of 2.0–2.4 GiB. One request returned a tool call; no attempt passed. Other apps
    held 5.4 GB at its start, less than at the default mlx-lm and oMLX starts (6.3 and 6.2 GB). The refusals began
    before Chrome, Xcode and Pages were opened (21:32–21:43). By the user's decision it is recorded as is,
    without a rerun.
- **Statistics.** The Wilson intervals and two-sided Fisher tests treat attempts as independent. Repeated tasks
  and shared failures limit that: one crash ended four attempts in one row, and another crash fourteen. The
  p-values are not adjusted for the number of pairs, and none of them tests timing.
- **Speed tests:** thinking off; a 4K-token prompt for prefill; up to 512 tokens of decode on two prompts, three
  runs each. They are separate measurements, not the speeds of the agent runs.
  - LM Studio (guardrails off) and MTPLX: no speed result. The prefill request asked for one token, and got none
    back:
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
  - LM Studio's two speed tries with the guardrail off: one while the crashed engine still held 19 GB, and one at
    20:44 whose 1-token prefill request got nothing back;
  - two later oMLX speed reruns: one ended on critical memory pressure, and the other was refused by its guard;
  - a run recorded while LM Studio's download had been aborted.
- **Ollama 0.40.16 has no scored result.** Memory pressure stopped its 64K run during loading and its 16K run
  during the first attempt, where its process grew past its own memory estimate (see `not-run.json` and
  [ollama/ollama#19757](https://github.com/ollama/ollama/issues/19757)). Its native 4K window was not tried.

## Agent tasks, each engine build pooled over its task sets

Solved: the attempt's final workspace passed the task's check, work done before an error included, with a 95% Wilson interval. "Failed checks after a repeated-format exit": attempts the agent stopped after three replies in a row it could not use, whose check then failed. "Requests": every request the agent sent, the ones that failed included. "Agent tool-call error notices" and "Agent output-limit notices": the agent's own feedback messages, for a reply without a usable tool call and for a reply cut off at the output limit. "Estimated available before launch": the memory macOS could hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages), measured after the harness's GPU calibration; runs made before 2026-11-18 did not record it. A run whose engine never served an attempt is listed with how it stopped, and no score.

| Model | Engine | Build | Settings | Bits per weight | Estimated available before launch | Tasks | Solved | Failed checks after a repeated-format exit | Requests | Agent tool-call error notices | Agent output-limit notices | Median time, solved attempts | Other failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | qwen3.8-27b (FP4, 4.0 bits, with its MTP head) | defaults | 4.04 | 11.9 GB | 8 (core) | 16/16 (100%, 95% CI 81-100%) | 0/16 | 171 | 0 | 3 | 119 s | none |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 15.6 GB | 8 (core) | 14/16 (88%, 95% CI 64-97%) | 1/16 | 147 | 0 | 7 | 162 s | wrong 1 |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 16.4 GB | 8 (core) | 16/16 (100%, 95% CI 81-100%) | 0/16 | 159 | 0 | 7 | 154 s | none |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | 4.43 | 10.4 GB | 8 (core) | 9/16 (56%, 95% CI 33-77%) | 7/16 | 128 | 0 | 41 | 422 s | none |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 14.1 GB | 8 (core) | 9/16 (56%, 95% CI 33-77%) | 0/16 | 109 | 0 | 0 | 552 s | error 4, timeout 3 |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b (lmstudio-community/Qwen3.8-27B-MLX-4bit) | context 16384, guardrails off | 4.43 | 15.2 GB | 8 (core) | 2/16 (12%, 95% CI 3-36%) | 0/16 | 171 | 0 | 0 | 451 s | error 14 |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | memory guard aggressive | 4.43 | 15.6 GB | 8 (core) | 12/16 (75%, 95% CI 51-90%) | 0/16 | 208 | 0 | 0 | 601 s | error 3, loop 1 |
| qwen3.8-27b | MTPLX 2.12.3 | Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed | defaults | 4.5 | 13.3 GB | 8 (core) | 0/16 (0%, 95% CI 0-19%) | 0/16 | 484 | 0 | 0 |  | error 16 |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults |  | 11.1 GB | | 0 attempts scored — server started, warm-up refused: its first request failed: HTTP 507: Model 'Qwen3.8-27B-4bit' (15.70GB) does not fit under the dynamic memory ceiling (13.18GB). Close other apps to free RAM (macOS and other apps leave oMLX 13.18GB of the 21.00GB static cap, and oMLX itself uses 208.84MB of that), raise memory_guard_tier (safe → balanced → aggressive), or use a smaller model. | | | | | | |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b | context 16384 |  | 15.1 GB | | 0 attempts scored — engine exited before answering: exited with code -15: Error: Model loading was stopped due to insufficient system resources. Continuing to load the model would likely overload your system and cause it to freeze. If you think this is incorrect, you can adjust the model loading guardrails in settings. | | | | | | |
| qwen3.8-27b | Ollama 0.40.16 | qwen3.8:27b (q4_K_M, 17.74 GB) | |  | | | not run: no scored result (stopped at a 64K and a 16K window) | | | | | | Memory pressure turned critical twice on this 24 GB Mac, and the watchdog stopped it: during loading at a 64K window (Ollama's own estimate: 20.6 GB of GPU memory, 18.2 GB free), and in the first attempt at a 16K window (llama-server reached 21 GB against its 17.3 GB estimate; Ollama disables mmap here). Its native default window here, 4K ("default_num_ctx=4096"), was not run: it would cut an agent's conversation. |


## Pairwise differences

Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.

| Model | A | B | Solved, A vs B | p | Failed checks after a repeated-format exit, A vs B | p |
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

Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over tasks solved. Summed proxy request duration: the requests' durations added up; with retries it can exceed the elapsed time. Decode and prefill during the run: from each engine's own log, medians over the requests that wrote at least 64 tokens and that read at least 256 new tokens. Elapsed totals describe each listed configuration, and include the time spent on failed attempts.

| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | Tokens written | Summed proxy request duration (can exceed elapsed attempt time) | Decode during the run (median) | Prefill during the run (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | core | defaults | 16 | 16 | 45.8 min | 172 s | 171 | 92734 | 45.7 min | 40 tok/s (170 requests) | 267 tok/s (70 requests) |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | core | defaults | 16 | 14 | 65.8 min | 282 s | 147 | 85648 | 65.7 min | 25 tok/s (145 requests) | 287 tok/s (65 requests) |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | core | defaults | 16 | 16 | 66.4 min | 249 s | 159 | 92140 | 66.3 min | 27 tok/s (155 requests) | 287 tok/s (70 requests) |
| qwen3.8-27b | mlx-lm 0.32.2 | core | defaults | 16 | 9 | 98.0 min | 653 s | 128 | 44082 | 97.9 min | not logged | not logged |
| qwen3.8-27b | mlx-lm 0.32.2 | core | max tokens 16384 (times not comparable) | 16 | 9 | 162.7 min | 1085 s | 109 | 76960 | 162.9 min | not logged | not logged |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | core | context 16384, guardrails off (times not comparable) | 16 | 2 | 85.0 min | 2549 s | 171 | 14026 | 27.0 min | not logged | not logged |
| qwen3.8-27b | oMLX 0.8.0 | core | memory guard aggressive (times not comparable) | 16 | 12 | 276.1 min | 1380 s | 208 | not reported | 247.8 min | not logged | not logged |
| qwen3.8-27b | MTPLX 2.12.3 | core | defaults | 16 | 0 | 71.4 min |  | 484 | 138 | 1.6 min | not logged | not logged |


**Corrected on 2026-11-21.** oMLX's "Tokens written" reads "not reported", here and in the table below, but oMLX
did report usage with each reply.
- **The real figure.** Its 121 replies to the agent hold 59,883 tokens. Tokens of aborted requests and of replies
  the agent did not keep are not in that figure.
- **Why.** solvetime's proxy, which counts the tokens, took oMLX's chunked replies for streams and lost their
  counts. It is fixed in solvetime #58.

The records stay as measured.

## Each run

Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it understates an engine that maps its weights from the file. To compare memory across engines, use the speed section's memory taken from the system.

| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, solved attempts | Total time per solved task | Total time | Median tokens written, among attempts reporting usage | Requests (median) | Failures | Paged out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | qwen3.8-27b (FP4, 4.0 bits, with its MTP head) | defaults | 4.04 | 13.8 GB | 7.2 GB | 4 s | 16/16 | 119 s | 172 s | 45.8 min | 4330 | 10 | none | 0.0 GB |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 20.2 GB | 18.5 GB | 5 s | 14/16 | 162 s | 282 s | 65.8 min | 3752 | 10 | malformed output 1, wrong 1 | 0.0 GB |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | kapitsa qwen3.8-27b: qwen3.8-27b-q4 | defaults | 4.43 | 20.2 GB | 18.5 GB | 7 s | 16/16 | 154 s | 249 s | 66.4 min | 3891 | 9 | none | 0.0 GB |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | | | | | 0 attempts scored — server started, warm-up refused: its first request failed: HTTP 507: Model 'Qwen3.8-27B-4bit' (15.70GB) does not fit under the dynamic memory ceiling (13.18GB). Close other apps to free RAM (macOS and other apps leave oMLX 13.18GB of the 21.00GB static cap, and oMLX itself uses 208.84MB of that), raise memory_guard_tier (safe → balanced → aggressive), or use a smaller model. | | | | | | | |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | defaults | 4.43 | 16.1 GB | 19.4 GB | 6 s | 9/16 | 422 s | 653 s | 98.0 min | 2569 | 8 | malformed output 4, loop 3 | 0.0 GB |
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 16.1 GB | 19.5 GB | 5 s | 9/16 | 552 s | 1085 s | 162.7 min | 6034 (12/16 attempts) | 8 | error 4, timeout 3 | 0.0 GB |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b | context 16384 | | | | | 0 attempts scored — engine exited before answering: exited with code -15: Error: Model loading was stopped due to insufficient system resources. Continuing to load the model would likely overload your system and cause it to freeze. If you think this is incorrect, you can adjust the model loading guardrails in settings. | | | | | | | |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | qwen/qwen3.8-27b (lmstudio-community/Qwen3.8-27B-MLX-4bit) | context 16384, guardrails off | 4.43 | 16.1 GB | not measured | 17 s | 2/16 | 451 s | 2549 s | 85.0 min | 3940 (3/16 attempts) | 10 | error 14 | 0.1 GB |
| qwen3.8-27b | oMLX 0.8.0 | mlx-community/Qwen3.8-27B-4bit (b935172) | memory guard aggressive | 4.43 | 16.1 GB | 19.6 GB | 20 s | 12/16 | 601 s | 1380 s | 276.1 min | not reported | 11 | error 3, loop 1 | 0.0 GB |
| qwen3.8-27b | MTPLX 2.12.3 | Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed | defaults | 4.5 | 16.3 GB | 19.1 GB | 10 s | 0/16 |  | nothing solved | 71.4 min | 138 (1/16 attempts) | 30 | error 16 | 0.0 GB |


## Speed

`solvetime speed`: thinking off, after one uncounted warm-up; prefill on a 4K-token prompt, decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what the system had available before the engine started, minus the least it had while the engine ran (free, inactive, speculative and purgeable pages), on an otherwise quiet machine.

Runs left out of a cell are named in its row's notes, and listed with the reason in speed-exclusions.json.

| Model | Engine | Settings | Bits per weight | Prefill 4K | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 | Memory taken (system) | RSS | Footprint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | raceline 0.8.1 (ours) | defaults | 4.04 | 292 (279-296) | 35.4 (33.3-38.5) | 31.7 (30.8-31.9) | 40.6 (39.8-41.0) | 40.5 (40.2-40.5) | 14.9 GB | 1.9 GB | 6.1 GB |  |
| qwen3.8-27b | kapitsa 0.2.36 (release, fe6a17a) | defaults | 4.43 | 384 (377-387) | 26.3 (24.1-27.6) | 27.6 (27.6-27.8) | 30.8 (30.4-30.9) | 30.8 (30.7-30.8) | 14.6 GB | 16.5 GB | 18.3 GB |  |
| qwen3.8-27b | kapitsa main c1c42e6 (fixes #1279, #1302) | defaults | 4.43 | 385 (379-386) | 24.9 (23.7-26.9) | 27.7 (27.3-28.2) | 30.5 (29.9-30.7) | 30.7 (30.7-30.8) | 14.7 GB | 16.5 GB | 18.3 GB |  |
| qwen3.8-27b | mlx-lm 0.32.2 | defaults | 4.43 | 289 (286-291) | 9.3 (9.2-9.4) | 9.4 (9.3-9.4) | 9.5 (9.4-9.5) | 9.5 (9.5-9.6) | 13.0 GB | 14.2 GB | 16.9 GB |  |
| qwen3.8-27b | LM Studio llmster 0.0.27-1 (MLX runtime 1.14.0) | context 16384 | | | | | | | | | | no speed result — engine did not start: exited with code -15: Error: Model loading was stopped due to insufficient system resources. Continuing to load the model would likely overload your system and cause it to freeze. If you think this is incorrect, you can adjust the model loading guardrails in settings. |
| qwen3.8-27b | oMLX 0.8.0 | defaults | | | | | | | | | | no speed result — server started, warm-up refused: its first request failed: HTTP 507: Cannot load Qwen3.8-27B-4bit: projected memory 15.90GB would exceed the dynamic memory ceiling 15.85GB (current: 208.78MB, model: 15.70GB). Close other apps to free RAM (macOS and other apps leave oMLX 15.84GB of the 21.00GB static cap, and oMLX itself uses 208.78MB of that), raise memory_guard_tier (safe → balanced → aggressive), or unload another model. |
| qwen3.8-27b | oMLX 0.8.0 | memory guard aggressive | 4.43 | 317 (1 retained of 3) | 9.9 (9.4-9.9) | 9.9 (9.9-9.9) | 9.9 (9.9-9.9) | 9.9 (9.9-9.9) | 13.0 GB | 1.1 GB | 17.8 GB | prefill runs 2, 3 left out: they overlapped 14 s of unit tests run on the same Mac; the user's #51 kept run 1 only |


- qwen3.8-27b, LM Studio llmster 0.0.27-1, mlx-llm 1.14.0 (qwen/qwen3.8-27b@4bit (lmstudio-community MLX 4-bit, 4.43 bits per weight), 16K window, guardrails off (non-default)): no speed result. First try (2026-11-19 15:48): memory pressure turned critical while the model loaded, because the engine of the agent run before it had crashed (Metal out of memory) but kept its 19 GB until stopped, and the reload overlapped its teardown. Retry from a settled state (20:44): the model loaded, but the prefill test got no token back: LM Studio's MLX runtime returns max_tokens - 1 tokens (asked for 1, it returns none; lmstudio-ai/lmstudio-bug-tracker#2602), and it ignores chat_template_kwargs.enable_thinking (only reasoning_effort "none" turns its thinking off), so the speed test's requests did not fit it. Found with qwen3.5-4b at 20:47; the speed test is fixed after the round. The 27B was removed after the retry, so it was not measured again. The user's choice for a second failure (#44): record it as not measured.
- qwen3.8-27b, MTPLX 2.12.3 (Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed (4.5 bpw, its tight fit for 24 GB)): no speed result. The speed test (speed-20261119-222857, 22:28) loaded it (cold start 9.5 s), and its warm-up requests decoded at 19.5-23.8 tok/s by its own log. The prefill request asked for 1 token and got none back; MTPLX's log warns "max_tokens leaves no room to answer". The speed test asks for 8 since solvetime #37, but the pack had been removed by then, and the user's #47 kept the MTPLX row as measured.

## Not run

- qwen3.8-27b, Ollama 0.40.16 (qwen3.8:27b (q4_K_M, 17.74 GB)): no scored result (stopped at a 64K and a 16K window). Memory pressure turned critical twice on this 24 GB Mac, and the watchdog stopped it: during loading at a 64K window (Ollama's own estimate: 20.6 GB of GPU memory, 18.2 GB free), and in the first attempt at a 16K window (llama-server reached 21 GB against its 17.3 GB estimate; Ollama disables mmap here). Its native default window here, 4K ("default_num_ctx=4096"), was not run: it would cut an agent's conversation.
