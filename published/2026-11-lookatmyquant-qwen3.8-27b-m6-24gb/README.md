# LookAtMyQuant's 2.1-bit Qwen3.8-27B on a 24 GB Mac mini (M6), November 2026

Machine: Apple M6, 24 GB, macOS 27.2.
Runs from 2026-11-19 11:28:04 to 2026-11-22 14:07:55. Agent: mini-swe-agent 2.4.6; limits: 50 steps, and a 900 s budget the agent checks before each query (a query in flight and its retries can run past it). Sampling sent with every request: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0, thinking on. Tables made with solvetime 0.1.0.dev0 (4490ae8-dirty).

## About these runs

- **What the tables compare.** Qwen3.8-27B through the same server, mlx-lm 0.32.2. Both rows ran the 8 core tasks
  twice, with the same task order, agent settings and sampling overrides:
  - `--max-tokens 16384` and `--prompt-cache-size 1`;
  - thinking explicitly on;
  - 50 steps and 900 s per attempt;
  - solvetime's Qwen sampling: temperature 0.6, top_p 0.95, top_k 20.
    - **Where 0.6 comes from.** It is the Qwen3.5 and 3.6 cards' setting for precise coding tasks.
    - **What Qwen3.8's card says.** It recommends temperature 1.0 in thinking mode, as the build's own
      `generation_config.json` does.
    - **Later runs.** After this round, solvetime runs Qwen3.8 at its card's 1.0, so later 27B results with
      thinking are not directly comparable with this page's.

  The rows:
  - **Round 4 (2026-11-22):** [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant)'s 7.2 GB build, 2.1
    bits per weight. It runs on LookAtMyQuant's Qwen loader and fused kernels, inside mlx-lm's HTTP server and tool
    parser (solvetime #61).
  - **Round 2 (2026-11-19):** mlx-community's 4-bit build, 4.43 bits per weight, through mlx-lm's own loader. It is
    the same run as on the [27B page](../2026-11-qwen3.8-27b-m6-24gb/README.md).

  They also differ in date, harness version and the conditions below. So the tables compare these setups as they
  ran.
- **The attempts.**
  - **The first 12 scheduled attempts:** both builds passed the same nine.
  - **Round 2's failure.** Its server's generation thread failed during attempt 12 (cli-flag 2), after a passing edit:
    "[METAL] Command buffer execution failed: Insufficient Memory". Its final four attempts received no model reply
    and counted as failures.
  - **Round 4** passed two of those four (config-defaults 2, docstrings 2).
  - **Time-outs.** All five round 4 failures, and three of round 2's seven, ended at the agent's time limit.
    - **In round 4:** 9 of its 16 attempts reached the limit, 4 of them with a passing edit already in place.
    - **What can't be known:** whether more time would have solved the others. bisect 2's final check also timed out.
  - **What can't be inferred:** why round 4 avoided the out-of-memory failure, or what round 2 would have solved
    with a healthy server.
- **The statistics are nominal.** The Fisher test (p = 0.716) and the per-row intervals treat the 32 attempts as
  independent. But the same eight tasks were repeated, and one server failure accounts for four of round 2's failures.
- **Speed.**
  - **The speed tests** (thinking off). Both clients requested 512 decode tokens and got them. Round 2's speed server
    used mlx-lm's default reply cap of 512, at harness 984f250; round 4's used 16,384.

    | | Prefill, ~4,000-token prompt | Decode |
    |---|---|---|
    | 2.1-bit build | 166–168 tok/s | 5.39–5.48 tok/s |
    | 4-bit build | 286–291 tok/s | 9.24–9.55 tok/s |

    The 2.1-bit build gave about 58% of the other's throughput on this M6.
  - **Five matched attempts both agents submitted normally** (inventory 1 and 2, rename 1, config-defaults 1, bisect
    1):
    - **Tokens:** the 2.1-bit build generated 32% fewer completion tokens (12,460 against 18,252).
    - **Time:** it took 20% more elapsed time (2,983 against 2,493 s).
    - **Reasoning:** 46% against 47% of the characters, counting reasoning, assistant content and tool-call arguments.
  - **The prompt cache.** After each first reply, every reply reported 319 cached prompt tokens. So the server
    processed the rest of each prompt again: over round 4's 125 replies, a median of 2,769 new tokens and up to
    10,315. The longest full prompt was 10,634 tokens.
  - **Total agent-loop time** includes failed attempts.
    - **The first 12 slots:** 164.2 min in round 4, against 162.7 in round 2.
    - **The last four:** 56.3 min in round 4, while in round 2 they failed at once.
    - **What the timer covers:** it stops when the agent stops, before the final check.
  - **The Time table's label.** The Time table marks both rows "times not comparable", as it does any row not at the
    engine's defaults. These two rows share their settings, so their times compare with each other.
- **Memory.**
  - **The agent run.** The 7.2 GB build's process footprint peaked at 18.6 GB, against 19.5 GB for round 2's 15 GB of
    weights.
  - **A separate test** ([memory-check.md](memory-check.md)) gave these readings:
    - **Loaded:** MLX reported 7.57 GB active.
    - **A 10,634-token prefill** in the server's 2,048-token chunks peaked at 15.93 GB active, whether or not MLX's
      cache was cleared after each chunk, as the server does.
    - **So a long prefill** needs about 8 GB above the loaded model.
  - **What wasn't measured:** the server's own allocator breakdown.
  - **Pressure.** During the scored run the watchdog recorded 66 warning snapshots and no critical one.
    - **The engine's reading.** macOS `top` displayed the engine at 17G in 62 of them.
    - **The bias.** Those snapshots are taken at pressure warnings, so they don't cover ordinary moments.
- **Memory before launch.** The recorded estimate of available memory was 18.6 GB in round 4 and 14.1 GB in round 2.
  - **Round 2's is biased low.** Its older harness sampled before macOS released the calibration's memory (the 27B
    page's correction of 2026-11-21). Its speed test's "Memory taken (system)" of 13.0 GB carries the same baseline
    bias.
  - **So** the difference shows neither how much more room round 4 had, nor why round 2 ran out.
- **LookAtMyQuant's own results.** For this build, against the original model, its card reports:

  | Benchmark | This build | Original model |
  |---|---|---|
  | Math, 540 exact-answer problems | 74.3% | 74.3% (401/540 each) |
  | HumanEval | 73.8% | 77.4% |
  | MMLU | 78.8% | 83.5% |
  | GSM8K | 94.5% | 95.5% |

  The card says the build was calibrated and fine-tuned on English prose, chat, code and math. So this comparison
  takes in its checkpoint, its runtime's kernels and the serving conditions together.
- **Not run.**
  - **The 27B's DSpine drafter.** The README links one (`Qwen3.8-27B-LookAtMyQuant-7.2GB-drafter`), which it says
    roughly doubles decode speed on math and code. We could not access it on 2026-11-22: its repo answered 401.
    Decoding was serial throughout.
  - **Other engines.** None of the engines we checked reads LookAtMyQuant's format (`opgraph-packed-1`).
    LookAtMyQuant's own runtime has a CLI and a Python API, but no HTTP server and no tool calls, which solvetime's
    agent needs.
  - **LookAtMyQuant's Inkling 2 Mini builds:** in the table of builds not run.
- **Builds.**
  - **shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB at 6083eff.**
    - **Size.** `weights.bin` is 7,188,022,016 bytes, SHA-256
      `a3b2ca7174a8897a2bb2495e2b0bae216026b99e05724eddb7e21ca63c8545b3`.
    - **Bits per weight.** Its `artifact.json` lists every tensor's bytes, with no vision tower. That is 2.1 bits per
      weight over the 27.32B text parameters.
    - **License.** Its repo lists no separate license for the weights.
    - **Checked.** All seven files matched the repo's hashes, recorded before the download.
  - **LookAtMyQuant 0.3.0 at 51069235** (MIT), installed from git into a venv of its own.
    - **Pinned beside it:** round 2's mlx 0.32.4, mlx-lm 0.32.2 and transformers 5.22.2.
    - **Settings:** its loader's defaults: verification on, and its fused kernels.
  - **mlx-community/Qwen3.8-27B-4bit at b935172,** as in round 2.
- **Machine and conditions.** Mac mini (2026), Apple M6, 24 GB, macOS 27.2.
  - **The CPU.** System-wide CPU use was 1,173%, 218% and 107% of 1,200% at the start of the smoke test, the scored
    run and the speed test. Another project's CPU-bound search accounted for most of the first.
  - **Other sessions** were asked to keep memory low while the GPU was in use.
- **Harness.** solvetime f643e93 for round 4, with #61 letting mlx-lm's server load a LookAtMyQuant build. Round 2's
  run was at 9f793fc.
  - **What changed in between:** the speed test counts reasoning (#52), memory is read after the calibration's
    release (#53), thinking switches go as template variables (#56), and the proxy accounts chunked replies (#58).
  - **What stayed the same:** the two runs' recorded sampling overrides are identical.

## Agent tasks, each engine build pooled over its task sets

Solved: the attempt's final workspace passed the task's check, work done before an error included, with a 95% Wilson interval. "Failed checks after a repeated-format exit": attempts the agent stopped after three replies in a row it could not use, whose check then failed. "Requests": every request the agent sent, the ones that failed included. "Agent tool-call error notices" and "Agent output-limit notices": the agent's own feedback messages, for a reply without a usable tool call and for a reply cut off at the output limit. "Estimated available before launch": the memory macOS could hand out just before the engine started (vm_stat: free, inactive, speculative and purgeable pages), measured after the harness's GPU calibration; runs made before 2026-11-18 did not record it. A run whose engine never served an attempt is listed with how it stopped, and no score.

| Model | Engine | Build | Settings | Bits per weight | Estimated available before launch | Tasks | Solved | Failed checks after a repeated-format exit | Requests | Agent tool-call error notices | Agent output-limit notices | Median time, solved attempts | Other failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 14.1 GB | 8 (core) | 9/16 (56%, 95% CI 33-77%) | 0/16 | 109 | 0 | 0 | 552 s | error 4, timeout 3 |
| qwen3.8-27b | mlx-lm 0.32.2 | shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB (6083eff) | LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384 | 2.1 | 18.6 GB | 8 (core) | 11/16 (69%, 95% CI 44-86%) | 0/16 | 125 | 0 | 0 | 726 s | timeout 5 |
| inkling-2-mini | mlx-lm 0.32.2, with LookAtMyQuant 0.3.0 (51069235) | shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB (3579ec9, 12.76 GB) | |  | | | not run: no working tool-call route found | | | | | | LookAtMyQuant supplies an mlx-lm model adapter for it, but its cache lacks the nbytes that mlx-lm 0.32.2's server reads when it keeps a prompt cache. mlx-lm 0.32.2 also has no parser for Inkling's tool calls; its Inkling 2 support is a draft pull request. We did not establish a working tool-call route in this setup. By the maintainer's decision, it waits for mlx-lm's Inkling support or a LookAtMyQuant server. |


## Pairwise differences

Two-sided Fisher exact tests. With a few dozen attempts per engine, only large differences reach p < 0.05; a larger p means the counts cannot tell the two apart, not that they are equal.

| Model | A | B | Solved, A vs B | p | Failed checks after a repeated-format exit, A vs B | p |
|---|---|---|---|---|---|---|
| qwen3.8-27b | mlx-lm 0.32.2 (max tokens 16384) | mlx-lm 0.32.2 (LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384) | 9/16 vs 11/16 | 0.716 | 0/16 vs 0/16 | 1.000 |


## Time

Total time: the attempts end to end (agent, sandbox and engine). Per solved task: total time over tasks solved. Summed proxy request duration: the requests' durations added up; with retries it can exceed the elapsed time. Decode and prefill during the run: from each engine's own log, medians over the requests that wrote at least 64 tokens and that read at least 256 new tokens; an engine whose log has no timing of its own for these phases says so. Elapsed totals describe each listed configuration, and include the time spent on failed attempts.

| Model | Engine | Task set | Settings | Attempts | Solved | Total time | Per solved task | Requests | Tokens written | Summed proxy request duration (can exceed elapsed attempt time) | Decode during the run (median) | Prefill during the run (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | mlx-lm 0.32.2 | core | max tokens 16384 (times not comparable) | 16 | 9 | 162.7 min | 1085 s | 109 | 76960 (12/16 attempts) | 162.9 min | no comparable timing logged | no comparable timing logged |
| qwen3.8-27b | mlx-lm 0.32.2 | core | LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384 (times not comparable) | 16 | 11 | 220.6 min | 1203 s | 125 | 58033 | 216.5 min | no comparable timing logged | no comparable timing logged |


## Each run

Peak footprint is the engine's phys_footprint, which leaves out clean memory-mapped pages: it understates an engine that maps its weights from the file. To compare memory across engines, use the speed section's memory taken from the system.

| Model | Engine | Build | Settings | Bits per weight | Size | Peak footprint | Cold start | Solved | Median time, solved attempts | Total time per solved task | Total time | Median tokens written, among attempts reporting usage | Requests (median) | Failures | Paged out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | mlx-lm 0.32.2 | mlx-community/Qwen3.8-27B-4bit (b935172) | max tokens 16384 | 4.43 | 16.1 GB | 19.5 GB | 5 s | 9/16 | 552 s | 1085 s | 162.7 min | 6034 (12/16 attempts) | 8 | error 4, timeout 3 | 0.0 GB |
| qwen3.8-27b | mlx-lm 0.32.2 | shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB (6083eff) | LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384 | 2.1 | 7.2 GB | 18.6 GB | 10 s | 11/16 | 726 s | 1203 s | 220.6 min | 3598 | 8 | timeout 5 | 0.0 GB |


## Speed

`solvetime speed`: thinking off, by each model's own switch, after one uncounted warm-up; prefill on a prompt of Python standard-library source (its size in the model's own tokens is in the cell), decode of up to 512 tokens on two prompts, three runs each (median, range). Memory taken (system): what the system had available before the engine started, minus the least it had while the decode requests ran, which is when it is sampled (free, inactive, speculative and purgeable pages); other processes can change it.

| Model | Engine | Settings | Bits per weight | Prefill | Write T 0.6 | Write T 0 | Continue T 0.6 | Continue T 0 | Memory taken (system) | RSS | Footprint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | mlx-lm 0.32.2 | defaults | 4.43 | 289 (286-291), 4.0K tokens | 9.3 (9.2-9.4) | 9.4 (9.3-9.4) | 9.5 (9.4-9.5) | 9.5 (9.5-9.6) | 13.0 GB | 14.2 GB | 16.9 GB |  |
| qwen3.8-27b | mlx-lm 0.32.2 | LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384 | 2.1 | 168 (166-168), 4.0K tokens | 5.4 (5.4-5.4) | 5.4 (5.4-5.4) | 5.5 (5.4-5.5) | 5.5 (5.5-5.5) | 11.8 GB | 7.5 GB | 12.9 GB |  |


## Not run

- inkling-2-mini, mlx-lm 0.32.2, with LookAtMyQuant 0.3.0 (51069235) (shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB (3579ec9, 12.76 GB)): no working tool-call route found. LookAtMyQuant supplies an mlx-lm model adapter for it, but its cache lacks the nbytes that mlx-lm 0.32.2's server reads when it keeps a prompt cache. mlx-lm 0.32.2 also has no parser for Inkling's tool calls; its Inkling 2 support is a draft pull request. We did not establish a working tool-call route in this setup. By the maintainer's decision, it waits for mlx-lm's Inkling support or a LookAtMyQuant server.
