# solvetime

**How long does your Mac take to actually finish a coding task with a local model?**

solvetime measures local LLM setups the way agent users live with them. A fixed, open-source agent works on
coding tasks that have hidden checks, and solvetime records whether each task was solved and how long it took,
end to end. Tokens per second are recorded too, but they are not the headline: a build that thinks longer,
drops its tool calls or reads its weights from the SSD again and again is slow, however fast it decodes.

## First result: Qwen3.6-35B-A3B on a 24 GB Mac

On a 24 GB Mac, the builds of Qwen3.6-35B-A3B that fit in memory are about 2-bit. At 2 bits the model often
writes a tool call without its opening `<tool_call>` tag, and a server that parses strictly hands the agent
plain text instead of a command. 16 coding tasks × 2 attempts per build, on a Mac mini M6:

| Build | Bits | Tasks solved (95% CI) | Attempts lost to malformed replies | Core set, 16 attempts |
|---|---|---|---|---|
| kapitsa 0.2.36 (release) | 2.48 | 19/32 (42-74%) | 13/32 | 6.6 min (failures end early) |
| kapitsa main 7445a6c (fix [#1279](https://github.com/nace-ai/kapitsa/pull/1279)) | 2.48 | 28/32 (72-95%) | 0/32 | 13.5 min |
| llama.cpp 0.9.0, unsloth UD-IQ2_XXS | 2.42 | 23/32 (55-84%) | 5/32 | 12.7 min |
| raceline (ours), 4-bit, experts read from the SSD | 4.06 | 30/32 (80-98%) | 0/32 | 19.6 min |

What we learned:

- **A parser fix recovers most of the loss.** We reported the dropped tag with a repro
  ([kapitsa#1277](https://github.com/nace-ai/kapitsa/issues/1277)), and kapitsa merged a parser fallback the
  same day: 19/32 → 28/32 solved (p = 0.022). llama.cpp on the same weights is still affected (a related
  issue is open there: [ggml-org/llama.cpp#30118](https://github.com/ggml-org/llama.cpp/issues/30118)).
- **Synthetic speed tests point the wrong way here.** Our own engine looked faster in a 4K-prompt speed test
  (prefill 1268-1333 vs 1091 tok/s) and solved about as many tasks as the fixed kapitsa build (30/32 vs
  28/32; at this sample size the two can't be told apart). But it took 1.45x as long end to end. In agent use
  each tool output adds a few hundred new tokens that route to most of the model's 256 experts, and the ones
  raceline keeps on the SSD (29% at its default) are read again for every such chunk: its prefill during the
  runs was 420 tok/s (306 with a smaller expert cache) against kapitsa's 925. The synthetic prompt spread
  that cost over 4,000 tokens. (A solvetime defect, fixed in [#27](https://github.com/audenvenn/solvetime/pull/27),
  held memory macOS could have used to cache the experts raceline reads from the SSD, so the gap may be
  smaller than 1.45x; it wasn't remeasured. See the results page.)
- **24 GB is tight.** On our Mac, macOS and the programs we had open held about 7-10 GB (cached files
  included), and in every run before 2026-11-18 solvetime itself kept another ~3.5 GB, a defect now fixed
  ([#27](https://github.com/audenvenn/solvetime/pull/27)). An engine holding 17-19 GB on top leaves
  little room: with the dense 27B loaded, this Mac sat at normal or warning pressure from one run to the
  next, and three times it turned critical. Close other large apps before you
  measure; each run now records the memory available at its start.

The story behind it, in one page: [On a 24 GB Mac, ~2-bit Qwen3.6-35B drops its tool-call tag](https://audenvenn.github.io/2026/11/17/qwen36-35b-2bit-drops-tool-call-tag.html).

Every table, the exact builds and their hashes, the excluded runs and why, and the raw trajectories are in
[published/2026-11-qwen3.6-35b-a3b-m6-24gb](published/2026-11-qwen3.6-35b-a3b-m6-24gb/README.md).
Not run there: MTPLX (its own catalog says none of its 35B packs fit 24 GB) and mlx-lm (its 4-bit weights
exceed the memory Metal lets the GPU use).

## Second result: Qwen3.8-27B on a 24 GB Mac

A dense 27B at about 4 bits nearly fills a 24 GB Mac. We ran the 8 core tasks twice in each configuration below,
on a Mac mini M6. The rows differ in engine, model build, settings and the memory free at the start, so the table
compares these setups as they ran; it doesn't say how much any one factor contributed.

| Engine and build | Settings | Solved | Total time | Time per solved task |
|---|---|---|---|---|
| raceline 0.8.1 (ours) | defaults (32K window) | 16/16 | 45.8 min | 172 s |
| kapitsa main c1c42e6 (PRs #1279, #1302; in 0.2.37) | defaults (16K window) | 16/16 | 66.4 min | 249 s |
| kapitsa 0.2.36 | defaults (16K window) | 14/16 | 65.8 min | 282 s |
| oMLX 0.8.0 | memory guard "aggressive" (the default guard declined to load the model) | 12/16 | 276.1 min | 1,380 s |
| mlx-lm 0.32.2 | defaults (replies capped at 512 tokens) | 9/16 | 98.0 min | 653 s |
| mlx-lm 0.32.2 | replies up to 16,384 tokens | 9/16 | 162.7 min | 1,085 s |
| LM Studio (llmster 0.0.27-1) | 4-bit build pinned, 16K window, guardrails off (the default guardrail declined to load the model) | 2/16 | 85.0 min | 2,549 s |
| MTPLX 2.12.3 | defaults (its memory plan chose a 4K window); other apps held 5.4 GB at its start, and Chrome, Xcode and Pages were opened mid-run, after its refusals began | 0/16 | 71.4 min | none solved |

Total time covers all 16 attempts, failed ones included, and the last column divides it by the tasks solved. A
task counts as solved when its hidden check passes on the final workspace, even if the agent then ended on an
error.

Ollama 0.40.16 has no scored result: memory pressure stopped its 64K run during loading and its 16K run during
the first attempt. Its native 4K window was not tried.

What we learned:

- **Defaults shaped several rows.** mlx_lm.server caps a reply at 512 tokens unless the request asks for more.
  In mlx-lm's default row, 43 of 128 replies reached the cap, 41 of them without a tool call, and all seven failed
  attempts ended after three cut-off replies in a row. Allowing 16,384 tokens also gave 9/16: two failures became
  passes, and two passes became failures, one by running out of time and one after a crash. Under the recorded
  conditions, LM Studio's guardrail and oMLX's memory guard declined to load the model at their defaults. MTPLX's
  own memory plan set a 4,096-token window, and its memory guard refused 483 of its 484 requests.
- **Going past a default has a cost.** With LM Studio's guardrail off, its MLX engine ran out of GPU memory during
  attempt 3, then kept its 19 GB while attempts 4–16 spent about 54 minutes on requests answered with "No models
  loaded". mlx-lm's server, allowed longer replies, ran out of GPU memory during attempt 12, which still passed,
  and answered 404 to the last four. oMLX with its aggressive guard stayed up through 64 aborted requests, 8
  rejected prompts and 4 model evictions followed by reloads, and solved 12/16 in 276.1 minutes.
- **Two bugs we reported in kapitsa were fixed within a day** (#1279, the parser; #1302, the memory fit). On the
  same pack, 0.2.36 solved 14/16 and main at c1c42e6 16/16. One 0.2.36 attempt ended after three tool calls cut
  off at the length limit, and the other failed its check. This comparison doesn't show which of main's changes,
  if any, made the difference.
- **raceline (ours) and kapitsa main both solved 16/16.** Their 16 attempts took 45.8 and 66.4 minutes, in one run
  each, on different builds and windows.

Every row's conditions, including the memory available at each start, the speed tests, the statistical tests and
the user's decisions on non-default settings, are on the [results page](published/2026-11-qwen3.8-27b-m6-24gb/README.md).

## Experiment: an agent LoRA on Qwen3.8-27B

harness-hub's pi adapter is a LoRA for Qwen3.8-27B, trained to work behind the pi coding agent. On the same 24 GB
Mac mini, we ran the same 4-bit GGUF in llama.cpp without and with it, under solvetime's agent rather than pi. Each
arm ran the 8 core and 8 extended tasks twice.

| Arm | Solved | Steps per attempt (mean) | Tokens written per attempt (mean) | Total time |
|---|---|---|---|---|
| Without the adapter | 28/32 | 6.75 | 1,028 | 69.4 min |
| With the adapter | 28/32 | 5.09 | 743 | 53.6 min |

Both arms failed the same two tasks. With the adapter the agent took fewer steps on 13 of the 16 tasks (two-sided
sign test, p = 0.021 unadjusted, 0.064 after adjusting for the three measures compared). In total across all
attempts it took 23% less time, although llama.cpp, which applies the adapter at run time, processed prompts 30%
slower with it. This is one model build and one engine, with 32 attempts per arm, so the result is exploratory. The
[experiment page](published/2026-11-experiment-pi-lora-qwen3.8-27b-m6-24gb/README.md) has the results per task, the
conditions, the first try we discarded, and the limits.

## Third result: Inkling 2 Mini on a 24 GB Mac

Inkling 2 Mini (Thinking Machines) is a 26B mixture of experts whose middle layers run three times per token; its
card asks for a 32 GB Mac. Two configurations completed scored runs here, each on a build of about 3.6 to 3.8 bits
per weight. We ran the 8 core tasks twice in each, on a Mac mini M6.

| Engine and build | Settings | Attempts solved | Total time | Total attempt time per solve |
|---|---|---|---|---|
| llama.cpp b12600, InterpreticAI IQ3_XXS (3.6 bits per weight) | 32K window, RAM prompt cache off | 16/16 | 51.1 min | 192 s |
| oMLX 0.8.1, mlx-community oQ3e-mtp (3.8 bits per weight) | memory guard "aggressive" (at its default guard, two launches reached no scored attempt); its DSpark drafter on, by default | 13/16 | 82.4 min | 380 s |

What we learned:

- **Support decided most of it.**
  - **llama.cpp.** At b12600, its first release with the model, its Mac backend does not support multiplying
    NVFP4 weights, and the unsloth GGUFs we checked keep the model's NVFP4 experts. Measured with llama-bench,
    unsloth's build processed prompts 11x slower and generated 2.2x slower than the scored file: an NVFP4-free
    3-bit build of IQ3_XXS from another uploader.
  - **oMLX.** Its support was released the morning of the round (0.8.1).
  - **The rest:**
    - Ollama's only builds are the 20 GB checkpoint, which we did not try;
    - mlx-lm's support is a draft pull request;
    - in LM Studio, kapitsa, MTPLX and raceline (ours) we found none.
- **oMLX's default guard stopped it twice.** Once it refused to load the model, with its ceiling at 11.1 GiB. The
  second time it loaded the model, then refused the first prompt. With its aggressive guard it ran, with 80 request
  aborts under memory pressure in its log. Its three failed attempts ended in API errors.
- **The model kept thinking at its lowest reasoning effort.** In the speed tests, every 512-token reply was all
  reasoning.

The [results page](published/2026-11-inkling-2-mini-m6-24gb/README.md) has each run's conditions, the speed tests
and the engines not run.

## What it measures

- **Solved**: a hidden check decides; the agent never sees it, and files it must not edit (such as the tests)
  are verified unchanged.
- **Time to solution**: wall clock from the agent's first request to its exit: prompt processing, thinking,
  tool calls and all.
- **Tokens and speed**: tokens written, requests made, and the decode and prefill speeds each engine logs
  during the runs.
- **Failures by kind**: wrong result, timeout, step limit, loop (the same reply, or the same command with the
  same output, again and again), malformed output (three replies in a row without a usable tool call),
  engine crash.
- **Memory**: memory taken from the system while the engine runs, and whether the Mac paged out.

## Fairness

- The agent is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6 (MIT) with its default
  prompts and one bash tool. It is not ours and favours no engine.
- Each engine runs the build it recommends or picks for the Mac at hand, with its default settings
  (llama.cpp, which picks none, gets the GGUF kapitsa picks, as a control); the build, its bits per weight
  and any setting beyond the defaults appear in every table. "Does not fit" is a
  result too, and the tables say how it was determined.
- Every engine gets the same tasks, prompts, sampling (the model card's) and limits: 50 steps and 15
  minutes per attempt.
- Each task is attempted twice, and the tables show 95% intervals and exact tests rather than a best run.
- Every attempt's trajectory and the raw numbers are published next to the tables.
- Disclosure: the author also maintains [raceline](https://github.com/audenvenn/raceline), one of the
  engines measured. It is labelled in every table and reported like any other.
- The code and the results pages are written with Claude's assistance, and Codex reviews them independently
  before they are published.

## How it is measured

- **Sandboxed agent**: every command the agent runs may write only inside its attempt's directory, has no
  network, and can't read your home directory (apart from the task's own Python). The hidden checks run in
  the same sandbox.
- **Warm before measuring**: one uncounted request after the server starts; cold start is reported
  separately.
- **Round-robin**: every task once, then every task again, so no engine gets an unrealistically warm cache
  from repeating the same task back to back. Reusing the agent's system prompt across tasks is real-world
  behaviour and allowed.
- **Speed test** (`solvetime speed`): thinking off, a 4K-token prompt for prefill and up to 512 tokens of
  decode on two prompts (writing new code, continuing code), three runs each. It measures memory taken from
  the system on an otherwise quiet machine.

## Run it on your Mac

Needs macOS on Apple silicon and Python 3.10 or later. Install the engines you want to test:

| Engine | Install |
|---|---|
| raceline | `pip install git+https://github.com/audenvenn/raceline`, then `raceline pull qwen3.6-35b-a3b` (M5/M6 only) |
| kapitsa | `brew install nace-ai/tap/kapitsa`; for the #1279 fix before its next release, `brew install --HEAD nace-ai/tap/kapitsa` |
| llama.cpp | `brew install llama.cpp`, plus the GGUF named in the results, passed as `SOLVETIME_GGUF=/path/to.gguf` |
| mtplx | `pip install mtplx` |
| mlx-lm | `pip install mlx-lm` |
| ollama | `brew install ollama`, then `ollama pull qwen3.8:27b` (solvetime starts its own server) |
| lmstudio | `curl -fsSL https://lmstudio.ai/install.sh \| bash` (the headless daemon with `lms`), then `lms get qwen/qwen3.8-27b` |
| omlx | `brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx`, plus an MLX model folder in `SOLVETIME_OMLX_MODEL` |

On a 24 GB Mac none of the last three can load the 35B; they are for the 27B and smaller models. Ollama picks a 4096-token context there by default, which would cut an agent's conversation without saying so, so solvetime gives it 65536, as it does llama.cpp, and the tables show the setting.

```sh
pip install git+https://github.com/audenvenn/solvetime
solvetime doctor                               # which engines it finds; starts none
solvetime run --engine kapitsa --model qwen3.6-35b-a3b --set core --attempts 2
solvetime speed --engine kapitsa --model qwen3.6-35b-a3b
solvetime report results/                      # add --speed for the speed tables
```

A core × 2 run takes 13-20 minutes per engine on an M6, plus the first model download (10-20 GB).
`--set extended|all`, `--tasks a,b` and `--thinking no-thinking` select other runs.

The models are `qwen3.6-35b-a3b`, `qwen3.8-27b` and `qwen3.5-9b`. [`solvetime/models.py`](solvetime/models.py)
holds each one's checkpoint, its card's sampling and every engine's build of it; `solvetime doctor --model
<model>` shows the builds this Mac has. A variable points an engine at another build (`SOLVETIME_OLLAMA_TAG`,
`SOLVETIME_MLX_MODEL`, ...; listed there), and a new model is one entry in that file.

Results from M1 to M5 Macs, and from other memory sizes, are the point of the project. To share yours:

```sh
solvetime submit results/          # one .tar.gz; home paths, host names, emails and your name replaced
solvetime validate <bundle>        # replays every attempt's commands in the sandbox and re-scores it
```

Then open a [Results from your Mac](https://github.com/audenvenn/solvetime/issues/new/choose) issue with the bundle. Every attempt published
here was replayed and re-scored the same way, and all 128 match what was recorded:
`solvetime validate published/2026-11-qwen3.6-35b-a3b-m6-24gb` reruns that check in about 90 seconds.

## The tasks

16 small coding tasks in two sets. The core 8 are single-file work: a bug that failing tests point at, a
rename across files, a function from its docstring, a new command-line option, config defaults, docstrings
without code changes, `os.path` to `pathlib`, a binary-search bug. The extended 8 are closer to a working day:
several files, a 600-line file, tests that must catch broken versions of the code, a new module from a
description, a signature change across files.

## License

Apache-2.0.
