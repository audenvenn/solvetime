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
  that cost over 4,000 tokens.
- **24 GB is tight.** With ordinary apps open (about 5-6 GB), an engine holding 17-19 GB leaves little room:
  one overnight run of a dense 27B pushed memory pressure to critical. Close other large apps before you
  measure.

Every table, the exact builds and their hashes, the excluded runs and why, and the raw trajectories are in
[published/2026-11-qwen3.6-35b-a3b-m6-24gb](published/2026-11-qwen3.6-35b-a3b-m6-24gb/README.md).
Not run there: MTPLX (its own catalog says none of its 35B packs fit 24 GB) and mlx-lm (its 4-bit weights
exceed the memory Metal lets the GPU use).

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

```sh
pip install git+https://github.com/audenvenn/solvetime
solvetime doctor                               # which engines it finds; starts none
solvetime run --engine kapitsa --model qwen3.6-35b-a3b --set core --attempts 2
solvetime speed --engine kapitsa --model qwen3.6-35b-a3b
solvetime report results/                      # add --speed for the speed tables
```

A core × 2 run takes 13-20 minutes per engine on an M6, plus the first model download (10-20 GB).
`--set extended|all`, `--tasks a,b` and `--thinking no-thinking` select other runs. A command to bundle a
run for submission, and a template for results from other Macs, are coming this week; results from M1 to M5
Macs, and from other memory sizes, are the point of the project.

## The tasks

16 small coding tasks in two sets. The core 8 are single-file work: a bug that failing tests point at, a
rename across files, a function from its docstring, a new command-line option, config defaults, docstrings
without code changes, `os.path` to `pathlib`, a binary-search bug. The extended 8 are closer to a working day:
several files, a 600-line file, tests that must catch broken versions of the code, a new module from a
description, a signature change across files.

## License

Apache-2.0.
