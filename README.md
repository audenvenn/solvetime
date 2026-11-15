# solvetime

**How long does your Mac take to actually finish a coding task with a local model?**

solvetime measures local LLM setups the way agent users live with them. A fixed, open-source agent works on
coding tasks that have hidden checks, and solvetime records whether each task was solved and how long it took,
end to end. Tokens per second are recorded too, but they are not the headline: a build that thinks twice as
long, or loops until the timeout, is slow however fast it decodes.

> Status: in development. First results (Qwen3.6-35B-A3B on a Mac mini M6, 24 GB) are being measured.

## What it measures

- **Solved**: a hidden check decides; the agent never sees it, and files it must not edit (such as the tests)
  are verified unchanged.
- **Time to solution**: wall clock from the agent's first request to its exit: prompt processing, thinking,
  tool calls and all.
- **Tokens and speed**: tokens written, requests made, prefill and decode speed, from a metering proxy that
  sits between the agent and the engine.
- **Failures by kind**: wrong result, timeout, loop (the same request or output again and again), malformed
  output, engine crash.
- **Peak memory** of the engine process.

## What it compares

Any local server that speaks the OpenAI or Anthropic API: kapitsa, MTPLX, oMLX, LM Studio, Ollama, llama.cpp,
mlx-lm and others. Each engine runs its own recommended ~4-bit build of the model, because that is what its
users get; the build's name and size are part of every result. "Does not fit in memory" is a result too.

## Fairness

- The agent is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT), pinned to one version. It
  is not ours and favours no engine.
- Every engine gets the same tasks, prompts, sampling settings (the model card's), thinking setting and timeout.
- Each task is attempted several times; tables show the spread, not a best run.
- Every attempt's trajectory and raw numbers are published next to the tables.
- Disclosure: the author also maintains [raceline](https://github.com/audenvenn/raceline), one of the engines
  measured. It is labelled in every table, and its results are reported like any other's.

## How it is measured

- **Each engine as its users get it**: the build the engine recommends or picks for the Mac at hand, with
  its default settings. No engine is tuned beyond what its documentation tells users to do.
- **The agent is sandboxed**: its shell commands may write only inside the attempt's own directory and
  have no network, so it can neither damage the machine nor look answers up.
- **Warm before measuring**: one uncounted request after the server starts, so loading weights and
  compiling kernels don't land in the first task. Cold start is reported separately.
- **Attempts run round-robin** (every task once, then every task again), so no engine gets an
  unrealistically warm cache from repeating the same task back to back. Reusing the agent's system
  prompt across tasks is real-world behaviour and allowed.
- **Speed comes from a separate test**, a 4K-token prompt and a 512-token coding answer. The proxy
  passes the agent's requests through unchanged instead of converting them to streaming, because that
  would change how engines answer.
- **Memory**: the engine's peak physical footprint, plus whether the Mac swapped during the run. The
  footprint leaves out memory-mapped weight files, so the build's size is shown next to it.
- **Same rules for all**: tasks, prompts, the model card's sampling settings, thinking on or off, a
  50-step limit and a 15-minute wall-time limit per attempt. Which settings each engine actually
  honours is checked and recorded.

## Results

Coming: Qwen3.6-35B-A3B and Qwen3.8-27B on a Mac mini M6 (24 GB), across the engines above.

## Run it on your Mac

Coming with the first release. Results from other Macs (M1 to M6, any memory size) are the point of the
project: a run takes about an hour per engine and model, and a pull request with the results file adds your
Mac to the tables.

## The tasks

16 small coding tasks in two sets. The core 8 are single-file work: a bug that failing tests point at, a
rename across files, a function from its docstring, a new command-line option, config defaults, docstrings
without code changes, `os.path` to `pathlib`, a binary-search bug. The extended 8 are closer to a working day:
several files, a 600-line file, tests that must catch broken versions of the code, a new module from a
description, a signature change across files.
