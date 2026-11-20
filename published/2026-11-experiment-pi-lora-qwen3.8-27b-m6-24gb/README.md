# Experiment: an agent LoRA on Qwen3.8-27B, 24 GB Mac mini (M6), November 2026

Does an adapter trained for one coding agent change how another agent works with the model? We ran the same 4-bit
Qwen3.8-27B in llama.cpp without and with harness-hub's pi LoRA, under solvetime's agent, on the same tasks.

Machine: Apple M6, 24 GB, macOS 27.2. Runs from 2026-11-20 11:45:57 to 13:53:05. Agent: mini-swe-agent 2.4.6;
limits: 50 steps, and a 900 s budget the agent checks before each query. Sampling sent with every request:
temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, thinking off.

## Result

Each arm ran the 8 core and the 8 extended tasks twice: 32 attempts per arm.

| Arm | Solved | Steps per attempt, mean (median) | Tokens written per attempt, mean (median) | Time per attempt, mean (median) | Total time |
|---|---|---|---|---|---|
| Without the adapter | 28/32 | 6.75 (7) | 1,028 (916) | 130 s (114 s) | 69.4 min |
| With the adapter | 28/32 | 5.09 (5) | 743 (649) | 100 s (90 s) | 53.6 min |

A step is one request from the agent to the model. Total time covers all 32 attempts, failed ones included. A task
counts as solved when its hidden check passes on the final workspace. All 64 attempts ended with the agent
submitting; none ran out of steps or time.

- **The same tasks were solved.** In both arms, 14 tasks passed both attempts and the same two (slugify and
  signature) failed both.
- **Fewer steps with the adapter.** It took fewer steps on 13 of the 16 tasks and more on 3. A two-sided exact
  sign test over tasks gives p = 0.021, or 0.064 after a Bonferroni adjustment across steps, tokens and time. It
  wrote fewer tokens and took less time on 12 of the 16 tasks: p = 0.077 each, 0.23 adjusted.
- **Less time despite a slower engine.** llama.cpp applies the adapter at run time, which made prompt processing
  slower (below). The adapter arm still took 23% less time in total, because it wrote 28% fewer tokens, and
  writing tokens took 84–90% of each arm's time. Of the 15.9 minutes saved, 9.9 came from the 14 tasks both arms
  solved, and 5.9 from the two that neither solved.

## Each task

Means of the two attempts per arm, without / with the adapter, rounded half up.

| Task | Set | Solved | Steps | Tokens written | Seconds |
|---|---|---|---|---|---|
| bisect | core | 2/2 / 2/2 | 7.0 / 4.5 | 800 / 602 | 98 / 82 |
| cli-flag | core | 2/2 / 2/2 | 7.0 / 4.0 | 1,181 / 565 | 146 / 75 |
| config-defaults | core | 2/2 / 2/2 | 5.0 / 4.0 | 715 / 523 | 87 / 69 |
| docstrings | core | 2/2 / 2/2 | 5.0 / 5.5 | 742 / 1,017 | 96 / 134 |
| inventory | core | 2/2 / 2/2 | 7.5 / 5.5 | 1,071 / 588 | 132 / 82 |
| pathlib | core | 2/2 / 2/2 | 9.5 / 4.5 | 1,612 / 905 | 192 / 117 |
| rename | core | 2/2 / 2/2 | 6.5 / 5.0 | 569 / 578 | 77 / 82 |
| slugify | core | 0/2 / 0/2 | 7.5 / 7.0 | 1,593 / 1,245 | 189 / 157 |
| cart | extended | 2/2 / 2/2 | 7.0 / 5.5 | 1,004 / 710 | 124 / 95 |
| csv-report | extended | 2/2 / 2/2 | 7.5 / 5.0 | 726 / 620 | 95 / 82 |
| email | extended | 2/2 / 2/2 | 7.5 / 4.5 | 993 / 552 | 128 / 74 |
| long-file | extended | 2/2 / 2/2 | 6.5 / 5.5 | 1,027 / 834 | 158 / 142 |
| lru | extended | 2/2 / 2/2 | 4.0 / 5.0 | 654 / 891 | 84 / 112 |
| signature | extended | 0/2 / 0/2 | 9.0 / 4.5 | 1,859 / 667 | 234 / 89 |
| todo-cli | extended | 2/2 / 2/2 | 6.5 / 7.5 | 824 / 958 | 109 / 130 |
| write-tests | extended | 2/2 / 2/2 | 5.0 / 4.0 | 1,076 / 637 | 132 / 84 |

## What the adapter costs in llama.cpp

llama.cpp keeps the adapter beside the base weights and computes its extra products on every request (`--lora`);
it is not merged into the model. These are llama-server's own timings, summed over the agent's requests. Each
run's three setup requests are left out: a warm-up and two probes of the thinking switch.

| Arm | Agent requests | Prompt processing | Writing tokens (decode) |
|---|---|---|---|
| Without the adapter | 216 | 70,973 tokens in 6.85 min (172.7 tok/s) | 32,883 tokens in 62.40 min (8.78 tok/s) |
| With the adapter | 163 | 61,102 tokens in 8.47 min (120.2 tok/s) | 23,776 tokens in 44.95 min (8.81 tok/s) |

- **Decoding** ran at the same speed in both arms.
- **Prompt processing** was 30% slower with the adapter: 44% more time per token. The arms processed different
  prompts, but each task set's first prompt is the same in both:

  | First prompt | Without the adapter | With the adapter |
  |---|---|---|
  | Core set, 1,095 tokens | 190.5 tok/s | 126.5 tok/s |
  | Extended set, 1,082 tokens | 202.9 tok/s | 126.1 tok/s |

- **The cost in minutes, an estimate.** At the arm without the adapter's overall rate, the adapter arm's 61,102
  prompt tokens would have taken 5.90 minutes instead of 8.47. That 2.57-minute difference is an estimate for
  these workloads.
- **The net effect.** The adapter arm spent 17.45 fewer minutes decoding and 1.62 more processing prompts.

A single request in a check before the runs suggested a 20% slower decode; the runs did not show it. Merging the
adapter into the weights before quantizing would remove the run-time products but change what is quantized. We did
not test it.

## How it ran

- **Order.** The four runs went without, with, with, without the adapter (core, core, extended, extended). That
  reverses which arm goes first across the two task sets. It does not randomize the order or remove drift over the
  day. Before each start, the queue waited until no engine process was left and memory had settled. Each run takes
  two readings just before it launches the engine:
  - the estimated available memory, from four `vm_stat` counters;
  - the system-wide anonymous memory: all apps together.

  | Run | Started | Solved | Total time | Estimated available before launch | System-wide anonymous memory |
  |---|---|---|---|---|---|
  | Without the adapter, core | 11:45:57 | 14/16 | 33.9 min | 9.9 GB | 6.7 GB |
  | With the adapter, core | 12:21:10 | 14/16 | 26.7 min | 15.1 GB | 2.7 GB |
  | With the adapter, extended | 12:49:03 | 14/16 | 26.9 min | 15.3 GB | 2.5 GB |
  | Without the adapter, extended | 13:17:15 | 14/16 | 35.5 min | 15.9 GB | 1.9 GB |

  Both runs without the adapter, one before and one after the runs with it, took longer than both runs with it.
  The first one started with the least estimated available memory and the most anonymous memory of the four.
  Its memory pressure alternated between normal and warning in its first minutes. The last one started with the
  most estimated available memory and took about as long.
- **The adapter was applied.** The run records show the launch command with `--lora`. llama-server's log, at its
  default verbosity, does not mention adapters. Before the runs, we sent the same prompt at temperature 0 with the
  adapter's scale set to 1 and then to 0, through llama-server's `/lora-adapters`. The text came out the same, but
  the token log-probabilities differed by up to 0.19. The same request sent twice at scale 0 gave identical
  numbers.
- **A first try, discarded.** The first planned run was the core set without the adapter, from 11:09 to 11:28. It
  ran with llama-server's RAM prompt cache at its default size (`--cache-ram`, 8,192 MiB), which keeps earlier
  conversations' states beside the model. Our memory watchdog stopped it at critical pressure during its 10th
  attempt, after 8 passes in 9; the other three planned runs had not started.
  - **The cache.** The server had begun evicting entries from the cache about seven minutes in, by which time
    swap had grown by about 7.8 GiB. It was saving another entry when it was stopped.
  - **Our reading.** We take the cache to be the main cause. The records do not measure its occupancy throughout,
    and a Lean 4 install and compile also ran on the Mac from about 11:15 to 11:25.
  - **What we did.** We discarded the run and started a fresh four-run sequence with the RAM prompt cache off
    throughout (`SOLVETIME_LLAMA_CACHE_RAM`, solvetime #45). The discarded run's records and its timeline are in
    [excluded/](excluded/).

## Limits

- **Sample size and tests.** 16 tasks, with two attempts per task in each arm, sampled at temperature 0.7, in
  four runs. The sign tests compare per-task means of two attempts and drop ties. They treat the 16 tasks'
  directions as independent; with eight tasks sharing each run and a fixed task suite, that holds only
  approximately. Steps, tokens and time were all tested, so read the results as exploratory:

  | Measure | Unadjusted p | Bonferroni-adjusted p |
  |---|---|---|
  | Steps | 0.021 | 0.064 |
  | Tokens written | 0.077 | 0.23 |
  | Time | 0.077 | 0.23 |

- **Pass rates.** Both arms solved the same tasks. That leaves room for differences on other tasks or over more
  attempts. With 14 of the 16 tasks solved in every attempt, these tasks leave little room to show a change in pass
  rate. The adapter's own card reports gains on harder benchmarks.
- **One setup.** Everything here comes from a single configuration:
  - the model: a 4-bit GGUF, while the adapter was trained on the BF16 model;
  - the engine: llama.cpp, applying the adapter at run time;
  - the agent: mini-swe-agent, which is not the pi harness the adapter was trained for;
  - thinking off, a 16K window, and one Mac.

  The adapter's GGUF card says the GGUF has not been evaluated through llama.cpp and that a quantized base will
  move the results.
- **Training data.** The adapter's card lists terminal and software-engineering task sets among its training data
  (harness-hub/scale800-harbor, harness-hub/termgrade-train-harbor, and others). We have not checked them for
  overlap with solvetime's tasks. Those tasks entered this repository on 2026-11-15 (git history); the adapter's
  repository on Hugging Face was created on 2026-11-16.

## The adapter, the model and the engine

- **Adapter:** [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted), a
  PEFT LoRA (rank 32, alpha 64, on the q, k, v, o, gate, up and down projections), Apache-2.0. It was trained with
  reinforcement learning, thinking off, as the model behind the [pi](https://github.com/earendil-works/pi) coding
  agent. Its card, at revision 80d05c9, reports results under pi, with the BF16 adapter served by vLLM and one
  attempt per task:
  - **Terminal-Bench 2.1:** from 39.5% to 55.8%, on 86 tasks graded for both.
  - **SWE-bench Pro (hard 51):** from 39.2% to 64.7%. An earlier version of the card gave 29.4% to 62.7%; the
    authors corrected it after finding that nine of the base model's rollouts there had failed because a network
    tunnel in their setup went down.
  - **Turns:** on the tasks both solved, the median number of turns fell from 59 to 23 (SWE-bench Pro, 18 tasks)
    and from 18 to 8 (Terminal-Bench 2.1, 32 tasks).

  Its turns and our steps count requests under different harnesses and limits.
- **The adapter file we ran:** `Qwen3.8-27B-pi-lora-F16.gguf` (0.32 GB) from
  [harness-hub/Qwen3.8-27B-pi-adapted-GGUF](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF) at
  revision e8c5252, Apache-2.0, converted with llama.cpp's `convert_lora_to_gguf.py`. We downloaded it under its
  earlier name, harness-hub/Qwen3.8-27B-pi-GGUF, which Hugging Face now resolves to that repository. SHA-256:
  `a64f91844f19e67da500d8628bdf9383aa334e61b7ab687805802e7d272d7b67`.
- **Base model file:** `Qwen3.8-27B-UD-IQ4_XS.gguf` from
  [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision 4af51b3, Apache-2.0.
  It is 14.35 GB, at 4.20 bits per weight over the 27.3B parameters in the file, and was made from
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). SHA-256:
  `4b194b34ecec99fdfb74caef84445ccd01c987de87d96ae8ca50dc434b4b0a38`. Both files' digests matched Hugging Face's.
- **Engine:** llama.cpp 0.9.0 (build 12110, commit cd959bae9):
  `llama-server -ngl 999 -fa on --jinja -c 16384 --cache-ram 0`, plus `--lora` in the adapter arm. Its default
  of 4 slots sharing one KV cache was left as is.

## Run it again

To remake the tables from the records in this folder:

```sh
python3 compare_arms.py runs/*
```

To run the experiment again, use solvetime at fd87284 or later. Run the four steps in this order, into one new
folder:

```sh
export SOLVETIME_GGUF=/path/to/Qwen3.8-27B-UD-IQ4_XS.gguf SOLVETIME_LLAMA_CTX=16384 SOLVETIME_LLAMA_CACHE_RAM=0
LORA=/path/to/Qwen3.8-27B-pi-lora-F16.gguf
RUN="solvetime run --engine llama.cpp --model qwen3.8-27b --attempts 2 --thinking no-thinking --out rerun"
$RUN --set core                                  # 1. without the adapter
SOLVETIME_LLAMA_LORA=$LORA $RUN --set core       # 2. with it
SOLVETIME_LLAMA_LORA=$LORA $RUN --set extended   # 3. with it
$RUN --set extended                              # 4. without it
python3 /path/to/compare_arms.py rerun/*
```

## Files

- `runs/`: the four runs (run.json, attempts.jsonl, engine.log, and each attempt's trajectory).
  `solvetime validate runs` replays every attempt and checks it again: 4 runs, 0 problems, on these files.
- `excluded/`: the discarded first try and its timeline.
- `compare_arms.py`: the tables above, from the runs' records. It refuses an incomplete or mixed set of runs.
- All records have the paths, account name, host names and network addresses that would identify the Mac
  replaced (solvetime #47). Workspace paths keep their shape for the replay.
