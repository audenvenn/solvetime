# The discarded first try: without the adapter, core set, 2026-11-20 11:09–11:28

Not scored. Its records are in `20261120-110902-llama.cpp-qwen3.8-27b/`. It was the first of the four planned
runs; the other three had not started.

**What happened.** It ran with llama-server's RAM prompt cache at its default size. Our memory watchdog stopped it
at critical pressure (11:28:37) during its 10th attempt; 8 of its first 9 attempts had passed. Memory pressure was
at warning from the first minute. The estimated available memory at launch was 13.1 GB, for a 14.35 GB model at a
16K window.

**The likely cause: llama-server's RAM prompt cache** (`--cache-ram`, 8,192 MiB by default), which keeps the
states of earlier conversations beside the model.
- **Evictions.** Its log records eight of them ("making room for prompt cache entry, removing oldest entry (size =
  2585.213 MiB)"). The largest evicted entry was 2,636.9 MiB.
- **Timing.** The first eviction came 6 min 58 s into the server's log, about 11:16. Swap grew from 1,841 MiB at
  11:09 to 9,784 MiB at 11:16. The last eviction began 19 min 28 s in, as the 10th attempt started, and the
  critical reading came within seconds.
- **What the records can't show.** They do not measure the cache's occupancy throughout. Other memory use on the
  Mac (below) can't be separated from it.
- **The server's memory.** `top`'s MEM column, in the watchdog's snapshots, read 11G during the run and 10G at
  critical. In a two-request check before the run, `footprint`, a different instrument, gave 2.3 GB for the same
  server.
- **Slot use.** All agent requests used one slot, slot 3 of 4; the harness's three setup requests used the others.
  Requests within an attempt reused the slot's prefix. The default-verbosity log records only the evictions; we did
  not measure how often the cache was hit.

**How it ended.** The watchdog's interrupts arrived while the server was saving that entry. llama-server printed
"Received second interrupt, terminating immediately". Then came a Metal assertion at exit
(`ggml-metal-device.m:1041`, `GGML_ASSERT([rsets->data count] == 0)`), with a backtrace inside the prompt-cache
copy. That backtrace comes from the forced exit; it is not a separate crash.

**Something else on the Mac.** Another job installed and compiled a Lean 4 project from about 11:15 to 11:25.
- **Process lists.** The watchdog took 58 snapshots in that window, each listing the eight processes using the most
  memory. None shows a Lean process, but that does not measure what it used.
- **Swap.** After its early rise, swap fluctuated. It fell from 10,224 to 8,186 MiB between 11:17:19 and 11:17:59.
  It rose from 9,067 to 10,271 MiB between 11:23:53 and 11:24:53, the minute of a cache save.

**What changed.** solvetime #45 added `SOLVETIME_LLAMA_CACHE_RAM`. A fresh sequence of all four runs started at
11:45:57 with the cache off (`--cache-ram 0`). In its first 12 minutes the server's memory read:
- 3.5–4.9 GB in the watchdog's 22 warning snapshots (top's MEM);
- 3.2–4.9 GB in a separate `footprint` reading each minute.
