# MTPLX 2.13.1 on Inkling 2 Mini: the addendum's check

**What ran.** On 2026-11-22, from 21:49:36 to 21:55:46: one attempt of the inventory task (299.5 s), on solvetime
48f61a0.
- **The server:** MTPLX's default memory and context settings,
  `mtplx serve --model Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed --port 8890 --no-auth --yes`.
- **The model's settings:** the card's sampling (temperature 1.0, top_p 1.0, top_k 0, min_p 0), with no thinking
  override.
- **The build:** the pack at 84a7701, pulled with `mtplx pull --revision`. All 21 pack files matched their Hugging
  Face digests, recorded before the download (git blob ids or LFS SHA-256). All 18 files that the pack's own
  `SHA256SUMS` lists matched it.
- **Its source.** The pack lists the same upstream model-shard and drafter hashes as Inkling 2 Mini at ed40e7d. Its
  quantization differs from round 3's two scored builds.

**MTPLX's own plan at load**, from its startup log:

```
[5/6] Context window: 24576 tokens
[5/6] Memory plan: 24G Mac: engine budget 18.0G, weights 14.9G, context 24576 (machine-bound), session bank up to 2.1G (yields to 0.6G under long-context load) (tight machine: no bank floor, restores from SSD)
[5/6] Memory plan: tight machine: the session bank has no floor (the warm cache yields fully to live KV; restores come from SSD) and KV is counted at its dense width; weights 14.9 GiB + runtime 1.04 GiB leave 1.42 GiB for KV under the 18.0 GiB engine budget
[5/6] Model loaded in 55.7s
```

- **MTPLX's warm-up.** It completed its own internal warm-up.
- **solvetime's clock.** It read 64.9 s at the end of solvetime's separate warm-up request, which the guard refused.

**The agent's requests.** 45 in all, retries included, for three prompts: one of 994 tokens, 14 of 1,250 and 30 of
1,844.

| Outcome | Requests |
|---|---|
| Answered with a bash call that ran (requests 1 and 15) | 2 |
| Declined by MTPLX's memory guard before prefill | 42 |
| Admitted after the guard reclaimed memory, then stopped during prefill | 1 |

The task check failed.

**Setup.** The guard also declined solvetime's warm-up (18 tokens) and the two thinking probes (27 and 29 tokens).
- **The probe's record.** It recorded that as "ignored: never thinks", because MTPLX sends a refusal as an error
  event in a stream answered with HTTP 200. The record is corrected, and solvetime #69 fixes the probe.
- **Under the fixed harness,** the same refused warm-up would have stopped the run before the probe or the agent.

**The engine log's receipts.** It holds 50 `prefill_admission_shed` receipts:
- **45 refusals:** the 3 setup requests and the 42 agent requests above;
- **5 admissions** after the guard reclaimed memory. One of those was then stopped by a `prefill_system_abort`.

**The guard's figures.** For the 42 agent refusals, it estimated 1.71 to 2.29 GiB available after reclaiming its own
caches, with shortfalls of 0.04 to 0.66 GiB.

**A refusal, as the agent got it** (its session id left out):

> insufficient memory: the other apps on this Mac leave 2.0 GiB free, and this prompt needs about 0.2 GiB more than
> that (1250 prompt tokens, 321 not cached). The engine gave back its own caches first. Held after reclamation: model
> weights 14.9 GiB; cached conversations 0.5 GiB: … 0.5 GiB (restore_source_or_not_reached); allocator pool 0.0 GiB;
> 0.7 GiB outside MLX. A retry can succeed once other apps give memory back: close some apps and try again, or
> shorten the prompt. The request was refused before prefill instead of pushing the Mac into swap, which can freeze
> the whole desktop; --allow-swap admits it anyway.

**The two answered replies.** Each was a bash tool call, and both ran with exit code 0:
1. `ls -la && which test_inventory.py inventory.py 2>/dev/null; pwd`
2. `cat inventory.py && echo '=====' && cat test_inventory.py`

**Memory.**
- **Just before launch**, by solvetime's counters: 13.6 GB estimated available, 7.4 GB of anonymous pages, 2.3 GB
  wired and 0.4 GB in the compressor. Other applications were open.
  - **Nine Claude Code processes,** about seven minutes before launch, summed to 3.68 GiB of resident memory, which
    can include shared pages.
- **During the check.** At the first pressure warning, at 21:50:19, 18.6 GB was wired system-wide and the compressor
  held 3.2 GB. It held 3.5 and 3.6 GB at the next two warnings.
  - **What top showed.** The MTPLX server was at 17G, 17G and 16G in those three snapshots. Beside it, top's eight
    largest showed the window server, WebKit, four Claude Code processes and a remote-desktop app.
  - **The peak.** solvetime recorded a peak footprint of 18.9 GB for the server.

**solvetime's own gate.**
- **The first try.** Its 17 GB pre-launch check timed out before launching MTPLX. Its last reading was 15.9 GB of room
  (14.0 GB estimated available).
- **The second try.** It was lowered to 15.5 GB, and MTPLX launched at 15.6 GB of room (13.5 GB estimated available).
- **What room counts:** reclaimable file cache. Neither threshold changes MTPLX's guard, and MTPLX's memory settings
  stayed at their defaults.

**Not scored.** The maintainer stopped after this check: no scored suite or second attempt was run, and neither
`--memory-limit` nor `--allow-swap` was used. Round 2's MTPLX run of the 27B was scored, 0/16, and was likewise not
rerun.
