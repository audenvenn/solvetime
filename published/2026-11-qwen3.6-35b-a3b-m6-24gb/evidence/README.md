# Qwen3.6-35B-A3B at ~2 bits drops the opening `<tool_call>` token

**What happens.** With the unsloth UD-IQ2_XXS build of Qwen3.6-35B-A3B, the model often writes a tool call
without its opening `<tool_call>` token. The reply ends its thinking, then writes
`<function=bash>\n<parameter=command>\n...\n</parameter>\n</function>\n</tool_call>` as plain text. The server finds no
tool call and returns `finish_reason: "stop"`, `tool_calls: null`, and the call text as `content`.

The intent is unambiguous: the function name, its parameters and the closing `</tool_call>` are all there. The
same model at 4 bits does this too, but rarely enough that an agent recovers.

## Environment

- Mac mini (2026), Apple M6, 12-core GPU, 24 GB, macOS 27.2 (26B123), on AC power.
- Weights: `unsloth/Qwen3.6-35B-A3B-GGUF` at revision `a483e9e6cbd595906af30beda3187c2663a1118c`, file
  `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`, 10,756,586,464 bytes, sha256
  `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef`.
- **llama.cpp** 0.9.0 (build 12110, commit cd959bae9), Homebrew:
  `llama-server -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --port 8105 --host 127.0.0.1 -ngl 999 -fa on --jinja -c 32768`
- **kapitsa** 0.2.36 (fe6a17a), Homebrew. `kapitsa serve --model 35b --private --local` resolves to `35b:16gb` on this Mac:
  the same GGUF, imported. Its log reports no MTP head, no DFlash2 drafter, plain decoding.

## Minimal repro

`request.json` is the first turn of a coding task exactly as mini-swe-agent 2.4.6 sends it:
- its default system prompt and instance prompt, with one `bash` tool;
- the task: rename `calc_area` to `area`;
- thinking on (`chat_template_kwargs.enable_thinking: true`);
- the model card's sampling: temperature 0.6, top-p 0.95, top-k 20;
- `max_tokens` 2048.

```bash
python repro_missing_tool_call.py http://127.0.0.1:8105 1 2 3 4 5 6 7 8 9 10
# or, for one seed:
jq '. + {seed: 1, model: "any"}' request.json | curl -s http://127.0.0.1:8105/v1/chat/completions \
  -H 'Content-Type: application/json' -d @- | jq '.choices[0] | {finish_reason, tool_calls: .message.tool_calls, content: .message.content}'
```

Results (`seeds_1-10.jsonl`; whole replies in `replies.jsonl`):

| Server | Seeds | Opening `<tool_call>` missing | Proper tool call | Other |
|---|---|---|---|---|
| llama.cpp | 1-10 | 8 (1, 3, 5, 6, 7, 8, 9, 10) | 1 (seed 2) | 1 (seed 4: prose, no call) |
| llama.cpp, repeated | 1, 1, 2, 2 | seed 1 both times | seed 2 both times | |
| kapitsa | 1-5 | 5 of 5 | 0 | 0 |

So llama.cpp is deterministic for a seed: seed 1 always reproduces the drop, and seed 2 always gives a proper call.

llama.cpp, seed 1, `content` (`finish_reason: "stop"`, no `tool_calls`):

```
<function=bash>
<parameter=command>
find / -name "main.py" 2>/dev/null | head -20
</parameter>
</function>
</tool_call>
```

kapitsa, seed 1, `content`:

```
<function=bash>
<parameter=command>
find / -name "main.py" -type f 2>/dev/null | head -20
</parameter>
</function>
</tool_call>
```

In both, the thinking that comes before ends normally (`...Let me start by exploring the project structure.`). The
opening token isn't hidden in the reasoning; the model never wrote it.

## What it does to an agent

solvetime: the 8 core tasks × 2 attempts, mini-swe-agent 2.4.6, thinking on, card sampling. The agent rejects a reply
without a tool call, tells the model so, and gives up after 3 in a row.

| Engine and build | Solved | Replies without a tool call | Attempts ended on 3 in a row |
|---|---|---|---|
| raceline (main after 0.7.1, released as 0.8.0), the same model at 4 bits (experts streamed from the SSD) | 14/16 | 5 (3 attempts, all recovered) | 0 |
| llama.cpp 0.9.0, UD-IQ2_XXS | 10/16 | 16 (6 attempts, 2 recovered) | 4 |
| kapitsa 0.2.36, `35b:16gb` (UD-IQ2_XXS imported) | 7/16 | 29 (10 attempts, 1 recovered) | 9 |

In every rejected llama.cpp reply logged (16 of 16), the call text was there without its opening tag.

## A fix any server can make

Accept a tool call whose opening `<tool_call>` is missing when the rest is a well-formed call. That means content that
starts, after optional whitespace, with `<function=NAME>`, has its `<parameter=...>` blocks, and ends with
`</function>` and `</tool_call>`, where `NAME` is one of the request's tools. Parse it as the call it is.
- The model's intent is unambiguous, and no ordinary text has that shape.
- This would rescue nearly all of the attempts above, on any engine and at any bit width.
- Alternatively, a 24 GB default could prefer a ~3-bit expert variant if one fits, but that doesn't remove the
  quirk.

## After the fix

kapitsa fixed it in main the next day: 7445a6c5, PR #1279 on
[nace-ai/kapitsa#1277](https://github.com/nace-ai/kapitsa/issues/1277). A reply that is a single well-formed
call to a declared tool is now returned as a tool call. On the same request, seeds 1-5 all come back as proper
tool calls (`seeds_1-5-main-7445a6c.jsonl`; the whole replies are in `replies.jsonl`). On the agent tasks,
the fixed build no longer ends an attempt on malformed output, and it solves as many tasks as the 4-bit
build. The counts are in this directory's parent.
