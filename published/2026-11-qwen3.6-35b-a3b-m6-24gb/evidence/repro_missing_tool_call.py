"""Minimal repro: Qwen3.6-35B-A3B at ~2 bits (unsloth UD-IQ2_XXS) sometimes writes a tool call without its opening
<tool_call> token, so the server returns it as plain content and no tool_calls.

The request is the first turn of a coding task as mini-swe-agent 2.4.6 sends it (its default system and instance
prompts, one `bash` tool), thinking on, the model card's sampling (T 0.6, top-p 0.95, top-k 20), with a fixed seed.

  python repro_missing_tool_call.py http://127.0.0.1:8080 [seeds...]    (writes request.json once)
"""
import json, os, sys, http.client, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
url = urllib.parse.urlparse(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]] or list(range(1, 11))
req_path = os.path.join(HERE, "request.json")
if not os.path.exists(req_path):
    import platform
    import yaml
    from jinja2 import StrictUndefined, Template
    from minisweagent import package_dir
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    cfg = yaml.safe_load(open(os.path.join(package_dir, "config", "mini.yaml")))["agent"]
    env = dict(platform.uname()._asdict(), task="Rename the function calc_area to area everywhere in the project, "
               "including all call sites. python3 main.py must print exactly what it prints now.")
    render = lambda s: Template(s, undefined=StrictUndefined).render(**env)
    json.dump({"messages": [{"role": "system", "content": render(cfg["system_template"])},
                            {"role": "user", "content": render(cfg["instance_template"])}],
               "tools": [BASH_TOOL], "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 2048,
               "chat_template_kwargs": {"enable_thinking": True}}, open(req_path, "w"), indent=1)
base = json.load(open(req_path))
for seed in seeds:
    body = dict(base, seed=seed, model="any")
    c = http.client.HTTPConnection(url.hostname, url.port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    m, finish = r["choices"][0]["message"], r["choices"][0].get("finish_reason")
    content = m.get("content") or ""
    broken = not m.get("tool_calls") and "<function=" in content and "<tool_call>" not in content
    print(json.dumps({"seed": seed, "finish": finish, "tool_calls": bool(m.get("tool_calls")),
                      "missing_open_tag": broken, "content_head": content[:140]}))
    with open(os.path.join(HERE, "replies.jsonl"), "a") as f:   # the whole reply, for the record
        f.write(json.dumps({"server": sys.argv[1], "seed": seed, "finish": finish, "content": content,
                            "reasoning": m.get("reasoning_content"), "tool_calls": m.get("tool_calls")}) + "\n")
