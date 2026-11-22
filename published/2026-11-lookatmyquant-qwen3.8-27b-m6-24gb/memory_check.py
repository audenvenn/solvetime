"""Round 4, after the round: what LookAtMyQuant's 7.2 GB Qwen3.8-27B build holds in memory, and what a long prefill
needs.

This loads the build as the launcher does (runtime_mlx.load with its defaults) and reads MLX's own counters: active
memory (the live arrays), the allocator's cache of freed buffers, and the peak of active memory. It prefills the
round's longest prompt length, 10,634 tokens of synthetic text, in mlx-lm's server's 2,048-token chunks, twice, each
time with a fresh KV cache:
1. **as mlx-lm 0.32.2's server does** (generate.py's batched prompt loop): evaluate the cache after each chunk, then
   mx.clear_cache(), and clear once more at the end;
2. **without clearing,** as the first version of this check did (2026-11-22, 14:34). It left 8.97 GB in the cache.

The model's arrays are also split into the module tree's parameters and the arrays LookAtMyQuant's own structures
hold. That split counts arrays reachable from those objects, to a depth of 4 and each once.
Run with ~/venvs/lamq/bin/python, under the GPU lock.
"""

import json
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from runtime_mlx import load

GB = 1e9
STEP = 2048     # mlx_lm.server's --prefill-step-size default
LENGTH = 10634  # the longest prompt of the round, in tokens


def mem(at: str) -> None:
    print(json.dumps({"at": at, "active_gb": round(mx.get_active_memory() / GB, 2),
                      "cache_gb": round(mx.get_cache_memory() / GB, 2),
                      "peak_active_gb": round(mx.get_peak_memory() / GB, 2)}), flush=True)


def held_arrays(obj, seen: set, depth: int = 0) -> int:
    """Bytes of the mx.arrays an object holds in its attributes, lists and dicts, each array counted once."""
    if isinstance(obj, mx.array):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return obj.nbytes
    if depth > 4:
        return 0
    if isinstance(obj, dict):
        return sum(held_arrays(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(held_arrays(v, seen, depth + 1) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(held_arrays(v, seen, depth + 1) for v in vars(obj).values())
    return 0


def prefill(ids: list, clear: bool) -> None:
    cache = make_prompt_cache(model)
    mx.reset_peak_memory()
    t = time.time()
    for i in range(0, len(ids), STEP):
        model(mx.array([ids[i:i + STEP]]), cache=cache)
        mx.eval([c.state for c in cache])
        if clear:
            mx.clear_cache()
    if clear:
        mx.clear_cache()
    mx.synchronize()
    how = "clearing the cache after each chunk, as the server does" if clear else "without clearing"
    print(json.dumps({"prefill": how, "tokens": len(ids), "seconds": round(time.time() - t, 1)}), flush=True)
    mem(f"after the prefill, {how}, its KV cache still held")
    del cache
    mx.clear_cache()
    mem("after dropping that KV cache and clearing MLX's cache")


build = sys.argv[1]
t = time.time()
model, tok = load(build)
mx.synchronize()
print(json.dumps({"load_seconds": round(time.time() - t, 1)}), flush=True)
mem("after load")
params = tree_flatten(model.parameters())
seen = {id(v) for _, v in params}
print(json.dumps({
    "parameter_gb": round(sum(v.nbytes for _, v in params) / GB, 2),
    "vision_tower_gb": round(sum(v.nbytes for k, v in params if "vision" in k or "visual" in k) / GB, 2),
    "float32_parameter_gb": round(sum(v.nbytes for _, v in params if v.dtype == mx.float32) / GB, 2),
    "lookatmyquant_structures_gb": round(sum(held_arrays(s, seen) for s in getattr(model, "lamq_stages", [])) / GB, 2),
}), flush=True)
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 1200)[:LENGTH]
prefill(ids, clear=True)
prefill(ids, clear=False)
