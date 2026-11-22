"""The models solvetime runs: the official checkpoint, the model card's sampling, and each engine's build.

A build is the engine's own name for its build of the model: a pulled model, a catalog key, a Hugging Face
repo, a tag. An engine without a build of a model says so (`solvetime doctor --model ...`), and its
variable (BUILD_VARIABLE) points it at one. Adding a model is one entry here: its checkpoint, its card's
sampling, and the builds the engines publish.

Bits per weight are reported against the text model's parameters: the vision tower is left out, since every
engine here runs text, and the MTP head is counted. When the count is not written down, it is read from the
official repo's safetensors headers (nothing else is downloaded) and kept in ~/.cache/solvetime.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "solvetime", "params.json")

# The Qwen3.5 and 3.6 cards' sampling, sent with every request of a run: in thinking mode the cards' setting for
# precise coding tasks (they give 1.0 for general tasks), and their non-thinking setting.
QWEN3_SAMPLING = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                 "enable_thinking": True},
    "no-thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                    "enable_thinking": False},
}
# Qwen3.8's card gives one thinking setting, temperature 1.0 with the same top_p, top_k, min_p and presence
# penalty, and the same non-thinking one.
# - **Through round 4 (2026-11-22)** solvetime ran Qwen3.8 at the 3.5 and 3.6 cards' 0.6: round 2's and round 4's
#   thinking runs.
#   The pi-LoRA experiment ran with thinking off.
# - **So** later Qwen3.8 results with thinking are not directly comparable with those.
QWEN38_SAMPLING = {"thinking": dict(QWEN3_SAMPLING["thinking"], temperature=1.0),
                   "no-thinking": QWEN3_SAMPLING["no-thinking"]}

# Inkling 2's card gives one sampling setting, temperature 1.0; the rest is sent neutral so that engines' own
# defaults (llama.cpp's top_k 40, top_p 0.95, min_p 0.05) do not differ. Its template ignores enable_thinking: depth
# is `reasoning_effort` (none ... max, or 0.0-0.99), "high" when absent. Thinking runs send none, so the template's
# default applies. No-thinking runs send "none" as a template variable only: llama.cpp b12600 gives a top-level
# reasoning_effort precedence and then renders the default, 0.9 (its /apply-template, 2026-11-21), and oMLX 0.8.1
# forwards a top-level one only where the template variable is not set. Even at "none" the model writes a short
# thinking block (llama.cpp, 2026-11-21), so a probe of the switch says "ignored: always thinks".
INKLING2_OFF = {"chat_template_kwargs": {"reasoning_effort": "none"}}
INKLING2_SAMPLING = {
    "thinking": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
    "no-thinking": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "extra": INKLING2_OFF},
}


@dataclass(frozen=True)
class Model:
    name: str                      # solvetime's name for it (--model)
    hf: str                        # the official checkpoint on Hugging Face
    params: int | None             # text-model parameters; None: read from `hf`
    sampling: dict                 # "thinking" / "no-thinking" -> the card's request fields
    tokenizer: str                 # the Hugging Face repo whose tokenizer.json counts tokens in the speed test
    builds: dict = field(default_factory=dict)   # engine -> its build of this model
    drafter_files: tuple = ()      # files of a build that hold a separate drafter, not among `params`: left out of bits


MODELS = {m.name: m for m in [
    Model("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B", 35_951_822_704 - 446_571_248, QWEN3_SAMPLING, "Qwen/Qwen3.5-9B", {
        "raceline": "qwen3.6-35b-a3b",                                   # `raceline pull`
        "kapitsa": "qwen3.6-35b-a3b",    # its catalog's full name: the short one ("35b") may move to a newer model
        "mtplx": "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",      # for larger Macs: none fits 24 GB
        "mlx-lm": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "omlx": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "ollama": "qwen3.6:35b-a3b",                                     # its default tag
        "lmstudio": "qwen/qwen3.6-35b-a3b",                              # its staff pick
    }),
    Model("qwen3.8-27b", "Qwen/Qwen3.8-27B", 27_781_427_952 - 460_730_096, QWEN38_SAMPLING, "Qwen/Qwen3.5-9B", {
        "raceline": "qwen3.8-27b",
        "kapitsa": "qwen3.8-27b",
        "mtplx": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",                # its "tight_fit" for 24 GB
        "mlx-lm": "mlx-community/Qwen3.8-27B-4bit",
        "omlx": "mlx-community/Qwen3.8-27B-4bit",
        "ollama": "qwen3.8:27b",
        "lmstudio": "qwen/qwen3.8-27b",
    }),
    # 128 experts, 8 active; layers 6-17 run three times per token. Text parameters from the official checkpoint's
    # safetensors headers (ed40e7d), counted by hand: each NVFP4 byte holds two weights and the quantization scales
    # are left out, which text_params() would not do. The checkpoint carries no MTP head; the GGUF counts 40 fewer.
    Model("inkling-2-mini", "thinkingmachines/Inkling-2-Mini", 26_070_390_312, INKLING2_SAMPLING,
          "thinkingmachines/Inkling-2-Mini", {
        "omlx": "mlx-community/Inkling-2-Mini-oQ3e-mtp",                 # oMLX's oQ build ("needs oMLX 0.8.1")
        "ollama": "inkling-2-mini",                                      # its default tag: the 20 GB checkpoint
    }, drafter_files=("mtp.safetensors",)),   # oQ builds keep their DSpark drafter there (0.64 GB)
    Model("qwen3.5-9b", "Qwen/Qwen3.5-9B", 9_653_104_368 - 456_010_480, QWEN3_SAMPLING, "Qwen/Qwen3.5-9B", {
        "raceline": "qwen3.5-9b",
        "kapitsa": "qwen3.5-9b",
        "mtplx": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
        "mlx-lm": "mlx-community/Qwen3.5-9B-4bit",
        "omlx": "mlx-community/Qwen3.5-9B-4bit",
        "ollama": "qwen3.5:9b",
        "lmstudio": "qwen/qwen3.5-9b",
    }),
]}

# The variable that points each engine at a build of the model (the table's when it is not set).
BUILD_VARIABLE = {
    "raceline": "SOLVETIME_RACELINE_FILE",      # a pulled model's name, or its .rlm file
    "kapitsa": "SOLVETIME_KAPITSA_MODEL",       # a catalog name (<model>[:<variant>])
    "mtplx": "SOLVETIME_MTPLX_PACK",            # a Hugging Face repo of an MTPLX pack
    "mlx-lm": "SOLVETIME_MLX_MODEL",            # a Hugging Face repo or a model folder
    "llama.cpp": "SOLVETIME_GGUF",              # a .gguf file
    "ollama": "SOLVETIME_OLLAMA_TAG",           # a tag
    "lmstudio": "SOLVETIME_LMSTUDIO_MODEL",     # a model key
    "omlx": "SOLVETIME_OMLX_MODEL",             # a Hugging Face repo or a model folder
}


def model(name: str) -> Model:
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; solvetime knows {', '.join(MODELS)} (solvetime/models.py)")
    return MODELS[name]


def build(name: str, engine: str) -> str:
    """The engine's build of the model: its variable's, else the table's; "" when neither names one."""
    return os.environ.get(BUILD_VARIABLE[engine]) or model(name).builds.get(engine) or ""


def text_params(name: str) -> int | None:
    """Text-model parameters: written down, cached, or read from the official repo's safetensors headers."""
    m = model(name)
    if m.params:
        return m.params
    try:
        cached = json.load(open(CACHE))
    except (OSError, ValueError):
        cached = {}
    if m.hf in cached:
        return cached[m.hf]
    try:
        from huggingface_hub import HfApi
        meta = HfApi().get_safetensors_metadata(m.hf)
    except Exception:
        return None
    count = 0
    for f in meta.files_metadata.values():
        for tensor, info in f.tensors.items():
            if not re.search(r"visual|vision", tensor):
                n = 1
                for d in info.shape:
                    n *= d
                count += n
    if count:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        cached[m.hf] = count
        with open(CACHE, "w") as f:
            json.dump(cached, f, indent=1)
    return count or None


def thinking_fields(name: str, mode: str, engine_no_thinking: dict | None = None) -> dict:
    """The request fields that select thinking ("thinking") or none ("no-thinking") for a model. Qwen: the
    template's enable_thinking, plus the engine's own fields for "off" (LM Studio and Ollama need reasoning_effort
    "none"). A model with its own switch (Inkling 2: reasoning_effort) gives its fields in `extra`."""
    s = model(name).sampling[mode]
    if "enable_thinking" in s:
        on = s["enable_thinking"]
        return {"chat_template_kwargs": {"enable_thinking": on}, **({} if on else (engine_no_thinking or {}))}
    return dict(s.get("extra") or {})
