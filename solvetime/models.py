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

# The Qwen3.5, 3.6 and 3.8 cards' sampling for thinking and non-thinking use, sent with every request of a run.
QWEN3_SAMPLING = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                 "enable_thinking": True},
    "no-thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                    "enable_thinking": False},
}


@dataclass(frozen=True)
class Model:
    name: str                      # solvetime's name for it (--model)
    hf: str                        # the official checkpoint on Hugging Face
    params: int | None             # text-model parameters; None: read from `hf`
    sampling: dict                 # "thinking" / "no-thinking" -> the card's request fields
    tokenizer: str                 # the Hugging Face repo whose tokenizer.json counts tokens in the speed test
    builds: dict = field(default_factory=dict)   # engine -> its build of this model


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
    Model("qwen3.8-27b", "Qwen/Qwen3.8-27B", 27_781_427_952 - 460_730_096, QWEN3_SAMPLING, "Qwen/Qwen3.5-9B", {
        "raceline": "qwen3.8-27b",
        "kapitsa": "qwen3.8-27b",
        "mtplx": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",                # its "tight_fit" for 24 GB
        "mlx-lm": "mlx-community/Qwen3.8-27B-4bit",
        "omlx": "mlx-community/Qwen3.8-27B-4bit",
        "ollama": "qwen3.8:27b",
        "lmstudio": "qwen/qwen3.8-27b",
    }),
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
