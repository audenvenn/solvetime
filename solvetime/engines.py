"""Engine adapters: how to start each engine's server with its own recommended build of a model.

Each engine runs the build its users get on this Mac: the one it picks by
itself or its documentation recommends for the memory at hand. The build's
name, bits and size go into every result. An engine that cannot load the
model here is a result too ("does not fit").

Models are named by the original checkpoint: qwen3.6-35b-a3b, qwen3.8-27b,
qwen3.5-9b. `pack()` maps that name to the engine's own build.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class Engine:
    name: str
    model: str                      # the checkpoint, e.g. qwen3.6-35b-a3b
    pack: str                       # the engine's build of it
    command: list[str]              # starts the server in the foreground
    port: int
    api_model: str                  # the "model" field of requests
    health_path: str = "/v1/models"
    start_timeout: int = 900        # seconds; the first start may download the build
    env: dict = field(default_factory=dict)
    version_command: list[str] = field(default_factory=list)
    pack_dir: str | None = None     # where the build's files are, for its size
    ours: bool = False              # raceline: the author's own engine, labelled as such

    def version(self) -> str:
        if not self.version_command:
            return ""
        try:
            out = subprocess.run(self.version_command, capture_output=True, text=True, timeout=60)
            lines = (out.stdout + "\n" + out.stderr).strip().splitlines()
            named = [line for line in lines if "version" in line.lower()]
            return (named[0] if named else lines[-1]).strip()[:200]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            return ""

    info: object = None             # engine-specific pack_info(), when the build's files alone don't tell

    def pack_info(self) -> dict:
        """Size of the build's files, its weights in bytes and bits per weight (weights / the original
        checkpoint's parameter count: bf16 is 16)."""
        if callable(self.info):
            return self.info(self)
        size = self.pack_bytes()
        return {"pack_bytes": size, "weights_bytes": size, "bits_per_weight": bits(size, self.model)}

    def pack_bytes(self) -> int | None:
        if not self.pack_dir or not os.path.exists(self.pack_dir):
            return None
        if os.path.isfile(self.pack_dir):
            return os.path.getsize(self.pack_dir)
        total = 0
        for root, _, files in os.walk(self.pack_dir):
            for f in files:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    total += os.path.getsize(p)
        return total


HOME = os.path.expanduser("~")

# Parameters of the original checkpoints without their vision tower (Hugging Face safetensors metadata, minus
# the tower's tensors): the text model and its MTP head. Engines load the tower only when an image comes.
PARAMS = {"qwen3.5-9b": 9_653_104_368 - 456_010_480, "qwen3.8-27b": 27_781_427_952 - 460_730_096,
          "qwen3.6-35b-a3b": 35_951_822_704 - 446_571_248}


def text_weight_bytes(folder: str | None) -> int | None:
    """Bytes of the tensors in a folder's *.safetensors that are not the vision tower (headers only)."""
    if not folder or not os.path.isdir(folder):
        return None
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".safetensors"):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n))
            for name, meta in header.items():
                if name != "__metadata__" and not re.search(r"visual|vision", name):
                    start, end = meta["data_offsets"]
                    total += end - start
    return total or None


def _files_info(engine: "Engine") -> dict:
    return {"pack_bytes": engine.pack_bytes(), "weights_bytes": (w := text_weight_bytes(engine.pack_dir)),
            "bits_per_weight": bits(w, engine.model)}


def bits(weights_bytes: int | None, model: str) -> float | None:
    """Bits per weight of the text model: bf16 is 16."""
    return round(8 * weights_bytes / PARAMS[model], 2) if weights_bytes and model in PARAMS else None


def raceline(model: str, port: int) -> Engine:
    alias = {"qwen3.6-35b-a3b": "qwen3.6-35b-a3b", "qwen3.8-27b": "qwen3.8-27b", "qwen3.5-9b": "qwen3.5-9b"}[model]
    exe = os.environ.get("SOLVETIME_RACELINE", shutil.which("raceline") or "raceline")
    target = os.environ.get("SOLVETIME_RACELINE_FILE", alias)   # a pulled model's name, or its .rlm file
    # The server's default window is 32768 tokens; an agent's conversation can outgrow it, so 65536 here
    # (the other engines default to 16K-256K on a 24 GB Mac). The 27B keeps 32768: its weights leave no room
    # for a 65536-token cache on 24 GB. The longest conversation in the core tasks is about 13K tokens.
    ctx = os.environ.get("SOLVETIME_RACELINE_CTX", "32768" if model == "qwen3.8-27b" else "65536")
    return Engine("raceline", model, f"{alias} (FP4, 4.0 bits, with its MTP head)",
                  [exe, "serve", target, "--port", str(port), "--ctx", ctx],
                  port, alias, version_command=[os.path.join(os.path.dirname(exe), "python"), "-c",
                                                "import raceline; print('raceline', raceline.__version__)"],
                  pack_dir=os.environ.get("SOLVETIME_RACELINE_FILE"), ours=True)


def _kapitsa_info(engine: Engine) -> dict:
    """The variant kapitsa resolves on this Mac, its folder, and the weights and formats `kapitsa doctor` lists."""
    key = engine.command[engine.command.index("--model") + 1]
    dry = subprocess.run(["kapitsa", "pull", "--dry-run", key], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"folder:\s+(\S+)", dry)
    folder = os.path.expanduser(m.group(1)) if m else None
    variant = re.search(r"variant:\s+(.+)", dry)
    engine.pack_dir = folder
    out = {"pack": f"kapitsa {key}: {os.path.basename(folder) if folder else '?'}", "pack_bytes": engine.pack_bytes(),
           "variant": variant.group(1).strip() if variant else None}
    doctor = subprocess.run(["kapitsa", "doctor"], capture_output=True, text=True, timeout=300).stdout
    name = os.path.basename(folder) if folder else ""
    block = re.search(rf"^  {re.escape(name)}[ ,(].*?(?=^  \S|\Z)", doctor, re.M | re.S) if name else None
    if block:
        text = block.group(0)
        w = re.search(r"weights ([0-9.]+) GB", text)
        formats = re.search(r"weights ([^)]*)\)", text.splitlines()[0])
        out["weights_bytes"] = int(float(w.group(1)) * 1e9) if w else None
        out["weight_formats"] = formats.group(1) if formats else None
        decoding = re.search(r"decoding: (.+)", text)
        out["decoding"] = decoding.group(1).strip() if decoding else None
    out["bits_per_weight"] = bits(out.get("weights_bytes"), engine.model)
    return out


def kapitsa(model: str, port: int) -> Engine:
    key = {"qwen3.6-35b-a3b": "35b", "qwen3.8-27b": "27b", "qwen3.5-9b": "9b"}[model]
    return Engine("kapitsa", model, f"kapitsa {key} (the variant `kapitsa doctor` plans for this Mac)",
                  ["kapitsa", "serve", "--model", key, "--port", str(port), "--private", "--local"],
                  port, {"35b": "qwen3.6-35b-a3b", "27b": "qwen3.8-27b", "9b": "qwen3.5-9b"}[key],
                  version_command=["kapitsa", "--version"], info=_kapitsa_info)


MTPLX_PACKS = {  # the catalog's recommendation for a 24 GB Mac where it has one (README, "Models and recommended settings")
    "qwen3.5-9b": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
    "qwen3.8-27b": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
    "qwen3.6-35b-a3b": "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
}


def mtplx(model: str, port: int) -> Engine:
    repo = MTPLX_PACKS[model]
    exe = os.environ.get("SOLVETIME_MTPLX", os.path.join(HOME, "venvs/mtplx/bin/mtplx"))
    return Engine("mtplx", model, repo, [exe, "serve", "--model", repo, "--port", str(port), "--no-auth", "--yes"],
                  port, repo, health_path="/health", version_command=[exe, "--version"],
                  pack_dir=os.path.join(HOME, ".mtplx/models", repo.replace("/", "--")), info=_files_info)


MLX_PACKS = {"qwen3.5-9b": "mlx-community/Qwen3.5-9B-4bit", "qwen3.8-27b": "mlx-community/Qwen3.8-27B-4bit",
             "qwen3.6-35b-a3b": "mlx-community/Qwen3.6-35B-A3B-4bit"}


def mlx_lm(model: str, port: int) -> Engine:
    path = os.environ.get("SOLVETIME_MLX_MODEL", MLX_PACKS[model])
    py = os.environ.get("SOLVETIME_MLX_PYTHON", "python3")
    # One cached prompt: a single user's agent only ever extends its one conversation. The server's default keeps
    # the KV caches of several earlier prompts, and with the 27B's 15 GB of weights those outgrow the GPU memory
    # of a 24 GB Mac: the next 4K-token prompt fails with "Insufficient Memory" and the server stops generating.
    return Engine("mlx-lm", model, path, [py, "-m", "mlx_lm.server", "--model", path, "--port", str(port),
                                          "--prompt-cache-size", "1"],
                  port, path, version_command=[py, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
                  pack_dir=path if os.path.isdir(path) else None, info=_files_info)


def _gguf_info(engine: Engine) -> dict:
    """A GGUF's size, and its bits per weight from the tensor sizes its header lists (vision tower excluded)."""
    size = engine.pack_bytes()
    return {"pack_bytes": size, "weights_bytes": size, "bits_per_weight": bits(size, engine.model)}


def llama_cpp(model: str, port: int) -> Engine:
    """llama.cpp's llama-server on a GGUF (SOLVETIME_GGUF), all layers on the GPU, flash attention, the model's
    own chat template, and its MTP head as the drafter (--spec-type draft-mtp) when the GGUF carries one."""
    gguf = os.environ.get("SOLVETIME_GGUF", "")
    ctx = os.environ.get("SOLVETIME_LLAMA_CTX", "65536")
    command = ["llama-server", "-m", gguf, "--port", str(port), "--host", "127.0.0.1", "-ngl", "999", "-fa", "on",
               "--jinja", "-c", ctx]
    if os.environ.get("SOLVETIME_LLAMA_SPEC", "draft-mtp") != "none":
        command += ["--spec-type", os.environ.get("SOLVETIME_LLAMA_SPEC", "draft-mtp")]
    return Engine("llama.cpp", model, os.path.basename(gguf) or "(set SOLVETIME_GGUF)", command, port,
                  os.path.basename(gguf), health_path="/health", version_command=["llama-server", "--version"],
                  pack_dir=gguf or None, info=_gguf_info)


ENGINES = {"raceline": raceline, "kapitsa": kapitsa, "mtplx": mtplx, "mlx-lm": mlx_lm, "llama.cpp": llama_cpp}
