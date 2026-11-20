"""Engine adapters: how to start each engine's server with its own recommended build of a model.

Each engine runs the build its users get on this Mac: the one it picks by
itself or its documentation recommends for the memory at hand. The build's
name, bits and size go into every result. An engine that cannot load the
model here is a result too ("does not fit").

Models are named by the original checkpoint (qwen3.6-35b-a3b, qwen3.8-27b,
qwen3.5-9b); models.py names each engine's build of each.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from .models import BUILD_VARIABLE, build, model as model_entry, text_params


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
    settings: str = ""              # runtime options beyond the defaults, as passed (shown in the tables)
    clear_env: str = ""             # a prefix: the user's own variables with it are not passed to the server
    no_thinking: dict = field(default_factory=dict)   # request fields that turn thinking off, beyond
                                                     # chat_template_kwargs.enable_thinking (which it may ignore)
    scratch: list = field(default_factory=list)   # folders made for this run alone (oMLX's settings and SSD
                                                  # cache), removed when the run ends

    def server_env(self) -> dict:
        base = {k: v for k, v in os.environ.items() if not (self.clear_env and k.startswith(self.clear_env))}
        return dict(base, **self.env)

    def cleanup(self) -> None:
        """Remove the folders made for this run (oMLX's SSD cache can take half the free disk)."""
        for folder in self.scratch:
            shutil.rmtree(folder, ignore_errors=True)

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
    """Bits per weight of the text model (its parameters without the vision tower, which engines load only when
    an image comes): bf16 is 16."""
    params = text_params(model) if weights_bytes else None
    return round(8 * weights_bytes / params, 2) if params else None


# The package version, and the commit when raceline runs from a git checkout (where the version alone names
# the last release, not what was measured).
RACELINE_VERSION = """
import os, subprocess, raceline
d = os.path.dirname(raceline.__file__)
git = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
tracked = git("ls-files", "--error-unmatch", "__init__.py").returncode == 0
build = git("describe", "--tags", "--always", "--dirty").stdout.strip() if tracked else ""
print("raceline", raceline.__version__ + (f" ({build})" if build else ""))
"""


def _python_beside(exe: str) -> str:
    """The Python of the environment a console script was installed into (through symlinks, e.g. pipx), so
    that its package can be imported; this one if there is none."""
    real = shutil.which(exe) or exe
    py = os.path.join(os.path.dirname(os.path.realpath(real)), "python")
    return py if os.path.exists(py) else sys.executable


def raceline(model: str, port: int) -> Engine:
    alias = model_entry(model).builds.get("raceline", "")
    exe = os.environ.get("SOLVETIME_RACELINE", shutil.which("raceline") or "raceline")
    target = build(model, "raceline")   # a pulled model's name, or its .rlm file
    # The server's default window is 32768 tokens; an agent's conversation can outgrow it, so 65536 here
    # (the other engines default to 16K-256K on a 24 GB Mac). The 27B keeps 32768: its weights leave no room
    # for a 65536-token cache on 24 GB. The longest conversation in the core tasks is about 13K tokens.
    ctx = os.environ.get("SOLVETIME_RACELINE_CTX", "32768" if model == "qwen3.8-27b" else "65536")
    # Extra server options (e.g. "--expert-cache 8": 8 GB of experts in memory, the rest read from the SSD),
    # kept in the command and shown as the run's settings. The build stays the same.
    extra = os.environ.get("SOLVETIME_RACELINE_ARGS", "").split()
    return Engine("raceline", model, f"{alias or os.path.basename(target)} (FP4, 4.0 bits, with its MTP head)",
                  [exe, "serve", target, "--port", str(port), "--ctx", ctx, *extra],
                  port, alias, version_command=[_python_beside(exe), "-c", RACELINE_VERSION],
                  pack_dir=os.environ.get("SOLVETIME_RACELINE_FILE"), ours=True, settings=" ".join(extra))


def _kapitsa_info(engine: Engine) -> dict:
    """The variant kapitsa resolves on this Mac, its folder, and the weights and formats `kapitsa doctor` lists."""
    key = engine.command[engine.command.index("--model") + 1]
    exe = engine.command[0]
    dry = subprocess.run([exe, "pull", "--dry-run", key], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"folder:\s+(\S+)", dry)
    folder = os.path.expanduser(m.group(1)) if m else None
    variant = re.search(r"variant:\s+(.+)", dry)
    engine.pack_dir = folder
    out = {"pack": f"kapitsa {key}: {os.path.basename(folder) if folder else '?'}", "pack_bytes": engine.pack_bytes(),
           "variant": variant.group(1).strip() if variant else None}
    doctor = subprocess.run([exe, "doctor"], capture_output=True, text=True, timeout=300).stdout
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
    key = build(model, "kapitsa")   # its catalog's name for the model
    # SOLVETIME_KAPITSA: another kapitsa binary, e.g. a build of main (its version names the commit); the models
    # are shared (~/.cache/kapitsa/models)
    exe = os.environ.get("SOLVETIME_KAPITSA", "kapitsa")
    return Engine("kapitsa", model, f"kapitsa {key} (the variant `kapitsa doctor` plans for this Mac)",
                  [exe, "serve", "--model", key, "--port", str(port), "--private", "--local"],
                  port, model,
                  version_command=[exe, "--version"], info=_kapitsa_info)


def mtplx(model: str, port: int) -> Engine:
    """MTPLX on the pack of the same checkpoint that its catalog rates as fitting a 24 GB Mac (27B: Bare Speed,
    "tight_fit"; the catalog's "recommended" 24 GB pick, Ternary Bonsai 2 27B, is a different model). No 35B pack
    fits (insufficient_memory; see the results' not-run notes): that entry is for larger Macs."""
    repo = build(model, "mtplx")
    exe = os.environ.get("SOLVETIME_MTPLX", shutil.which("mtplx") or "mtplx")
    return Engine("mtplx", model, repo, [exe, "serve", "--model", repo, "--port", str(port), "--no-auth", "--yes"],
                  port, repo, health_path="/health", version_command=[exe, "--version"],
                  pack_dir=os.path.join(HOME, ".mtplx/models", repo.replace("/", "--")) if repo else None,
                  info=_files_info)


def mlx_lm(model: str, port: int) -> Engine:
    path = build(model, "mlx-lm")
    # The Python that has mlx-lm: SOLVETIME_MLX_PYTHON, else this one if mlx-lm is installed beside solvetime.
    py = os.environ.get("SOLVETIME_MLX_PYTHON") or (sys.executable if importlib.util.find_spec("mlx_lm") else "python3")
    # One cached prompt: a single user's agent only ever extends its one conversation. The server's default keeps
    # the KV caches of several earlier prompts, and with the 27B's 15 GB of weights those outgrow the GPU memory
    # of a 24 GB Mac: the next 4K-token prompt fails with "Insufficient Memory" and the server stops generating.
    # SOLVETIME_MLX_MAX_TOKENS: the longest reply the server writes when a request names no max_tokens (the
    # agent's requests don't). Its default, 512, cuts a thinking model off mid-thought: in the 27B's run of
    # 2026-11-19, 43 of 128 replies, and each failed attempt ended on three in a row. Shown in the tables when set.
    limit = os.environ.get("SOLVETIME_MLX_MAX_TOKENS", "")
    limit = str(int(limit)) if limit else ""
    return Engine("mlx-lm", model, path, [py, "-m", "mlx_lm.server", "--model", path, "--port", str(port),
                                          "--prompt-cache-size", "1", *(["--max-tokens", limit] if limit else [])],
                  port, path, version_command=[py, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
                  pack_dir=path if os.path.isdir(path) else None, info=_files_info,
                  settings=f"max tokens {limit}" if limit else "")


def _gguf_info(engine: Engine) -> dict:
    """A GGUF's size, and its bits per weight from the tensor sizes its header lists (vision tower excluded)."""
    size = engine.pack_bytes()
    return {"pack_bytes": size, "weights_bytes": size, "bits_per_weight": bits(size, engine.model)}


def llama_cpp(model: str, port: int) -> Engine:
    """llama.cpp's llama-server on a GGUF (SOLVETIME_GGUF), all layers on the GPU, flash attention, the model's
    own chat template, plain decoding. SOLVETIME_LLAMA_SPEC=draft-mtp drafts with the model's MTP head, for a GGUF
    that carries one (the Qwen3.6-35B-A3B UD-IQ2_XXS GGUF of the first results does not). SOLVETIME_LLAMA_LORA: a
    LoRA adapter GGUF applied on top (`--lora`), e.g. harness-hub's pi adapter for Qwen3.8-27B; shown in the tables.
    The window (SOLVETIME_LLAMA_CTX) is shown when it is not the default. SOLVETIME_LLAMA_CACHE_RAM: the RAM prompt
    cache in MiB (`--cache-ram`; 0 turns it off). Its default, 8192, keeps whole earlier conversation states (2.0-2.6
    GB each for Qwen3.8-27B at 16K) beside the model, which an agent's one conversation does not use: on a 24 GB
    Mac it pushed a 14.35 GB model into 9.5 GB of swap and critical pressure (2026-11-20)."""
    gguf = build(model, "llama.cpp")
    ctx = os.environ.get("SOLVETIME_LLAMA_CTX", "65536")
    command = ["llama-server", "-m", gguf, "--port", str(port), "--host", "127.0.0.1", "-ngl", "999", "-fa", "on",
               "--jinja", "-c", ctx]
    spec = os.environ.get("SOLVETIME_LLAMA_SPEC", "none")
    if spec != "none":
        command += ["--spec-type", spec]
    lora = os.environ.get("SOLVETIME_LLAMA_LORA", "")
    if lora:
        if not os.path.isfile(lora):
            raise ValueError(f"SOLVETIME_LLAMA_LORA={lora}: no such file")
        command += ["--lora", lora]
    cache_ram = os.environ.get("SOLVETIME_LLAMA_CACHE_RAM", "")
    if cache_ram:
        command += ["--cache-ram", str(int(cache_ram))]
    settings = [f"context {ctx}"] if ctx != "65536" else []
    settings += [("prompt cache off" if int(cache_ram) == 0 else f"prompt cache {int(cache_ram)} MiB")] if cache_ram else []
    settings += [f"speculation {spec}"] if spec != "none" else []
    settings += [f"lora {os.path.basename(lora)}"] if lora else []
    return Engine("llama.cpp", model, os.path.basename(gguf) or "(set SOLVETIME_GGUF)", command, port,
                  os.path.basename(gguf), health_path="/health", version_command=["llama-server", "--version"],
                  pack_dir=gguf or None, info=_gguf_info, settings=", ".join(settings))


def _ollama_models() -> str:
    return os.environ.get("OLLAMA_MODELS", os.path.join(HOME, ".ollama", "models"))


def _ollama_manifest(tag: str) -> str:
    name, _, version = tag.partition(":")
    if "/" not in name:
        name = "library/" + name
    return os.path.join(_ollama_models(), "manifests", "registry.ollama.ai", name, version or "latest")


def _ollama_info(engine: "Engine") -> dict:
    """The tag's size (its blobs, from the manifest) and bits per weight."""
    try:
        manifest = json.load(open(_ollama_manifest(engine.api_model)))
        size = sum(layer.get("size", 0) for layer in manifest.get("layers", [])
                   if layer.get("mediaType", "").endswith(".model"))
    except (OSError, ValueError):
        return {}
    return {"pack_bytes": size, "weights_bytes": size, "bits_per_weight": bits(size, engine.model)}


def ollama(model: str, port: int) -> Engine:
    """A dedicated `ollama serve` on its own port, over the user's model store, so the desktop app's settings
    and loaded models stay out. Its OpenAI endpoint reads temperature, top_p and presence_penalty from a request
    but not top_k or min_p: those come from the tag's own parameters (top_k 20 and min_p 0 for these tags, as
    the model card says). Thinking is turned off with reasoning_effort "none", not chat_template_kwargs. The
    27B tag drafts with its MTP head as shipped (draft_num_predict 4). The tag as Ollama ships it (SOLVETIME_OLLAMA_TAG for another). Its context
    window: 65536 tokens, as llama.cpp gets here; Ollama's own default depends on the GPU's memory and can be
    4096, which an agent's conversation outgrows without an error (the oldest messages are cut). The tag is
    Ollama's own default for the model (ollama.com/library): its q4_K_M GGUF, as `ollama run` pulls it."""
    exe = os.environ.get("SOLVETIME_OLLAMA", shutil.which("ollama") or "ollama")
    tag = build(model, "ollama")
    ctx = os.environ.get("SOLVETIME_OLLAMA_CTX", "65536")
    return Engine("ollama", model, tag, [exe, "serve"], port, tag, health_path="/api/version",
                  env={"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_CONTEXT_LENGTH": ctx, "OLLAMA_NOPRUNE": "1",
                       "OLLAMA_KEEP_ALIVE": "-1"},
                  # "ollama version is X" with a server running, "Warning: client version is X" without
                  version_command=["sh", "-c", f"{shlex.quote(exe)} --version 2>&1 | sed -n -E "
                                   "'s/.*(client )?version is /ollama /p' | head -1"],
                  info=_ollama_info, settings=f"context {ctx}",
                  clear_env="OLLAMA_", no_thinking={"reasoning_effort": "none"})


def _lmstudio_info(engine: "Engine") -> dict:
    """The loaded model's files: `lms ls` names its key, size and variant (e.g. qwen/qwen3.5-4b@4bit); its folder
    under ~/.lmstudio/models is the one named after the model and the variant (lmstudio-community/Qwen3.5-4B-MLX-4bit).
    Bits per weight from that folder's weights (vision tower excluded)."""
    lms = engine.env.get("SOLVETIME_LMS", "lms")
    try:
        listed = json.loads(subprocess.run([lms, "ls", "--json"], capture_output=True, text=True, timeout=60).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    entry = next((m for m in listed if isinstance(m, dict) and m.get("modelKey") == engine.pack), None)
    if not entry:
        return {}
    variant = (entry.get("selectedVariant") or "").partition("@")[2].lower()
    name = engine.pack.split("/")[-1].lower()
    root = os.path.join(HOME, ".lmstudio", "models")
    folders = [d for d in sorted(glob.glob(os.path.join(root, "*", "*"))) if os.path.isdir(d)
               and name in os.path.basename(d).lower() and (not variant or variant in os.path.basename(d).lower())]
    folder = folders[0] if folders else None
    weights = text_weight_bytes(folder) if folder else None
    return {"pack": engine.pack + (f" ({os.path.relpath(folder, root)})" if folder else ""),
            "pack_bytes": entry.get("sizeBytes"), "weights_bytes": weights,
            "bits_per_weight": bits(weights or entry.get("sizeBytes"), engine.model)}


# From LM Studio's files, not from `lms`: any lms command starts the daemon. The llmster release, and the
# runtime it picks for each model format with its version.
LMSTUDIO_VERSION = """
import json, os
home = os.path.expanduser("~/.lmstudio")
releases = sorted(os.listdir(os.path.join(home, "llmster"))) if os.path.isdir(os.path.join(home, "llmster")) else []
try:
    prefs = json.load(open(os.path.join(home, ".internal", "backend-preferences-v2.json")))
    runtimes = [f"{name}@{prefs.get('versions', {}).get(name, '?')} ({fmt})" for fmt, name in prefs.get("formatDefaults", {}).items()]
except (OSError, ValueError):
    runtimes = []
print("lmstudio llmster " + (releases[-1] if releases else "?") + (": " + ", ".join(runtimes) if runtimes else ""))
"""


def lmstudio(model: str, port: int) -> Engine:
    """LM Studio's headless daemon (llmster) with the newest runtimes it offers, its staff pick for the model
    in its MLX 4-bit build from lmstudio-community (`lms get <key>@4bit --mlx`: with -y alone, `lms get` takes
    the variant it preselects for the hardware, and on a 24 GB Mac it also started the MLX 5-bit), one request at a time, a 65536-token window as the other engines get
    here. launchers/lmstudio.sh starts a fresh daemon, loads the model, starts the server, and stops them all
    when the run ends."""
    lms = os.environ.get("SOLVETIME_LMS", os.path.join(HOME, ".lmstudio", "bin", "lms"))
    key = build(model, "lmstudio")
    ctx = os.environ.get("SOLVETIME_LMSTUDIO_CTX", "65536")
    # SOLVETIME_LMSTUDIO_GUARDRAILS: its model-loading guardrail for this run (localModels.modelLoadingGuardrails in
    # the daemon's .internal/settings.json; the launcher sets it and puts the previous one back). llmster wrote
    # "relaxed" there on its first start, and "relaxed" refuses the 27B on a 24 GB Mac: it estimates 20.97 GiB, 1.4x
    # the model's 14.98 GiB of files, at 8K, 16K and 32K windows alike ("Confidence: LOW"); "off" loads it.
    guard = os.environ.get("SOLVETIME_LMSTUDIO_GUARDRAILS", "")
    launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launchers", "lmstudio.sh")
    return Engine("lmstudio", model, key, ["/bin/bash", launcher, str(port), key, "solvetime", ctx], port, "solvetime",
                  env={"SOLVETIME_LMS": lms, **({"SOLVETIME_LMSTUDIO_GUARDRAILS": guard} if guard else {})},
                  start_timeout=900,
                  version_command=[sys.executable, "-c", LMSTUDIO_VERSION],
                  info=_lmstudio_info, settings=f"context {ctx}" + (f", guardrails {guard}" if guard else ""),
                  # chat_template_kwargs is ignored; reasoning_effort "none" turns thinking off (probed on qwen3.5-4b)
                  no_thinking={"reasoning_effort": "none"})


def _hf_snapshot(repo: str) -> str | None:
    """A Hugging Face repo's local snapshot, if it is already downloaded (nothing is fetched)."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo, local_files_only=True)
    except Exception:
        return None


def omlx(model: str, port: int) -> Engine:
    """oMLX (`omlx serve`, headless) on the same MLX 4-bit weights as mlx-lm (SOLVETIME_OMLX_MODEL: a model folder),
    with its own defaults otherwise: memory guard "balanced", its paged SSD KV cache on (up to half the free disk:
    16 GB in the 27B's agent run on 2026-11-19), its default speculation. It loads
    models from a directory of model folders, so the folder is linked into a directory of its own.
    SOLVETIME_OMLX_MEMORY_GUARD: another tier (`--memory-guard`: safe, aggressive, off), shown in the tables.
    "balanced" refuses the 27B (15.70 GiB) on a 24 GB Mac: its ceiling was 13.2-15.5 GiB (2026-11-19).
    Each run gets a settings folder of its own (--base-path): oMLX saves its command-line flags into its
    settings.json, so with the user's ~/.omlx one run's flags (a tier, a port) would carry over to the next.
    The folder, SSD cache included, is removed when the run ends."""
    exe = os.environ.get("SOLVETIME_OMLX", shutil.which("omlx") or "omlx")
    src = build(model, "omlx")
    folder = os.path.realpath(src) if os.path.isdir(src) else (_hf_snapshot(src) or src)
    name = os.path.basename(src.rstrip("/"))
    models = os.path.join(tempfile.gettempdir(), f"solvetime-omlx-models-{port}")
    if os.path.isdir(folder):
        os.makedirs(models, exist_ok=True)
        link = os.path.join(models, name)
        if os.path.islink(link) and os.readlink(link) != folder:
            os.remove(link)
        if not os.path.exists(link):
            os.symlink(folder, link)
    guard = os.environ.get("SOLVETIME_OMLX_MEMORY_GUARD", "")
    if guard and guard not in ("safe", "balanced", "aggressive", "off"):
        raise ValueError(f"SOLVETIME_OMLX_MEMORY_GUARD={guard}: one of safe, balanced, aggressive, off")
    base = os.path.join(tempfile.gettempdir(), f"solvetime-omlx-base-{port}")
    shutil.rmtree(base, ignore_errors=True)   # its defaults, not what an earlier run saved
    os.makedirs(base)
    return Engine("omlx", model, src,
                  [exe, "serve", "--base-path", base, "--model-dir", models, "--headless", "--host", "127.0.0.1",
                   "--port", str(port), *(["--memory-guard", guard] if guard else [])],
                  port, name, env={"OMLX_PORT": str(port)}, clear_env="OMLX_",
                  settings=f"memory guard {guard}" if guard else "", scratch=[base],
                  version_command=[exe, "--version"], pack_dir=folder if os.path.isdir(folder) else None,
                  info=_files_info, start_timeout=900)


ENGINES = {"raceline": raceline, "kapitsa": kapitsa, "mtplx": mtplx, "mlx-lm": mlx_lm, "llama.cpp": llama_cpp,
           "ollama": ollama, "lmstudio": lmstudio, "omlx": omlx}

INSTALL = {
    "raceline": "pip install git+https://github.com/audenvenn/raceline (Apple M5 or later), then `raceline pull <model>`; "
                "or point SOLVETIME_RACELINE at its `raceline` command",
    "kapitsa": "brew install nace-ai/tap/kapitsa; or point SOLVETIME_KAPITSA at a kapitsa binary",
    "mtplx": "pip install mtplx; or point SOLVETIME_MTPLX at its `mtplx` command",
    "mlx-lm": "pip install mlx-lm beside solvetime; or set SOLVETIME_MLX_PYTHON to a Python that has it",
    "llama.cpp": "brew install llama.cpp",
    "ollama": "brew install ollama (or the Ollama app); then `ollama pull <tag>` with a server running",
    "lmstudio": "curl -fsSL https://lmstudio.ai/install.sh | bash (LM Studio's headless daemon, with `lms`), then "
                "`lms get <model>`; or point SOLVETIME_LMS at an `lms`",
    "omlx": "brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx; or point "
            "SOLVETIME_OMLX at an `omlx`",
}


def harness_version() -> str:
    """solvetime's own version: the package version, and the commit when it runs from a git checkout."""
    from . import __version__
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(["git", "-C", here, "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True)
    return __version__ + (f" ({r.stdout.strip()})" if r.returncode == 0 and r.stdout.strip() else "")


def downloaded(folder: str) -> bool:
    """A model folder whose download has finished: no partial file left (LM Studio writes
    `downloading_<name>.part` until a file is complete) and every shard its safetensors index lists."""
    files = os.listdir(folder)
    if any(f.startswith("downloading_") or f.endswith(".part") for f in files):
        return False
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = set(json.load(f).get("weight_map", {}).values())
        return bool(shards) and all(os.path.isfile(os.path.join(folder, s)) for s in shards)
    return any(f.endswith((".safetensors", ".gguf")) for f in files)


def missing(engine: Engine) -> str | None:
    """Why this engine cannot start here (not installed, no model file), with how to fix it; None if it can."""
    exe = engine.env.get("SOLVETIME_LMS", engine.command[0]) if engine.name == "lmstudio" else engine.command[0]
    if not (shutil.which(exe) or os.path.isfile(exe)):
        return f"{engine.name}: `{exe}` not found. Install it: {INSTALL[engine.name]}"
    if engine.name != "llama.cpp" and not build(engine.model, engine.name):
        return (f"{engine.name}: no build of {engine.model} is known for it yet; set {BUILD_VARIABLE[engine.name]} "
                "to one")
    if engine.name == "mlx-lm" and subprocess.run([exe, "-c", "import mlx_lm"], capture_output=True).returncode:
        return f"mlx-lm: {exe} cannot import mlx_lm. {INSTALL['mlx-lm']}"
    if engine.name == "ollama" and not os.path.isfile(_ollama_manifest(engine.api_model)):
        return f"ollama: {engine.api_model} is not pulled. `ollama pull {engine.api_model}` (with `ollama serve` running)"
    if engine.name == "lmstudio":   # from its files: any lms command starts the daemon
        name = engine.pack.split("/")[-1].lower()
        folders = [d for d in glob.glob(os.path.join(HOME, ".lmstudio", "models", "*", "*"))
                   if os.path.isdir(d) and name in os.path.basename(d).lower()]
        if not any(downloaded(d) for d in folders):
            return (f"lmstudio: {engine.pack} is not downloaded{' completely' if folders else ''}. "
                    f"`lms get {engine.pack}`")
    if engine.name == "mtplx":   # `mtplx pull` can stop half-way: the pack then lacks shards
        folder = engine.pack_dir
        if not (folder and os.path.isdir(folder) and downloaded(folder)):
            done = " completely" if folder and os.path.isdir(folder) else ""
            return f"mtplx: {engine.pack} is not downloaded{done}. `mtplx pull {engine.pack}`"
    if engine.name == "omlx" and not engine.pack_dir:
        return (f"omlx: {engine.pack} is not on this Mac. Download it (`hf download {engine.pack}`), or set "
                "SOLVETIME_OMLX_MODEL to an MLX model folder")
    if engine.name == "llama.cpp":
        gguf = os.environ.get("SOLVETIME_GGUF", "")
        if not os.path.isfile(gguf):
            return ("llama.cpp: set SOLVETIME_GGUF to the model's .gguf file"
                    + (f" ({gguf} does not exist)" if gguf else ""))
    return None
