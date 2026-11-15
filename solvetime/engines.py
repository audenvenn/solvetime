"""Engine adapters: how to start each engine's server with its own recommended build of a model.

Each engine runs the build its users get on this Mac: the one it picks by
itself or its documentation recommends for the memory at hand. The build's
name, bits and size go into every result. An engine that cannot load the
model here is a result too ("does not fit").

Models are named by the original checkpoint: qwen3.6-35b-a3b, qwen3.8-27b,
qwen3.5-9b. `pack()` maps that name to the engine's own build.
"""

from __future__ import annotations

import os
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
            return (out.stdout or out.stderr).strip().splitlines()[-1][:200]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            return ""

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


def raceline(model: str, port: int) -> Engine:
    alias = {"qwen3.6-35b-a3b": "qwen3.6-35b-a3b", "qwen3.8-27b": "qwen3.8-27b", "qwen3.5-9b": "qwen3.5-9b"}[model]
    exe = os.environ.get("SOLVETIME_RACELINE", shutil.which("raceline") or "raceline")
    target = os.environ.get("SOLVETIME_RACELINE_FILE", alias)   # a pulled model's name, or its .rlm file
    return Engine("raceline", model, f"{alias} (FP4, 4.0 bits, with its MTP head)", [exe, "serve", target, "--port", str(port)],
                  port, alias, version_command=[os.path.join(os.path.dirname(exe), "python"), "-c",
                                                "import raceline; print('raceline', raceline.__version__)"],
                  pack_dir=os.environ.get("SOLVETIME_RACELINE_FILE"), ours=True)


def kapitsa(model: str, port: int) -> Engine:
    key = {"qwen3.6-35b-a3b": "35b", "qwen3.8-27b": "27b", "qwen3.5-9b": "9b"}[model]
    return Engine("kapitsa", model, f"kapitsa {key} (the variant `kapitsa doctor` plans for this Mac)",
                  ["kapitsa", "serve", "--model", key, "--port", str(port), "--private", "--local"],
                  port, {"35b": "qwen3.6-35b-a3b", "27b": "qwen3.8-27b", "9b": "qwen3.5-9b"}[key],
                  version_command=["kapitsa", "--version"])


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
                  pack_dir=os.path.join(HOME, ".mtplx/models", repo.replace("/", "--")))


MLX_PACKS = {"qwen3.5-9b": "mlx-community/Qwen3.5-9B-4bit", "qwen3.8-27b": "mlx-community/Qwen3.8-27B-4bit",
             "qwen3.6-35b-a3b": "mlx-community/Qwen3.6-35B-A3B-4bit"}


def mlx_lm(model: str, port: int) -> Engine:
    path = os.environ.get("SOLVETIME_MLX_MODEL", MLX_PACKS[model])
    py = os.environ.get("SOLVETIME_MLX_PYTHON", "python3")
    return Engine("mlx-lm", model, path, [py, "-m", "mlx_lm.server", "--model", path, "--port", str(port)],
                  port, path, version_command=[py, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
                  pack_dir=path if os.path.isdir(path) else None)


ENGINES = {"raceline": raceline, "kapitsa": kapitsa, "mtplx": mtplx, "mlx-lm": mlx_lm}
