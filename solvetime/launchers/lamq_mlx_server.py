"""Serve a LookAtMyQuant build through mlx-lm's own server, unchanged but for the function that loads the model.

mlx-lm's server loads its model with mlx_lm.utils.load. This sets that name, inside mlx_lm.server, to a function that
builds the model and tokenizer with LookAtMyQuant's runtime_mlx.load instead. For its Qwen builds that is an mlx-lm
model whose linear layers and fused Metal kernels are LookAtMyQuant's. The rest is mlx-lm's own:
- the chat template and the tool-call parser;
- sampling and the prompt cache;
- the HTTP API.

Run it with a Python that has both packages, and with mlx_lm.server's own arguments. solvetime's mlx-lm engine does so
when SOLVETIME_MLX_LOADER=lookatmyquant.
"""

import json
import sys
from pathlib import Path

import mlx_lm.server as server
from runtime_mlx import load as lookatmyquant_load


def load(path, adapter_path=None, tokenizer_config=None, return_config=False, **_):
    """mlx_lm.utils.load's interface, for a local folder that holds a LookAtMyQuant build."""
    if adapter_path is not None:
        raise SystemExit("lamq_mlx_server: a LookAtMyQuant build is served without adapters")
    # The build's own tokenizer files are read, so the server's tokenizer options must be its defaults.
    if (tokenizer_config or {}).get("chat_template") or (tokenizer_config or {}).get("trust_remote_code"):
        raise SystemExit("lamq_mlx_server: --chat-template and --trust-remote-code are not supported")
    model, tokenizer = lookatmyquant_load(path)
    if not return_config:
        return model, tokenizer
    return model, tokenizer, json.loads((Path(path) / "config.json").read_text())


server.load = load

if __name__ == "__main__":
    sys.exit(server.main())
