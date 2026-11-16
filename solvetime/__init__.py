"""solvetime: how long a Mac takes to finish coding tasks with a local model."""

import os as _os

__version__ = "0.1.0.dev0"

# mini-swe-agent loads a global .env from the user's config directory when it is imported. A benchmark must
# not pick up the user's own mini-swe-agent settings, so it gets an empty directory of solvetime's own, and
# no start-up banner.
_os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = _os.path.join(_os.path.expanduser("~"), ".cache", "solvetime", "mini-swe-agent")
_os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
