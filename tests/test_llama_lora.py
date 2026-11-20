"""llama.cpp with a LoRA adapter on top (SOLVETIME_LLAMA_LORA): passed as --lora and shown in the run's settings,
with a non-default window and speculation shown too."""

import os
import tempfile
import unittest
from unittest import mock

from solvetime import engines


class LlamaLora(unittest.TestCase):
    def engine(self, **env):
        with mock.patch.dict(os.environ, env):
            for k in ("SOLVETIME_LLAMA_LORA", "SOLVETIME_LLAMA_CTX", "SOLVETIME_LLAMA_SPEC", "SOLVETIME_LLAMA_CACHE_RAM"):
                if k not in env:
                    os.environ.pop(k, None)
            return engines.llama_cpp("qwen3.8-27b", 8894)

    def test_the_default(self):
        e = self.engine(SOLVETIME_GGUF="/m/base.gguf")
        self.assertNotIn("--lora", e.command)
        self.assertEqual(e.settings, "")

    def test_a_lora_and_a_window(self):
        with tempfile.NamedTemporaryFile(suffix="-pi-lora-F16.gguf") as f:
            e = self.engine(SOLVETIME_GGUF="/m/base.gguf", SOLVETIME_LLAMA_LORA=f.name, SOLVETIME_LLAMA_CTX="16384")
            self.assertEqual(e.command[e.command.index("--lora") + 1], f.name)
            self.assertEqual(e.settings, f"context 16384, lora {os.path.basename(f.name)}")

    def test_the_prompt_cache_off(self):
        e = self.engine(SOLVETIME_GGUF="/m/base.gguf", SOLVETIME_LLAMA_CTX="16384", SOLVETIME_LLAMA_CACHE_RAM="0")
        self.assertEqual(e.command[e.command.index("--cache-ram") + 1], "0")
        self.assertEqual(e.settings, "context 16384, prompt cache off")
        e = self.engine(SOLVETIME_GGUF="/m/base.gguf", SOLVETIME_LLAMA_CACHE_RAM="1024")
        self.assertEqual(e.settings, "prompt cache 1024 MiB")

    def test_a_missing_lora(self):
        with self.assertRaises(ValueError):
            self.engine(SOLVETIME_GGUF="/m/base.gguf", SOLVETIME_LLAMA_LORA="/nonexistent/lora.gguf")


if __name__ == "__main__":
    unittest.main()
