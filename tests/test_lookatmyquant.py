"""LookAtMyQuant builds through mlx-lm's own server:
- SOLVETIME_MLX_LOADER=lookatmyquant starts the launcher, which replaces only the server's loader;
- the tables show LookAtMyQuant's version;
- a LookAtMyQuant artifact's tensors count toward bits per weight."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from solvetime import engines


class Loader(unittest.TestCase):
    def engine(self, **env):
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(engines, "_lookatmyquant", return_value="0.3.0 (5106923)"):
            for k in ("SOLVETIME_MLX_LOADER", "SOLVETIME_MLX_MAX_TOKENS"):
                if k not in env:
                    os.environ.pop(k, None)
            return engines.mlx_lm("qwen3.8-27b", 8890)

    def test_mlx_lm_by_default(self):
        e = self.engine()
        self.assertEqual(e.command[1:3], ["-m", "mlx_lm.server"])
        self.assertEqual(e.settings, "")

    def test_the_launcher_and_its_settings(self):
        e = self.engine(SOLVETIME_MLX_LOADER="lookatmyquant", SOLVETIME_MLX_MAX_TOKENS="16384")
        self.assertEqual(e.name, "mlx-lm")
        self.assertEqual(e.command[1], engines.LOOKATMYQUANT_LAUNCHER)
        self.assertTrue(os.path.isfile(e.command[1]))
        self.assertIn("--prompt-cache-size", e.command)
        self.assertEqual(e.command[-2:], ["--max-tokens", "16384"])
        self.assertEqual(e.settings, "LookAtMyQuant runtime 0.3.0 (5106923), max tokens 16384")

    def test_an_unknown_loader(self):
        with self.assertRaises(ValueError):
            self.engine(SOLVETIME_MLX_LOADER="other")


class Launcher(unittest.TestCase):
    """The launcher sets mlx_lm.server.load and nothing else, tested with stand-ins for both packages."""

    def setUp(self):
        self.server = types.ModuleType("mlx_lm.server")
        self.server.load = "mlx-lm's own"
        self.server.main = lambda: 0
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.server = self.server
        runtime = types.ModuleType("runtime_mlx")
        runtime.load = lambda path: (f"model of {path}", "tokenizer")
        self.modules = mock.patch.dict(sys.modules, {"mlx_lm": mlx_lm, "mlx_lm.server": self.server,
                                                     "runtime_mlx": runtime})
        self.modules.start()
        spec = importlib.util.spec_from_file_location("lamq_mlx_server", engines.LOOKATMYQUANT_LAUNCHER)
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)
        self.folder = tempfile.mkdtemp()
        with open(os.path.join(self.folder, "config.json"), "w") as f:
            json.dump({"model_type": "qwen3_5"}, f)

    def tearDown(self):
        self.modules.stop()

    def test_only_the_loader_is_replaced(self):
        self.assertIs(self.server.load, self.launcher.load)

    def test_the_server_gets_model_tokenizer_and_config(self):
        got = self.server.load(self.folder, adapter_path=None, tokenizer_config={"trust_remote_code": False},
                               return_config=True, trust_remote_code=False)
        self.assertEqual(got, (f"model of {self.folder}", "tokenizer", {"model_type": "qwen3_5"}))

    def test_options_it_cannot_honour(self):
        with self.assertRaises(SystemExit):
            self.server.load(self.folder, adapter_path="adapters")
        with self.assertRaises(SystemExit):
            self.server.load(self.folder, tokenizer_config={"chat_template": "{{ messages }}"})


class PackedBytes(unittest.TestCase):
    def artifact(self, packed: dict) -> str:
        folder = tempfile.mkdtemp()
        with open(os.path.join(folder, "artifact.json"), "w") as f:
            json.dump(packed, f)
        return folder

    def test_its_tensors_count_but_not_a_vision_tower(self):
        folder = self.artifact({"format": "opgraph-packed-1", "blob": "weights.bin", "tensors": {
            "model.layers.0.mlp.down_proj.weight.b0.g0.codes": {"nbytes": 1000, "offset": 0},
            "lm_head.weight.b0.g0.codes": {"nbytes": 200, "offset": 1000},
            "model.visual.blocks.0.attn.qkv.weight": {"nbytes": 50, "offset": 1200}}})
        self.assertEqual(engines.text_weight_bytes(folder), 1200)

    def test_another_format_is_read_as_safetensors(self):
        self.assertIsNone(engines.text_weight_bytes(self.artifact({"format": "something-else", "tensors": {}})))


if __name__ == "__main__":
    unittest.main()
