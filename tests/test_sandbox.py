"""The sandbox blocks what it must and lets the tasks work: python tests/test_sandbox.py"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solvetime.sandbox import HOME, Sandbox, SandboxedEnvironment  # noqa: E402
from solvetime.tasks import TASKS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class SandboxTest(unittest.TestCase):
    def setUp(self):
        os.environ["SOLVETIME_TEST_SECRET"] = "s3cret"
        self.box = Sandbox()
        self.env = SandboxedEnvironment(self.box, timeout=30)

    def tearDown(self):
        self.box.close()

    def sh(self, command):
        return self.env.execute({"command": command})

    def test_writes_only_in_the_attempt(self):
        self.assertEqual(self.sh("echo hi > inside.txt && cat inside.txt")["output"].strip(), "hi")
        probe = os.path.join(HOME, "solvetime-sandbox-probe")
        out = self.sh(f"echo x > {probe}")
        self.assertNotEqual(out["returncode"], 0)
        self.assertFalse(os.path.exists(probe))
        out = self.sh("echo x > /private/tmp/solvetime-sandbox-probe")
        self.assertNotEqual(out["returncode"], 0)
        self.assertFalse(os.path.exists("/private/tmp/solvetime-sandbox-probe"))

    def test_home_is_unreadable(self):
        paths = [os.path.join(HERE, "..", "solvetime", "tasks.py"), os.path.join(HOME, ".ssh"),
                 os.path.join(HOME, ".cache", "huggingface", "token"), os.path.join(HOME, ".config"),
                 os.path.join(HOME, "Library", "Keychains"), os.path.join(HOME, "project"), os.path.join(HOME, "Documents")]
        for path in [p for p in paths if os.path.exists(p)]:    # a missing path says "no such file" either way
            out = self.sh(f"cat {os.path.realpath(path)} || ls {os.path.realpath(path)}")
            self.assertNotEqual(out["returncode"], 0, path)
            self.assertIn("not permitted", out["output"].lower(), path)

    def test_home_and_the_way_to_the_task_python_cannot_be_listed(self):
        # The directories between the home directory and the task Python stay traversable (metadata only),
        # so that the interpreter finds itself; their contents cannot be listed.
        from solvetime.sandbox import TASK_VENV
        d, dirs = os.path.dirname(os.path.realpath(TASK_VENV)), []
        while d.startswith(HOME):
            dirs.append(d)
            d = os.path.dirname(d)
        for path in dirs:
            out = self.sh(f"ls {path}")
            self.assertNotEqual(out["returncode"], 0, path)
            self.assertIn("not permitted", out["output"].lower(), path)

    def test_no_network(self):
        out = self.sh("python3 -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)\"")
        self.assertNotEqual(out["returncode"], 0)
        out = self.sh("curl -s -m 3 http://127.0.0.1:8000/ ; echo rc=$?")
        self.assertNotIn("rc=0", out["output"])

    def test_python_tempfiles_and_a_clean_environment(self):
        out = self.sh("python3 -c \"import sys, tempfile; print(sys.version_info[:2]); "
                      "f = tempfile.NamedTemporaryFile(); print('tmp ok')\"; env")
        self.assertEqual(out["returncode"], 0, out)
        self.assertIn("(3, 12)", out["output"])
        self.assertIn("tmp ok", out["output"])
        self.assertNotIn("s3cret", out["output"])
        self.assertNotIn("SOLVETIME_TEST_SECRET", out["output"])

    def test_a_hidden_check_runs_in_the_sandbox(self):
        task = TASKS["inventory"]
        for name, text in task["files"].items():
            with open(os.path.join(self.box.work, name), "w") as f:
                f.write(text)
        code = f"ORIGINAL_FILES = {task['files']!r}\n{task['check']}"
        self.assertNotEqual(self.box.run_python(code, 60).returncode, 0)   # the bug is still there
        fixed = task["files"]["inventory.py"].replace('sum(item["price"] for', 'sum(item["price"] * item["quantity"] for')
        fixed = fixed.replace('if item["quantity"] < 0:', 'if item["quantity"] <= 0:')
        with open(os.path.join(self.box.work, "inventory.py"), "w") as f:
            f.write(fixed)
        r = self.box.run_python(code, 60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_a_file_named_like_a_module_does_not_shadow_it(self):
        with open(os.path.join(self.box.work, "json.py"), "w") as f:
            f.write("raise SystemExit('shadowed')\n")
        r = self.box.run_python("import json\nprint(json.dumps([1]))", 30)
        self.assertEqual(r.stdout.strip(), "[1]", r.stderr)


if __name__ == "__main__":
    unittest.main()
