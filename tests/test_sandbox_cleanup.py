"""Nothing an attempt started outlives it (2026-11-22).

Round 4's bisect 2 hidden check timed out on an endless loop, and so did four replays of it by validate. Each time,
subprocess.run killed its direct child only. That left 13 processes running in their workspaces for hours, each busy
on a core, through the measurements that came after."""

import os
import subprocess
import sys
import time
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solvetime.sandbox import Sandbox, kill_inside, run_in_group  # noqa: E402


def running(marker: str) -> bool:
    return subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 0


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox()
        self.marker = f"solvetime-test-{uuid.uuid4().hex[:12]}"   # in the child's command line only

    def tearDown(self):
        self.box.close()
        subprocess.run(["pkill", "-KILL", "-f", self.marker], capture_output=True)

    def sleeper(self) -> str:
        return f"import time; time.sleep(120)  # {self.marker}"

    def test_a_timed_out_check_leaves_nothing_running(self):
        # the check starts a test runner of its own, as bisect's did, and then hangs
        code = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {self.sleeper()!r}])\ntime.sleep(120)"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.box.run_python(code, timeout=3)
        time.sleep(0.5)
        self.assertFalse(running(self.marker))

    def test_what_a_check_leaves_behind_is_ended(self):
        code = f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {self.sleeper()!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        self.assertEqual(self.box.run_python(code, timeout=30).returncode, 0)
        time.sleep(0.5)
        self.assertFalse(running(self.marker))

    def test_a_timed_out_replayed_command_leaves_nothing_running(self):
        command = self.box.command(f"python3 -c '{self.sleeper()}' & sleep 120")
        with self.assertRaises(subprocess.TimeoutExpired):
            run_in_group(command, shell=True, cwd=self.box.work, env=self.box.env, timeout=3)
        time.sleep(0.5)
        self.assertFalse(running(self.marker))

    def test_closing_a_sandbox_ends_what_left_its_group(self):
        # a process in its own session, as `setsid` or a daemon would make it: only its working directory gives it away
        proc = subprocess.Popen(self.box.command(f"exec python3 -c '{self.sleeper()}'"), shell=True,
                                cwd=self.box.work, env=self.box.env, start_new_session=True)
        time.sleep(1)
        self.assertTrue(running(self.marker))
        self.assertGreaterEqual(kill_inside(self.box.base), 1)
        proc.wait(timeout=10)
        self.assertFalse(running(self.marker))

    def test_output_and_exit_codes_as_subprocess_run(self):
        r = run_in_group(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], timeout=10)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (3, "out\n", "err\n"))


if __name__ == "__main__":
    unittest.main()
