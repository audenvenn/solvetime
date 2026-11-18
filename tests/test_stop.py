"""A run ended with SIGTERM (by the memory watchdog) still runs its `finally`: it writes what it measured."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Stop(unittest.TestCase):
    def test_sigterm_runs_finally(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "written")
            code = (f"import sys, time; sys.path.insert(0, {HERE!r})\n"
                    "from solvetime import run\n"
                    "run.stop_on_term()\n"
                    "try:\n    print('ready', flush=True); time.sleep(60)\n"
                    f"finally:\n    open({out!r}, 'w').write(str(run.STOPPED))\n")
            p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
            self.assertEqual(p.stdout.readline().strip(), "ready")
            p.send_signal(signal.SIGTERM)
            p.wait(30)
            self.assertEqual(p.returncode, 128 + signal.SIGTERM)
            self.assertEqual(open(out).read(), "SIGTERM")


if __name__ == "__main__":
    unittest.main()
