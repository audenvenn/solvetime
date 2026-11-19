"""An engine that exits before it answers is recorded with the last error its log reports, not only an exit code
(LM Studio's launcher exits -15 when `lms load` refuses; the reason was only in engine.log)."""

import os
import subprocess
import sys
import tempfile
import unittest

from solvetime.run import last_error, wait_up

REFUSAL = ("\x1b[K\x1b[?25hError: Model loading was stopped due to insufficient system resources. Continuing to "
           "load the model would likely overload your system and cause it to freeze.")


class LoadReason(unittest.TestCase):
    def run_engine(self, *lines: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "engine.log")
            code = "import sys\nfor line in sys.argv[1:]:\n    print(line)\nsys.exit(1)\n"
            with open(log_path, "w") as log:
                proc = subprocess.Popen([sys.executable, "-c", code, *lines], stdout=log, stderr=subprocess.STDOUT)
                proc.wait(30)
                return wait_up("http://127.0.0.1:9/v1/models", proc, 30, log_path)

    def test_the_reason_is_kept(self):
        problem = self.run_engine("llmster started (PID: 1).", "### lms load qwen/qwen3.8-27b", REFUSAL)
        self.assertEqual(problem, "exited with code 1: Error: Model loading was stopped due to insufficient system "
                                  "resources. Continuing to load the model would likely overload your system and "
                                  "cause it to freeze.")

    def test_no_error_line(self):
        self.assertEqual(self.run_engine("starting", "stopping"), "exited with code 1")

    def test_the_last_error_wins_and_is_short(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "engine.log")
            with open(p, "w") as f:
                f.write("error: the first\r\nprogress 10%\rprogress 20%\nFATAL: " + "x" * 600 + "\n")
            self.assertEqual(last_error(p), ("FATAL: " + "x" * 600)[:400])
        self.assertEqual(last_error(None), "")
        self.assertEqual(last_error("/nonexistent/engine.log"), "")


if __name__ == "__main__":
    unittest.main()
