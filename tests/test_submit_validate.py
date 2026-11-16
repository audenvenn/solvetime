"""submit's scrubbing and validate's replay, on hand-made trajectories (no engine)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solvetime.submit import HOME, scrubber, targets  # noqa: E402
from solvetime.tasks import TASKS  # noqa: E402
from solvetime.validate import commands, disk_scan, original_work_dir, replay, settings_problems  # noqa: E402


def call(cmd):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": json.dumps({"command": cmd})}, "id": "c", "type": "function"}]}


class Scrub(unittest.TestCase):
    def test_what_could_name_the_user_goes_and_task_data_stays(self):
        scrub = scrubber()
        out = scrub(f"{HOME}/projects/x, someone@mail.com and first+tag@example.co.uk")
        self.assertNotIn(HOME, out)
        self.assertIn("~/projects/x", out)
        self.assertIn("someone@mail.com", out)               # email-shaped task data stays as it was
        for old, new in targets():                           # the user's own (git email, full name, host)
            self.assertNotIn(old, scrub(f"before {old} after"))   # never printed: the test must not leak them

    def test_json_stays_valid(self):
        # A JSON string where an escape runs into an address ("\nx@y.io"): scrubbing the text would break
        # the escape; scrubbing value by value cannot.
        doc = json.dumps({"output": f"line\nx@y.io\n{HOME}/a", "n": [1, f"{HOME}"]})
        out = json.loads(scrubber()(doc, "t.json"))
        self.assertEqual(out["output"], "line\nx@y.io\n~/a")
        self.assertEqual(out["n"], [1, "~"])
        lines = scrubber()(doc + "\n" + doc + "\n", "t.jsonl").splitlines()
        self.assertEqual([json.loads(line)["n"] for line in lines], [[1, "~"], [1, "~"]])


class Replay(unittest.TestCase):
    def test_commands_and_work_dir(self):
        old = "/private/var/folders/ab/cd/T/solvetime-xyz_1/work"
        traj = {"messages": [call("pwd"), {"role": "tool", "content": json.dumps({"returncode": 0, "output": old})},
                             call("ls -la")]}
        self.assertEqual(commands(traj), ["pwd", "ls -la"])
        self.assertEqual(original_work_dir(traj), old)

    def test_disk_scans(self):
        self.assertTrue(disk_scan('find / -name "x.py" 2>/dev/null | head -20'))
        for cmd in ["find / -name x | xargs rm", "find / -name x; rm y", "find / -delete", "find . -name x",
                    "find ~ -type f > out.txt", "find / -name x -exec rm {} \\;"]:
            self.assertFalse(disk_scan(cmd), cmd)

    def test_a_fix_replays_as_solved_and_none_as_not(self):
        task = TASKS["bisect"]
        src = task["files"]["search.py"]
        # the fix the hidden check accepts, written the way an agent would, with an absolute path to its
        # original work directory that the replay has to map onto its own
        old = "/private/var/folders/ab/cd/T/solvetime-xyz_1/work"
        self.assertIn("lo, hi = 0, len(items) - 1", src)          # the task's bug
        fixed = src.replace("lo, hi = 0, len(items) - 1", "lo, hi = 0, len(items)")
        write = f"cat > {old}/search.py <<'EOF'\n{fixed}EOF"
        traj = {"messages": [call("pwd"), {"role": "tool", "content": json.dumps({"output": old})}, call(write),
                             call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]}
        self.assertEqual(replay(task, traj)[0], True)
        self.assertEqual(replay(task, {"messages": [call("ls")]})[0], False)

    def test_an_edit_right_after_an_import_is_seen(self):
        # Replayed commands come within milliseconds: a same-size edit right after an import would leave
        # Python's cached bytecode looking current. The replay must run the edited code.
        task = {"files": {"m.py": "X = 1\n"}, "check": "import m\nassert m.X == 2, m.X", "keep": ()}
        traj = {"messages": [call("python3 -c 'import m'"), call("sed -i '' 's/X = 1/X = 2/' m.py")]}
        self.assertEqual(replay(task, traj), (True, ""))

    def test_settings(self):
        meta = {"agent": "mini-swe-agent 2.4.6", "step_limit": 50, "wall_seconds": 900, "thinking": "thinking",
                "overrides_sent": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                   "presence_penalty": 0.0, "enable_thinking": True},
                "thinking_switch": "honoured", "tasks": ["bisect"]}
        self.assertEqual(settings_problems(meta), [])
        self.assertTrue(settings_problems(dict(meta, step_limit=100)))
        self.assertTrue(settings_problems(dict(meta, overrides_sent={"temperature": 0.0})))


if __name__ == "__main__":
    unittest.main()
