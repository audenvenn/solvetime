"""The agent's commands run in a macOS sandbox (sandbox-exec), with a minimal environment.

Model-written commands run on the host, as the user, on a Mac that holds
other projects, credentials and models. So each command runs under a
Seatbelt profile that:

- allows writes only in the attempt's directory and its own TMPDIR;
- denies reading the home directory, except the Python that runs the tasks,
  so other projects (solvetime's own hidden checks among them), credentials,
  caches and the user's files stay out of reach;
- denies all network access, so nothing can be looked up or sent.

The environment is minimal too: a PATH with that Python first, HOME, a
locale, TMPDIR, and the agent config's own variables; nothing from the
harness process (no tokens) reaches the commands.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile

from minisweagent.environments.local import LocalEnvironment, _run

HOME = os.path.realpath(os.path.expanduser("~"))
TASK_VENV = os.environ.get("SOLVETIME_TASK_VENV", os.path.join(HOME, "venvs", "solvetime-tasks"))


def task_python() -> str:
    """The bare Python (no pytest, nothing installed) the tasks are written for; created on first use."""
    py = os.path.join(TASK_VENV, "bin", "python3")
    if not os.path.exists(py):
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", "-q", TASK_VENV, "--python", "3.12"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", TASK_VENV], check=True)
    return py


def _python_roots() -> list[str]:
    """Where the task Python and its standard library live (they must stay readable): the venv, every
    directory its interpreter's symlink chain passes through, and the interpreter's base prefix. For a uv
    Python that is uv's whole Python store (interpreters only, nothing private)."""
    py = task_python()
    roots = {os.path.realpath(TASK_VENV)}
    path = py
    for _ in range(16):                      # follow the chain link by link: each link must be readable
        roots.add(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        if not os.path.islink(path):
            break
        target = os.readlink(path)
        path = target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target)
    out = subprocess.run([py, "-c", "import sys; print(sys.base_prefix)"], capture_output=True, text=True).stdout
    if out.strip():
        roots.add(os.path.realpath(out.strip()))
    uv_store = os.path.join(HOME, ".local", "share", "uv", "python")
    if any(r.startswith(uv_store) for r in roots):
        roots = {r for r in roots if not r.startswith(uv_store)} | {uv_store}
    return sorted(roots)


def _q(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(writable: list[str]) -> str:
    writes = "\n".join(f"  (subpath {_q(os.path.realpath(p))})" for p in writable)
    reads = "\n".join(f"  (subpath {_q(p)})" for p in _python_roots())
    parents = sorted({os.path.dirname(p) for p in _python_roots()} | {HOME})
    literals = "\n".join(f"  (literal {_q(p)})" for p in parents)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
{writes}
  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper")
  (regex #"^/dev/tty") (regex #"^/dev/fd/"))
(deny file-read* (subpath {_q(HOME)}))
(allow file-read*
{reads}
{literals})
"""


class Sandbox:
    """One attempt's directories: `work` (the agent's), `tmp` (its TMPDIR) and the profile, kept where the
    agent cannot write it."""

    def __init__(self):
        self.base = tempfile.mkdtemp(prefix="solvetime-")
        self.work = os.path.join(self.base, "work")
        self.tmp = os.path.join(self.base, "tmp")
        os.makedirs(self.work)
        os.makedirs(self.tmp)
        self.harness = tempfile.mkdtemp(prefix="solvetime-profile-")
        self.profile_path = os.path.join(self.harness, "profile.sb")
        with open(self.profile_path, "w") as f:
            f.write(profile([self.work, self.tmp]))
        bin_dir = os.path.dirname(task_python())
        self.env = {"PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": HOME, "LANG": "en_US.UTF-8",
                    "LC_ALL": "en_US.UTF-8", "TMPDIR": self.tmp + "/", "TERM": "dumb"}

    def command(self, shell_command: str) -> str:
        """The shell command line that runs `shell_command` in the sandbox."""
        return f"/usr/bin/sandbox-exec -f {shlex.quote(self.profile_path)} /bin/bash -c {shlex.quote(shell_command)}"

    def run_python(self, code: str, timeout: int) -> subprocess.CompletedProcess:
        """Harness code (a hidden check) in the sandbox: it imports the agent's files, so it runs what the agent
        wrote. Isolated mode, with the directory appended to sys.path: a file the agent created cannot shadow a
        standard module the check imports."""
        code = "import sys\nsys.path.append('')\n" + code
        return subprocess.run(["/usr/bin/sandbox-exec", "-f", self.profile_path, task_python(), "-I", "-c", code],
                              cwd=self.work, env=self.env, capture_output=True, text=True, timeout=timeout,
                              start_new_session=True)

    def close(self):
        shutil.rmtree(self.base, ignore_errors=True)
        shutil.rmtree(self.harness, ignore_errors=True)


class SandboxedEnvironment(LocalEnvironment):
    """mini-swe-agent's LocalEnvironment, each command in the attempt's sandbox with its minimal environment."""

    def __init__(self, sandbox: Sandbox, **kwargs):
        super().__init__(cwd=sandbox.work, **kwargs)
        self.sandbox = sandbox
        self.env = dict(sandbox.env, **self.config.env)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        try:
            result = _run(self.sandbox.command(command), cwd or self.sandbox.work, self.env,
                          timeout or self.config.timeout)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw = getattr(e, "output", None)
            raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            output = {"output": raw, "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {e}",
                      "extra": {"exception_type": type(e).__name__, "exception": str(e)}}
        self._check_finished(output)
        return output
