"""Eval plumbing: put a recorded `rocm` on PATH for the cases that have one.

The grading machines carry no `rocm` binary, so an agent following the skill
stops at the install offer and never reads a diagnosis. For a case that has a
directory under `fixtures/`, `setup` writes a stub `rocm` into a temporary
directory, puts it first on PATH, and `teardown` takes it away again. A case
with no directory runs exactly as it did before: no `rocm` on PATH.

The stub replays recorded output and nothing else. It answers `--version`,
`fix --json` and `diagnose`, and refuses every other command with exit code 2,
so a replay cannot change a machine or send anything. The recordings are
generated from the real catalog by `cargo xtask eval-fixtures`, and CI fails
when they go stale; do not edit them by hand.

Stdlib only. No network, no subprocess, nothing written outside the temporary
directory.
"""

import os
import shlex
import shutil
import stat
import sys
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPLIES = "replies.json"

STUB = r"""import json
import sys
from pathlib import Path

replies = json.loads(
    (Path(__file__).resolve().parent / "replies.json").read_text(encoding="utf-8")
)
args = sys.argv[1:]
# Options that would make `diagnose` do something other than replay a diagnosis.
refused = {"--report", "--send", "--distro", "--model"}


def reply(entry):
    sys.stdout.buffer.write(entry["stdout"].encode("utf-8"))
    sys.stderr.buffer.write(entry["stderr"].encode("utf-8"))
    sys.exit(entry["rc"])


if args in (["--version"], ["-V"], ["version"]):
    reply(replies["version"])
elif args == ["fix", "--json"]:
    reply(replies["fix_json"])
elif args[:1] == ["diagnose"] and not refused & set(args):
    reply(replies["diagnose_json" if "--json" in args else "diagnose_text"])
sys.stderr.write(
    "error: this evaluation replays only `--version`, `fix --json` and "
    "`diagnose`; it runs nothing else\n"
)
sys.exit(2)
"""


def _write_launcher(directory: Path) -> None:
    """A `rocm` the agent's shell can find, which runs the stub."""
    python = sys.executable
    if os.name == "nt":
        (directory / "rocm.cmd").write_text(
            f'@echo off\r\n"{python}" "%~dp0rocm.py" %*\r\n', encoding="utf-8"
        )
        return
    launcher = directory / "rocm"
    launcher.write_text(
        f"#!/bin/sh\nexec {shlex.quote(python)} "
        f'{shlex.quote(str(directory / "rocm.py"))} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def setup(workspace, case, ctx):
    source = FIXTURES / case.id / REPLIES
    if not source.is_file():
        return {}
    directory = Path(tempfile.mkdtemp(prefix="rocm-replay-"))
    shutil.copyfile(source, directory / REPLIES)
    (directory / "rocm.py").write_text(STUB, encoding="utf-8")
    _write_launcher(directory)
    ctx["_replay"] = (str(directory), os.environ.get("PATH", ""))
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    return {}


def teardown(workspace, case, ctx):
    replay = ctx.pop("_replay", None)
    if replay is None:
        return
    directory, saved_path = replay
    os.environ["PATH"] = saved_path
    shutil.rmtree(directory, ignore_errors=True)
