#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate PowerShell script syntax for the pre-commit/prek hook.

Mirrors the syntax check CI runs on Windows, but works as a local hook on any
platform: it parses each ``*.ps1`` file with the PowerShell language parser and
fails if any parse errors are reported.

PowerShell is only available on some developer machines, so when neither
``pwsh`` nor ``powershell`` is found this hook skips cleanly (exit 0) rather than
blocking the commit. CI still enforces the check on Windows.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

# PowerShell snippet: parse the file given as the first argument and emit any
# syntax errors. Exits non-zero when the file does not parse.
_PARSE_SCRIPT = (
    "$errors = $null; "
    "[void][System.Management.Automation.Language.Parser]::ParseFile("
    "$args[0], [ref]$null, [ref]$errors); "
    "if ($errors) { $errors | ForEach-Object { "
    "Write-Output $_.ToString() }; exit 1 }"
)


def find_powershell() -> str | None:
    for exe in ("pwsh", "powershell"):
        path = shutil.which(exe)
        if path:
            return path
    return None


def main(argv: list[str]) -> int:
    files = argv[1:]
    if not files:
        return 0

    powershell = find_powershell()
    if powershell is None:
        print(
            "check_powershell_syntax: PowerShell (pwsh/powershell) not found; "
            "skipping. CI validates PowerShell syntax on Windows.",
            file=sys.stderr,
        )
        return 0

    failed = False
    for file in files:
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _PARSE_SCRIPT,
                file,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed = True
            print(f"{file}: PowerShell syntax errors:", file=sys.stderr)
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
