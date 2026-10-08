#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Run lychee for the `lychee` prek hook, skipping with a warning when absent.

A missing lychee must not block a commit: the `docs-links` CI job stays the
hard gate. When lychee is installed, its output is shown only on failure, so
the hook (which prek runs `verbose` to surface the skip warning) stays quiet
on a passing commit. Arguments are passed through to lychee unchanged.
"""

import shutil
import subprocess
import sys


def main() -> int:
    lychee = shutil.which("lychee")
    if lychee is None:
        print(
            "warning: lychee not found, skipping the markdown-link check; "
            "the docs-links CI job still runs it. Install the pinned version "
            "from CONTRIBUTING.md to check locally."
        )
        return 0
    result = subprocess.run(
        [lychee, *sys.argv[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        # Raw bytes: lychee's summary has emoji, which a cp1252 console
        # stdout (Windows) can't encode, so a text write would replace the
        # report with a UnicodeEncodeError.
        sys.stdout.flush()
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
