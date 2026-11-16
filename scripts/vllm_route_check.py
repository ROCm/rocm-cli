#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Check that vLLM has an install route for the SDKs an unpinned install picks.

Runs ``rocm install sdk --channel C --family F --dry-run`` for the release and
the nightly channel on each Instinct arch (the targets whose default engine is
vLLM) and fails when the ROCm SDK version the plan selects has neither a
discovery row nor a static pin in the vLLM engine's tables. Without a route,
``rocm engines install vllm`` refuses the runtime, and every vLLM-default GPU
lane runs no scenarios on it (the nightly queue lanes did exactly that on the
10.3.0a nightly SDKs).

The tables are read from the engine's own source, so this cannot drift from
them, and the matching mirrors the engine's: a discovery row matches on
``major.minor``, a static pin on ``major.minor.patch``, both ignoring any
``aYYYYMMDD`` suffix. Nothing is installed and no GPU is needed.

It resolves the live indexes, so it can go red with no change in the PR: an
index that moved on to an SDK vLLM has no wheels for. The workflow that runs it
is advisory for that reason. Add the missing row, or wait for the wheels.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DISCOVER_FILE = Path("engines/vllm/src/discover_builds.rs")
STATIC_FILE = Path("engines/vllm/src/wheel_builds.rs")
CHANNELS = ("release", "nightly")
# Exact arches, because the next layout selects its device payload by arch and
# refuses a grouped family. gfx942 is MI300X/MI325X, gfx950 is MI350/MI355.
FAMILIES = ("gfx942", "gfx950")
DRY_RUN_TIMEOUT_SECS = 300


class CheckError(Exception):
    """The check could not run to a verdict."""


def read_routes(discover_text: str, static_text: str) -> dict[str, set]:
    """The ROCm SDK lines the vLLM engine installs for, from its two tables.

    ``discover`` holds ``(major, minor)`` lines and ``static`` holds
    ``(major, minor, patch)`` releases. Only the code above each file's test
    module is read, so a test fixture cannot count as a route.
    """

    def literals(text: str) -> list[tuple[int, ...]]:
        text = text.split("#[cfg(test)]", 1)[0]
        return [
            tuple(int(part) for part in found.split("."))
            for found in re.findall(r'rocm_sdk_version:\s*"(\d+\.\d+\.\d+)"', text)
        ]

    discover = {version[:2] for version in literals(discover_text)}
    static = set(literals(static_text))
    if not discover or not static:
        raise CheckError(
            "could not read the vLLM build tables; VLLM_ROCM_DISCOVER_BUILD_TABLE "
            "or VLLM_ROCM_BUILD_TABLE was renamed or moved"
        )
    return {"discover": discover, "static": static}


def route_problem(version: str, routes: dict[str, set]) -> str | None:
    """Why vLLM cannot be installed on ``version``; ``None`` when it can."""
    found = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not found:
        return f"selected version {version!r} is not a ROCm release version"
    triple = tuple(int(part) for part in found.groups())
    if triple[:2] in routes["discover"] or triple in routes["static"]:
        return None
    return (
        f"vLLM has no install route for ROCm SDK {version}: no discovery row for "
        f"{triple[0]}.{triple[1]} and no static pin for "
        f"{triple[0]}.{triple[1]}.{triple[2]}, so `rocm engines install vllm` "
        "refuses that runtime"
    )


def selected_version(output: str) -> str | None:
    """The ``selected_rocm_version`` line of a dry-run plan."""
    for line in output.splitlines():
        match = re.match(r"^\s+selected_rocm_version: (\S+)\s*$", line)
        if match:
            return match.group(1)
    return None


def probe(rocm: str, channel: str, family: str, routes: dict[str, set]) -> str | None:
    """Resolve one channel and family; return a failure, or ``None`` when fine."""
    label = f"{channel}/{family}"
    command = [
        rocm, "install", "sdk", "--channel", channel, "--family", family,
        "--dry-run",
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DRY_RUN_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"{label}: dry run did not finish in {DRY_RUN_TIMEOUT_SECS}s"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return f"{label}: dry run failed ({result.returncode}): {' | '.join(tail)}"
    version = selected_version(result.stdout)
    if version is None:
        return f"{label}: the plan has no selected_rocm_version line"
    problem = route_problem(version, routes)
    if problem:
        return f"{label}: {problem}"
    print(f"{label}: ROCm SDK {version} has a vLLM route")
    return None


SAMPLE_DISCOVER = """
    VllmRocmDiscoverBuild {
        rocm_sdk_version: "10.0.0",
    },
    VllmRocmDiscoverBuild {
        rocm_sdk_version: "10.1.0",
    },
#[cfg(test)]
        rocm_sdk_version: "10.9.0",
"""
SAMPLE_STATIC = """
const VLLM_ROCM_BUILD_TABLE: &[VllmRocmBuild] = &[VllmRocmBuild {
    rocm_sdk_version: "7.2.3",
}];
"""


def self_test() -> None:
    """Each rule must reject what it exists for, and accept a covered version."""
    routes = read_routes(SAMPLE_DISCOVER, SAMPLE_STATIC)
    assert routes == {"discover": {(10, 0), (10, 1)}, "static": {(7, 2, 3)}}, routes
    assert route_problem("10.1.0", routes) is None
    assert route_problem("10.1.0a20261101", routes) is None, "suffix is ignored"
    assert route_problem("10.1.7", routes) is None, "a discovery row spans patches"
    assert route_problem("7.2.3a20260423", routes) is None
    assert route_problem("7.2.4", routes) is not None, "a static pin is exact"
    assert route_problem("10.9.0", routes) is not None, "test fixtures are no route"
    new_line = route_problem("10.3.0a20261115", routes)
    assert new_line and "10.3" in new_line, new_line
    assert route_problem("nightly", routes) is not None
    for broken in (("", SAMPLE_STATIC), (SAMPLE_DISCOVER, "")):
        try:
            read_routes(*broken)
        except CheckError:
            pass
        else:
            raise AssertionError("a renamed table must fail loudly, not pass")
    plan = "sdk install\n  channel: nightly\n  selected_rocm_version: 10.3.0a20261101\n"
    assert selected_version(plan) == "10.3.0a20261101"
    assert selected_version("sdk install\n") is None
    print("vllm route check self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rocm", help="path to the rocm binary to probe")
    parser.add_argument("--discover-file", type=Path, default=DISCOVER_FILE)
    parser.add_argument("--static-file", type=Path, default=STATIC_FILE)
    parser.add_argument("--channel", action="append", dest="channels")
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.rocm:
        parser.error("--rocm is required unless --self-test is given")
    routes = read_routes(
        args.discover_file.read_text(encoding="utf-8"),
        args.static_file.read_text(encoding="utf-8"),
    )
    failures = []
    for channel in args.channels or CHANNELS:
        for family in args.families or FAMILIES:
            failure = probe(args.rocm, channel, family, routes)
            if failure:
                print(f"::error::{failure}")
                failures.append(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as error:
        print(f"::error::{error}")
        sys.exit(2)
