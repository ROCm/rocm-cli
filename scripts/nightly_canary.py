#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Metadata-only canary for the canonical TheRock nightly stream.

Runs ``rocm install sdk --channel nightly --family F --dry-run`` for each
family and checks that the plan it prints still comes from the canonical
nightly stream: the source constants are read from the resolver's own source
file, so the canary cannot drift from them, and no build version is
hard-coded. Nothing is installed and no GPU is needed.

What it fails on:

* the dry run itself failing (unknown layout, redirect, no complete build);
* a channel, source layout generation or source URL other than the canonical
  nightly ones, which covers a legacy endpoint or the stable stream taking
  part in nightly resolution;
* a selected version that is not a dated nightly (``X.Y.ZaYYYYMMDD``);
* a selected build older than ``--max-age-days``, which is how a resolver that
  quietly keeps choosing a stale train shows up.

It also lists the ``rocm`` wheel versions the index publishes and reports, as a
warning rather than a failure, when a newer one exists than the one selected:
the resolver skips a build that is not yet complete, so a newer listing alone
is not a fault.

What it does not cover: the hardware install lane (a real GPU operation), and
whether a newer listed build is complete. See docs/testing.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_SOURCE_FILE = Path("apps/rocm/src/therock/source.rs")
# Instinct families (``-dcgpu``) are Linux-only, so a Windows run probing one
# would fail for a reason that is not drift.
DEFAULT_FAMILIES = ("gfx110X-all", "gfx120X-all") + (
    () if sys.platform == "win32" else ("gfx950-dcgpu",)
)
DEFAULT_MAX_AGE_DAYS = 3
NIGHTLY_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+a(?P<date>\d{8})$")
WHEEL_VERSION_RE = re.compile(r"rocm-(?P<version>\d+\.\d+\.\d+a\d{8})[-.]")
FETCH_TIMEOUT_SECS = 30
DRY_RUN_TIMEOUT_SECS = 300


class CanaryError(Exception):
    """The canary could not run to a verdict."""


def read_constants(source_text: str) -> dict[str, str]:
    """The canonical nightly base and layout generation, from the resolver source."""
    base = re.search(
        r'const THEROCK_NIGHTLY_PIP_INDEX_BASE: &str = "([^"]+)";', source_text
    )
    generation = re.search(
        r'const THEROCK_SOURCE_LAYOUT_GENERATION: &str = "([^"]+)";', source_text
    )
    if not base or not generation:
        raise CanaryError(
            "could not read the nightly index base and layout generation from "
            "the resolver source; the constants were renamed or moved"
        )
    return {"base": base.group(1).rstrip("/"), "generation": generation.group(1)}


def parse_plan(output: str) -> dict[str, str]:
    """The ``key: value`` lines of a dry-run plan."""
    plan: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"^\s+([a-z_]+): (.*\S)\s*$", line)
        if match:
            plan.setdefault(match.group(1), match.group(2))
    return plan


def check_plan(
    plan: dict[str, str],
    constants: dict[str, str],
    today: dt.date,
    max_age_days: int,
) -> list[str]:
    """Why this plan is not the canonical nightly one; empty when it is."""
    failures: list[str] = []
    if plan.get("channel") != "nightly":
        failures.append(f"channel is {plan.get('channel')!r}, expected 'nightly'")
    generation = plan.get("source_layout_generation")
    if generation != constants["generation"]:
        failures.append(
            f"source layout generation is {generation!r}, "
            f"expected {constants['generation']!r}"
        )
    for key in ("canonical_source", "index_url"):
        value = plan.get(key)
        if value is None:
            failures.append(f"the plan has no {key} line")
        elif not value.rstrip("/").startswith(constants["base"]):
            failures.append(
                f"{key} is {value!r}, which is not under the canonical "
                f"nightly stream {constants['base']!r}"
            )
    version = plan.get("selected_rocm_version", "")
    match = NIGHTLY_VERSION_RE.match(version)
    if not match:
        failures.append(
            f"selected version {version!r} is not a dated nightly (X.Y.ZaYYYYMMDD)"
        )
    else:
        built = dt.datetime.strptime(match.group("date"), "%Y%m%d").date()
        age = (today - built).days
        if age > max_age_days:
            failures.append(
                f"selected build {version} is {age} days old "
                f"(limit {max_age_days}); the stream may have moved on"
            )
    return failures


def newest_listed_version(listing_html: str) -> str | None:
    """The newest dated nightly ``rocm`` wheel version in an index listing."""
    versions = set(WHEEL_VERSION_RE.findall(listing_html))
    return max(versions, key=lambda v: v.rsplit("a", 1)[1], default=None)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECS) as response:
        return response.read().decode("utf-8", errors="replace")


def probe(rocm: str, family: str, constants: dict[str, str], max_age_days: int):
    """Resolve one family; return ``(failures, warnings)``."""
    command = [
        rocm, "install", "sdk", "--channel", "nightly",
        "--family", family, "--dry-run",
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
        return [f"{family}: dry run did not finish in {DRY_RUN_TIMEOUT_SECS}s"], []
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return [
            f"{family}: dry run failed ({result.returncode}): {' | '.join(tail)}"
        ], []
    plan = parse_plan(result.stdout)
    failures = [
        f"{family}: {problem}"
        for problem in check_plan(
            plan, constants, dt.datetime.now(dt.UTC).date(), max_age_days
        )
    ]
    warnings: list[str] = []
    index_url = plan.get("index_url")
    if index_url:
        try:
            newest = newest_listed_version(fetch(index_url.rstrip("/") + "/rocm/"))
        except OSError as error:
            warnings.append(f"{family}: could not read the index listing: {error}")
        else:
            selected = plan.get("selected_rocm_version", "")
            print(f"{family}: selected {selected}, newest listed {newest}")
            if newest and newest.rsplit("a", 1)[1] > selected.rsplit("a", 1)[-1]:
                warnings.append(
                    f"{family}: index lists {newest}, newer than the selected "
                    f"{selected}; fine if that build is still incomplete"
                )
    return failures, warnings


SAMPLE_PLAN = """sdk install
  channel: nightly
  canonical_source: https://rocm.nightlies.amd.com/whl-multi-arch
  selected_rocm_version: 7.14.0a20260601
  build_date: 2026-06-01
  source_layout_generation: multi-arch-v2
  format: wheel
  device_target: <undetermined>
  index_url: https://rocm.nightlies.amd.com/whl-multi-arch
  mode: dry-run
"""
SAMPLE_SOURCE = """
const THEROCK_NIGHTLY_PIP_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/whl-multi-arch";
pub(super) const THEROCK_SOURCE_LAYOUT_GENERATION: &str = "multi-arch-v2";
"""


def self_test() -> None:
    """Each rule must reject the plan it exists for, and accept a good one."""
    constants = read_constants(SAMPLE_SOURCE)
    assert constants == {
        "base": "https://rocm.nightlies.amd.com/whl-multi-arch",
        "generation": "multi-arch-v2",
    }, constants
    today = dt.date(2026, 6, 2)

    def failures(**overrides: str | None) -> list[str]:
        plan = parse_plan(SAMPLE_PLAN)
        for key, value in overrides.items():
            if value is None:
                plan.pop(key)
            else:
                plan[key] = value
        return check_plan(plan, constants, today, DEFAULT_MAX_AGE_DAYS)

    assert failures() == []
    assert any("channel" in f for f in failures(channel="release"))
    assert any(
        "layout generation" in f for f in failures(source_layout_generation="next-v1")
    )
    legacy = "https://rocm.nightlies.amd.com/v2/gfx110X-all"
    assert any("canonical_source" in f for f in failures(canonical_source=legacy))
    assert any("index_url" in f for f in failures(index_url=legacy))
    assert any("index_url" in f for f in failures(index_url=None))
    assert any("dated nightly" in f for f in failures(selected_rocm_version="7.14.1"))
    assert any(
        "days old" in f for f in failures(selected_rocm_version="7.14.0a20260520")
    )
    assert failures(selected_rocm_version="7.14.0a20260530") == []
    try:
        read_constants('const OTHER: &str = "x";')
    except CanaryError:
        pass
    else:
        raise AssertionError("a renamed constant must fail loudly, not pass")
    listing = (
        '<a href="rocm-7.14.0a20260601-py3-none-linux_x86_64.whl">x</a>'
        '<a href="rocm-7.14.0a20260603.tar.gz">y</a>'
    )
    assert newest_listed_version(listing) == "7.14.0a20260603"
    assert newest_listed_version("<html></html>") is None
    print("nightly canary self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rocm", help="path to the rocm binary to probe")
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.rocm:
        parser.error("--rocm is required unless --self-test is given")
    constants = read_constants(args.source_file.read_text(encoding="utf-8"))
    all_failures: list[str] = []
    for family in args.families or DEFAULT_FAMILIES:
        failures, warnings = probe(args.rocm, family, constants, args.max_age_days)
        for warning in warnings:
            print(f"::warning::{warning}")
        for failure in failures:
            print(f"::error::{failure}")
        if not failures:
            print(f"{family}: canonical nightly resolution ok")
        all_failures += failures
    return 1 if all_failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CanaryError as error:
        print(f"::error::{error}")
        sys.exit(2)
