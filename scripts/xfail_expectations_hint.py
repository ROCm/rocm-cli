#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Advisory PR-time hint: warn when a pull request that claims to fix a bug
leaves a stale known-failure marker behind.

The end-to-end suite records per-scenario expected failures ("xfail") in
``tests/e2e-cucumber/expectations.toml``, each keyed to a ticket via its ``bug``
field. When that bug is fixed the row goes stale, but the reconciler only
notices if the fix's CI lane happens to exercise the very platform/engine the
row is keyed to — otherwise the stale row lurks until an unrelated later PR
turns red with a mis-attributed XPASS.

This check closes that gap at authoring time: it collects every ticket id the
pull request references (title, body, commit messages) and, if any of them
still has a live xfail row, writes an advisory note to the CI step summary
asking the author to remove or narrow it. It never fails the build.

Ticket ids are matched in two shapes so the check keeps working as the project
opens to public GitHub issues:
  * project tracker style ``[A-Z]+-<n>`` (the ticket ids used today), and
  * GitHub issues ``#<n>`` — both bare and with a ``Fixes/Closes/Resolves``
    closing keyword.
Bare ``#<n>`` is matched deliberately. A bare ``#123`` sometimes means "PR 123"
rather than "issue 123", so this can occasionally nudge on an unrelated number.
That is acceptable because the check is advisory only: a stray nudge costs a
glance, whereas a missed stale marker costs a mis-attributed failure on an
unrelated pull request later.

Usage:
    xfail_expectations_hint.py [--title T] [--body B] [--commits-file F]
    xfail_expectations_hint.py --self-test
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

# Project tracker ids (e.g. ABC-123) and GitHub issue refs (#123). The GitHub
# pattern intentionally matches bare #123 as well as closing-keyword forms; see
# the module docstring for why higher recall is preferred for an advisory check.
_TICKET_PATTERNS = (
    re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"),
    re.compile(r"(?<![\w/])#\d+\b"),
)


def extract_ticket_refs(text: str) -> set[str]:
    """Return every ticket id referenced in ``text`` (both id shapes)."""
    refs: set[str] = set()
    for pattern in _TICKET_PATTERNS:
        refs.update(pattern.findall(text))
    return refs


def live_xfail_bugs(expectations_toml: str) -> dict[str, list[str]]:
    """Map each ticket id that has xfail rows to the affected scenario ids.

    Takes the *contents* of ``expectations.toml`` (an array-of-tables keyed by
    scenario id; each row carries a ``bug`` ticket reference). Ids are taken
    verbatim — no assumption about the tracker prefix.
    """
    data = tomllib.loads(expectations_toml)
    bugs: dict[str, list[str]] = {}
    for scenario_id, rows in data.items():
        for row in rows:
            bug = row.get("bug")
            if bug:
                bugs.setdefault(bug, []).append(scenario_id)
    return bugs


def find_stale_hits(
    referenced: set[str], bugs: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Tickets the PR references that still have live xfail rows."""
    return {
        bug: sorted(set(scenarios))
        for bug, scenarios in bugs.items()
        if bug in referenced
    }


def format_note(hits: dict[str, list[str]]) -> str:
    """Advisory markdown note naming each stale ticket and its scenarios."""
    lines = [
        "### Stale expected-failure marker?",
        "",
        "This pull request references bug(s) that still have expected-failure "
        "(xfail) rows in `tests/e2e-cucumber/expectations.toml`. If this PR "
        "fixes the bug, remove or narrow the matching row(s) so a later "
        "unrelated PR does not go red with a mis-attributed XPASS:",
        "",
    ]
    for bug in sorted(hits):
        scenarios = ", ".join(f"`{s}`" for s in hits[bug])
        lines.append(f"- **{bug}** — scenario(s): {scenarios}")
    return "\n".join(lines)


def collect_pr_text(title: str, body: str, commits: str) -> str:
    return "\n".join(part for part in (title, body, commits) if part)


def run_check(
    expectations_toml: str, title: str, body: str, commits: str
) -> dict[str, list[str]]:
    referenced = extract_ticket_refs(collect_pr_text(title, body, commits))
    return find_stale_hits(referenced, live_xfail_bugs(expectations_toml))


def emit(hits: dict[str, list[str]]) -> None:
    """Print the note; also append to the CI step summary when available."""
    if not hits:
        print("xfail hint: no referenced bug has a live xfail row.")
        return
    note = format_note(hits)
    print(note)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        # GITHUB_STEP_SUMMARY is a trusted path provided by the CI runner, not
        # user input.
        with open(summary, "a", encoding="utf-8") as fh:  # codeql[py/path-injection]
            fh.write(note + "\n")


# --- Self-test --------------------------------------------------------------
# Each case corresponds to a locked behavior scenario. These are the source of
# truth for the check's behavior.

_SELFTEST_EXPECTATIONS = """
[["serve-vllm-inference"]]
when = { effective_engine = "vllm" }
bug = "EAI-7333"
reason = "example"

[["serve-default-engine-inference"]]
when = {}
bug = "#123"
reason = "example"
"""


def run_self_test() -> None:
    def check(title="", body="", commits=""):
        return run_check(_SELFTEST_EXPECTATIONS, title, body, commits)

    # Scenario 1 - a fix that references a bug with a live xfail is warned,
    # naming that bug and the affected scenario.
    hits = check(body="Fixes EAI-7333 in the vllm path.")
    assert hits == {"EAI-7333": ["serve-vllm-inference"]}, hits
    note = format_note(hits)
    assert "EAI-7333" in note and "serve-vllm-inference" in note, note
    print("self-test: scenario 1 (tracker ref with live xfail warns) ok")

    # Scenario 1 (github issue form) - a closing-keyword issue ref is warned.
    hits = check(body="Fixes #123")
    assert hits == {"#123": ["serve-default-engine-inference"]}, hits
    print("self-test: scenario 1 (github 'Fixes #123' warns) ok")

    # Scenario 1 (bare issue ref) - bare #123 is matched too (recall > precision).
    hits = check(body="Follow-up to #123, same root cause.")
    assert hits == {"#123": ["serve-default-engine-inference"]}, hits
    print("self-test: scenario 1 (bare #123 warns) ok")

    # Scenario 2 - a referenced bug with no xfail row produces no warning.
    hits = check(body="Fixes EAI-9999, unrelated area.")
    assert hits == {}, hits
    print("self-test: scenario 2 (referenced bug, no xfail row) ok")

    # Scenario 3 - a PR referencing none of the tracked bugs is silent.
    hits = check(title="Refactor logging", body="No ticket.")
    assert hits == {}, hits
    print("self-test: scenario 3 (references nothing tracked) ok")

    # A '#123' embedded in a URL path must NOT be read as an issue ref.
    hits = check(body="see https://example.test/pulls/123 for context")
    assert hits == {}, hits
    print("self-test: url path is not a bare issue ref ok")

    # Refs may come from any of title, body, or commit messages.
    hits = check(commits="abc123 Fixes EAI-7333\n")
    assert hits == {"EAI-7333": ["serve-vllm-inference"]}, hits
    print("self-test: refs read from commit messages ok")

    print("xfail hint self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="", help="Pull request title.")
    parser.add_argument("--body", default="", help="Pull request body.")
    parser.add_argument(
        "--commits-file",
        type=Path,
        help="File containing commit messages (one range dump); optional.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in behavior checks instead of scanning a PR.",
    )
    return parser.parse_args()


# The expectations matrix lives at a fixed repo-relative path; it is not a
# command-line input (avoids threading an untrusted path into a file read).
EXPECTATIONS_PATH = Path("tests/e2e-cucumber/expectations.toml")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    commits = ""
    if args.commits_file and args.commits_file.exists():
        commits = args.commits_file.read_text(encoding="utf-8")
    expectations_toml = EXPECTATIONS_PATH.read_text(encoding="utf-8")
    hits = run_check(expectations_toml, args.title, args.body, commits)
    emit(hits)
    # Advisory only: always succeed so this never blocks a merge.
    sys.exit(0)


if __name__ == "__main__":
    main()
