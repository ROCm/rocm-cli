#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""First-pass triage of a newly opened issue.

Run by ``.github/workflows/issue-triage.yml``. For one issue it:

* parses the issue-template sections and flags a report that is missing its
  reproduction steps or version (``needs-info``, plus one comment asking for them);
* adds a ``platform:*`` label from the template's ``Platform:`` line;
* applies a type, area and priority classified by a model, and
* assigns the next maintainer in round-robin order.

The script never calls the model. The workflow runs it in three phases, in
separate jobs so the credential that reaches the model is never the one that
can write issues:

1. ``--emit-prompt FILE`` (read-only token): fetch the issue and write the
   untrusted title and body, size-capped, to FILE. No file means skip.
2. The workflow's model step reads FILE with the system prompt in
   ``.github/prompts/issue-triage.txt`` (no write token in that job).
3. ``--issue N`` with the model's reply in ``TRIAGE_MODEL_RESPONSE`` (write
   token): the reply is reduced to values from the fixed enums below, and the
   script, not the model, decides which labels exist, which are applied and who
   is assigned. A missing or malformed reply still gets the deterministic
   labels and the assignment.

Idempotent: an issue that already carries a priority or area label or an assignee is
left alone for that part, so a re-run (``workflow_dispatch``) does not stack
labels or reassign.

Usage::

    GH_TOKEN=... python scripts/triage_issue.py --repo OWNER/REPO --issue 123 --emit-prompt out.txt
    GH_TOKEN=... TRIAGE_MODEL_RESPONSE='{"type": "bug", ...}' \\
        python scripts/triage_issue.py --repo OWNER/REPO --issue 123 [--dry-run]
    python scripts/triage_issue.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_BODY_CHARS = 6000
MAINTAINERS_FILE = REPO_ROOT / ".github" / "triage-maintainers.txt"
SYSTEM_PROMPT_FILE = REPO_ROOT / ".github" / "prompts" / "issue-triage.txt"
RESPONSE_ENV = "TRIAGE_MODEL_RESPONSE"

TYPES = ("bug", "enhancement", "question", "documentation")
# Template-applied or maintainer-applied labels that already settle the type.
TYPE_LABELS = frozenset(TYPES)

# model enum value -> label. `installation`, `engines` and `documentation` are
# long-standing labels; the rest are created on first use.
AREAS = {
    "cli": "area:cli",
    "dash-tui": "area:dash-tui",
    "daemon": "area:rocmd",
    "assistant": "area:assistant",
    "engines": "engines",
    "install": "installation",
    "docs": "documentation",
}
PRIORITIES = ("P0", "P1", "P2", "P3")
PRIORITY_PREFIX = "priority:"

# label -> (colour, description) for labels this script may have to create.
NEW_LABELS = {
    "area:cli": ("1d76db", "The rocm command line"),
    "area:dash-tui": ("1d76db", "The dash terminal UI"),
    "area:rocmd": ("1d76db", "The rocmd background daemon"),
    "area:assistant": ("1d76db", "The built-in assistant"),
    "platform:linux": ("5319e7", "Reported on Linux"),
    "platform:windows": ("5319e7", "Reported on Windows"),
    "platform:wsl": ("5319e7", "Reported on WSL2"),
    "needs-info": ("fbca04", "Waiting on details from the reporter"),
    "priority:P0": ("b60205", "Crash or data loss on a supported path"),
    "priority:P1": ("d93f0b", "Core flow broken, no workaround"),
    "priority:P2": ("fbca04", "Real problem with a workaround"),
    "priority:P3": ("0e8a16", "Minor, cosmetic or nice-to-have"),
}


def parse_sections(body: str) -> dict[str, str]:
    """Split a template body on its ``**Heading**`` lines."""
    sections: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        m = re.fullmatch(r"\s*\*\*(.+?)\*\*\s*", line)
        if m:
            current = m.group(1).strip().lower()
            sections[current] = ""
        elif current is not None:
            sections[current] += line + "\n"
    return {k: v.strip() for k, v in sections.items()}


def field_value(body: str, label: str) -> str:
    """The text after ``- <label>:`` on one line of the environment list."""
    m = re.search(
        rf"^\s*[-*]\s*{re.escape(label)}[^:\n]*:[ \t]*(.*)$", body, re.I | re.M
    )
    return m.group(1).strip() if m else ""


# The bug template's own hint under **Steps to reproduce**; left unedited it
# is not a set of steps.
STEPS_PLACEHOLDER = "Commands, flags, or a minimal example that reproduces the issue"


def is_bug_template(body: str) -> bool:
    """True when the body still has the bug template's headings.

    A free-form report labelled ``bug`` (blank issues are allowed) is not asked
    for the template's fields: it may already say all of it in prose.
    """
    sections = parse_sections(body)
    return "steps to reproduce" in sections and "your environment" in sections


def missing_info(body: str, labels: set[str]) -> list[str]:
    """What a bug report still has to say.

    Only a ``bug`` report written in the bug template is checked.
    """
    if "bug" not in labels or not is_bug_template(body):
        return []
    sections = parse_sections(body)
    steps = sections.get("steps to reproduce", "")
    if steps.strip() == STEPS_PLACEHOLDER:
        steps = ""
    missing = []
    if len(steps) < 10:
        missing.append("steps to reproduce")
    if not field_value(body, "rocm-cli version"):
        missing.append("the `rocm --version` output")
    return missing


def platform_labels(body: str) -> list[str]:
    value = field_value(body, "Platform").lower()
    found = []
    if re.search(r"\bwsl", value):
        found.append("platform:wsl")
    if re.search(r"\blinux\b", value):
        found.append("platform:linux")
    if re.search(r"\bwindows\b", value):
        found.append("platform:windows")
    # The template's own `Linux / WSL2 / Windows` line, never edited, names all
    # three; that is no report of a platform.
    return [] if len(found) == 3 else found


def load_maintainers(path: Path = MAINTAINERS_FILE) -> list[str]:
    logins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            logins.append(line)
    return logins


def pick_assignee(issue_number: int, maintainers: list[str], author: str) -> str | None:
    """Round-robin on the issue number, so it needs no stored state.

    The author is skipped (an issue is not assigned to its reporter) by taking
    the next maintainer along.
    """
    if not maintainers:
        return None
    for offset in range(len(maintainers)):
        candidate = maintainers[(issue_number + offset) % len(maintainers)]
        if candidate.lower() != author.lower():
            return candidate
    return None


def validate_classification(raw: object) -> dict[str, str]:
    """Keep only values inside the fixed enums; anything else is dropped."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    if raw.get("type") in TYPES:
        out["type"] = raw["type"]
    if raw.get("area") in AREAS:
        out["area"] = raw["area"]
    if raw.get("priority") in PRIORITIES:
        out["priority"] = raw["priority"]
    return out


def parse_response(text: str) -> dict[str, str]:
    """Turn the model's reply into a validated classification.

    The reply is meant to be one JSON object, but a chat model may wrap it in a
    code fence or a sentence, so the first ``{`` to the last ``}`` is parsed.
    Anything unparseable gives ``{}``, never an error.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return {}
    try:
        return validate_classification(json.loads(text[start : end + 1]))
    except ValueError:
        return {}


def build_prompt(issue: dict) -> str:
    """The untrusted issue text, marked and size-capped, for the model step."""
    body = (issue.get("body") or "")[:MAX_BODY_CHARS]
    return f"<<<ISSUE START>>>\nTitle: {issue['title']}\n\n{body}\n<<<ISSUE END>>>\n"


def skip_reason(issue: dict) -> str | None:
    if "pull_request" in issue:
        return "not an issue"
    if (issue.get("user") or {}).get("type") == "Bot":
        return "opened by a bot"
    return None


def plan(issue: dict, classification: dict[str, str], maintainers: list[str]) -> dict:
    """Decide the actions for one issue. Pure: no network, no writes."""
    body = issue.get("body") or ""
    existing = {label["name"] for label in issue.get("labels", [])}
    author = (issue.get("user") or {}).get("login", "")

    add: list[str] = []
    add += [label for label in platform_labels(body) if label not in existing]
    if not existing & TYPE_LABELS and "type" in classification:
        add.append(classification["type"])
    if "area" in classification and not existing & set(AREAS.values()):
        add.append(AREAS[classification["area"]])
    already_prioritised = any(name.startswith(PRIORITY_PREFIX) for name in existing)
    if not already_prioritised and "priority" in classification:
        add.append(PRIORITY_PREFIX + classification["priority"])

    missing = missing_info(body, existing | set(add))
    if missing and "needs-info" not in existing:
        add.append("needs-info")

    assignee = None
    if not issue.get("assignees"):
        assignee = pick_assignee(issue["number"], maintainers, author)

    return {
        "labels": [label for label in dict.fromkeys(add) if label not in existing],
        "assignee": assignee,
        "comment": needs_info_comment(missing)
        if missing and "needs-info" not in existing
        else None,
    }


def needs_info_comment(missing: list[str]) -> str:
    items = "\n".join(f"- {item}" for item in missing)
    return (
        "Thanks for the report. To help us reproduce this, could you add:\n\n"
        f"{items}\n\n"
        "A maintainer will take another look once it is here."
    )


def gh(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()}")
    return result.stdout


def ensure_labels(repo: str, names: list[str]) -> None:
    for name in names:
        if name in NEW_LABELS:
            color, description = NEW_LABELS[name]
            # --force updates an existing label instead of failing on it.
            gh(
                "label",
                "create",
                name,
                "--repo",
                repo,
                "--color",
                color,
                "--description",
                description,
                "--force",
            )


def apply(repo: str, number: int, decided: dict) -> None:
    if decided["labels"]:
        ensure_labels(repo, decided["labels"])
        gh(
            "issue",
            "edit",
            str(number),
            "--repo",
            repo,
            "--add-label",
            ",".join(decided["labels"]),
        )
    if decided["assignee"]:
        try:
            gh(
                "issue",
                "edit",
                str(number),
                "--repo",
                repo,
                "--add-assignee",
                decided["assignee"],
            )
        except RuntimeError as err:
            # A maintainer who cannot be assigned must not fail the whole run.
            print(f"::warning::could not assign {decided['assignee']}: {err}")
    if decided["comment"]:
        gh(
            "issue",
            "comment",
            str(number),
            "--repo",
            repo,
            "--body-file",
            "-",
            stdin=decided["comment"],
        )


def fetch_issue(repo: str, number: int) -> dict:
    return json.loads(gh("api", f"repos/{repo}/issues/{number}"))


def emit_prompt(repo: str, number: int, out: Path) -> int:
    """Write the model prompt to ``out``, or leave ``out`` absent to mean skip."""
    issue = fetch_issue(repo, number)
    reason = skip_reason(issue)
    if reason:
        print(f"{reason}; skipping")
        return 0
    out.write_text(build_prompt(issue), encoding="utf-8")
    return 0


def run(repo: str, number: int, dry_run: bool) -> int:
    issue = fetch_issue(repo, number)
    reason = skip_reason(issue)
    if reason:
        print(f"{reason}; skipping")
        return 0
    response = os.environ.get(RESPONSE_ENV, "")
    classification = parse_response(response)
    if not classification:
        print("::warning::no usable model classification; labelling without it")
    decided = plan(issue, classification, load_maintainers())
    print(json.dumps({"classification": classification, **decided}, indent=2))
    if not dry_run:
        apply(repo, number, decided)
    return 0


def self_test() -> int:
    bug_body = (
        "**Current behavior**\nIt crashes\n\n**Steps to reproduce**\nrocm serve foo --port 1\n\n"
        "**Your environment**\n- rocm-cli channel: nightly\n- rocm-cli version (`rocm --version`): 0.4.1\n"
        "- Platform: WSL2 / Windows\n- GPU/APU: gfx1151\n"
    )
    thin_body = "**Steps to reproduce**\n\n**Your environment**\n- rocm-cli version (`rocm --version`):\n"
    team = ["a", "b", "c"]

    sections = parse_sections(bug_body)
    assert sections["steps to reproduce"] == "rocm serve foo --port 1", sections
    assert field_value(bug_body, "rocm-cli version") == "0.4.1"
    assert missing_info(bug_body, {"bug"}) == []
    assert missing_info(thin_body, {"bug"}) == [
        "steps to reproduce",
        "the `rocm --version` output",
    ]
    # Feature requests and questions are never asked for repro steps.
    assert missing_info(thin_body, {"enhancement"}) == []
    assert platform_labels(bug_body) == ["platform:wsl", "platform:windows"]

    # An untouched template asks for steps and gets no platform label.
    raw_template = (
        "**Current behavior**\nx\n\n**Steps to reproduce**\n"
        f"{STEPS_PLACEHOLDER}\n\n**Your environment**\n"
        "- rocm-cli version (`rocm --version`): 0.4.1\n"
        "- Platform: Linux / WSL2 / Windows\n"
    )
    assert missing_info(raw_template, {"bug"}) == ["steps to reproduce"]
    assert platform_labels(raw_template) == []
    # A free-form `bug` report is not held to the template's fields.
    free_form = "rocm serve crashes on gfx1151 with 0.4.1, run `rocm serve foo`."
    assert missing_info(free_form, {"bug"}) == []
    free_issue = {
        "number": 7,
        "body": free_form,
        "labels": [],
        "user": {"login": "rep"},
        "assignees": [],
    }
    got = plan(free_issue, {"type": "bug"}, team)
    assert got["labels"] == ["bug"] and got["comment"] is None, got

    # Round robin walks the team and skips the reporter.
    assert [pick_assignee(n, team, "zz") for n in (3, 4, 5, 6)] == ["a", "b", "c", "a"]
    assert pick_assignee(3, team, "A") == "b"
    assert pick_assignee(3, [], "x") is None
    assert pick_assignee(3, ["a"], "a") is None

    # Model output outside the enums is dropped, never applied.
    assert validate_classification(
        {"type": "bug", "area": "cli", "priority": "P1"}
    ) == {"type": "bug", "area": "cli", "priority": "P1"}
    assert (
        validate_classification(
            {"type": "wontfix", "area": "../x", "priority": "P0; rm -rf"}
        )
        == {}
    )
    assert validate_classification(["bug"]) == {}

    # A chat model may fence or introduce the JSON; only the object is parsed,
    # and a reply with no usable object gives {} rather than an error.
    reply = '{"type": "bug", "area": "engines", "priority": "P2", "summary": "x"}'
    want = {"type": "bug", "area": "engines", "priority": "P2"}
    assert parse_response(reply) == want
    assert parse_response(f"```json\n{reply}\n```") == want
    assert parse_response(f"Here you go: {reply}") == want
    assert parse_response("") == {}
    assert parse_response("OK") == {}
    assert parse_response("{not json}") == {}
    assert parse_response('{"type": "bug"') == {}

    # The prompt carries the issue text between markers and is size-capped.
    prompt = build_prompt({"title": "T", "body": "x" * (MAX_BODY_CHARS + 500)})
    assert prompt.startswith("<<<ISSUE START>>>\nTitle: T")
    assert prompt.endswith("<<<ISSUE END>>>\n")
    assert prompt.count("x") == MAX_BODY_CHARS
    assert build_prompt({"title": "T", "body": None}).count("<<<") == 2

    # Bots and pull requests are never triaged.
    assert skip_reason({"user": {"type": "Bot"}}) == "opened by a bot"
    assert skip_reason({"pull_request": {}, "user": {"type": "User"}}) == "not an issue"
    assert skip_reason({"user": {"type": "User"}}) is None

    # The system prompt (a separate file the model step reads) and this script
    # share one vocabulary; a value on one side only would never be applied.
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    for value in (*TYPES, *AREAS, *PRIORITIES):
        assert value in system_prompt, f"{value} missing from {SYSTEM_PROMPT_FILE}"

    issue = {
        "number": 5,
        "body": bug_body,
        "labels": [{"name": "bug"}],
        "user": {"login": "rep"},
        "assignees": [],
    }
    got = plan(issue, {"type": "question", "area": "install", "priority": "P1"}, team)
    # The template's `bug` label is not overridden; the rest is added.
    assert got["labels"] == [
        "platform:wsl",
        "platform:windows",
        "installation",
        "priority:P1",
    ], got
    assert got["assignee"] == "c" and got["comment"] is None, got

    # Thin bug report: needs-info plus a comment naming both gaps.
    thin = {
        "number": 6,
        "body": thin_body,
        "labels": [{"name": "bug"}],
        "user": {"login": "rep"},
        "assignees": [],
    }
    got = plan(thin, {}, team)
    assert got["labels"] == ["needs-info"] and "steps to reproduce" in got["comment"], (
        got
    )

    # Re-run is idempotent: existing priority, needs-info and assignee are respected.
    again = {
        "number": 6,
        "body": thin_body,
        "user": {"login": "rep"},
        "labels": [
            {"name": "bug"},
            {"name": "needs-info"},
            {"name": "priority:P2"},
            {"name": "installation"},
        ],
        "assignees": [{"login": "a"}],
    }
    got = plan(again, {"priority": "P0", "area": "cli"}, team)
    assert got == {"labels": [], "assignee": None, "comment": None}, got

    # Every label the script can apply is creatable or pre-existing.
    created = set(NEW_LABELS)
    assert {PRIORITY_PREFIX + p for p in PRIORITIES} <= created
    assert {
        v
        for v in AREAS.values()
        if v not in {"engines", "installation", "documentation"}
    } <= created
    print("triage_issue self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo")
    parser.add_argument("--issue", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, change nothing"
    )
    parser.add_argument(
        "--emit-prompt",
        type=Path,
        metavar="FILE",
        help="write the model prompt for the issue to FILE (nothing if it is skipped)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.repo or not args.issue:
        parser.error("--repo and --issue are required")
    if args.emit_prompt:
        return emit_prompt(args.repo, args.issue, args.emit_prompt)
    return run(args.repo, args.issue, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
