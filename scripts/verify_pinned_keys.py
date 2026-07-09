#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Assert the release/metadata public keys pinned in the installers and rocm-core
match the canonical published keys under ``docs/keys/``.

This closes the gap where CI verifies release artifacts against the public key
configured in its environment (see ``release_readiness.py``) while installers and
the binary trust a *separately embedded* copy. If those diverge, CI can bless a
release that installers then reject once default-on verification lands. This check
enforces a single source of truth: the committed ``docs/keys/*.pem`` files.

Design notes:

- **Dormant by default.** Until a canonical ``docs/keys/`` file exists and is
  non-empty, its key is skipped. So this passes as a no-op today (empty pinned
  sentinels, no canonical files) and only starts enforcing once the owner ceremony
  publishes real keys.
- **Embedding-agnostic comparison.** The pinned key lives in three different
  syntaxes (shell double-quoted string, PowerShell string/here-string, Rust string
  literal). Rather than parse each, we reduce both the canonical key and the source
  file to their base64 alphabet only (``[A-Za-z0-9+/=]``) and check containment.
  A 2048-bit SPKI body is ~360 base64 chars — distinctive enough that this is a
  reliable equality proxy immune to quoting, ``\\n`` escapes, line continuations,
  and here-strings.
- **CI cross-check.** When ``ROCM_CLI_SIGNING_PUBLIC_KEY_PATH/PEM`` is set (as in
  release CI), it must equal the canonical *current* release key, so the key CI
  verifies against is exactly the one users pin.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

PEM_BEGIN = "-----BEGIN PUBLIC KEY-----"
PEM_END = "-----END PUBLIC KEY-----"

# canonical published key -> source files that must embed the identical bytes.
PINNED_KEYS = (
    {
        "name": "release-current",
        "canonical": "docs/keys/rocm-cli-release-current-public.pem",
        "sources": ("install.sh", "install.ps1"),
    },
    {
        "name": "release-next",
        "canonical": "docs/keys/rocm-cli-release-next-public.pem",
        "sources": ("install.sh", "install.ps1"),
    },
    {
        "name": "metadata",
        "canonical": "docs/keys/rocm-cli-metadata-public.pem",
        "sources": ("apps/rocm/src/therock.rs",),
    },
)

# The canonical key that CI signs and verifies against under production trust.
CURRENT_RELEASE_KEY = "release-current"


class ConsistencyError(Exception):
    """A pinned key does not match its canonical source of truth."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def base64_only(text: str) -> str:
    """Reduce arbitrary text to its base64 alphabet, dropping everything else.

    Newline escape sequences are removed first: a literal ``\\n`` (shell/Rust) or
    ``\\`n`` (PowerShell) would otherwise leave a stray ``n``/``r``/``t`` — all
    base64-alphabet letters — and corrupt the payload. Real newlines are plain
    whitespace and drop out with everything else non-base64.
    """
    without_escapes = re.sub(r"[\\`][nrt]", "", text)
    return re.sub(r"[^A-Za-z0-9+/=]", "", without_escapes)


def pem_body(text: str) -> str | None:
    """Return the whitespace-stripped base64 body of a PUBLIC KEY PEM, or None.

    Uses plain string search (not a regex) so it is linear-time regardless of
    input — no catastrophic/polynomial backtracking on adversarial content.
    """
    begin = text.find(PEM_BEGIN)
    if begin == -1:
        return None
    start = begin + len(PEM_BEGIN)
    end = text.find(PEM_END, start)
    if end == -1:
        return None
    body = base64_only(text[start:end])
    return body or None


def fingerprint(pem_text: str) -> str:
    """SHA-256 over the normalized (LF, no trailing blank lines) PEM bytes."""
    normalized = "\n".join(
        line.rstrip() for line in pem_text.replace("\r\n", "\n").split("\n")
    ).strip()
    return hashlib.sha256((normalized + "\n").encode("utf-8")).hexdigest()


def check_pinned_keys(root: Path) -> list[str]:
    """Verify every populated canonical key is embedded verbatim in its sources.

    Returns human-readable status messages. Raises ConsistencyError on a mismatch.
    """
    messages: list[str] = []
    current_release_body: str | None = None

    for entry in PINNED_KEYS:
        name = entry["name"]
        canonical_path = root / entry["canonical"]
        if (
            not canonical_path.is_file()
            or not canonical_path.read_text(encoding="utf-8").strip()
        ):
            messages.append(f"{name}: no canonical key yet — skipped (dormant)")
            continue

        canonical_text = canonical_path.read_text(encoding="utf-8")
        canonical_body = pem_body(canonical_text)
        if canonical_body is None:
            raise ConsistencyError(
                f"{name}: {entry['canonical']} is not a valid PUBLIC KEY PEM"
            )
        if name == CURRENT_RELEASE_KEY:
            current_release_body = canonical_body

        for source in entry["sources"]:
            source_path = root / source
            if not source_path.is_file():
                raise ConsistencyError(f"{name}: source file is missing: {source}")
            if canonical_body not in base64_only(
                source_path.read_text(encoding="utf-8")
            ):
                raise ConsistencyError(
                    f"{name}: {source} does not embed the canonical key "
                    f"{entry['canonical']} — the pinned constant and the published "
                    f"key have diverged"
                )
        messages.append(
            f"{name}: pinned in {', '.join(entry['sources'])} "
            f"(sha256 {fingerprint(canonical_text)})"
        )

    messages.extend(check_ci_public_key(current_release_body))
    return messages


def check_ci_public_key(current_release_body: str | None) -> list[str]:
    """When a CI signing public key is configured, it must equal the canonical
    current release key — the key CI verifies release artifacts against.

    Only the inline PEM form (ROCM_CLI_SIGNING_PUBLIC_KEY_PEM) is checked, which is
    what release/nightly CI wire from the signing-key secret."""
    env_pem = os.environ.get("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM", "").strip()
    if not env_pem:
        return []

    env_body = pem_body(env_pem)
    if env_body is None:
        raise ConsistencyError(
            "the configured CI signing public key is not a valid PUBLIC KEY PEM"
        )

    if current_release_body is None:
        return [
            "ci public key: configured, but no canonical current release key to "
            "compare against yet — skipped"
        ]
    if env_body != current_release_body:
        raise ConsistencyError(
            "the CI signing public key does not match the canonical current "
            f"release key ({PINNED_KEYS[0]['canonical']}); CI would verify against "
            "a different key than installers pin"
        )
    return ["ci public key: matches the canonical current release key"]


def fail(message: str) -> None:
    print(f"pinned key check: {message}", file=sys.stderr)
    sys.exit(1)


def run_self_test() -> None:
    sample = (
        "-----BEGIN PUBLIC KEY-----\n"
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAselftestselftest+/\n"
        "abcDEF0123456789selftestselftestselftestselftestselftestABCD==\n"
        "-----END PUBLIC KEY-----\n"
    )
    other = sample.replace("selftest", "different")
    work = Path(tempfile.mkdtemp(prefix="pinned-key-self-test-"))
    try:
        keys_dir = work / "docs" / "keys"
        keys_dir.mkdir(parents=True)
        (work / "apps" / "rocm" / "src").mkdir(parents=True)

        # Dormant: no canonical files -> passes as a no-op.
        (work / "install.sh").write_text('PINNED=""\n', encoding="utf-8")
        (work / "install.ps1").write_text('$Pinned = ""\n', encoding="utf-8")
        (work / "apps" / "rocm" / "src" / "therock.rs").write_text(
            'const PINNED: &str = "";\n', encoding="utf-8"
        )
        check_pinned_keys(work)
        print("pinned key self-test: dormant (no canonical keys) accepted")

        # Populated and matching (embedded with LF, CRLF, and \n escapes).
        (keys_dir / "rocm-cli-metadata-public.pem").write_text(sample, encoding="utf-8")
        escaped = sample.replace("\n", "\\n")
        (work / "apps" / "rocm" / "src" / "therock.rs").write_text(
            f'const PINNED: &str = "{escaped}";\n', encoding="utf-8"
        )
        check_pinned_keys(work)
        print("pinned key self-test: matching embedded key (\\n-escaped) accepted")

        # Divergent: source embeds a different key -> rejected.
        (work / "apps" / "rocm" / "src" / "therock.rs").write_text(
            f'const PINNED: &str = "{other.replace(chr(10), chr(92) + "n")}";\n',
            encoding="utf-8",
        )
        try:
            check_pinned_keys(work)
        except ConsistencyError:
            print("pinned key self-test: divergent embedded key rejected as expected")
        else:
            raise ConsistencyError("divergent embedded key unexpectedly passed")

        # CI public-key cross-check against a mismatched key -> rejected.
        (keys_dir / "rocm-cli-release-current-public.pem").write_text(
            sample, encoding="utf-8"
        )
        (work / "install.sh").write_text(f'PINNED="{sample}"\n', encoding="utf-8")
        (work / "install.ps1").write_text(
            f'$Pinned = @"\n{sample}\n"@\n', encoding="utf-8"
        )
        (work / "apps" / "rocm" / "src" / "therock.rs").write_text(
            f'const PINNED: &str = "{escaped}";\n', encoding="utf-8"
        )
        saved = os.environ.get("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM")
        os.environ["ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"] = other
        try:
            check_pinned_keys(work)
        except ConsistencyError:
            print("pinned key self-test: mismatched CI public key rejected as expected")
        else:
            raise ConsistencyError("mismatched CI public key unexpectedly passed")
        finally:
            if saved is None:
                os.environ.pop("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM", None)
            else:
                os.environ["ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"] = saved

        print("pinned key self-test: ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in fixtures instead of checking the repository",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            return
        messages = check_pinned_keys(repo_root())
    except ConsistencyError as error:
        fail(str(error))
    for message in messages:
        print(f"pinned key check: {message}")


if __name__ == "__main__":
    main()
