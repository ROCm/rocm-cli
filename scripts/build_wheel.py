#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Build the `rocm-cli` PyPI wheel from already-built rocm/rocmd binaries.

The wheel is a thin distribution channel: it carries no Python modules and no
console-script shims. The native binaries are placed in the wheel's
`.data/scripts/` directory so pip installs them verbatim into the environment's
`bin`/`Scripts` directory, keeping `std::env::current_exe()` pointed at the real
executable (the CLI re-execs itself and locates `rocmd` as a sibling file).

Only prebuilt binaries are packaged here; this script never invokes cargo.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DISTRIBUTION = "rocm-cli"
NORMALIZED_NAME = "rocm_cli"
GENERATOR = "rocm-cli build_wheel.py"
SUMMARY = "Command-line control plane for local ROCm AI inference (rocm, rocmd)."

PLATFORM_TAGS = {
    "linux-amd64": "manylinux_2_17_x86_64.manylinux2014_x86_64",
    "windows-amd64": "win_amd64",
}
BINARY_STEMS = ("rocm", "rocmd")
LICENSE_FILES = ("LICENSE.TXT", "THIRD_PARTY_NOTICES.txt")

# Reproducible zip timestamp: the earliest value the zip format can encode.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
SCRIPT_MODE = 0o755
DATA_MODE = 0o644
S_IFREG = 0o100000
CREATE_SYSTEM_UNIX = 3

TAG_RE = re.compile(
    r"^v?(?P<release>\d+\.\d+\.\d+)"
    r"(?:-(?P<kind>alpha|beta|rc|experimental)\.(?P<serial>\d+))?$"
)
PRE_RELEASE_MARKERS = {
    "alpha": "a",
    "experimental": "a",
    "beta": "b",
    "rc": "rc",
}
# Every version this builder is allowed to emit. Deliberately narrower than
# PEP 440: local versions (`+local`) are illegal in a wheel filename, and dev,
# post, and epoch versions are rejected by PyPI or sort confusingly against the
# release tags this project actually pushes. `--version` bypasses the tag
# mapper, so it is validated against the same shape rather than trusted.
PUBLISHABLE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")
WORKSPACE_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')

DESCRIPTION = """\
# rocm-cli

`rocm-cli` distributes the prebuilt ROCm AI command-line control plane as a
Python wheel. Installing it places two native executables on your `PATH`:

- `rocm` — the user-facing CLI: discover hardware, install and serve models,
  chat with a local endpoint, and inspect system health.
- `rocmd` — the background daemon the CLI supervises for long-running engine
  services.

The wheel contains no Python code. It is a delivery vehicle for the same native
binaries published on the GitHub release page, so `pip install rocm-cli` is
simply a convenient way to get them.

Supported platforms: Linux x86-64 (manylinux2014 or newer) and Windows x86-64.
No other platform wheels are published.

Source, issues, and documentation: https://github.com/ROCm/rocm-cli
"""


class WheelBuildError(Exception):
    """The wheel could not be built."""


def fail(message: str) -> None:
    print(f"wheel build failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def pep440_version_from_tag(tag: str) -> str:
    """Map a release git tag onto its PEP 440 version.

    `vX.Y.Z` -> `X.Y.Z`, `-alpha.N`/`-experimental.N` -> `X.Y.ZaN`,
    `-beta.N` -> `X.Y.ZbN`, `-rc.N` -> `X.Y.ZrcN`. Every other shape (nightly
    and staging tags included) is a hard error.

    `-alpha.N` and `-experimental.N` intentionally collapse onto the same
    `aN` form, because PEP 440 has one alpha marker and both tag spellings mean
    the same thing here. The consequence is that only one of the two spellings
    may be used per serial: pushing both `v1.2.3-experimental.1` and
    `v1.2.3-alpha.1` produces the same wheel version twice, and PyPI refuses
    the second upload because a released version can never be replaced.
    """
    match = TAG_RE.fullmatch(tag.strip())
    if match is None:
        raise WheelBuildError(f"tag is not a publishable release tag: {tag}")
    release = match.group("release")
    kind = match.group("kind")
    if kind is None:
        return release
    serial = int(match.group("serial"))
    return f"{release}{PRE_RELEASE_MARKERS[kind]}{serial}"


def release_segment(version: str) -> str:
    """Return the `X.Y.Z` release segment of a mapped PEP 440 version."""
    match = re.match(r"^(\d+\.\d+\.\d+)", version)
    if match is None:
        raise WheelBuildError(f"version is not a PEP 440 release version: {version}")
    return match.group(1)


def workspace_version(repo_root: Path) -> str:
    """Parse `[workspace.package] version` out of the root `Cargo.toml`."""
    manifest = repo_root / "Cargo.toml"
    if not manifest.is_file():
        raise WheelBuildError(f"workspace manifest not found: {manifest}")
    in_section = False
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            # A section header may carry a trailing comment. Strip it before
            # matching so `[workspace.package] # ...` is still recognized.
            header = line.split("#", 1)[0].strip()
            if not header.endswith("]"):
                continue
            in_section = header == "[workspace.package]"
            continue
        if not in_section:
            continue
        match = WORKSPACE_VERSION_RE.match(line)
        if match is not None:
            return match.group(1)
    raise WheelBuildError(f"[workspace.package] version not found in {manifest}")


def resolve_version(repo_root: Path, *, tag: str | None, version: str | None) -> str:
    """Resolve the wheel version and cross-check it against the workspace."""
    if (tag is None) == (version is None):
        raise WheelBuildError("exactly one of --tag or --version is required")
    resolved = pep440_version_from_tag(tag) if tag is not None else str(version)
    if PUBLISHABLE_VERSION_RE.fullmatch(resolved) is None:
        source = f"tag {tag}" if tag is not None else f"version {version}"
        raise WheelBuildError(
            f"{source} resolves to {resolved}, which is not a publishable wheel "
            "version (expected X.Y.Z with an optional aN, bN, or rcN suffix; "
            "local, dev, post, and epoch versions are refused)"
        )
    expected = workspace_version(repo_root)
    actual = release_segment(resolved)
    if actual != expected:
        source = f"tag {tag}" if tag is not None else f"version {version}"
        raise WheelBuildError(
            f"{source} maps to release segment {actual}, but the workspace version "
            f"in {repo_root / 'Cargo.toml'} is {expected}"
        )
    return resolved


def wheel_platform_tag(platform: str) -> str:
    """Return the wheel platform tag for a supported release platform."""
    try:
        return PLATFORM_TAGS[platform]
    except KeyError:
        raise WheelBuildError(f"unsupported platform: {platform}") from None


def binary_names(platform: str) -> tuple[str, ...]:
    suffix = ".exe" if platform == "windows-amd64" else ""
    return tuple(f"{stem}{suffix}" for stem in BINARY_STEMS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: Path) -> Path:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return sidecar


def record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def metadata_text(version: str) -> str:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {DISTRIBUTION}",
        f"Version: {version}",
        f"Summary: {SUMMARY}",
        "License-Expression: MIT",
        "License-File: LICENSE.TXT",
        "License-File: THIRD_PARTY_NOTICES.txt",
        "Requires-Python: >=3.9",
        "Project-URL: Source, https://github.com/ROCm/rocm-cli",
        "Classifier: Environment :: Console",
        "Classifier: Intended Audience :: Developers",
        "Classifier: Operating System :: POSIX :: Linux",
        "Classifier: Operating System :: Microsoft :: Windows",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Description-Content-Type: text/markdown",
        "",
        DESCRIPTION,
    ]
    return "\n".join(lines)


def wheel_text(platform_tag: str) -> str:
    return (
        "Wheel-Version: 1.0\n"
        f"Generator: {GENERATOR}\n"
        "Root-Is-Purelib: false\n"
        f"Tag: py3-none-{platform_tag}\n"
    )


def zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = CREATE_SYSTEM_UNIX
    # The regular-file bit is mandatory: without S_IFREG pip installs the
    # binaries without their executable bit. The low 16 bits are MS-DOS
    # attributes and are left at zero, as conventional wheel writers do.
    info.external_attr = (S_IFREG | mode) << 16
    return info


def collect_members(
    *,
    repo_root: Path,
    bin_dir: Path,
    platform: str,
    version: str,
) -> list[tuple[str, bytes, int]]:
    """Return `(arcname, payload, mode)` triples in stable write order."""
    data_dir = f"{NORMALIZED_NAME}-{version}.data/scripts"
    dist_info = f"{NORMALIZED_NAME}-{version}.dist-info"
    members: list[tuple[str, bytes, int]] = []

    for name in binary_names(platform):
        source = bin_dir / name
        if not source.is_file():
            raise WheelBuildError(f"required binary not found: {source}")
        members.append((f"{data_dir}/{name}", source.read_bytes(), SCRIPT_MODE))

    members.append(
        (f"{dist_info}/METADATA", metadata_text(version).encode("utf-8"), DATA_MODE)
    )
    members.append(
        (
            f"{dist_info}/WHEEL",
            wheel_text(wheel_platform_tag(platform)).encode("utf-8"),
            DATA_MODE,
        )
    )
    for name in LICENSE_FILES:
        source = repo_root / name
        if not source.is_file():
            raise WheelBuildError(f"required license file not found: {source}")
        members.append((f"{dist_info}/licenses/{name}", source.read_bytes(), DATA_MODE))
    return members


def record_text(members: list[tuple[str, bytes, int]], record_name: str) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for arcname, payload, _mode in members:
        writer.writerow([arcname, record_hash(payload), len(payload)])
    writer.writerow([record_name, "", ""])
    return buffer.getvalue()


def build_wheel(
    *,
    repo_root: Path,
    bin_dir: Path,
    platform: str,
    version: str,
    out_dir: Path,
) -> Path:
    """Write the wheel (and its `.sha256` sidecar) and return the wheel path."""
    platform_tag = wheel_platform_tag(platform)
    if not bin_dir.is_dir():
        raise WheelBuildError(f"binary directory not found: {bin_dir}")

    members = collect_members(
        repo_root=repo_root, bin_dir=bin_dir, platform=platform, version=version
    )
    record_name = f"{NORMALIZED_NAME}-{version}.dist-info/RECORD"
    members.append(
        (record_name, record_text(members, record_name).encode("utf-8"), DATA_MODE)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = out_dir / f"{NORMALIZED_NAME}-{version}-py3-none-{platform_tag}.whl"
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as package:
        for arcname, payload, mode in members:
            package.writestr(zip_info(arcname, mode), payload)
    write_sha256(wheel_path)
    return wheel_path


def expect_error(label: str, func) -> None:
    try:
        func()
    except WheelBuildError:
        return
    raise WheelBuildError(f"{label} unexpectedly succeeded")


def self_test_tag_mapping() -> None:
    cases = {
        "v1.2.3": "1.2.3",
        "1.2.3": "1.2.3",
        "v1.2.3-alpha.4": "1.2.3a4",
        "v1.2.3-experimental.4": "1.2.3a4",
        "v1.2.3-beta.4": "1.2.3b4",
        "v1.2.3-rc.4": "1.2.3rc4",
        "v0.1.0-experimental.1": "0.1.0a1",
    }
    for tag, expected in cases.items():
        actual = pep440_version_from_tag(tag)
        if actual != expected:
            raise WheelBuildError(f"tag {tag} mapped to {actual}, expected {expected}")
    for bad in (
        "nightly-20260804-abc1234",
        "staging-v1.2.3",
        "v1.2.3-experimental",
        "v1.2.3-alpha",
        "v1.2.3-pre.1",
        "1.2",
        "v1.2.3.4",
        "",
    ):
        expect_error(f"tag {bad!r}", lambda bad=bad: pep440_version_from_tag(bad))


def make_fake_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo).mkdir(parents=True)
    (repo / "Cargo.toml").write_text(
        "[workspace]\n"
        'members = ["apps/rocm"]\n'
        "\n"
        "[workspace.dependencies]\n"
        'version = "9.9.9"\n'
        "\n"
        "[workspace.package]\n"
        'version = "0.1.0"\n'
        'edition = "2024"\n',
        encoding="utf-8",
    )
    for name in LICENSE_FILES:
        (repo / name).write_text(f"{name} test content\n", encoding="utf-8")
    return repo


def self_test_wheel(root: Path, repo: Path) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name in BINARY_STEMS:
        (bin_dir / name).write_bytes(b"\x7fELF fake " + name.encode("ascii") + b"\n")

    version = resolve_version(repo, tag="v0.1.0-experimental.1", version=None)
    if version != "0.1.0a1":
        raise WheelBuildError(f"unexpected resolved version: {version}")

    first = build_wheel(
        repo_root=repo,
        bin_dir=bin_dir,
        platform="linux-amd64",
        version=version,
        out_dir=root / "out-a",
    )
    second = build_wheel(
        repo_root=repo,
        bin_dir=bin_dir,
        platform="linux-amd64",
        version=version,
        out_dir=root / "out-b",
    )
    if (
        first.name
        != "rocm_cli-0.1.0a1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    ):
        raise WheelBuildError(f"unexpected wheel name: {first.name}")
    if first.read_bytes() != second.read_bytes():
        # Byte-identical within one interpreter, which is what CI reproduces;
        # deflate output is not guaranteed identical across zlib versions.
        raise WheelBuildError("consecutive builds are not byte-identical")
    sidecar = first.with_suffix(first.suffix + ".sha256")
    if sidecar.read_text(encoding="ascii") != f"{sha256_file(first)}  {first.name}\n":
        raise WheelBuildError("sha256 sidecar contents are wrong")

    dist_info = "rocm_cli-0.1.0a1.dist-info"
    data_scripts = "rocm_cli-0.1.0a1.data/scripts"
    expected_names = {
        f"{data_scripts}/rocm",
        f"{data_scripts}/rocmd",
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/licenses/LICENSE.TXT",
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.txt",
        f"{dist_info}/RECORD",
    }
    with zipfile.ZipFile(first) as package:
        names = set(package.namelist())
        if names != expected_names:
            raise WheelBuildError(f"unexpected wheel members: {sorted(names)}")
        for name in (f"{data_scripts}/rocm", f"{data_scripts}/rocmd"):
            info = package.getinfo(name)
            attrs = info.external_attr >> 16
            if attrs & 0o7777 != SCRIPT_MODE:
                raise WheelBuildError(f"{name} mode is {oct(attrs & 0o7777)}")
            if attrs & 0o170000 != S_IFREG:
                raise WheelBuildError(f"{name} is missing the S_IFREG bit")
            if info.date_time != ZIP_DATE_TIME:
                raise WheelBuildError(f"{name} has a non-fixed timestamp")
        wheel_meta = package.read(f"{dist_info}/WHEEL").decode("utf-8")
        if "Root-Is-Purelib: false" not in wheel_meta:
            raise WheelBuildError("WHEEL is missing Root-Is-Purelib: false")
        if "Tag: py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64" not in wheel_meta:
            raise WheelBuildError("WHEEL is missing the expected platform tag")
        metadata = package.read(f"{dist_info}/METADATA").decode("utf-8")
        for required in ("Metadata-Version: 2.4", "Name: rocm-cli", "Version: 0.1.0a1"):
            if required not in metadata:
                raise WheelBuildError(f"METADATA is missing {required!r}")

        rows = list(
            csv.reader(package.read(f"{dist_info}/RECORD").decode("utf-8").splitlines())
        )
        listed = {row[0] for row in rows}
        if listed != expected_names:
            raise WheelBuildError(f"RECORD lists {sorted(listed)}")
        for path, digest, size in rows:
            if path == f"{dist_info}/RECORD":
                if digest or size:
                    raise WheelBuildError("RECORD's own row must have empty fields")
                continue
            payload = package.read(path)
            if digest != record_hash(payload):
                raise WheelBuildError(f"RECORD hash mismatch for {path}")
            if int(size) != len(payload):
                raise WheelBuildError(f"RECORD size mismatch for {path}")

    windows_bin = root / "bin-win"
    windows_bin.mkdir()
    expect_error(
        "windows wheel with missing binaries",
        lambda: build_wheel(
            repo_root=repo,
            bin_dir=windows_bin,
            platform="windows-amd64",
            version=version,
            out_dir=root / "out-win",
        ),
    )

    windows_ok_bin = root / "bin-win-ok"
    windows_ok_bin.mkdir()
    for name in BINARY_STEMS:
        (windows_ok_bin / f"{name}.exe").write_bytes(b"MZ fake " + name.encode("ascii"))
    windows_wheel = build_wheel(
        repo_root=repo,
        bin_dir=windows_ok_bin,
        platform="windows-amd64",
        version=version,
        out_dir=root / "out-win-ok",
    )
    if windows_wheel.name != "rocm_cli-0.1.0a1-py3-none-win_amd64.whl":
        raise WheelBuildError(f"unexpected windows wheel name: {windows_wheel.name}")
    with zipfile.ZipFile(windows_wheel) as package:
        names = set(package.namelist())
        expected_windows = {
            f"{data_scripts}/rocm.exe",
            f"{data_scripts}/rocmd.exe",
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/licenses/LICENSE.TXT",
            f"{dist_info}/licenses/THIRD_PARTY_NOTICES.txt",
            f"{dist_info}/RECORD",
        }
        if names != expected_windows:
            raise WheelBuildError(f"unexpected windows wheel members: {sorted(names)}")
        wheel_meta = package.read(f"{dist_info}/WHEEL").decode("utf-8")
        if "Tag: py3-none-win_amd64" not in wheel_meta:
            raise WheelBuildError("windows WHEEL is missing the win_amd64 tag")


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        self_test_tag_mapping()

        if wheel_platform_tag("windows-amd64") != "win_amd64":
            raise WheelBuildError("windows platform tag is wrong")
        if (
            wheel_platform_tag("linux-amd64")
            != "manylinux_2_17_x86_64.manylinux2014_x86_64"
        ):
            raise WheelBuildError("linux platform tag is wrong")
        expect_error(
            "platform darwin-arm64", lambda: wheel_platform_tag("darwin-arm64")
        )

        repo = make_fake_repo(root)
        if workspace_version(repo) != "0.1.0":
            raise WheelBuildError(
                f"workspace version parsed as {workspace_version(repo)}"
            )
        expect_error(
            "workspace version cross-check",
            lambda: resolve_version(repo, tag="v9.9.9", version=None),
        )
        expect_error(
            "explicit version cross-check",
            lambda: resolve_version(repo, tag=None, version="2.0.0"),
        )
        for rejected in ("0.1.0+local", "0.1.0.dev1", "0.1.0.post1", "1!0.1.0"):
            expect_error(
                f"unpublishable explicit version {rejected}",
                lambda value=rejected: resolve_version(repo, tag=None, version=value),
            )
        if resolve_version(repo, tag=None, version="0.1.0a1") != "0.1.0a1":
            raise WheelBuildError("a mapped pre-release version must be accepted")
        expect_error(
            "both tag and version",
            lambda: resolve_version(repo, tag="v0.1.0", version="0.1.0"),
        )
        expect_error(
            "neither tag nor version",
            lambda: resolve_version(repo, tag=None, version=None),
        )
        expect_error(
            "missing manifest", lambda: workspace_version(root / "does-not-exist")
        )
        commented = root / "commented"
        commented.mkdir()
        (commented / "Cargo.toml").write_text(
            '[workspace.package] # release metadata\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        if workspace_version(commented) != "0.1.0":
            raise WheelBuildError("a commented section header must still be parsed")

        self_test_wheel(root, repo)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("wheel builder self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir", type=Path, help="Directory holding the built rocm/rocmd binaries."
    )
    parser.add_argument("--platform", choices=sorted(PLATFORM_TAGS))
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--tag", help="Release git tag, e.g. v0.1.0-rc.1.")
    version_group.add_argument(
        "--version", help="Already-mapped PEP 440 version, e.g. 0.1.0rc1."
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "dist" / "wheels")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--self-test", action="store_true", help="Run offline wheel policy tests."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            with tempfile.TemporaryDirectory(prefix="rocm-wheel-selftest-") as temp:
                run_self_test(Path(temp) / "work")
            return 0
        if args.bin_dir is None:
            raise WheelBuildError("--bin-dir is required")
        if args.platform is None:
            raise WheelBuildError("--platform is required")
        repo_root = args.repo_root.resolve()
        version = resolve_version(repo_root, tag=args.tag, version=args.version)
        wheel = build_wheel(
            repo_root=repo_root,
            bin_dir=args.bin_dir.resolve(),
            platform=args.platform,
            version=version,
            out_dir=args.out_dir.resolve(),
        )
        print(f"wheel: {wheel}")
        return 0
    except (WheelBuildError, OSError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
