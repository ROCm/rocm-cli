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
import json
import re
import shutil
import subprocess
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
# PEP 440, as policy: local versions (`+local`) are illegal in a wheel
# filename, and this project's release tags never spell a dev, post or epoch
# version, so a version of that shape is a mistake rather than a release.
# `--version` bypasses the tag mapper, so it is validated against the same
# shape rather than trusted.
#
# The wheel's version is the tag's. It is deliberately not compared with
# `[workspace.package] version`: that number is shared by every crate and is
# not bumped per release (see `apps/rocm/build.rs`), and `rocm --version`
# reports the tag, so the release job's smoke test is what ties the installed
# binaries to the tag.
PUBLISHABLE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")

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


class UnmappableTagError(WheelBuildError):
    """The release tag has no PEP 440 spelling, so no wheel exists for it.

    Kept apart from the other build errors because it is the one failure that
    is not a defect: the release workflow accepts any `vX.Y.Z-<pre>` tag (a
    `-preview.1`, for one), and PEP 440 cannot express most of them. Release CI
    passes `--skip-unmappable-tag` so such a tag ships its archives without a
    wheel instead of failing the whole release. Every other error still fails.
    """


def fail(message: str) -> None:
    print(f"wheel build failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def pep440_version_from_tag(tag: str) -> str:
    """Map a release git tag onto its PEP 440 version.

    `vX.Y.Z` -> `X.Y.Z`, `-alpha.N`/`-experimental.N` -> `X.Y.ZaN`,
    `-beta.N` -> `X.Y.ZbN`, `-rc.N` -> `X.Y.ZrcN`. Every other shape (nightly,
    staging and `-preview.N` tags included) raises [`UnmappableTagError`].

    `-alpha.N` and `-experimental.N` intentionally collapse onto the same
    `aN` form, because PEP 440 has one alpha marker and both tag spellings mean
    the same thing here. The consequence is that only one of the two spellings
    may be used per serial: pushing both `v1.2.3-experimental.1` and
    `v1.2.3-alpha.1` produces the same wheel version twice, and PyPI refuses
    the second upload because a released version can never be replaced.
    """
    match = TAG_RE.fullmatch(tag.strip())
    if match is None:
        raise UnmappableTagError(f"tag is not a publishable release tag: {tag}")
    release = match.group("release")
    kind = match.group("kind")
    if kind is None:
        return release
    serial = int(match.group("serial"))
    return f"{release}{PRE_RELEASE_MARKERS[kind]}{serial}"


def resolve_version(*, tag: str | None, version: str | None) -> str:
    """Resolve the wheel version from a release tag or an already-mapped version."""
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
        *(f"License-File: {name}" for name in LICENSE_FILES),
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
    # The filename compresses a platform tag set with dots; the `WHEEL` file
    # lists each tag of the set on its own line.
    tags = "".join(f"Tag: py3-none-{tag}\n" for tag in platform_tag.split("."))
    return f"Wheel-Version: 1.0\nGenerator: {GENERATOR}\nRoot-Is-Purelib: false\n{tags}"


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


# The oldest glibc the Linux wheel's platform tag promises (manylinux_2_17 is
# manylinux2014). The tag is asserted by PLATFORM_TAGS and verified, not
# trusted, by `check_manylinux_report`.
MANYLINUX_GLIBC = (2, 17)
LEGACY_MANYLINUX_GLIBC = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}
MANYLINUX_TAG_RE = re.compile(
    r"^(?:manylinux_(?P<major>\d+)_(?P<minor>\d+)|(?P<legacy>manylinux(?:1|2010|2014)))"
    r"_x86_64$"
)


def check_manylinux_report(report: str) -> str:
    """Judge `auditwheel show --json` output against the Linux wheel's tag.

    Returns the platform tag auditwheel says the wheel's binaries need. Raises
    [`WheelBuildError`] when that tag needs a newer glibc than manylinux_2_17,
    when the binaries use CPU extensions beyond the baseline, or when the
    report cannot be read as the object this check understands: an
    unreadable report is a failed check, never a passed one.
    """
    try:
        data = json.loads(report)
    except json.JSONDecodeError as error:
        raise WheelBuildError(f"auditwheel report is not JSON: {error}") from None
    tag = data.get("overall_tag") if isinstance(data, dict) else None
    if not isinstance(tag, str):
        raise WheelBuildError("auditwheel report has no overall_tag")
    match = MANYLINUX_TAG_RE.fullmatch(tag)
    if match is None:
        raise WheelBuildError(f"auditwheel says the wheel needs {tag}, not manylinux")
    legacy = match.group("legacy")
    glibc = (
        LEGACY_MANYLINUX_GLIBC[legacy]
        if legacy is not None
        else (int(match.group("major")), int(match.group("minor")))
    )
    if glibc > MANYLINUX_GLIBC:
        raise WheelBuildError(
            f"the wheel is tagged manylinux_2_17 but its binaries need {tag}"
        )
    if data.get("unsupported_isa") is not False:
        raise WheelBuildError("auditwheel reports CPU extensions beyond the baseline")
    return tag


def check_manylinux(wheel: Path, auditwheel: str) -> str:
    """Run `auditwheel show --json` on `wheel` and judge it."""
    try:
        result = subprocess.run(
            [auditwheel, "show", "--json", str(wheel)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WheelBuildError(f"cannot run {auditwheel}: {error}") from None
    if result.returncode != 0:
        raise WheelBuildError(
            f"auditwheel show failed ({result.returncode}): {result.stderr.strip()}"
        )
    return check_manylinux_report(result.stdout)


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
        "v0.3.0": "0.3.0",
        "v0.3.0-rc.1": "0.3.0rc1",
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
        "v0.1.0-preview.1",
        "v1.2.3+build.5",
        "1.2",
        "v1.2.3.4",
        "",
    ):
        try:
            pep440_version_from_tag(bad)
        except UnmappableTagError:
            continue
        raise WheelBuildError(f"tag {bad!r} must raise UnmappableTagError")


# Real `auditwheel show --json` reports (auditwheel 6.8.2, run in the
# quay.io/pypa/manylinux2014_x86_64 image the release builds in, on a wheel made
# by build_wheel.py). Captured, not written by hand, so the keys the check reads
# are the real ones. This one is a program built on Ubuntu 24.04 (glibc 2.39)
# that calls `strlcpy`: the report this check has to reject.
REAL_REPORT_NEEDS_GLIBC_2_38 = """{
    "version": 1,
    "wheel": "rocm_cli-0.0.1rc1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "pure": false,
    "overall_tag": "manylinux_2_38_x86_64",
    "sym_tag": "manylinux_2_38_x86_64",
    "pyfpe": false,
    "ucs2": false,
    "unsupported_isa": false,
    "versioned_symbols": {
        "libc.so.6": [
            "GLIBC_2.2.5",
            "GLIBC_2.34",
            "GLIBC_2.38",
            "GLIBC_2.4"
        ]
    },
    "external_libs": {},
    "policy_upgrades": {}
}"""


# The same command on the `rocm` and `rocmd` the release container builds
# (`cargo build --release -p rocm -p rocmd`): the report it has to accept.
REAL_REPORT_RELEASE_BINARIES = """{
    "version": 1,
    "wheel": "rocm_cli-0.0.1rc1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "pure": false,
    "overall_tag": "manylinux_2_17_x86_64",
    "sym_tag": "manylinux_2_17_x86_64",
    "pyfpe": false,
    "ucs2": false,
    "unsupported_isa": false,
    "versioned_symbols": {
        "libc.so.6": [
            "GLIBC_2.10",
            "GLIBC_2.14",
            "GLIBC_2.15",
            "GLIBC_2.16",
            "GLIBC_2.2.5",
            "GLIBC_2.3",
            "GLIBC_2.3.2",
            "GLIBC_2.3.4",
            "GLIBC_2.4",
            "GLIBC_2.5",
            "GLIBC_2.6",
            "GLIBC_2.7",
            "GLIBC_2.9"
        ],
        "libdl.so.2": [
            "GLIBC_2.2.5"
        ],
        "libgcc_s.so.1": [
            "GCC_3.0",
            "GCC_3.3",
            "GCC_4.2.0"
        ],
        "libm.so.6": [
            "GLIBC_2.2.5"
        ],
        "libpthread.so.0": [
            "GLIBC_2.12",
            "GLIBC_2.2.5"
        ],
        "librt.so.1": [
            "GLIBC_2.2.5"
        ]
    },
    "external_libs": {},
    "policy_upgrades": {}
}"""


def self_test_manylinux_report() -> None:
    try:
        check_manylinux_report(REAL_REPORT_NEEDS_GLIBC_2_38)
    except WheelBuildError as error:
        if "manylinux_2_38_x86_64" not in str(error):
            raise WheelBuildError(f"wrong rejection reason: {error}") from None
    else:
        raise WheelBuildError("a glibc 2.38 report must be rejected")
    if check_manylinux_report(REAL_REPORT_RELEASE_BINARIES) != "manylinux_2_17_x86_64":
        raise WheelBuildError("the release binaries' report must pass")

    def report(tag: object, isa: object = False) -> str:
        return json.dumps({"overall_tag": tag, "unsupported_isa": isa})

    for good in (
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
        "manylinux_2_12_x86_64",
        "manylinux2010_x86_64",
        "manylinux_2_5_x86_64",
        "manylinux1_x86_64",
    ):
        if check_manylinux_report(report(good)) != good:
            raise WheelBuildError(f"{good} must pass")
    # A binary built on a current glibc needs a tag newer than 2_17, which is the
    # failure this check exists to catch (`REAL_REPORT_NEEDS_GLIBC_2_38`).
    for bad in (
        "manylinux_2_38_x86_64",
        "manylinux_2_18_x86_64",
        "manylinux_3_0_x86_64",
        "musllinux_1_2_x86_64",
        "manylinux_2_17_aarch64",
        "linux_x86_64",
        "",
        None,
        17,
    ):
        expect_error(
            f"overall_tag {bad!r}", lambda bad=bad: check_manylinux_report(report(bad))
        )
    expect_error(
        "unsupported ISA",
        lambda: check_manylinux_report(report("manylinux_2_17_x86_64", True)),
    )
    expect_error(
        "missing ISA field",
        lambda: check_manylinux_report('{"overall_tag": "manylinux_2_17_x86_64"}'),
    )
    expect_error("not JSON", lambda: check_manylinux_report("The wheel is consistent"))
    expect_error("not an object", lambda: check_manylinux_report("[]"))
    expect_error(
        "missing auditwheel",
        lambda: check_manylinux(Path("x.whl"), "no-such-auditwheel-here"),
    )


def make_fake_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    for name in LICENSE_FILES:
        (repo / name).write_text(f"{name} test content\n", encoding="utf-8")
    return repo


def self_test_wheel(root: Path, repo: Path) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name in BINARY_STEMS:
        (bin_dir / name).write_bytes(b"\x7fELF fake " + name.encode("ascii") + b"\n")

    version = resolve_version(tag="v0.1.0-experimental.1", version=None)
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
        for tag in ("manylinux_2_17_x86_64", "manylinux2014_x86_64"):
            if f"Tag: py3-none-{tag}\n" not in wheel_meta:
                raise WheelBuildError(f"WHEEL is missing the {tag} tag line")
        metadata = package.read(f"{dist_info}/METADATA").decode("utf-8")
        license_lines = tuple(f"License-File: {name}" for name in LICENSE_FILES)
        for required in (
            "Metadata-Version: 2.4",
            "Name: rocm-cli",
            "Version: 0.1.0a1",
            *license_lines,
        ):
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
        if "Tag: py3-none-win_amd64\n" not in wheel_meta:
            raise WheelBuildError("windows WHEEL is missing the win_amd64 tag")


def self_test_unmappable_tag_policy(root: Path, repo: Path) -> None:
    """A tag with no wheel spelling skips only when the caller opts in, and
    nothing else is forgiven by that opt-in."""
    bin_dir = root / "bin"
    out_dir = root / "out-skip"
    common = [
        "--bin-dir",
        str(bin_dir),
        "--platform",
        "linux-amd64",
        "--repo-root",
        str(repo),
        "--out-dir",
        str(out_dir),
    ]
    preview = ["--tag", "v0.1.0-preview.1"]
    if main([*common, *preview, "--skip-unmappable-tag"]) != 0:
        raise WheelBuildError("an unmappable tag with the skip flag must exit 0")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise WheelBuildError("a skipped tag must not leave a wheel behind")
    try:
        main([*common, *preview])
    except SystemExit as error:
        if error.code != 1:
            raise WheelBuildError(
                f"an unmappable tag must exit 1, got {error.code}"
            ) from None
    else:
        raise WheelBuildError("an unmappable tag without the skip flag must fail")
    # The opt-in covers the tag shape only: a mappable tag whose binaries are
    # missing is a real release defect and still fails.
    missing = [*common, "--tag", "v0.1.0", "--skip-unmappable-tag"]
    missing[missing.index(str(bin_dir))] = str(root / "no-such-bin-dir")
    try:
        main(missing)
    except SystemExit as error:
        if error.code != 1:
            raise WheelBuildError(
                f"missing binaries must exit 1, got {error.code}"
            ) from None
    else:
        raise WheelBuildError("the skip flag must not forgive a build failure")


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        self_test_tag_mapping()
        self_test_manylinux_report()

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
        for rejected in ("0.1.0+local", "0.1.0.dev1", "0.1.0.post1", "1!0.1.0"):
            expect_error(
                f"unpublishable explicit version {rejected}",
                lambda value=rejected: resolve_version(tag=None, version=value),
            )
        if resolve_version(tag=None, version="0.1.0a1") != "0.1.0a1":
            raise WheelBuildError("a mapped pre-release version must be accepted")
        if resolve_version(tag="v0.3.0", version=None) != "0.3.0":
            raise WheelBuildError("a release tag need not match the workspace version")
        expect_error(
            "both tag and version",
            lambda: resolve_version(tag="v0.1.0", version="0.1.0"),
        )
        expect_error(
            "neither tag nor version",
            lambda: resolve_version(tag=None, version=None),
        )

        self_test_wheel(root, repo)
        self_test_unmappable_tag_policy(root, repo)
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
        "--skip-unmappable-tag",
        action="store_true",
        help=(
            "When --tag has no PEP 440 spelling (for example v0.1.0-preview.1), "
            "print a notice, write no wheel and exit 0 instead of failing. Any "
            "other error still fails."
        ),
    )
    parser.add_argument(
        "--check-manylinux",
        type=Path,
        metavar="WHEEL",
        help="Verify the Linux wheel's binaries satisfy manylinux_2_17 with auditwheel.",
    )
    parser.add_argument(
        "--auditwheel", default="auditwheel", help="auditwheel executable to run."
    )
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
        if args.check_manylinux is not None:
            tag = check_manylinux(args.check_manylinux, args.auditwheel)
            print(f"manylinux check: {args.check_manylinux.name} needs {tag}")
            return 0
        if args.bin_dir is None:
            raise WheelBuildError("--bin-dir is required")
        if args.platform is None:
            raise WheelBuildError("--platform is required")
        repo_root = args.repo_root.resolve()
        try:
            version = resolve_version(tag=args.tag, version=args.version)
        except UnmappableTagError as error:
            if not args.skip_unmappable_tag:
                raise
            print(f"wheel skipped: {error}; no Python wheel is built for this tag")
            return 0
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
