#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Verify rocm-cli release dist assets before publication."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".tar.gz", ".zip")
ROCM_RELEASE_ASSET_RE = re.compile(
    r"^rocm-cli-(?:(?:v[0-9][A-Za-z0-9._+-]*|nightly(?:-[0-9]{8}-[0-9A-Fa-f]+)?)-)?"
    r"(?P<os>linux|windows)-amd64(?P<suffix>\.tar\.gz|\.zip)$"
)
# First-party engines are built into rocm/rocm.exe (run in-process); the
# standalone rocm-engine-* binaries are an external plugin fallback and are not
# part of the shipped bundle.
LINUX_REQUIRED = (
    "bin/rocm",
    "bin/rocmd",
    "README.md",
    "LICENSE.TXT",
    "install.sh",
)
WINDOWS_REQUIRED = (
    "bin/rocm.exe",
    "bin/rocmd.exe",
    "README.md",
    "LICENSE.TXT",
    "install.ps1",
)
LINUX_EXECUTABLES = (
    "bin/rocm",
    "bin/rocmd",
    "install.sh",
)
SIGNING_PUBLIC_KEY_PATH_ENV = "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH"
SIGNING_PUBLIC_KEY_ENV = "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"
PRODUCTION_TRUST_ENV_NAMES = (
    SIGNING_PUBLIC_KEY_PATH_ENV,
    SIGNING_PUBLIC_KEY_ENV,
    "ROCM_CLI_METADATA_PUBLIC_KEY_PATH",
    "ROCM_CLI_METADATA_PUBLIC_KEY_PEM",
    "ROCM_CLI_MODEL_RECIPE_INDEX_PATH",
    "ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH",
    "ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH",
)


class ReadinessError(Exception):
    """A release artifact failed a readiness check."""


def fail(message: str) -> None:
    print(f"release readiness failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def validate_requested_asset_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ReadinessError(f"asset name must be a file name, got: {name!r}")
    if "/" in name or "\\" in name:
        raise ReadinessError(f"asset name must not contain path separators: {name}")
    if ":" in name:
        raise ReadinessError(
            f"asset name must not contain drive or URI separators: {name}"
        )
    if Path(name).name != name:
        raise ReadinessError(f"asset name must be a plain file name: {name}")
    if not is_archive(Path(name)):
        raise ReadinessError(f"asset name is not a supported release archive: {name}")


def publishable_asset_base_name(name: str) -> str | None:
    lower_name = name.lower()
    if any(lower_name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
        return name
    for sidecar_suffix in (".sha256", ".sig"):
        if lower_name.endswith(sidecar_suffix):
            base_name = name[: -len(sidecar_suffix)]
            if is_archive(Path(base_name)):
                return base_name
    return None


def validate_exact_dist_assets(
    dist: Path,
    archives: list[Path],
    *,
    expect_signatures: bool,
) -> list[str]:
    if not dist.is_dir():
        raise ReadinessError(f"dist directory does not exist: {dist}")

    expected_names: set[str] = set()
    for archive in archives:
        expected_names.add(archive.name)
        expected_names.add(f"{archive.name}.sha256")
        if expect_signatures:
            expected_names.add(f"{archive.name}.sig")

    publishable_names: set[str] = set()
    for path in dist.iterdir():
        if not path.is_file():
            continue
        if publishable_asset_base_name(path.name) is not None:
            publishable_names.add(path.name)

    extra_names = sorted(publishable_names - expected_names)
    if extra_names:
        joined = ", ".join(extra_names)
        raise ReadinessError(f"dist contains unverified publishable asset(s): {joined}")

    missing_names = sorted(expected_names - publishable_names)
    if missing_names:
        joined = ", ".join(missing_names)
        raise ReadinessError(f"dist is missing expected publishable asset(s): {joined}")

    return [f"exact dist asset set ok: {len(expected_names)} file(s)"]


def validate_rocm_release_asset_name(archive: Path) -> None:
    match = ROCM_RELEASE_ASSET_RE.fullmatch(archive.name)
    if match is None:
        raise ReadinessError(
            f"release archive name is not a supported rocm-cli asset name: {archive.name}"
        )
    platform = match.group("os")
    suffix = match.group("suffix")
    if platform == "linux" and suffix != ".tar.gz":
        raise ReadinessError(f"Linux release archive must be .tar.gz: {archive.name}")
    if platform == "windows" and suffix != ".zip":
        raise ReadinessError(f"Windows release archive must be .zip: {archive.name}")


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_sidecar(path: Path) -> tuple[str, str | None]:
    if not path.is_file():
        raise ReadinessError(f"missing checksum sidecar: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ReadinessError(f"checksum sidecar is empty: {path}")
    parts = lines[0].split()
    digest = parts[0].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ReadinessError(
            f"checksum sidecar does not start with a sha256 digest: {path}"
        )
    recorded_name = parts[1] if len(parts) > 1 else None
    return digest, recorded_name


def normalized_archive_name(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."}:
        raise ReadinessError(f"archive contains an empty path entry: {name!r}")
    if normalized.startswith("/") or name.startswith("\\"):
        raise ReadinessError(f"archive contains an absolute path entry: {name!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ReadinessError(f"archive contains an unsafe path entry: {name!r}")
    if ":" in parts[0]:
        raise ReadinessError(f"archive contains a drive-qualified path entry: {name!r}")
    return normalized


def relative_bundle_path(path: str, top_level: str) -> str | None:
    if path == top_level:
        return None
    prefix = f"{top_level}/"
    if not path.startswith(prefix):
        raise ReadinessError(f"archive path escaped top-level directory: {path}")
    return path[len(prefix) :]


def validate_top_level(paths: set[str], archive: Path) -> str:
    top_levels = {path.split("/", 1)[0] for path in paths}
    if len(top_levels) != 1:
        joined = ", ".join(sorted(top_levels))
        raise ReadinessError(
            f"{archive.name} must contain exactly one top-level directory; found {joined}"
        )
    return next(iter(top_levels))


def required_for_archive(archive: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    name = archive.name.lower()
    if name.endswith(".zip"):
        return WINDOWS_REQUIRED, ()
    if name.endswith(".tar.gz"):
        return LINUX_REQUIRED, LINUX_EXECUTABLES
    raise ReadinessError(f"unsupported archive extension: {archive.name}")


def validate_tar_archive(archive: Path) -> list[str]:
    required, executables = required_for_archive(archive)
    all_paths: set[str] = set()
    files: dict[str, tarfile.TarInfo] = {}
    try:
        with tarfile.open(archive, "r:gz") as package:
            for member in package.getmembers():
                normalized = normalized_archive_name(member.name)
                if normalized in all_paths:
                    raise ReadinessError(
                        f"{archive.name} contains duplicate path: {normalized}"
                    )
                all_paths.add(normalized)
                if member.isdev() or member.issym() or member.islnk():
                    raise ReadinessError(
                        f"{archive.name} contains unsupported special entry: {normalized}"
                    )
                if member.isfile():
                    files[normalized] = member
    except tarfile.TarError as error:
        raise ReadinessError(
            f"failed to read tar archive {archive.name}: {error}"
        ) from error

    top_level = validate_top_level(all_paths, archive)
    bundle_files: dict[str, tarfile.TarInfo] = {}
    for path, member in files.items():
        relative = relative_bundle_path(path, top_level)
        if relative is not None:
            bundle_files[relative] = member

    for required_file in required:
        member = bundle_files.get(required_file)
        if member is None:
            raise ReadinessError(f"{archive.name} is missing {required_file}")
        if member.size <= 0:
            raise ReadinessError(
                f"{archive.name} contains empty required file {required_file}"
            )
    for executable in executables:
        member = bundle_files.get(executable)
        if member is None:
            continue
        if member.mode & 0o111 == 0:
            raise ReadinessError(
                f"{archive.name} required executable is not executable: {executable}"
            )
    return [f"bundle contents ok: {archive.name}"]


def zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def validate_zip_archive(archive: Path) -> list[str]:
    required, _executables = required_for_archive(archive)
    all_paths: set[str] = set()
    files: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                normalized = normalized_archive_name(info.filename)
                if normalized in all_paths:
                    raise ReadinessError(
                        f"{archive.name} contains duplicate path: {normalized}"
                    )
                all_paths.add(normalized)
                if zip_entry_is_symlink(info):
                    raise ReadinessError(
                        f"{archive.name} contains unsupported symlink entry: {normalized}"
                    )
                if not info.is_dir():
                    files[normalized] = info
    except zipfile.BadZipFile as error:
        raise ReadinessError(
            f"failed to read zip archive {archive.name}: {error}"
        ) from error

    top_level = validate_top_level(all_paths, archive)
    bundle_files: dict[str, zipfile.ZipInfo] = {}
    for path, info in files.items():
        relative = relative_bundle_path(path, top_level)
        if relative is not None:
            bundle_files[relative] = info

    for required_file in required:
        info = bundle_files.get(required_file)
        if info is None:
            raise ReadinessError(f"{archive.name} is missing {required_file}")
        if info.file_size <= 0:
            raise ReadinessError(
                f"{archive.name} contains empty required file {required_file}"
            )
    return [f"bundle contents ok: {archive.name}"]


def validate_archive_contents(archive: Path) -> list[str]:
    if archive.name.lower().endswith(".tar.gz"):
        return validate_tar_archive(archive)
    if archive.name.lower().endswith(".zip"):
        return validate_zip_archive(archive)
    raise ReadinessError(f"unsupported archive extension: {archive.name}")


def verify_signature(archive: Path, signature: Path, public_key: Path | None) -> None:
    """Verify ``archive`` against ``public_key`` via ``cargo xtask verify``.

    When ``public_key`` is ``None`` the ``--public-key`` flag is omitted and
    ``cargo xtask verify`` reads the key from ``ROCM_CLI_SIGNING_PUBLIC_KEY_PEM``
    (the inline PEM release/nightly CI wire from the signing-key secret), so no
    temporary key file has to be materialized here.
    """
    if public_key is not None and not public_key.is_file():
        raise ReadinessError(f"public key does not exist: {public_key}")
    cargo = shutil.which("cargo")
    if cargo is None:
        raise ReadinessError("cargo is required for signature verification")
    key_args = (
        ["--public-key", str(public_key.resolve())] if public_key is not None else []
    )
    # Resolve to absolute paths because the subprocess runs from the repo root
    # (so the `cargo xtask` alias resolves); relative paths would otherwise be
    # interpreted against the repo root rather than the caller's working directory.
    completed = subprocess.run(
        [
            cargo,
            "xtask",
            "verify",
            *key_args,
            "--in",
            str(archive.resolve()),
            "--signature",
            str(signature.resolve()),
        ],
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise ReadinessError(
            f"signature verification failed for {archive.name}{suffix}"
        )


def validate_archive(
    archive: Path,
    *,
    require_signatures: bool,
    verify: bool,
    public_key: Path | None,
    require_rocm_asset_names: bool,
) -> list[str]:
    messages: list[str] = []
    if not archive.is_file():
        raise ReadinessError(f"missing archive: {archive}")
    if archive.stat().st_size <= 0:
        raise ReadinessError(f"archive is empty: {archive}")
    if require_rocm_asset_names:
        validate_rocm_release_asset_name(archive)
        messages.append(f"asset name ok: {archive.name}")

    messages.extend(validate_archive_contents(archive))

    sidecar = Path(f"{archive}.sha256")
    expected, recorded_name = parse_sha256_sidecar(sidecar)
    actual = sha256_hex(archive)
    if expected != actual:
        raise ReadinessError(
            f"checksum mismatch for {archive.name}: sidecar={expected} actual={actual}"
        )
    if recorded_name is not None and Path(recorded_name).name != archive.name:
        raise ReadinessError(
            f"checksum sidecar {sidecar.name} names {recorded_name}, expected {archive.name}"
        )
    messages.append(f"checksum ok: {archive.name}")

    signature = Path(f"{archive}.sig")
    if require_signatures or verify:
        if not signature.is_file():
            raise ReadinessError(f"missing signature sidecar: {signature}")
        if signature.stat().st_size <= 0:
            raise ReadinessError(f"signature sidecar is empty: {signature}")
        messages.append(f"signature present: {archive.name}.sig")

    if verify:
        verify_signature(archive, signature, public_key)
        messages.append(f"signature verified: {archive.name}")

    return messages


def discover_archives(dist: Path, asset_names: list[str]) -> list[Path]:
    if asset_names:
        seen_names: set[str] = set()
        for name in asset_names:
            validate_requested_asset_name(name)
            if name in seen_names:
                raise ReadinessError(f"asset name was requested more than once: {name}")
            seen_names.add(name)
        return [dist / name for name in asset_names]
    if not dist.is_dir():
        raise ReadinessError(f"dist directory does not exist: {dist}")
    archives = sorted(
        path for path in dist.iterdir() if path.is_file() and is_archive(path)
    )
    if not archives:
        raise ReadinessError(f"no release archives found in {dist}")
    return archives


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return Path(value)


def env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def verification_is_required(
    require_signatures: bool,
    require_production_trust: bool,
    explicit_public_key: Path | None,
) -> bool:
    """Whether this run must cryptographically verify signatures.

    Requiring signatures means requiring they verify. Checking only that a
    ``.sig`` exists would pass an artifact signed by the wrong key, truncated,
    or corrupted -- so whenever signatures are required, so is verification,
    and a key that cannot be resolved is a hard failure rather than a quiet
    downgrade to the presence check.

    Separate from ``main`` so ``--self-test`` can exercise the decision
    directly; ``main`` returns on the ``--self-test`` branch before the gate.
    """
    return (
        require_signatures
        or require_production_trust
        or explicit_public_key is not None
    )


def resolve_verification(
    require_signatures: bool,
    require_production_trust: bool,
    explicit_public_key: Path | None,
) -> tuple[bool, Path | None, str | None]:
    """The whole verify wiring: decide, then resolve the key when required.

    Returns ``(verify, public_key, key_source)``; ``key_source`` is ``None``
    when no verification is required.

    Deciding and resolving belong together: a run that requires verification
    but resolves no key would verify nothing while reporting success. Keeping
    both here means ``main`` holds no copy of the sequence.
    """
    verify = verification_is_required(
        require_signatures, require_production_trust, explicit_public_key
    )
    if not verify:
        return False, None, None
    public_key, key_source = resolve_signing_key(explicit_public_key)
    return True, public_key, key_source


def resolve_signing_key(explicit: Path | None) -> tuple[Path | None, str]:
    """Resolve the public key release signatures are verified against.

    Returns the key path and a label naming where it came from. A ``None`` path
    means `cargo xtask verify` reads the inline PEM from
    ``ROCM_CLI_SIGNING_PUBLIC_KEY_PEM`` itself, so no temporary key file has to
    be materialized here.

    A key file path is preferred over the inline PEM, matching how `install.sh`
    and `cargo xtask package` resolve their signing keys. A path that does not
    exist is an error from `verify_signature`, not a reason to fall through to
    the next source: silently verifying against a different key than the
    operator named is the failure mode this gate exists to prevent.

    Raises when no key resolves. That is the point of this function: GitHub
    expands an unset secret to the empty string, so a missing key is
    indistinguishable from a deliberately absent one. Verification has to fail
    closed, or rotating the secret away would silently downgrade the release
    gate to "a .sig file exists".
    """
    if explicit is not None:
        return explicit, f"--public-key {explicit}"
    if (path := env_path(SIGNING_PUBLIC_KEY_PATH_ENV)) is not None:
        return path, f"${SIGNING_PUBLIC_KEY_PATH_ENV} ({path})"
    if env_text(SIGNING_PUBLIC_KEY_ENV) is not None:
        return None, f"${SIGNING_PUBLIC_KEY_ENV}"
    raise ReadinessError(
        "signature verification requires a release signing public key: pass "
        f"--public-key or set {SIGNING_PUBLIC_KEY_PATH_ENV} or "
        f"{SIGNING_PUBLIC_KEY_ENV}. Verification is requested by "
        "--require-signatures, --require-production-trust, --public-key, or the "
        "ROCM_CLI_REQUIRE_SIGNATURE / ROCM_CLI_REQUIRE_PRODUCTION_TRUST "
        "environment variables -- the last of which packaging steps export."
    )


def require_any(label: str, names: list[str]) -> None:
    if any(env_text(name) for name in names):
        return
    joined = " or ".join(names)
    raise ReadinessError(f"production trust requires {label}: set {joined}")


def require_existing_env_path(name: str) -> Path:
    path = env_path(name)
    if path is None:
        raise ReadinessError(f"production trust requires {name}")
    if not path.is_file():
        raise ReadinessError(f"{name} does not point to a file: {path}")
    return path


def validate_production_trust() -> list[str]:
    """Validate only explicit owner-provided production trust inputs."""

    messages: list[str] = []
    require_any(
        "the release signing public key",
        ["ROCM_CLI_SIGNING_PUBLIC_KEY_PATH", "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"],
    )
    if (
        path := env_path("ROCM_CLI_SIGNING_PUBLIC_KEY_PATH")
    ) is not None and not path.is_file():
        raise ReadinessError(
            f"ROCM_CLI_SIGNING_PUBLIC_KEY_PATH does not point to a file: {path}"
        )
    messages.append("release signing public key configured")

    require_any(
        "the runtime metadata public key",
        ["ROCM_CLI_METADATA_PUBLIC_KEY_PATH", "ROCM_CLI_METADATA_PUBLIC_KEY_PEM"],
    )
    if (
        path := env_path("ROCM_CLI_METADATA_PUBLIC_KEY_PATH")
    ) is not None and not path.is_file():
        raise ReadinessError(
            f"ROCM_CLI_METADATA_PUBLIC_KEY_PATH does not point to a file: {path}"
        )
    messages.append("runtime metadata public key configured")

    index_path = require_existing_env_path("ROCM_CLI_MODEL_RECIPE_INDEX_PATH")
    signature_path = env_path("ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH")
    if signature_path is None:
        signature_path = Path(f"{index_path}.sig")
    if not signature_path.is_file():
        raise ReadinessError(
            f"model recipe index signature is missing: {signature_path}"
        )
    require_existing_env_path("ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH")
    messages.append("model recipe index, signature, and public key configured")

    return messages


def validate_release(
    dist: Path,
    *,
    assets: list[str],
    require_signatures: bool,
    verify: bool,
    public_key: Path | None,
    require_production_trust: bool,
    require_rocm_asset_names: bool,
    require_exact_assets: bool,
) -> list[str]:
    archives = discover_archives(dist, assets)
    messages: list[str] = []
    if require_exact_assets:
        if not assets:
            raise ReadinessError("--require-exact-assets requires at least one --asset")
        messages.extend(
            validate_exact_dist_assets(
                dist,
                archives,
                expect_signatures=require_signatures or verify,
            )
        )
    for archive in archives:
        messages.extend(
            validate_archive(
                archive,
                require_signatures=require_signatures,
                verify=verify,
                public_key=public_key,
                require_rocm_asset_names=require_rocm_asset_names,
            )
        )
    if require_production_trust:
        messages.extend(validate_production_trust())
    return messages


def write_sha(path: Path, *, archive_name: str | None = None) -> None:
    target = archive_name or path.name
    Path(f"{path}.sha256").write_text(
        f"{sha256_hex(path)}  {target}\n", encoding="ascii"
    )


def add_tar_file(
    package: tarfile.TarFile, root: str, relative: str, executable: bool = False
) -> None:
    data = f"test content for {relative}\n".encode()
    info = tarfile.TarInfo(f"{root}/{relative}")
    info.size = len(data)
    info.mode = 0o755 if executable else 0o644
    package.addfile(info, io.BytesIO(data))


def create_test_tar(path: Path, root: str) -> None:
    with tarfile.open(path, "w:gz") as package:
        root_info = tarfile.TarInfo(root)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        package.addfile(root_info)
        for required in LINUX_REQUIRED:
            add_tar_file(
                package, root, required, executable=required in LINUX_EXECUTABLES
            )


def create_test_zip(path: Path, root: str) -> None:
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(f"{root}/", "")
        for required in WINDOWS_REQUIRED:
            package.writestr(f"{root}/{required}", f"test content for {required}\n")


def expect_failure(label: str, func) -> None:
    try:
        func()
    except ReadinessError:
        print(f"release readiness self-test: {label} rejected as expected")
        return
    raise ReadinessError(f"{label} unexpectedly passed")


def run_with_env(values: dict[str, str | Path | None], func):
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        return func()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _assert_verification_wiring(expected_key: Path) -> None:
    """`resolve_verification` must decide AND resolve, not just decide."""
    verify, public_key, key_source = resolve_verification(True, False, None)
    if not verify:
        raise ReadinessError(
            "--require-signatures must force verification through the wiring main uses"
        )
    if public_key != expected_key or not key_source:
        raise ReadinessError(
            f"verification was required but no key was resolved: {public_key!r} / {key_source!r}; "
            "a required verification that silently resolves no key is the fail-open this gate closes"
        )
    off_verify, off_key, off_source = resolve_verification(False, False, None)
    if off_verify or off_key is not None or off_source is not None:
        raise ReadinessError(
            "verification must stay off, and resolve no key, when nothing asks for it"
        )


def _run_main(dist: Path, *flags: str) -> tuple[int | None, str, str]:
    """Drive `main` in-process.

    Returns ``(exit code or None if it returned, stdout, stderr)``. Both streams
    are captured: stderr so a case expecting failure does not print
    ``release readiness failed: ...`` into the log of a passing run, and stdout
    so cases can assert on what the run claimed it did.
    """
    saved_argv = sys.argv
    sys.argv = ["release_readiness.py", "--dist", str(dist), *flags]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            try:
                main()
            except SystemExit as exit_error:
                code = exit_error.code
                return (
                    (code if isinstance(code, int) else 1),
                    out.getvalue(),
                    err.getvalue(),
                )
        return None, out.getvalue(), err.getvalue()
    finally:
        sys.argv = saved_argv


def _assert_main_refuses_without_a_key(dist: Path) -> None:
    """`--require-signatures` with no key must fail, and say so."""
    code, _stdout, stderr = _run_main(dist, "--require-signatures")
    if code is None:
        raise ReadinessError(
            "main accepted a signature-required run with no key configured; "
            "that is the fail-open this gate exists to close"
        )
    # The reason matters, not just the exit code: a `main` that skipped the
    # gate and happened to fail for some unrelated reason would otherwise pass.
    if "requires a release signing public key" not in stderr:
        raise ReadinessError(
            "main failed, but not on the missing signing key -- the gate case is "
            f"no longer exercising what it claims. stderr was: {stderr.strip()!r}"
        )


def _run_main_verifying(
    dist: Path, *flags: str, verification_fails: bool = False
) -> tuple[int | None, str, str, list[list[str]]]:
    """Drive `main` with `cargo xtask verify` intercepted at the subprocess call.

    Returns `(exit code, stdout, stderr, argvs)`, where `argvs` holds every
    command verification actually ran.

    Interception sits at the subprocess boundary rather than stubbing
    `verify_signature`, so the real argv is built and the `--public-key` handoff
    is exercised; stubbing the function leaves `key_args` unreachable, and
    dropping it would stay green while verification silently fell back to
    whatever the inline PEM holds.
    """
    argvs: list[list[str]] = []
    returncode = 1 if verification_fails else 0

    class _Completed:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = "stub: signature did not verify" if returncode else ""

    module = sys.modules[__name__]
    original_run, original_which = module.subprocess.run, module.shutil.which
    module.subprocess.run = lambda argv, **_kwargs: (
        argvs.append(list(argv)),
        _Completed(),
    )[1]
    module.shutil.which = lambda name: (
        "/fake/cargo" if name == "cargo" else original_which(name)
    )
    try:
        code, stdout, stderr = _run_main(dist, *flags)
    finally:
        module.subprocess.run, module.shutil.which = original_run, original_which
    return code, stdout, stderr, argvs


def _assert_main_verifies(dist: Path, expected_key: Path) -> None:
    """`main` must verify, against the resolved key, and say which key that was.

    Three things have to hold together, because each is separately silent:
    verification has to happen at all; it has to use the key
    `resolve_signing_key` chose, since `cargo xtask verify` falls back to the
    inline PEM when `--public-key` is omitted and would then check a different
    key than the run reports; and the `signature verification key:` line has to
    name that same key, or the log makes a claim nothing backs.
    """
    code, stdout, stderr, argvs = _run_main_verifying(dist, "--require-signatures")
    if code is not None:
        raise ReadinessError(
            f"main rejected a run it should have accepted (exit {code}): "
            f"{stderr.strip()!r}"
        )
    if not argvs:
        raise ReadinessError(
            "main completed a --require-signatures run without verifying any "
            "signature; the verify decision is not reaching validate_release"
        )
    resolved = str(expected_key.resolve())
    for argv in argvs:
        if "--public-key" not in argv or resolved not in argv:
            raise ReadinessError(
                f"verification ran without --public-key {resolved}: {argv!r}. "
                "`cargo xtask verify` falls back to the inline PEM when the flag "
                "is omitted, so this would check a different key than the run "
                "reports."
            )
    expected_line = (
        f"signature verification key: ${SIGNING_PUBLIC_KEY_PATH_ENV} ({expected_key})"
    )
    if expected_line not in stdout:
        raise ReadinessError(
            f"the run did not report the key it used; expected {expected_line!r} "
            f"in stdout, got {stdout.strip()!r}"
        )


def _assert_key_line_survives_a_failed_verification(dist: Path) -> None:
    """The key line must appear on the run that fails, not only on the one that passes.

    `docs/release-trust.md` promises it names the key the run actually used; a
    failed verification is when that matters most. Collecting the line into
    `messages`, which is flushed only on success, would keep every other case
    green while deleting it from exactly that run.
    """
    code, stdout, _stderr, argvs = _run_main_verifying(
        dist, "--require-signatures", verification_fails=True
    )
    if code is None or not argvs:
        raise ReadinessError(
            "main accepted a run whose signature verification failed "
            f"(exit {code!r}, {len(argvs)} verification call(s))"
        )
    if "signature verification key:" not in stdout:
        raise ReadinessError(
            "the key line is missing from a run that failed verification, which "
            "is the run that most needs it; it must not be deferred to the "
            f"success-only message flush. stdout was {stdout.strip()!r}"
        )


def _assert_inline_pem_run_omits_the_key_flag(dist: Path) -> None:
    """With only the inline PEM set, the run must say so and pass no `--public-key`.

    This is the source release and nightly actually use. `cargo xtask verify`
    reads the PEM itself, so materialising a key file would be wrong -- but the
    label still has to name the PEM, and gating the line on a resolved path
    would silently drop it for every real CI run.
    """
    code, stdout, stderr, argvs = _run_main_verifying(dist, "--require-signatures")
    if code is not None:
        raise ReadinessError(
            f"main rejected an inline-PEM run it should have accepted (exit {code}): "
            f"{stderr.strip()!r}"
        )
    if not argvs:
        raise ReadinessError("an inline-PEM run verified nothing")
    for argv in argvs:
        if "--public-key" in argv:
            raise ReadinessError(
                f"an inline-PEM run passed --public-key: {argv!r}; `cargo xtask "
                "verify` reads the PEM itself and no key file exists to name"
            )
    expected_line = f"signature verification key: ${SIGNING_PUBLIC_KEY_ENV}"
    if expected_line not in stdout:
        raise ReadinessError(
            f"an inline-PEM run did not report its key source; expected "
            f"{expected_line!r}, got {stdout.strip()!r}"
        )


def _assert_main_honours_trigger(dist: Path, *flags: str) -> None:
    """Each trigger must force verification *through `main`*, not just in the helper.

    `verification_is_required` is pinned directly, but `main` chooses what to
    feed it. Passing a literal `False` for production trust, or `None` for
    `--public-key`, leaves those helper cases green while the flag stops doing
    anything.
    """
    _code, _stdout, _stderr, argvs = _run_main_verifying(dist, *flags)
    if not argvs:
        raise ReadinessError(
            f"{' '.join(flags)} did not make main verify anything; the flag is no "
            "longer reaching resolve_verification"
        )


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        dist = root / "dist"
        dist.mkdir()
        linux_archive = dist / "rocm-cli-test-linux-amd64.tar.gz"
        windows_archive = dist / "rocm-cli-test-windows-amd64.zip"
        create_test_tar(linux_archive, "rocm-cli-test-linux-amd64")
        create_test_zip(windows_archive, "rocm-cli-test-windows-amd64")
        for archive in (linux_archive, windows_archive):
            write_sha(archive)
            Path(f"{archive}.sig").write_bytes(b"detached signature placeholder\n")
        validate_release(
            dist,
            assets=[],
            require_signatures=True,
            verify=False,
            public_key=None,
            require_production_trust=False,
            require_rocm_asset_names=False,
            require_exact_assets=False,
        )
        print("release readiness self-test: valid signed dist accepted")

        exact_dist = root / "exact-dist"
        exact_dist.mkdir()
        exact_linux_archive = exact_dist / "rocm-cli-v1.2.3-linux-amd64.tar.gz"
        exact_windows_archive = exact_dist / "rocm-cli-v1.2.3-windows-amd64.zip"
        create_test_tar(exact_linux_archive, "rocm-cli-v1.2.3-linux-amd64")
        create_test_zip(exact_windows_archive, "rocm-cli-v1.2.3-windows-amd64")
        for archive in (exact_linux_archive, exact_windows_archive):
            write_sha(archive)
            Path(f"{archive}.sig").write_bytes(b"detached signature placeholder\n")
        validate_release(
            exact_dist,
            assets=[exact_linux_archive.name, exact_windows_archive.name],
            require_signatures=True,
            verify=False,
            public_key=None,
            require_production_trust=False,
            require_rocm_asset_names=True,
            require_exact_assets=True,
        )
        print("release readiness self-test: exact signed dist accepted")

        stale_archive = exact_dist / "rocm-cli-v9.9.9-linux-amd64.tar.gz"
        create_test_tar(stale_archive, "rocm-cli-v9.9.9-linux-amd64")
        write_sha(stale_archive)
        Path(f"{stale_archive}.sig").write_bytes(b"detached signature placeholder\n")
        expect_failure(
            "stale publishable asset",
            lambda: validate_release(
                exact_dist,
                assets=[exact_linux_archive.name, exact_windows_archive.name],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=True,
                require_exact_assets=True,
            ),
        )
        stale_archive.unlink()
        Path(f"{stale_archive}.sha256").unlink()
        Path(f"{stale_archive}.sig").unlink()

        orphan_sidecar = exact_dist / "rocm-cli-v9.9.9-windows-amd64.zip.sha256"
        orphan_sidecar.write_text(
            f"{'0' * 64}  rocm-cli-v9.9.9-windows-amd64.zip\n", encoding="ascii"
        )
        expect_failure(
            "orphan publishable sidecar",
            lambda: validate_release(
                exact_dist,
                assets=[exact_linux_archive.name, exact_windows_archive.name],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=True,
                require_exact_assets=True,
            ),
        )
        orphan_sidecar.unlink()

        strict_linux_archive = dist / "rocm-cli-v1.2.3-linux-amd64.tar.gz"
        strict_windows_archive = dist / "rocm-cli-v1.2.3-windows-amd64.zip"
        strict_nightly_archive = (
            dist / "rocm-cli-nightly-20260602-abcdef0-linux-amd64.tar.gz"
        )
        strict_nightly_alias = dist / "rocm-cli-nightly-windows-amd64.zip"
        create_test_tar(strict_linux_archive, "rocm-cli-v1.2.3-linux-amd64")
        create_test_zip(strict_windows_archive, "rocm-cli-v1.2.3-windows-amd64")
        create_test_tar(
            strict_nightly_archive, "rocm-cli-nightly-20260602-abcdef0-linux-amd64"
        )
        create_test_zip(strict_nightly_alias, "rocm-cli-nightly-windows-amd64")
        for archive in (
            strict_linux_archive,
            strict_windows_archive,
            strict_nightly_archive,
            strict_nightly_alias,
        ):
            write_sha(archive)
            Path(f"{archive}.sig").write_bytes(b"detached signature placeholder\n")
        validate_release(
            dist,
            assets=[
                strict_linux_archive.name,
                strict_windows_archive.name,
                strict_nightly_archive.name,
                strict_nightly_alias.name,
            ],
            require_signatures=True,
            verify=False,
            public_key=None,
            require_production_trust=False,
            require_rocm_asset_names=True,
            require_exact_assets=False,
        )
        print("release readiness self-test: valid rocm-cli asset names accepted")

        Path(f"{linux_archive}.sig").unlink()
        expect_failure(
            "missing signature",
            lambda: validate_release(
                dist,
                assets=[],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )
        Path(f"{linux_archive}.sig").write_bytes(b"detached signature placeholder\n")

        Path(f"{linux_archive}.sha256").write_text(
            f"{'0' * 64}  {linux_archive.name}\n",
            encoding="ascii",
        )
        expect_failure(
            "bad checksum",
            lambda: validate_release(
                dist,
                assets=[],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )

        write_sha(linux_archive, archive_name="other-name.tar.gz")
        expect_failure(
            "checksum filename mismatch",
            lambda: validate_release(
                dist,
                assets=[],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )

        with tarfile.open(linux_archive, "w:gz") as package:
            add_tar_file(package, "bad-a", "bin/rocm", executable=True)
            add_tar_file(package, "bad-b", "bin/rocmd", executable=True)
        write_sha(linux_archive)
        expect_failure(
            "multiple top-level archive roots",
            lambda: validate_release(
                dist,
                assets=[],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )

        create_test_tar(linux_archive, "rocm-cli-test-linux-amd64")
        write_sha(linux_archive)
        expect_failure(
            "missing production trust inputs",
            lambda: run_with_env(
                dict.fromkeys(PRODUCTION_TRUST_ENV_NAMES),
                lambda: validate_release(
                    dist,
                    assets=[],
                    require_signatures=True,
                    verify=False,
                    public_key=None,
                    require_production_trust=True,
                    require_rocm_asset_names=False,
                    require_exact_assets=False,
                ),
            ),
        )

        trust_root = root / "production-trust"
        trust_root.mkdir()
        release_public_key = trust_root / "release-signing-public.pem"
        metadata_public_key = trust_root / "metadata-public.pem"
        recipe_index = trust_root / "model-recipes.json"
        recipe_index_signature = trust_root / "model-recipes.json.sig"
        recipe_index_public_key = trust_root / "model-recipes-public.pem"
        for path in (
            release_public_key,
            metadata_public_key,
            recipe_index,
            recipe_index_signature,
            recipe_index_public_key,
        ):
            path.write_text(
                f"self-test placeholder for {path.name}\n", encoding="ascii"
            )
        run_with_env(
            {
                **dict.fromkeys(PRODUCTION_TRUST_ENV_NAMES),
                "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH": release_public_key,
                "ROCM_CLI_METADATA_PUBLIC_KEY_PATH": metadata_public_key,
                "ROCM_CLI_MODEL_RECIPE_INDEX_PATH": recipe_index,
                "ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH": recipe_index_signature,
                "ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH": recipe_index_public_key,
            },
            lambda: validate_release(
                dist,
                assets=[],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=True,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )
        print("release readiness self-test: valid production trust inputs accepted")

        expect_failure(
            "missing explicit asset",
            lambda: validate_release(
                dist,
                assets=[
                    "rocm-cli-test-linux-amd64.tar.gz",
                    "missing-installer-alias.tar.gz",
                ],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )

        expect_failure(
            "asset path traversal",
            lambda: validate_release(
                dist,
                assets=["../rocm-cli-linux-amd64.tar.gz"],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=False,
                require_exact_assets=False,
            ),
        )

        create_test_tar(linux_archive, "rocm-cli-test-linux-amd64")
        write_sha(linux_archive)
        expect_failure(
            "malformed rocm release asset name",
            lambda: validate_release(
                dist,
                assets=["rocm-cli-test-linux-amd64.tar.gz"],
                require_signatures=True,
                verify=False,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=True,
                require_exact_assets=False,
            ),
        )

        expect_failure(
            "signature verification with no resolvable key",
            lambda: run_with_env(
                {SIGNING_PUBLIC_KEY_ENV: None, SIGNING_PUBLIC_KEY_PATH_ENV: None},
                lambda: resolve_signing_key(None),
            ),
        )
        # GitHub expands an unset secret to the empty string, so this is what an
        # unconfigured ROCM_CLI_SIGNING_PUBLIC_KEY_PEM actually looks like in CI.
        # It must fail exactly like an absent one rather than resolving to a key.
        expect_failure(
            "signature verification with an empty key",
            lambda: run_with_env(
                {SIGNING_PUBLIC_KEY_ENV: "", SIGNING_PUBLIC_KEY_PATH_ENV: ""},
                lambda: resolve_signing_key(None),
            ),
        )

        inline_pem = "-----BEGIN PUBLIC KEY-----\nself-test\n"
        env_key, env_source = run_with_env(
            {SIGNING_PUBLIC_KEY_ENV: inline_pem, SIGNING_PUBLIC_KEY_PATH_ENV: None},
            lambda: resolve_signing_key(None),
        )
        if env_key is not None or SIGNING_PUBLIC_KEY_ENV not in env_source:
            raise ReadinessError(
                f"expected the environment key to resolve, got {env_source}"
            )

        # A key file the operator named must be used, not passed over in favour of
        # the inline PEM — and never ignored in favour of failing, which would tell
        # them to configure a key they had already configured.
        key_path = root / "configured-public-key.pem"
        path_key, path_source = run_with_env(
            {SIGNING_PUBLIC_KEY_ENV: inline_pem, SIGNING_PUBLIC_KEY_PATH_ENV: key_path},
            lambda: resolve_signing_key(None),
        )
        if path_key != key_path or SIGNING_PUBLIC_KEY_PATH_ENV not in path_source:
            raise ReadinessError(
                f"expected the key path to win over the inline PEM, got {path_source}"
            )

        explicit = root / "explicit-public-key.pem"
        explicit_key, explicit_source = run_with_env(
            {
                SIGNING_PUBLIC_KEY_ENV: inline_pem,
                SIGNING_PUBLIC_KEY_PATH_ENV: key_path,
            },
            lambda: resolve_signing_key(explicit),
        )
        if explicit_key != explicit or str(explicit) not in explicit_source:
            raise ReadinessError(
                f"expected --public-key to win over the environment, got {explicit_source}"
            )
        print("release readiness self-test: signing key resolution ok")

        # Each flag must force verification on, and nothing else may turn it on.
        for label, flags in (
            ("--require-signatures", (True, False, None)),
            ("--require-production-trust", (False, True, None)),
            ("--public-key", (False, False, Path("/nonexistent/key.pem"))),
        ):
            if not verification_is_required(*flags):
                raise ReadinessError(
                    f"{label} must force cryptographic verification, not just a .sig check"
                )
        if verification_is_required(False, False, None):
            raise ReadinessError(
                "verification must stay off when nothing asks for it, or an ordinary "
                "readiness run would demand a signing key it has no reason to need"
            )
        print("release readiness self-test: verify gate ok")
        # Requiring verification must also resolve a key, not just set a flag.
        run_with_env(
            {
                SIGNING_PUBLIC_KEY_PATH_ENV: key_path,
                SIGNING_PUBLIC_KEY_ENV: None,
            },
            lambda: _assert_verification_wiring(key_path),
        )
        print("release readiness self-test: verify wiring ok")

        # `main`'s own behaviour, on a dist that is valid on every other axis so
        # the verify gate is the only thing either case can turn on.
        gate_dist = root / "gate-dist"
        gate_dist.mkdir()
        gate_archive = gate_dist / "rocm-cli-test-linux-amd64.tar.gz"
        create_test_tar(gate_archive, "rocm-cli-test-linux-amd64")
        write_sha(gate_archive)
        Path(f"{gate_archive}.sig").write_bytes(b"detached signature placeholder\n")
        no_key_env = {
            SIGNING_PUBLIC_KEY_PATH_ENV: None,
            SIGNING_PUBLIC_KEY_ENV: None,
            "ROCM_CLI_REQUIRE_SIGNATURE": None,
            "ROCM_CLI_REQUIRE_PRODUCTION_TRUST": None,
        }
        run_with_env(no_key_env, lambda: _assert_main_refuses_without_a_key(gate_dist))
        print("release readiness self-test: main refuses without a key ok")
        # A key that exists on disk, because this case lets the real
        # `verify_signature` run and it rejects a named key that is not a file.
        # `key_path` above stays absent on purpose, to pin that a named-but-
        # missing key still wins resolution rather than falling through.
        real_key = root / "present-public-key.pem"
        real_key.write_text(inline_pem, encoding="ascii")
        path_env = {**no_key_env, SIGNING_PUBLIC_KEY_PATH_ENV: real_key}
        run_with_env(path_env, lambda: _assert_main_verifies(gate_dist, real_key))
        print("release readiness self-test: main actually verifies ok")

        run_with_env(
            path_env, lambda: _assert_key_line_survives_a_failed_verification(gate_dist)
        )
        print("release readiness self-test: key line survives a failed verify ok")

        # The source release and nightly actually use.
        run_with_env(
            {**no_key_env, SIGNING_PUBLIC_KEY_ENV: inline_pem},
            lambda: _assert_inline_pem_run_omits_the_key_flag(gate_dist),
        )
        print("release readiness self-test: inline-PEM run reports its source ok")

        # The other two triggers, driven through `main` rather than the helper.
        run_with_env(
            path_env,
            lambda: _assert_main_honours_trigger(
                gate_dist, "--require-production-trust"
            ),
        )
        run_with_env(
            no_key_env,
            lambda: _assert_main_honours_trigger(
                gate_dist, "--public-key", str(real_key)
            ),
        )
        print("release readiness self-test: production-trust and --public-key ok")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("release readiness self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist", default="dist", help="Directory containing release archives."
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Archive filename under --dist to validate. Repeat to check selected assets only.",
    )
    parser.add_argument(
        "--require-signatures",
        action="store_true",
        help="Require every archive to carry a detached signature that verifies "
        "against the release signing public key.",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        help="Verify detached signatures with this public key instead of the one "
        f"named by {SIGNING_PUBLIC_KEY_PATH_ENV} or {SIGNING_PUBLIC_KEY_ENV}.",
    )
    parser.add_argument(
        "--require-production-trust",
        action="store_true",
        help="Require explicit owner-provided production trust root inputs.",
    )
    parser.add_argument(
        "--require-rocm-asset-names",
        action="store_true",
        help="Require rocm-cli release asset names such as rocm-cli-v1.2.3-linux-amd64.tar.gz.",
    )
    parser.add_argument(
        "--require-exact-assets",
        action="store_true",
        help="Reject stale publishable release files under --dist that are not named by --asset.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local verifier self-tests instead of checking dist.",
    )
    parser.add_argument(
        "--self-test-root",
        type=Path,
        default=repo_root() / ".rocm-work" / "tests" / "release-readiness",
        help="Workspace-local root used by --self-test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        try:
            run_self_test(args.self_test_root)
        except ReadinessError as error:
            fail(str(error))
        return

    require_signatures = args.require_signatures or truthy(
        os.environ.get("ROCM_CLI_REQUIRE_SIGNATURE")
    )
    require_production_trust = args.require_production_trust or truthy(
        os.environ.get("ROCM_CLI_REQUIRE_PRODUCTION_TRUST")
    )
    messages: list[str] = []
    try:
        verify, public_key, key_source = resolve_verification(
            require_signatures, require_production_trust, args.public_key
        )
        if key_source is not None:
            # Printed now rather than collected into `messages`, which is only
            # flushed on success: a run that fails verification is exactly when
            # you need to know which key it used. Flushed so it cannot be
            # reordered after the unbuffered stderr of `fail()` when piped.
            print(
                f"release readiness: signature verification key: {key_source}",
                flush=True,
            )
        messages.extend(
            validate_release(
                Path(args.dist),
                assets=args.asset,
                require_signatures=require_signatures,
                verify=verify,
                public_key=public_key,
                require_production_trust=require_production_trust,
                require_rocm_asset_names=args.require_rocm_asset_names,
                require_exact_assets=args.require_exact_assets,
            )
        )
    except ReadinessError as error:
        fail(str(error))

    for message in messages:
        print(f"release readiness: {message}")
    print("release readiness: ok")


if __name__ == "__main__":
    main()
