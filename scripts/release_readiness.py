#!/usr/bin/env python3
"""Verify rocm-cli release dist assets before publication."""

from __future__ import annotations

import argparse
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
LINUX_REQUIRED = (
    "bin/rocm",
    "bin/rocmd",
    "bin/rocm-engine-pytorch",
    "bin/rocm-engine-llama-cpp",
    "bin/rocm-engine-lemonade",
    "bin/rocm-engine-atom",
    "bin/rocm-engine-vllm",
    "bin/rocm-engine-sglang",
    "bin/rocm-codex",
    "README.md",
    "LICENSE",
    "install.sh",
)
WINDOWS_REQUIRED = (
    "bin/rocm.exe",
    "bin/rocmd.exe",
    "bin/rocm-engine-pytorch.exe",
    "bin/rocm-engine-llama-cpp.exe",
    "bin/rocm-engine-lemonade.exe",
    "bin/rocm-engine-atom.exe",
    "bin/rocm-engine-vllm.exe",
    "bin/rocm-engine-sglang.exe",
    "bin/rocm-codex.exe",
    "README.md",
    "LICENSE",
    "install.ps1",
)
LINUX_EXECUTABLES = (
    "bin/rocm",
    "bin/rocmd",
    "bin/rocm-engine-pytorch",
    "bin/rocm-engine-llama-cpp",
    "bin/rocm-engine-lemonade",
    "bin/rocm-engine-atom",
    "bin/rocm-engine-vllm",
    "bin/rocm-engine-sglang",
    "bin/rocm-codex",
    "install.sh",
)
PRODUCTION_TRUST_ENV_NAMES = (
    "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH",
    "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM",
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
        raise ReadinessError(f"asset name must not contain drive or URI separators: {name}")
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
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ReadinessError(f"checksum sidecar is empty: {path}")
    parts = lines[0].split()
    digest = parts[0].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ReadinessError(f"checksum sidecar does not start with a sha256 digest: {path}")
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
        raise ReadinessError(f"{archive.name} must contain exactly one top-level directory; found {joined}")
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
                    raise ReadinessError(f"{archive.name} contains duplicate path: {normalized}")
                all_paths.add(normalized)
                if member.isdev() or member.issym() or member.islnk():
                    raise ReadinessError(f"{archive.name} contains unsupported special entry: {normalized}")
                if member.isfile():
                    files[normalized] = member
    except tarfile.TarError as error:
        raise ReadinessError(f"failed to read tar archive {archive.name}: {error}") from error

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
            raise ReadinessError(f"{archive.name} contains empty required file {required_file}")
    for executable in executables:
        member = bundle_files.get(executable)
        if member is None:
            continue
        if member.mode & 0o111 == 0:
            raise ReadinessError(f"{archive.name} required executable is not executable: {executable}")
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
                    raise ReadinessError(f"{archive.name} contains duplicate path: {normalized}")
                all_paths.add(normalized)
                if zip_entry_is_symlink(info):
                    raise ReadinessError(f"{archive.name} contains unsupported symlink entry: {normalized}")
                if not info.is_dir():
                    files[normalized] = info
    except zipfile.BadZipFile as error:
        raise ReadinessError(f"failed to read zip archive {archive.name}: {error}") from error

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
            raise ReadinessError(f"{archive.name} contains empty required file {required_file}")
    return [f"bundle contents ok: {archive.name}"]


def validate_archive_contents(archive: Path) -> list[str]:
    if archive.name.lower().endswith(".tar.gz"):
        return validate_tar_archive(archive)
    if archive.name.lower().endswith(".zip"):
        return validate_zip_archive(archive)
    raise ReadinessError(f"unsupported archive extension: {archive.name}")


def verify_signature(archive: Path, signature: Path, public_key: Path) -> None:
    if not public_key.is_file():
        raise ReadinessError(f"public key does not exist: {public_key}")
    openssl = shutil.which("openssl")
    if openssl is None:
        raise ReadinessError("openssl is required for signature verification")
    completed = subprocess.run(
        [
            openssl,
            "dgst",
            "-sha256",
            "-verify",
            str(public_key),
            "-signature",
            str(signature),
            str(archive),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise ReadinessError(f"signature verification failed for {archive.name}{suffix}")


def validate_archive(
    archive: Path,
    *,
    require_signatures: bool,
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
    if require_signatures or public_key is not None:
        if not signature.is_file():
            raise ReadinessError(f"missing signature sidecar: {signature}")
        if signature.stat().st_size <= 0:
            raise ReadinessError(f"signature sidecar is empty: {signature}")
        messages.append(f"signature present: {archive.name}.sig")

    if public_key is not None:
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
    archives = sorted(path for path in dist.iterdir() if path.is_file() and is_archive(path))
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
    if (path := env_path("ROCM_CLI_SIGNING_PUBLIC_KEY_PATH")) is not None and not path.is_file():
        raise ReadinessError(f"ROCM_CLI_SIGNING_PUBLIC_KEY_PATH does not point to a file: {path}")
    messages.append("release signing public key configured")

    require_any(
        "the runtime metadata public key",
        ["ROCM_CLI_METADATA_PUBLIC_KEY_PATH", "ROCM_CLI_METADATA_PUBLIC_KEY_PEM"],
    )
    if (path := env_path("ROCM_CLI_METADATA_PUBLIC_KEY_PATH")) is not None and not path.is_file():
        raise ReadinessError(f"ROCM_CLI_METADATA_PUBLIC_KEY_PATH does not point to a file: {path}")
    messages.append("runtime metadata public key configured")

    index_path = require_existing_env_path("ROCM_CLI_MODEL_RECIPE_INDEX_PATH")
    signature_path = env_path("ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH")
    if signature_path is None:
        signature_path = Path(f"{index_path}.sig")
    if not signature_path.is_file():
        raise ReadinessError(f"model recipe index signature is missing: {signature_path}")
    require_existing_env_path("ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH")
    messages.append("model recipe index, signature, and public key configured")

    return messages


def validate_release(
    dist: Path,
    *,
    assets: list[str],
    require_signatures: bool,
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
                expect_signatures=require_signatures or public_key is not None,
            )
        )
    for archive in archives:
        messages.extend(
            validate_archive(
                archive,
                require_signatures=require_signatures,
                public_key=public_key,
                require_rocm_asset_names=require_rocm_asset_names,
            )
        )
    if require_production_trust:
        messages.extend(validate_production_trust())
    return messages


def write_sha(path: Path, *, archive_name: str | None = None) -> None:
    target = archive_name or path.name
    Path(f"{path}.sha256").write_text(f"{sha256_hex(path)}  {target}\n", encoding="ascii")


def add_tar_file(package: tarfile.TarFile, root: str, relative: str, executable: bool = False) -> None:
    data = f"test content for {relative}\n".encode("utf-8")
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
            add_tar_file(package, root, required, executable=required in LINUX_EXECUTABLES)


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
        orphan_sidecar.write_text(f"{'0' * 64}  rocm-cli-v9.9.9-windows-amd64.zip\n", encoding="ascii")
        expect_failure(
            "orphan publishable sidecar",
            lambda: validate_release(
                exact_dist,
                assets=[exact_linux_archive.name, exact_windows_archive.name],
                require_signatures=True,
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=True,
                require_exact_assets=True,
            ),
        )
        orphan_sidecar.unlink()

        strict_linux_archive = dist / "rocm-cli-v1.2.3-linux-amd64.tar.gz"
        strict_windows_archive = dist / "rocm-cli-v1.2.3-windows-amd64.zip"
        strict_nightly_archive = dist / "rocm-cli-nightly-20260602-abcdef0-linux-amd64.tar.gz"
        strict_nightly_alias = dist / "rocm-cli-nightly-windows-amd64.zip"
        create_test_tar(strict_linux_archive, "rocm-cli-v1.2.3-linux-amd64")
        create_test_zip(strict_windows_archive, "rocm-cli-v1.2.3-windows-amd64")
        create_test_tar(strict_nightly_archive, "rocm-cli-nightly-20260602-abcdef0-linux-amd64")
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
                {name: None for name in PRODUCTION_TRUST_ENV_NAMES},
                lambda: validate_release(
                    dist,
                    assets=[],
                    require_signatures=True,
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
            path.write_text(f"self-test placeholder for {path.name}\n", encoding="ascii")
        run_with_env(
            {
                **{name: None for name in PRODUCTION_TRUST_ENV_NAMES},
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
                assets=["rocm-cli-test-linux-amd64.tar.gz", "missing-installer-alias.tar.gz"],
                require_signatures=True,
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
                public_key=None,
                require_production_trust=False,
                require_rocm_asset_names=True,
                require_exact_assets=False,
            ),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("release readiness self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", default="dist", help="Directory containing release archives.")
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Archive filename under --dist to validate. Repeat to check selected assets only.",
    )
    parser.add_argument(
        "--require-signatures",
        action="store_true",
        help="Require every archive to have a non-empty .sig sidecar.",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        help="Verify detached signatures with this public key.",
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

    require_signatures = args.require_signatures or truthy(os.environ.get("ROCM_CLI_REQUIRE_SIGNATURE"))
    require_production_trust = args.require_production_trust or truthy(
        os.environ.get("ROCM_CLI_REQUIRE_PRODUCTION_TRUST")
    )
    try:
        messages = validate_release(
            Path(args.dist),
            assets=args.asset,
            require_signatures=require_signatures,
            public_key=args.public_key,
            require_production_trust=require_production_trust,
            require_rocm_asset_names=args.require_rocm_asset_names,
            require_exact_assets=args.require_exact_assets,
        )
    except ReadinessError as error:
        fail(str(error))

    for message in messages:
        print(f"release readiness: {message}")
    print("release readiness: ok")


if __name__ == "__main__":
    main()
