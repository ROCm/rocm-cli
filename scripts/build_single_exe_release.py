#!/usr/bin/env python3
"""Build platform-native rocm-cli release artifacts.

This helper copies or archives the native `rocm`/`rocm.exe` binary for the
current platform. It is useful for development packages, but it is not the
Cosmopolitan universal-binary release path. Use
`scripts/rust_cosmopolitan_spike.py build-rocm --release` and
`scripts/single_exe_release_gate.py` for the true no-extract universal binary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

WINDOWS_BINARIES = [
    "rocm.exe",
    "rocmd.exe",
    "rocm-engine-pytorch.exe",
    "rocm-engine-llama-cpp.exe",
    "rocm-engine-lemonade.exe",
    "rocm-engine-atom.exe",
    "rocm-engine-vllm.exe",
    "rocm-engine-sglang.exe",
]
LINUX_BINARIES = [name[:-4] for name in WINDOWS_BINARIES]
PLATFORMS = {"windows-amd64", "linux-amd64"}


class ReleaseBuildError(Exception):
    """The release artifact could not be built."""


def fail(message: str) -> None:
    print(f"release build failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(args))
    completed = subprocess.run(args, cwd=cwd, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise ReleaseBuildError(
            f"command failed with exit {completed.returncode}: {' '.join(args)}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: Path) -> None:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def current_platform() -> str:
    if os.name == "nt":
        return "windows-amd64"
    if sys.platform.startswith("linux"):
        return "linux-amd64"
    raise ReleaseBuildError(f"unsupported build host: {sys.platform}")


def cargo_build_release(repo_root: Path, jobs: int | None) -> None:
    env = os.environ.copy()
    if jobs is not None:
        env["CARGO_BUILD_JOBS"] = str(jobs)
    run(
        ["cargo", "build", "--release", "--workspace", "--bins"], cwd=repo_root, env=env
    )


def cargo_build_rocm_release(repo_root: Path, jobs: int | None) -> None:
    env = os.environ.copy()
    if jobs is not None:
        env["CARGO_BUILD_JOBS"] = str(jobs)
    run(
        ["cargo", "build", "--release", "-p", "rocm", "--bin", "rocm"],
        cwd=repo_root,
        env=env,
    )


def strip_tool_for(platform: str) -> str | None:
    candidates: list[str] = []
    configured = os.environ.get("ROCM_CLI_STRIP")
    if configured:
        candidates.append(configured)
    if platform == "windows-amd64":
        candidates.extend(
            [
                r"D:\jam\venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-strip.exe",
                r"D:\jam\venv\Lib\site-packages\_rocm_sdk_devel\lib\llvm\bin\llvm-strip.exe",
                "llvm-strip",
            ]
        )
    else:
        candidates.extend(["llvm-strip", "strip"])
    for candidate in candidates:
        found = shutil.which(candidate) if not Path(candidate).is_file() else candidate
        if found:
            return str(found)
    return None


def maybe_strip(path: Path, *, platform: str, strip_debug_only: bool = False) -> None:
    tool = strip_tool_for(platform)
    if tool is None:
        return
    flag = "--strip-debug" if strip_debug_only else "--strip-all"
    completed = subprocess.run(
        [tool, flag, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 and not strip_debug_only:
        subprocess.run(
            [tool, "--strip-debug", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ReleaseBuildError(f"required file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def platform_binaries(platform: str) -> list[str]:
    if platform == "windows-amd64":
        return WINDOWS_BINARIES
    if platform == "linux-amd64":
        return LINUX_BINARIES
    raise ReleaseBuildError(f"unsupported platform: {platform}")


def standalone_binary(platform: str) -> str:
    if platform == "windows-amd64":
        return "rocm.exe"
    if platform == "linux-amd64":
        return "rocm"
    raise ReleaseBuildError(f"unsupported platform: {platform}")


def platform_install_script(platform: str) -> str:
    return "install.ps1" if platform == "windows-amd64" else "install.sh"


def create_zip(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as package:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            package.write(path, path.relative_to(root.parent).as_posix())


def reset_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def create_tar_gz(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as gz,
        tarfile.open(fileobj=gz, mode="w") as package,
    ):
        package.add(root, arcname=root.name, filter=reset_tar_info)


def assert_no_codex(archive: Path) -> None:
    names: list[str]
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            names = package.getnames()
    else:
        raise ReleaseBuildError(f"unsupported archive type: {archive}")
    offenders = [name for name in names if "rocm-codex" in name or "/codex" in name]
    if offenders:
        raise ReleaseBuildError(
            f"Codex binary leaked into release archive: {offenders[:3]}"
        )


def stage_platform_release(
    *,
    repo_root: Path,
    output_dir: Path,
    version: str,
    platform: str,
    skip_cargo_build: bool,
    strip_binaries: bool,
    jobs: int | None,
) -> Path:
    if platform not in PLATFORMS:
        raise ReleaseBuildError(f"platform must be one of {sorted(PLATFORMS)}")
    if not skip_cargo_build:
        cargo_build_release(repo_root, jobs)

    dist_name = f"rocm-cli-v{version}-{platform}"
    staging_root = output_dir / "staging" / platform / dist_name
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    bin_dir = staging_root / "bin"
    bin_dir.mkdir()

    profile_dir = repo_root / "target" / "release"
    for binary in platform_binaries(platform):
        destination = bin_dir / binary
        copy_required(profile_dir / binary, destination)
        if strip_binaries:
            maybe_strip(destination, platform=platform)

    for name in ["README.md", "LICENSE", platform_install_script(platform)]:
        copy_required(repo_root / name, staging_root / name)

    archive = output_dir / (
        f"{dist_name}.zip" if platform == "windows-amd64" else f"{dist_name}.tar.gz"
    )
    if archive.exists():
        archive.unlink()
    if platform == "windows-amd64":
        create_zip(staging_root, archive)
    else:
        create_tar_gz(staging_root, archive)
    assert_no_codex(archive)
    write_sha256(archive)
    print(f"wrote {archive}")
    print(f"sha256 {sha256_file(archive)}")
    return archive


def build_standalone_release(
    *,
    repo_root: Path,
    output_dir: Path,
    output: Path | None,
    platform: str,
    skip_cargo_build: bool,
    strip_binary: bool,
    write_digest: bool,
    jobs: int | None,
) -> Path:
    if platform not in PLATFORMS:
        raise ReleaseBuildError(f"platform must be one of {sorted(PLATFORMS)}")
    if platform != current_platform() and not skip_cargo_build:
        raise ReleaseBuildError(
            f"standalone builds target the current host by default ({current_platform()}); "
            f"run this command on {platform} or pass --skip-cargo-build after staging the binary"
        )
    if not skip_cargo_build:
        cargo_build_rocm_release(repo_root, jobs)

    binary = standalone_binary(platform)
    source = repo_root / "target" / "release" / binary
    destination = (
        output.resolve() if output is not None else output_dir.resolve() / binary
    )
    copy_required(source, destination)
    if strip_binary:
        maybe_strip(destination, platform=platform)
    if platform == "linux-amd64":
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    if write_digest:
        write_sha256(destination)
    print(f"wrote {destination}")
    if write_digest:
        print(f"sha256 {sha256_file(destination)}")
    return destination


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    fake_repo = root / "repo"
    fake_out = root / "out"
    try:
        for platform in sorted(PLATFORMS):
            profile = fake_repo / "target" / "release"
            profile.mkdir(parents=True, exist_ok=True)
            for binary in platform_binaries(platform):
                path = profile / binary
                path.write_bytes(b"fake release binary\n")
                if platform == "linux-amd64":
                    path.chmod(path.stat().st_mode | stat.S_IXUSR)
            for name in ["README.md", "LICENSE", "install.ps1", "install.sh"]:
                (fake_repo / name).write_text(f"fake {name}\n", encoding="utf-8")
            standalone = build_standalone_release(
                repo_root=fake_repo,
                output_dir=fake_out / "standalone" / platform,
                output=None,
                platform=platform,
                skip_cargo_build=True,
                strip_binary=False,
                write_digest=False,
                jobs=None,
            )
            assert standalone.name == standalone_binary(platform)
            archive = stage_platform_release(
                repo_root=fake_repo,
                output_dir=fake_out,
                version="0.0.0-test",
                platform=platform,
                skip_cargo_build=True,
                strip_binaries=False,
                jobs=None,
            )
            assert_no_codex(archive)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("release builder self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    standalone = subparsers.add_parser(
        "standalone", help="Build the standalone rocm-cli binary."
    )
    standalone.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    standalone.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".rocm-work" / "standalone-release",
    )
    standalone.add_argument("--output", type=Path)
    standalone.add_argument(
        "--platform", choices=sorted(PLATFORMS), default=current_platform()
    )
    standalone.add_argument("--skip-cargo-build", action="store_true")
    standalone.add_argument("--no-strip", action="store_true")
    standalone.add_argument("--write-sha256", action="store_true")
    standalone.add_argument("--jobs", type=int, default=96)

    stage = subparsers.add_parser(
        "stage-platform", help="Build and archive the current platform payload."
    )
    stage.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    stage.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "platform-release"
    )
    stage.add_argument("--version", default="0.2.0")
    stage.add_argument(
        "--platform", choices=sorted(PLATFORMS), default=current_platform()
    )
    stage.add_argument("--skip-cargo-build", action="store_true")
    stage.add_argument("--no-strip", action="store_true")
    stage.add_argument("--jobs", type=int, default=96)

    self_test = subparsers.add_parser(
        "self-test", help="Run offline archive policy tests."
    )
    self_test.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / ".rocm-work" / "tests" / f"release-builder-{os.getpid()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "standalone":
            build_standalone_release(
                repo_root=args.repo_root.resolve(),
                output_dir=args.output_dir.resolve(),
                output=args.output,
                platform=args.platform,
                skip_cargo_build=args.skip_cargo_build,
                strip_binary=not args.no_strip,
                write_digest=args.write_sha256,
                jobs=args.jobs,
            )
            return 0
        if args.command == "stage-platform":
            stage_platform_release(
                repo_root=args.repo_root.resolve(),
                output_dir=args.output_dir.resolve(),
                version=args.version,
                platform=args.platform,
                skip_cargo_build=args.skip_cargo_build,
                strip_binaries=not args.no_strip,
                jobs=args.jobs,
            )
            return 0
        if args.command == "self-test":
            run_self_test(args.root.resolve())
            return 0
    except (ReleaseBuildError, OSError, subprocess.SubprocessError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
