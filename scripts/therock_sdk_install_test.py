#!/usr/bin/env python3
"""Opt-in acceptance test for managed TheRock SDK pip installs.

This test creates an isolated rocm-cli state root, creates a local bootstrap
Python venv, runs `rocm install sdk --format pip`, and verifies the installed
TheRock SDK venv with `python -m rocm_sdk` commands.

It downloads TheRock wheels and can take a while, so it is not part of the
default smoke suite.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THEROCK_SDK_PACKAGE_SPEC = "rocm[libraries,devel]"
THEROCK_TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]
THEROCK_RUNTIME_PACKAGES = ["rocm", "rocm-sdk-core"]


def fail(message: str) -> None:
    print(f"therock-sdk-install failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def venv_python_path(venv_root: Path) -> Path:
    if platform.system() == "Windows":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def run(
    label: str,
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    timeout: int = 1800,
    expect_failure: bool = False,
) -> str:
    print(f"therock-sdk-install: {label}")
    print("  " + " ".join(str(arg) for arg in argv))
    completed = subprocess.run(
        argv,
        cwd=cwd or repo_root(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.strip()
    if output:
        print(output)
    if expect_failure:
        if completed.returncode == 0:
            fail(f"{label} unexpectedly succeeded")
    elif completed.returncode != 0:
        fail(f"{label} exited with status {completed.returncode}")
    return output


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label} did not contain expected text: {needle}\n{text}")


def assert_not_contains(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"{label} contained unexpected text: {needle}\n{text}")


def install_output_field(text: str, field: str) -> Path:
    prefix = f"  {field}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix) :].strip())
    fail(f"sdk install output did not include field: {field}\n{text}")


def parse_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        fail(f"failed to parse {label} at {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} at {path} was not a JSON object")
    return value


def resolve_cargo() -> str:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    fallback = Path.home() / ".cargo" / "bin" / exe_name("cargo")
    if fallback.is_file():
        return str(fallback)
    fail(
        "missing developer build command: cargo\n  Build rocm first and rerun this test with --skip-build."
    )


def require_command(name: str, install_hint: str | None = None) -> None:
    if shutil.which(name):
        return
    message = f"missing required command: {name}"
    if install_hint:
        message += f"\n  {install_hint}"
    fail(message)


def report_windows_tool_hints() -> None:
    if platform.system() != "Windows":
        return

    print("therock-sdk-install: Windows tool check")
    checks = [
        ("winget", "optional package installer for missing build tools"),
        ("git", "source-build helper"),
        ("cmake", "source-build helper"),
        ("ninja", "source-build helper"),
        ("cl", "MSVC compiler for source builds"),
        ("dvc", "source-build helper"),
        ("perl", "Strawberry Perl / gfortran helper for source builds"),
        ("ccache", "optional source-build cache"),
    ]
    missing = []
    for command, purpose in checks:
        found = shutil.which(command) is not None
        print(f"  {command}: {'found' if found else 'missing'} ({purpose})")
        if not found:
            missing.append(command)
    if missing:
        print(
            "  note: this SDK wheel install test does not require the source-build tools above."
        )
        print(
            "  TheRock's Windows source-build docs suggest winget/manual installs for them."
        )
        print(
            "  rocm-cli should avoid installing global tools unless the user explicitly asks."
        )


def isolated_env(
    test_root: Path, python_executable: Path, family: str | None
) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_CLI_UPDATE_USER_PATH"] = "0"
    env["ROCM_CLI_CONFIG_DIR"] = str(test_root / "rocm-config")
    env["ROCM_CLI_DATA_DIR"] = str(test_root / "rocm-data")
    env["ROCM_CLI_CACHE_DIR"] = str(test_root / "rocm-cache")
    env["ROCM_CLI_PYTHON"] = str(python_executable)
    env.setdefault("ROCM_CLI_PIP_TIMEOUT_SECS", "900")
    env.setdefault("ROCM_CLI_PIP_RETRIES", "8")
    if family:
        env["ROCM_CLI_THEROCK_FAMILY"] = family

    if platform.system() == "Windows":
        env["APPDATA"] = str(test_root / "appdata")
        env["LOCALAPPDATA"] = str(test_root / "localappdata")
    else:
        env["XDG_CONFIG_HOME"] = str(test_root / "xdg-config")
        env["XDG_DATA_HOME"] = str(test_root / "xdg-data")
        env["XDG_CACHE_HOME"] = str(test_root / "xdg-cache")
        env["HOME"] = str(test_root / "home")
    return env


def ensure_bootstrap_python(test_root: Path, requested_python: str | None) -> Path:
    if requested_python:
        python = Path(requested_python)
        if not python.is_file():
            fail(f"--python does not point to a file: {python}")
        return python

    bootstrap_venv = test_root / "python-bootstrap"
    python = venv_python_path(bootstrap_venv)
    if python.is_file():
        return python

    bootstrap_venv.parent.mkdir(parents=True, exist_ok=True)
    run(
        "create local bootstrap Python venv",
        [sys.executable, "-m", "venv", str(bootstrap_venv)],
        env=os.environ.copy(),
        timeout=300,
    )
    run(
        "bootstrap pip in local Python venv",
        [str(python), "-m", "ensurepip", "--upgrade"],
        env=os.environ.copy(),
        timeout=300,
    )
    return python


def binary_path(profile: str, target_dir: Path | None, name: str) -> Path:
    binary_dir = cargo_target_dir(target_dir) / profile
    return binary_dir / exe_name(name)


def cargo_target_dir(target_dir: Path | None) -> Path:
    if target_dir is not None:
        return target_dir
    override = os.environ.get("CARGO_TARGET_DIR")
    if override:
        path = Path(override).expanduser()
        if path.is_absolute():
            return path
        return repo_root() / path
    return repo_root() / "target"


def discover_manifest(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    registry_dir = data_dir / "runtimes" / "registry"
    manifests = sorted(registry_dir.glob("*.json"))
    if not manifests:
        fail(f"no runtime manifests found in {registry_dir}")
    newest = max(manifests, key=lambda path: path.stat().st_mtime)
    return newest, parse_json_file(newest, "runtime manifest")


def remove_tree_with_retries(path: Path, attempts: int = 5) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def verify_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    expected_pip_cache_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    if manifest.get("format") != "pip":
        fail(f"expected pip runtime manifest, got: {manifest.get('format')}")
    python = Path(str(manifest.get("python_executable") or ""))
    if not python.is_file():
        fail(f"runtime python executable is missing: {python}")
    pip_cache_dir = Path(str(manifest.get("pip_cache_dir") or ""))
    if not pip_cache_dir.is_dir():
        fail(f"runtime pip cache dir is missing: {pip_cache_dir}")
    if pip_cache_dir.resolve() != expected_pip_cache_dir.resolve():
        fail(
            "runtime pip cache dir did not match expected location: "
            f"{pip_cache_dir} != {expected_pip_cache_dir}"
        )
    rocm_sdk = manifest.get("rocm_sdk")
    if not isinstance(rocm_sdk, dict) or not rocm_sdk.get("import_ok"):
        fail(f"manifest did not record a successful rocm_sdk probe: {manifest_path}")
    root = Path(str(rocm_sdk.get("root_path") or ""))
    bin_dir = Path(str(rocm_sdk.get("bin_path") or ""))
    if not root.is_dir():
        fail(f"rocm_sdk root does not exist: {root}")
    if not bin_dir.is_dir():
        fail(f"rocm_sdk bin does not exist: {bin_dir}")
    packages = rocm_sdk.get("packages")
    if not isinstance(packages, list) or not packages:
        fail("manifest rocm_sdk probe did not include any ROCm distributions")
    resolved_libraries = rocm_sdk.get("resolved_libraries")
    if not isinstance(resolved_libraries, list):
        fail("manifest rocm_sdk probe did not include resolved_libraries")
    resolved_names = {
        str(entry.get("shortname") or "")
        for entry in resolved_libraries
        if isinstance(entry, dict)
    }
    for shortname in ["amdhip64", "hipblas"]:
        if shortname not in resolved_names:
            fail(f"manifest rocm_sdk probe did not resolve {shortname}")
    return python, root, bin_dir, pip_cache_dir


def verify_python_distributions(
    python: Path,
    packages: list[str],
    env: dict[str, str],
) -> dict[str, str]:
    code = (
        "import importlib.metadata as md, json; "
        f"names = {packages!r}; "
        "print(json.dumps({name: md.version(name) for name in names}, sort_keys=True))"
    )
    output = run(
        "verify Python distributions in TheRock runtime",
        [str(python), "-c", code],
        env=env,
        timeout=120,
    )
    try:
        versions = json.loads(output)
    except json.JSONDecodeError as error:
        fail(f"failed to parse Python distribution versions: {error}\n{output}")
    if not isinstance(versions, dict):
        fail(f"Python distribution version output was not an object: {versions!r}")
    missing = [
        name
        for name in packages
        if not isinstance(versions.get(name), str) or not versions[name]
    ]
    if missing:
        fail(f"missing Python distributions in runtime: {', '.join(missing)}")
    return {name: str(versions[name]) for name in packages}


def verify_rocm_manifest_packages(manifest: dict[str, Any]) -> None:
    rocm_sdk = manifest.get("rocm_sdk")
    if not isinstance(rocm_sdk, dict):
        fail("manifest did not include rocm_sdk details")
    packages = rocm_sdk.get("packages")
    if not isinstance(packages, list):
        fail("manifest rocm_sdk packages were not a list")
    names = {
        str(package.get("name") or "")
        for package in packages
        if isinstance(package, dict)
    }
    missing = [name for name in THEROCK_RUNTIME_PACKAGES if name not in names]
    if missing:
        fail(
            f"manifest rocm_sdk packages missed runtime dependencies: {', '.join(missing)}"
        )
    if not any(name.lower().startswith("rocm-sdk-libraries") for name in names):
        fail("manifest rocm_sdk packages missed the family-specific ROCm library wheel")


def verify_rocm_runtime_libraries(
    python: Path,
    env: dict[str, str],
) -> dict[str, list[str]]:
    code = (
        "import json, rocm_sdk; "
        "names = ['amdhip64', 'hipblas']; "
        "print(json.dumps({name: [str(path) for path in rocm_sdk.find_libraries(name)] for name in names}, sort_keys=True))"
    )
    output = run(
        "rocm_sdk runtime library discovery",
        [str(python), "-c", code],
        env=env,
        timeout=120,
    )
    try:
        libraries = json.loads(output)
    except json.JSONDecodeError as error:
        fail(f"failed to parse rocm_sdk runtime library discovery: {error}\n{output}")
    if not isinstance(libraries, dict):
        fail(f"rocm_sdk runtime library discovery was not an object: {libraries!r}")
    for name in ["amdhip64", "hipblas"]:
        paths = libraries.get(name)
        if not isinstance(paths, list) or not paths:
            fail(f"rocm_sdk.find_libraries did not resolve {name}: {libraries!r}")
        for path in paths:
            if not Path(str(path)).is_file():
                fail(
                    f"rocm_sdk.find_libraries returned missing path for {name}: {path}"
                )
    return {
        str(name): [str(path) for path in paths] for name, paths in libraries.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["debug", "release"], default="debug")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument(
        "--root", type=Path, default=repo_root() / "target" / "therock-sdk-install"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="delete the test root before running"
    )
    parser.add_argument("--channel", choices=["release", "nightly"], default="release")
    parser.add_argument(
        "--family", help="override ROCM_CLI_THEROCK_FAMILY, e.g. gfx120X-all"
    )
    parser.add_argument(
        "--python", help="Python executable used by rocm-cli to create the SDK venv"
    )
    parser.add_argument(
        "--prefix", type=Path, help="explicit ROCm runtime folder to pass to rocm-cli"
    )
    parser.add_argument("--check-windows-tools", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the install plan without downloading wheels",
    )
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    if args.target_dir is not None and not args.target_dir.is_absolute():
        args.target_dir = repo_root() / args.target_dir
    if args.prefix is not None and not args.prefix.is_absolute():
        args.prefix = repo_root() / args.prefix
    test_root = args.root if args.root.is_absolute() else repo_root() / args.root

    if args.fresh and test_root.exists():
        remove_tree_with_retries(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    if args.check_windows_tools:
        report_windows_tool_hints()

    if not args.skip_build:
        cargo = resolve_cargo()
        run(
            "build rocm and llama.cpp adapter",
            [cargo, "build", "-p", "rocm", "-p", "rocm-engine-llama-cpp"],
            env=os.environ.copy(),
            timeout=1200,
        )

    rocm = binary_path(args.profile, args.target_dir, "rocm")
    llama = binary_path(args.profile, args.target_dir, "rocm-engine-llama-cpp")
    if not rocm.is_file():
        fail(f"missing rocm binary: {rocm}")
    if not llama.is_file():
        fail(f"missing llama.cpp adapter binary: {llama}")

    bootstrap_python = ensure_bootstrap_python(test_root, args.python)
    env = isolated_env(test_root, bootstrap_python, args.family)

    doctor = run(
        "rocm doctor before SDK install", [str(rocm), "doctor"], env=env, timeout=120
    )
    assert_contains(doctor, "rocm doctor", "doctor")

    install_argv = [
        str(rocm),
        "install",
        "sdk",
        "--channel",
        args.channel,
        "--format",
        "pip",
    ]
    if args.prefix is not None:
        install_argv.extend(["--prefix", str(args.prefix)])
    if args.dry_run:
        install_argv.append("--dry-run")
    install_output = run(
        "rocm install sdk pip",
        install_argv,
        env=env,
        timeout=args.timeout,
    )
    assert_contains(install_output, "sdk install", "sdk install")
    assert_contains(
        install_output,
        "summary: rocm-cli will install the ROCm SDK and matching PyTorch packages",
        "sdk install",
    )
    assert_contains(install_output, "format: pip", "sdk install")
    assert_contains(install_output, "pip_cache_dir:", "sdk install")
    assert_contains(install_output, "latest_compatible_version:", "sdk install")
    assert_contains(install_output, "python_wheel_tag:", "sdk install")
    assert_contains(install_output, "platform_wheel_tags:", "sdk install")
    assert_contains(install_output, "package_specs:", "sdk install")
    assert_contains(install_output, f"{THEROCK_SDK_PACKAGE_SPEC}==", "sdk install")
    for package in THEROCK_TORCH_PACKAGES:
        assert_contains(install_output, f"{package}==", "sdk install")
    assert_not_contains(install_output, "rocm[devel]", "sdk install")
    if args.dry_run:
        assert_contains(install_output, "mode: dry-run", "sdk dry-run")
        dry_run_target = install_output_field(install_output, "target")
        dry_run_pip_cache = install_output_field(install_output, "pip_cache_dir")
        expected_dry_run_pip_cache = dry_run_target / "pip-cache"
        if dry_run_pip_cache != expected_dry_run_pip_cache:
            fail(
                "sdk dry-run pip cache was not inside the selected ROCm folder: "
                f"{dry_run_pip_cache} != {expected_dry_run_pip_cache}"
            )
        if dry_run_pip_cache.exists():
            fail(
                "sdk dry-run created the pip cache directory before pip ran: "
                f"{dry_run_pip_cache}"
            )
        print("therock-sdk-install: dry-run ok")
        return 0

    assert_contains(install_output, "python_executable:", "sdk install")
    assert_contains(install_output, "rocm_sdk_root:", "sdk install")
    assert_contains(install_output, "rocm_sdk_bin:", "sdk install")
    assert_contains(install_output, "ROCm SDK installed successfully.", "sdk install")
    assert_contains(install_output, "next step: run `rocm help`", "sdk install")

    manifest_path, manifest = discover_manifest(Path(env["ROCM_CLI_DATA_DIR"]))
    install_root = Path(str(manifest.get("install_root") or ""))
    expected_pip_cache_dir = install_root / "pip-cache"
    runtime_python, manifest_root, manifest_bin, pip_cache_dir = verify_manifest(
        manifest_path,
        manifest,
        expected_pip_cache_dir,
    )

    version = run(
        "rocm_sdk version",
        [str(runtime_python), "-m", "rocm_sdk", "version"],
        env=env,
        timeout=120,
    )
    if not version.strip():
        fail("rocm_sdk version returned empty output")

    targets = run(
        "rocm_sdk targets",
        [str(runtime_python), "-m", "rocm_sdk", "targets"],
        env=env,
        timeout=120,
    )

    if not targets.strip():
        fail("rocm_sdk targets returned empty output")
    runtime_libraries = verify_rocm_runtime_libraries(runtime_python, env)

    torch_versions = verify_python_distributions(
        runtime_python, THEROCK_TORCH_PACKAGES, env
    )
    runtime_versions = verify_python_distributions(
        runtime_python, THEROCK_RUNTIME_PACKAGES, env
    )
    verify_rocm_manifest_packages(manifest)

    llama_detect = run(
        "llama.cpp adapter detects TheRock HIP env",
        [str(llama), "detect"],
        env=env,
        timeout=120,
    )
    assert_contains(
        llama_detect,
        "TheRock HIP runtime env available",
        "llama.cpp TheRock HIP env detection",
    )

    print("therock-sdk-install: ok")
    print(f"  test_root: {test_root}")
    print(f"  manifest: {manifest_path}")
    print(f"  runtime_python: {runtime_python}")
    print(f"  pip_cache_dir: {pip_cache_dir}")
    print(f"  rocm_sdk_root: {manifest_root}")
    print(f"  rocm_sdk_bin: {manifest_bin}")
    print(f"  targets: {targets.strip()}")
    print("  runtime_libraries:")
    for name, paths in runtime_libraries.items():
        print(f"    {name}: {', '.join(paths)}")
    print("  torch_packages:")
    for name, version_value in torch_versions.items():
        print(f"    {name}: {version_value}")
    print("  sdk_packages:")
    for name, version_value in runtime_versions.items():
        print(f"    {name}: {version_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
