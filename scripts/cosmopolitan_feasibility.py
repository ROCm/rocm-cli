#!/usr/bin/env python3
"""Probe the true no-extract Cosmopolitan binary path.

This script verifies the local Rust/Cosmopolitan prerequisites and checks that
repo wording still separates platform-native helper artifacts from the true
universal rocm-cli APE built by `rust_cosmopolitan_spike.py`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COSMO_TARGET_KEYWORDS = ("cosmo", "cosmopolitan", "ape")
DEFAULT_RUSTUP_HOME = REPO_ROOT / ".rocm-work" / "tools" / "rustup"
DEFAULT_CARGO_HOME = REPO_ROOT / ".rocm-work" / "tools" / "cargo"
DEFAULT_TOOLCHAIN = "nightly"


class FeasibilityError(Exception):
    """The feasibility probe failed."""


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_capture(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(args, completed.returncode, completed.stdout, completed.stderr)


def local_rust_env() -> dict[str, str] | None:
    cargo_bin = DEFAULT_CARGO_HOME / "bin"
    if not DEFAULT_RUSTUP_HOME.is_dir():
        return None
    env = os.environ.copy()
    env["RUSTUP_HOME"] = str(DEFAULT_RUSTUP_HOME)
    env["CARGO_HOME"] = str(DEFAULT_CARGO_HOME)
    if cargo_bin.is_dir():
        env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")
    return env


def rust_target_list(rustc: str) -> tuple[str, list[str]]:
    env = local_rust_env()
    result = run_capture([rustc, "--print", "target-list"], env=env)
    label = rustc
    if result.returncode != 0 and "no default is configured" in result.stderr.lower():
        fallback = run_capture(
            [rustc, f"+{DEFAULT_TOOLCHAIN}", "--print", "target-list"], env=env
        )
        if fallback.returncode == 0:
            result = fallback
            label = f"{rustc} +{DEFAULT_TOOLCHAIN}"
    if result.returncode != 0:
        raise FeasibilityError(
            result.stderr.strip() or result.stdout.strip() or "rustc target-list failed"
        )
    return label, [line.strip() for line in result.stdout.splitlines() if line.strip()]


def has_builtin_cosmopolitan_target(targets: list[str]) -> bool:
    return any(
        any(keyword in target.lower() for keyword in COSMO_TARGET_KEYWORDS)
        for target in targets
    )


def matching_cosmopolitan_targets(targets: list[str]) -> list[str]:
    return [
        target
        for target in targets
        if any(keyword in target.lower() for keyword in COSMO_TARGET_KEYWORDS)
    ]


def compiler_looks_cosmopolitan(compiler: str) -> bool:
    name = Path(compiler).name.lower()
    return "cosmo" in name or name.startswith("ape-") or "unknown-cosmo" in name


def find_cosmocc(configured: str | None) -> str | None:
    if configured:
        return configured
    env = os.environ.get("ROCM_CLI_COSMOCC") or os.environ.get("ROCM_CLI_APE_CC")
    if env:
        return env
    for candidate in (
        REPO_ROOT / ".rocm-work" / "tools" / "cosmocc-wsl-elf" / "bin" / "cosmocc",
        REPO_ROOT / ".rocm-work" / "tools" / "cosmocc" / "bin" / "cosmocc",
        REPO_ROOT
        / ".rocm-work"
        / "tools"
        / "cosmocc"
        / "bin"
        / "x86_64-unknown-cosmo-cc",
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("cosmocc") or shutil.which("x86_64-unknown-cosmo-cc")


def compile_minimal_c_ape(compiler: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "hello.c"
    output = work_dir / "hello.com"
    source.write_text(
        '#include <stdio.h>\nint main(void) { puts("hello from cosmopolitan"); return 0; }\n',
        encoding="ascii",
    )
    result = run_capture([compiler, "-O2", "-o", str(output), str(source)])
    if result.returncode != 0:
        raise FeasibilityError(
            "cosmocc failed to compile a minimal C APE:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    return output


def repo_contract_messages() -> list[str]:
    release_script = (REPO_ROOT / "scripts" / "build_single_exe_release.py").read_text(
        encoding="utf-8"
    )
    messages: list[str] = []
    if "Cosmopolitan universal-binary release path" in release_script:
        messages.append(
            "platform-native release helper is explicitly not the universal-binary path"
        )
    else:
        raise FeasibilityError(
            "native release helper wording does not distinguish platform-native from universal"
        )
    if (
        "rust_cosmopolitan_spike.py" in release_script
        and "single_exe_release_gate.py" in release_script
    ):
        messages.append(
            "release helper points universal builds to the Rust/Cosmopolitan scripts"
        )
    else:
        raise FeasibilityError(
            "release helper does not point universal builds to the Rust/Cosmopolitan scripts"
        )
    return messages


def print_probe(args: argparse.Namespace) -> None:
    rustc = args.rustc
    rustc_label, targets = rust_target_list(rustc)
    matches = matching_cosmopolitan_targets(targets)
    print("Rust target probe")
    print(f"  rustc: {rustc_label}")
    print(f"  builtin Cosmopolitan/APE target: {'yes' if matches else 'no'}")
    if matches:
        print(f"  matching targets: {', '.join(matches)}")
    else:
        print(
            "  next step: use a pinned custom target/build-std spike or port the bootstrap/core to C/C++"
        )
    print()

    print("Repo release shape")
    for message in repo_contract_messages():
        print(f"  {message}")
    print()

    compiler = find_cosmocc(args.compiler)
    print("Cosmopolitan C probe")
    if not compiler:
        print("  cosmocc: not found")
        print(
            "  next step: run scripts/setup-cosmocc.sh from WSL/Linux or pass --compiler"
        )
        return
    print(f"  compiler: {compiler}")
    print(
        f"  compiler name check: {'cosmopolitan-looking' if compiler_looks_cosmopolitan(compiler) else 'not cosmopolitan-looking'}"
    )
    if args.compile_c:
        with tempfile.TemporaryDirectory(prefix="rocm-cosmo-probe-") as temp:
            output = compile_minimal_c_ape(compiler, Path(temp))
            print(f"  minimal C APE compiled: {output}")
    else:
        print("  compile check: skipped; pass --compile-c to build a minimal C APE")


def run_self_test() -> None:
    assert not has_builtin_cosmopolitan_target(
        ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"]
    )
    assert has_builtin_cosmopolitan_target(["x86_64-unknown-cosmo"])
    assert has_builtin_cosmopolitan_target(["x86_64-unknown-ape"])
    assert matching_cosmopolitan_targets(
        ["x86_64-unknown-linux-gnu", "x86_64-unknown-cosmo"]
    ) == ["x86_64-unknown-cosmo"]
    assert compiler_looks_cosmopolitan("cosmocc")
    assert compiler_looks_cosmopolitan("x86_64-unknown-cosmo-cc")
    assert not compiler_looks_cosmopolitan("clang.exe")
    messages = repo_contract_messages()
    assert len(messages) == 2
    print("cosmopolitan feasibility self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser(
        "probe", help="Inspect local Rust/Cosmopolitan feasibility."
    )
    probe.add_argument("--rustc", default="rustc")
    probe.add_argument("--compiler", help="Path to cosmocc or x86_64-unknown-cosmo-cc.")
    probe.add_argument(
        "--compile-c",
        action="store_true",
        help="Compile a minimal C APE with the selected compiler.",
    )
    subparsers.add_parser("self-test", help="Run offline parser/contract checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "probe":
            print_probe(args)
            return 0
        if args.command == "self-test":
            run_self_test()
            return 0
    except (FeasibilityError, OSError, subprocess.SubprocessError) as error:
        print(f"cosmopolitan feasibility failed: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
