#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Cross-platform local smoke tests for rocm-cli debug/release binaries."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"smoke failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def resolve_cargo() -> str:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    fallback = Path.home() / ".cargo" / "bin" / exe_name("cargo")
    if fallback.is_file():
        return str(fallback)
    fail("missing required command: cargo")


def run(
    label: str,
    argv: list[str],
    *,
    env: dict[str, str],
    expect_failure: bool = False,
) -> str:
    print(f"smoke: {label}")
    completed = subprocess.run(
        argv,
        cwd=repo_root(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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


def assert_contains_any(text: str, needles: list[str], label: str) -> None:
    if not any(needle in text for needle in needles):
        expected = " or ".join(repr(needle) for needle in needles)
        fail(f"{label} did not contain expected text: {expected}\n{text}")


def assert_not_contains(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"{label} contained unexpected text: {needle}\n{text}")


def assert_path_missing(path: Path, label: str) -> None:
    if path.exists():
        fail(f"{label} unexpectedly exists: {path}")


def parse_json(text: str, label: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        fail(f"{label} did not return valid JSON: {error}\n{text}")


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def isolated_env(smoke_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_CLI_UPDATE_USER_PATH"] = "0"
    env["ROCM_CLI_CONFIG_DIR"] = str(smoke_root / "rocm-config")
    env["ROCM_CLI_DATA_DIR"] = str(smoke_root / "rocm-data")
    env["ROCM_CLI_CACHE_DIR"] = str(smoke_root / "rocm-cache")
    if platform.system() == "Windows":
        env["APPDATA"] = str(smoke_root / "appdata")
        env["LOCALAPPDATA"] = str(smoke_root / "localappdata")
    else:
        env["XDG_CONFIG_HOME"] = str(smoke_root / "xdg-config")
        env["XDG_DATA_HOME"] = str(smoke_root / "xdg-data")
        env["XDG_CACHE_HOME"] = str(smoke_root / "xdg-cache")
        env["HOME"] = str(smoke_root / "home")
    return env


def binary_paths(
    root: Path, profile: str, target_dir: Path | None = None
) -> dict[str, Path]:
    binary_dir = (target_dir if target_dir is not None else root / "target") / profile
    return {
        "rocm": binary_dir / exe_name("rocm"),
        "rocmd": binary_dir / exe_name("rocmd"),
        "lemonade": binary_dir / exe_name("rocm-engine-lemonade"),
        "vllm": binary_dir / exe_name("rocm-engine-vllm"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["debug", "release"], default="debug")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()

    root = repo_root()
    profile = "release" if args.profile == "release" else "debug"
    smoke_root = root / "target" / "smoke-local"
    env = isolated_env(smoke_root)

    if not args.skip_build:
        cargo = resolve_cargo()
        run(
            "build workspace all targets",
            [cargo, "build", "--workspace", "--all-targets"],
            env=env,
        )

    target_dir = args.target_dir
    if target_dir is not None and not target_dir.is_absolute():
        target_dir = root / target_dir
    paths = binary_paths(root, profile, target_dir)
    for name, path in paths.items():
        if not path.is_file():
            fail(f"missing smoke binary {name}: {path}")

    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)
    reject_port = free_tcp_port()

    rocm = str(paths["rocm"])
    rocmd = str(paths["rocmd"])
    vllm = str(paths["vllm"])

    version = run("rocm version", [rocm, "version"], env=env)
    assert_contains(version, "rocm ", "rocm version")

    examine = run("rocm examine", [rocm, "examine"], env=env)
    assert_contains(examine, "rocm examine", "rocm examine")
    assert_contains(examine, "default_engine:", "rocm examine")
    assert_contains(examine, "managed_runtimes: 0", "rocm examine first-run state")
    assert_contains(examine, "managed_services: 0", "rocm examine first-run state")

    engines = run("rocm engines list", [rocm, "engines", "list"], env=env)
    assert_contains(engines, "lemonade", "rocm engines list")
    assert_contains(engines, "vllm", "rocm engines list")

    telemetry = run(
        "rocm config set telemetry off",
        [rocm, "config", "set-telemetry", "off"],
        env=env,
    )
    assert_contains(telemetry, "telemetry mode set to off", "telemetry off")
    assert_contains(telemetry, "policy: disabled", "telemetry off")
    config_show = run("rocm config show", [rocm, "config", "show"], env=env)
    assert_contains(config_show, "telemetry_mode: off", "config telemetry off")
    assert_contains(config_show, "telemetry_policy: disabled", "config telemetry off")

    engine_install = run(
        "rocm engines install requires exact runtime",
        [rocm, "engines", "install", "vllm"],
        env=env,
        expect_failure=True,
    )
    assert_contains(
        engine_install,
        "no active ROCm runtime is configured",
        "rocm engines install requires exact runtime",
    )

    chat = run("rocm chat local status", [rocm, "chat", "--provider", "local"], env=env)
    assert_contains_any(
        chat,
        ["Provider: local", "Assistant source: local model on this computer"],
        "rocm chat local status",
    )

    freeform_status = run(
        "rocm freeform installed status question",
        [rocm, "is rocm installed?"],
        env=env,
    )
    assert_contains(
        freeform_status, "ROCm status", "rocm freeform installed status question"
    )
    assert_contains(
        freeform_status,
        "Nothing was changed.",
        "rocm freeform installed status question",
    )
    assert_not_contains(
        freeform_status,
        "No ROCm action selected",
        "rocm freeform installed status question",
    )

    freeform_comfy_help = run(
        "rocm freeform comfyui help question",
        [rocm, "how do i setup comfyui"],
        env=env,
    )
    assert_contains(
        freeform_comfy_help, "ComfyUI status", "rocm freeform comfyui help question"
    )
    assert_contains(
        freeform_comfy_help,
        "Nothing was changed.",
        "rocm freeform comfyui help question",
    )
    assert_not_contains(
        freeform_comfy_help,
        "No ROCm action selected",
        "rocm freeform comfyui help question",
    )

    freeform_comfy_install = run(
        "rocm freeform comfyui install request",
        [rocm, "can you setup comfyui for me"],
        env=env,
    )
    assert_contains(
        freeform_comfy_install,
        "Install ComfyUI",
        "rocm freeform comfyui install request",
    )
    assert_contains(
        freeform_comfy_install,
        "approval: required",
        "rocm freeform comfyui install request",
    )

    plan = run(
        "rocm freeform vllm plan",
        [rocm, "serve qwen with vllm"],
        env=env,
    )
    assert_contains(plan, "engine: vllm", "rocm freeform plan")
    assert_contains(
        plan,
        "no CPU fallback is implied",
        "rocm freeform plan no fallback note",
    )

    tiny_plan = run(
        "rocm freeform tiny gpu recipe plan",
        [rocm, "serve tiny-gpt2"],
        env=env,
    )
    assert_contains(tiny_plan, "model: sshleifer/tiny-gpt2", "tiny gpu recipe plan")
    assert_contains(tiny_plan, "device_policy: gpu_required", "tiny gpu recipe plan")
    assert_contains(tiny_plan, "--device gpu_required", "tiny gpu recipe plan")
    assert_contains(tiny_plan, "approval: required", "tiny gpu recipe plan")

    if platform.system() == "Windows":
        tarball = run(
            "windows tarball sdk rejection",
            [rocm, "install", "sdk", "--format", "tarball", "--dry-run"],
            env=env,
            expect_failure=True,
        )
        assert_contains(
            tarball,
            "TheRock tarball installs are not supported on Windows",
            "windows tarball rejection",
        )

    status = run("rocmd status", [rocmd, "status"], env=env)
    assert_contains(status, "rocmd status", "rocmd status")

    bridge = parse_json(
        run("rocmd bridge snapshot", [rocmd, "bridge-snapshot"], env=env),
        "bridge snapshot",
    )
    if (
        not isinstance(bridge, dict)
        or bridge.get("protocol") != "rocmd-codex-bridge-v0"
    ):
        fail(f"unexpected bridge snapshot protocol: {bridge}")

    sandbox_examine = parse_json(
        run(
            "rocmd sandbox examine snapshot",
            [rocmd, "sandbox-run", "examine_snapshot", "--allow-native-fallback"],
            env=env,
        ),
        "sandbox examine snapshot",
    )
    if (
        not isinstance(sandbox_examine, dict)
        or sandbox_examine.get("tool") != "examine_snapshot"
        or not sandbox_examine.get("ok")
    ):
        fail(f"unexpected sandbox examine result: {sandbox_examine}")

    sandbox_servers = parse_json(
        run(
            "rocmd sandbox list servers",
            [rocmd, "sandbox-run", "list_servers", "--allow-native-fallback"],
            env=env,
        ),
        "sandbox list servers",
    )
    if (
        not isinstance(sandbox_servers, dict)
        or sandbox_servers.get("tool") != "list_servers"
        or not sandbox_servers.get("ok")
    ):
        fail(f"unexpected sandbox list result: {sandbox_servers}")

    prefetch_failure = run(
        "rocmd sandbox prefetch validation",
        [rocmd, "sandbox-run", "prefetch_artifact", "--allow-native-fallback"],
        env=env,
        expect_failure=True,
    )
    assert_contains(
        prefetch_failure, "prefetch_artifact requires", "sandbox prefetch validation"
    )

    parse_json(run("vllm detect", [vllm, "detect"], env=env), "vllm detect")
    vllm_capabilities = parse_json(
        run("vllm capabilities", [vllm, "capabilities"], env=env),
        "vllm capabilities",
    )
    if (
        not isinstance(vllm_capabilities, dict)
        or not vllm_capabilities.get("openai_compatible")
        or vllm_capabilities.get("cpu")
    ):
        fail(
            f"vllm capabilities did not report GPU-only OpenAI serving: {vllm_capabilities}"
        )

    vllm_model = run("vllm resolve qwen", [vllm, "resolve-model", "qwen"], env=env)
    assert_contains(vllm_model, "qwen", "vllm resolve qwen")
    vllm_cpu = run(
        "vllm reject cpu",
        [vllm, "resolve-model", "qwen", "--device-policy", "cpu_only"],
        env=env,
        expect_failure=True,
    )
    assert_contains(vllm_cpu, "no CPU fallback is used", "vllm reject cpu")

    vllm_gpu_required = run(
        "vllm reject required gpu",
        [
            rocm,
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--device",
            "gpu_required",
            "--foreground",
            "--port",
            str(reject_port),
        ],
        env=env,
        expect_failure=True,
    )
    assert_contains(vllm_gpu_required, "gpu_required", "vllm reject required gpu")
    assert_not_contains(
        vllm_gpu_required,
        "CPU fallback",
        "vllm reject required gpu",
    )

    vllm_cpu_serve = run(
        "rocm vllm reject cpu serve",
        [
            rocm,
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--device",
            "cpu",
            "--foreground",
            "--port",
            str(reject_port),
        ],
        env=env,
        expect_failure=True,
    )
    assert_contains(
        vllm_cpu_serve,
        "CPU mode is not a fallback path",
        "rocm vllm reject cpu serve",
    )

    assert_path_missing(
        smoke_root / "rocm-cache" / "uv",
        "first-run smoke uv cache",
    )
    assert_path_missing(
        smoke_root / "rocm-data" / "runtimes" / "registry",
        "first-run smoke runtime registry",
    )

    print("smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
