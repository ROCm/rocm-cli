#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ATOM ROCm GPU smoke test for rocm-cli managed TheRock runtimes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from vllm_therock_gpu_test import (
    DEFAULT_MATH_MODULES,
    completion_text,
    first_model_id,
    get_json,
    http_request,
    load_runtime_registry_manifests,
    post_json,
    resolve_path,
    resolve_runtime_selector,
    restore_env,
    rocm_cli_state_paths,
    run_json,
    stop_service,
    verify_loaded_modules,
    verify_managed_env,
    write_config,
    write_runtime_manifest,
)


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    if platform.system() == "Windows":
        raise SystemExit(
            "ATOM GPU acceptance is Linux/WSL only; no CPU fallback is allowed"
        )

    repo_root = Path(__file__).resolve().parents[1]
    engine = resolve_path(args.engine, repo_root)
    runtime_id = resolve_runtime_id(args.runtime_id)
    math_modules = args.math_module or DEFAULT_MATH_MODULES

    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    reject_external_runtime_env()

    env = os.environ.copy()
    detect = run_json([str(engine), "detect"], env=env, timeout=args.timeout)
    assert_atom_gpu_detected(detect)

    capabilities = run_json(
        [str(engine), "capabilities"], env=env, timeout=args.timeout
    )
    if capabilities.get("cpu"):
        raise RuntimeError("ATOM capabilities unexpectedly report CPU support")
    if not capabilities.get("rocm_gpu"):
        raise RuntimeError("ATOM capabilities did not report ROCm GPU support")
    if not capabilities.get("openai_compatible"):
        raise RuntimeError("ATOM capabilities did not report OpenAI-compatible serving")
    assert_atom_cpu_policy_rejected(engine, args.model, env=env, timeout=args.timeout)

    process, state_path, log_path = start_atom(
        engine=engine,
        model=args.model,
        runtime_id=runtime_id,
        args=args,
        env=env,
        repo_root=repo_root,
    )

    try:
        health = wait_atom_health(args.host, args.port, args.timeout)
        models = get_json(args.host, args.port, "/v1/models", timeout=args.timeout)
        served_model = first_model_id(models)
        completion = post_json(
            args.host,
            args.port,
            "/v1/completions",
            {
                "model": served_model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            timeout=args.timeout,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        verify_atom_state(state)
        env_values = verify_managed_env(state)
        module_paths = verify_loaded_modules(state, math_modules)
        text = completion_text(completion)
        if not text.strip():
            raise RuntimeError("ATOM completion returned empty text")

        summary = {
            "ok": True,
            "launch_mode": args.launch_mode,
            "service_id": args.service_id,
            "model": args.model,
            "served_model": served_model,
            "runtime_id": runtime_id,
            "health": health,
            "completion_text": text,
            "state_path": str(state_path),
            "log_path": str(log_path),
            "server_pid": state.get("server_pid") or state.get("pid"),
            "therock_runtime_env": state.get("therock_runtime_env"),
            "verified_env": env_values,
            "verified_modules": module_paths,
        }
        print(json.dumps(summary, indent=2))
    finally:
        if not args.keep_running:
            stop_service(process, state_path)

    return 0


def parse_args() -> argparse.Namespace:
    default_engine = (
        Path("target/debug/rocm-engine-atom.exe")
        if platform.system() == "Windows"
        else Path("target/debug/rocm-engine-atom")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=str(default_engine))
    parser.add_argument(
        "--model",
        default=os.environ.get("ROCM_CLI_ATOM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        help="small Hugging Face model id or local model path served by ATOM",
    )
    parser.add_argument(
        "--runtime-id",
        help=(
            "managed TheRock runtime key or unambiguous runtime id; defaults to "
            "the active rocm-cli runtime"
        ),
    )
    parser.add_argument("--service-id", default="atom-gpu-e2e")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11443)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--launch-mode",
        choices=("serve-http", "launch"),
        default="serve-http",
        help="serve-http runs the wrapper directly; launch verifies background launch",
    )
    parser.add_argument(
        "--math-module",
        action="append",
        help=(
            "math-library basename where at least one must load from the managed "
            "TheRock SDK wheel directories; defaults to libhipblas/libhipblaslt/librocblas"
        ),
    )
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline selection checks for this script and exit",
    )
    return parser.parse_args()


def reject_external_runtime_env() -> None:
    blocked = [
        name
        for name in [
            "ROCM_CLI_ATOM_COMMAND",
            "ATOM_COMMAND",
            "ROCM_CLI_ATOM_PYTHON",
            "ATOM_PYTHON",
        ]
        if os.environ.get(name)
    ]
    if blocked:
        raise RuntimeError(
            "ATOM GPU acceptance requires discovery through a rocm-cli managed "
            f"TheRock runtime manifest; unset external runtime overrides: {', '.join(blocked)}"
        )


def resolve_runtime_id(explicit: str | None) -> str:
    config_path, registry_dir = rocm_cli_state_paths()
    manifests = load_runtime_registry_manifests(registry_dir)
    if explicit:
        return resolve_runtime_selector(explicit, manifests, "explicit --runtime-id")
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        active_key = config.get("active_runtime_key")
        if isinstance(active_key, str) and active_key.strip():
            return resolve_runtime_selector(
                active_key.strip(),
                manifests,
                f"active_runtime_key in {config_path}",
                exact_key_only=True,
            )
        runtime_id = config.get("default_runtime_id")
        if isinstance(runtime_id, str) and runtime_id.strip():
            return resolve_runtime_selector(
                runtime_id.strip(),
                manifests,
                f"default_runtime_id in {config_path}",
            )
    raise RuntimeError(
        "no active managed TheRock runtime was found; run "
        "`rocm runtimes activate <runtime_key>` or pass --runtime-id <runtime_key>"
    )


def assert_atom_gpu_detected(detect: dict[str, Any]) -> None:
    devices = detect.get("available_devices", [])
    rocm_gpu = next((item for item in devices if item.get("kind") == "rocm_gpu"), None)
    if not detect.get("installed") or not rocm_gpu or not rocm_gpu.get("available"):
        raise RuntimeError(
            "ATOM ROCm GPU runtime was not detected through a managed TheRock "
            "runtime; no CPU fallback is allowed:\n" + json.dumps(detect, indent=2)
        )
    if not detect.get("managed_env"):
        raise RuntimeError(
            "ATOM acceptance requires a rocm-cli managed TheRock runtime manifest"
        )


def assert_atom_cpu_policy_rejected(
    engine: Path,
    model: str,
    *,
    env: dict[str, str],
    timeout: int,
) -> None:
    completed = subprocess.run(
        [str(engine), "resolve-model", model, "--device-policy", "cpu_only"],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise RuntimeError(
            "ATOM accepted cpu_only during GPU acceptance; no fallback is allowed"
        )
    if "no CPU fallback is used" not in output:
        raise RuntimeError(
            "ATOM cpu_only rejection did not include the no-fallback explanation:\n"
            + output
        )


def verify_atom_state(state: dict[str, Any]) -> None:
    if state.get("device_policy") != "gpu_required":
        raise RuntimeError(f"ATOM service state did not record gpu_required: {state!r}")
    if not (state.get("server_pid") or state.get("pid")):
        raise RuntimeError("ATOM service state is missing server_pid/pid")
    runtime_env = state.get("therock_runtime_env")
    if not isinstance(runtime_env, dict):
        raise RuntimeError("ATOM service state is missing therock_runtime_env")
    source = runtime_env.get("source")
    if not isinstance(source, str) or not source.startswith("managed_runtime_manifest"):
        raise RuntimeError(
            "ATOM service did not launch from a managed TheRock runtime manifest; "
            f"source={source!r}"
        )


def wait_atom_health(host: str, port: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, body = http_request(host, port, "GET", "/health", None, timeout=3)
            if status < 400:
                return {
                    "status_code": status,
                    "body": body.decode("utf-8", errors="replace"),
                }
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"ATOM server did not become healthy: {last_error}")


def start_atom(
    *,
    engine: Path,
    model: str,
    runtime_id: str,
    args: argparse.Namespace,
    env: dict[str, str],
    repo_root: Path,
) -> tuple[subprocess.Popen[bytes] | None, Path, Path]:
    if args.launch_mode == "launch":
        return start_launch(engine, model, runtime_id, args, env)

    state_path = repo_root / "target" / "test-state" / f"{args.service_id}.json"
    log_path = repo_root / "target" / "test-logs" / f"{args.service_id}.log"
    command = [
        str(engine),
        "serve-http",
        args.service_id,
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device-policy",
        "gpu_required",
        "--runtime-id",
        runtime_id,
        "--state-path",
        str(state_path),
    ]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state_path.unlink()
    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_file.close()
    return process, state_path, log_path


def start_launch(
    engine: Path,
    model: str,
    runtime_id: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[None, Path, Path]:
    command = [
        str(engine),
        "launch",
        args.service_id,
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device-policy",
        "gpu_required",
        "--runtime-id",
        runtime_id,
    ]
    started = time.monotonic()
    response = run_json(command, env=env, timeout=args.timeout)
    elapsed = time.monotonic() - started
    if elapsed > 10:
        raise RuntimeError(
            f"launch took {elapsed:.1f}s; background launch should return promptly"
        )
    return None, Path(response["state_path"]), Path(response["log_path"])


def run_self_test() -> int:
    scratch_root = (
        Path(__file__).resolve().parents[1] / ".rocm-work" / "script-self-tests"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atom-", dir=scratch_root) as temp:
        root = Path(temp)
        config_dir = root / "config"
        data_dir = root / "data"
        registry_dir = data_dir / "runtimes" / "registry"
        registry_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        write_runtime_manifest(
            registry_dir, "runtime-old", "therock-release:gfx120X-all"
        )
        write_runtime_manifest(
            registry_dir, "runtime-new", "therock-release:gfx120X-all"
        )
        write_runtime_manifest(registry_dir, "runtime-other", "therock-release:gfx1151")
        write_config(
            config_dir,
            {
                "active_runtime_key": "runtime-old",
                "default_runtime_id": "therock-release:gfx120X-all",
            },
        )

        old_env = {
            "ROCM_CLI_CONFIG_DIR": os.environ.get("ROCM_CLI_CONFIG_DIR"),
            "ROCM_CLI_DATA_DIR": os.environ.get("ROCM_CLI_DATA_DIR"),
            "ROCM_CLI_ATOM_COMMAND": os.environ.get("ROCM_CLI_ATOM_COMMAND"),
            "ATOM_COMMAND": os.environ.get("ATOM_COMMAND"),
            "ROCM_CLI_ATOM_PYTHON": os.environ.get("ROCM_CLI_ATOM_PYTHON"),
            "ATOM_PYTHON": os.environ.get("ATOM_PYTHON"),
        }
        try:
            os.environ["ROCM_CLI_CONFIG_DIR"] = str(config_dir)
            os.environ["ROCM_CLI_DATA_DIR"] = str(data_dir)
            for key in [
                "ROCM_CLI_ATOM_COMMAND",
                "ATOM_COMMAND",
                "ROCM_CLI_ATOM_PYTHON",
                "ATOM_PYTHON",
            ]:
                os.environ.pop(key, None)

            assert resolve_runtime_id(None) == "runtime-old"
            assert resolve_runtime_id("runtime-new") == "runtime-new"
            assert resolve_runtime_id("therock-release:gfx1151") == "runtime-other"

            os.environ["ROCM_CLI_ATOM_COMMAND"] = "/tmp/not-used-atom"
            try:
                reject_external_runtime_env()
            except RuntimeError as exc:
                assert "managed TheRock runtime manifest" in str(exc)
            else:
                raise AssertionError("external ATOM override did not fail")
            os.environ.pop("ROCM_CLI_ATOM_COMMAND", None)

            write_config(
                config_dir, {"default_runtime_id": "therock-release:gfx120X-all"}
            )
            try:
                resolve_runtime_id(None)
            except RuntimeError as exc:
                message = str(exc)
                assert "runtime-old" in message and "runtime-new" in message
            else:
                raise AssertionError("ambiguous default_runtime_id did not fail")

            write_config(config_dir, {"active_runtime_key": "missing-runtime"})
            try:
                resolve_runtime_id(None)
            except RuntimeError as exc:
                assert "missing-runtime" in str(exc)
            else:
                raise AssertionError("missing active runtime key did not fail")
        finally:
            restore_env(old_env)

    detect = {
        "installed": True,
        "managed_env": True,
        "available_devices": [{"kind": "rocm_gpu", "available": True}],
    }
    assert_atom_gpu_detected(detect)
    verify_atom_state(
        {
            "server_pid": 123,
            "device_policy": "gpu_required",
            "therock_runtime_env": {
                "runtime_id": "therock-release:gfx120X-all",
                "source": "managed_runtime_manifest:runtime-old",
            },
        }
    )
    print("ATOM GPU script self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
