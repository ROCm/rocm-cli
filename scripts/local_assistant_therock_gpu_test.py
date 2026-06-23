#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Live local-assistant acceptance for rocm-cli managed GPU services.

This opt-in test starts a managed Lemonade service with GPU-required policy,
runs `rocm chat --tools --provider local`, verifies the request reached a
ready managed local service, then stops the service it started. It never uses
CPU fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen"
DEFAULT_ENGINE = "lemonade"
DEFAULT_PROMPT = "Use ROCm tools to check this computer and summarize the setup."


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    rocm = resolve_path(args.rocm, repo_root)
    if not rocm.is_file():
        raise SystemExit(
            f"rocm binary not found: {rocm}\n"
            "Build it with `cargo build -p rocm --bin rocm`, or pass --rocm."
        )

    env = os.environ.copy()
    temp_state_root: Path | None = None
    try:
        if args.temp_state:
            temp_state_root = Path(tempfile.mkdtemp(prefix="rocm-cli-assistant-state-"))
            apply_state_root(env, temp_state_root)
            print_step(f"Using temporary ROCm CLI state under {temp_state_root}.")
        elif args.state_root:
            state_root = resolve_path(args.state_root, repo_root)
            apply_state_root(env, state_root)
            print_step(f"Using ROCm CLI state under {state_root}.")
        if args.copy_runtime_state_from:
            if "ROCM_CLI_CONFIG_DIR" not in env or "ROCM_CLI_DATA_DIR" not in env:
                raise RuntimeError(
                    "--copy-runtime-state-from requires --temp-state or --state-root"
                )
            source_state = resolve_path(args.copy_runtime_state_from, repo_root)
            copy_runtime_state(source_state, env)
            print_step(f"Copied ROCm runtime registry from {source_state}.")

        localize_huggingface_cache(env, repo_root)
        return run_live_acceptance(args, rocm, env, temp_state_root)
    finally:
        if temp_state_root is not None and not args.keep_state:
            print_step(f"Removing temporary ROCm CLI state under {temp_state_root}.")
            shutil.rmtree(temp_state_root, ignore_errors=True)


def run_live_acceptance(
    args: argparse.Namespace,
    rocm: Path,
    env: dict[str, str],
    temp_state_root: Path | None,
) -> int:
    data_dir = rocm_cli_data_dir(env)
    service_id = args.service_id
    manifest_path: Path | None = None
    started_service = False

    try:
        if args.skip_serve:
            if not service_id:
                service_id = find_ready_service_id(
                    data_dir, args.chat_model or args.model, args.engine
                )
            print_step(f"Using existing managed service {service_id}.")
        else:
            serve_cmd = build_serve_command(args, rocm)
            print_step(
                f"Starting managed {args.engine} local assistant service in GPU-required mode."
            )
            serve_output = run_text(serve_cmd, env=env, timeout=args.timeout)
            print(serve_output, end="" if serve_output.endswith("\n") else "\n")
            manifest_path = parse_manifest_path(serve_output)
            service_id = parse_service_id(serve_output) or find_latest_service_id(
                data_dir, args.model, args.engine
            )
            started_service = True

        manifest = (
            wait_ready_manifest_path(manifest_path, args.timeout)
            if manifest_path is not None
            else wait_ready_manifest(data_dir, service_id, args.timeout)
        )
        assert_service_manifest(manifest, expected_engine=args.engine)
        wait_local_endpoint(manifest, args.timeout)
        chat_model = args.chat_model or manifest.get("canonical_model_id") or args.model
        chat_cmd = build_chat_command(args, rocm, str(chat_model))
        print_step("Running local assistant chat with ROCm tools enabled.")
        chat_output = run_text(chat_cmd, env=env, timeout=args.timeout)
        print(chat_output, end="" if chat_output.endswith("\n") else "\n")
        assert_chat_output(chat_output, require_tool_call=args.require_tool_call)

        summary = {
            "ok": True,
            "message": "Success: local assistant reached a managed ROCm GPU service.",
            "service_id": service_id,
            "model": chat_model,
            "endpoint": manifest.get("endpoint_url"),
            "runtime_id": manifest.get("runtime_id"),
            "env_id": manifest.get("env_id"),
            "device_policy": manifest.get("device_policy"),
            "manifest_path": str(
                manifest_path or service_manifest_path(data_dir, service_id)
            ),
        }
        print_step("Success: local assistant used the managed local service.")
        print(json.dumps(summary, indent=2))
    finally:
        if started_service and not args.keep_running and service_id:
            print_step(f"Stopping managed service {service_id}.")
            stop_cmd = build_rocm_command(
                args, rocm, "services", "stop", service_id, "--yes"
            )
            stop_output = run_text(stop_cmd, env=env, timeout=args.timeout, check=False)
            print(stop_output, end="" if stop_output.endswith("\n") else "\n")

    return 0


def parse_args() -> argparse.Namespace:
    default_rocm = cargo_binary_path("debug", "rocm")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", default=str(default_rocm))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument(
        "--chat-model",
        help="model filter passed to `rocm chat`; defaults to the launched service canonical id",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11442)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument(
        "--rocm-launch-prefix",
        nargs="+",
        default=[],
        help="optional prefix used to launch rocm, for example a wrapper command",
    )
    parser.add_argument("--runtime-id", help="exact managed TheRock runtime key")
    parser.add_argument("--env-id", help="managed engine env id")
    parser.add_argument("--service-id", help="service id to use with --skip-serve")
    parser.add_argument(
        "--state-root",
        help="use this root for ROCM_CLI_CONFIG_DIR/DATA_DIR/CACHE_DIR",
    )
    parser.add_argument(
        "--temp-state",
        action="store_true",
        help="create a temporary isolated ROCm CLI state root for this run",
    )
    parser.add_argument("--keep-state", action="store_true")
    parser.add_argument(
        "--copy-runtime-state-from",
        help=(
            "copy config.json and runtimes/ from an existing ROCm CLI state root "
            "into the isolated state before the test starts"
        ),
    )
    parser.add_argument(
        "--skip-serve",
        action="store_true",
        help="reuse an existing ready managed service instead of launching one",
    )
    parser.add_argument(
        "--require-tool-call",
        action="store_true",
        help="fail unless the model requests and rocm-cli renders a ROCm tool result",
    )
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline checks for this script and exit",
    )
    args = parser.parse_args()
    if args.temp_state and args.state_root:
        parser.error("--temp-state and --state-root are mutually exclusive")
    return args


def print_step(message: str) -> None:
    print(f"[local-assistant-gpu-test] {message}", flush=True)


def cargo_binary_path(profile: str, name: str) -> Path:
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", "target")).expanduser()
    return target_root / profile / exe_name(name)


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def apply_state_root(env: dict[str, str], state_root: Path) -> None:
    env["ROCM_CLI_CONFIG_DIR"] = str(state_root / "config")
    env["ROCM_CLI_DATA_DIR"] = str(state_root / "data")
    env["ROCM_CLI_CACHE_DIR"] = str(state_root / "cache")


def copy_runtime_state(source_root: Path, env: dict[str, str]) -> None:
    if not source_root.exists():
        raise RuntimeError(f"runtime state root does not exist: {source_root}")
    config_dir = Path(env["ROCM_CLI_CONFIG_DIR"])
    data_dir = Path(env["ROCM_CLI_DATA_DIR"])
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    copy_first_existing_file(
        [
            source_root / "config.json",
            source_root / "config" / "config.json",
        ],
        config_dir / "config.json",
    )
    copied_runtime_dir = copy_first_existing_dir(
        [
            source_root / "runtimes",
            source_root / "data" / "runtimes",
        ],
        data_dir / "runtimes",
    )
    if not copied_runtime_dir:
        raise RuntimeError(
            f"no runtimes directory found under {source_root}; "
            "install TheRock first or pass a state root with runtimes/"
        )


def copy_first_existing_file(candidates: list[Path], destination: Path) -> bool:
    for source in candidates:
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return True
    return False


def copy_first_existing_dir(candidates: list[Path], destination: Path) -> bool:
    for source in candidates:
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            return True
    return False


def build_rocm_command(args: argparse.Namespace, rocm: Path, *parts: str) -> list[str]:
    return [*args.rocm_launch_prefix, str(rocm), *parts]


def build_serve_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    command = build_rocm_command(
        args,
        rocm,
        "serve",
        args.model,
        "--engine",
        args.engine,
        "--device",
        "gpu_required",
        "--managed",
        "--host",
        args.host,
        "--port",
        str(args.port),
    )
    if args.runtime_id:
        command.extend(["--runtime-id", args.runtime_id])
    if args.env_id:
        command.extend(["--env-id", args.env_id])
    return command


def build_chat_command(args: argparse.Namespace, rocm: Path, model: str) -> list[str]:
    return build_rocm_command(
        args,
        rocm,
        "chat",
        "--tools",
        "--provider",
        "local",
        "--model",
        model,
        "--prompt",
        args.prompt,
    )


def run_text(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {format_command(command)}\n{output}"
        )
    return output


def format_command(command: list[str]) -> str:
    return " ".join(command)


def parse_service_id(output: str) -> str | None:
    match = re.search(r"(?m)^\s*service_id:\s*([A-Za-z0-9_.-]+)\s*$", output)
    return match.group(1) if match else None


def parse_manifest_path(output: str) -> Path | None:
    match = re.search(r"(?m)^\s*manifest_path:\s*(.+?)\s*$", output)
    if not match:
        return None
    return runtime_path_to_host_path(match.group(1).strip())


def rocm_cli_data_dir(env: dict[str, str]) -> Path:
    override = env.get("ROCM_CLI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".rocm"


def service_manifest_path(data_dir: Path, service_id: str) -> Path:
    return data_dir / "services" / f"{service_id}.json"


def wait_ready_manifest(
    data_dir: Path, service_id: str, timeout: int
) -> dict[str, Any]:
    path = service_manifest_path(data_dir, service_id)
    return wait_ready_manifest_path(path, timeout)


def wait_ready_manifest_path(path: Path, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status = "<missing>"
    last_engine_status = "<missing>"
    while time.monotonic() < deadline:
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            last_status = str(manifest.get("status", "<missing>"))
            if last_status == "ready":
                return manifest
            engine_state_path = manifest.get("engine_state_path")
            if isinstance(engine_state_path, str) and engine_state_path:
                engine_state = runtime_path_to_host_path(engine_state_path)
                if engine_state.is_file():
                    state = json.loads(engine_state.read_text(encoding="utf-8"))
                    last_engine_status = str(state.get("status", "<missing>"))
                    if last_engine_status == "ready":
                        manifest["status"] = "ready"
                        manifest["engine_state"] = state
                        return manifest
            if last_status in {"failed", "exited", "unreachable"}:
                raise RuntimeError(
                    f"managed service manifest reached {last_status}; inspect {path}"
                )
        time.sleep(1)
    raise RuntimeError(
        "managed service did not become ready before timeout; "
        f"last status: {last_status}; engine status: {last_engine_status}; manifest: {path}"
    )


def runtime_path_to_host_path(value: str) -> Path:
    if platform.system() == "Windows":
        normalized = value.replace("\\", "/")
        if (
            len(normalized) >= 3
            and normalized[0] == "/"
            and normalized[1].isalpha()
            and normalized[2] == "/"
        ):
            return Path(
                f"{normalized[1].upper()}:\\" + normalized[3:].replace("/", "\\")
            )
    return Path(value)


def assert_service_manifest(manifest: dict[str, Any], *, expected_engine: str) -> None:
    if manifest.get("engine") != expected_engine:
        raise RuntimeError(
            f"expected {expected_engine} service, got {manifest.get('engine')}"
        )
    if manifest.get("device_policy") != "gpu_required":
        raise RuntimeError(
            "managed service did not record gpu_required device policy; "
            f"manifest device_policy={manifest.get('device_policy')!r}"
        )
    endpoint = manifest.get("endpoint_url")
    if not isinstance(endpoint, str) or not endpoint.startswith("http://"):
        raise RuntimeError(f"managed service endpoint is invalid: {endpoint!r}")
    canonical = manifest.get("canonical_model_id")
    if not isinstance(canonical, str) or not canonical.strip():
        raise RuntimeError("managed service manifest is missing canonical_model_id")


def wait_local_endpoint(manifest: dict[str, Any], timeout: int) -> None:
    endpoint = str(manifest.get("endpoint_url") or "")
    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        raise RuntimeError(f"managed service endpoint is invalid: {endpoint!r}")
    engine = str(manifest.get("engine") or "")
    model_names = {
        str(manifest.get("model_ref") or ""),
        str(manifest.get("canonical_model_id") or ""),
    }
    engine_state = manifest.get("engine_state")
    if not isinstance(engine_state, dict):
        engine_state = {}
    lemonade_direct_rocm = engine == "lemonade" and str(
        engine_state.get("backend_requested") or manifest.get("backend_requested") or ""
    ).strip().lower().startswith("rocm")
    probes = (
        [("/v1/health", "health"), ("/v1/models", "models")]
        if engine == "lemonade"
        else [("/v1/models", "models")]
    )
    deadline = time.monotonic() + timeout
    last_error = "endpoint was not checked"
    while time.monotonic() < deadline:
        for path, source in probes:
            try:
                conn = http.client.HTTPConnection(host, port, timeout=5)
                conn.request("GET", path)
                response = conn.getresponse()
                body = response.read().decode("utf-8", errors="replace")
                require_rocm_backend = engine == "lemonade" and not (
                    source == "models" and lemonade_direct_rocm
                )
                if response.status == 200 and endpoint_payload_has_loaded_model(
                    body,
                    model_names,
                    require_rocm_backend=require_rocm_backend,
                    source=source,
                ):
                    return
                last_error = (
                    f"model was not loaded yet; HTTP {response.status} from {path}; "
                    f"endpoint={endpoint}"
                )
            except OSError as error:
                last_error = str(error)
            finally:
                with contextlib.suppress(Exception):
                    conn.close()  # type: ignore[name-defined]
        time.sleep(2)
    raise RuntimeError(
        "managed service endpoint did not report the requested loaded model "
        f"before timeout: {last_error}"
    )


def endpoint_payload_has_loaded_model(
    body: str,
    model_names: set[str],
    *,
    require_rocm_backend: bool,
    source: str,
) -> bool:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    if source == "health":
        entries = payload.get("all_models_loaded")
    else:
        entries = payload.get("data")
    if not isinstance(entries, list):
        return False
    return any(
        payload_entry_matches_model(
            entry, model_names, require_rocm_backend=require_rocm_backend
        )
        for entry in entries
        if isinstance(entry, dict)
    )


def payload_entry_matches_model(
    entry: dict[str, Any],
    model_names: set[str],
    *,
    require_rocm_backend: bool,
) -> bool:
    loaded_names = [
        str(entry.get(field) or "") for field in ("model_name", "id", "model", "name")
    ]
    name_matches = any(
        service_model_names_match(loaded, expected)
        for loaded in loaded_names
        for expected in model_names
        if expected
    )
    if not name_matches:
        return False
    return not require_rocm_backend or payload_entry_reports_rocm_backend(entry)


def payload_entry_reports_rocm_backend(entry: dict[str, Any]) -> bool:
    recipe_options = entry.get("recipe_options")
    backend = None
    if isinstance(recipe_options, dict):
        backend = recipe_options.get("llamacpp_backend")
    if backend is None:
        backend = entry.get("llamacpp_backend")
    return isinstance(backend, str) and backend.strip().lower().startswith("rocm")


def service_model_names_match(left: str, right: str) -> bool:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return True
    left = strip_model_suffix(left).lower()
    right = strip_model_suffix(right).lower()
    return left in right or right in left


def strip_model_suffix(value: str) -> str:
    lower = value.lower()
    for suffix in (".gguf", ".safetensors"):
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return value


def assert_chat_output(output: str, *, require_tool_call: bool) -> None:
    required = [
        "chat response",
        "provider: local",
        "rocm tools: enabled",
    ]
    for needle in required:
        if needle not in output:
            raise RuntimeError(f"chat output did not contain `{needle}`:\n{output}")
    forbidden = [
        "No local assistant is running yet",
        "local provider has no ready managed service",
        "CPU fallback",
        "cpu_only",
    ]
    for needle in forbidden:
        if needle in output:
            raise RuntimeError(f"chat output contained forbidden `{needle}`:\n{output}")
    if require_tool_call and (
        "ROCm checks used" not in output or "none requested" in output
    ):
        raise RuntimeError(
            "local assistant did not request a ROCm tool call; retry without "
            "--require-tool-call or choose a tool-calling model/prompt.\n" + output
        )


def find_ready_service_id(data_dir: Path, model: str, engine: str) -> str:
    services_dir = data_dir / "services"
    if not services_dir.is_dir():
        raise RuntimeError(f"no managed services directory found: {services_dir}")
    matches: list[tuple[int, str]] = []
    for path in services_dir.glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") != "ready":
            continue
        if manifest.get("engine") != engine:
            continue
        names = {
            str(manifest.get("model_ref", "")),
            str(manifest.get("canonical_model_id", "")),
        }
        if model and not any(name.lower() == model.lower() for name in names):
            continue
        created = int(manifest.get("created_at_unix_ms") or 0)
        service_id = manifest.get("service_id")
        if isinstance(service_id, str) and service_id:
            matches.append((created, service_id))
    if not matches:
        raise RuntimeError(
            f"no ready {engine} managed service for `{model}`; run without --skip-serve"
        )
    matches.sort(reverse=True)
    return matches[0][1]


def find_latest_service_id(data_dir: Path, model: str, engine: str) -> str:
    services_dir = data_dir / "services"
    if not services_dir.is_dir():
        raise RuntimeError(
            "managed serve output did not include a service_id and no services "
            f"directory was found: {services_dir}"
        )
    matches: list[tuple[int, str]] = []
    for path in services_dir.glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("engine") != engine:
            continue
        names = {
            str(manifest.get("model_ref", "")),
            str(manifest.get("canonical_model_id", "")),
        }
        if model and not any(name.lower() == model.lower() for name in names):
            continue
        service_id = manifest.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            continue
        created = int(
            manifest.get("created_at_unix_ms") or path.stat().st_mtime_ns // 1_000_000
        )
        matches.append((created, service_id))
    if not matches:
        raise RuntimeError(
            "managed serve output did not include a service_id and no matching "
            f"{engine} service manifest was found for `{model}` under {services_dir}"
        )
    matches.sort(reverse=True)
    return matches[0][1]


def localize_huggingface_cache(env: dict[str, str], repo_root: Path) -> None:
    cache_root = Path(
        env.get("ROCM_CLI_CACHE_DIR", repo_root / "target" / "test-cache")
    )
    hf_root = cache_root / "huggingface"
    env.setdefault("HF_HOME", str(hf_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))


def run_self_test() -> int:
    fake_rocm = Path("target/debug") / exe_name("rocm")
    args = argparse.Namespace(
        model="qwen",
        engine="lemonade",
        chat_model=None,
        prompt=DEFAULT_PROMPT,
        host="127.0.0.1",
        port=11442,
        rocm_launch_prefix=["sh"],
        runtime_id="release-pip-gfx120x-all-7-14-0a20260601",
        env_id="windows-release-pip-gfx120x-all-7-14-0a20260601-3-12",
    )
    serve = build_serve_command(args, fake_rocm)
    assert serve[:2] == ["sh", str(fake_rocm)]
    assert "--device" in serve
    assert "gpu_required" in serve
    assert "cpu" not in " ".join(serve).lower()
    chat = build_chat_command(args, fake_rocm, "Qwen3-0.6B-GGUF")
    assert chat[:2] == ["sh", str(fake_rocm)]
    assert "--tools" in chat
    assert "--provider" in chat and "local" in chat
    assert (
        parse_service_id("managed service launched\n  service_id: svc-qwen\n")
        == "svc-qwen"
    )
    assert parse_service_id("managed service launched\n") is None
    assert parse_manifest_path(
        "managed service launched\n  manifest_path: /tmp/svc.json\n"
    ) == Path("/tmp/svc.json")
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        services = data_dir / "services"
        services.mkdir()
        manifest_path = services / "svc-qwen.json"
        manifest = {
            "service_id": "svc-qwen",
            "engine": "lemonade",
            "model_ref": "qwen",
            "canonical_model_id": "Qwen3-0.6B-GGUF",
            "endpoint_url": "http://127.0.0.1:11442/v1",
            "status": "ready",
            "device_policy": "gpu_required",
            "created_at_unix_ms": 1,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = wait_ready_manifest(data_dir, "svc-qwen", timeout=1)
        assert_service_manifest(loaded, expected_engine="lemonade")
        ready_payload = json.dumps(
            {
                "all_models_loaded": [
                    {
                        "model_name": "Qwen3-0.6B-GGUF",
                        "recipe_options": {"llamacpp_backend": "rocm"},
                    }
                ]
            }
        )
        assert endpoint_payload_has_loaded_model(
            ready_payload,
            {"qwen", "Qwen3-0.6B-GGUF"},
            require_rocm_backend=True,
            source="health",
        )
        cpu_payload = json.dumps(
            {
                "all_models_loaded": [
                    {
                        "model_name": "Qwen3-0.6B-GGUF",
                        "recipe_options": {"llamacpp_backend": "cpu"},
                    }
                ]
            }
        )
        assert not endpoint_payload_has_loaded_model(
            cpu_payload,
            {"qwen", "Qwen3-0.6B-GGUF"},
            require_rocm_backend=True,
            source="health",
        )
        assert find_ready_service_id(data_dir, "qwen", "lemonade") == "svc-qwen"
        assert find_latest_service_id(data_dir, "qwen", "lemonade") == "svc-qwen"
    assert rocm_cli_data_dir({"ROCM_CLI_DATA_DIR": "custom-data"}) == Path(
        "custom-data"
    )
    with (
        tempfile.TemporaryDirectory() as src_text,
        tempfile.TemporaryDirectory() as dst_text,
    ):
        source = Path(src_text)
        dest = Path(dst_text)
        (source / "runtimes" / "registry").mkdir(parents=True)
        (source / "runtimes" / "active.json").write_text("{}", encoding="utf-8")
        (source / "config.json").write_text(
            json.dumps({"active_runtime_key": "runtime-a"}),
            encoding="utf-8",
        )
        env = {}
        apply_state_root(env, dest)
        copy_runtime_state(source, env)
        assert (dest / "config" / "config.json").is_file()
        assert (dest / "data" / "runtimes" / "active.json").is_file()
    assert_chat_output(
        "chat response\n  provider: local\n  model: qwen\n  rocm tools: enabled\n\nhello",
        require_tool_call=False,
    )
    print("local-assistant-gpu-test self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
