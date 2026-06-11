#!/usr/bin/env python3
"""End-to-end vLLM ROCm GPU smoke test for rocm-cli managed TheRock runtimes."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


HIP_MODULE = "libamdhip64"
DEFAULT_MATH_MODULES = ["libhipblas", "libhipblaslt", "librocblas"]


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    if platform.system() == "Windows":
        print(
            json.dumps(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": (
                        "vLLM GPU acceptance is skipped on native Windows; "
                        "use WSL/Linux for ROCm GPU serving. No CPU fallback is allowed."
                    ),
                },
                indent=2,
            )
        )
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    engine = resolve_path(args.engine, repo_root)
    runtime_id = resolve_runtime_id(args.runtime_id)
    math_modules = args.math_module or DEFAULT_MATH_MODULES

    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    reject_external_runtime_env()

    env = os.environ.copy()
    detect = run_json([str(engine), "detect"], env=env, timeout=args.timeout)
    assert_vllm_gpu_detected(detect)

    capabilities = run_json([str(engine), "capabilities"], env=env, timeout=args.timeout)
    if capabilities.get("cpu"):
        raise RuntimeError("vLLM capabilities unexpectedly report CPU support")
    if not capabilities.get("rocm_gpu"):
        raise RuntimeError("vLLM capabilities did not report ROCm GPU support")

    process, state_path, log_path = start_vllm(
        engine=engine,
        model=args.model,
        runtime_id=runtime_id,
        args=args,
        env=env,
        repo_root=repo_root,
    )

    try:
        health = wait_health(args.host, args.port, args.timeout)
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
        env_values = verify_managed_env(state)
        module_paths = verify_loaded_modules(state, math_modules)
        text = completion_text(completion)
        if not text.strip():
            raise RuntimeError("vLLM completion returned empty text")

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
        Path("target/debug/rocm-engine-vllm.exe")
        if platform.system() == "Windows"
        else Path("target/debug/rocm-engine-vllm")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=str(default_engine))
    parser.add_argument(
        "--model",
        default=os.environ.get("ROCM_CLI_VLLM_TEST_MODEL", "facebook/opt-125m"),
        help="small Hugging Face model id or local model path served by vLLM",
    )
    parser.add_argument(
        "--runtime-id",
        help=(
            "managed TheRock runtime key or unambiguous runtime id; defaults to "
            "the active rocm-cli runtime"
        ),
    )
    parser.add_argument("--service-id", default="vllm-gpu-e2e")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11440)
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


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def reject_external_runtime_env() -> None:
    blocked = [
        name
        for name in [
            "ROCM_CLI_VLLM_COMMAND",
            "VLLM_COMMAND",
            "ROCM_CLI_VLLM_PYTHON",
            "VLLM_PYTHON",
        ]
        if os.environ.get(name)
    ]
    if blocked:
        raise RuntimeError(
            "vLLM GPU acceptance requires discovery through a rocm-cli managed "
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


def load_runtime_registry_manifests(registry_dir: Path) -> list[dict[str, str]]:
    manifests: list[dict[str, str]] = []
    if not registry_dir.is_dir():
        return manifests
    for path in registry_dir.glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runtime_key = manifest.get("runtime_key")
        runtime_id = manifest.get("runtime_id")
        if not isinstance(runtime_key, str) or not runtime_key.strip():
            runtime_key = path.stem
        if not isinstance(runtime_id, str) or not runtime_id.strip():
            continue
        manifests.append(
            {
                "runtime_key": runtime_key.strip(),
                "runtime_id": runtime_id.strip(),
            }
        )
    return manifests


def resolve_runtime_selector(
    selector: str,
    manifests: list[dict[str, str]],
    source: str,
    *,
    exact_key_only: bool = False,
) -> str:
    exact_key_matches = [
        manifest
        for manifest in manifests
        if manifest["runtime_key"].lower() == selector.lower()
    ]
    if len(exact_key_matches) == 1:
        return exact_key_matches[0]["runtime_key"]
    if exact_key_only:
        raise RuntimeError(
            f"{source} points to `{selector}`, but that exact runtime key is not "
            "registered; run `rocm runtimes list` and activate an installed runtime"
        )

    runtime_id_matches = [
        manifest
        for manifest in manifests
        if manifest["runtime_id"].lower() == selector.lower()
    ]
    if len(runtime_id_matches) == 1:
        return runtime_id_matches[0]["runtime_key"]
    if len(runtime_id_matches) > 1:
        raise RuntimeError(
            f"{source} selector `{selector}` matches more than one installed runtime. "
            "Activate one by runtime_key first: "
            + runtime_keys_text(runtime_id_matches)
        )
    raise RuntimeError(
        f"{source} selector `{selector}` did not match any registered runtime; "
        "run `rocm runtimes list` and activate an installed runtime"
    )


def runtime_keys_text(manifests: list[dict[str, str]]) -> str:
    keys = sorted({manifest["runtime_key"] for manifest in manifests})
    return ", ".join(keys) if keys else "<none>"


def rocm_cli_state_paths() -> tuple[Path, Path]:
    config_dir = os.environ.get("ROCM_CLI_CONFIG_DIR")
    data_dir = os.environ.get("ROCM_CLI_DATA_DIR")
    if config_dir or data_dir:
        config_base = (
            Path(config_dir)
            if config_dir
            else default_rocm_cli_config_dir()
        )
        data_base = (
            Path(data_dir)
            if data_dir
            else default_rocm_cli_data_dir(config_base)
        )
        return (
            config_base / "config.json",
            Path(data_base) / "runtimes" / "registry",
        )
    return (
        default_rocm_cli_config_dir() / "config.json",
        default_rocm_cli_data_dir(default_rocm_cli_config_dir()) / "runtimes" / "registry",
    )


def default_rocm_cli_config_dir() -> Path:
    return Path.home() / ".rocm"


def default_rocm_cli_data_dir(config_base: Path) -> Path:
    config_path = config_base / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Path.home() / ".rocm"
    setup = config.get("setup")
    if isinstance(setup, dict):
        therock_venv = setup.get("therock_venv")
        if isinstance(therock_venv, str) and therock_venv.strip():
            return Path(therock_venv).expanduser()
    return Path.home() / ".rocm"


def run_self_test() -> int:
    scratch_root = Path(__file__).resolve().parents[1] / ".rocm-work" / "script-self-tests"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="vllm-", dir=scratch_root
    ) as temp:
        root = Path(temp)
        config_dir = root / "config"
        data_dir = root / "data"
        registry_dir = data_dir / "runtimes" / "registry"
        registry_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        write_runtime_manifest(registry_dir, "runtime-old", "therock-release:gfx120X-all")
        write_runtime_manifest(registry_dir, "runtime-new", "therock-release:gfx120X-all")
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
            "ROCM_CLI_VLLM_COMMAND": os.environ.get("ROCM_CLI_VLLM_COMMAND"),
            "VLLM_COMMAND": os.environ.get("VLLM_COMMAND"),
            "ROCM_CLI_VLLM_PYTHON": os.environ.get("ROCM_CLI_VLLM_PYTHON"),
            "VLLM_PYTHON": os.environ.get("VLLM_PYTHON"),
        }
        try:
            os.environ["ROCM_CLI_CONFIG_DIR"] = str(config_dir)
            os.environ["ROCM_CLI_DATA_DIR"] = str(data_dir)
            for key in [
                "ROCM_CLI_VLLM_COMMAND",
                "VLLM_COMMAND",
                "ROCM_CLI_VLLM_PYTHON",
                "VLLM_PYTHON",
            ]:
                os.environ.pop(key, None)

            assert resolve_runtime_id(None) == "runtime-old"
            assert resolve_runtime_id("runtime-new") == "runtime-new"
            assert resolve_runtime_id("therock-release:gfx1151") == "runtime-other"

            os.environ["ROCM_CLI_VLLM_COMMAND"] = "/tmp/not-used-vllm"
            try:
                reject_external_runtime_env()
            except RuntimeError as exc:
                assert "managed TheRock runtime manifest" in str(exc)
            else:
                raise AssertionError("external vLLM override did not fail")
            os.environ.pop("ROCM_CLI_VLLM_COMMAND", None)

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
    print("vLLM GPU script self-test passed")
    return 0


def write_runtime_manifest(registry_dir: Path, runtime_key: str, runtime_id: str) -> None:
    (registry_dir / f"{runtime_key}.json").write_text(
        json.dumps({"runtime_key": runtime_key, "runtime_id": runtime_id}),
        encoding="utf-8",
    )


def write_config(config_dir: Path, payload: dict[str, str]) -> None:
    (config_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def restore_env(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_json(command: list[str], *, env: dict[str, str], timeout: int) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"command did not return JSON: {' '.join(command)}\n{completed.stdout}"
        ) from exc


def assert_vllm_gpu_detected(detect: dict[str, Any]) -> None:
    devices = detect.get("available_devices", [])
    rocm_gpu = next((item for item in devices if item.get("kind") == "rocm_gpu"), None)
    if not detect.get("installed") or not rocm_gpu or not rocm_gpu.get("available"):
        raise RuntimeError(
            "vLLM ROCm GPU runtime was not detected through a managed TheRock "
            "runtime; no CPU fallback is allowed:\n" + json.dumps(detect, indent=2)
        )


def start_vllm(
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


def wait_health(host: str, port: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, body = http_request(host, port, "GET", "/health", None, timeout=3)
            if status < 400:
                return {"status_code": status, "body": body.decode("utf-8", errors="replace")}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"vLLM server did not become healthy: {last_error}")


def get_json(host: str, port: int, path: str, *, timeout: int) -> dict[str, Any]:
    status, body = http_request(host, port, "GET", path, None, timeout=timeout)
    if status >= 400:
        raise RuntimeError(f"GET {path} failed: HTTP {status}: {body!r}")
    return json.loads(body.decode("utf-8"))


def post_json(
    host: str, port: int, path: str, payload: dict[str, Any], *, timeout: int
) -> dict[str, Any]:
    status, body = http_request(
        host,
        port,
        "POST",
        path,
        json.dumps(payload).encode("utf-8"),
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"POST {path} failed: HTTP {status}: {body!r}")
    return json.loads(body.decode("utf-8"))


def http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | None,
    *,
    timeout: int,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        return response.status, response_body
    finally:
        connection.close()


def completion_text(completion: dict[str, Any]) -> str:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("completion response did not include choices")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("completion response did not include text")
    return text


def first_model_id(models: dict[str, Any]) -> str:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("/v1/models did not return a non-empty data list")
    model_id = data[0].get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise RuntimeError("/v1/models first entry did not include an id")
    return model_id


def verify_managed_env(state: dict[str, Any]) -> dict[str, str]:
    pid = state.get("server_pid") or state.get("pid")
    runtime_env = state.get("therock_runtime_env") or {}
    runtime_root = runtime_env.get("root")
    runtime_bin = runtime_env.get("bin")
    if not pid or not runtime_root:
        raise RuntimeError("state is missing server_pid/pid or therock_runtime_env.root")
    environ_path = Path("/proc") / str(int(pid)) / "environ"
    if not environ_path.is_file():
        raise RuntimeError(f"process environ file was not found: {environ_path}")
    entries = environ_path.read_bytes().split(b"\0")
    env: dict[str, str] = {}
    for entry in entries:
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )

    expected = {
        "ROCM_SDK_ROOT": runtime_root,
        "ROCM_PATH": runtime_root,
        "ROCM_HOME": runtime_root,
        "HIP_PATH": runtime_root,
    }
    if runtime_bin:
        expected["ROCM_CLI_THEROCK_SDK_BIN"] = runtime_bin
    for key, value in expected.items():
        if env.get(key) != value:
            raise RuntimeError(
                f"managed TheRock env mismatch for {key}: expected {value!r}, got {env.get(key)!r}"
            )
    if env.get("ROCM_CLI_THEROCK_RUNTIME_ID") != state.get("runtime_id"):
        raise RuntimeError("ROCM_CLI_THEROCK_RUNTIME_ID does not match service state")
    return {key: env[key] for key in expected if key in env}


def verify_loaded_modules(state: dict[str, Any], math_modules: list[str]) -> dict[str, str]:
    pid = state.get("server_pid") or state.get("pid")
    runtime_env = state.get("therock_runtime_env") or {}
    runtime_root = runtime_env.get("root")
    if not pid or not runtime_root:
        raise RuntimeError("state is missing server_pid/pid or therock_runtime_env.root")
    maps_path = Path("/proc") / str(int(pid)) / "maps"
    if not maps_path.is_file():
        raise RuntimeError(f"process maps file was not found: {maps_path}")

    module_paths: dict[str, str] = {}
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "/" not in line:
            continue
        path = line.split(maxsplit=5)[-1]
        name = Path(path).name.lower()
        for module in [HIP_MODULE, *math_modules]:
            if name.startswith(f"{module.lower()}.so"):
                module_paths.setdefault(module, path)
    if HIP_MODULE not in module_paths:
        raise RuntimeError(f"missing loaded HIP module: {HIP_MODULE}")
    math_loaded = sorted(set(math_modules) & set(module_paths))
    if not math_loaded:
        raise RuntimeError(f"missing loaded ROCm math module; expected one of {math_modules}")

    roots = managed_therock_module_roots(Path(runtime_root))
    for module, path in module_paths.items():
        lower = str(Path(path)).lower()
        if not any(lower.startswith(root) for root in roots):
            raise RuntimeError(
                f"{module} did not load from managed TheRock SDK wheel directories: {path}"
            )
    return module_paths


def managed_therock_module_roots(runtime_root: Path) -> set[str]:
    roots = {str(runtime_root).lower(), str(runtime_root.resolve()).lower()}
    parent = runtime_root.parent
    if runtime_root.name.startswith("_rocm_sdk_") and parent.is_dir():
        for sibling in parent.glob("_rocm_sdk_*"):
            if sibling.is_dir():
                roots.add(str(sibling).lower())
                roots.add(str(sibling.resolve()).lower())
    return roots


def stop_service(process: subprocess.Popen[bytes] | None, state_path: Path) -> None:
    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    process_pid = process.pid if process else None
    for pid in [state.get("server_pid"), state.get("pid"), process_pid]:
        if not pid:
            continue
        subprocess.run(
            ["kill", str(int(pid))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(0.5)
    for pid in [state.get("server_pid"), state.get("pid"), process_pid]:
        if not pid:
            continue
        subprocess.run(
            ["kill", "-9", str(int(pid))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
