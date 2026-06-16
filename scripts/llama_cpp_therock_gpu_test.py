#!/usr/bin/env python3
"""End-to-end llama.cpp GPU smoke test for rocm-cli managed TheRock runtimes."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import platform
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL_URL = (
    "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories260K.gguf"
)
MODULES_TO_VERIFY = {
    "amdhip64_7.dll",
    "ggml-hip.dll",
    "hipblas.dll",
    "libhipblaslt.dll",
    "rocblas.dll",
}
LINUX_MODULES_TO_VERIFY = {
    "libamdhip64": "libamdhip64",
    "libhipblas": "libhipblas",
    "libhipblaslt": "libhipblaslt",
    "librocblas": "librocblas",
}


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    engine = resolve_path(args.engine, repo_root)
    llama_server = resolve_path(args.llama_server, repo_root)
    model_path = resolve_path(args.model_path, repo_root)
    runtime_id = resolve_runtime_id(args.runtime_id)

    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    if not llama_server.is_file():
        raise SystemExit(
            f"HIP llama-server not found: {llama_server}; no CPU fallback is allowed"
        )
    ensure_model(model_path, args.model_url)

    env = os.environ.copy()
    env["ROCM_CLI_LLAMA_CPP_SERVER"] = str(llama_server)

    detect = run_json([str(engine), "detect"], env=env, timeout=args.timeout)
    assert_rocm_gpu_detected(detect)

    process, state_path, log_path = start_serve_http(
        engine=engine,
        model_path=model_path,
        runtime_id=runtime_id,
        args=args,
        env=env,
        repo_root=repo_root,
    )

    try:
        health = wait_health(args.host, args.port, args.timeout)
        completion = post_json(
            args.host,
            args.port,
            "/v1/completions",
            {
                "model": model_path.name,
                "prompt": "Once upon a time",
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=args.timeout,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert_gpu_log(log_text)
        module_paths = verify_loaded_modules(state)

        summary = {
            "ok": True,
            "launch_mode": args.launch_mode,
            "service_id": args.service_id,
            "health": health,
            "completion_text": completion["choices"][0]["text"],
            "runtime_id": runtime_id,
            "state_path": str(state_path),
            "log_path": str(log_path),
            "launcher_pid": process.pid if process else state.get("pid"),
            "staged_runtime_dir": state.get("staged_runtime_dir"),
            "verified_modules": module_paths,
        }
        print(json.dumps(summary, indent=2))
    finally:
        if not args.keep_running:
            stop_service(process, state_path)

    return 0


def parse_args() -> argparse.Namespace:
    default_engine = cargo_binary_path("debug", "rocm-engine-llama-cpp")
    default_server = (
        Path("target/llama.cpp-build-hip/bin/llama-server.exe")
        if platform.system() == "Windows"
        else Path("target/llama.cpp-build-hip/bin/llama-server")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=str(default_engine))
    parser.add_argument(
        "--llama-server",
        default=os.environ.get("ROCM_CLI_LLAMA_CPP_SERVER", str(default_server)),
    )
    parser.add_argument("--model-path", default="target/models/stories260K.gguf")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument(
        "--runtime-id",
        help=(
            "managed TheRock runtime key or unambiguous runtime id; defaults to the "
            "active rocm-cli runtime"
        ),
    )
    parser.add_argument("--service-id", default="llama-gpu-e2e")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11439)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--launch-mode",
        choices=("serve-http", "launch"),
        default="serve-http",
        help="serve-http runs the wrapper directly; launch verifies captured-output background launch",
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


def cargo_binary_path(profile: str, name: str) -> Path:
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", "target")).expanduser()
    return target_root / profile / exe_name(name)


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


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
        config_base = Path(config_dir) if config_dir else default_rocm_cli_config_dir()
        data_base = Path(data_dir) if data_dir else default_rocm_cli_data_dir()
        return (
            config_base / "config.json",
            data_base / "runtimes" / "registry",
        )
    return default_rocm_cli_config_dir() / "config.json", (
        default_rocm_cli_data_dir() / "runtimes" / "registry"
    )


def default_rocm_cli_config_dir() -> Path:
    return Path.home() / ".rocm"


def default_rocm_cli_data_dir() -> Path:
    return Path.home() / ".rocm"


def run_self_test() -> int:
    scratch_root = (
        Path(__file__).resolve().parents[1] / ".rocm-work" / "script-self-tests"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-cpp-", dir=scratch_root) as temp:
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
        }
        try:
            os.environ["ROCM_CLI_CONFIG_DIR"] = str(config_dir)
            os.environ["ROCM_CLI_DATA_DIR"] = str(data_dir)
            assert resolve_runtime_id(None) == "runtime-old"
            assert resolve_runtime_id("runtime-new") == "runtime-new"
            assert resolve_runtime_id("therock-release:gfx1151") == "runtime-other"

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
    print("llama.cpp GPU script self-test passed")
    return 0


def write_runtime_manifest(
    registry_dir: Path, runtime_key: str, runtime_id: str
) -> None:
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


def ensure_model(path: Path, url: str) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    if path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty GGUF model: {path}")


def run_json(
    command: list[str], *, env: dict[str, str], timeout: int
) -> dict[str, Any]:
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


def start_serve_http(
    *,
    engine: Path,
    model_path: Path,
    runtime_id: str,
    args: argparse.Namespace,
    env: dict[str, str],
    repo_root: Path,
) -> tuple[subprocess.Popen[bytes] | None, Path, Path]:
    if args.launch_mode == "launch":
        return start_launch(engine, model_path, runtime_id, args, env=env)

    data_root = test_data_root(repo_root)
    state_path = data_root / "test-state" / f"{args.service_id}.json"
    log_path = data_root / "test-logs" / f"{args.service_id}.log"
    command = [
        str(engine),
        "serve-http",
        args.service_id,
        str(model_path),
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


def test_data_root(repo_root: Path) -> Path:
    override = os.environ.get("ROCM_CLI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root / "target"


def start_launch(
    engine: Path,
    model_path: Path,
    runtime_id: str,
    args: argparse.Namespace,
    *,
    env: dict[str, str],
) -> tuple[None, Path, Path]:
    command = [
        str(engine),
        "launch",
        args.service_id,
        str(model_path),
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
            f"launch took {elapsed:.1f}s; captured-output background launch should return promptly"
        )
    state_path = Path(response["state_path"])
    log_path = Path(response["log_path"])
    return None, state_path, log_path


def assert_rocm_gpu_detected(detect: dict[str, Any]) -> None:
    devices = detect.get("available_devices", [])
    rocm_gpu = next((d for d in devices if d.get("kind") == "rocm_gpu"), None)
    if not rocm_gpu or not rocm_gpu.get("available"):
        raise RuntimeError(
            "llama.cpp ROCm GPU was not detected; no CPU fallback is allowed:\n"
            + json.dumps(detect, indent=2)
        )


def wait_health(host: str, port: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = get_json(host, port, "/health", timeout=3)
            if health.get("status") == "ok":
                return health
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def get_json(host: str, port: int, path: str, *, timeout: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status >= 400:
        raise RuntimeError(f"GET {path} failed: HTTP {response.status}: {body!r}")
    return json.loads(body.decode("utf-8"))


def post_json(
    host: str, port: int, path: str, payload: dict[str, Any], *, timeout: int
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    try:
        connection.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        response_body = response.read()
    finally:
        connection.close()
    if response.status >= 400:
        raise RuntimeError(
            f"POST {path} failed: HTTP {response.status}: {response_body!r}"
        )
    return json.loads(response_body.decode("utf-8"))


def assert_gpu_log(log_text: str) -> None:
    if "ROCm0" not in log_text:
        raise RuntimeError("llama.cpp log did not show ROCm0 GPU usage")
    if "C:\\WINDOWS\\SYSTEM32\\amdhip64_7.dll".lower() in log_text.lower():
        raise RuntimeError("llama.cpp loaded amdhip64_7.dll from System32")


def verify_loaded_modules(state: dict[str, Any]) -> dict[str, str]:
    if platform.system() == "Windows":
        return verify_windows_modules(state)
    return verify_proc_maps(state)


def verify_windows_modules(state: dict[str, Any]) -> dict[str, str]:
    server_pid = state.get("server_pid")
    staged_dir = state.get("staged_runtime_dir")
    if not server_pid or not staged_dir:
        raise RuntimeError("state is missing server_pid or staged_runtime_dir")
    command = (
        "$names=@("
        + ",".join(f"'{name}'" for name in sorted(MODULES_TO_VERIFY))
        + "); "
        + f"Get-Process -Id {int(server_pid)} -Module | "
        + "Where-Object { $names -contains $_.ModuleName } | "
        + "Select-Object ModuleName,FileName | ConvertTo-Json -Depth 4"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"module inspection failed:\n{completed.stderr}")
    if not completed.stdout.strip():
        raise RuntimeError("module inspection returned no HIP modules")
    parsed = json.loads(completed.stdout)
    rows = parsed if isinstance(parsed, list) else [parsed]
    modules = {row["ModuleName"].lower(): row["FileName"] for row in rows}
    missing = sorted(MODULES_TO_VERIFY - set(modules))
    if missing:
        raise RuntimeError(f"missing loaded HIP modules: {missing}")
    staged_prefix = str(Path(staged_dir)).lower()
    for name, filename in modules.items():
        lower = filename.lower()
        if "\\system32\\" in lower:
            raise RuntimeError(f"{name} loaded from System32: {filename}")
        if not lower.startswith(staged_prefix):
            raise RuntimeError(f"{name} did not load from staged runtime: {filename}")
    return modules


def verify_proc_maps(state: dict[str, Any]) -> dict[str, str]:
    server_pid = state.get("server_pid")
    runtime_env = state.get("therock_runtime_env") or {}
    runtime_root = runtime_env.get("root")
    if not server_pid or not runtime_root:
        raise RuntimeError("state is missing server_pid or therock_runtime_env.root")
    maps_path = Path("/proc") / str(int(server_pid)) / "maps"
    if not maps_path.is_file():
        raise RuntimeError(f"process maps file was not found: {maps_path}")
    module_paths: dict[str, str] = {}
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "/" not in line:
            continue
        path = line.split(maxsplit=5)[-1]
        name = Path(path).name.lower()
        for label, needle in LINUX_MODULES_TO_VERIFY.items():
            if name.startswith(f"{needle}.so"):
                module_paths.setdefault(label, path)
    missing = sorted(set(LINUX_MODULES_TO_VERIFY) - set(module_paths))
    if missing:
        raise RuntimeError(f"missing loaded HIP/BLAS modules: {missing}")
    roots = managed_therock_module_roots(runtime_env)
    for name, path in module_paths.items():
        lower = str(Path(path)).lower()
        if not any(lower.startswith(root) for root in roots):
            raise RuntimeError(
                f"{name} did not load from managed TheRock SDK wheel directories: {path}"
            )
    return module_paths


def managed_therock_module_roots(runtime_env: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for key in ["root", "bin"]:
        add_module_root(roots, runtime_env.get(key))
    for key in ["bin_paths", "library_paths"]:
        values = runtime_env.get(key)
        if isinstance(values, list):
            for value in values:
                add_module_root(roots, value)

    runtime_root = runtime_env.get("root")
    if isinstance(runtime_root, str) and runtime_root.strip():
        root_path = Path(runtime_root)
        parent = root_path.parent
        if root_path.name.startswith("_rocm_sdk_") and parent.is_dir():
            for sibling in parent.glob("_rocm_sdk_*"):
                if sibling.is_dir():
                    add_module_root(roots, sibling)
    return roots


def add_module_root(roots: set[str], value: Any) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return
    path = Path(value)
    roots.add(str(path).lower())
    with contextlib.suppress(OSError):
        roots.add(str(path.resolve()).lower())


def stop_service(process: subprocess.Popen[bytes] | None, state_path: Path) -> None:
    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    process_pid = process.pid if process else None
    for pid in [state.get("server_pid"), state.get("pid"), process_pid]:
        if not pid:
            continue
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            subprocess.run(["kill", str(int(pid))], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
