#!/usr/bin/env python3
"""End-to-end PyTorch GPU smoke test for rocm-cli managed TheRock runtimes."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_REF = "hf-internal-testing/tiny-random-gpt2"
WINDOWS_MODULE_PREFIXES = ("amdhip64", "hipblas", "rocblas", "torch_hip")
LINUX_MODULE_PREFIXES = ("libamdhip64", "libhipblas", "librocblas", "libtorch_hip")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    engine = resolve_path(args.engine, repo_root)

    if not engine.is_file():
        raise SystemExit(
            f"PyTorch engine binary not found: {engine}\n"
            "Build it with `cargo build -p rocm-engine-pytorch`, or pass "
            "`--engine <path-to-rocm-engine-pytorch>`."
        )

    env = os.environ.copy()
    localize_huggingface_cache(env, repo_root)
    print_step("Checking for an AMD GPU and a managed PyTorch folder...")
    runtime_id = resolve_runtime_id(args.runtime_id)
    detect = run_json([str(engine), "detect"], env=env, timeout=args.timeout)
    assert_rocm_gpu_detected(detect)

    if args.env_id:
        env_id = args.env_id.strip()
        manifest = load_engine_manifest(env_id)
        assert_env_matches_runtime(manifest, runtime_id)
    else:
        env_id, manifest = resolve_engine_env_for_runtime(runtime_id)
    assert_managed_env_manifest_ready(manifest)

    process, state_path, log_path = start_serve_http(
        engine=engine,
        model_ref=args.model_ref,
        env_id=env_id,
        runtime_id=runtime_id,
        args=args,
        env=env,
        repo_root=repo_root,
    )
    print_step("Starting the PyTorch test server in AMD GPU mode.")
    print(f"Log file: {log_path}", flush=True)

    try:
        health = wait_health(
            args.host, args.port, args.timeout, process, state_path, log_path
        )
        print_step("PyTorch is running on the AMD GPU. Sending a tiny prompt...")
        models = get_json(args.host, args.port, "/v1/models", timeout=args.timeout)
        completion = post_json(
            args.host,
            args.port,
            "/v1/completions",
            {
                "model": args.model_ref,
                "prompt": "ROCm is",
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=args.timeout,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert_gpu_state(state, health)
        assert_models(models, args.model_ref)
        assert_completion(completion)
        module_paths = (
            {}
            if args.skip_module_check
            else verify_loaded_modules(state, manifest.get("env_path"))
        )

        summary = {
            "ok": True,
            "message": "Success: PyTorch ran on your AMD GPU using TheRock ROCm.",
            "service_id": args.service_id,
            "model_ref": args.model_ref,
            "health": health,
            "completion_text": completion["choices"][0]["text"],
            "env_id": env_id,
            "runtime_id": state.get("runtime_id"),
            "state_path": str(state_path),
            "log_path": str(log_path),
            "worker_pid": state.get("pid"),
            "verified_modules": module_paths,
        }
        print_step("Success: PyTorch ran on your AMD GPU using TheRock ROCm.")
        print(json.dumps(summary, indent=2))
    finally:
        if not args.keep_running:
            stop_service(process, state_path)

    return 0


def print_step(message: str) -> None:
    print(f"[pytorch-gpu-test] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    default_engine = cargo_binary_path("debug", "rocm-engine-pytorch")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=str(default_engine))
    parser.add_argument("--model-ref", default=DEFAULT_MODEL_REF)
    parser.add_argument("--service-id", default="pytorch-gpu-e2e")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11441)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--env-id", help="managed PyTorch env id; defaults to engine detect"
    )
    parser.add_argument(
        "--runtime-id",
        help=(
            "managed TheRock runtime key or unambiguous runtime id; defaults to the "
            "active rocm-cli runtime"
        ),
    )
    parser.add_argument(
        "--skip-module-check",
        action="store_true",
        help="skip OS loaded-module inspection; AMD GPU state is still required",
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


def localize_huggingface_cache(env: dict[str, str], repo_root: Path) -> None:
    cache_root = test_cache_root(repo_root) / "huggingface"
    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))


def assert_rocm_gpu_detected(detect: dict[str, Any]) -> None:
    devices = detect.get("available_devices", [])
    rocm_gpu = next(
        (device for device in devices if device.get("kind") == "rocm_gpu"),
        None,
    )
    if not rocm_gpu or not rocm_gpu.get("available"):
        raise RuntimeError(
            "PyTorch could not see an AMD GPU. No CPU fallback is allowed.\n"
            "Run `rocm doctor`, fix AMD driver/GPU detection there, then retry this test.\n"
            + json.dumps(detect, indent=2)
        )


def assert_managed_env_manifest_ready(manifest: dict[str, Any]) -> None:
    env_path = manifest.get("env_path")
    if not isinstance(env_path, str) or not env_path.strip():
        raise RuntimeError("managed PyTorch env manifest is missing env_path")
    if not Path(env_path).is_dir():
        raise RuntimeError(
            f"managed PyTorch env path from manifest does not exist: {env_path}"
        )


def assert_env_matches_runtime(manifest: dict[str, Any], runtime_id: str) -> None:
    manifest_runtime = manifest.get("runtime_id")
    if not isinstance(manifest_runtime, str) or not manifest_runtime.strip():
        raise RuntimeError("managed PyTorch env manifest is missing runtime_id")
    if manifest_runtime.lower() != runtime_id.lower():
        raise RuntimeError(
            "managed PyTorch env does not belong to the selected ROCm runtime: "
            f"env_id={manifest.get('env_id')}, env_runtime_id={manifest_runtime}, "
            f"selected_runtime_key={runtime_id}"
        )


def start_serve_http(
    *,
    engine: Path,
    model_ref: str,
    env_id: str,
    runtime_id: str | None,
    args: argparse.Namespace,
    env: dict[str, str],
    repo_root: Path,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    data_root = test_data_root(repo_root)
    state_path = data_root / "test-state" / f"{args.service_id}.json"
    log_path = data_root / "test-logs" / f"{args.service_id}.log"
    command = [
        str(engine),
        "serve-http",
        args.service_id,
        model_ref,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device-policy",
        "gpu_required",
        "--env-id",
        env_id,
        "--state-path",
        str(state_path),
    ]
    if runtime_id:
        command.extend(["--runtime-id", runtime_id])

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


def test_cache_root(repo_root: Path) -> Path:
    override = os.environ.get("ROCM_CLI_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "test-cache"
    return repo_root / "target" / "test-cache"


def test_data_root(repo_root: Path) -> Path:
    override = os.environ.get("ROCM_CLI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root / "target"


def wait_health(
    host: str,
    port: int,
    timeout: int,
    process: subprocess.Popen[bytes],
    state_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    last_log_line_count = 0
    last_progress = 0.0
    while time.monotonic() < deadline:
        try:
            health = get_json(host, port, "/healthz", timeout=3)
            if health.get("status") == "ok":
                return health
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"PyTorch worker exited before it became healthy (exit {exit_code}).\n"
                + failure_context(state_path, log_path)
            )
        now = time.monotonic()
        if now - last_progress >= 5:
            last_log_line_count = print_new_log_lines(log_path, last_log_line_count)
            print_state_progress(state_path)
            last_progress = now
        time.sleep(0.5)
    raise RuntimeError(
        f"PyTorch worker did not become healthy: {last_error}\n"
        + failure_context(state_path, log_path)
    )


def print_new_log_lines(log_path: Path, previous_count: int) -> int:
    if not log_path.is_file():
        print_step("Waiting for PyTorch to start. The log file is not ready yet.")
        return previous_count
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= previous_count:
        print_step("Still waiting for PyTorch to load the tiny model...")
        return len(lines)
    for line in lines[previous_count:][-8:]:
        print(f"[pytorch-gpu-test][log] {line}", flush=True)
    return len(lines)


def print_state_progress(state_path: Path) -> None:
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    status = state.get("status")
    if status == "starting":
        print_step(
            "The test server is starting. This can take a little while on first run."
        )
    elif status:
        print_step(f"Current status: {status}")


def failure_context(state_path: Path, log_path: Path) -> str:
    parts = [f"state file: {state_path}", f"log file: {log_path}"]
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            visible_state = {
                key: state.get(key)
                for key in [
                    "status",
                    "error",
                    "device",
                    "device_policy",
                    "runtime_id",
                    "env_id",
                ]
                if key in state
            }
            parts.append("state: " + json.dumps(visible_state, indent=2))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"state could not be read: {exc}")
    if log_path.is_file():
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if log_lines:
            parts.append("last log lines:\n" + "\n".join(log_lines[-40:]))
    return "\n".join(parts)


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


def assert_gpu_state(state: dict[str, Any], health: dict[str, Any]) -> None:
    for label, payload in [("state", state), ("health", health)]:
        if payload.get("device") != "cuda":
            raise RuntimeError(
                f"PyTorch {label} reported {payload.get('device')!r}, not cuda; "
                "no CPU fallback is allowed"
            )
        if payload.get("device_policy") != "gpu_required":
            raise RuntimeError(
                f"PyTorch {label} reported device_policy={payload.get('device_policy')!r}, "
                "expected gpu_required"
            )
        gpu_count = payload.get("gpu_count")
        if not isinstance(gpu_count, int) or gpu_count < 1:
            raise RuntimeError(f"PyTorch {label} did not report a visible GPU")
    if state.get("status") != "ready":
        raise RuntimeError(f"PyTorch state is not ready: {state}")


def assert_models(models: dict[str, Any], model_ref: str) -> None:
    rows = models.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("/v1/models did not return any models")
    ids = {row.get("id") for row in rows if isinstance(row, dict)}
    if model_ref not in ids:
        raise RuntimeError(f"/v1/models did not include {model_ref!r}: {models}")


def assert_completion(completion: dict[str, Any]) -> None:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"/v1/completions returned no choices: {completion}")
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if not isinstance(text, str):
        raise RuntimeError(f"/v1/completions did not return text: {completion}")


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
    config_base = (
        Path(config_dir).expanduser() if config_dir else default_rocm_cli_dir()
    )
    data_base = (
        Path(data_dir).expanduser()
        if data_dir
        else default_rocm_cli_data_dir(config_base)
    )
    return config_base / "config.json", data_base / "runtimes" / "registry"


def default_rocm_cli_dir() -> Path:
    return Path.home() / ".rocm"


def rocm_cli_data_dir() -> Path:
    override = os.environ.get("ROCM_CLI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    config_dir = os.environ.get("ROCM_CLI_CONFIG_DIR")
    config_base = (
        Path(config_dir).expanduser() if config_dir else default_rocm_cli_dir()
    )
    return default_rocm_cli_data_dir(config_base)


def default_rocm_cli_data_dir(config_base: Path) -> Path:
    config_path = config_base / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_rocm_cli_dir()
    setup = config.get("setup")
    if isinstance(setup, dict):
        therock_venv = setup.get("therock_venv")
        if isinstance(therock_venv, str) and therock_venv.strip():
            return Path(therock_venv).expanduser()
    return default_rocm_cli_dir()


def load_engine_manifest(env_id: str) -> dict[str, Any]:
    manifest_path = (
        rocm_cli_data_dir() / "engines" / "pytorch" / "manifests" / f"{env_id}.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(
            f"managed PyTorch env manifest was not found: {manifest_path}"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_engine_env_for_runtime(runtime_id: str) -> tuple[str, dict[str, Any]]:
    manifest_dir = rocm_cli_data_dir() / "engines" / "pytorch" / "manifests"
    if not manifest_dir.is_dir():
        raise RuntimeError(
            "no managed PyTorch environments were found. Install one with "
            f"`rocm engines install pytorch --runtime-id {runtime_id}`"
        )
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for path in manifest_dir.glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        manifest_runtime = manifest.get("runtime_id")
        env_id = manifest.get("env_id")
        if (
            isinstance(manifest_runtime, str)
            and manifest_runtime.lower() == runtime_id.lower()
            and isinstance(env_id, str)
            and env_id.strip()
        ):
            try:
                modified = path.stat().st_mtime_ns
            except OSError:
                modified = 0
            matches.append((modified, env_id.strip(), manifest))
    if not matches:
        raise RuntimeError(
            "no managed PyTorch environment matches the selected ROCm runtime "
            f"`{runtime_id}`. Install one with "
            f"`rocm engines install pytorch --runtime-id {runtime_id}`"
        )
    matches.sort(reverse=True, key=lambda row: (row[0], row[1]))
    _, env_id, manifest = matches[0]
    return env_id, manifest


def run_self_test() -> int:
    scratch_root = (
        Path(__file__).resolve().parents[1] / ".rocm-work" / "script-self-tests"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pytorch-", dir=scratch_root) as temp:
        root = Path(temp)
        config_dir = root / "config"
        data_dir = root / "data"
        registry_dir = data_dir / "runtimes" / "registry"
        manifest_dir = data_dir / "engines" / "pytorch" / "manifests"
        env_root = data_dir / "engines" / "pytorch" / "envs"
        registry_dir.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)
        env_root.mkdir(parents=True)
        config_dir.mkdir(parents=True)

        write_runtime_manifest(
            registry_dir, "runtime-old", "therock-release:gfx120X-all"
        )
        write_runtime_manifest(
            registry_dir, "runtime-new", "therock-release:gfx120X-all"
        )
        write_runtime_manifest(registry_dir, "runtime-other", "therock-release:gfx1151")
        write_engine_manifest(manifest_dir, env_root, "env-old", "runtime-old")
        write_engine_manifest(manifest_dir, env_root, "env-new", "runtime-new")
        write_engine_manifest(manifest_dir, env_root, "env-other", "runtime-other")
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
            env_id, manifest = resolve_engine_env_for_runtime("runtime-old")
            assert env_id == "env-old"
            assert_env_matches_runtime(manifest, "runtime-old")
            assert_managed_env_manifest_ready(manifest)

            assert resolve_runtime_id("runtime-new") == "runtime-new"
            assert resolve_runtime_id("therock-release:gfx1151") == "runtime-other"
            try:
                assert_env_matches_runtime(
                    load_engine_manifest("env-new"), "runtime-old"
                )
            except RuntimeError as exc:
                assert "selected ROCm runtime" in str(exc)
            else:
                raise AssertionError("mismatched env/runtime pair did not fail")

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
    print("PyTorch GPU script self-test passed")
    return 0


def write_runtime_manifest(
    registry_dir: Path, runtime_key: str, runtime_id: str
) -> None:
    (registry_dir / f"{runtime_key}.json").write_text(
        json.dumps({"runtime_key": runtime_key, "runtime_id": runtime_id}),
        encoding="utf-8",
    )


def write_engine_manifest(
    manifest_dir: Path,
    env_root: Path,
    env_id: str,
    runtime_id: str,
) -> None:
    env_path = env_root / env_id
    env_path.mkdir(parents=True)
    payload = {
        "env_id": env_id,
        "runtime_id": runtime_id,
        "env_path": str(env_path),
    }
    (manifest_dir / f"{env_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def write_config(config_dir: Path, payload: dict[str, str]) -> None:
    (config_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def restore_env(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def verify_loaded_modules(state: dict[str, Any], env_path: Any) -> dict[str, str]:
    if not isinstance(env_path, str) or not env_path.strip():
        raise RuntimeError("managed PyTorch env manifest is missing env_path")
    if platform.system() == "Windows":
        return verify_windows_modules(state, Path(env_path))
    return verify_proc_maps(state, Path(env_path))


def verify_windows_modules(state: dict[str, Any], env_path: Path) -> dict[str, str]:
    worker_pid = state.get("pid")
    if not worker_pid:
        raise RuntimeError("state is missing the PyTorch worker pid")
    command = (
        f"Get-Process -Id {int(worker_pid)} -Module | "
        "Select-Object ModuleName,FileName | ConvertTo-Json -Depth 4"
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
        raise RuntimeError("module inspection returned no loaded modules")
    parsed = json.loads(completed.stdout)
    rows = parsed if isinstance(parsed, list) else [parsed]
    modules: dict[str, str] = {}
    for row in rows:
        name = str(row.get("ModuleName", "")).lower()
        filename = str(row.get("FileName", ""))
        if any(name.startswith(prefix) for prefix in WINDOWS_MODULE_PREFIXES):
            modules[name] = filename
    ensure_expected_modules(modules, WINDOWS_MODULE_PREFIXES, env_path)
    return modules


def verify_proc_maps(state: dict[str, Any], env_path: Path) -> dict[str, str]:
    worker_pid = state.get("pid")
    if not worker_pid:
        raise RuntimeError("state is missing the PyTorch worker pid")
    maps_path = Path("/proc") / str(int(worker_pid)) / "maps"
    if not maps_path.is_file():
        raise RuntimeError(f"process maps file was not found: {maps_path}")
    modules: dict[str, str] = {}
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "/" not in line:
            continue
        path = line.split(maxsplit=5)[-1]
        if not path.startswith("/"):
            continue
        name = Path(path).name.lower()
        if any(name.startswith(prefix) for prefix in LINUX_MODULE_PREFIXES):
            modules.setdefault(name, path)
    ensure_expected_modules(modules, LINUX_MODULE_PREFIXES, env_path)
    return modules


def ensure_expected_modules(
    modules: dict[str, str],
    prefixes: tuple[str, ...],
    env_path: Path,
) -> None:
    if not modules:
        raise RuntimeError("no loaded PyTorch HIP modules were found")
    if not any(name.startswith(prefixes[0]) for name in modules):
        raise RuntimeError(f"missing loaded HIP runtime module matching {prefixes[0]}")

    env_prefix = str(env_path.resolve()).lower()
    for name, filename in modules.items():
        lower_path = str(Path(filename).resolve()).lower()
        if platform.system() == "Windows" and "\\system32\\" in lower_path:
            raise RuntimeError(f"{name} loaded from System32: {filename}")
        if not lower_path.startswith(env_prefix):
            raise RuntimeError(
                f"{name} did not load from the managed PyTorch env: {filename}"
            )


def stop_service(process: subprocess.Popen[bytes], state_path: Path) -> None:
    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    for pid in [state.get("pid"), process.pid]:
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
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
