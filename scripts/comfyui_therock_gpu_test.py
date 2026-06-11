#!/usr/bin/env python3
"""End-to-end ComfyUI GPU smoke test for rocm-cli managed TheRock runtimes.

This opt-in test installs or reuses the rocm-cli managed ComfyUI app, starts it
with the active managed TheRock runtime, verifies the local ComfyUI HTTP
endpoint, checks rocm-cli status/log output, then stops the process it started
unless --keep-running is set. With --generate-cat, it also places a checkpoint
in ComfyUI's model folder when needed and submits a real cat image workflow
through ComfyUI's HTTP API. It never uses CPU fallback.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import shutil
import tempfile
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18188
DEFAULT_CAT_PROMPT = "a cute orange tabby cat sitting on a sunny windowsill"
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, extra limbs, text, watermark"
DEFAULT_CHECKPOINT_NAME = "sd-v1-5-tiny.safetensors"
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/ehristoforu/stable-diffusion-v1-5-tiny/resolve/main/"
    + DEFAULT_CHECKPOINT_NAME
)
DEFAULT_CHECKPOINT_SIZE_BYTES = 2_132_625_894
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_STEPS = 18
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "normal"


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
    started_pid: int | None = None
    started_service = False
    try:
        if args.temp_state:
            temp_state_root = Path(tempfile.mkdtemp(prefix="rocm-cli-comfyui-state-"))
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

        comfyui_root: Path | None = None
        if not args.skip_install:
            install_command = build_install_command(args, rocm)
            print_step("Installing ComfyUI with the selected managed ROCm runtime.")
            install_output = run_text(install_command, env=env, timeout=args.timeout)
            print(install_output, end="" if install_output.endswith("\n") else "\n")
            assert_install_output(install_output)
            comfyui_root = parse_comfyui_root(install_output)

        if args.generate_cat:
            if args.comfyui_root:
                comfyui_root = resolve_path(args.comfyui_root, repo_root)
            if comfyui_root is None:
                status_for_folder = run_text(
                    build_rocm_command(args, rocm, "comfyui", "status"),
                    env=env,
                    timeout=args.timeout,
                )
                comfyui_root = parse_comfyui_root(status_for_folder)
            if comfyui_root is None:
                raise RuntimeError(
                    "could not find the ComfyUI folder from `rocm comfyui status`; "
                    "install ComfyUI first or pass --comfyui-root"
                )
            checkpoint_path = ensure_checkpoint(args, comfyui_root)
            print_step(f"Using checkpoint {checkpoint_path}.")
        elif args.comfyui_root:
            comfyui_root = resolve_path(args.comfyui_root, repo_root)

        start_command = build_start_command(args, rocm)
        print_step("Starting ComfyUI in GPU-required mode.")
        start_output = run_text(start_command, env=env, timeout=args.timeout)
        print(start_output, end="" if start_output.endswith("\n") else "\n")
        started_service = True
        assert_start_output(start_output, args.host, args.port)
        started_pid = parse_pid(start_output)

        system_stats = wait_comfyui_endpoint(args.host, args.port, args.timeout)
        assert_comfyui_reports_gpu(system_stats)

        status_output = run_text(
            build_rocm_command(args, rocm, "comfyui", "status"),
            env=env,
            timeout=args.timeout,
        )
        print(status_output, end="" if status_output.endswith("\n") else "\n")
        assert_status_output(status_output, args.host, args.port)

        logs_output = run_text(
            build_rocm_command(args, rocm, "comfyui", "logs", "--lines", "80"),
            env=env,
            timeout=args.timeout,
        )
        print(logs_output, end="" if logs_output.endswith("\n") else "\n")
        assert_logs_output(logs_output)

        generated_image: Path | None = None
        if args.generate_cat:
            print_step("Generating a cat image through the ComfyUI HTTP API.")
            if comfyui_root is None:
                comfyui_root = parse_comfyui_root(status_output)
            checkpoint_name = args.checkpoint_name or DEFAULT_CHECKPOINT_NAME
            generated_image = generate_cat_image(args, checkpoint_name, repo_root)
            print_step(f"Generated image: {generated_image}")

        print_step("Success: ComfyUI is reachable through ROCm CLI with AMD GPU checks.")
        print(json.dumps(
            {
                "ok": True,
                "url": f"http://{args.host}:{args.port}",
                "pid": started_pid,
                "installed": not args.skip_install,
                "generated_image": str(generated_image) if generated_image else None,
            },
            indent=2,
        )
        )
    finally:
        if started_service and not args.keep_running:
            print_step("Stopping ComfyUI through ROCm CLI.")
            try:
                stop_output = run_text(
                    build_rocm_command(args, rocm, "comfyui", "stop"),
                    env=env,
                    timeout=args.timeout,
                )
                print(stop_output, end="" if stop_output.endswith("\n") else "\n")
            except Exception as error:
                if started_pid:
                    print_step(f"ROCm CLI stop failed; stopping process {started_pid}.")
                    stop_pid(started_pid)
                else:
                    print_step(f"ROCm CLI stop failed: {error}")
        if temp_state_root is not None and not args.keep_state:
            print_step(f"Removing temporary ROCm CLI state under {temp_state_root}.")
            shutil.rmtree(temp_state_root, ignore_errors=True)

    return 0


def parse_args() -> argparse.Namespace:
    default_rocm = cargo_binary_path("debug", "rocm")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", default=str(default_rocm))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument(
        "--rocm-launch-prefix",
        nargs="+",
        default=[],
        help="prefix used to launch rocm, for example `sh` for WSL APE validation",
    )
    parser.add_argument("--runtime-id", help="exact managed TheRock runtime key")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument(
        "--state-root",
        help=(
            "use this root for ROCM_CLI_CONFIG_DIR/DATA_DIR/CACHE_DIR so a live "
            "test does not touch the default user state"
        ),
    )
    parser.add_argument(
        "--temp-state",
        action="store_true",
        help=(
            "create a temporary ROCm CLI state root for this run; this is safest "
            "for isolated tests, but it requires the selected runtime to exist in "
            "that temporary state or be installed before this script runs"
        ),
    )
    parser.add_argument(
        "--keep-state",
        action="store_true",
        help="keep the temporary state root created by --temp-state",
    )
    parser.add_argument(
        "--copy-runtime-state-from",
        help=(
            "copy only config.json and runtimes/ from an existing ROCm CLI state "
            "root into the isolated state before the test starts; useful with "
            "--temp-state so ComfyUI installs are isolated but can reuse an "
            "already installed managed TheRock runtime"
        ),
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="reuse an existing rocm-cli managed ComfyUI install",
    )
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--generate-cat",
        action="store_true",
        help="after launch, run a real text-to-image cat workflow through ComfyUI",
    )
    parser.add_argument(
        "--comfyui-root",
        help=(
            "ComfyUI source folder; normally discovered from `rocm comfyui status` "
            "and only needed if status output is unavailable"
        ),
    )
    parser.add_argument("--cat-prompt", default=DEFAULT_CAT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument(
        "--checkpoint-size-bytes",
        type=int,
        default=DEFAULT_CHECKPOINT_SIZE_BYTES,
        help="expected checkpoint size; set to 0 to skip size validation",
    )
    parser.add_argument(
        "--no-download-checkpoint",
        action="store_true",
        help="fail if the requested checkpoint is not already present",
    )
    parser.add_argument("--output-dir", help="folder for the downloaded generated image")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--sampler", default=DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
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
    print(f"[comfyui-gpu-test] {message}", flush=True)


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


def build_install_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    command = build_rocm_command(args, rocm, "comfyui", "install")
    if args.runtime_id:
        command.extend(["--runtime-id", args.runtime_id])
    if args.reinstall:
        command.append("--reinstall")
    return command


def build_start_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    return build_rocm_command(
        args,
        rocm,
        "comfyui",
        "start",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--no-open-browser",
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


def http_json(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body: bytes | None = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
    finally:
        conn.close()

    text = raw.decode("utf-8", errors="replace")
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"{method} {path} returned HTTP {response.status}: {text}")
    if not text.strip():
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} returned non-object JSON: {text}")
    return parsed


def http_bytes(host: str, port: int, path: str, *, timeout: int = 60) -> bytes:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read()
    finally:
        conn.close()

    if response.status < 200 or response.status >= 300:
        text = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {path} returned HTTP {response.status}: {text}")
    return raw


def format_command(command: list[str]) -> str:
    return " ".join(command)


def parse_pid(output: str) -> int | None:
    match = re.search(r"(?m)^\s*pid:\s*(\d+)\s*$", output)
    if not match:
        return None
    pid = int(match.group(1))
    return pid if pid > 0 else None


def parse_comfyui_root(output: str) -> Path | None:
    match = re.search(r"(?m)^\s*folder:\s*(.+?)\s*$", output)
    if not match:
        return None
    value = match.group(1).strip()
    if not value:
        return None
    return Path(value)


def wait_comfyui_endpoint(host: str, port: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "endpoint was not checked"
    while time.monotonic() < deadline:
        try:
            stats = http_json(host, port, "GET", "/system_stats", timeout=5)
            if "system" in stats or "devices" in stats:
                return stats
            last_error = "GET /system_stats returned unexpected JSON"
        except OSError as error:
            last_error = str(error)
        except RuntimeError as error:
            last_error = str(error)
        time.sleep(2)
    raise RuntimeError(
        f"ComfyUI endpoint did not become reachable before timeout: {last_error}"
    )


def assert_comfyui_reports_gpu(stats: dict[str, Any]) -> None:
    devices = stats.get("devices")
    if not isinstance(devices, list) or not devices:
        raise RuntimeError(f"ComfyUI did not report any devices:\n{json.dumps(stats, indent=2)}")

    gpu_devices = []
    cpu_devices = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        label = " ".join(
            str(device.get(key, "")) for key in ("name", "type", "device")
        ).lower()
        if "cpu" in label:
            cpu_devices.append(device)
        if any(needle in label for needle in ("amd", "hip", "cuda", "rocm")):
            gpu_devices.append(device)

    if not gpu_devices:
        raise RuntimeError(
            "ComfyUI did not report an AMD/ROCm GPU device; CPU fallback is not allowed:\n"
            f"{json.dumps(devices, indent=2)}"
        )
    if len(cpu_devices) == len(devices):
        raise RuntimeError(
            "ComfyUI reported only CPU devices; CPU fallback is not allowed:\n"
            f"{json.dumps(devices, indent=2)}"
        )


def ensure_checkpoint(
    args: argparse.Namespace,
    comfyui_root: Path,
) -> Path:
    checkpoint_dir = comfyui_root / "models" / "checkpoints"
    checkpoint_name = args.checkpoint_name or DEFAULT_CHECKPOINT_NAME
    checkpoint_path = checkpoint_dir / checkpoint_name
    if checkpoint_path.is_file():
        validate_checkpoint_size(
            checkpoint_path,
            args.checkpoint_size_bytes,
            label="existing checkpoint",
        )
        return checkpoint_path

    if args.no_download_checkpoint:
        raise RuntimeError(
            f"checkpoint is missing and downloads are disabled: {checkpoint_path}\n"
            f"Download URL: {args.checkpoint_url}"
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print_step(
        "Downloading checkpoint for the cat workflow "
        f"({format_bytes(args.checkpoint_size_bytes)} expected)."
    )
    download_file(args.checkpoint_url, checkpoint_path, args.checkpoint_size_bytes)
    validate_checkpoint_size(
        checkpoint_path,
        args.checkpoint_size_bytes,
        label="downloaded checkpoint",
    )
    return checkpoint_path


def validate_checkpoint_size(path: Path, expected_size: int, *, label: str) -> None:
    if expected_size <= 0:
        return
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch for {path}: expected "
            f"{expected_size} bytes, got {actual_size} bytes"
        )


def download_file(url: str, destination: Path, expected_size: int) -> None:
    part = destination.with_name(destination.name + ".part")
    if part.exists():
        part.unlink()

    request = Request(url, headers={"User-Agent": "rocm-cli-comfyui-gpu-test"})
    downloaded = 0
    last_report = time.monotonic()
    with urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else expected_size
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    print_step(
                        "Checkpoint download "
                        f"{format_bytes(downloaded)}"
                        + (
                            f" / {format_bytes(total_size)}"
                            if total_size > 0
                            else ""
                        )
                    )
                    last_report = now
    part.replace(destination)


def format_bytes(size: int) -> str:
    if size <= 0:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def generate_cat_image(
    args: argparse.Namespace,
    checkpoint_name: str,
    repo_root: Path,
) -> Path:
    assert_checkpoint_available(args.host, args.port, checkpoint_name)
    workflow = build_cat_workflow(
        checkpoint_name=checkpoint_name,
        prompt=args.cat_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
    )
    prompt_id = queue_prompt(args.host, args.port, workflow)
    history = poll_prompt_history(
        lambda: fetch_prompt_history(args.host, args.port, prompt_id),
        prompt_id,
        timeout=args.timeout,
        interval=2,
    )
    images = extract_output_images(history)
    if not images:
        raise RuntimeError(
            f"ComfyUI finished prompt {prompt_id}, but no output images were reported:\n"
            f"{json.dumps(history, indent=2)}"
        )
    return download_generated_image(args, images[0], repo_root)


def build_cat_workflow(
    *,
    checkpoint_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
) -> dict[str, Any]:
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("ComfyUI latent image width and height must be divisible by 8")
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "rocm_cli_cat", "images": ["8", 0]},
        },
    }


def assert_checkpoint_available(host: str, port: int, checkpoint_name: str) -> None:
    info = http_json(
        host,
        port,
        "GET",
        "/object_info/CheckpointLoaderSimple",
        timeout=30,
    )
    choices = checkpoint_choices_from_object_info(info)
    if checkpoint_name not in choices:
        raise RuntimeError(
            f"ComfyUI does not list checkpoint `{checkpoint_name}`. "
            "If it was just downloaded, restart ComfyUI and run this test again.\n"
            f"Available checkpoints: {choices}"
        )


def checkpoint_choices_from_object_info(info: dict[str, Any]) -> list[str]:
    node = info.get("CheckpointLoaderSimple")
    if not isinstance(node, dict):
        return []
    input_info = node.get("input")
    if not isinstance(input_info, dict):
        return []
    required = input_info.get("required")
    if not isinstance(required, dict):
        return []
    ckpt_info = required.get("ckpt_name")
    if (
        isinstance(ckpt_info, list)
        and ckpt_info
        and isinstance(ckpt_info[0], list)
    ):
        return [str(value) for value in ckpt_info[0]]
    return []


def queue_prompt(host: str, port: int, workflow: dict[str, Any]) -> str:
    response = http_json(
        host,
        port,
        "POST",
        "/prompt",
        payload={"prompt": workflow, "client_id": "rocm-cli-comfyui-gpu-test"},
        timeout=30,
    )
    node_errors = response.get("node_errors")
    if node_errors:
        raise RuntimeError(f"ComfyUI rejected the workflow:\n{json.dumps(response, indent=2)}")
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt did not return a prompt_id:\n{response}")
    return prompt_id


def fetch_prompt_history(host: str, port: int, prompt_id: str) -> dict[str, Any]:
    history = http_json(host, port, "GET", f"/history/{prompt_id}", timeout=30)
    entry = history.get(prompt_id)
    if isinstance(entry, dict):
        return entry
    return {}


def poll_prompt_history(
    fetch_history: Callable[[], dict[str, Any]],
    prompt_id: str,
    *,
    timeout: int,
    interval: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    deadline = monotonic() + timeout
    last_entry: dict[str, Any] = {}
    while monotonic() < deadline:
        entry = fetch_history()
        if entry:
            last_entry = entry
        status = entry.get("status") if isinstance(entry, dict) else None
        outputs = entry.get("outputs") if isinstance(entry, dict) else None
        if isinstance(outputs, dict) and outputs:
            if prompt_failed(status):
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} failed:\n{json.dumps(entry, indent=2)}"
                )
            return entry
        if prompt_failed(status):
            raise RuntimeError(
                f"ComfyUI prompt {prompt_id} failed:\n{json.dumps(entry, indent=2)}"
            )
        sleep(interval)
    raise RuntimeError(
        f"ComfyUI prompt {prompt_id} did not finish before timeout. "
        f"Last history entry:\n{json.dumps(last_entry, indent=2)}"
    )


def prompt_failed(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    status_text = str(status.get("status_str", "")).lower()
    completed = status.get("completed")
    return status_text in {"error", "failed"} or completed is False and "error" in status_text


def extract_output_images(history_entry: dict[str, Any]) -> list[dict[str, str]]:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    images: list[dict[str, str]] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        raw_images = output.get("images")
        if not isinstance(raw_images, list):
            continue
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            images.append(
                {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder", "")),
                    "type": str(image.get("type", "output")),
                }
            )
    return images


def download_generated_image(
    args: argparse.Namespace,
    image: dict[str, str],
    repo_root: Path,
) -> Path:
    output_dir = (
        resolve_path(args.output_dir, repo_root)
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="rocm-cli-comfyui-cat-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    query = urlencode(
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    data = http_bytes(args.host, args.port, f"/view?{query}", timeout=60)
    destination = output_dir / Path(image["filename"]).name
    destination.write_bytes(data)
    if not data.startswith(b"\x89PNG") and not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"downloaded output does not look like an image: {destination}")
    return destination


def assert_install_output(output: str) -> None:
    require_any_contains(
        output,
        ["AMD GPU check: ready", "AMD GPU: ready"],
        "install output",
    )
    require_contains(output, "next step: rocm comfyui start", "install output")
    reject_cpu_fallback(output, "install output")


def assert_start_output(output: str, host: str, port: int) -> None:
    if "status: running" not in output:
        require_contains(output, "status: starting", "start output")
    require_contains(output, "AMD GPU check: ready", "start output")
    require_any_contains(
        output,
        [f"url: http://{host}:{port}", f"URL: http://{host}:{port}"],
        "start output",
    )
    require_contains(output, "browser: not opened", "start output")
    reject_cpu_fallback(output, "start output")


def assert_status_output(output: str, host: str, port: int) -> None:
    require_contains(output, "status: running", "status output")
    require_contains(output, f"url: http://{host}:{port}", "status output")
    if "status: stopped" in output:
        raise RuntimeError(f"ComfyUI status reports stopped:\n{output}")
    reject_cpu_fallback(output, "status output")


def assert_logs_output(output: str) -> None:
    require_contains(output, "ComfyUI logs", "logs output")
    if "Run log" not in output and "Install log" not in output:
        raise RuntimeError(f"ComfyUI logs did not include saved app logs:\n{output}")
    reject_cpu_fallback(output, "logs output")


def require_contains(output: str, needle: str, label: str) -> None:
    if needle not in output:
        raise RuntimeError(f"{label} did not contain `{needle}`:\n{output}")


def require_any_contains(output: str, needles: list[str], label: str) -> None:
    if not any(needle in output for needle in needles):
        joined = "`, `".join(needles)
        raise RuntimeError(f"{label} did not contain one of `{joined}`:\n{output}")


def reject_cpu_fallback(output: str, label: str) -> None:
    forbidden = [
        "CPU fallback",
        "cpu fallback",
        "falling back to CPU",
        "Running on CPU",
        "cpu_only",
    ]
    for needle in forbidden:
        if needle in output:
            raise RuntimeError(f"{label} contained forbidden `{needle}`:\n{output}")


def stop_pid(pid: int) -> None:
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        subprocess.run(["kill", str(pid)], check=False)


def run_self_test() -> int:
    fake_rocm = Path("target/debug") / exe_name("rocm")
    args = argparse.Namespace(
        host="127.0.0.1",
        port=18188,
        rocm_launch_prefix=["sh"],
        runtime_id="release-pip-gfx120x-all-7-14-0a20260601",
        reinstall=True,
    )
    install = build_install_command(args, fake_rocm)
    assert install[:4] == ["sh", str(fake_rocm), "comfyui", "install"]
    assert "--runtime-id" in install
    assert "--reinstall" in install
    assert "cpu" not in " ".join(install).lower()

    start = build_start_command(args, fake_rocm)
    assert start[:4] == ["sh", str(fake_rocm), "comfyui", "start"]
    assert "--no-open-browser" in start
    assert "--port" in start and "18188" in start

    assert parse_pid("ComfyUI\n  pid: 12345\n") == 12345
    assert parse_pid("ComfyUI\n  pid: 0\n") is None

    assert_install_output(
        "ComfyUI\n  installed: yes\n  AMD GPU check: ready (1 device)\n"
        "  next step: rocm comfyui start\n"
    )
    assert_install_output(
        "ComfyUI\n  installed: yes\n  AMD GPU: ready (1 device)\n"
        "  next step: rocm comfyui start\n"
    )
    assert_start_output(
        "ComfyUI\n  status: starting\n  AMD GPU check: ready (1 device)\n"
        "  url: http://127.0.0.1:18188\n  browser: not opened (--no-open-browser)\n"
        "  pid: 12345\n",
        "127.0.0.1",
        18188,
    )
    assert_start_output(
        "ComfyUI\n  status: starting\n  AMD GPU check: ready (1 device)\n"
        "  URL: http://127.0.0.1:18188\n  browser: not opened (--no-open-browser)\n"
        "  pid: 12345\n",
        "127.0.0.1",
        18188,
    )
    assert_status_output(
        "ComfyUI\n\nRunning\n  status: running\n  url: http://127.0.0.1:18188\n",
        "127.0.0.1",
        18188,
    )
    assert_logs_output("ComfyUI logs\n\nRun log\n  latest output:\n    started\n")
    try:
        assert_status_output(
            "ComfyUI\n\nRunning\n  status: stopped\n  url: http://127.0.0.1:18188\n",
            "127.0.0.1",
            18188,
        )
    except RuntimeError as error:
        assert "status: running" in str(error) or "stopped" in str(error)
    else:
        raise AssertionError("stopped status was incorrectly accepted")
    try:
        assert_start_output(
            "ComfyUI\n  status: starting\n  AMD GPU check: ready (1 device)\n"
            "  url: http://127.0.0.1:18188\n  browser: not opened (--no-open-browser)\n"
            "  pid: 12345\n  CPU fallback enabled\n",
            "127.0.0.1",
            18188,
        )
    except RuntimeError as error:
        assert "CPU fallback" in str(error)
    else:
        raise AssertionError("CPU fallback output was incorrectly accepted")

    assert parse_comfyui_root("ComfyUI\n  folder: /tmp/ComfyUI\n") == Path("/tmp/ComfyUI")
    with tempfile.TemporaryDirectory(
        prefix="rocm-cli-comfyui-copy-source-"
    ) as src_text, tempfile.TemporaryDirectory(
        prefix="rocm-cli-comfyui-copy-dest-"
    ) as dst_text:
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
    assert DEFAULT_CHECKPOINT_URL.endswith("/sd-v1-5-tiny.safetensors")
    assert format_bytes(DEFAULT_CHECKPOINT_SIZE_BYTES).endswith("GB")

    workflow = build_cat_workflow(
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
        prompt=DEFAULT_CAT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=1,
        width=512,
        height=512,
        steps=4,
        cfg=3.5,
        sampler="euler",
        scheduler="normal",
    )
    assert workflow["4"]["class_type"] == "CheckpointLoaderSimple"
    assert workflow["4"]["inputs"]["ckpt_name"] == DEFAULT_CHECKPOINT_NAME
    assert workflow["9"]["class_type"] == "SaveImage"
    try:
        build_cat_workflow(
            checkpoint_name=DEFAULT_CHECKPOINT_NAME,
            prompt="cat",
            negative_prompt="",
            seed=1,
            width=513,
            height=512,
            steps=4,
            cfg=3.5,
            sampler="euler",
            scheduler="normal",
        )
    except ValueError as error:
        assert "divisible by 8" in str(error)
    else:
        raise AssertionError("non-latent-aligned width was incorrectly accepted")

    object_info = {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [[DEFAULT_CHECKPOINT_NAME, "other.ckpt"], {}],
                },
            },
        },
    }
    assert checkpoint_choices_from_object_info(object_info) == [
        DEFAULT_CHECKPOINT_NAME,
        "other.ckpt",
    ]

    calls = {"count": 0}

    def fake_fetch() -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {}
        return {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "9": {
                    "images": [
                        {
                            "filename": "rocm_cli_cat_00001_.png",
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            },
        }

    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        return clock["value"]

    def fake_sleep(seconds: float) -> None:
        clock["value"] += seconds

    history = poll_prompt_history(
        fake_fetch,
        "prompt-1",
        timeout=5,
        interval=1,
        sleep=fake_sleep,
        monotonic=fake_monotonic,
    )
    images = extract_output_images(history)
    assert images == [
        {
            "filename": "rocm_cli_cat_00001_.png",
            "subfolder": "",
            "type": "output",
        }
    ]

    try:
        poll_prompt_history(
            lambda: {
                "status": {"completed": False, "status_str": "error"},
                "outputs": {},
            },
            "prompt-2",
            timeout=5,
            interval=1,
            sleep=fake_sleep,
            monotonic=fake_monotonic,
        )
    except RuntimeError as error:
        assert "failed" in str(error)
    else:
        raise AssertionError("failed ComfyUI prompt was incorrectly accepted")

    assert_comfyui_reports_gpu(
        {"devices": [{"name": "AMD Radeon RX 9070 XT", "type": "cuda"}]}
    )
    try:
        assert_comfyui_reports_gpu({"devices": [{"name": "CPU", "type": "cpu"}]})
    except RuntimeError as error:
        assert "CPU fallback" in str(error)
    else:
        raise AssertionError("CPU-only ComfyUI device report was incorrectly accepted")

    print("ComfyUI GPU script self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
