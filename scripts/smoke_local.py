#!/usr/bin/env python3
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


def binary_paths(root: Path, profile: str, target_dir: Path | None = None) -> dict[str, Path]:
    binary_dir = (target_dir if target_dir is not None else root / "target") / profile
    return {
        "rocm": binary_dir / exe_name("rocm"),
        "rocmd": binary_dir / exe_name("rocmd"),
        "pytorch": binary_dir / exe_name("rocm-engine-pytorch"),
        "llama": binary_dir / exe_name("rocm-engine-llama-cpp"),
        "lemonade": binary_dir / exe_name("rocm-engine-lemonade"),
        "atom": binary_dir / exe_name("rocm-engine-atom"),
        "vllm": binary_dir / exe_name("rocm-engine-vllm"),
        "sglang": binary_dir / exe_name("rocm-engine-sglang"),
    }


def create_fake_llama_server(smoke_root: Path) -> Path:
    fake_dir = smoke_root / "fake-bin"
    fake_dir.mkdir(parents=True, exist_ok=True)
    server_py = fake_dir / "fake_llama_server.py"
    server_py.write_text(
        """
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import time


def arg_value(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


host = arg_value("--host", "127.0.0.1")
port = int(arg_value("--port", "11435"))
ready_port = int(os.environ.get("ROCM_CLI_FAKE_LLAMA_READY_PORT", "11437"))
print("fake llama-server " + " ".join(sys.argv[1:]), flush=True)

if port != ready_port:
    raise SystemExit(0)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        if self.path == "/v1/models":
            payload = json.dumps({"data": [{"id": "tiny.gguf"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer((host, port), Handler) as httpd:
    httpd.timeout = 0.2
    deadline = time.time() + 10
    while time.time() < deadline:
        httpd.handle_request()
""".lstrip(),
        encoding="utf-8",
    )
    if platform.system() == "Windows":
        path = fake_dir / "llama-server.cmd"
        path.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "{server_py}" %*\r\n',
            encoding="utf-8",
        )
    else:
        path = fake_dir / "llama-server"
        path.write_text(
            "#!/usr/bin/env sh\n"
            f'exec "{sys.executable}" "{server_py}" "$@"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)
    return path


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
    env["ROCM_CLI_LLAMA_CPP_SERVER"] = str(create_fake_llama_server(smoke_root))

    rocm = str(paths["rocm"])
    rocmd = str(paths["rocmd"])
    pytorch = str(paths["pytorch"])
    llama = str(paths["llama"])
    atom = str(paths["atom"])
    vllm = str(paths["vllm"])
    sglang = str(paths["sglang"])

    version = run("rocm version", [rocm, "version"], env=env)
    assert_contains(version, "rocm ", "rocm version")

    doctor = run("rocm doctor", [rocm, "doctor"], env=env)
    assert_contains(doctor, "rocm doctor", "rocm doctor")
    assert_contains(doctor, "default_engine:", "rocm doctor")
    assert_contains(doctor, "managed_runtimes: 0", "rocm doctor first-run state")
    assert_contains(doctor, "managed_services: 0", "rocm doctor first-run state")

    engines = run("rocm engines list", [rocm, "engines", "list"], env=env)
    assert_contains(engines, "llama.cpp", "rocm engines list")
    assert_contains(engines, "pytorch", "rocm engines list")
    assert_contains(engines, "atom", "rocm engines list")
    assert_contains(engines, "vllm", "rocm engines list")
    assert_contains(engines, "sglang", "rocm engines list")

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

    llama_install = parse_json(
        run(
            "llama.cpp direct external install probe",
            [llama, "install", "--runtime-id", "external"],
            env=env,
        ),
        "llama.cpp direct external install probe",
    )
    if (
        not isinstance(llama_install, dict)
        or llama_install.get("runtime_kind") != "external_llama_server"
        or llama_install.get("managed_env") is not False
        or "python_executable:" in json.dumps(llama_install)
    ):
        fail(f"unexpected llama.cpp external install probe: {llama_install}")

    engine_install = run(
        "rocm engines install requires exact runtime",
        [rocm, "engines", "install", "llama.cpp"],
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
    assert_contains(freeform_status, "ROCm status", "rocm freeform installed status question")
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
    assert_contains(freeform_comfy_help, "ComfyUI status", "rocm freeform comfyui help question")
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
        "rocm freeform llama plan",
        [rocm, "serve qwen with llama.cpp"],
        env=env,
    )
    assert_contains(plan, "engine: llama.cpp", "rocm freeform plan")
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
            "TheRock tarball installs are not supported on Windows V1",
            "windows tarball rejection",
        )

    status = run("rocmd status", [rocmd, "status"], env=env)
    assert_contains(status, "rocmd status", "rocmd status")

    bridge = parse_json(run("rocmd bridge snapshot", [rocmd, "bridge-snapshot"], env=env), "bridge snapshot")
    if not isinstance(bridge, dict) or bridge.get("protocol") != "rocmd-codex-bridge-v0":
        fail(f"unexpected bridge snapshot protocol: {bridge}")

    sandbox_doctor = parse_json(
        run(
            "rocmd sandbox doctor snapshot",
            [rocmd, "sandbox-run", "doctor_snapshot", "--allow-native-fallback"],
            env=env,
        ),
        "sandbox doctor snapshot",
    )
    if (
        not isinstance(sandbox_doctor, dict)
        or sandbox_doctor.get("tool") != "doctor_snapshot"
        or not sandbox_doctor.get("ok")
    ):
        fail(f"unexpected sandbox doctor result: {sandbox_doctor}")

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
    assert_contains(prefetch_failure, "prefetch_artifact requires", "sandbox prefetch validation")

    parse_json(run("pytorch detect", [pytorch, "detect"], env=env), "pytorch detect")
    pytorch_capabilities = parse_json(
        run("pytorch capabilities", [pytorch, "capabilities"], env=env),
        "pytorch capabilities",
    )
    if not isinstance(pytorch_capabilities, dict) or not pytorch_capabilities.get("openai_compatible"):
        fail(f"pytorch capabilities did not report OpenAI-compatible serving: {pytorch_capabilities}")

    pytorch_model = run("pytorch resolve qwen", [pytorch, "resolve-model", "qwen"], env=env)
    assert_contains(pytorch_model, "Qwen/Qwen2.5-1.5B-Instruct", "pytorch resolve qwen")
    qwen35_failure = run(
        "pytorch reject qwen3.5",
        [pytorch, "resolve-model", "qwen3.5"],
        env=env,
        expect_failure=True,
    )
    assert_contains(
        qwen35_failure,
        "not supported by the managed PyTorch engine",
        "pytorch reject qwen3.5",
    )
    tiny_pytorch = parse_json(
        run("pytorch resolve tiny gpu recipe", [pytorch, "resolve-model", "tiny-gpt2"], env=env),
        "pytorch resolve tiny-gpt2",
    )
    if (
        not isinstance(tiny_pytorch, dict)
        or tiny_pytorch.get("canonical_model_id") != "sshleifer/tiny-gpt2"
        or tiny_pytorch.get("device_policy") != "gpu_required"
        or tiny_pytorch.get("dtype") != "float16"
    ):
        fail(f"pytorch tiny-gpt2 did not resolve as GPU-required recipe: {tiny_pytorch}")

    parse_json(run("llama.cpp detect", [llama, "detect"], env=env), "llama.cpp detect")
    llama_capabilities = parse_json(
        run("llama.cpp capabilities", [llama, "capabilities"], env=env),
        "llama.cpp capabilities",
    )
    if (
        not isinstance(llama_capabilities, dict)
        or not llama_capabilities.get("openai_compatible")
        or llama_capabilities.get("quantized_models") != "gguf"
    ):
        fail(f"llama.cpp capabilities did not report expected GGUF/OpenAI support: {llama_capabilities}")

    llama_model = run("llama.cpp resolve gguf", [llama, "resolve-model", "tiny.gguf"], env=env)
    assert_contains(llama_model, "tiny.gguf", "llama.cpp resolve gguf")
    llama_cpu = run(
        "llama.cpp reject cpu",
        [llama, "launch", "smoke-cpu", "tiny.gguf", "--device-policy", "cpu_only"],
        env=env,
        expect_failure=True,
    )
    assert_contains(llama_cpu, "no CPU fallback is used", "llama.cpp reject cpu")

    parse_json(run("atom detect", [atom, "detect"], env=env), "atom detect")
    atom_capabilities = parse_json(
        run("atom capabilities", [atom, "capabilities"], env=env),
        "atom capabilities",
    )
    if (
        not isinstance(atom_capabilities, dict)
        or not atom_capabilities.get("openai_compatible")
        or atom_capabilities.get("cpu")
    ):
        fail(f"atom capabilities did not report GPU-only OpenAI serving: {atom_capabilities}")

    atom_model = run("atom resolve qwen", [atom, "resolve-model", "qwen"], env=env)
    assert_contains(atom_model, "qwen", "atom resolve qwen")
    atom_cpu = run(
        "atom reject cpu",
        [atom, "resolve-model", "qwen", "--device-policy", "cpu_only"],
        env=env,
        expect_failure=True,
    )
    assert_contains(atom_cpu, "no CPU fallback is used", "atom reject cpu")

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
        fail(f"vllm capabilities did not report GPU-only OpenAI serving: {vllm_capabilities}")

    vllm_model = run("vllm resolve qwen", [vllm, "resolve-model", "qwen"], env=env)
    assert_contains(vllm_model, "qwen", "vllm resolve qwen")
    vllm_cpu = run(
        "vllm reject cpu",
        [vllm, "resolve-model", "qwen", "--device-policy", "cpu_only"],
        env=env,
        expect_failure=True,
    )
    assert_contains(vllm_cpu, "no CPU fallback is used", "vllm reject cpu")

    parse_json(run("sglang detect", [sglang, "detect"], env=env), "sglang detect")
    sglang_capabilities = parse_json(
        run("sglang capabilities", [sglang, "capabilities"], env=env),
        "sglang capabilities",
    )
    if (
        not isinstance(sglang_capabilities, dict)
        or not sglang_capabilities.get("openai_compatible")
        or sglang_capabilities.get("cpu")
    ):
        fail(f"sglang capabilities did not report GPU-only OpenAI serving: {sglang_capabilities}")

    sglang_model = run("sglang resolve qwen", [sglang, "resolve-model", "qwen"], env=env)
    assert_contains(sglang_model, "qwen", "sglang resolve qwen")
    sglang_cpu = run(
        "sglang reject cpu",
        [sglang, "resolve-model", "qwen", "--device-policy", "cpu_only"],
        env=env,
        expect_failure=True,
    )
    assert_contains(sglang_cpu, "no CPU fallback is used", "sglang reject cpu")

    llama_gpu_required = run(
        "llama.cpp reject required gpu",
        [
            rocm,
            "serve",
            "tiny.gguf",
            "--engine",
            "llama.cpp",
            "--device",
            "gpu_required",
            "--foreground",
            "--port",
            str(reject_port),
        ],
        env=env,
        expect_failure=True,
    )
    assert_contains(llama_gpu_required, "gpu_required", "llama.cpp reject required gpu")
    assert_not_contains(
        llama_gpu_required,
        "CPU fallback",
        "llama.cpp reject required gpu",
    )

    llama_cpu_serve = run(
        "rocm llama.cpp reject cpu serve",
        [
            rocm,
            "serve",
            "tiny.gguf",
            "--engine",
            "llama.cpp",
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
        llama_cpu_serve,
        "CPU mode is not a fallback path",
        "rocm llama.cpp reject cpu serve",
    )

    assert_path_missing(
        smoke_root / "rocm-cache" / "pip",
        "first-run smoke pip cache",
    )
    assert_path_missing(
        smoke_root / "rocm-data" / "runtimes" / "registry",
        "first-run smoke runtime registry",
    )

    print("smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
