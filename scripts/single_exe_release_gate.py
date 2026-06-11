#!/usr/bin/env python3
"""Release gate for the single-file rocm APE artifact.

The default checks are safe and isolated: they do not install TheRock,
ComfyUI, models, or write the user's real `.rocm` state. Live GPU checks are
opt-in because they need an already installed managed TheRock runtime.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERNAL_ARTIFACT = (
    REPO_ROOT / ".rocm-work" / "tests" / "rust-cosmopolitan" / "rocm-rust-cosmo-release.exe"
)
DEFAULT_RELEASE_ARTIFACT = REPO_ROOT / ".rocm-work" / "single-exe-release" / "rocm.exe"


class GateError(Exception):
    pass


def main() -> int:
    args = parse_args()
    artifact = resolve_path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"single-exe artifact not found: {artifact}")
    if args.stage_release:
        artifact = stage_release_artifact(artifact, resolve_path(args.release_output))

    run_python_self_tests()
    run_windows_smoke(artifact)
    if args.wsl:
        run_wsl_smoke(artifact)
    if args.live_assistant:
        run_live_assistant(args, artifact)
    if args.live_comfyui:
        run_live_comfyui(args, artifact)
    print("[single-exe-gate] ok")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(default_artifact()))
    parser.add_argument(
        "--stage-release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the checked artifact to the clean release name before smoke tests",
    )
    parser.add_argument("--release-output", default=str(DEFAULT_RELEASE_ARTIFACT))
    parser.add_argument(
        "--wsl",
        action=argparse.BooleanOptionalAction,
        default=os.name == "nt",
        help="also verify the same artifact through the WSL Linux launch path",
    )
    parser.add_argument(
        "--runtime-state",
        help="existing ROCm CLI state root with config.json and runtimes/ for live GPU tests",
    )
    parser.add_argument("--live-assistant", action="store_true")
    parser.add_argument("--live-comfyui", action="store_true")
    parser.add_argument("--generate-cat", action="store_true")
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    if (args.live_assistant or args.live_comfyui) and not args.runtime_state:
        parser.error("--runtime-state is required for live GPU checks")
    return args


def default_artifact() -> Path:
    if DEFAULT_INTERNAL_ARTIFACT.is_file():
        return DEFAULT_INTERNAL_ARTIFACT
    return DEFAULT_RELEASE_ARTIFACT


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def stage_release_artifact(source: Path, destination: Path) -> Path:
    if source == destination:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        destination.chmod(destination.stat().st_mode | 0o755)
    except OSError:
        pass
    print(f"[single-exe-gate] staged release artifact: {destination}")
    return destination


def run_python_self_tests() -> None:
    print("[single-exe-gate] running assistant and ComfyUI harness self-tests")
    run([sys.executable, "scripts/local_assistant_therock_gpu_test.py", "--self-test"])
    run([sys.executable, "scripts/comfyui_therock_gpu_test.py", "--self-test"])
    run([sys.executable, "scripts/rust_cosmopolitan_spike.py", "self-test"])


def run_windows_smoke(artifact: Path) -> None:
    if os.name != "nt":
        return
    print("[single-exe-gate] running Windows single-exe doctor smoke")
    with tempfile.TemporaryDirectory(prefix="rocm-cli-single-exe-win-") as root_text:
        root = Path(root_text)
        env = isolated_env(root)
        version = run([str(artifact), "version"], env=env)
        require("rocm " in version, "Windows version smoke did not print rocm version")
        doctor = run([str(artifact), "doctor"], env=env)
        require("os: windows" in doctor, "Windows doctor did not report os: windows")
        require("detected_gfx_target:" in doctor, "Windows doctor did not report GPU target")
        run_windows_safe_command_smokes(artifact, env)
        run([sys.executable, "scripts/tui_e2e_smoke.py", "--rocm", str(artifact)], env=env)


def run_windows_safe_command_smokes(artifact: Path, env: dict[str, str]) -> None:
    print("[single-exe-gate] running Windows safe command smokes")
    services = run([str(artifact), "services", "list"], env=env)
    require("No local servers are running." in services, "services list did not use the empty-state text")
    services_all = run([str(artifact), "services", "list", "--all"], env=env)
    require("No local server records yet." in services_all, "services list --all did not use the empty all-records text")
    runtimes = run([str(artifact), "runtimes", "list"], env=env)
    require("installed: none" in runtimes, "runtimes list did not report the empty install state")
    comfyui = run([str(artifact), "comfyui", "status"], env=env)
    require("installed: no" in comfyui, "comfyui status did not report the empty install state")
    engines = run([str(artifact), "engines", "list"], env=env)
    require("lemonade" in engines and "default embedded Lemonade" in engines, "engines list did not report Lemonade")


def run_wsl_smoke(artifact: Path) -> None:
    print("[single-exe-gate] running WSL Linux-path single-exe smoke")
    command = (
        "python3 scripts/rust_cosmopolitan_spike.py smoke-wsl-linux-path "
        f"--release --artifact {shell_quote(wsl_path(artifact))}"
    )
    run(["wsl", "-u", "jam", "--cd", "/mnt/d/jam/rocm-cli", "--", "bash", "-lc", command])


def run_live_assistant(args: argparse.Namespace, artifact: Path) -> None:
    print("[single-exe-gate] running live local-assistant GPU check")
    run(
        [
            sys.executable,
            "scripts/local_assistant_therock_gpu_test.py",
            "--rocm",
            str(artifact),
            "--temp-state",
            "--copy-runtime-state-from",
            args.runtime_state,
            "--timeout",
            str(args.timeout),
        ],
        timeout=args.timeout + 120,
    )


def run_live_comfyui(args: argparse.Namespace, artifact: Path) -> None:
    print("[single-exe-gate] running live ComfyUI GPU check")
    command = [
        sys.executable,
        "scripts/comfyui_therock_gpu_test.py",
        "--rocm",
        str(artifact),
        "--temp-state",
        "--copy-runtime-state-from",
        args.runtime_state,
        "--timeout",
        str(args.timeout),
    ]
    if args.generate_cat:
        command.extend(["--generate-cat", "--output-dir", ".rocm-work/tests/comfyui-cat"])
    run(command, timeout=args.timeout + 120)


def isolated_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_CLI_CONFIG_DIR"] = str(root / "config")
    env["ROCM_CLI_DATA_DIR"] = str(root / "data")
    env["ROCM_CLI_CACHE_DIR"] = str(root / "cache")
    env["NO_COLOR"] = "1"
    env.pop("VIRTUAL_ENV", None)
    return env


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    if completed.returncode != 0:
        raise GateError(f"command failed ({completed.returncode}): {format_command(command)}\n{output}")
    return output


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def wsl_path(path: Path) -> str:
    text = str(path)
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def format_command(command: list[str]) -> str:
    return " ".join(command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, subprocess.SubprocessError, OSError) as error:
        print(f"[single-exe-gate] failed: {error}", file=sys.stderr)
        raise SystemExit(1)
