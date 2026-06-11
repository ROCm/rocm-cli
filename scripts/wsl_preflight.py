#!/usr/bin/env python3
"""Read-only WSL/ROCDXG preflight for rocm-cli.

This script intentionally does not install packages or modify global WSL state.
It can run from Windows and inspect a WSL distro, or run directly inside WSL.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

LINUX_COLLECTOR = r"""
import glob
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

def parse_os_release(text):
    result = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip("\"'")
    return result

def command_output(argv, timeout=20):
    try:
        completed = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "output": str(error)}
    return {"ok": completed.returncode == 0, "output": completed.stdout.strip()}

tools = ["git", "cmake", "gcc", "g++", "make", "python3", "rocminfo", "cargo"]
tool_paths = {tool: shutil.which(tool) for tool in tools}
venv_probe = {"ok": False, "output": "python3 missing"}
if tool_paths.get("python3"):
    venv_probe = command_output([tool_paths["python3"], "-m", "venv", "--help"], timeout=20)

ldconfig = command_output(["ldconfig", "-p"], timeout=20)
ldconfig_text = ldconfig.get("output", "")
proc_version = read_text("/proc/version")
sdk_headers = sorted(
    glob.glob("/mnt/c/Program Files (x86)/Windows Kits/10/Include/*/shared/dxcore_interface.h")
    + glob.glob("/mnt/c/Program Files (x86)/Windows Kits/10/Include/*/um/dxcore_interface.h")
    + glob.glob("/mnt/c/Program Files (x86)/Windows Kits/10/Include/*/um/dxcore.h")
)

state = {
    "collector": "linux",
    "kernel": platform.release(),
    "machine": platform.machine(),
    "proc_version": proc_version.strip(),
    "is_wsl": "microsoft" in proc_version.lower() or Path("/dev/dxg").exists(),
    "os_release": parse_os_release(read_text("/etc/os-release")),
    "paths": {
        "/dev/dxg": Path("/dev/dxg").exists(),
        "/usr/lib/wsl/lib/libdxcore.so": Path("/usr/lib/wsl/lib/libdxcore.so").exists(),
        "/opt/rocm/lib/librocdxg.so": Path("/opt/rocm/lib/librocdxg.so").exists(),
        "/opt/rocm/share/rocdxg/dids.conf": Path("/opt/rocm/share/rocdxg/dids.conf").exists(),
    },
    "ldconfig": {
        "ok": bool(ldconfig.get("ok")),
        "libdxcore": "libdxcore.so" in ldconfig_text,
        "librocdxg": "librocdxg.so" in ldconfig_text,
        "libhsa_runtime64": "libhsa-runtime64.so" in ldconfig_text,
        "libamdhip64": "libamdhip64.so" in ldconfig_text,
    },
    "tools": tool_paths,
    "python_venv": bool(venv_probe.get("ok")),
    "windows_sdk_headers": sdk_headers,
    "env": {
        "HSA_ENABLE_DXG_DETECTION": os.environ.get("HSA_ENABLE_DXG_DETECTION"),
        "ROCM_ROOT": os.environ.get("ROCM_ROOT"),
        "ROCM_PATH": os.environ.get("ROCM_PATH"),
        "HIP_PATH": os.environ.get("HIP_PATH"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    },
}
print(json.dumps(state, sort_keys=True))
"""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def clean_output(text: str) -> str:
    return text.replace("\x00", "")


def run(argv: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def parse_wsl_list(text: str) -> list[str]:
    distros: list[str] = []
    for raw in clean_output(text).splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        if line.startswith("*"):
            line = line[1:].strip()
        parts = line.split()
        if parts:
            distros.append(parts[0])
    return distros


def collect_from_windows(distro: str | None) -> dict[str, Any]:
    if not shutil.which("wsl.exe"):
        return {"collector": "windows", "error": "wsl.exe was not found on PATH"}

    listed = run(["wsl.exe", "-l", "-v"], timeout=30)
    if listed.returncode != 0:
        return {
            "collector": "windows",
            "error": "failed to list WSL distributions",
            "output": clean_output(listed.stdout).strip(),
        }

    distros = parse_wsl_list(listed.stdout)
    if distro is None and len(distros) > 1:
        return {
            "collector": "windows",
            "error": "multiple WSL distributions found; pass --distro explicitly",
            "distros": distros,
            "wsl_list": clean_output(listed.stdout).strip(),
        }
    selected = distro or (distros[0] if distros else None)
    if not selected:
        return {
            "collector": "windows",
            "error": "no WSL distributions were found",
            "wsl_list": clean_output(listed.stdout).strip(),
        }

    inspected = run(
        [
            "wsl.exe",
            "-d",
            selected,
            "--exec",
            "/bin/bash",
            "-lc",
            f"python3 - <<'PY'\n{LINUX_COLLECTOR}\nPY",
        ],
        timeout=90,
    )
    if inspected.returncode != 0:
        return {
            "collector": "windows",
            "distro": selected,
            "error": "failed to inspect WSL distribution; python3 may be missing",
            "output": clean_output(inspected.stdout).strip(),
        }

    try:
        state = json.loads(clean_output(inspected.stdout))
    except json.JSONDecodeError as error:
        return {
            "collector": "windows",
            "distro": selected,
            "error": f"failed to parse WSL inspection JSON: {error}",
            "output": clean_output(inspected.stdout).strip(),
        }
    state["collector"] = "windows-wsl"
    state["distro"] = selected
    state["wsl_list"] = clean_output(listed.stdout).strip()
    return state


def collect_local_linux() -> dict[str, Any]:
    completed = run(
        ["python3", "-c", LINUX_COLLECTOR],
        timeout=60,
    )
    if completed.returncode != 0:
        return {
            "collector": "linux",
            "error": "failed to inspect local Linux host",
            "output": completed.stdout.strip(),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "collector": "linux",
            "error": f"failed to parse local inspection JSON: {error}",
            "output": completed.stdout.strip(),
        }


def collect_state(distro: str | None) -> dict[str, Any]:
    if platform.system() == "Windows":
        return collect_from_windows(distro)
    if platform.system() == "Linux":
        return collect_local_linux()
    return {
        "collector": platform.system().lower(),
        "error": "only Windows and Linux are supported",
    }


def evaluate(
    state: dict[str, Any],
    require_rocm_tools: bool,
    require_build_tools: bool = False,
) -> list[Check]:
    if state.get("error"):
        return [Check("inspect", False, str(state["error"]))]

    paths = state.get("paths") if isinstance(state.get("paths"), dict) else {}
    tools = state.get("tools") if isinstance(state.get("tools"), dict) else {}
    os_release = (
        state.get("os_release") if isinstance(state.get("os_release"), dict) else {}
    )
    ldconfig = state.get("ldconfig") if isinstance(state.get("ldconfig"), dict) else {}
    sdk_headers = state.get("windows_sdk_headers") or []

    version_id = str(os_release.get("VERSION_ID") or "")
    supported_ubuntu = os_release.get("ID") == "ubuntu" and version_id in {
        "22.04",
        "24.04",
        "26.04",
    }
    build_tools = all(
        tools.get(name) for name in ["git", "cmake", "gcc", "g++", "make"]
    )

    checks = [
        Check("wsl", bool(state.get("is_wsl")), "WSL marker or /dev/dxg detected"),
        Check(
            "ubuntu", supported_ubuntu, f"Ubuntu VERSION_ID={version_id or '<unknown>'}"
        ),
        Check("dxg_device", bool(paths.get("/dev/dxg")), "/dev/dxg"),
        Check(
            "dxcore",
            bool(paths.get("/usr/lib/wsl/lib/libdxcore.so")),
            "/usr/lib/wsl/lib/libdxcore.so",
        ),
        Check(
            "windows_sdk",
            bool(sdk_headers),
            "Windows SDK dxcore headers visible from WSL",
            required=require_build_tools,
        ),
        Check(
            "librocdxg",
            bool(paths.get("/opt/rocm/lib/librocdxg.so")),
            "/opt/rocm/lib/librocdxg.so",
        ),
        Check(
            "rocdxg_dids",
            bool(paths.get("/opt/rocm/share/rocdxg/dids.conf")),
            "/opt/rocm/share/rocdxg/dids.conf (not shipped by rocdxg-roct 1.2.0)",
            required=False,
        ),
        Check(
            "ldconfig_librocdxg",
            bool(ldconfig.get("librocdxg")),
            "librocdxg visible through ldconfig -p",
        ),
        Check(
            "build_tools",
            build_tools,
            "git cmake gcc g++ make",
            required=require_build_tools,
        ),
        Check("python_venv", bool(state.get("python_venv")), "python3 -m venv"),
        Check(
            "rocminfo",
            bool(tools.get("rocminfo")),
            "rocminfo command after ROCm/TheRock activation",
            required=require_rocm_tools,
        ),
        Check(
            "cargo",
            bool(tools.get("cargo")),
            "cargo for building rocm-cli in WSL",
            required=False,
        ),
    ]
    return checks


def render_human(state: dict[str, Any], checks: list[Check]) -> str:
    lines = ["WSL preflight"]
    if state.get("distro"):
        lines.append(f"  distro: {state['distro']}")
    if state.get("kernel"):
        lines.append(f"  kernel: {state['kernel']}")
    os_release = (
        state.get("os_release") if isinstance(state.get("os_release"), dict) else {}
    )
    if os_release.get("PRETTY_NAME"):
        lines.append(f"  os: {os_release['PRETTY_NAME']}")
    if state.get("error"):
        lines.append(f"  error: {state['error']}")
    lines.append("  checks:")
    for check in checks:
        status = (
            "ok" if check.ok else ("missing" if check.required else "optional-missing")
        )
        lines.append(f"    {check.name}: {status} ({check.detail})")
    if any(not check.ok and check.required for check in checks):
        lines.append("  next:")
        lines.append("    install/verify ROCDXG before running GPU HIP apps in WSL")
        lines.append("    run with --json for machine-readable details")
    return "\n".join(lines)


def self_test() -> None:
    distros = parse_wsl_list(
        "  NAME      STATE           VERSION\n* Ubuntu    Stopped         2\n"
    )
    assert distros == ["Ubuntu"], distros
    distros = parse_wsl_list(
        "  NAME      STATE           VERSION\n* Ubuntu    Running         2\n  Debian    Stopped         2\n"
    )
    assert distros == ["Ubuntu", "Debian"], distros

    base_state = {
        "is_wsl": True,
        "os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
        "paths": {
            "/dev/dxg": True,
            "/usr/lib/wsl/lib/libdxcore.so": True,
            "/opt/rocm/lib/librocdxg.so": False,
            "/opt/rocm/share/rocdxg/dids.conf": False,
        },
        "ldconfig": {"librocdxg": False},
        "tools": {
            "git": "/usr/bin/git",
            "cmake": "/usr/bin/cmake",
            "gcc": "/usr/bin/gcc",
            "g++": "/usr/bin/g++",
            "make": "/usr/bin/make",
            "rocminfo": None,
            "cargo": None,
        },
        "python_venv": True,
        "windows_sdk_headers": ["/mnt/c/sdk/shared/dxcore_interface.h"],
    }
    checks = evaluate(base_state, require_rocm_tools=False)
    assert any(check.name == "librocdxg" and not check.ok for check in checks)
    assert not all(check.ok for check in checks if check.required)

    ready_state = json.loads(json.dumps(base_state))
    ready_state["paths"]["/opt/rocm/lib/librocdxg.so"] = True
    ready_state["ldconfig"]["librocdxg"] = True
    checks = evaluate(ready_state, require_rocm_tools=False)
    assert all(check.ok for check in checks if check.required), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distro", help="WSL distribution name when running from Windows"
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero unless WSL and ROCDXG are ready",
    )
    parser.add_argument(
        "--require-rocm-tools", action="store_true", help="also require rocminfo"
    )
    parser.add_argument(
        "--require-build-tools",
        action="store_true",
        help="also require source-build tools such as the Windows SDK and CMake toolchain",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run parser/status self-tests without touching WSL",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("wsl-preflight self-test ok")
        return 0

    state = collect_state(args.distro)
    checks = evaluate(
        state,
        require_rocm_tools=args.require_rocm_tools,
        require_build_tools=args.require_build_tools,
    )
    payload = {
        "state": state,
        "checks": [check.__dict__ for check in checks],
        "ready": all(check.ok for check in checks if check.required),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_human(state, checks))

    if args.require_ready and not payload["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
