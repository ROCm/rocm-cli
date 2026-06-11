#!/usr/bin/env python3
"""Build Rust/Cosmopolitan proof artifacts for rocm-cli.

This is a spike harness, not the release builder. It follows the public
Rust/Cosmopolitan recipe used by `ahgamut/rust-ape-example`:

- use a custom `x86_64-unknown-linux-cosmo` target JSON;
- build Rust `std` with nightly Cargo `-Z build-std`;
- link with Cosmopolitan through a tiny GCC-style wrapper;
- keep all generated files under `.rocm-work`.

The first goal is to prove a Rust `std` executable can become an APE. The next
goal is to move from the generated hello fixture to a small rocm-cli subset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_WORK_ROOT = REPO_ROOT / ".rocm-work" / "tests" / "rust-cosmopolitan"
DEFAULT_RELEASE_ROOT = REPO_ROOT / ".rocm-work" / "single-exe-release"
DEFAULT_RUSTUP_HOME = REPO_ROOT / ".rocm-work" / "tools" / "rustup"
DEFAULT_CARGO_HOME = REPO_ROOT / ".rocm-work" / "tools" / "cargo"
DEFAULT_COSMOCC_NATIVE_ROOT = REPO_ROOT / ".rocm-work" / "tools" / "cosmocc"
DEFAULT_COSMOCC_WSL_ELF_ROOT = REPO_ROOT / ".rocm-work" / "tools" / "cosmocc-wsl-elf"
DEFAULT_COSMOPOLITAN_SOURCE_ROOT = REPO_ROOT / ".rocm-work" / "tools" / "cosmopolitan-src"
DEFAULT_TOOLCHAIN = "nightly"


class RustCosmoError(Exception):
    """The Rust/Cosmopolitan spike failed."""


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def fail(message: str) -> None:
    print(f"rust-cosmopolitan spike failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_capture(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result = CommandResult(args, completed.returncode, completed.stdout, completed.stderr)
    if check and result.returncode != 0:
        raise RustCosmoError(format_command_failure(result))
    return result


def format_command_failure(result: CommandResult) -> str:
    output = (result.stdout + result.stderr).strip()
    command = " ".join(result.args)
    if output:
        return f"command failed with exit {result.returncode}: {command}\n{output}"
    return f"command failed with exit {result.returncode}: {command}"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def host_is_wsl() -> bool:
    if os.name == "nt":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "microsoft" in version.lower() or "wsl" in version.lower()


def cosmocc_root_has_minimal_tools(root: Path) -> bool:
    return all(
        (root / "bin" / name).is_file()
        for name in (
            "cosmocc",
            "cosmocross",
            "x86_64-unknown-cosmo-cc",
            "ape-x86_64.elf",
        )
    )


def default_cosmocc_root() -> Path:
    if host_is_wsl() or cosmocc_root_has_minimal_tools(DEFAULT_COSMOCC_WSL_ELF_ROOT):
        return DEFAULT_COSMOCC_WSL_ELF_ROOT
    return DEFAULT_COSMOCC_NATIVE_ROOT


def require_posix_host() -> None:
    if os.name == "nt":
        raise RustCosmoError(
            "Rust/Cosmopolitan builds must run from WSL or native Linux for now; "
            "run this script through `wsl -u jam -- bash -lc ...`"
        )


def path_text(path: Path) -> str:
    return path.as_posix()


def prepend_path(env: dict[str, str], *entries: Path) -> None:
    values = [path_text(entry) for entry in entries]
    old = env.get("PATH")
    if old:
        values.append(old)
    env["PATH"] = os.pathsep.join(values)


def spike_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    linker_root = args.work_root.resolve() / "cosmocc-link-root"
    env["RUSTUP_HOME"] = path_text(args.rustup_home.resolve())
    env["CARGO_HOME"] = path_text(args.cargo_home.resolve())
    env["ROCM_CLI_COSMOCC_ROOT"] = path_text(args.cosmocc_root.resolve())
    env["ROCM_CLI_RUST_COSMO_ROOT"] = path_text(linker_root)
    env["COSMO"] = path_text(linker_root)
    prepend_path(
        env,
        args.cargo_home.resolve() / "bin",
        linker_root / "bin",
        args.cosmocc_root.resolve() / "bin",
    )
    return env


def check_cosmocc_root(path: Path) -> None:
    missing = [
        path / "bin" / "cosmocc",
        path / "bin" / "cosmocross",
        path / "bin" / "x86_64-unknown-cosmo-cc",
        path / "bin" / "ape-x86_64.elf",
    ]
    missing = [candidate for candidate in missing if not candidate.is_file()]
    if missing:
        raise RustCosmoError(
            "Cosmopolitan toolchain is incomplete; missing:\n"
            + "\n".join(f"  {candidate}" for candidate in missing)
        )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def target_json(linker: Path) -> str:
    # This intentionally remains a Linux-ish target. Existing Rust/Cosmo work
    # uses this shape; Windows behavior must be tested and fixed explicitly.
    return """{
  "llvm-target": "x86_64-unknown-linux-musl",
  "target-pointer-width": 64,
  "arch": "x86_64",
  "data-layout": "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  "cpu": "x86-64",
  "os": "linux",
  "env": "musl",
  "vendor": "cosmo",
  "panic-strategy": "abort",
  "requires-uwtable": false,
  "dynamic-linking": false,
  "executables": true,
  "exe-suffix": ".com.dbg",
  "emit-debug-gdb-scripts": false,
  "crt-static-default": true,
  "crt-static-respected": true,
  "linker-is-gnu": true,
  "has-rpath": false,
  "has-thread-local": false,
  "trap-unreachable": true,
  "position-independent-executables": false,
  "static-position-independent-executables": false,
  "relocation-model": "static",
  "disable-redzone": true,
  "frame-pointer": "always",
  "requires-lto": false,
  "eh-frame-header": false,
  "no-default-libraries": true,
  "max-atomic-width": 64,
  "linker-flavor": "gcc",
  "linker": "%s",
  "pre-link-args": {
    "gcc": ["-static", "-pg", "-mnop-mcount"]
  },
  "stack-probes": { "kind": "none" },
  "target-family": ["unix"]
}
""" % path_text(linker)


def linker_wrapper() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROCM_CLI_COSMOCC_ROOT:?ROCM_CLI_COSMOCC_ROOT is required}"
ROOT="${ROCM_CLI_RUST_COSMO_ROOT:-$ROOT}"
ARCH="${ROCM_CLI_RUST_COSMO_ARCH:-x86_64}"
export PATH="$ROOT/bin:$PATH"

args=()
for arg in "$@"; do
  case "$arg" in
    -lunwind|-Wl,-Bdynamic|-Wl,-Bstatic)
      continue
      ;;
  esac
  args+=("$arg")
done

exec "$ROOT/bin/${ARCH}-unknown-cosmo-cc" "${args[@]}"
"""


def reset_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            remove_tree(path)
        else:
            path.unlink()


def remove_tree(path: Path) -> None:
    last_error: Exception | None = None
    for _attempt in range(8):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    if last_error is not None:
        raise last_error


def symlink_or_copy_link(link: Path, target: Path, *, is_dir: bool = False) -> None:
    reset_path(link)
    link.symlink_to(target, target_is_directory=is_dir)


def resolved_tool_target(source_bin: Path, tool: Path) -> Path:
    if not tool.is_file() or tool.stat().st_size > 256:
        return tool
    try:
        text = tool.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return tool
    if not text or "/" in text or "\\" in text or any(char.isspace() for char in text):
        return tool
    candidate = source_bin / text
    if candidate.exists():
        return candidate
    return tool


def prepare_linker_toolchain(args: argparse.Namespace) -> None:
    ensure_cosmopolitan_waitid(args)
    source_root = args.cosmocc_root.resolve()
    link_root = args.work_root.resolve() / "cosmocc-link-root"
    source_bin = source_root / "bin"
    link_bin = link_root / "bin"
    reset_path(link_root)
    link_bin.mkdir(parents=True, exist_ok=True)

    for child in source_bin.iterdir():
        target = resolved_tool_target(source_bin, child)
        symlink_or_copy_link(link_bin / child.name, target, is_dir=target.is_dir())

    for name in ("include", "libexec", "x86_64-linux-cosmo", "aarch64-linux-cosmo"):
        source = source_root / name
        if source.exists():
            symlink_or_copy_link(link_root / name, source, is_dir=source.is_dir())

    patch_rust_std_for_cosmopolitan(args)


def cosmopolitan_waitid_source() -> str:
    return r"""/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-*/
#include "libc/calls/calls.h"
#include "libc/calls/cp.internal.h"
#include "libc/calls/struct/siginfo.h"
#include "libc/calls/weirdtypes.h"
#include "libc/dce.h"
#include "libc/intrin/strace.h"
#include "libc/sysv/errfuns.h"

int sys_waitid(int, id_t, siginfo_t *, int, void *);

/**
 * Waits for status to change on process.
 *
 * This wraps the Linux waitid syscall. Hosts that do not expose waitid return
 * ENOSYS, matching Cosmopolitan's syscall-wrapper convention.
 *
 * @param idtype can be P_ALL, P_PID, P_PGID, or Linux P_PIDFD
 * @param id identifies the process, process group, or pidfd
 * @param opt_out_siginfo optionally receives child status details
 * @param options can have WEXITED, WSTOPPED, WNOHANG, WNOWAIT, etc.
 * @return 0 on success or -1 w/ errno
 * @cancelationpoint
 * @asyncsignalsafe
 * @restartable
 */
int waitid(int idtype, id_t id, siginfo_t *opt_out_siginfo, int options) {
  int rc;
  BEGIN_CANCELATION_POINT;
  if (IsLinux() || IsXnu() || IsFreebsd()) {
    rc = sys_waitid(idtype, id, opt_out_siginfo, options, 0);
  } else {
    rc = enosys();
  }
  END_CANCELATION_POINT;
  STRACE("waitid(%d, %d, %p, %d) -> %d% m", idtype, id, opt_out_siginfo,
         options, rc);
  return rc;
}
"""


def cosmopolitan_libraries(cosmocc_root: Path) -> list[Path]:
    return [
        cosmocc_root / "x86_64-linux-cosmo" / "lib" / "libcosmo.a",
        cosmocc_root / "x86_64-linux-cosmo" / "lib" / "optlinux" / "libcosmo.a",
    ]


def archive_exports_symbol(library: Path, symbol: str) -> bool:
    if not library.is_file():
        return False
    result = run_capture(["nm", "-g", path_text(library)], cwd=library.parent)
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == symbol and parts[-2] in {"T", "W"}:
            return True
    return False


def cosmopolitan_exports_symbol(cosmocc_root: Path, symbol: str) -> bool:
    return any(archive_exports_symbol(library, symbol) for library in cosmopolitan_libraries(cosmocc_root))


def ensure_cosmopolitan_source(args: argparse.Namespace) -> Path:
    source_root = args.cosmopolitan_source_root.resolve()
    if (source_root / ".git").is_dir():
        return source_root
    source_root.parent.mkdir(parents=True, exist_ok=True)
    result = run_capture(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/jart/cosmopolitan",
            path_text(source_root),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RustCosmoError(format_command_failure(result))
    return source_root


def patch_cosmopolitan_waitid_source(source_root: Path) -> Path:
    calls_header = source_root / "libc" / "calls" / "calls.h"
    if not calls_header.is_file():
        raise RustCosmoError(f"Cosmopolitan calls.h was not found: {calls_header}")
    header_text = calls_header.read_text(encoding="utf-8")
    if "struct siginfo;" not in header_text:
        header_text = header_text.replace("typedef int sig_atomic_t;\n", "typedef int sig_atomic_t;\nstruct siginfo;\n", 1)
    if "int waitid(int, int, struct siginfo *, int)" not in header_text:
        anchor = "int wait(int *) libcesque __write_only(1);\n"
        if anchor not in header_text:
            raise RustCosmoError(f"could not locate wait() anchor in {calls_header}")
        header_text = header_text.replace(
            anchor,
            anchor + "int waitid(int, int, struct siginfo *, int) libcesque __write_only(3);\n",
            1,
        )
    calls_header.write_text(header_text, encoding="utf-8", newline="\n")

    source_path = source_root / "libc" / "proc" / "waitid.c"
    source = cosmopolitan_waitid_source()
    if not source_path.is_file() or source_path.read_text(encoding="utf-8") != source:
        write_text(source_path, source)
    return source_path


def compile_cosmopolitan_waitid_object(args: argparse.Namespace, source_path: Path) -> Path:
    work_root = args.work_root.resolve()
    object_path = work_root / "cosmopolitan-source-patch" / "waitid.o"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = args.cosmocc_root.resolve() / "bin" / "cosmocc"
    env = os.environ.copy()
    prepend_path(env, args.cosmocc_root.resolve() / "bin")
    result = run_capture(
        [
            path_text(compiler),
            "-mcosmo",
            "-O2",
            f"-I{path_text(source_path.parent.parent.parent)}",
            "-c",
            path_text(source_path),
            "-o",
            path_text(object_path),
        ],
        cwd=work_root,
        env=env,
    )
    if result.returncode != 0:
        write_text(work_root / "cosmopolitan-waitid.stdout.log", result.stdout)
        write_text(work_root / "cosmopolitan-waitid.stderr.log", result.stderr)
        raise RustCosmoError(format_command_failure(result))
    return object_path


def ensure_cosmopolitan_waitid(args: argparse.Namespace) -> None:
    cosmocc_root = args.cosmocc_root.resolve()
    if cosmopolitan_exports_symbol(cosmocc_root, "waitid"):
        return
    source_root = ensure_cosmopolitan_source(args)
    source_path = patch_cosmopolitan_waitid_source(source_root)
    object_path = compile_cosmopolitan_waitid_object(args, source_path)
    ar = cosmocc_root / "bin" / "x86_64-linux-cosmo-ar"
    for library in cosmopolitan_libraries(cosmocc_root):
        if not library.is_file():
            continue
        backup = library.with_suffix(library.suffix + ".before-waitid-source-patch")
        if not backup.is_file():
            shutil.copy2(library, backup)
        result = run_capture([path_text(ar), "rcs", path_text(library), path_text(object_path)], cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RustCosmoError(format_command_failure(result))
        if not archive_exports_symbol(library, "waitid"):
            raise RustCosmoError(f"patched Cosmopolitan archive still does not export waitid: {library}")


def rust_sysroot(args: argparse.Namespace) -> Path:
    result = run_capture(
        ["rustc", f"+{args.toolchain}", "--print", "sysroot"],
        cwd=REPO_ROOT,
        env=spike_env(args),
    )
    if result.returncode != 0:
        raise RustCosmoError(format_command_failure(result))
    return Path(result.stdout.strip())


def rust_std_unix_error_path(args: argparse.Namespace) -> Path:
    return (
        rust_sysroot(args)
        / "lib"
        / "rustlib"
        / "src"
        / "rust"
        / "library"
        / "std"
        / "src"
        / "sys"
        / "io"
        / "error"
        / "unix.rs"
    )


def rust_std_linux_random_path(args: argparse.Namespace) -> Path:
    return (
        rust_sysroot(args)
        / "lib"
        / "rustlib"
        / "src"
        / "rust"
        / "library"
        / "std"
        / "src"
        / "sys"
        / "random"
        / "linux.rs"
    )


def rust_std_kernel_copy_mod_path(args: argparse.Namespace) -> Path:
    return (
        rust_sysroot(args)
        / "lib"
        / "rustlib"
        / "src"
        / "rust"
        / "library"
        / "std"
        / "src"
        / "sys"
        / "io"
        / "kernel_copy"
        / "mod.rs"
    )


def rust_std_socket_mod_path(args: argparse.Namespace) -> Path:
    return (
        rust_sysroot(args)
        / "lib"
        / "rustlib"
        / "src"
        / "rust"
        / "library"
        / "std"
        / "src"
        / "sys"
        / "net"
        / "connection"
        / "socket"
        / "mod.rs"
    )


def cosmo_errno_std_patch() -> str:
    return r"""// ROCM_CLI_COSMO_ERRNO_PATCH_BEGIN
#[cfg(target_vendor = "cosmo")]
unsafe extern "C" {
    #[link_name = "E2BIG"]
    static COSMO_E2BIG: c_int;
    #[link_name = "EACCES"]
    static COSMO_EACCES: c_int;
    #[link_name = "EADDRINUSE"]
    static COSMO_EADDRINUSE: c_int;
    #[link_name = "EADDRNOTAVAIL"]
    static COSMO_EADDRNOTAVAIL: c_int;
    #[link_name = "EAGAIN"]
    static COSMO_EAGAIN: c_int;
    #[link_name = "EBUSY"]
    static COSMO_EBUSY: c_int;
    #[link_name = "ECONNABORTED"]
    static COSMO_ECONNABORTED: c_int;
    #[link_name = "ECONNREFUSED"]
    static COSMO_ECONNREFUSED: c_int;
    #[link_name = "ECONNRESET"]
    static COSMO_ECONNRESET: c_int;
    #[link_name = "EDEADLK"]
    static COSMO_EDEADLK: c_int;
    #[link_name = "EDQUOT"]
    static COSMO_EDQUOT: c_int;
    #[link_name = "EEXIST"]
    static COSMO_EEXIST: c_int;
    #[link_name = "EFBIG"]
    static COSMO_EFBIG: c_int;
    #[link_name = "EHOSTUNREACH"]
    static COSMO_EHOSTUNREACH: c_int;
    #[link_name = "EINPROGRESS"]
    static COSMO_EINPROGRESS: c_int;
    #[link_name = "EINTR"]
    static COSMO_EINTR: c_int;
    #[link_name = "EINVAL"]
    static COSMO_EINVAL: c_int;
    #[link_name = "EISDIR"]
    static COSMO_EISDIR: c_int;
    #[link_name = "ELOOP"]
    static COSMO_ELOOP: c_int;
    #[link_name = "EMLINK"]
    static COSMO_EMLINK: c_int;
    #[link_name = "ENAMETOOLONG"]
    static COSMO_ENAMETOOLONG: c_int;
    #[link_name = "ENETDOWN"]
    static COSMO_ENETDOWN: c_int;
    #[link_name = "ENETUNREACH"]
    static COSMO_ENETUNREACH: c_int;
    #[link_name = "ENOMEM"]
    static COSMO_ENOMEM: c_int;
    #[link_name = "ENOENT"]
    static COSMO_ENOENT: c_int;
    #[link_name = "ENOSPC"]
    static COSMO_ENOSPC: c_int;
    #[link_name = "ENOSYS"]
    static COSMO_ENOSYS: c_int;
    #[link_name = "ENOTCONN"]
    static COSMO_ENOTCONN: c_int;
    #[link_name = "ENOTDIR"]
    static COSMO_ENOTDIR: c_int;
    #[link_name = "ENOTEMPTY"]
    static COSMO_ENOTEMPTY: c_int;
    #[link_name = "EOPNOTSUPP"]
    static COSMO_EOPNOTSUPP: c_int;
    #[link_name = "EPERM"]
    static COSMO_EPERM: c_int;
    #[link_name = "EPIPE"]
    static COSMO_EPIPE: c_int;
    #[link_name = "EROFS"]
    static COSMO_EROFS: c_int;
    #[link_name = "ESPIPE"]
    static COSMO_ESPIPE: c_int;
    #[link_name = "ESTALE"]
    static COSMO_ESTALE: c_int;
    #[link_name = "ETIMEDOUT"]
    static COSMO_ETIMEDOUT: c_int;
    #[link_name = "ETXTBSY"]
    static COSMO_ETXTBSY: c_int;
    #[link_name = "EXDEV"]
    static COSMO_EXDEV: c_int;
    #[link_name = "EWOULDBLOCK"]
    static COSMO_EWOULDBLOCK: c_int;
}

#[cfg(target_vendor = "cosmo")]
#[inline]
fn cosmo_errno_value(value: c_int) -> i32 {
    value as i32
}

#[cfg(target_vendor = "cosmo")]
#[inline]
fn is_cosmo_interrupted(errno: i32) -> bool {
    unsafe { errno == cosmo_errno_value(COSMO_EINTR) }
}

#[cfg(target_vendor = "cosmo")]
fn decode_cosmo_error_kind(errno: i32) -> Option<io::ErrorKind> {
    use io::ErrorKind::*;
    let kind = unsafe {
        if errno == cosmo_errno_value(COSMO_E2BIG) {
            ArgumentListTooLong
        } else if errno == cosmo_errno_value(COSMO_EADDRINUSE) {
            AddrInUse
        } else if errno == cosmo_errno_value(COSMO_EADDRNOTAVAIL) {
            AddrNotAvailable
        } else if errno == cosmo_errno_value(COSMO_EBUSY) {
            ResourceBusy
        } else if errno == cosmo_errno_value(COSMO_ECONNABORTED) {
            ConnectionAborted
        } else if errno == cosmo_errno_value(COSMO_ECONNREFUSED) {
            ConnectionRefused
        } else if errno == cosmo_errno_value(COSMO_ECONNRESET) {
            ConnectionReset
        } else if errno == cosmo_errno_value(COSMO_EDEADLK) {
            Deadlock
        } else if errno == cosmo_errno_value(COSMO_EDQUOT) {
            QuotaExceeded
        } else if errno == cosmo_errno_value(COSMO_EEXIST) {
            AlreadyExists
        } else if errno == cosmo_errno_value(COSMO_EFBIG) {
            FileTooLarge
        } else if errno == cosmo_errno_value(COSMO_EHOSTUNREACH) {
            HostUnreachable
        } else if errno == cosmo_errno_value(COSMO_EINTR) {
            Interrupted
        } else if errno == cosmo_errno_value(COSMO_EINVAL) {
            InvalidInput
        } else if errno == cosmo_errno_value(COSMO_EISDIR) {
            IsADirectory
        } else if errno == cosmo_errno_value(COSMO_ELOOP) {
            FilesystemLoop
        } else if errno == cosmo_errno_value(COSMO_ENOENT) {
            NotFound
        } else if errno == cosmo_errno_value(COSMO_ENOMEM) {
            OutOfMemory
        } else if errno == cosmo_errno_value(COSMO_ENOSPC) {
            StorageFull
        } else if errno == cosmo_errno_value(COSMO_ENOSYS) {
            Unsupported
        } else if errno == cosmo_errno_value(COSMO_EMLINK) {
            TooManyLinks
        } else if errno == cosmo_errno_value(COSMO_ENAMETOOLONG) {
            InvalidFilename
        } else if errno == cosmo_errno_value(COSMO_ENETDOWN) {
            NetworkDown
        } else if errno == cosmo_errno_value(COSMO_ENETUNREACH) {
            NetworkUnreachable
        } else if errno == cosmo_errno_value(COSMO_ENOTCONN) {
            NotConnected
        } else if errno == cosmo_errno_value(COSMO_ENOTDIR) {
            NotADirectory
        } else if errno == cosmo_errno_value(COSMO_ENOTEMPTY) {
            DirectoryNotEmpty
        } else if errno == cosmo_errno_value(COSMO_EPIPE) {
            BrokenPipe
        } else if errno == cosmo_errno_value(COSMO_EROFS) {
            ReadOnlyFilesystem
        } else if errno == cosmo_errno_value(COSMO_ESPIPE) {
            NotSeekable
        } else if errno == cosmo_errno_value(COSMO_ESTALE) {
            StaleNetworkFileHandle
        } else if errno == cosmo_errno_value(COSMO_ETIMEDOUT) {
            TimedOut
        } else if errno == cosmo_errno_value(COSMO_ETXTBSY) {
            ExecutableFileBusy
        } else if errno == cosmo_errno_value(COSMO_EXDEV) {
            CrossesDevices
        } else if errno == cosmo_errno_value(COSMO_EINPROGRESS) {
            InProgress
        } else if errno == cosmo_errno_value(COSMO_EOPNOTSUPP) {
            Unsupported
        } else if errno == cosmo_errno_value(COSMO_EACCES)
            || errno == cosmo_errno_value(COSMO_EPERM)
        {
            PermissionDenied
        } else if errno == cosmo_errno_value(COSMO_EAGAIN)
            || errno == cosmo_errno_value(COSMO_EWOULDBLOCK)
        {
            WouldBlock
        } else {
            return None;
        }
    };
    Some(kind)
}
// ROCM_CLI_COSMO_ERRNO_PATCH_END
"""


def cosmo_random_std_patch() -> str:
    return r"""// ROCM_CLI_COSMO_RANDOM_PATCH_BEGIN
#[cfg(target_vendor = "cosmo")]
unsafe extern "C" {
    #[link_name = "getrandom"]
    fn cosmo_getrandom(
        buffer: *mut libc::c_void,
        length: libc::size_t,
        flags: libc::c_uint,
    ) -> libc::ssize_t;
}

#[cfg(target_vendor = "cosmo")]
fn cosmo_fill_random_bytes(bytes: &mut [u8]) {
    let mut remaining = bytes;
    while !remaining.is_empty() {
        let read = unsafe {
            cosmo_getrandom(remaining.as_mut_ptr().cast(), remaining.len(), 0)
        };
        if read <= 0 {
            panic!("failed to generate random data");
        }
        remaining = &mut remaining[read as usize..];
    }
}
// ROCM_CLI_COSMO_RANDOM_PATCH_END
"""


def patch_rust_std_for_cosmopolitan(args: argparse.Namespace) -> None:
    path = rust_std_unix_error_path(args)
    if not path.is_file():
        raise RustCosmoError(f"rust-src std unix error file was not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "ROCM_CLI_COSMO_ERRNO_PATCH_BEGIN" not in text:
        anchor = "#[inline]\npub fn is_interrupted(errno: i32) -> bool {\n"
        if anchor not in text:
            raise RustCosmoError(f"could not locate is_interrupted anchor in {path}")
        text = text.replace(anchor, cosmo_errno_std_patch() + "\n" + anchor, 1)
    old_interrupted = "pub fn is_interrupted(errno: i32) -> bool {\n    errno == libc::EINTR\n}\n"
    new_interrupted = (
        "pub fn is_interrupted(errno: i32) -> bool {\n"
        "    #[cfg(target_vendor = \"cosmo\")]\n"
        "    {\n"
        "        return is_cosmo_interrupted(errno);\n"
        "    }\n"
        "    #[cfg(not(target_vendor = \"cosmo\"))]\n"
        "    {\n"
        "        errno == libc::EINTR\n"
        "    }\n"
        "}\n"
    )
    if old_interrupted in text:
        text = text.replace(old_interrupted, new_interrupted, 1)
    elif new_interrupted not in text:
        raise RustCosmoError(f"could not patch is_interrupted in {path}")
    old_decode = "pub fn decode_error_kind(errno: i32) -> io::ErrorKind {\n    use io::ErrorKind::*;\n    match errno as libc::c_int {\n"
    new_decode = (
        "pub fn decode_error_kind(errno: i32) -> io::ErrorKind {\n"
        "    use io::ErrorKind::*;\n"
        "    #[cfg(target_vendor = \"cosmo\")]\n"
        "    if let Some(kind) = decode_cosmo_error_kind(errno) {\n"
        "        return kind;\n"
        "    }\n"
        "    match errno as libc::c_int {\n"
    )
    if old_decode in text:
        text = text.replace(old_decode, new_decode, 1)
    elif new_decode not in text:
        raise RustCosmoError(f"could not patch decode_error_kind in {path}")
    path.write_text(text, encoding="utf-8", newline="\n")

    random_path = rust_std_linux_random_path(args)
    if not random_path.is_file():
        raise RustCosmoError(f"rust-src std linux random file was not found: {random_path}")
    random_text = random_path.read_text(encoding="utf-8")
    if "ROCM_CLI_COSMO_RANDOM_PATCH_BEGIN" in random_text:
        start = random_text.index("// ROCM_CLI_COSMO_RANDOM_PATCH_BEGIN")
        end_marker = "// ROCM_CLI_COSMO_RANDOM_PATCH_END"
        end = random_text.index(end_marker, start) + len(end_marker)
        random_text = random_text[:start] + cosmo_random_std_patch().rstrip() + random_text[end:]
    else:
        anchor = "use crate::sys::pal::weak::syscall;\n"
        if anchor not in random_text:
            raise RustCosmoError(f"could not locate random import anchor in {random_path}")
        random_text = random_text.replace(anchor, anchor + "\n" + cosmo_random_std_patch(), 1)
    old_fill = "pub fn fill_bytes(bytes: &mut [u8]) {\n    getrandom(bytes, false);\n}\n"
    new_fill = (
        "pub fn fill_bytes(bytes: &mut [u8]) {\n"
        "    #[cfg(target_vendor = \"cosmo\")]\n"
        "    {\n"
        "        cosmo_fill_random_bytes(bytes);\n"
        "        return;\n"
        "    }\n"
        "    #[cfg(not(target_vendor = \"cosmo\"))]\n"
        "    {\n"
        "        getrandom(bytes, false);\n"
        "    }\n"
        "}\n"
    )
    if old_fill in random_text:
        random_text = random_text.replace(old_fill, new_fill, 1)
    elif new_fill not in random_text:
        raise RustCosmoError(f"could not patch fill_bytes in {random_path}")
    old_hashmap = (
        "pub fn hashmap_random_keys() -> (u64, u64) {\n"
        "    let mut bytes = [0; 16];\n"
        "    getrandom(&mut bytes, true);\n"
        "    let k1 = u64::from_ne_bytes(bytes[..8].try_into().unwrap());\n"
        "    let k2 = u64::from_ne_bytes(bytes[8..].try_into().unwrap());\n"
        "    (k1, k2)\n"
        "}\n"
    )
    new_hashmap = (
        "pub fn hashmap_random_keys() -> (u64, u64) {\n"
        "    let mut bytes = [0; 16];\n"
        "    #[cfg(target_vendor = \"cosmo\")]\n"
        "    cosmo_fill_random_bytes(&mut bytes);\n"
        "    #[cfg(not(target_vendor = \"cosmo\"))]\n"
        "    getrandom(&mut bytes, true);\n"
        "    let k1 = u64::from_ne_bytes(bytes[..8].try_into().unwrap());\n"
        "    let k2 = u64::from_ne_bytes(bytes[8..].try_into().unwrap());\n"
        "    (k1, k2)\n"
        "}\n"
    )
    if old_hashmap in random_text:
        random_text = random_text.replace(old_hashmap, new_hashmap, 1)
    elif new_hashmap not in random_text:
        raise RustCosmoError(f"could not patch hashmap_random_keys in {random_path}")
    random_path.write_text(random_text, encoding="utf-8", newline="\n")

    kernel_copy_path = rust_std_kernel_copy_mod_path(args)
    if not kernel_copy_path.is_file():
        raise RustCosmoError(f"rust-src std kernel_copy module was not found: {kernel_copy_path}")
    kernel_copy_text = kernel_copy_path.read_text(encoding="utf-8")
    old_kernel_copy = (
        "cfg_select! {\n"
        "    any(target_os = \"linux\", target_os = \"android\") => {\n"
        "        mod linux;\n"
        "        pub use linux::kernel_copy;\n"
        "    }\n"
        "    _ => {\n"
    )
    new_kernel_copy = (
        "cfg_select! {\n"
        "    target_vendor = \"cosmo\" => {\n"
        "        use crate::io::{Result, Read, Write};\n"
        "\n"
        "        pub fn kernel_copy<R: ?Sized, W: ?Sized>(_reader: &mut R, _writer: &mut W) -> Result<CopyState>\n"
        "        where\n"
        "            R: Read,\n"
        "            W: Write,\n"
        "        {\n"
        "            Ok(CopyState::Fallback(0))\n"
        "        }\n"
        "    }\n"
        "    any(target_os = \"linux\", target_os = \"android\") => {\n"
        "        mod linux;\n"
        "        pub use linux::kernel_copy;\n"
        "    }\n"
        "    _ => {\n"
    )
    if old_kernel_copy in kernel_copy_text:
        kernel_copy_text = kernel_copy_text.replace(old_kernel_copy, new_kernel_copy, 1)
    elif new_kernel_copy not in kernel_copy_text:
        raise RustCosmoError(f"could not patch kernel_copy fallback in {kernel_copy_path}")
    kernel_copy_path.write_text(kernel_copy_text, encoding="utf-8", newline="\n")

    socket_path = rust_std_socket_mod_path(args)
    if not socket_path.is_file():
        raise RustCosmoError(f"rust-src std socket module was not found: {socket_path}")
    socket_text = socket_path.read_text(encoding="utf-8")
    old_socket = "#[cfg(not(windows))]\n            unsafe {\n                setsockopt(&sock, c::SOL_SOCKET, c::SO_REUSEADDR, 1 as c_int)?\n            };"
    new_socket = "#[cfg(not(any(windows, target_vendor = \"cosmo\")))]\n            unsafe {\n                setsockopt(&sock, c::SOL_SOCKET, c::SO_REUSEADDR, 1 as c_int)?\n            };"
    if old_socket in socket_text:
        socket_text = socket_text.replace(old_socket, new_socket, 1)
    elif new_socket not in socket_text:
        raise RustCosmoError(f"could not patch SO_REUSEADDR guard in {socket_path}")
    socket_path.write_text(socket_text, encoding="utf-8", newline="\n")


def write_spike_files(work_root: Path) -> tuple[Path, Path]:
    linker = work_root / "gcc-linker-wrapper.bash"
    target = work_root / "x86_64-unknown-linux-cosmo.json"
    write_text(linker, linker_wrapper())
    linker.chmod(0o755)
    write_text(target, target_json(linker.resolve()))
    return target, linker


def cargo_config() -> str:
    return """[unstable]
build-std-features = [""]
build-std = ["libc", "panic_abort", "std"]

[profile.dev]
panic = "abort"
opt-level = "s"

[profile.release]
panic = "abort"
opt-level = "s"
"""


def create_hello_project(project_dir: Path) -> None:
    write_text(
        project_dir / "Cargo.toml",
        """[package]
name = "rocm-rust-cosmo-hello"
version = "0.0.0"
edition = "2024"

[workspace]

[profile.dev]
panic = "abort"
opt-level = "s"

[profile.release]
panic = "abort"
opt-level = "s"
""",
    )
    write_text(
        project_dir / "src" / "main.rs",
        """fn main() {
    println!("hello from Rust std on Cosmopolitan");
}
""",
    )
    write_text(project_dir / ".cargo" / "config.toml", cargo_config())


def rustup(args: argparse.Namespace, rustup_args: Iterable[str]) -> CommandResult:
    env = spike_env(args)
    args.rustup_home.mkdir(parents=True, exist_ok=True)
    args.cargo_home.mkdir(parents=True, exist_ok=True)
    return run_capture(["rustup", *rustup_args], cwd=REPO_ROOT, env=env)


def toolchain_usable(args: argparse.Namespace) -> bool:
    env = spike_env(args)
    cargo_result = run_capture(["cargo", f"+{args.toolchain}", "--version"], cwd=REPO_ROOT, env=env)
    rustc_result = run_capture(["rustc", f"+{args.toolchain}", "--version"], cwd=REPO_ROOT, env=env)
    return cargo_result.returncode == 0 and rustc_result.returncode == 0


def cargo(args: argparse.Namespace, cargo_args: Iterable[str], *, cwd: Path) -> CommandResult:
    env = spike_env(args)
    return run_capture(["cargo", *cargo_args], cwd=cwd, env=env)


def ensure_toolchain(args: argparse.Namespace) -> None:
    require_posix_host()
    result = rustup(
        args,
        [
            "toolchain",
            "install",
            args.toolchain,
            "--profile",
            "minimal",
            "--component",
            "rust-src",
        ],
    )
    if result.returncode != 0:
        if not toolchain_usable(args):
            raise RustCosmoError(format_command_failure(result))
        print("rust-cosmopolitan spike: rustup returned a non-zero status, but the requested toolchain is usable")
    print(f"rust-cosmopolitan spike: toolchain ready: {args.toolchain}")


def build_hello(args: argparse.Namespace) -> Path:
    require_posix_host()
    check_cosmocc_root(args.cosmocc_root.resolve())
    work_root = args.work_root.resolve()
    if args.clean and work_root.exists():
        remove_tree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    prepare_linker_toolchain(args)
    target, _linker = write_spike_files(work_root)
    project_dir = work_root / "hello-std"
    if project_dir.exists():
        remove_tree(project_dir)
    create_hello_project(project_dir)

    build_args = [
        f"+{args.toolchain}",
        "-Zjson-target-spec",
        "build",
        "--target",
        path_text(target),
    ]
    if args.release:
        build_args.append("--release")
    result = cargo(args, build_args, cwd=project_dir)
    if result.returncode != 0:
        write_text(work_root / "hello-build.stdout.log", result.stdout)
        write_text(work_root / "hello-build.stderr.log", result.stderr)
        raise RustCosmoError(format_command_failure(result))

    profile = "release" if args.release else "debug"
    artifact_dir = project_dir / "target" / "x86_64-unknown-linux-cosmo" / profile
    candidates = sorted(artifact_dir.glob("rocm-rust-cosmo-hello*.com.dbg"))
    if not candidates:
        raise RustCosmoError(f"Cargo build finished but no .com.dbg artifact was found in {artifact_dir}")
    artifact = candidates[0]
    ape = work_root / ("rocm-rust-cosmo-hello-release.exe" if args.release else "rocm-rust-cosmo-hello.exe")
    apelink = args.cosmocc_root.resolve() / "bin" / "apelink"
    ape_loader = args.cosmocc_root.resolve() / "bin" / "ape-x86_64.elf"
    result = run_capture(
        [
            path_text(apelink),
            "-l",
            path_text(ape_loader),
            "-o",
            path_text(ape),
            path_text(artifact),
        ],
        cwd=work_root,
        env=spike_env(args),
    )
    if result.returncode != 0:
        write_text(work_root / "hello-apelink.stdout.log", result.stdout)
        write_text(work_root / "hello-apelink.stderr.log", result.stderr)
        raise RustCosmoError(format_command_failure(result))
    ape.chmod(0o755)
    print(f"rust-cosmopolitan spike: hello ELF artifact: {artifact}")
    print(f"rust-cosmopolitan spike: hello APE artifact: {ape}")
    return ape


def build_rocm(args: argparse.Namespace) -> Path:
    require_posix_host()
    check_cosmocc_root(args.cosmocc_root.resolve())
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    prepare_linker_toolchain(args)
    target, _linker = write_spike_files(work_root)
    target_dir = work_root / "rocm-target"
    if args.clean and target_dir.exists():
        remove_tree(target_dir)

    build_args = [
        f"+{args.toolchain}",
        "-Zjson-target-spec",
        "-Zbuild-std=std,panic_abort",
        "-Zbuild-std-features=",
        "build",
        "--package",
        "rocm",
        "--bin",
        "rocm",
        "--target",
        path_text(target),
    ]
    if args.release:
        build_args.append("--release")
    if args.jobs:
        build_args.extend(["--jobs", str(args.jobs)])

    env = spike_env(args)
    env["CARGO_TARGET_DIR"] = path_text(target_dir)
    result = run_capture(["cargo", *build_args], cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        write_text(work_root / "rocm-build.stdout.log", result.stdout)
        write_text(work_root / "rocm-build.stderr.log", result.stderr)
        raise RustCosmoError(format_command_failure(result))

    profile = "release" if args.release else "debug"
    elf = target_dir / "x86_64-unknown-linux-cosmo" / profile / "rocm.com.dbg"
    if not elf.is_file():
        raise RustCosmoError(f"Cargo build finished but no rocm ELF artifact was found at {elf}")

    ape = work_root / ("rocm-rust-cosmo-release.exe" if args.release else "rocm-rust-cosmo.exe")
    apelink = args.cosmocc_root.resolve() / "bin" / "apelink"
    ape_loader = args.cosmocc_root.resolve() / "bin" / "ape-x86_64.elf"
    link_result = run_capture(
        [
            path_text(apelink),
            "-l",
            path_text(ape_loader),
            "-o",
            path_text(ape),
            path_text(elf),
        ],
        cwd=work_root,
        env=spike_env(args),
    )
    if link_result.returncode != 0:
        write_text(work_root / "rocm-apelink.stdout.log", link_result.stdout)
        write_text(work_root / "rocm-apelink.stderr.log", link_result.stderr)
        raise RustCosmoError(format_command_failure(link_result))
    ape.chmod(0o755)
    if args.release:
        release_ape = DEFAULT_RELEASE_ROOT / "rocm.exe"
        release_ape.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ape, release_ape)
        release_ape.chmod(0o755)
        print(f"rust-cosmopolitan spike: rocm release artifact: {release_ape}")
    print(f"rust-cosmopolitan spike: rocm ELF artifact: {elf}")
    print(f"rust-cosmopolitan spike: rocm APE artifact: {ape}")
    return ape


def run_hello(args: argparse.Namespace) -> None:
    artifact = build_hello(args)
    env = spike_env(args)
    result = run_capture([path_text(artifact)], cwd=artifact.parent, env=env)
    if result.returncode != 0:
        raise RustCosmoError(format_command_failure(result))
    print(result.stdout.strip())


def probe(args: argparse.Namespace) -> None:
    print("Rust/Cosmopolitan spike probe")
    print(f"  host: {sys.platform}")
    print(f"  work_root: {args.work_root.resolve()}")
    print(f"  rustup_home: {args.rustup_home.resolve()}")
    print(f"  cargo_home: {args.cargo_home.resolve()}")
    print(f"  cosmocc_root: {args.cosmocc_root.resolve()}")
    print(f"  toolchain: {args.toolchain}")
    try:
        check_cosmocc_root(args.cosmocc_root.resolve())
        print("  cosmocc: present")
    except RustCosmoError as error:
        print(f"  cosmocc: missing ({error})")
    env = spike_env(args)
    commands = [
        ("rustup", ["rustup", "--version"]),
        ("cargo", ["cargo", f"+{args.toolchain}", "--version"]),
        ("rustc", ["rustc", f"+{args.toolchain}", "--version"]),
    ]
    for name, command in commands:
        result = run_capture(command, cwd=REPO_ROOT, env=env)
        status = result.stdout.strip() or result.stderr.strip() or f"exit {result.returncode}"
        print(f"  {name}: {status}")


def expected_rocm_ape(args: argparse.Namespace, release: bool) -> Path:
    if release:
        return args.work_root.resolve() / "rocm-rust-cosmo-release.exe"
    return args.work_root.resolve() / "rocm-rust-cosmo.exe"


def smoke_wsl_linux_path(args: argparse.Namespace) -> None:
    require_posix_host()
    artifact = args.artifact.resolve() if args.artifact else expected_rocm_ape(args, args.release)
    if not artifact.is_file():
        raise RustCosmoError(f"single-file rocm APE was not found: {artifact}")

    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="wsl-linux-path-", dir=work_root))
    try:
        env = spike_env(args)
        env.update(
            {
                "ROCM_CLI_CONFIG_DIR": path_text(temp_root / "config"),
                "ROCM_CLI_DATA_DIR": path_text(temp_root / "data"),
                "ROCM_CLI_CACHE_DIR": path_text(temp_root / "cache"),
                "NO_COLOR": "1",
                "TERM": "xterm-256color",
            }
        )
        env.pop("VIRTUAL_ENV", None)
        result = run_capture(["sh", path_text(artifact), "doctor"], cwd=temp_root, env=env)
        combined = result.stdout + result.stderr
        if result.returncode != 0:
            raise RustCosmoError(format_command_failure(result))
        if "os: linux" not in combined:
            raise RustCosmoError(f"WSL/Linux smoke did not report the Linux runtime path:\n{combined}")
        if "os: windows" in combined:
            raise RustCosmoError(f"WSL/Linux smoke accidentally reported the Windows runtime path:\n{combined}")
        if host_is_wsl() and "wsl: true" not in combined:
            raise RustCosmoError(f"WSL smoke did not identify WSL:\n{combined}")

        print(f"rust-cosmopolitan spike: WSL/Linux path smoke passed for {artifact}")
        print("  launch: sh <same rocm APE> doctor")
        print("  observed: os: linux")
        if host_is_wsl():
            print("  observed: wsl: true")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_self_test() -> None:
    with_dir = DEFAULT_WORK_ROOT / "self-test"
    target, linker = write_spike_files(with_dir)
    target_text = target.read_text(encoding="utf-8")
    linker_text = linker.read_text(encoding="utf-8")
    assert "x86_64-unknown-linux-musl" in target_text
    assert "${ARCH}-unknown-cosmo-cc" in linker_text
    assert "ROCM_CLI_RUST_COSMO_ROOT" in linker_text
    assert "ROCM_CLI_RUST_COSMO_WAITID_OBJECT" not in linker_text
    assert "ROCM_CLI_RUST_COSMO_COMPAT_OBJECT" not in linker_text
    assert "-Wl,--allow-multiple-definition" not in linker_text
    assert "-lunwind" in linker_text
    waitid_text = cosmopolitan_waitid_source()
    assert "int waitid(" in waitid_text
    assert "getrandom" not in waitid_text
    assert "epoll" not in waitid_text
    assert "eventfd" not in waitid_text
    create_hello_project(with_dir / "hello")
    assert (with_dir / "hello" / "Cargo.toml").is_file()
    assert "[workspace]" in (with_dir / "hello" / "Cargo.toml").read_text(encoding="utf-8")
    expected_cosmocc_root = (
        DEFAULT_COSMOCC_WSL_ELF_ROOT
        if host_is_wsl() or cosmocc_root_has_minimal_tools(DEFAULT_COSMOCC_WSL_ELF_ROOT)
        else DEFAULT_COSMOCC_NATIVE_ROOT
    )
    assert default_cosmocc_root() == expected_cosmocc_root
    print("rust-cosmopolitan spike self-test: ok")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--rustup-home", type=Path, default=DEFAULT_RUSTUP_HOME)
    parser.add_argument("--cargo-home", type=Path, default=DEFAULT_CARGO_HOME)
    parser.add_argument("--cosmocc-root", type=Path, default=default_cosmocc_root())
    parser.add_argument("--cosmopolitan-source-root", type=Path, default=DEFAULT_COSMOPOLITAN_SOURCE_ROOT)
    parser.add_argument("--toolchain", default=DEFAULT_TOOLCHAIN)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Print local spike environment details.")
    add_common_args(probe_parser)

    install = subparsers.add_parser("install-toolchain", help="Install nightly + rust-src into the configured local rustup home.")
    add_common_args(install)

    build = subparsers.add_parser("build-hello", help="Build the Rust std hello APE fixture.")
    add_common_args(build)
    build.add_argument("--release", action="store_true")
    build.add_argument("--clean", action="store_true")

    build_rocm_parser = subparsers.add_parser("build-rocm", help="Attempt to build the real rocm binary as a Rust APE.")
    add_common_args(build_rocm_parser)
    build_rocm_parser.add_argument("--release", action="store_true")
    build_rocm_parser.add_argument("--clean", action="store_true")
    build_rocm_parser.add_argument("--jobs", type=int, default=0)

    smoke_wsl = subparsers.add_parser(
        "smoke-wsl-linux-path",
        help="Run the built rocm APE through the POSIX/WSL launch path and assert it reports Linux.",
    )
    add_common_args(smoke_wsl)
    smoke_wsl.add_argument("--artifact", type=Path)
    smoke_wsl.add_argument("--release", action="store_true")

    run = subparsers.add_parser("run-hello", help="Build and run the Rust std hello APE fixture.")
    add_common_args(run)
    run.add_argument("--release", action="store_true")
    run.add_argument("--clean", action="store_true")

    subparsers.add_parser("self-test", help="Run offline script-generation checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "probe":
            probe(args)
            return 0
        if args.command == "install-toolchain":
            ensure_toolchain(args)
            return 0
        if args.command == "build-hello":
            build_hello(args)
            return 0
        if args.command == "build-rocm":
            build_rocm(args)
            return 0
        if args.command == "smoke-wsl-linux-path":
            smoke_wsl_linux_path(args)
            return 0
        if args.command == "run-hello":
            run_hello(args)
            return 0
        if args.command == "self-test":
            run_self_test()
            return 0
    except (RustCosmoError, OSError, subprocess.SubprocessError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
