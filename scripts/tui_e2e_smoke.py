#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PTY-driven ROCm CLI TUI smoke tests.

These checks exercise the real terminal UI in an isolated temporary ROCm CLI
state root. They are intentionally safe: no TheRock install, model download,
ComfyUI launch, or GPU workload is started.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import pyte

REPO_ROOT = Path(__file__).resolve().parents[1]


class TuiSmokeError(Exception):
    pass


class TuiSession:
    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        columns: int = 118,
        rows: int = 34,
    ) -> None:
        self.columns = columns
        self.rows = rows
        self.screen = pyte.Screen(columns, rows)
        self.stream = pyte.Stream(self.screen)
        self._queue: queue.Queue[str | BaseException] = queue.Queue()
        self._closed = False
        if os.name == "nt":
            import winpty

            self._kind = "winpty"
            self._process = winpty.PtyProcess.spawn(
                command,
                cwd=str(cwd),
                env=env,
                dimensions=(rows, columns),
            )
            self._reader = threading.Thread(target=self._read_winpty, daemon=True)
        else:
            import fcntl
            import pty
            import struct
            import termios

            master_fd, slave_fd = pty.openpty()
            winsize = struct.pack("HHHH", rows, columns, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
            self._kind = "pty"
            self._master_fd = master_fd
            self._process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
            os.close(slave_fd)
            self._reader = threading.Thread(target=self._read_pty, daemon=True)
        self._reader.start()

    def _read_winpty(self) -> None:
        while True:
            try:
                data = self._process.read(4096)
            except EOFError:
                return
            except BaseException as error:  # pragma: no cover - defensive reader thread
                self._queue.put(error)
                return
            if data:
                self._queue.put(data)

    def _read_pty(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self._master_fd], [], [], 0.2)
                if not ready:
                    if self._process.poll() is not None:
                        return
                    continue
                data = os.read(self._master_fd, 4096)
            except OSError:
                return
            except BaseException as error:  # pragma: no cover - defensive reader thread
                self._queue.put(error)
                return
            if not data:
                return
            self._queue.put(data.decode("utf-8", errors="replace"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._kind == "winpty":
                if self._process.isalive():
                    self._process.terminate(force=True)
            else:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                os.close(self._master_fd)
        except Exception:
            pass

    def write(self, text: str) -> None:
        if self._kind == "winpty":
            self._process.write(text)
        else:
            os.write(self._master_fd, text.encode("utf-8"))

    def wait_for(self, needles: str | Iterable[str], *, timeout: float = 12.0) -> str:
        if isinstance(needles, str):
            needles = [needles]
        needles = list(needles)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain(timeout=0.2)
            rendered = self.rendered()
            if all(needle in rendered for needle in needles):
                return rendered
        raise TuiSmokeError(
            "timed out waiting for "
            + ", ".join(repr(needle) for needle in needles)
            + "\n\nLast screen:\n"
            + self.rendered()
        )

    def wait_not_contains(self, needle: str, *, settle: float = 0.8) -> str:
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            self._drain(timeout=0.1)
        rendered = self.rendered()
        if needle in rendered:
            raise TuiSmokeError(f"unexpected {needle!r} on screen\n\n{rendered}")
        return rendered

    def _drain(self, *, timeout: float) -> None:
        end = time.monotonic() + timeout
        while True:
            remaining = max(0.0, end - time.monotonic())
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                return
            if isinstance(item, BaseException):
                raise TuiSmokeError(f"PTY reader failed: {item}") from item
            self.stream.feed(item)
            if time.monotonic() >= end:
                return

    def rendered(self) -> str:
        return "\n".join(self.screen.display)


def main() -> int:
    args = parse_args()
    rocm = resolve_rocm(args.rocm)
    if not rocm.is_file():
        raise SystemExit(f"rocm executable not found: {rocm}")
    run_setup_smoke(rocm, args)
    run_main_tui_smoke(rocm, args)
    print("[tui-e2e-smoke] ok")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", default=str(default_rocm()))
    parser.add_argument("--columns", type=int, default=118)
    parser.add_argument("--rows", type=int, default=34)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def default_rocm() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "target" / "debug" / f"rocm{suffix}"


def resolve_rocm(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def isolated_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_CLI_CONFIG_DIR"] = str(root / "config")
    env["ROCM_CLI_DATA_DIR"] = str(root / "data")
    env["ROCM_CLI_CACHE_DIR"] = str(root / "cache")
    env["NO_COLOR"] = "1"
    env.pop("VIRTUAL_ENV", None)
    return env


def run_setup_smoke(rocm: Path, args: argparse.Namespace) -> None:
    with temp_state("rocm-cli-tui-setup-", args.keep_temp) as root:
        session = TuiSession(
            [str(rocm)],
            cwd=REPO_ROOT,
            env=isolated_env(root),
            columns=args.columns,
            rows=args.rows,
        )
        try:
            screen = session.wait_for(
                ["Set Up ROCm", "Install folder:", "Install ROCm"]
            )
            require(
                "Message" not in screen,
                "setup screen should not show the chat prompt box",
            )
            session.write("\x1b[A")
            session.wait_for("Install folder:")
            session.write("\r")
            session.wait_for(["Choose ROCm Folder", "Use current folder"])
            session.write("\x1b")
            session.wait_for(["Set Up ROCm", "Install ROCm"])
            session.write("?")
            session.wait_not_contains("Command List")
            session.write("\x1b")
            session.wait_for(["Quit setup", "Install folder:"])
        finally:
            session.close()


def run_main_tui_smoke(rocm: Path, args: argparse.Namespace) -> None:
    with temp_state("rocm-cli-tui-main-", args.keep_temp) as root:
        seed_completed_config(root)
        session = TuiSession(
            [str(rocm)],
            cwd=REPO_ROOT,
            env=isolated_env(root),
            columns=args.columns,
            rows=args.rows,
        )
        try:
            screen = session.wait_for(["ROCm CLI", "Choose"])
            require(
                "Message" not in screen,
                "main menu should not show a chat prompt before chat starts",
            )
            session.write("\x1bOP")
            session.wait_for(["Help", "Keyboard"])
            session.write("\x1b[6~")
            session.write("\x1b[<65;10;10M")
            session.wait_for("Help")
            session.write("\x1b")
            session.wait_not_contains("Keyboard shortcuts")
            session.write("/")
            session.wait_for(["Command", "/"])
            session.write("\t")
            session.wait_for("/")
            session.write("\x1b")
            session.wait_not_contains("Completions")
            session.write("\x1b")
            session.wait_for(["Quit", "ROCm CLI"])
        finally:
            session.close()


def seed_completed_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "onboarding_dismissed": True,
                "setup": {"completed": True},
                "default_engine": "lemonade",
                "permissions": {"mode": "ask_first"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class temp_state:
    def __init__(self, prefix: str, keep: bool) -> None:
        self.prefix = prefix
        self.keep = keep
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.keep or self.path is None:
            if self.path is not None:
                print(f"[tui-e2e-smoke] kept temp state: {self.path}")
            return
        shutil.rmtree(self.path, ignore_errors=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TuiSmokeError(message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TuiSmokeError, OSError, subprocess.SubprocessError) as error:
        print(f"[tui-e2e-smoke] failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
