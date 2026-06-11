# Cosmopolitan Universal Binary Plan

This document defines what "single universal binary" means for rocm-cli and
separates it from the retired payload-delegating APE launcher spike.

## Definitions

- **Platform-native standalone**: the Rust `rocm` or `rocm.exe` binary built for
  one OS. This remains useful for development, but it is not the target
  release shape for the single-exe work.
- **Payload-delegating APE launcher**: a retired compatibility spike that
  appended platform-specific release payloads to a small launcher and delegated
  to one of them at runtime. It is not the final universal binary and is no
  longer active in this tree.
- **True no-extract APE**: the actual rocm-cli program is compiled and linked as
  one Cosmopolitan executable. The executable can run on supported OSes without
  unpacking separate `rocm.exe`/`rocm` binaries first.

The target product requirement is the third shape.

## Upstream Facts

- Cosmopolitan Libc is for C/C++ programs and reconfigures GCC/Clang to emit an
  Actually Portable Executable that runs across supported OSes without a VM:
  https://github.com/jart/cosmopolitan
- APE loading maps the executable into memory. If the system APE loader is not
  installed, the embedded loader may be copied to a small `.ape` helper path,
  but the normal APE model is not "extract this app into OS-specific payloads":
  https://justine.lol/apeloader/
- Rust targets are selected with `--target`, and built-in targets are visible
  through `rustc --print target-list`:
  https://doc.rust-lang.org/rustc/targets/index.html
- Rust custom targets exist, but the Rust book says the target JSON/build-std
  path is unstable and must pin the compiler:
  https://doc.rust-lang.org/rustc/targets/custom.html
- Independent Rust/Cosmopolitan experiments exist, including `std` experiments,
  but they are not an official Rust target and are not a drop-in guarantee for
  this repo:
  https://ahgamut.github.io/2022/07/27/ape-rust-example/

Local probe on 2026-06-05:

```text
rustc 1.98.0-nightly (2026-06-04) does not report a built-in target containing
cosmo, cosmopolitan, or ape.
```

## Current Repo Reality

The repo currently has these runnable artifact paths:

- `scripts/build_single_exe_release.py standalone`
  - builds/copies the platform-native Rust `rocm` or `rocm.exe`;
  - no extraction;
  - not universal across OSes.

As of 2026-06-06, the repo has a true Rust/Cosmopolitan feasibility
builder:

- `scripts/setup-cosmocc.sh`
  - downloads `cosmocc` under `.rocm-work/tools/cosmocc`;
  - creates an executable ELF-converted toolchain under
    `.rocm-work/tools/cosmocc-wsl-elf` when the host needs it, including WSL
    and native Linux containers without APE `execve` support.
- `scripts/rust_cosmopolitan_spike.py install-toolchain`
  - installs a workspace-local nightly Rust toolchain plus `rust-src` under
    `.rocm-work/tools`.
- `scripts/rust_cosmopolitan_spike.py build-rocm --release --clean --jobs 96`
  - builds the real Rust `rocm` binary as a no-extract Cosmopolitan APE;
  - writes `.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe`.

The Rust APE currently uses a Linux-shaped custom Rust target so `std` can link
against Cosmopolitan. Product code must use `rocm_core::RuntimePlatform` for
runtime OS decisions; direct `cfg!(windows)` and `std::env::consts::OS` are
compile-target facts and are not enough for a universal APE.

Current Rust `std::process` also compiles Linux pidfd support for this custom
target. Rust's pidfd code references the public POSIX `waitid` symbol. The
downloaded Cosmopolitan toolchain exposes the lower-level `sys_waitid` syscall
entry but not the public libc wrapper, so `scripts/rust_cosmopolitan_spike.py`
now patches the workspace-local Cosmopolitan source tree under `.rocm-work` and
archives a real `libc/proc/waitid.c` implementation into `libcosmo.a` before
building rocm-cli. The linker wrapper does not inject rocm-cli-local syscall
objects or compatibility aliases.

Validated clean rebuild and E2E checkpoint on 2026-06-06:

- `.rocm-work` was deleted, then recreated by the setup/build scripts above.
- Revalidated after removing the old linker-injected waitid object path:
  `libcosmo.a` now exports `waitid` from the Cosmopolitan source-level patch,
  and the generated linker wrapper contains no `ROCM_CLI_RUST_COSMO_WAITID_OBJECT`,
  no `ROCM_CLI_RUST_COSMO_COMPAT_OBJECT`, and no `--allow-multiple-definition`.
- Native Windows smoke:
  - `version` prints `rocm 0.2.0`;
  - `doctor` reports `os: windows`, AMD Ryzen Threadripper PRO 9995WX,
    Radeon RX 9070 XT, `detected_gfx_target: gfx1201`,
    `compatible_therock_family: gfx120X-all`, and AMD display driver
    `32.0.23033.1002`.
- WSL smoke through `sh <same APE> doctor` reports `os: linux`, `wsl: true`,
  `driver_policy: wsl_rocdxg`, `driver_status: wsl_rocdxg_ready`, and
  `detected_gfx_target: gfx1201`.
- WSL Linux-path smoke currently uses `sh <same APE> doctor`. Direct WSL
  execution after renaming the same file to `rocm` is still a caveat: if it
  reports `os: windows`, WSLInterop intercepted the file before rocm-cli
  started.
- Windows and WSL TUI smoke pass with the same universal artifact.
- Windows and WSL Lemonade local-assistant E2E pass with no CPU/Vulkan fallback.
- Windows and WSL PyTorch local-assistant E2E pass with no CPU fallback.
- Windows and WSL ComfyUI install/start/status/stop E2E pass through the same
  universal artifact using temporary state roots.

WSL caveat: WSLInterop registers an `MZ` binfmt handler that can intercept
direct `./rocm` execution before rocm-cli starts, even when the file has no
`.exe` extension. `sh ./rocm ...` proves the Linux payload works. Direct
`./rocm` without WSLInterop interception remains the desired user experience.

Current release shape: first-party engine adapters and the small `rocmd`
service/tool helper surface are built into `rocm`. The universal APE no longer
requires sibling `rocm-engine-*` or `rocmd` files for doctor/setup, bridge
snapshots, service list/stop/restart helpers, foreground serving, or managed
serving launch. External Python environments, TheRock wheels, ComfyUI, ROCm
libraries, and models are still installed on disk as user-managed runtime
content.

## Implementation Tracks

### Track A: Native Cosmopolitan Bootstrap/Core

This is no longer the active path for `jam/updates`. The Rust/Cosmopolitan
artifact runs the real rocm-cli setup/TUI, so duplicating setup in C/C++ would
increase product drift.

Historical shape:

1. Detect OS and AMD GPU basics.
2. If AMD runtime/driver is missing, show simple driver guidance.
3. Provide the cross-platform folder picker.
4. Download or locate Python 3.12.10 when Python is missing.
5. Install TheRock wheels into the selected folder using rocm-cli's current
   package-selection rules.
6. Write the normal `~/.rocm` JSON config and runtime registry.

Keep this track only as a fallback if Rust/Cosmopolitan stops being viable.

### Track B: Rust-To-Cosmopolitan Feasibility

Prove whether the existing Rust CLI can become a true APE:

1. Done: pin a workspace-local nightly Rust toolchain.
2. Done: create a custom Linux-shaped `x86_64-unknown-linux-cosmo` target JSON.
3. Done: build a tiny Rust `std` APE with `cosmocc`.
4. Done: build the real `rocm` binary as one APE.
5. Done: smoke `version` and `doctor` on native Windows and WSL.
6. Done: no-arg first-time setup TUI opens from the APE on Windows in a PTY.
7. Partial: same artifact runs the Linux/WSL runtime path through
   `sh <same file>`; direct renamed `./rocm` remains a WSLInterop caveat.
8. Done: Windows and WSL universal-binary E2E passes for TUI smoke, Lemonade
   assistant, PyTorch assistant, and ComfyUI start/stop.
9. Remaining: graduate the spike script into the release pipeline, validate
   direct WSL execution, and validate native Linux outside WSL.

Acceptance for this track is no longer "hello world". The current artifact runs
useful rocm-cli doctor/setup code as one APE; the remaining blocker list is now
about release hardening and helper binaries, not Rust/Cosmopolitan feasibility.

### Track C: Collapse Multi-Binary Runtime Shape

A true no-extract rocm-cli cannot depend on sibling engine adapter executables
such as `rocm-engine-pytorch`, `rocm-engine-llama-cpp`, or `rocmd` unless those
are also compiled into the same executable or replaced by in-process modules.

Work items:

1. Done: inventory every spawned rocm-cli helper binary.
2. Done: first-party engines are linked into `rocm` and exposed through hidden
   internal `__engine-stdio` and `__engine-serve-http` routes.
3. Done: the `rocmd` status/bridge/sandbox service helper surface used by the
   TUI is implemented inside `rocm`; `rocm_core::daemon_binary_path()` now
   resolves to the current executable for same-file helper invocations.
4. Keep external Python environments, TheRock wheels, ComfyUI, models, and
   ROCm libraries as managed installed content. Those are runtime payloads the
   user asks rocm-cli to install, not part of the CLI executable itself.

## Acceptance Criteria

A true universal rocm-cli binary is accepted only when:

1. One produced file runs on Windows, WSL/Linux, and native Linux without
   extracting separate `rocm.exe` or `rocm` binaries.
2. The first-run setup/bootstrap UI opens directly.
3. The folder picker works cross-platform.
4. Python 3.12.10 can be installed or selected if missing.
5. TheRock installs into the user-selected folder.
6. Read-only system/GPU checks do not require approval.
7. Mutating install/uninstall/config operations still require clear approval.
8. The executable does not claim GPU readiness unless the AMD runtime needed by
   the selected workflow is actually available.

Current status on 2026-06-06: Windows execution and `sh <same APE>` WSL
execution pass for the local `gfx1201` host. Direct renamed `./rocm` execution
inside WSL remains a caveat when WSLInterop intercepts it. Native Linux
validation and production release-pipeline promotion remain open.

## Automation

Run the current feasibility probe:

```bash
python scripts/cosmopolitan_feasibility.py self-test
python scripts/cosmopolitan_feasibility.py probe
```

When a Cosmopolitan compiler is available:

```bash
python scripts/cosmopolitan_feasibility.py probe \
  --compiler .rocm-work/tools/cosmocc-wsl-elf/bin/cosmocc \
  --compile-c
```

The probe is intentionally small. It should fail loudly if repo wording drifts
back to calling any payload-delegating wrapper a true universal binary.

Build the current Rust APE from a clean workspace-local tool cache:

```bash
rm -rf .rocm-work
scripts/setup-cosmocc.sh
python3 scripts/rust_cosmopolitan_spike.py install-toolchain
python3 scripts/rust_cosmopolitan_spike.py build-rocm --release --clean --jobs 96
python3 scripts/rust_cosmopolitan_spike.py smoke-wsl-linux-path --release
```

The output is:

```text
.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe
```

For release packaging, copy that artifact to the user-facing name `rocm.exe`
on Windows. The same bytes may be copied to `rocm` for WSL/Linux smoke tests.
