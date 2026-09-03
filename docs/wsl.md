<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# WSL Support Notes

This note tracks the WSL path for `rocm-cli` with TheRock-managed Python
virtual environments and ROCDXG (`librocdxg`).

## Prerequisites

The AMD WSL path is:

1. Windows 11 with the AMD Adrenalin WSL-capable driver.
2. WSL2 with Ubuntu 24.04 or newer. Ubuntu 22.04 is not supported: it ships
   glibc 2.35, below the glibc 2.38 / `GLIBCXX_3.4.32` floor that every
   published Lemonade embeddable requires, so the Lemonade engine cannot start
   there. Ubuntu 24.04 provides glibc 2.39 and `GLIBCXX_3.4.33`.
3. ROCDXG (`librocdxg`) installed inside WSL.
4. A TheRock runtime installed by `rocm-cli` into a managed Python venv.

Useful read-only preflight, from inside the distro:

```bash
rocm diagnose
rocm diagnose --json
```

From Windows PowerShell, inspect a distro without installing anything in it:

```powershell
rocm diagnose --distro          # the only distro installed
rocm diagnose --distro Ubuntu   # a named one
```

The host-side form collects the facts over `wsl.exe` and runs the same catalog.
It needs no `rocm-cli`, and no Python, inside the target distro — which is the
point, since the distro being checked is usually the one that is not set up yet.

It sees less than a run from inside, so prefer the in-distro form where you can:

- It probes the conventional ROCm roots (`/opt/rocm*`, `/usr/local/rocm*`) but
  cannot honour a `$ROCM_PATH` pointing elsewhere. `wsl.exe --exec` runs a
  non-login, non-interactive shell, so nothing exported from a shell profile is
  set.
- For the same reason it collects no environment, so the checks that read one —
  `HSA_OVERRIDE_GFX_VERSION`, `PATH`, and the framework/ROCm version pairing —
  do not run. It reports on the WSL GPU stack, not on the whole installation.

## Install ROCDXG In WSL

Install build/runtime prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git cmake build-essential python3 python3-venv
```

Preferred package install for the current public release:

```bash
curl -L -o /tmp/rocdxg-roct_1.2.0_amd64.deb \
  https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
sudo apt install -y /tmp/rocdxg-roct_1.2.0_amd64.deb
sudo ldconfig
```

From this repo inside WSL, the same supported path is wrapped as:

```bash
bash scripts/wsl_setup_rocdxg.sh
rocm diagnose
```

To require checksum verification before installing the downloaded `.deb`, set
`ROCDXG_SHA256` to the trusted 64-character SHA-256 digest for that exact
ROCDXG package:

```bash
ROCDXG_SHA256=<64-hex-sha256> bash scripts/wsl_setup_rocdxg.sh
```

The wrapper intentionally does not guess or embed a production checksum. If
`ROCDXG_SHA256` is set and the downloaded package does not match, installation
stops before `apt install`.

Source-build alternative:

```bash
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
export win_sdk="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0"
if [ -f "${win_sdk}/shared/dxcore_interface.h" ]; then
  win_sdk_include="${win_sdk}/shared"
elif [ -f "${win_sdk}/um/dxcore_interface.h" ]; then
  win_sdk_include="${win_sdk}/um"
else
  echo "DXCore headers were not found under ${win_sdk}" >&2
  exit 1
fi
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk_include}"
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

For legacy ROCm releases, set:

```bash
export HSA_ENABLE_DXG_DETECTION=1
```

ROCK/TheRock 7.13 and newer should not require that variable, but setting it is
still useful for compatibility checks while the WSL path is being hardened.

## TheRock Runtime Env In WSL

`rocm-cli` should continue to own the Python venv. Do not rely on an externally
created venv such as `D:\ROCm\venv`.

The WSL activation environment for HIP applications that do not preload ROCm
the way PyTorch does must include the managed TheRock runtime package paths and
WSL DXCore path. With the managed `rocm[libraries,devel]` install, these paths
come from the rocm-cli runtime manifest, `rocm-sdk path --root`, and
`rocm_sdk.find_libraries(...)`.

```bash
export ROCM_ROOT="<managed _rocm_sdk_core or devel root from the runtime manifest>"
export ROCM_PATH="${ROCM_ROOT}"
export ROCM_HOME="${ROCM_ROOT}"
export HIP_PATH="${ROCM_ROOT}"
export PATH="<managed TheRock bin dirs>:${PATH}"
export LD_LIBRARY_PATH="<managed TheRock library dirs>:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

For `rocm-cli`, the command itself should resolve the managed runtime manifest
and apply that environment before launching HIP apps such as Lemonade's bundled
`llama.cpp` backend. Users should not have to hand-export these values.

## Diagnosing A WSL Host

`rocm diagnose` carries a WSL catalog, separate from the bare-metal Linux one.
The bare-metal checks (render group, `/dev/kfd`, `modprobe amdgpu`, `iommu=pt`)
never run here: WSL has no `amdgpu` module and no `/dev/kfd`, so a finding
naming one would send you after a fault that cannot exist on this platform.

```bash
rocm diagnose
rocm diagnose --json
```

The WSL entries, in the order a broken stack usually reveals them:

| Fix id | Reported when |
| --- | --- |
| `fix-wsl-7-wsl1` | The distro runs under WSL 1, which has no GPU path at all |
| `fix-wsl-1-gpu-not-exposed` | `/dev/dxg` is missing. Names which of the three causes applies: a container started without the device, a distro with no WSL GPU support wired in, or the Windows host driver |
| `fix-wsl-2-dxcore-missing` | `libdxcore.so` is absent or off the loader path |
| `fix-wsl-3-rocdxg-missing` | ROCDXG is not installed in the distro |
| `fix-wsl-4-rocdxg-not-linked` | ROCDXG is installed but absent from the linker cache |
| `fix-wsl-5-distro-too-old` | The distro release is below the floor in the prerequisites above |
| `fix-wsl-6-host-driver-too-old` | The distro-side plumbing is complete but the Windows host driver is missing or too old |

Every WSL remedy is print-only. `rocm fix <id>` shows the commands and does not
run them: they either install packages with `sudo`, edit loader configuration, or
belong to the Windows host, and none of that meets the bar the four
auto-applicable fixes clear.

Two deliberate silences, so a report can be trusted:

- `fix-wsl-6` **abstains** when WSL interop cannot reach the Windows host, rather
  than reading "could not ask" as "driver is too old". Inside a container, or
  with interop switched off, the host driver is reported as unknown and no
  finding blames it.
- `fix-wsl-5` **abstains** when the distro release cannot be parsed. An
  unreadable release is not evidence of a supported one, but it is not evidence
  of an old one either.

## What `rocm examine` Reports On WSL

`rocm examine` detects WSL cheaply and reports:

- `wsl: true`
- WSL distro/version, and whether the release clears the supported floor
- WSL major version (1 or 2)
- `/dev/dxg` presence
- `/usr/lib/wsl/lib/libdxcore.so` presence
- `librocdxg.so` presence, resolved across every ROCm install rather than
  assuming `/opt/rocm`
- `librocdxg` linker-cache visibility from `ldconfig -p`
- the Windows host AMD driver version, when WSL interop can reach the host
- whether `HSA_ENABLE_DXG_DETECTION` is set
- managed TheRock runtime count and active/default runtime

The driver, device-node and group probes stay skipped, so `has_amd_gpu` and
`gpus` describe the bare-metal view and are not populated here.

## Install UX Recommendations

`rocm install sdk` inside WSL should:

- default to a managed pip venv, same as native Linux and Windows
- avoid installing global WSL packages unless the user explicitly approves
- fail clearly when WSL GPU prerequisites are absent
- after install, validate `python -m rocm_sdk version`,
  `python -m rocm_sdk targets`, runtime library discovery through
  `rocm_sdk.find_libraries`, and at least one HIP-visible GPU probe when
  ROCDXG is ready

`rocm setup` in WSL should offer a staged plan:

1. Verify WSL/DXCore.
2. Explain missing ROCDXG if absent.
3. Ask before any `sudo apt install` or `sudo make install`.
4. Install TheRock into a managed venv.
5. Install a serving engine (Lemonade or vLLM).
6. Run a tiny GPU smoke test with CPU fallback disabled.

## Non-Destructive Tests

Safe tests that do not mutate global WSL state:

- `rocm diagnose --json`
- `rocm diagnose --distro <name>` from the Windows host
- `cargo test -p rocm-core wsl` for the catalog's own tests
- `rocm install sdk --channel release --format wheel --dry-run` inside WSL with
  isolated `ROCM_CLI_*` directories

Gated tests that require explicit opt-in because they may download packages or
need `sudo`:

- install ROCDXG `.deb`
- build ROCDXG from source
- install TheRock wheels into a fresh managed WSL venv
- install a serving engine (Lemonade or vLLM)
- run tiny inference on GPU
