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

Useful read-only preflight:

```bash
python scripts/wsl_preflight.py
python scripts/wsl_preflight.py --json
python scripts/wsl_preflight.py --require-ready
```

From Windows PowerShell, target a specific distro:

```powershell
python scripts\wsl_preflight.py --distro Ubuntu
```

## Install ROCDXG In WSL

Use the CLI:

```bash
rocm install driver            # print the plan
rocm install driver --yes      # run it
```

On WSL2 this installs ROCDXG rather than Linux DKMS — there is no in-tree
amdgpu driver to build, because the GPU comes from the Windows host driver
through `/dev/dxg`. The plan is printed for review first and only runs with
`--yes`, the same as on bare metal. It checks `/dev/dxg` and dxcore before
touching anything, installs the release package, runs `ldconfig`, and then
verifies that `/opt/rocm/lib/librocdxg.so` exists and is visible to the linker.
No reboot is needed: ROCDXG is userspace.

Confirm afterwards with `rocm examine`, which should report
`driver_status: wsl_rocdxg_ready`.

To install a release other than the pinned one, set `ROCM_CLI_ROCDXG_VERSION`.
To require checksum verification before the package is installed, set
`ROCM_CLI_ROCDXG_SHA256` to the trusted 64-character SHA-256 digest for that
exact package:

```bash
ROCM_CLI_ROCDXG_SHA256=<64-hex-sha256> rocm install driver --yes
```

No production checksum is embedded, so verification is opt-in; when the
variable is set and the download does not match, the install stops before
`apt install`.

### Doing it by hand

The equivalent manual steps, for reference or for a host where the CLI is not
installed yet:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
curl -L -o /tmp/rocdxg-roct_1.2.0_amd64.deb \
  https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
sudo apt install -y /tmp/rocdxg-roct_1.2.0_amd64.deb
sudo ldconfig
```

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

## Examine And Install UX Recommendations

`rocm examine` should detect WSL cheaply and report:

- `wsl: true`
- WSL distro/version
- `/dev/dxg` presence
- `/usr/lib/wsl/lib/libdxcore.so` presence
- `/opt/rocm/lib/librocdxg.so` presence
- `librocdxg` linker-cache visibility from `ldconfig -p`
- `rocminfo` from the active TheRock runtime after activation
- whether `HSA_ENABLE_DXG_DETECTION` is needed or set
- managed TheRock runtime count and active/default runtime

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

- `python scripts/wsl_preflight.py --self-test`
- `python scripts/wsl_preflight.py --json`
- `python scripts/wsl_preflight.py --require-ready` on a prepared WSL machine
- `rocm install sdk --channel release --format wheel --dry-run` inside WSL with
  isolated `ROCM_CLI_*` directories

Gated tests that require explicit opt-in because they may download packages or
need `sudo`:

- install ROCDXG `.deb`
- build ROCDXG from source
- install TheRock wheels into a fresh managed WSL venv
- install a serving engine (Lemonade or vLLM)
- run tiny inference on GPU
