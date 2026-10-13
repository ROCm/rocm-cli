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
   glibc 2.35, below the minimum that every published Lemonade embeddable
   requires (glibc 2.38 and `GLIBCXX_3.4.32`), so the Lemonade engine cannot
   start there. Ubuntu 24.04 provides glibc 2.39 and `GLIBCXX_3.4.33`.
3. ROCDXG (`librocdxg`) installed inside WSL.
4. A TheRock runtime installed by ROCm CLI into a managed Python venv.

To run a read-only preflight check from inside the distro:

```bash
rocm diagnose
rocm diagnose --json
```

To check a distro from Windows without installing anything in it, see
[Diagnosing a WSL host](#diagnosing-a-wsl-host) later on this page.

## Install ROCDXG In WSL

Use the CLI:

```bash
rocm install driver            # print the plan
rocm install driver --yes      # run it
```

On WSL2 this installs ROCDXG rather than Linux DKMS. There is no in-tree
amdgpu driver to build, because the GPU comes from the Windows host driver
through `/dev/dxg`. The plan is printed for review first and only runs with
`--yes`, the same as on bare metal. It checks `/dev/dxg` and dxcore before
touching anything, installs the release package, runs `ldconfig`, and then
verifies that `/opt/rocm/lib/librocdxg.so` exists and is visible to the linker.
No reboot is needed: ROCDXG is userspace.

Confirm afterwards with
`rocm examine`, which should report
`driver_status: wsl_rocdxg_ready`.

The download is verified before it is installed. `rocm install driver` carries a
SHA-256 digest for each published ROCDXG release and checks the `.deb` against
it; on a mismatch the plan stops before `apt install` runs the package's
maintainer scripts as root. Nothing needs to be set for this.

To install a release other than the pinned one, set `ROCM_CLI_ROCDXG_VERSION`.
Because ROCm CLI has no digest for a release it predates, supply one:

```bash
ROCM_CLI_ROCDXG_VERSION=<version> \
ROCM_CLI_ROCDXG_SHA256=<64-hex-sha256> rocm install driver --yes
```

`ROCM_CLI_ROCDXG_SHA256` also overrides the pinned digest for a known version.
To install without verifying, you must opt in explicitly. The plan then prints
a warning that names the version it does not check:

```bash
ROCM_CLI_ROCDXG_VERSION=<version> \
ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED=1 rocm install driver --yes
```

Without one of those, an unpinned version is refused rather than installed
unverified.

### Doing it by hand

The equivalent manual steps, for reference or for a host where the CLI is not
installed yet. Do not skip the `sha256sum -c` line, because `apt install` runs
the package's maintainer scripts as root. `rocm install driver` checks the
download against a digest pinned in the ROCm CLI source
(`ROCDXG_PINNED_DIGESTS` in `apps/rocm/src/driver_install.rs`), which also
catches a replaced release asset. A digest read from the release page at
install time catches only a corrupted download. For a version listed in
`ROCDXG_PINNED_DIGESTS`, use the pinned digest. Set `ROCDXG_VERSION` to the
release you want and `ROCDXG_SHA256` to its digest:

```bash
ROCDXG_VERSION=<version>
ROCDXG_SHA256=<64-hex-sha256>
sudo apt-get update
sudo apt-get install -y ca-certificates curl
curl -L -o "/tmp/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" \
  "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb"
printf '%s  %s\n' "${ROCDXG_SHA256}" \
  "/tmp/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" | sha256sum -c -
sudo apt install -y "/tmp/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb"
sudo ldconfig
```

Source-build alternative. Set `win_sdk` to the Windows SDK version installed on
your host; the version in this example may differ from yours:

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

If you use a legacy ROCm release instead of the managed TheRock runtime, set:

```bash
export HSA_ENABLE_DXG_DETECTION=1
```

TheRock 7.13 and newer should not require this variable, but setting it on a
newer release is harmless.

## TheRock Runtime Env In WSL

`rocm-cli` should continue to own the Python venv. Do not rely on an externally
created venv such as `D:\ROCm\venv`.

The WSL activation environment for HIP applications that do not preload ROCm
the way PyTorch does must include the managed TheRock runtime package paths and
WSL DXCore path. With the managed `rocm[libraries]` install, these paths
come from the rocm-cli runtime manifest, `rocm-sdk path --root`, and
`rocm_sdk.find_libraries(...)`. `rocm[libraries]` is enough to run a pre-built
HIP application; compiling one from source under WSL needs the compiler and
headers, so run `rocm install sdk --devel` instead.

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

`rocm diagnose` has a separate set of
WSL checks, called the WSL catalog, distinct from the bare-metal Linux ones.
The bare-metal checks (render group, `/dev/kfd`, `modprobe amdgpu`, `iommu=pt`)
never run here: WSL has no `amdgpu` module and no `/dev/kfd`, so a finding
naming one would send you after a fault that cannot exist on this platform.

Run `rocm diagnose`, or `rocm diagnose --json` for machine-readable output.

The WSL catalog does not check for build tools or Python `venv` support, so a
distro missing only those reports clean here.

The WSL entries, in the order a broken stack usually reveals them:

| Fix id | Reported when |
| --- | --- |
| `fix-wsl-7-wsl1` | The distro runs under WSL 1, which has no GPU path at all |
| `fix-wsl-1-gpu-not-exposed` | `/dev/dxg` is missing. Names which of the three causes applies: a container started without the device, a distro with no WSL GPU support wired in, or the Windows host driver |
| `fix-wsl-2-dxcore-missing` | `libdxcore.so` is absent or off the loader path |
| `fix-wsl-3-rocdxg-missing` | ROCDXG is not installed in the distro |
| `fix-wsl-4-rocdxg-not-linked` | ROCDXG is installed but absent from the linker cache |
| `fix-wsl-5-distro-too-old` | The distro release is below the floor in the [prerequisites](#prerequisites) above |
| `fix-wsl-6-host-driver-too-old` | The distro-side plumbing is complete but the Windows host driver is missing or too old |

Seven entries that are not WSL-specific also run on WSL: `fix-1-arch`,
`fix-2-unset-override`, `fix-6-path`, `fix-8-wheel-rocm`, `fix-16-vllm-oom`,
`fix-18-comgr-conflict`, and `fix-19-shm-too-small`. `fix-16-vllm-oom` is driven
by the error text you pass with `--symptom` and reads nothing from the
environment or the ROCm install. `fix-18-comgr-conflict` and
`fix-19-shm-too-small` apply when you run `rocm diagnose` inside the distro. A
`--distro` run skips both (see
[Diagnose a distro from Windows](#diagnose-a-distro-from-windows) later in this
section).

`fix-19-shm-too-small` is not a WSL entry, but WSL2 ships the same 64 MiB `/dev/shm` a container does, and a
serving workload needs gigabytes of it. It reports below 1 GiB, so a WSL2 user
can meet it on an otherwise healthy stack. Its guidance names the container and
bare-metal cases; under WSL2 the remedy is the host one, remounting `/dev/shm`
larger and adding the matching `/etc/fstab` line inside the distro.

`fix-18-comgr-conflict` also applies on WSL2: a wheel copy and a system copy of
the code object manager library collide there exactly as they do on bare
metal, through the same `LD_LIBRARY_PATH`/loader-cache search, so the finding
is not specific to either platform.

Every WSL remedy is print-only.
`rocm fix <id>` shows the commands and
does not run them: they either install packages with `sudo`, edit loader configuration, or
belong to the Windows host, and none of that meets the bar an auto-applied fix
has to clear. On WSL the CLI carries out exactly one catalog entry itself,
`fix-6-path`, which is not one of the WSL entries.

Two checks deliberately stay silent when they cannot be sure, so you can trust
the report:

- `fix-wsl-6` **abstains** when WSL interop cannot reach the Windows host, rather
  than reading "could not ask" as "driver is too old". Inside a container, or
  with interop switched off, the host driver is reported as unknown and no
  finding blames it. It also abstains on WSL 1, which has no GPU path for a
  driver to serve.
- `fix-wsl-5` **abstains** when the distro release cannot be parsed. An
  unreadable release is not evidence of a supported one, but it is not evidence
  of an old one either. It also abstains on any distro other than Ubuntu, because
  the floor is defined only for Ubuntu.

### Diagnose a distro from Windows

To inspect a distro without installing anything in it, install ROCm CLI on
Windows (see the Windows steps in the
[installation guide](../README.md#windows-x86_64-powershell))
and run this from PowerShell:

```powershell
rocm diagnose --distro          # the only distro installed
rocm diagnose --distro Ubuntu   # a named distro
```

Name the distro whenever more than one is installed. Without a name, the
command stops and lists the installed distros.

ROCm CLI runs on Windows and calls `wsl.exe` to run a short read-only script
inside the distro and collect diagnostic data. It then checks that data against
the WSL catalog, except for the checks that need data it cannot collect.
Nothing is installed in the distro, and neither ROCm CLI nor Python needs to be
there. That matters because the distro you are checking is usually not set up
yet.

The script runs under `/bin/sh -c` through `wsl.exe --exec`. That shell is
non-login and non-interactive, so it never reads your shell profile, and it
collects a fixed set of data. This leaves four limits:

- **Fixed ROCm locations.** The script looks for the ROCDXG files only in the
  conventional ROCm roots (`/opt/rocm*`, `/usr/local/rocm*`) and cannot honor a
  `$ROCM_PATH` set elsewhere.
- **`rocminfo` on the default PATH only.** The script finds `rocminfo` with
  `command -v`, and the default WSL `PATH` has neither `/opt/rocm/bin` nor a
  TheRock virtual environment. When the script does not find `rocminfo`, the
  `fix-wsl-6` case "plumbing complete but no GPU enumerated" cannot fire, so
  `--distro` can miss a host-driver fault that `rocm diagnose` inside the
  distro reports.
- **No environment checks.** The script collects no environment, so the checks
  that read one do not run: `HSA_OVERRIDE_GFX_VERSION`, `PATH`, and the
  framework and ROCm version pairing.
- **No library or shared-memory checks.** The script does not collect the data
  that `fix-18-comgr-conflict` and `fix-19-shm-too-small` need, so those two do
  not run either.

`--distro` therefore reports on the WSL GPU stack only. For a full diagnosis,
run `rocm diagnose` inside the distro. `--distro` also cannot be combined with
`--model` or `--report`, because both describe the machine running the command.

If the command reports `could not inspect '<name>'; the distribution may be
stopped or unreachable`, `wsl.exe` could not run the script in the distro. To
see WSL's own error message, run:

```powershell
wsl -d <name> --exec /bin/sh -c "echo ok"
```

Resolve that error first, then run `rocm diagnose --distro` again.

## What `rocm examine` Reports On WSL

`rocm examine` detects WSL cheaply and reports the following. The text report
shows a subset. Four items appear only in `rocm examine --json`, and are marked
below.

- `wsl: true`
- WSL distro and version
- whether the release clears the supported floor (`--json` only)
- WSL major version, 1 or 2 (`--json` only)
- `/dev/dxg` presence
- `/usr/lib/wsl/lib/libdxcore.so` presence
- `librocdxg.so` presence, resolved across every ROCm install rather than
  assuming `/opt/rocm`
- `librocdxg` linker-cache visibility from `ldconfig -p`
- the Windows host AMD driver version, when WSL interop can reach the host
  (`--json` only)
- whether `HSA_ENABLE_DXG_DETECTION` is set (`--json` only)
- managed TheRock runtime count and active default runtime

The driver, device-node, and group probes stay skipped. In the `--json` output,
`has_amd_gpu` is therefore always false and `gpus` is empty on WSL, even on a
healthy host, because both describe the bare-metal view.

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
