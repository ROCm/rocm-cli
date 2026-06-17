# CI hardware (GPU / WSL) testing — planned

The hosted CI (`ubuntu-latest`, `windows-latest`) builds and unit-tests every
shipping target natively. By design it cannot exercise two things:

- **Real AMD GPU execution** — GitHub-hosted runners have no AMD GPU.
- **Real WSL behaviour** — the Windows↔WSL interop path needs an actual WSL host.

A dedicated hardware-test layer covers exactly those gaps. It is **not** part of
the CI workflow yet; it will be introduced in a follow-up PR. This document
records the intended design so it can be reviewed and wired up as a unit.

## Design

1. **Build once on hosted runners.** The hosted `build-and-test` (Linux) and
   `windows-build-and-test` (Windows) jobs publish their per-OS binaries
   (`rocm`, `rocmd`, `rocm-engine-*`) as workflow artifacts.
2. **Test on dedicated self-hosted runners.** Hardware-test jobs download those
   exact artifacts and run only the checks hosted runners cannot: `rocm examine`
   host/GPU detection, engine `detect`/`capabilities`, the no-CPU-fallback
   smoke (`scripts/smoke_local.py --skip-build`), and the
   `scripts/*_therock_gpu_test.py` end-to-end GPU harnesses.

### Targets

| Target | Runner | Notes |
|---|---|---|
| Pure Windows 11 (gfx1151) | self-hosted Windows + AMD GPU | consumes the Windows artifact |
| AMD Instinct, bare-metal | self-hosted Linux + Instinct GPU | consumes the Linux artifact |
| Ubuntu on WSL (primary) | self-hosted Windows + WSL | consumes the Linux artifact in WSL |
| Fedora on WSL (secondary) | self-hosted Windows + WSL | builds in-WSL to match the distro's glibc |

### Guards

Each hardware-test job is:

- **Opt-in** via a repository/org variable (`ENABLE_WIN11_GPU_CI`,
  `ENABLE_INSTINCT_CI`, `ENABLE_WSL_UBUNTU_CI`, `ENABLE_WSL_FEDORA_CI`) so it is
  enabled per target as each runner is wired into the repo.
- **Non-blocking** (`continue-on-error: true`) — a hardware result never gates a PR.
- **Fork-safe** — self-hosted runners do not execute untrusted fork PRs.

## Notes

- These jobs run **debug** binaries: they assert functional behaviour (device
  detection, engine launch, policy enforcement), not performance, so the
  debug-vs-release difference does not affect what they check. Release-fidelity,
  optimized validation already lives in the nightly/release pipeline, which
  builds `manylinux2014` (glibc 2.17) binaries.
- Linux artifacts are built on `ubuntu-latest`; a consuming runner on an older
  distro must have a compatible (≥) glibc, otherwise it should build in-place
  (as the Fedora-on-WSL target does).

The layer will be enabled end-to-end in a follow-up PR once the self-hosted
runners are connected to the repository and each path has been validated against
real hardware.
