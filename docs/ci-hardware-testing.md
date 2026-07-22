<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# CI hardware (GPU / WSL) testing

The E2E matrix covers native Linux, native Windows, and hosted WSL2. GitHub-hosted
runners have no AMD GPU, so real GPU execution remains on dedicated self-hosted
runners.

## Platforms

Each job runs the same cucumber-rs suite. Capability tags resolve scenarios to
pass, expected failure, or not applicable for that host, and `e2e-report`
consolidates the resulting platform reports.

| Job | Platform | Runner |
|---|---|---|
| `e2e` | Mock (no GPU) | GitHub-hosted Ubuntu |
| `e2e-wsl` | Ubuntu on WSL2 (no GPU) | GitHub-hosted Windows |
| `e2e-gpu` | MI300X | self-hosted Linux + AMD GPU |
| `e2e-gpu-strix-ubuntu` | Strix Halo (gfx1151) on Ubuntu | self-hosted Linux + AMD GPU |
| `e2e-gpu-strix-windows` | Strix Halo (gfx1151) on native Windows 11 | self-hosted Windows + AMD GPU |

The hosted `e2e-wsl` job registers an Ubuntu distro under WSL2, builds the CLI
inside it, and runs the black-box suite. It exercises real WSL detection and the
Windows-to-WSL execution boundary. GPU-only scenarios skip because the hosted
runner has no AMD GPU.

This is intentionally distinct from GPU-on-WSL validation. The hosted job cannot
exercise `/dev/dxg`, the Windows AMD driver, ROCDXG, or live model serving on an
AMD GPU. Those paths require a self-hosted Windows machine with both an AMD GPU
and a configured WSL distro.

## Triggers and blocking behavior

The hosted WSL and hardware jobs run automatically for heavy changes after the
hosted build succeeds. They can also be selected with `workflow_dispatch` using
`platform=wsl`, `app-dev-gpu`, `strix-ubuntu`, or `strix-windows`.

Only the hosted Linux `e2e` job is blocking. `e2e-wsl` and all three hardware
jobs use `continue-on-error: true`, so they provide advisory platform coverage
while their failures remain visible in the consolidated report.

Self-hosted runners are not used for untrusted fork pull requests. Manual
workflow dispatch requires repository write access.

## Notes

- E2E jobs build and run release binaries. They assert functional behavior such
  as host detection, engine launch, and policy enforcement; they are not
  performance benchmarks.
- `e2e-wsl` has its own platform slug even without a GPU, so its results do not
  collide with the ordinary mock report.
- A future GPU-on-WSL lane should remain a separate, non-blocking platform until
  its runner, distro, and GPU access have been validated end to end.
