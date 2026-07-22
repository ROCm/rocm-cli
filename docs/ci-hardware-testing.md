<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# CI hardware (GPU / WSL) testing

The hosted CI (`ubuntu-latest`, `windows-latest`) builds and unit-tests every
shipping target natively, but GitHub-hosted runners have no AMD GPU. A
dedicated hardware layer in `.github/workflows/ci.yml` covers that gap by
running the same cucumber-rs E2E suite on dedicated self-hosted runners with
real AMD GPUs.

## Platforms

The E2E suite (BDD scenarios in Gherkin `.feature` files backed by Rust step
functions) runs as one job per platform. Each job's harness resolves every
scenario to pass / xfail / skip for that host from its `@id` and
`@requires-*` tags, a capability probe, and `expectations.toml` — there is no
separate tier flag or tag filter to maintain.

| Job | Platform | Runner labels |
|---|---|---|
| `e2e` | Mock (no GPU) | GitHub-hosted `ubuntu-latest` |
| `e2e-gpu` | MI300X (AMD Instinct, bare-metal Linux) | self-hosted `[self-hosted, linux, amd-gpu]` |
| `e2e-gpu-strix-ubuntu` | Strix Halo (gfx1151) on Ubuntu | self-hosted `[self-hosted, linux, strix-halo]` |
| `e2e-gpu-strix-windows` | Strix Halo (gfx1151) on native Windows 11 | self-hosted `[self-hosted, windows, strix-halo]` |

`e2e` is the blocking, GitHub-hosted mock job: `@requires-gpu` scenarios
resolve to skip here, and known bugs resolve to xfail from
`expectations.toml`. It is a required check and must stay green.

The three GPU jobs (`e2e-gpu`, `e2e-gpu-strix-ubuntu`, `e2e-gpu-strix-windows`)
run on dedicated self-hosted runners with a real AMD GPU attached, so they
exercise host/GPU detection, engine `detect`/`capabilities`, and live serving
scenarios that the mock job cannot.

An `e2e-report` job consolidates every platform's report — including partial
or failed runs — into one HTML report and GitHub step summary, joined by
scenario id, so the (scenario × platform) expectation grid is visible in one
place.

## Triggers

The GPU jobs run automatically on `push`, `pull_request`, and `merge_group`
when both of the following hold:

- the `changes` job's `heavy` path filter is `true` (the change touches code
  that can affect runtime behavior, not just docs or unrelated files), and
- the hosted `build-and-test` job succeeded.

They can also be triggered manually via `workflow_dispatch`, independent of
the `heavy` gate, with these inputs:

- `platform` (choice: `all`, `mock`, `app-dev-gpu`, `strix-ubuntu`,
  `strix-windows`) — which job(s) to run. `mock` maps to `e2e`,
  `app-dev-gpu` to `e2e-gpu`, `strix-ubuntu` to `e2e-gpu-strix-ubuntu`, and
  `strix-windows` to `e2e-gpu-strix-windows`.
- `name_filter` (string) — a scenario-name regex forwarded to the cucumber
  harness (`cargo xtask e2e -- --name <regex>`) so a dispatch can run a
  single scenario instead of the full suite. Empty runs everything
  applicable to the selected platform(s).
- `include_nightly` (boolean, default `false`) — opts a dispatch into
  `@nightly`-tagged scenarios (e.g. large-model serves, cold installs) that
  are otherwise skipped on a normal push/PR run to keep it fast.

A manual dispatch skips the hosted `build-and-test` job for a faster loop; the
E2E jobs run directly against the dispatched ref.

## Blocking vs. non-blocking

Only `e2e` (the GitHub-hosted mock job) is a required, blocking check. The
three hardware jobs — `e2e-gpu`, `e2e-gpu-strix-ubuntu`, and
`e2e-gpu-strix-windows` — all run with `continue-on-error: true`, so a
hardware failure never gates a PR merge. Their results still surface in the
consolidated `e2e-report` for visibility.

## Fork safety

Self-hosted runners are not used for untrusted fork pull requests: GitHub
does not dispatch self-hosted-runner jobs from an external fork's
`pull_request` event without explicit approval. The hardware jobs only run
against branches and PRs within the repository (and on `workflow_dispatch`,
which requires write access to trigger).

## Notes

- The hardware jobs build and run **release** binaries: they assert
  functional behavior (device detection, engine launch, policy enforcement),
  not performance, so this is not a performance benchmark. Release-fidelity,
  `manylinux2014` (glibc 2.17) packaging validation is handled separately by
  the nightly/release pipeline.
- `e2e-report` collects whatever ran, including partial results from a
  cancelled or failed job, and renders one HTML report plus a step summary so
  all platforms are visible together.
