<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# CI hardware (GPU) testing

The hosted CI (`ubuntu-latest`, `windows-latest`) builds and unit-tests every
shipping target natively, but GitHub-hosted runners have no AMD GPU. A
dedicated hardware layer covers that gap by running the same cucumber-rs E2E
suite on dedicated self-hosted runners with real AMD GPUs.

That hardware layer lives in its **own workflow**, `.github/workflows/e2e-selfhosted.yml`,
separate from the main `ci.yml`. The split is deliberate: a job queued on an
**offline** self-hosted runner cannot be cancelled by GitHub, so if it shared
`ci.yml`'s concurrency group a superseded run would hold that group and the
newer run's merge-required (GitHub-hosted) checks would sit pending forever
(observed on PR #138). Giving the self-hosted lanes their own workflow — and
thus their own concurrency group — means an offline runner can only ever stall
that workflow's own supersession, never `ci.yml`'s required checks. See
`EAI-7548`.

## Platforms

The E2E suite (BDD scenarios in Gherkin `.feature` files backed by Rust step
functions) runs as one job per platform. Each job's harness resolves every
scenario to pass / xfail / skip for that host from its `@id` and
`@requires-*` tags, a capability probe, and `expectations.toml` — there is no
separate tier flag or tag filter to maintain.

| Job | Workflow | Platform | Runner labels |
|---|---|---|---|
| `e2e` | `ci.yml` | Mock (no GPU) | GitHub-hosted `ubuntu-latest` |
| `e2e-gpu` | `e2e-selfhosted.yml` | MI300X (AMD Instinct, bare-metal Linux) | self-hosted `[self-hosted, linux, amd-gpu]` |
| `e2e-gpu-strix-ubuntu` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Ubuntu | self-hosted `[self-hosted, linux, strix-halo, native]` |
| `e2e-gpu-strix-windows` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on native Windows 11 | self-hosted `[self-hosted, windows, strix-halo, native]` |

The Strix Halo lanes pin the extra `native` label because two Linux runners
share the `strix-halo` label (a native host and a WSL host) and the jobs'
hardcoded `/home/ubuntu/actions-runner` paths exist only on the native one.

`e2e` is the blocking, GitHub-hosted mock job: `@requires-gpu` scenarios
resolve to skip here, and known bugs resolve to xfail from
`expectations.toml`. It is a required check and must stay green.

The three GPU jobs (`e2e-gpu`, `e2e-gpu-strix-ubuntu`, `e2e-gpu-strix-windows`)
run on dedicated self-hosted runners with a real AMD GPU attached, so they
exercise host/GPU detection, engine `detect`/`capabilities`, and live serving
scenarios that the mock job cannot.

Each workflow has its own consolidated report job (both named `e2e-report`
internally). `ci.yml`'s `E2E consolidated report` covers the mock platform;
`e2e-selfhosted.yml`'s `E2E consolidated report (self-hosted)` covers the three
GPU platforms. Each joins its platforms' reports — including partial or failed
runs — by scenario id into one HTML report and GitHub step summary.

## Triggers

The GPU jobs (in `e2e-selfhosted.yml`) run automatically on `push`,
`pull_request`, and `merge_group` when the workflow's own `changes` job's
`serve` path filter is `true`. `serve` is narrower than `heavy`: it matches only
paths that can change serve *behaviour* or the GPU E2E harness (the engines, the
serve code path in `apps/rocm`/`apps/rocmd`, `rocm-core`, the e2e-cucumber crate,
plus broad-dependency safety nets), **not** a blanket `**/*.rs`. So a Rust change
that cannot affect serving — e.g. a dashboard-only or unrelated-crate PR — skips
the heavy GPU matrix, while compile coverage for every crate still runs on
`ci.yml`'s always-on build/test lanes. Off `pull_request` (push/merge_group) the
filter is forced `true`, so the full matrix always runs there. Unlike the
pre-split layout the GPU jobs do **not** gate on the hosted `build-and-test` job
— cross-workflow `needs` is not possible, so each GPU job builds the `rocm`
binary itself as its first real step (a broken build fails that job fast and
non-fatally). `ci.yml`'s required `build-and-test` and mock `e2e` remain the
authoritative pre-merge build gate.

They can also be triggered manually via `e2e-selfhosted.yml`'s
`workflow_dispatch`, independent of the `serve` gate, with these inputs:

- `platform` (choice: `all`, `app-dev-gpu`, `strix-ubuntu`, `strix-windows`) —
  which self-hosted job(s) to run. `app-dev-gpu` maps to `e2e-gpu`,
  `strix-ubuntu` to `e2e-gpu-strix-ubuntu`, and `strix-windows` to
  `e2e-gpu-strix-windows`. (The mock lane has its own `platform` input on
  `ci.yml`; it is not part of this workflow.)
- `name_filter` (string) — a scenario-name regex forwarded to the cucumber
  harness (`cargo xtask e2e -- --name <regex>`) so a dispatch can run a
  single scenario instead of the full suite. Empty runs everything applicable
  to the selected platform(s).
- `include_nightly` (boolean, default `false`) — opts a dispatch into
  `@nightly`-tagged scenarios (e.g. large-model serves, cold installs) that
  are otherwise skipped on a normal push/PR run to keep it fast.

Dispatch the GPU lanes with, e.g.:

```bash
gh workflow run e2e-selfhosted.yml --ref <ref> -f platform=app-dev-gpu
```

## Blocking vs. non-blocking

The three hardware jobs — `e2e-gpu`, `e2e-gpu-strix-ubuntu`, and
`e2e-gpu-strix-windows` — all run with `continue-on-error: true`, so a
hardware failure that RUNS never gates a PR merge. Their results still surface
in the self-hosted consolidated report for visibility.

**Required-check caveat.** These three job names (plus, historically, a
consolidated-report name) are still in `main`'s required-status-check list.
`continue-on-error` neutralizes a job that ran and failed, but a required check
that *never reports* — because its self-hosted runner is offline — is treated as
missing and still blocks the merge. The workflow split removes the catastrophic
concurrency stall (an offline runner can no longer freeze `ci.yml`'s hosted
required checks), but fully unblocking merges while a runner is offline
additionally requires removing these self-hosted checks from the required list —
a branch-protection change tracked separately from the workflow split.

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
- Each workflow's `e2e-report` job collects whatever ran in that workflow —
  including partial results from a cancelled or failed job — and renders one
  HTML report plus a step summary. `download-artifact@v8` flattens a
  single-match download straight into the artifacts directory (it uses the root
  path when exactly one artifact matches, regardless of the pattern), so after
  the split each report job usually has one artifact; `xtask e2e-report`'s
  discovery handles both the flattened and per-subdirectory layouts, labeling a
  root-level report from its `platform.json` slug.
