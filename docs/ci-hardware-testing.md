<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# CI hardware (GPU / WSL) testing

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
| `e2e-wsl` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Ubuntu under WSL2 | self-hosted `[self-hosted, linux, strix-halo, wsl]` |
| `e2e-gpu-rad3` | `e2e-selfhosted.yml` | Radeon AI PRO R9700 (gfx1201) on Linux | self-hosted `[self-hosted, linux, r9700]` |

The Strix Halo lanes pin the extra `native` label because two Linux runners
share the `strix-halo` label (a native host and a WSL host) and the jobs'
hardcoded `/home/ubuntu/actions-runner` paths exist only on the native one. The
WSL lane pins `wsl` for the same reason, from the other side.

`e2e` is the blocking, GitHub-hosted mock job: `@requires-gpu` scenarios
resolve to skip here, and known bugs resolve to xfail from
`expectations.toml`. It is a required check and must stay green.

The self-hosted jobs (`e2e-gpu`, `e2e-gpu-strix-ubuntu`,
`e2e-gpu-strix-windows`, `e2e-wsl`, and `e2e-gpu-rad3`) run on AMD GPU systems, so they exercise
host/GPU detection, engine `detect`/`capabilities`, and live serving scenarios
that the mock job cannot. GPU availability is advisory in the WSL lane, as
described below.

`e2e-wsl` runs on an Ubuntu distro hosted in WSL2 on the Strix Halo Windows box
and mirrors the sibling Linux lane step for step: stray-serve reclaim, GPU
preflight, toolchain bootstrap, shared-runtime pre-warm, then the full suite
with no hand filtering. It covers WSL host detection, the Windows-to-WSL
execution boundary, and whatever GPU access WSL exposes on that machine. The
GPU preflight is advisory here precisely because GPU-on-WSL is what the lane is
proving out: where it is unavailable the capability probe resolves those
scenarios to not-applicable and the rest of the suite still runs. Scenarios the
product deliberately routes around on WSL carry `@requires-bare-metal`; the one
scenario whose premise *is* a WSL host carries `@requires-wsl`, and this is the
only lane that runs it.

### What the WSL distro needs

The distro needs `pkg-config`, `build-essential` and `libcap-dev` to build the
workspace; the lane installs them itself where it has passwordless sudo, and
otherwise fails with the list of what is missing rather than hanging on a
password prompt.

GPU coverage additionally needs ROCm's WSL passthrough to be complete —
`/dev/dxg` and dxcore alone are not enough, `librocdxg.so` and its ldconfig
entry must be present too. `rocm examine` reports the verdict as
`driver_status: wsl_rocdxg_ready`; anything else (`wsl_rocdxg_missing`,
`wsl_gpu_plumbing_missing`) means the runtime cannot reach the device even
though `detected_gfx_target` still names it, because that target is read from
the Windows-side driver. The capability probe keys `@requires-gpu` on the
driver verdict rather than the target for exactly this reason, so a distro
without the passthrough runs the non-GPU suite and reports the GPU scenarios as
not applicable, instead of failing them on a premise the host cannot meet.

Each workflow has its own consolidated report job. `ci.yml`'s `e2e-report`
covers the mock platform;
`e2e-selfhosted.yml`'s `e2e-report` covers the GPU platforms;
`nightly.yml`'s `e2e-report-nightly` covers the same platforms with the
`@nightly` scenarios included. Each joins its platforms'
reports — including partial or failed runs — by scenario id into one HTML report
and GitHub step summary.

The lane artifacts are named canonically (`e2e-report`, `e2e-gpu-report`,
`e2e-gpu-rad3-report`, `e2e-gpu-strix-ubuntu-report`, `e2e-gpu-strix-windows-report`,
`e2e-gpu-strix-wsl-report`) in every workflow, because the report derives each
platform's name and OS from the artifact name. An unrecognised name renders as a
guessed platform on Linux, which would report a Windows lane as Linux; `xtask`'s
`every_uploaded_e2e_artifact_has_a_name_the_report_can_label` guards against it.

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

- `platform` (choice: `all`, `app-dev-gpu`, `strix-ubuntu`, `strix-windows`,
  `strix-wsl`, `rad3`) — which self-hosted job(s) to run. `app-dev-gpu` maps to
  `e2e-gpu`, `strix-ubuntu` to `e2e-gpu-strix-ubuntu`, `strix-windows` to
  `e2e-gpu-strix-windows`, `strix-wsl` to `e2e-wsl`, and `rad3` to
  `e2e-gpu-rad3`. (The mock lane has its own `platform` input on `ci.yml`; it is
  not part of this workflow.)
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

## The shared pre-warmed runtime

Nearly every GPU serve scenario points its `data/runtimes` at one shared,
pre-warmed managed runtime tree (`E2E_SHARED_RUNTIMES_DIR`), so a multi-GiB
`rocm install sdk` happens once per runner instead of once per scenario. The tree
lives on the runner's persistent workspace and survives `git clean`.

The tree may hold **more than one** runtime — the pre-warm installs a newer one
side by side when the channel index publishes it (below) — so scenarios must not
rely on the CLI auto-selecting a runtime, which it deliberately declines to do
once two are installed. Each scenario keeps its own config dir, so the pre-warm's
`--activate` is invisible to it; the precondition steps re-activate from the
tree's own `active.json`, which lives inside the shared tree and is therefore
visible through the symlink. Without that, a serve fails with `no active ROCm
runtime is configured` while the precondition still passes.

It is a **cache with invalidation and repair**, not a one-shot install. Each
self-hosted lane calls:

```bash
cargo xtask e2e-prewarm --channel release --prewarm-dir "$prewarm"
```

before the suite. `rocm update` compares both the channel version and the wheel
composition recorded in the runtime manifest (source-layout generation and exact
pinned package specs). A deterministic composition fingerprint is part of each new
wheel runtime key, so a repair is installed beside—not over—the old environment.
Pre-warm then:

- installs the SDK when nothing is present for that channel;
- installs a newer runtime **side-by-side** and activates it when the index is
  ahead;
- replaces a same-version runtime side-by-side when its manifest has an older or
  missing wheel composition, then activates the composition-keyed replacement;
- ensures the default engine is installed even when the runtime itself is reused;
- reuses the existing tree when it is `up_to_date`, when it is `ahead_of_index`
  (a pinned build newer than the index must not be rolled back), or when freshness
  cannot be established at all — an unreachable index reuses and warns rather than
  re-downloading gigabytes or failing the lane;
- prunes with `rocm storage remove-old-installs` after any install, update, or
  repair, so the multi-version cache stays bounded.

The runtime is always installed **in place**: `install sdk` bakes absolute paths
into the runtime manifest, so a tree that is moved after installation leaves every
serve pointing at a path that no longer exists.

This replaced an existence-only guard that never reinstalled, which had frozen both
MI300X runners on a 16-day-old runtime and left drift from a fresh install untested
(`EAI-8057`). The decision logic lives in `xtask` rather than the workflows because
the pre-warm block is duplicated across multiple jobs in two shells;
`xtask/src/e2e_prewarm.rs` carries unit tests for each freshness verdict, and the
`runtime-update-reports-freshness` scenario pins the `rocm update` output shape those tests assume.

## Blocking vs. non-blocking

The self-hosted jobs — `e2e-gpu`, `e2e-gpu-strix-ubuntu`,
`e2e-gpu-strix-windows`, `e2e-wsl`, and `e2e-gpu-rad3` — all run with `continue-on-error: true`, so a
hardware failure that RUNS never gates a PR merge. Their results still surface
in the self-hosted consolidated report for visibility.

### Timeouts on the shared Strix box

Three of those lanes — the two native Strix ones and `e2e-wsl` — run on the same
physical machine and can be in flight together, so a wait that is comfortable on
an idle runner can expire while a sibling lane loads a model. Two budgets are
raised there rather than letting contention read as a product failure:
`E2E_SERVE_TIMEOUT_SECS` for serve readiness, and `E2E_TUI_TIMEOUT_SECS` for the
PTY-driven dashboard waits. Both only lengthen how long a wait may take; a
genuine hang still fails, just later.

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
