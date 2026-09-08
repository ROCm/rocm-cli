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
| `e2e-gpu-linux` | `e2e-selfhosted.yml` | Linux GPU pool: MI300X (AMD Instinct), Strix Halo (gfx1151) on Ubuntu, or Radeon AI PRO R9700 (gfx1201) — whichever box is free | self-hosted `[self-hosted, linux, linux-gpu-pool]` |
| `e2e-gpu-strix-windows` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on native Windows 11 | self-hosted `[self-hosted, windows, strix-halo, native]` |
| `e2e-wsl` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Ubuntu under WSL2 | self-hosted `[self-hosted, linux, strix-halo, wsl]` |

**The Linux pool (`EAI-8558`).** MI300X, Strix Halo Ubuntu (native), and rad3's
Radeon R9700 all run Ubuntu, so instead of one dedicated job per box,
`linux-gpu-pool` is a NEW label shared by all three (added to each box) and
`e2e-gpu-linux`'s `runs-on` targets that shared label — GitHub dispatches the
job to whichever box is idle. This also closes a gap noted below: previously
only the Strix box could serve this required check, so it going offline made
the check go missing; now MI300X or rad3 can serve it instead. Each box also
keeps its own distinguishing label (`mi300x`, `strix-halo`+`native`, `r9700`)
so `nightly.yml` can still test all three individually and exhaustively, and
so a scoped `workflow_dispatch` can still force one specific box (see Triggers
below). The Strix Windows and WSL boxes do NOT carry `linux-gpu-pool`, so they
stay out of the Linux pool.

It is a NEW label rather than the existing `amd-gpu`, precisely because of the
bug described next: `amd-gpu` marks every AMD GPU runner, Strix WSL included,
so using it as the pool selector would pull the WSL box into this "Linux" pool
too — exactly the mixed-silicon-selection failure mode `mi300x` was introduced
to fix for MI300X specifically.

The Strix Halo Ubuntu and WSL runners additionally share the `strix-halo`
label (a native host and a WSL host); `native`/`wsl` disambiguate them because
the native host's hardcoded `/home/ubuntu/actions-runner` paths exist only
there.

Every label in a `runs-on` must narrow the pool to one kind of hardware.
`amd-gpu` reads specific but is not — it is carried by every AMD GPU runner,
Strix Halo included, so selecting on it alone (as the MI300X lane once did)
draws from a mixed-silicon pool: the lane could land on gfx1151, pass, and
publish its result under the MI300X column. `mi300x` (this lane), `r9700`, and
`linux-gpu-pool` all narrow to one kind of hardware or the intended pool;
`every_self_hosted_lane_pins_a_hardware_label` in
`xtask/src/workflow_contract.rs` enforces this, because the failure is quiet —
the lane simply passes on the wrong GPU and reports under the label it was
named for.

`e2e` is the blocking, GitHub-hosted mock job: `@requires-gpu` scenarios
resolve to skip here, and known bugs resolve to xfail from
`expectations.toml`. It is a required check and must stay green.

The self-hosted jobs (`e2e-gpu-linux`, `e2e-gpu-strix-windows`, and
`e2e-wsl`) run on AMD GPU systems, so they exercise
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

The lane artifacts are named canonically (`e2e-gpu-linux-report`,
`e2e-gpu-rad3-report`, `e2e-gpu-report`, `e2e-gpu-strix-ubuntu-report`,
`e2e-gpu-strix-windows-report`, `e2e-gpu-strix-wsl-report`, `e2e-report`) in every workflow,
because the report derives each platform's name and OS from
the artifact name. An unrecognised name renders as a guessed platform on
Linux, which would report a Windows lane as Linux; `xtask`'s
`every_uploaded_e2e_artifact_has_a_name_the_report_can_label` guards against
it. `e2e-gpu-linux-report` is the ONE exception: since it can be produced by
any of the three pooled boxes, `xtask e2e-report`'s `discover()` relabels it
at runtime from its own `platform.json` rather than trusting the name — see
`nightly.yml` vs the pool below.

## Triggers

All three self-hosted jobs — `e2e-gpu-linux`, `e2e-gpu-strix-windows`, and
`e2e-wsl` — run automatically on `push`, `pull_request`, and `merge_group`,
gated on the workflow's own `changes` job's `serve` path filter being `true`.
`serve` is narrower than `heavy`: it matches only paths that can change serve
*behaviour* or the GPU E2E harness (the engines, the serve code path in
`apps/rocm`/`apps/rocmd`, `rocm-core`, the e2e-cucumber crate, plus
broad-dependency safety nets), **not** a blanket `**/*.rs`. So a Rust change
that cannot affect serving — e.g. a dashboard-only or unrelated-crate PR —
skips these lanes, while compile coverage for every crate still runs on
`ci.yml`'s always-on build/test lanes. Off `pull_request` (push/merge_group)
the filter is forced `true`, so all three lanes always run there.

`e2e-gpu-linux`'s `runs-on` defaults to the shared `linux-gpu-pool` label (see
Platforms above), so on a normal push/PR/merge_group run GitHub picks whichever
of MI300X/Strix-Ubuntu/rad3 is idle. A scoped `workflow_dispatch` can still
force one specific box instead, via a `runs-on` expression that overrides the
pool label when `platform` names that box specifically (`app-dev-gpu` to
`mi300x`, `strix-ubuntu` to its native label, `rad3` to `r9700`; `all` falls
through to the pool).

Unlike the pre-split layout the GPU jobs do **not** gate on the hosted
`build-and-test` job — cross-workflow `needs` is not possible, so each GPU job
builds the `rocm` binary itself as its first real step (a broken build fails
that job fast and non-fatally). `ci.yml`'s required `build-and-test` and mock
`e2e` remain the authoritative pre-merge build gate.

Every lane can also be triggered manually via `e2e-selfhosted.yml`'s
`workflow_dispatch`, independent of the `serve` gate, with these inputs:

- `platform` (choice: `all`, `app-dev-gpu`, `strix-ubuntu`, `strix-windows`,
  `strix-wsl`, `rad3`) — which self-hosted job(s) to run, and (for the pooled
  Linux job) which box to force. `all` runs every job, with `e2e-gpu-linux`
  using the pool label. `app-dev-gpu`, `strix-ubuntu`, and `rad3` each run
  `e2e-gpu-linux` forced onto that specific box (via `mi300x`, `strix-halo`
  native, and `r9700` respectively) instead of the pool. `strix-windows` and
  `strix-wsl` run their own dedicated jobs (`e2e-gpu-strix-windows`, `e2e-wsl`)
  unchanged. (The mock lane has its own `platform` input on `ci.yml`; it is
  not part of this workflow.)
- `name_filter` (string) — a scenario-name regex forwarded to the cucumber
  harness (`cargo xtask e2e -- --name <regex>`) so a dispatch can run a
  single scenario instead of the full suite. Empty runs everything applicable
  to the selected platform(s).
- `include_nightly` (boolean, default `false`) — opts a dispatch into
  `@nightly`-tagged scenarios (e.g. large-model serves, cold installs) that
  are otherwise skipped on a normal push/PR run to keep it fast.

Dispatch a specific box in the Linux pool with, e.g.:

```bash
gh workflow run e2e-selfhosted.yml --ref <ref> -f platform=rad3
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

It is a **cache with invalidation**, not a one-shot install. Each self-hosted lane calls

```bash
cargo xtask e2e-prewarm --channel release --prewarm-dir "$prewarm"
```

before the suite, which asks `rocm update` whether the channel index has published
a newer version and then:

- installs the SDK when nothing is present for that channel;
- installs the newer runtime **side-by-side** and activates it
  (`rocm update --apply --runtime <key> --activate`) when the index is ahead;
- reuses the existing tree when it is `up_to_date`, when it is `ahead_of_index`
  (a pinned build newer than the index must not be rolled back), or when freshness
  cannot be established at all — an unreachable index reuses and warns rather than
  re-downloading gigabytes or failing the lane;
- prunes with `rocm storage remove-old-installs` after any install or update, so
  the multi-version cache stays bounded.

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

The self-hosted jobs — `e2e-gpu-linux`, `e2e-gpu-strix-windows`, and
`e2e-wsl` — all run with `continue-on-error: true`, so a
hardware failure that RUNS never gates a PR merge. Their results still surface
in the self-hosted consolidated report for visibility.

### Timeouts on the shared Strix box

`e2e-gpu-strix-windows` and `e2e-wsl` always run on the physical Strix Halo
box; `e2e-gpu-linux` runs there too whenever the pool (or a scoped dispatch)
lands it on the Strix Ubuntu runner. Up to three lanes can then be in flight on
that one machine together, so a wait that is comfortable on an idle runner can
expire while a sibling lane loads a model. Two budgets are raised unconditionally
in every self-hosted lane rather than letting contention read as a product
failure: `E2E_SERVE_TIMEOUT_SECS` for serve readiness, and
`E2E_TUI_TIMEOUT_SECS` for the PTY-driven dashboard waits — harmless on the
dedicated MI300X/rad3 boxes too. Both only lengthen how long a wait may take; a
genuine hang still fails, just later.

**Required-check caveat.** These three job **display names** (plus, historically, a
consolidated-report name) were historically documented as being in `main`'s
required-status-check list: `E2E tests (Linux GPU pool)` (`e2e-gpu-linux`,
formerly `E2E tests (Strix Halo, Ubuntu)` before the `EAI-8558` pooling
change), `E2E tests (Strix Halo, Windows)` (`e2e-gpu-strix-windows`), and
`E2E tests (Strix Halo, WSL2)` (`e2e-wsl`). `continue-on-error` neutralizes a
job that ran and failed, but a required check that *never reports* — because
its self-hosted runner is offline — would be treated as missing and block the
merge, IF these checks are actually required. As of 2026-09, recent evidence
says otherwise: PRs #334, #335, and #343 all landed while
`E2E tests (Strix Halo, Ubuntu)` reported `cancelled` on the `merge_group`
ref, which a required check under ALLGREEN would have blocked. Branch
protection is not readable without admin access, so this is inferred from
those merges rather than confirmed from the required list — renaming or
restructuring these lanes (as this pooling change does) is therefore likely
safe either way. Pooling still narrows the theoretical risk for the Linux
check specifically: previously only the Strix box could serve it, so it going
offline would make the check go missing if it is in fact required; now MI300X
or rad3 can serve it instead. If these checks *are* still required, fully
removing the risk for the remaining two (Windows/WSL) additionally requires
removing them from the required list — a branch-protection change tracked
separately from the workflow split.

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
