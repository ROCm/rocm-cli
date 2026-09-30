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
`EAI-7548`. `xtask/src/workflow_contract.rs`'s
`self_hosted_workflow_owns_the_gpu_lanes` test pins `e2e-gpu`,
`e2e-gpu-strix-ubuntu`, and `e2e-gpu-strix-windows` by name in
`e2e-selfhosted.yml`, so this split cannot be silently undone.

## The three-stage validation ladder

Three self-hosted GPU gates run at different points in a change's life. None
are required-status checks (see "Blocking vs. non-blocking" below); what
differs is when each runs and what hardware it covers.

| Rung | Workflow | Trigger | Lanes | Blocks a merge/tag? |
|---|---|---|---|---|
| Per-PR smoke gate | `e2e-selfhosted.yml` | `push`/`pull_request`/`merge_group` (see "Triggers") | The 4 jobs in the Platforms table below | No — absent from the required-status-check list |
| Nightly coverage gate | `nightly.yml` | `schedule` (06:00 UTC daily) + `workflow_dispatch` | The same 4 platforms plus R9700 (`e2e-gpu-nightly-rad3`) and MI350P (`e2e-gpu-nightly-mi350p`), each run against both the `release` and `nightly` package channel (`strategy.matrix.channel: [release, nightly]`, ROCMAI-429) | No — not part of any PR or push-to-main event |
| Release-candidate regression gate | `e2e-selfhosted.yml` | `push` to a `release/**` branch (ROCMAI-120/EAI-8761), ahead of cutting the `v*` tag `release.yml` publishes from | The same 4 per-PR lanes | No — same non-required status as the per-PR gate; a hardware signal for whoever cuts the tag, not an automated block |

R9700 and MI350P were dropped from the per-PR gate (ROCMAI-125): still
covered, but off every PR's critical path, so they moved to the nightly-only
rung instead of being removed outright.

The release-candidate rung's `release/**` trigger ships in PR #415; pinning
the SDK version that rung's pre-warmed runtime resolves to (`cargo xtask
e2e-prewarm --version <ver>` / `--build-date <date>`, so a release-branch run
can hold at `n-1`/`n-2` instead of always tracking the latest channel index)
ships in PR #464, stacked on #415. Neither is merged as of this writing, so
`e2e-selfhosted.yml` on `main` today still triggers on `push: branches:
[main]` only. Wiring an actual `sdk_version: [current, n-1, n-2]` matrix axis
into the release-branch trigger is not part of either PR: nothing in this
repo maps "n-1"/"n-2" to a concrete SDK version (`therock.rs`'s
index-version parsers are private), so the release gate's SDK version is
whatever `--version`/`--build-date` its caller passes, not an automatic
3-way matrix.

## Platforms

The E2E suite (BDD scenarios in Gherkin `.feature` files backed by Rust step
functions) runs as one job per platform. Each job's harness resolves every
scenario to pass / xfail / skip for that host from its `@id` and
`@requires-*` tags, a capability probe, and `expectations.toml` — there is no
separate tier flag or tag filter to maintain.

| Job | Workflow | Platform | Runner labels |
|---|---|---|---|
| `e2e` | `ci.yml` | Mock (no GPU) | GitHub-hosted `ubuntu-latest` |
| `e2e-gpu` | `e2e-selfhosted.yml` | MI300X (AMD Instinct, bare-metal Linux) | self-hosted `[self-hosted, linux, mi300x]` |
| `e2e-gpu-strix-ubuntu` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Ubuntu | self-hosted `[self-hosted, linux, devlab-dispatch, strix-halo]` |
| `e2e-gpu-strix-windows` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Windows 11 | self-hosted `[self-hosted, windows, devlab-dispatch, strix-halo]` |
| `e2e-wsl` | `e2e-selfhosted.yml` | Strix Halo (gfx1151) on Ubuntu under WSL2 | self-hosted `[self-hosted, windows, devlab-dispatch, strix-halo]` |

`e2e-gpu-strix-ubuntu` additionally runs a `strategy.matrix.channel: [release,
nightly]` axis — two concurrent legs on the same job, channel-suffixed
artifact names — so the per-PR smoke gate proves the nightly channel boots on
at least one platform without adding the axis (and its wall-clock cost) to
every per-PR lane. This mirrors `nightly.yml`'s channel matrix from
ROCMAI-429, applied to a single per-PR lane rather than all of them.

All three Strix Halo lanes run on the AMD Ryzen DevLab Dispatch pool: a fresh
runner is registered per job and destroyed after, opt-in only via the
`devlab-dispatch` label, so the pool never picks up a job by accident even
though its hosts also carry `strix-halo`. `e2e-wsl` lands on the same
Windows hosts as `e2e-gpu-strix-windows` — it installs WSL2 and an
Ubuntu-24.04 distro fresh inside the job rather than needing a dedicated
`wsl`-labeled runner. That distro is unregistered between jobs by the pool's
own design (confirmed by its maintainer, along with `/dev/dxg` passthrough
working on these hosts), so every run pays a ~5min distro-install cost before
cloning the checkout into the guest's native filesystem to avoid building
against the slow DrvFs mount. The distro is deliberately left without ROCm's
WSL driver bridge — see "What the WSL distro needs" below.

This migration is scoped to the per-PR lanes in this table, plus the nightly
WSL lane below — not every nightly lane. `nightly.yml`'s Ubuntu and Windows
Strix lanes (`e2e-gpu-nightly-strix`, `e2e-gpu-nightly-strix-windows`) still
pin the `native` label and run on the same static, always-on host these per-PR
lanes moved off of — that label still means what it always did there:
`native` disambiguates this host from any runner that might carry a bare
`wsl` label alongside `strix-halo` (the static WSL runner this repo used
before the WSL lanes below moved to the ephemeral pool). Whether that runner
remains registered is not something this migration changes either way, since
nothing here targets a `wsl`-labeled runner anymore. `nightly.yml`'s
`e2e-wsl-nightly`, however, already moved onto the same DevLab Dispatch pool
as the per-PR `e2e-wsl` lane above, for the same reason: WSL2 needs no
dedicated `wsl`-labeled runner, so there is nothing pool-incompatible about
it. Moving the two `native` nightly lanes onto the pool as well is a
separate, not-yet-made decision.

Every label in a `runs-on` must narrow the pool to one kind of hardware. In
particular the MI300X lane pins `mi300x` rather than `amd-gpu`: `amd-gpu` is
carried by every AMD GPU runner, Strix Halo included, so it selects a
mixed-silicon pool. `every_self_hosted_lane_pins_a_hardware_label` in
`xtask/src/workflow_contract.rs` enforces this, because the failure is quiet —
the lane simply passes on the wrong GPU and reports under the label it was
named for.

`e2e` is the blocking, GitHub-hosted mock job: `@requires-gpu` scenarios
resolve to skip here, and known bugs resolve to xfail from
`expectations.toml`. It is a required check and must stay green.

The self-hosted jobs (`e2e-gpu`, `e2e-gpu-strix-ubuntu`, `e2e-gpu-strix-windows`,
and `e2e-wsl`) run on AMD GPU systems, so they
exercise host/GPU detection, engine `detect`/`capabilities`, and live serving
scenarios that the mock job cannot. GPU availability is advisory in the WSL lane, as
described below.

`e2e-wsl` runs on the DevLab Dispatch pool. Its WSL2/Ubuntu-24.04 distro is
built fresh inside every job (see above), so there is nothing of its own for
a reclaim step to clean up. The sibling Linux lane still carries a `Reclaim
GPU from stray E2E processes` step, retained from when that lane ran on a
static host rather than added for anything specific to the pool; both lanes
now pin the same `devlab-dispatch` label, so whether that step still earns
its place there is a separate, open question, not something this paragraph
answers. Beyond that, `e2e-wsl` otherwise mirrors the Linux lane: GPU
preflight, toolchain bootstrap, shared-runtime pre-warm, then the full suite
with no hand filtering. It covers WSL host detection, the
Windows-to-WSL execution boundary, and whatever GPU access WSL exposes on that
machine. The GPU preflight is advisory here precisely because GPU-on-WSL is
what the lane is proving out: where it is unavailable the capability probe
resolves those scenarios to not-applicable and the rest of the suite still
runs. Scenarios the product deliberately routes around on WSL carry
`@requires-bare-metal`;
scenarios whose premise *is* a WSL host carry `@requires-wsl`, and this is the
only lane that runs them.

`rocm diagnose` is no longer one of the things routed around: it carries a WSL
catalog of its own, so the `@requires-wsl` diagnose scenarios — that a WSL host
is never given a bare-metal cause, and that a WSL remedy is explained rather
than carried out — are proven here and nowhere else.

### What the WSL distro needs

The distro is fresh every job (see above), so the lane provisions it from
scratch each run: `pkg-config`, `build-essential`, `libcap-dev` and friends to
build the workspace, installed unconditionally — the guest always runs as
`root`, so there's no passwordless-sudo gate to check. `sudo` is on that list
anyway, because `rocm install driver` emits a hard `command -v sudo`
precondition into its WSL plan even for a root caller.

What the lane deliberately does *not* provision is ROCm's WSL driver bridge.
Installing ROCDXG is `rocm install driver`'s own job, and the `@requires-wsl`
driver scenarios assert on the plan that command produces — a lane that
pre-installed it would hand those scenarios a host already in the state the
command under test exists to reach. That costs no coverage today, because
`rocm-smi` is absent too, so `@requires-gpu` scenarios resolve to skip either
way (see `e2e-wsl`'s own step comments in `e2e-selfhosted.yml`).

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
`e2e-selfhosted.yml`'s `e2e-report` covers the per-PR GPU platforms;
`nightly.yml`'s `e2e-report-nightly` covers a strict superset of those
platforms (it also runs `e2e-gpu-rad3` and `e2e-gpu-mi350p`, demoted from
per-PR to nightly-only per ROCMAI-125) with the `@nightly` scenarios
included. Each joins its platforms'
reports — including partial or failed runs — by scenario id into one HTML report
and GitHub step summary. Since ROCMAI-429 the report keys each column by
`(platform_slug, channel)` rather than platform alone, so `nightly.yml`'s
release/nightly channel matrix renders two columns per platform instead of
one channel's result silently overwriting the other.

The lane artifacts are named canonically (`e2e-report`, `e2e-gpu-report`,
`e2e-gpu-rad3-report`, `e2e-gpu-mi350p-report`, `e2e-gpu-strix-ubuntu-report`,
`e2e-gpu-strix-windows-report`, `e2e-gpu-strix-wsl-report`) in `ci.yml` and
`e2e-selfhosted.yml`, because the report derives each platform's name and OS
from the artifact name. `nightly.yml`'s self-hosted lanes run a `channel:
[release, nightly]` matrix and append the channel as the final segment before
the `-report` affix (e.g. `e2e-gpu-strix-windows-nightly-report`); the report
strips that segment before matching, so the platform/OS derivation is
unaffected, and the channel itself is read from each artifact's
`platform.json` (or, if absent, this suffix) to keep release and nightly runs
of the same platform in separate columns. An unrecognised name renders as a
guessed platform on Linux, which would report a Windows lane as Linux;
`xtask`'s `every_uploaded_e2e_artifact_has_a_name_the_report_can_label` guards
against it.

## Engine is not an independently selectable axis

None of the tables above vary the serve engine as a matrix dimension, because
the engine is not selectable independent of hardware and OS. `rocm serve`
picks it via `effective_serve_engine()` in
`tests/e2e-cucumber/src/capability.rs` (mirroring the product's own
`preferred_serve_engine_for_host_gpu_summary`):

```rust
pub fn effective_serve_engine(gfx_target: Option<&str>, os_family: &str) -> String {
    if os_family.eq_ignore_ascii_case("windows") {
        return "lemonade".to_owned();
    }
    if family_prefers_vllm(gfx_target) {
        "vllm".to_owned()
    } else {
        "lemonade".to_owned()
    }
}
```

Any Windows host resolves to `lemonade` (the vLLM adapter does not run
there); otherwise `vllm` is only preferred for the `*-dcgpu` families and
`gfx906`/`gfx908`/`gfx90a`. Consequences for the lanes above:

- Strix Halo is `gfx1151` on every OS, so all three per-PR Strix lanes
  (Ubuntu, Windows, WSL2) and their nightly counterparts are lemonade-only —
  no matrix value turns a Strix lane into a vLLM lane.
- `e2e-gpu` (MI300X) is the only per-PR lane where vLLM is the effective
  engine.
- Adding a vLLM lane means adding hardware from a vLLM-eligible family, not
  adding an `engine:` value to a workflow matrix.

## Triggers

The GPU jobs (in `e2e-selfhosted.yml`) run automatically on `push`,
`pull_request`, and `merge_group` when the workflow's own `changes` job's
`serve` path filter is `true`. `push` fires on `main` and on any `release/**`
branch, so pushing to a release branch runs the full self-hosted matrix ahead
of cutting the `v*` tag that `release.yml` builds/publishes from — this is the
release-candidate regression gate; `pull_request` fires against any base
branch. `serve` is narrower than `heavy`: it matches only
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
authoritative pre-merge build gate; `ci.yml`'s own `push` trigger stays
`main`-only, so it does not run on the release-branch commit itself — a
direct push or cherry-pick to a release branch is not covered by it. Coverage
instead comes from whatever pull request produced that commit
(`e2e-selfhosted.yml` gates on nothing but its own build, per the previous
paragraph, and the self-hosted lanes build the binary themselves rather than
compensating for the hosted gate).

They can also be triggered manually via `e2e-selfhosted.yml`'s
`workflow_dispatch`, independent of the `serve` gate, with these inputs:

- `platform` (choice: `all`, `app-dev-gpu`, `strix-ubuntu`, `strix-windows`,
  `strix-wsl`) — which self-hosted job(s) to run. `app-dev-gpu`
  maps to `e2e-gpu`, `strix-ubuntu` to `e2e-gpu-strix-ubuntu`, `strix-windows` to
  `e2e-gpu-strix-windows`, and `strix-wsl` to `e2e-wsl`. (The mock lane has its own
  `platform` input on `ci.yml`; it is not part of this workflow.)
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
lives on the runner's persistent workspace, survives `git clean`, and is namespaced
by source-layout generation (`e2e-prewarm-multi-arch-v2`) so a branch using a new
package layout cannot poison the cache consumed by code that only understands the
previous layout.

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

(`e2e-gpu-strix-ubuntu`'s two channel-matrix legs pass `--channel ${{
matrix.channel }}` instead, resolving to `release` or `nightly` per leg;
every other lane is release-only and keeps the literal `--channel release`
above.)

before the suite. `rocm update` compares both the channel version and the wheel
composition recorded in the runtime manifest (source-layout generation and exact
pinned package specs, including the `device-<target>` payload). A deterministic
composition fingerprint is part of each wheel runtime key, so a corrected
composition is installed beside — never over — the old environment. Each report
line also carries `target=<key>`: the runtime key an apply from that line would
produce, which on a superseded manifest is its already-installed replacement.
Pre-warm then:

- installs the SDK when nothing is present for that channel;
- installs a newer runtime **side-by-side** and activates it
  (`rocm update --apply --runtime <key> --activate`) when the index is ahead;
- replaces a same-version runtime side-by-side when its manifest has an older or
  missing wheel composition, then activates the composition-keyed replacement;
- treats that repair as complete while the matching replacement remains installed,
  so retained legacy manifests do not trigger repeated repairs or notifications;
- activates the runtime a reuse actually means — after a repair that is the
  replacement named by `target=`, not the superseded manifest the line belongs to;
- ensures the default engine is installed even when the runtime itself is reused;
- reuses the existing tree when it is `up_to_date`, when it is `ahead_of_index`
  (a pinned build newer than the index must not be rolled back and cannot be
  reproduced from the index, so it is never offered a repair), or when freshness
  cannot be established at all — an unreachable index reuses and warns rather than
  re-downloading gigabytes or failing the lane;
- prunes with `rocm storage remove-old-installs` after any install, update, or
  repair, so the multi-version cache stays bounded.

`e2e-prewarm` also accepts mutually exclusive `--version`/`--build-date`
flags (ROCMAI-430) that pin the SDK build the pre-warm resolves to instead of
always tracking whatever the channel index currently serves; the unpinned
invocation above is unaffected, and today no caller in this repo passes
either flag yet (see "The three-stage validation ladder" above).

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
`e2e-gpu-strix-windows`, and `e2e-wsl` — all run with
check names that are absent from branch protection's required-status-check
list (see below), so a hardware failure never gates a PR merge no matter how
it reports. All four run with `continue-on-error: false`, so a real
regression shows red instead of always green. Their results also
surface in the self-hosted consolidated report for visibility.

### Timeouts on the Strix lanes

Every Strix Halo lane raises two budgets rather than letting a slow host read
as a product failure: `E2E_SERVE_TIMEOUT_SECS` for serve readiness, and
`E2E_TUI_TIMEOUT_SECS` for the PTY-driven dashboard waits. On the pool-hosted
lanes this covers a busy or cold-starting pool host; `e2e-wsl` additionally
runs through WSL2's own virtualization overhead, which can make a wait that's
comfortable on native hardware run long. Both budgets only lengthen how long
a wait may take; a genuine hang still fails, just later.

**Required-check history.** These job names — and `E2E consolidated report
(self-hosted)` — used to be in `main`'s required-status-check list, where a
required check that *never reports* (because its self-hosted runner is
offline) is treated as missing and still blocks the merge, `continue-on-error`
notwithstanding. That branch-protection change has since landed (confirmed
2026-09-11): none of the self-hosted lane names, nor the self-hosted
consolidated report, are in the required list anymore (`ci.yml`'s own mock
`E2E tests` / `E2E consolidated report` are the required checks — similarly
prefixed but distinct from the self-hosted names above, so no name string
actually collides). An offline or unclaimed self-hosted runner can no longer
block a merge.

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
