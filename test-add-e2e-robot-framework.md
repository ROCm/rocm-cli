
# WIP: E2E BDD tests for rocm-cli (PR #69, cucumber-rs)

**Stage:** 8-awaiting-pr-approval
**Pipeline:** standard
**Branch:** test/add-e2e-robot-framework
**Last Updated:** 2026-07-10

**Token Usage:** in=2234 out=656753 cache_create=10780854 cache_read=345536971 calls=1124

---

## Problem

rocm-cli's existing tests (Rust unit tests, Python smoke scripts) validate
components in isolation but miss cross-component integration failures — alias
resolution not forwarded to engine adapters, TUI detection probing wrong ports,
hardcoded model names. Six bugs (EAI-7218..7223) were found during manual MI300X
testing, none caught by existing tests. Need a behavioral E2E layer exercising
full user journeys against the real `rocm` binary.

## Solution

E2E suite with BDD scenarios (Gherkin) exercising install → examine → serve →
detect → chat, black-box (no crate imports), two tiers (mock / GPU). **Started in
Robot Framework, switched to cucumber-rs** after team pushback (EAI-7164: one Rust
toolchain). The suite must genuinely gate CI, quarantine known bugs, and route
scenarios through the actual binary.

## Scenarios

`.feature` files under `tests/e2e-cucumber/features/` (chat, model_serving,
examine, runtime_setup). Tags: bare `@expected-failure` (+ `@expected-failure-EAI-NNNN`
traceability) and `@gpu` for hardware-dependent scenarios.

## Implementation Steps

### Completed ✅ (Robot era, Jun 30–Jul 1)
- ✅ Robot suites, keyword libs, mock server, CI jobs, self-hosted runner
- ✅ 6 bugs filed (EAI-7218..7223), each with a failing scenario
- ✅ Built cucumber-rs PoC and did the Robot-vs-cucumber comparison (below)

### Completed ✅ (cucumber migration + PR #69 review response, Jul 8)
- ✅ Replaced Robot with cucumber-rs suite
- ✅ Exit-code gating: summarized writer + `process::exit(1)` on failure (report first)
- ✅ Real tag split — 3 CI selections: `e2e` blocking (`not @gpu and not @expected-failure`), `e2e-known-bugs` + `e2e-gpu` non-blocking. Cucumber tags are exact-match → bare `@expected-failure` for filtering
- ✅ Scenarios exercise real `rocm` via planted service-record JSON (`register_mock_service`); untagged mislabeled-EAI-7220 services-list scenarios; dropped helper-only chat model-name scenario
- ✅ Per-scenario temp isolation; async poll loop (dropped `blocking`); crate clippy in CI; `.feature` path filter
- ✅ `run.sh` → `cargo xtask e2e`; report.rs bugs fixed (kept generator per user decision)
- ✅ Engine-set reconcile (lemonade/vllm only, #79); restored mock/gpu split lost in rewrite
- ✅ Committed `6741054` (signed+signed-off, no AI attribution), force-pushed, PR comment posted (#issuecomment-4916250315)

### Completed ✅ (CI-correctness + multi-runner + consolidated report, Jul 9–10)
- ✅ Fixed 3 CI failures that surfaced once clippy went green (heavy jobs had been
  skipped behind the lint gate): (1) `Test (affected crates)` — nextest couldn't
  `--list` the custom cucumber harness; (2) `windows-build-and-test` — ran all
  scenarios unfiltered with no mock. Both fixed by `test = false` on the `[[test]]
  e2e` target (nextest + `cargo test --all-targets` skip it; `xtask e2e`'s
  explicit `--test e2e` still runs it). (3) `E2E tests (known bugs)` exited red on
  expected failures → added **xfail inversion**.
- ✅ xfail inversion: `cargo xtask e2e --expect-failures` sets `E2E_EXPECT_FAILURES`;
  harness treats `@expected-failure` failing as green, red only on XPASS / untagged
  failure / parse-hook error. `report::evaluate_xfail` + `XfailReport`. Committed `55b3aec`.
- ✅ Hardened cross-engine expansion step (was `unwrap_or_default()` → vacuous
  `""=="" ` pass; now panics on missing `resolved model:` line). Confirmed EAI-7219
  is NOT fixed — the local "pass" was the vacuous artifact.
- ✅ Split GPU job into expect-pass + known-bugs tiers (commit `d33d182`).
- ✅ Added Strix Halo runners: two new self-hosted runners came online
  (`strix-halo-ubuntu` Linux gfx1151, `strix-halo-windows` Windows 11 iGPU).
  Added 4 jobs (each expect-pass + known-bugs), label-routed by `strix-halo` +
  os. Windows = first real Windows-GPU e2e coverage. Commit `93f03ef`.
  (app-dev-gpu = `amd-gpu` label; strix boxes = `strix-halo` label.)
- ✅ **Consolidated cross-platform report** (COMMITTED `96108e5`): extracted
  lean `crates/e2e-report` (maud+serde only, so xtask doesn't pull
  cucumber/axum/tokio); `generate_consolidated` + `consolidated_summary_markdown`
  (platform×tier matrix, xfail-aware, flags XPASS); `cargo xtask e2e-report`
  (globs `*-report` artifacts → auto-discovers platforms); CI `e2e-report` job
  (`if: always()`, step-summary + one HTML artifact). 52 tests green, clippy
  clean, Linux container green, browser-verified HTML render.
- ✅ Final clippy pass: made `ok()` & `status_text()` const; collapsed nested `if`;
  used `writeln!` instead of format_push_string. All 3 crates: 52 tests green,
  Linux container full suite green, harness re-export verified.
- ✅ Committed `96108e5` (signed via github-app fallback wrapper, signed-off);
  force-pushed to origin/test/add-e2e-robot-framework. PR now has 4 commits total:
  CI-correctness (`55b3aec`), GPU split (`d33d182`), Strix runners (`93f03ef`),
  consolidated report (`96108e5`).

### Completed ✅ (Isolation refactor, both-engine coverage, EAI triage, Jul 10)
- ✅ **Test isolation refactor**: moved `qwen2.5` alias from serve-setups to EAI-7219 scenarios only.
  Serve preconditions now use canonical `Qwen/Qwen2.5-1.5B-Instruct`, isolating failures to bugs being tested, not upstream unrelated bugs.
- ✅ **Timeout env-configurable**: `E2E_SERVE_TIMEOUT_SECS` (default 600s) to make serve readiness tunable per hardware.
- ✅ **Strengthened auto-engine assertion**: now parses actual engine value from serve output, asserts it's in {lemonade, vllm} (post-#79).
- ✅ **Both-engine explicit coverage**: added scenario 7 (lemonade serve+inference on Qwen3-0.6B-GGUF), tagged `@expected-failure-EAI-7052`.
- ✅ **Manual MI300X GPU repro**: built main from latest on app-dev pod, confirmed `setup_gpu_model` (vllm) works end-to-end, `setup_lemonade_model` inference hangs (Vulkan backend/ROCm mismatch).
- ✅ **EAI tickets filed/updated**: EAI-7333 (healthcheck readiness gap: /v1/models ≠ inference-ready), unassigned + rocm-cli component. Decided unit-test fix in engine crates (not E2E).
- ✅ **E2E suite final count**: 18 scenarios total (8 mock expect-pass, 2 mock known-bugs, 5 gpu expect-pass, 3 gpu known-bugs).

### Completed ✅ (readiness + default-engine tests, EAI-7218 recheck, Jul 10 late)
- ✅ **Reverted** the scenario-5 EAI-7219 tag: isolation refactor means setup uses canonical
  name, so scenario 5 (vllm inference) is a clean expect-pass again, NOT xfail.
- ✅ **EAI-7333 unit test** added in `crates/rocm-core/src/lib.rs`:
  `models_endpoint_readiness_does_not_imply_inference_ready` (mock HTTP server lists model on
  /v1/models but can't infer → readiness signal is a false positive). Passes.
- ✅ **Scenario 8** (`@gpu`, expect-pass): "A service reported ready can immediately serve
  inference" — asserts CLI-reported ready (`services list`) ⇒ inference works now. Verified on
  MI300X: `Status: 1 ready` → immediate inference returned real content. New steps:
  `the CLI reports the service as ready`, `an inference request succeeds immediately`.
- ✅ **Scenario 9** (`@gpu`, expect-pass): "vLLM is the default serving engine on Instinct" —
  serve a vLLM-capable model w/o `--engine` → asserts `engine: vllm`. Verified on MI300X:
  `Qwen/Qwen2.5-0.5B-Instruct` → `engine_selection: detected ROCm GPU family prefers vLLM`.
  New steps: `the user serves a vLLM-capable model without specifying an engine`,
  `vLLM is selected as the default engine`.
- ✅ **2 rocm-core unit tests** for default-engine logic: `instinct_dcgpu_family_prefers_vllm`
  (gfx94X-dcgpu → vllm; None on Windows), `consumer_gpu_family_has_no_vllm_preference`
  (gfx110X-all → None). Pin the GPU-family default engine-side.
- ✅ **EAI-7218 rechecked on fresh main**: pytorch engine REMOVED by #79 — `--engine pytorch`
  now rejected (`invalid value 'pytorch' [possible values: lemonade, vllm]`); auto-select for
  `qwen2.5` picks lemonade, never pytorch. The EAI-7218 error only reproduced on the STALE
  `/workload/rocm-cli` v0.3.0 binary (pre-#79). So EAI-7218 (Won't Fix) genuinely N/A on main.
- ✅ **Suite now: 20 scenarios** (8 mock expect-pass, 2 mock known-bugs, 7 gpu expect-pass,
  3 gpu known-bugs). model_serving.feature = 9 scenarios.

### DEFAULT-ENGINE SELECTION PRECEDENCE (verified in code + hardware, apps/rocm/src/main.rs ~3530)
1. explicit `--engine`  2. configured `default_engine`  3. **GPU family prefers vLLM**
(`*-dcgpu` incl. MI300X gfx94X-dcgpu → vllm) *only if recipe supports vllm*  4. recipe
`preferred_engines`  5. platform default (`default_engine_for_platform()` = "lemonade").
KEY: GGUF-only recipes (e.g. `qwen2.5`→Qwen3-4B-GGUF, `Qwen2.5-1.5B`) fall through to lemonade
even on Instinct because vllm can't serve them. vLLM-capable safetensors (e.g.
`Qwen/Qwen2.5-0.5B-Instruct`) → vllm on Instinct. This is CORRECT, not a bug.

### Todo 📋
- ✅ DONE: committed the 3 files as `c25c4eb` (signed via git-commit-with-fallback, signed-off,
  no AI refs), pushed `96108e5..c25c4eb`. Pre-commit verify: mock suite 8/8 green + Linux
  container suite fully green (rocm-core 170 passed incl. 3 new tests, e2e-report 18 passed).
  PR #69 now 5 commits: 55b3aec, d33d182, 93f03ef, 96108e5, c25c4eb.
- 📋 Add engine-level unit tests for EAI-7333 in engines/vllm + engines/lemonade healthcheck
  (the rocm-core test covers the shared helper; the per-engine healthcheck_service still trusts
  /v1/models — could add per-engine coverage too, lower priority).
- ✅ DONE: Awaited + inspected live CI run on PR #69 (run 29090245163). Core CI all green
  (clippy, build-and-test, Windows, coverage, signatures, CodeQL). E2E mock tiers passing 
  (blocking + xfail-inverted). GPU tiers in progress. Strix Halo all 4 fail at `setup-rust-toolchain` 
  (curl write-to-disk failure, infra not code). No code action needed.
- 📋 **Fix scenario numbering** in model_serving.feature (currently 1,2,3,4,5,7,6,9,8 — out of
  order after insertions). Cosmetic; renumber 1-9 sequentially for readability. LOW priority.
- 📋 (Optional) Update E2E cucumber memory ([[rocm-cli-e2e-cucumber]]) with new default-engine
  precedence + both-engine coverage once this lands — currently reflects the pre-multi-engine state.
- 📋 Persistent self-hosted GPU runner (currently ephemeral workspace pod).

### In progress 🚧 (Jul 10 — CI infra fixes + manual dispatch)
- ✅ **Inference timeout cap** (`f2fa495`): `send_chat` client had NO timeout → a hung
  backend (EAI-7052 lemonade) blocked the HTTP POST forever, stretching the GPU known-bugs
  job to 28+ min. Added `E2E_INFERENCE_TIMEOUT_SECS` (default **10s**). Not a product bug
  (hang is EAI-7052); harness gap only.
- ✅ **Strix Linux disk fix** (`f2ee84e`): runner root disk full/non-persistent → rustup
  bootstrap `curl (23)` write failure. Redirect CARGO_HOME/RUSTUP_HOME/TMPDIR to
  `/home/ubuntu/actions-runner` + pre-create tmp. Both pushed to #69.
- ✅ **Strix Windows bootstrap**: `setup-rust-toolchain` fails (`bash: command not found`);
  replaced with a pwsh `win.rustup.rs` install step (`--default-toolchain none`, rust-toolchain.toml
  pins 1.96.0), idempotent. Disk on Windows box = 1.8TB, no redirect needed. Committed as `96b8bbb`.
- ✅ **Manual dispatch plumbing** (task: [[ci-manual-e2e]]): added `workflow_dispatch` + platform/tier
  inputs to #69's ci.yml (byte-identical to PR #98), build-and-test skip on dispatch, all 8 E2E jobs
  guarded with platform×tier mapping, e2e-report runs on dispatch. actionlint clean. Committed as `96b8bbb` (signed, no AI refs), pushed. Container suite green pre-push (31 ok).
  - **Key insight**: `cargo xtask e2e` builds the binary in place (`cargo build --release -p rocm`)
    when ROCM_CLI_BINARY unset → E2E jobs are self-contained, `needs: build-and-test` is ordering-only,
    so skipping it on dispatch is safe (each job builds fresh from the dispatched ref).
  - **PR #98** (`ci-manual-e2e`, off main): just the `on:` trigger, so ci.yml becomes
    dispatchable (workflow_dispatch must exist on default branch). CI green, **in merge queue** (will land soon).
  - Once #98 merges: rebase #69 onto pinned main, then `gh workflow run ci.yml --ref test/add-e2e-robot-framework -f platform=strix-windows -f tier=known-bugs` runs E2E on-demand.
- ✅ **create_worktree.sh gotcha**: branched ci-manual-e2e/ci-harden-actions off STALE local main
  (18 behind origin, pre-`affected` subcommand) → `Test (affected crates)` failed with
  "unrecognized subcommand 'affected'". Fixed #98 + #99 by rebasing onto origin/main.
- ✅ **CI security hardening** (PR #99 `ci-harden-actions`): SHA-pinned 4 remaining actions (checkout, cache, setup-rust-toolchain, dtolnay/rust-toolchain) across ci.yml/nightly.yml/release.yml + added `.github/dependabot.yml` (github-actions, weekly). Also applied repo settings: default token read-only, no PR approvals by actions. Post-merge: flip `sha_pinning_required` (after rebasing #69/#98 onto pinned main to avoid CI rejections).

## Next Steps

- Monitor PR #69 CI + re-review. If report.rs maintainability is raised again,
  `maud` is the easy upgrade (user wants to keep the HTML report).
- On merge: post-merge cleanup, then remove worktree.

## Checklist

- [x] Scenarios written before implementation
- [x] All 6 bugs have corresponding scenarios (or documented as untestable)
- [x] Pre-commit checks pass
- [x] No internal/AI references in PR/commits (AGENTS.md compliance)
- [ ] PR reviewed and merged

## Blockers / Open Questions

- **3 files uncommitted** (see Todo) — commit + push pending.
- **Signing**: use `github-app` skill's `git-commit-with-fallback` (NOT raw git; 1Password SSH
  agent flaky this machine — wrapper has GPG fallback). Progress-branch WIP syncs may commit
  unsigned (skill-permitted for orphan branch only).
- **Jira ticket convention** (user pref): new tickets = assignee UNASSIGNED + component `rocm-cli`.
  acli edit lacks a component flag → set via REST: `curl -u "$JIRA_USERNAME:$JIRA_TOKEN" -X PUT
  "$JIRA_URL/rest/api/2/issue/KEY" -d '{"fields":{"assignee":null,"components":[{"name":"rocm-cli"}]}}'`.
  acli auth: `acli jira auth login` (env has JIRA_TOKEN/URL/USERNAME for REST).
- **Persistent runner**: self-hosted GPU runner is ephemeral; needs a k8s deploy.

## Robot Framework vs cucumber-rs (decision: cucumber-rs)

Team lead (EAI-7164) wanted to consolidate on Rust. cucumber-rs won on: real
Gherkin (`.feature` IS the test, no `.robot` rewrite/drift), one toolchain (no
Python in CI), structural safety (OS-assigned ports, `Drop` cleanup). Tradeoffs:
more boilerplate than Python; no built-in HTML report (added a generator); Robot's
`--skiponfailure` gave nicer fail→SKIP signal than cucumber's tag exclusion —
addressed here with the exit-code fix + dedicated known-bugs job.

## Notes

- **macOS dev constraint**: pre-push hook + local clippy fail on this Mac because
  rocm-cli targets Linux/Windows only (cfg-stub clippy lints; 3 `rocm`-bin
  python-launcher/comfyui tests). Verified the full workspace suite passes on
  **Linux** (Apple `container`, arm64, Rust 1.96, exit 0) → push used `--no-verify`
  with evidence. See project memory.
- **ALWAYS run the full suite in the Linux container before every push** (user rule).
  The Mac pre-push hook can't pass by design, so the container run IS the
  verification gate. Command: `workspace/wip/container-test.sh all` (from the
  worktree) — runs workspace tests excl. cucumber harness + e2e-cucumber lib
  tests + e2e mock blocking selection. Must be fully green (all `test result: ok`,
  e2e `8 scenarios (8 passed)`) before pushing with `--no-verify`. Never push on
  the strength of the native Mac run alone — the rocm-bin flake passes in
  isolation but fails under full parallel load, which is a false red, not a real
  signal. Container green is the real signal.
- EAI-7220 = TUI-only wrong-port bug (can't be black-box tested); EAI-7218 =
  Won't Fix (PyTorch engine removed by #79; rechecked on main — confirmed N/A).
  EAI-7219 (vllm alias not forwarded — CONFIRMED still open on HW), 7221/7223 still open.
- EAI-7222 (privacy notice) requires TUI interaction — documented known gap.
- **EAI-7052** (In Progress): lemonade should use installed ROCm libs — root cause of
  lemonade inference hang on MI300X (falls back to Vulkan backend). Scenario 7 tagged with it.
- **EAI-7333** (filed this session, Backlog, unassigned, component=rocm-cli): healthcheck
  reports ready from /v1/models, not actual inference. Covered by rocm-core unit test.
- PR #66 (README fixes) was a companion PR from the same testing session (merged).

## app-dev MI300X access (manual GPU verification)
- Cluster context `app-dev` (local Keycloak `kc.app-dev.silogen.ai/realms/airm`, password grant
  from `~/.kube/cluster-user`; needs Drift/Headscale up). Auth via authenticate-clusters skill;
  if `invalid_grant`, run `authenticate-clusters/scripts/diagnose-auth app-dev` (creds may just
  need the secrets-cache refreshed — it auto-recovered this session).
- Runner is EPHEMERAL (`e2e-test-runner` ns empty between jobs). For manual repro use the
  persistent dev pod: ns `rocm-cli`, pod `wb-dev-workspace-vscode-*` (1 GPU, MI300X gfx943).
- **Always build fresh main** — the pod's `/workload/rocm-cli` is STALE (v0.3.0, pre-#79).
  Recipe: `git clone --depth 1 https://github.com/ROCm/rocm-cli.git /workload/rocm-cli-main`
  (public, no auth), build with `/root/.rustup/toolchains/1.96.0-x86_64-unknown-linux-gnu/bin/cargo`
  + `RUSTUP_TOOLCHAIN=1.96.0-x86_64-unknown-linux-gnu` (rustup shim symlinks are DANGLING; use the
  toolchain bin directly). Release build ~13 min. libcap present.
- Serve repro pattern: isolated root (`ROCM_CLI_{CONFIG,DATA,CACHE}_DIR=/tmp/x/...`), symlink
  `ln -s /root/.rocm/runtimes /tmp/x/data/runtimes`, `rocm runtimes activate
  release-wheel-gfx94x-dcgpu-7-13-0`, then serve. Managed serve DETACHES — read plan from the
  redirected output file, not the exec stdout. Always stop services after + verify no stray
  vllm/llama-server procs (they hold the GPU). Never `pkill` broadly — it kills the exec shell.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/test-add-e2e-robot-framework`
- Recreate with: `create_worktree.sh test/add-e2e-robot-framework`
- Relocated from a nested-path worktree to this flat path to avoid a git
  admin-basename collision (see git-wt worktree memory).

## Work Log

### 2026-06-29..07-01 (Robot era)
- Manual MI300X testing found 13 issues (7 docs→PR #66, 6 bugs→EAI-7218..7223).
- Built Robot Framework E2E framework; PR #69 created; mock CI green.
- Built cucumber-rs PoC + comparison; team tooling decision pending.

### 2026-07-08 (cucumber migration + review response)
- Rebased onto origin/main; recovered from a worktree admin-dir collision
  (relocated to flat path); cleaned an orphaned docs worktree.
- Addressed all of @rominf's CHANGES_REQUESTED points; committed `6741054`,
  force-pushed, posted summary comment.
- Set up Apple `container` to verify the Linux test suite locally (green).
- Set up local-only progress branch for WIP storage (push-guarded against
  upstream). Next: await CI + re-review.

### 2026-07-09 (idle flush)
- **2026-07-09 (idle flush):** Session idle for 1 hour, auto-flushing WIP state.

### 2026-07-09..07-10 (CI correctness, multi-runner, consolidated report)
- Rebased on main (picked up #84 catalog curation, #87 const stubs, #88 dash port).
- Diagnosed why CI jobs "started failing": they were previously SKIPPED behind a
  failing clippy gate; once clippy passed, the real Test/Windows/known-bugs
  failures surfaced for the first time.
- Fixed all three: `test = false` (nextest + Windows), xfail inversion
  (known-bugs). Committed `55b3aec`, force-pushed.
- Established EAI-7220 (#88) was fixed via a dash-tui unit test, not e2e (it's
  TUI-only, untestable black-box) — that unit test passes on the rebased tree.
- Split GPU tier expect-pass/known-bugs (`d33d182`).
- Two new Strix Halo runners came online; added 4 jobs to use them (`93f03ef`).
- Built the consolidated cross-platform report (new `crates/e2e-report` +
  `xtask e2e-report` + `e2e-report` CI job). Verified: 52 tests, clippy clean,
  Linux container green, browser-rendered the HTML matrix. **Commit blocked on
  locked 1Password signing agent** — staged, awaiting unlock.

### 2026-07-10 (context switch, verify & finalize consolidated report, manual GPU repro)
- Resumed from previous session. Pulled summary, reviewed full branch & implementation.
- Completed final clippy fixes (3 lints): made `ok()` & `status_text()` const;
  collapsed nested `if` in e2e_report.rs; used `writeln!` instead of `format_push_string`.
- Ran full test suite: 52 tests green (18 new in e2e-report, 34 in xtask), harness
  still green via re-export, Linux container suite fully green.
- Updated WIP file with multi-runner & consolidated-report scope. Synced to
  progress branch (unsigned, 1Password locked, per skill design).
- Debugged signing blocker on `96108e5` commit: 1Password SSH agent was returning errors. 
  Switched to `git-commit-with-fallback` (github-app skill wrapper) which has GPG fallback;
  commit signed successfully with RSA SSH key. Pushed to origin/test/add-e2e-robot-framework 
  (fast-forward, no force). All 4 daily commits now landed & pushed.

### 2026-07-10 (test isolation refactor, both-engine coverage, manual hardware validation)
- **Verified consolidated report** on fresh main build (cloned rocm-cli-main, `cargo build --release`).
  Served vLLM model end-to-end (qwen2.5 canonical, 600s timeout), inference succeeds, confirms readiness contract works on vLLM path.
- **Test isolation applied**: pinned serve preconditions to canonical model names + explicit engines, moved alias `qwen2.5` to EAI-7219 scenarios only.
  Ensures downstream serve+inference tests fail only from bugs they test, not upstream alias issues.
- **Both-engine coverage**: added scenario 7 (lemonade serve + inference on Qwen3-0.6B-GGUF, `@expected-failure-EAI-7052`).
  Confirmed lemonade reaches `/v1/models` ready (~4s cached) but inference hangs (Vulkan backend, not ROCm — missing system-ROCm config).
- **Manual testing on app-dev MI300X**: reproduced EAI-7219 (vllm alias bug), EAI-7052 (lemonade inference hang), confirmed auto-select resolves aliases correctly.
- **EAI-7333 filed** (healthcheck reports ready via /v1/models, not inference). Decided: unit-test fix in engine crates (white-box logic), not E2E (black-box behavior doesn't isolate the code defect cleanly).
- **Suite metrics**: 18 scenarios total (breakdown: 8 mock blocking, 2 mock xfail, 5 gpu blocking, 3 gpu xfail).

### 2026-07-10 (readiness test, default-engine test, EAI-7218 recheck — context-clear checkpoint)
- Added EAI-7333 rocm-core unit test (readiness ≠ inference-ready), scenario 8 (readiness
  contract, expect-pass, HW-verified), scenario 9 (vLLM default on Instinct, expect-pass,
  HW-verified), + 2 rocm-core default-engine unit tests. Suite now 20 scenarios.
- Mapped full default-engine precedence in code + confirmed on MI300X: Instinct dcgpu prefers
  vLLM for vllm-capable models; GGUF-only recipes correctly fall through to lemonade.
- Rechecked EAI-7218 on fresh main: pytorch fully removed by #79 (`--engine pytorch` rejected);
  the earlier repro was only the stale pod binary. Reverted the tentative scenario-5 7219 tag.
- 3 files changed, all clippy/fmt clean, verified on hardware. NOT yet committed (next session).
- Documented app-dev manual-GPU-verification recipe in Notes for future sessions.

### 2026-07-10 (CI status checkpoint, Strix infra investigation)
- Verified live run (29090245163) core CI all green; E2E mock tiers (blocking + xfail) passing.
- GPU tiers in progress; Strix Halo infra issue confirmed (curl write-to-disk in rustup, not code).
- Checked Strix runner reach (GH API 403, no admin; not k8s pods, standalone hosts). No access path.
- Added brevity feedback to project memory (keep answers short for status questions).
- WIP updated, synced to progress branch.

### 2026-07-10 (runner provisioning + dispatch plumbing + security hardening)
- **Inference timeout**: added `E2E_INFERENCE_TIMEOUT_SECS` (default 10s) to cap unbounded HTTP client, fail fast on known hangs.
- **Strix Linux disk**: set `CARGO_HOME`/`RUSTUP_HOME`/`TMPDIR` to persistent `/home/ubuntu/actions-runner`, pre-create tmp, toolchain installs on full-disk runner.
- **Strix Windows bash**: replaced bash-based action with PowerShell `win.rustup.rs` bootstrap, `--default-toolchain none`, idempotent.
- **Manual dispatch workflow**: added `workflow_dispatch` trigger + platform/tier inputs to ci.yml (8 E2E jobs guarded, e2e-report runs on dispatch).
- **PR #98** (ci-manual-e2e): enabler, **in merge queue**. PR #99 (ci-harden-actions): SHA-pin all actions, add dependabot, apply repo security settings.

### 2026-07-10 (earlier — runner provisioning fixes + security hardening planning)
- **Inference timeout cap** (commit `f2fa495`): `send_chat` used unbounded HTTP client; moved to 10s timeout via `E2E_INFERENCE_TIMEOUT_SECS` env. Known-bugs xfail scenarios now fail fast (~1 min) instead of blocking until job limit.
- **Strix Linux disk fix** (commit `f2ee84e`): runner's full/non-persistent root disk → `rustup-init` download fails (curl 23). Set `CARGO_HOME`, `RUSTUP_HOME`, `TMPDIR` to `/home/ubuntu/actions-runner`, pre-create temp dir. Toolchain install + reuse now on persistent disk.
- **Strix Windows bash fix** (staged, not pushed): `setup-rust-toolchain@v1` runs bash script internally. Replaced with PowerShell step: `iwr win.rustup.rs`, `--default-toolchain none` (rust-toolchain.toml pins 1.96.0). Idempotent; cold box auto-provisions on first run.
- **Both commits pushed** (via `--no-verify` + green Linux container suite verification, per user rule). New PR head: `f2ee84e`. CI run 29092792483 triggered.
- **Security audit** (admin access now): confirmed fork-PR approval all-external, default token read-only, no PR approvals by workflows. Two settings still loose: `sha_pinning_required: false` (actions on mutable tags), `allowed_actions: all`. Planned: separate hardening PR to pin all action refs + add dependabot + new manual-dispatch E2E workflow.
- **Hardening scope**: (1) pin 5 unpinned actions to resolved SHAs (checkout v6.0.3, upload/download-artifact v4.6.2/4.3.0, cache v5.1.0, setup-rust-toolchain v1.17.0, dtolnay/rust-toolchain 1.96.0), (2) add `.github/dependabot.yml` for auto-bump, (3) add manual-dispatch E2E workflow (inputs: platform [app-dev-gpu/strix-ubuntu/strix-windows/all], tier [expect-pass/known-bugs/both], no build-and-test dependency), (4) post-merge: flip `sha_pinning_required: true`.
- **Design decision pending**: move provisioning fixes (Strix bash→pwsh, Linux disk env) OUT of PR #69 INTO hardening PR for coherent CI-infra focus, or keep in #69 + duplicate bootstrap in manual workflow?

### 2026-07-10 (Strix infra fixes + app-dev runner analysis)
- **Strix Windows pwsh → powershell fix**: updated 4 job steps (lines 848, 859, 892, 903) from `shell: pwsh` to `shell: powershell` (Windows PowerShell 5.1, only available shell on self-hosted Strix Windows box). Fixes `pwsh: command not found` error. **Staged, not committed.**
- **Strix Linux HOME redirect**: root disk (/) + `/home/ubuntu` both full; `/home/ubuntu/actions-runner` is 1.7TB nvme mount (1% used). Confirmed rustup's `.profile` write fails ENOSPC on root fs. Added `HOME: /home/ubuntu/actions-runner/temp-home` + mkdir to redirect all of rustup's home-dir writes. **Staged, not committed.**
- **Task #6 batch preparation**: combined 3 fixes (Windows pwsh, Linux HOME, 8 non-E2E skip-on-dispatch guards from earlier) into one uncommitted batch. All valid individually (tested during live run inspection); ready to commit + push once current PR CI run completes.
- **app-dev runner analysis**: extracted full current pod spec (image, GPU resource requests, node labels, PVC mounts). Currently ephemeral (lives in vscode dev-workspace pod, emptyDir `/workload`, dies on pod restart). User confirmed: want to keep MI300X gfx943 runner active as a CI target after shutting down vscode. **Decision needed**: GitHub credential (PAT or App token) to enable auto-registration on Deployment startup. Plan: hand-rolled Deployment (one fixed runner, not ARC) to replicate pod spec independently + self-bootstrap via token API on startup.
