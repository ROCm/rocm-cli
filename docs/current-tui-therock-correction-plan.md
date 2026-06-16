# Completed TUI and TheRock Correction Plan

This document captures a completed correction pass so the main agent, reviewer
agents, or a human tester can understand what was fixed without relying on chat
history. It is retained as historical context, not as an open task list.

## Status

Completed on the current `jam/updates` branch.

- Normal Windows users do not need to set `ROCM_CLI_THEROCK_FAMILY`; detection
  chooses the compatible TheRock family where GPU inventory is available.
- First-time setup opens automatically on a fresh `rocm` TUI launch.
- The default TUI surface no longer shows a persistent Activity pane; history
  and lifecycle details are available from `/logs`.
- The setup, activity, and TheRock detection behaviors have focused test
  coverage in the `rocm` binary test suite.

## Scope

The original scope fixed these user-facing issues in the ROCm CLI TUI and
TheRock setup flow:

1. Normal Windows users must not need to set `ROCM_CLI_THEROCK_FAMILY`.
2. First-time setup must trigger automatically on first `rocm` TUI launch.
3. The default TUI surface should not show a low-value Activity pane full of
   lifecycle noise such as startup update checks.
4. The tree must compile cleanly before further feature work resumes.

## Non-Goals

- Do not add new setup concepts beyond the existing implementation plan.
- Do not add CPU fallback behavior for GPU-required engine paths.
- Do not make `ROCM_CLI_THEROCK_FAMILY` part of normal user setup guidance.
- Do not remove lifecycle/action logging itself unless required; this task is
  about the default TUI surface and user guidance.

## Completed Execution Order

### 1. Restore a Clean Compile Surface

This was addressed during the correction pass. Later work completed the
launch-time recipe hint contract intentionally, so this section is historical.

Original expected cleanup:

- Remove the provisional `engine_recipe` field from `LaunchRequest` in
  `crates/rocm-engine-protocol/src/lib.rs`.
- Remove matching `engine_recipe: None` entries from launch request literals in
  the engine adapter CLI paths.
- Keep the completed `ResolveModelRequest`/`ResolveModelResponse` recipe hint
  contract intact.

### 2. Fix TheRock Family Guidance

Normal Windows usage should rely on detected GPU compatibility. The override is
only for deterministic tests or explicit troubleshooting.

Completed behavior:

- `rocm doctor` should report a compatible TheRock family when Windows GPU
  inventory is available.
- `rocm install sdk --format wheel` should use that detected compatible family.
- `ROCM_CLI_THEROCK_FAMILY` or `--family gfx120X-all` should appear only in
  deterministic test instructions, not normal quick-start instructions.

If the installer cannot choose a family on a machine where doctor detects one,
treat that as a bug.

### 3. Auto-Start First-Time Setup

Launching `rocm` with no completed setup state should immediately show the
first-time setup flow. A user should not have to type `/setup`.

Completed behavior:

- Fresh app/config state opens the onboarding view automatically.
- `/setup` remains as a manual way to re-enter or reset setup.
- The prompt/status text should tell the user the next direct action without
  requiring them to discover `/setup next` first.

### 4. Remove the Default Activity Pane Noise

The old Activity pane consumed screen space and showed low-value entries such as
startup update checks. It is no longer part of the default TUI surface.

Completed behavior:

- The default TUI should prioritize transcript/onboarding, status, prompt,
  completions, proposals, and command output.
- Lifecycle logs should remain accessible through explicit log commands.
- Startup lifecycle entries such as update checks should not appear in a
  permanent visible pane.

### 5. Update Tests

Tests were added or adjusted so this behavior is protected:

- Fresh TUI app state auto-renders onboarding.
- Default render does not show the Activity pane.
- Lifecycle update-check entries do not appear on the default surface.
- Existing `/logs` behavior still works where applicable.
- TheRock install tests distinguish normal detection from deterministic family
  overrides.

### 6. Verification Gates

Historical verification gates for this correction pass:

```powershell
cargo fmt --all --check
cargo test -p rocm --bin rocm setup
cargo test -p rocm --bin rocm activity
cargo test -p rocm --bin rocm therock
cargo test --workspace
cargo build --workspace
git diff --check
```

Historical WSL verification commands:

```powershell
wsl.exe --cd /mnt/d/jam/rocm-cli -e env -i HOME=/home/jam PATH=/home/jam/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CARGO_TARGET_DIR=/home/jam/.cache/rocm-cli-target cargo test -p rocm --bin rocm setup
wsl.exe --cd /mnt/d/jam/rocm-cli -e env -i HOME=/home/jam PATH=/home/jam/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CARGO_TARGET_DIR=/home/jam/.cache/rocm-cli-target cargo build --workspace
```

### 7. UX Review Gate

For each visible TUI behavior change during the pass, the UX reviewer was asked
to review:

- whether first-run flow is discoverable without command knowledge,
- whether the default layout has only useful persistent surfaces,
- whether status text gives the next concrete action,
- whether no new feature outside this plan was introduced.

Record the review outcome in `docs/active-implementation-log.md`.
