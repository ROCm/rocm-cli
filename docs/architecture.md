<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Architecture

This is the living module map for rocm-cli. It's a contributor-facing reference to where things live and why, updated in the same PR as the code it documents — not a design history or a project-tracking artifact.

> Before relying on any entry below, verify current file and function boundaries directly (e.g. `grep`) rather than trusting this doc's wording. Module boundaries shift as the codebase grows, and a stale-but-plausible-looking note is worse than an explicit prompt to check.

## Module organization convention

New subcommands and subsystems default to their own file from day one — they should not grow inside `main.rs`/`lib.rs` waiting for a future extraction pass. Two extraction patterns already exist in the codebase; use whichever fits:

- **Full domain extraction** — a subsystem's domain implementation (types, logic) moves into its own file. In binary crates (`apps/rocm`) it's a private `mod x;`, accessed via qualified paths (e.g. `comfyui::render_status(...)`) — the clap command enum (e.g. `ComfyuiCommand`) and its dispatch function stay in `main.rs`. In library crates (`crates/rocm-core`) it's `pub mod x;` plus a `pub use x::{...};` re-export, since the module is part of the crate's public API. This is the default for new subsystems. Examples: `apps/rocm/src/therock.rs`, `comfyui.rs`, `providers.rs`; `crates/rocm-core`'s `diagnose.rs`/`examine.rs`.
- **Mechanical relocation** — a `pub(crate) fn` moves out verbatim, with shared types/config staying at the crate root and reached via `crate::`. Used for dispatch-adjacent clusters where a minimal, easy-to-review diff matters more than full extraction. Examples: `apps/rocm/src/automations.rs`, `uninstall.rs`.

There is no file-line-count CI gate enforcing this — `too_many_lines = "allow"` in the workspace `Cargo.toml` is a deliberate, function-level choice, not an oversight. This convention is the guardrail instead.

## Module map

Scoped to the crates that make up the shipped CLI/daemon/dashboard/engine surface, plus `crates/e2e-report` (a shared exception: it's HTML/markdown reporting consumed only by `xtask` and `tests/e2e-cucumber`, but it's still one of the modularization effort's target files, so it's mapped below). Dev-tooling and test-harness workspace members (`xtask`, `tests/e2e-cucumber` themselves) are otherwise out of scope — they're not part of the modularization effort's inventory.

### `apps/rocm` — main CLI binary

Subsystem modules already following full domain extraction: `therock.rs`, `comfyui.rs`, `providers.rs`. Mechanically relocated dispatch-adjacent handlers: `automations.rs`, `uninstall.rs`. Other existing per-subsystem modules: `chat_host_facts.rs`, `dash.rs`, `dash_seam.rs`, `endpoint_keys.rs`, `provider_keys.rs`, `serve_summary.rs`, `storage.rs`, `bootstrap.rs`, `logging.rs`. Shared CLI-output components: `cli_progress.rs` (`Spinner`, `AnimatedSpinner`), `cli_report.rs` (`ActionReport`).

`main.rs` itself is **not yet modularized** — see EAI-7768, split planned across several PRs, one cluster at a time.

### `apps/rocmd` — background daemon

`lib.rs` is **not yet modularized** — see EAI-7768.

### `crates/rocm-core` — core library

Already-extracted subsystem modules include `diagnose.rs`, `examine.rs`, and several siblings following the same pattern. `lib.rs` itself is **not yet modularized** — see EAI-7768, planned last in the modularization effort: highest fan-in (every app and engine crate depends on it), but lowest novelty since the existing sibling modules already prove the pattern works.

### `crates/rocm-dash-core`, `rocm-dash-collectors`, `rocm-dash-daemon`, `rocm-dash-tui` — dashboard/telemetry

`rocm-dash-tui`'s `agent.rs` and `app/mod.rs` are **not yet modularized** — see EAI-7768. `crates/rocm-dash-tui/src/ui/approval.rs` is the shared component for approval-state prompts — reuse it rather than hand-rolling new approval UI.

### `crates/rocm-engine-protocol` — engine IPC protocol

A contract surface: verify all impacted engines after any protocol change here (see `AGENTS.md`).

### `crates/rocm-deps`

Pinned versions of the third-party runtimes rocm-cli manages (from workspace-root `runtime-deps.toml`, turned into constants by `build.rs`). Small and already single-purpose; not part of the modularization effort's target list.

### `crates/e2e-report`

`lib.rs` is **not yet modularized** — see EAI-7768.

### `engines/lemonade`, `engines/vllm` — inference engine adapters

Both crates' `lib.rs` are **not yet modularized** — see EAI-7768.
