# Codex TUI Vendoring Plan For `rocm-cli`

> **SUPERSEDED.** The vendored Codex approach has been
> dropped. The ~70-crate vendored tree caused the Linux CI timeout and shipped
> no essential functionality (it was reachable only via the hidden
> `--experimental-codex-tui` flag spawning a separate `rocm-codex` binary). The
> interactive agentic TUI will instead be delivered via an ACP client driving
> the Goose binary, with ROCm tools exposed over MCP. Retained for historical
> context only.

## Goal

Replace the current in-tree `rocm` TUI with a fully vendored Codex TUI stack from `openai/codex`, while retrofitting ROCm-specific semantics, tools, approvals, and background automation into the Codex interaction model.

This plan assumes:

- `rocm-cli` remains Apache-2.0 licensed.
- The vendored Codex tree is treated as a pinned upstream subtree/fork.
- `rocmd` becomes the primary backend contract for interactive sessions.
- `rocm` remains the user-facing binary.
- direct CLI subcommands remain available for scripting and automation.

## Non-Goals

- Keeping the current ad hoc `ratatui` implementation as the long-term primary UI.
- Rewriting the Codex TUI from scratch to look similar.
- Making the vendored TUI generic before it works for ROCm.
- Preserving every current `rocm-cli` semantic exactly as-is if a Codex-style session/tool model is better.

## Product Direction

The new interactive experience should be:

- Codex-style transcript-first UI
- plan/tool/approval oriented
- backed by structured ROCm tools instead of shell text blobs
- capable of local-only operation
- capable of cloud-backed chat when configured

The user-visible product remains `ROCm AI Command Center CLI`.

## Default Provider Policy

Interactive `rocm` must preserve a default cloud-backed chat path even when the user has not configured an API key.

Provider priority for interactive chat:

1. explicit provider selected in config or CLI flags
2. provider implied by configured API keys
3. ChatGPT sign-in session, including free-plan sign-in if available
4. local provider

This means:

- if the user has `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or another explicitly configured provider, use that
- if the user has no provider configured, `rocm` should offer the same ChatGPT sign-in flow Codex uses
- the ChatGPT free plan remains the default no-key interactive path
- local models remain available and should be easy to switch to, but they are not forced as the default merely because no API key exists

This is an explicit product choice. `rocm-cli` should feel usable immediately on a fresh system.

## Architectural Decision

We will vendor the Codex Rust TUI stack intact enough to preserve:

- session model
- transcript/history cells
- approval flows
- tool execution views
- sidebar/status model
- picker flows
- embedded or remote app-server architecture

We will not attempt to vendor only visual widgets. The upstream TUI is tightly coupled to an app-server protocol and session runtime.

## Target Architecture

### User-facing binaries

- `rocm`
  - primary CLI and interactive TUI entrypoint
  - launches vendored Codex-style TUI in interactive mode
  - retains subcommands for automation and scripting
- `rocmd`
  - long-lived backend daemon when automations, managed services, or remote session state are enabled
  - primary ROCm backend for tools, events, approvals, and service orchestration

### Core backend layers

- `rocm-core`
  - host detection
  - config
  - filesystem layout
  - GPU detection
  - managed runtime manifests
- `rocm-tools`
  - structured ROCm tool implementations
- `rocm-codex-bridge`
  - adapter between vendored Codex protocol expectations and `rocmd` tool/event model
- `rocm-engine-protocol`
  - engine plugin contract
- engine crates
  - `pytorch`
  - later `vllm`, `sglang`, `atom`, `llama.cpp`

### Vendored upstream tree

- `third_party/openai-codex/codex-rs/`

This vendored tree remains as close to upstream as practical.

## Repo Layout

Recommended target layout:

```text
rocm-cli/
  apps/
    rocm/
    rocmd/
  crates/
    rocm-core/
    rocm-tools/
    rocm-codex-bridge/
    rocm-engine-protocol/
  engines/
    pytorch/
    vllm/
    sglang/
    atom/
  third_party/
    openai-codex/
      codex-rs/
  docs/
    codex-vendoring-plan.md
    upstream-codex-sync.md
    rocmd-backend-contract.md
```

## Vendoring Rules

### Upstream management

- pin a specific upstream `openai/codex` commit SHA
- record that SHA in `docs/upstream-codex-sync.md`
- record local patch categories:
  - branding
  - provider policy
  - backend bridge
  - disabled upstream features
  - ROCm-specific UI additions

### Patch discipline

- prefer wrapper crates and adapters over invasive changes in the vendored tree
- avoid renaming upstream crates in the first pass
- isolate required source edits to a documented patch queue
- keep each local patch rebased and attributable

### License and notices

- preserve upstream `LICENSE` and `NOTICE`
- add attribution in our top-level docs
- keep vendored subtree provenance visible

## Semantics Retrofit

The key shift is from command-oriented local CLI behavior to a session/tool/event model.

### Old model

- `rocm` parses user input
- sometimes shells out to itself
- prints summaries
- uses local helper functions directly

### New model

- `rocm` interactive mode is a client for a Codex-style session runtime
- user requests become tool-backed turns
- tool calls are visible objects in the transcript
- approvals are first-class
- outputs stream as structured history cells

### ROCm-native tool surface

The backend tool surface should include:

- `examine`
- `gpu_snapshot`
- `list_gpus`
- `list_runtimes`
- `install_sdk`
- `update_sdk`
- `remove_sdk`
- `install_driver`
- `check_driver`
- `install_engine`
- `detect_engine`
- `resolve_model`
- `launch_server`
- `stop_server`
- `server_status`
- `tail_logs`
- `watcher_list`
- `watcher_enable`
- `watcher_disable`
- `service_restart`
- `prefetch_artifact`
- `config_get`
- `config_set`

Every user-visible interactive action should be representable as one or more of these tools.

## `rocmd` As The Backend

`rocmd` should become the canonical runtime for:

- session state
- tool execution
- approvals
- watchers
- managed services
- event streams
- telemetry sampling
- log aggregation

### `rocmd` responsibilities

- expose a stable internal API for the TUI bridge
- execute tool calls with typed inputs and outputs
- emit events suitable for transcript rendering
- manage background state for automations and service supervision
- arbitrate approvals for privileged operations

### Session modes

- on-demand mode
  - `rocmd` started in-process or transiently for interactive sessions
- long-lived mode
  - required when automations, managed services, or remote session continuity are enabled

## Provider Model

The vendored Codex provider model must be adapted but not stripped.

### Required providers

- `chatgpt`
  - sign-in based
  - includes free-plan path
  - default when no other provider is configured
- `openai`
  - API key based
- `anthropic`
  - API key based
- `local`
  - local engine backed

### Provider selection behavior

- if user explicitly selects a provider, honor it
- if config specifies a provider, honor it
- if API keys are present, prefer the matching configured provider
- otherwise offer ChatGPT sign-in
- if ChatGPT sign-in is unavailable or declined, fall back to local

### Important rule

Do not regress the no-key onboarding path. A fresh install should be able to drop into chat using ChatGPT sign-in, even on the free plan, if upstream Codex supports that flow.

## UI Retrofit

The vendored TUI should be re-skinned and re-contextualized rather than rewritten.

### Required transcript content

- plan blocks
- tool cards
- command summaries
- stdout/stderr output blocks where relevant
- approval prompts
- service status updates
- watcher activity
- update availability
- model and engine resolution details

### Required sidebar content

- host OS and arch
- detected GPU family and gfx targets
- live `amd-smi` telemetry
- installed TheRock runtime
- selected engine
- selected provider
- active server summary
- automation/watcher status
- update status

### Required slash commands

- `/examine`
- `/gpu`
- `/runtimes`
- `/install`
- `/update`
- `/engine`
- `/serve`
- `/services`
- `/logs`
- `/automations`
- `/provider`
- `/model`
- `/config`
- `/quit`

## Approval Model

Codex approval flows should be repurposed directly for ROCm operations.

### Approval required

- DKMS driver install or update
- uninstall apply
- switching runtime channel between `release` and `nightly`
- deleting caches or managed envs
- binding non-loopback ports
- writing to system prefixes like `/opt/rocm`
- any root or `sudo` operation

### No approval required

- `examine`
- GPU telemetry
- dry-run planning
- runtime listing
- model resolution
- local status inspection

## Automations And Watchers

The Codex app-server model is compatible with our watchers design, but the semantics must remain ROCm-specific.

### Watchers become backend-managed automations

Examples:

- `update.available`
- `server.crashed`
- `gpu.overtemp`
- `vram.pressure`
- `healthcheck.failed`
- `artifact.downloaded`

### Execution modes

- `observe`
- `propose`
- `contained`

These should map to visible transcript history cells and approval states in the vendored TUI.

## Migration Plan

### Phase 0: Inventory and freeze

- freeze current in-tree TUI feature work except critical bug fixes
- document the current `rocm` interactive feature set
- define the minimum viable replacement feature set

Deliverables:

- this plan
- `docs/upstream-codex-sync.md`
- `docs/rocmd-backend-contract.md`

### Phase 1: Vendor upstream Codex Rust workspace

- import pinned `codex-rs` subtree
- build vendored crates in CI without ROCm adaptations
- verify license and attribution requirements

Deliverables:

- vendored source tree
- CI job that compiles upstream snapshot

### Phase 2: Add wrapper entrypoint

- add a `rocm-codex-shell` integration target
- prove we can launch vendored TUI from `rocm`
- keep current TUI available behind a temporary fallback flag

Deliverables:

- `rocm --experimental-codex-tui`
- startup banner with ROCm branding

### Phase 3: Backend bridge

- design and implement `rocm-codex-bridge`
- map Codex session/tool protocol calls into `rocmd`
- stand up embedded backend mode first

Deliverables:

- transcript can render `examine`
- transcript can render GPU snapshot
- transcript can render config and engine inventory

### Phase 4: ROCm tool adoption

- move install/update/engine/server operations behind structured backend tools
- stop relying on shelling out to `rocm` from the TUI
- preserve CLI subcommands as thin clients over shared backend logic

Deliverables:

- `install sdk`
- `update`
- `engines install`
- `serve --managed`
- `services`

### Phase 5: Provider and auth adaptation

- preserve ChatGPT sign-in flow
- preserve free-plan onboarding path
- add `anthropic`, `openai`, and `local` provider selection
- expose provider state in the sidebar and settings flows

Deliverables:

- first-run provider picker
- working ChatGPT default path with no API key

### Phase 6: Approvals and watchers

- wire approvals to privileged ROCm actions
- map watcher activity into history cells and status panes
- ensure `rocmd` long-lived mode is stable

Deliverables:

- approval prompts for driver/system actions
- visible watcher state and event history

### Phase 7: Remove old TUI

- make vendored Codex-style UI the default interactive path
- remove or archive the old in-tree TUI implementation
- keep CLI commands for scripting

Deliverables:

- `rocm` launches vendored UI by default on interactive TTYs

## Compatibility Strategy

### Linux

- first-class support target
- full install/update/serve/watcher support

### Windows

- interactive TUI supported
- provider flows supported
- managed runtimes supported where available
- driver installation remains deferred
- Windows-specific backend/tool capability gating must be explicit

## Risks

### High risk

- upstream sync cost
- app-server protocol mismatch
- OpenAI-specific assumptions embedded in the vendored runtime
- over-modifying the vendored tree too early

### Medium risk

- duplicate config systems during migration
- authentication path complexity
- approval semantics drift

### Low risk

- branding changes
- sidebar/status customization

## Design Constraints

- preserve no-key interactive onboarding through ChatGPT sign-in
- preserve local-only functionality
- keep direct CLI automation intact
- avoid long-term dependence on shelling out from the TUI
- keep vendored patches small and documented

## Recommended First Implementation Slice

The first slice should not try to migrate everything.

Build this sequence first:

1. vendor Codex Rust workspace
2. compile vendored TUI in `rocm-cli` CI
3. wrap launch from `rocm --experimental-codex-tui`
4. implement backend bridge for:
   - `examine`
   - `gpu_snapshot`
   - `config_get`
   - `list_engines`
   - `list_services`
5. preserve ChatGPT sign-in default path
6. add ROCm branding and sidebar telemetry

If that slice works, the rest of the migration becomes practical.

## Success Criteria

- `rocm` launches a Codex-style TUI in interactive mode
- default fresh-install onboarding works without an API key via ChatGPT sign-in
- ROCm operations appear as structured tool calls, not opaque shell blobs
- GPU telemetry is visible in the sidebar
- approvals work for privileged operations
- `rocmd` is the backend authority for sessions, tools, and watchers
- direct CLI commands still work for scripts and CI
