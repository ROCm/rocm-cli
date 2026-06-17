# rocm-cli Implementation Plan

## Current Scope Note

The original plan text below contains early CPU-only and CPU-fallback ideas.
Those are superseded for the current implementation pass by the user-facing
no-fallback policy: GPU-required serving must fail loudly instead of falling
back to CPU, and rocm-cli does not offer local CPU serving as a workaround for
missing ROCm GPU support. CPU-only hosts remain useful for the TUI, provider
chat, artifact inspection, and non-serving diagnostics, but they are not a
reason to add CPU fallback paths to PyTorch, llama.cpp, vLLM, SGLang, or ATOM.

## Current Implementation State

Audited on 2026-06-06. The implementation has moved well beyond the original
greenfield plan. Use `plans/rocm-cli-remaining-implementation-plan.md` as the
live tracker.

- Implemented locally: the Rust CLI/TUI, first-time setup, TheRock managed
  `pip` venv installs, runtime activation/import/adopt/uninstall, bounded
  Examine reports, service records/logs/actions, the model registry CLI/TUI,
  provider adapters, PyTorch/llama.cpp/Lemonade/vLLM engine adapters,
  automation proposals/sandbox runner, packaging scripts, installer
  verification, release-readiness self-tests, and native per-OS release builds.
- Bootstrap/setup uses normal install scripts for clean machines with no ROCm,
  Python, Rust, or Cargo installed; these seed minimal config, verify before
  activation, and set PATH automatically. rocm-cli ships native per-OS binaries
  (with first-party engines and the `rocmd` helper surface linked into `rocm`);
  it installs TheRock wheels, Python environments, models, ComfyUI, and app
  caches on disk only after the user chooses a folder and approves setup.
- TheRock installs now use a single pinned TheRock-index transaction for
  `rocm[libraries,devel]`, `torch`, `torchvision`, and `torchaudio`, with the
  exact ROCm wheel suffix selected for the current Python/platform.
- Local Windows/WSL acceptance has passed for managed TheRock install,
  PyTorch GPU smoke, Lemonade ROCm serving, llama.cpp GPU smoke, ComfyUI
  start/stop plus HTTP smoke, WSL ROCDXG readiness, and WSL vLLM GPU smoke on
  the current local AMD GPU host.
- Current local serving caveat: saved historical `Qwen/Qwen3.5-4B` PyTorch
  service attempts fail because the managed Transformers package does not
  recognize the checkpoint's `qwen3_5` architecture. The short `qwen` alias now
  resolves to the recommended low-VRAM assistant recipe. Lemonade is the
  default local engine for assistant/server flows; PyTorch remains a supported
  managed engine and explicit Qwen3.5 PyTorch requests are gated before launch
  with a clear compatibility error.
- Remaining gates are mostly owner/upstream/infrastructure items: production
  signing keys and hosted indexes, native-Linux release validation beyond WSL,
  privileged Linux driver acceptance, live ATOM/SGLang acceptance on supported
  upstream GPU targets, broader GPU-family CI, and a production AMD
  driver-update source.

### Done vs Left Snapshot

Done locally:

- TUI, first-time setup, setup reset, navigable command screens, overlapping
  approval/progress/detail cards, logs, Examine, services, runtimes, update, and
  provider/local-chat surfaces.
- Managed TheRock wheel runtime installs with rocm-cli-owned venvs, local pip
  cache, exact TheRock torch/runtime selection, runtime activation, import,
  adopt, uninstall, and list-based install selection.
- Fast setup/Examine host detection: setup shows detected GPU name/target/package
  without running full Examine; Windows Examine uses native registry/system probes
  instead of PowerShell/CIM on the common path; native Linux uses KFD/sysfs/IP
  discovery before any ROCm userland tool; WSL avoids duplicate host display
  probes.
- PyTorch, llama.cpp, vLLM, SGLang, and ATOM adapters with explicit no-CPU-
  fallback policy. PyTorch and llama.cpp have live Windows/WSL GPU acceptance;
  vLLM has live WSL GPU acceptance on this host.
- Local served-model chat can use ROCm command tools and approval-routed
  mutating actions. ComfyUI install/status/logs/start is available as a managed
  app surface.
- Lemonade is the default managed local engine for assistant/server entry
  points. PyTorch and llama.cpp remain explicit engine choices with GPU-required
  validation.
- Packaging, release-readiness checks, generated-key signature verification,
  installer lifecycle acceptance, first-install PATH setup acceptance, and
  native per-OS release builds.

Left or externally gated:

- Publish repository-owned production signing keys, release secrets, hosted
  signed metadata, hosted recipe indexes, and hosted source-policy metadata.
- Run privileged Linux DKMS driver acceptance on supported hosts with root/sudo
  control.
- Run live ATOM and SGLang GPU acceptance on upstream-supported targets, or
  after upstream support expands to the current host GPU family.
- Expand live GPU-family CI beyond the current local RDNA4/gfx1201 and
  self-hosted adapter detect/capabilities coverage.
- Define a production AMD driver-update feed before wiring real driver-update
  event detection.

## Summary
- Build `rocm-cli` as the ROCm AI Command Center CLI for AMD systems. It should install and manage TheRock runtimes on Linux and Windows, optionally install Linux DKMS drivers through official AMD flows, run local model servers, and provide a chat-first terminal experience for ROCm/TheRock operations.
- Make `rocm` default to an interactive TUI on a real TTY. The TUI should feel like a Codex/Claude-style control room: stable arrow-key panes for browsing, focused overlapping modal cards for decisions/progress/details, and explicit Logs screens for history.
- Keep TheRock as the only managed runtime stream:
  - `release`: latest stable TheRock release
  - `nightly`: latest nightly for the selected platform and GPU family
- Default managed runtime installs to a user-owned `pip` virtual environment backed by TheRock wheels. Keep tarball installs as an explicit alternative, especially for system prefixes such as `/opt/rocm`.
- On Windows, constrain V1 runtime management to `pip` venv installs and existing-driver validation. Use a managed Lemonade serving path as the default native local assistant/server engine, with PyTorch and llama.cpp available as explicit GPU-required engines.
- Use pluggable serving engines instead of hard-coding one runtime. V1 should support `pytorch`, `llama.cpp`, `vllm`, `sglang`, and `atom` through a common plugin contract.
- Use pluggable chat providers:
  - `local`: talk to a locally served model
  - `anthropic`: use Anthropic API when configured
  - `openai`: use OpenAI API when configured
- Implement contained automations and watchers natively inside `rocm-cli`. Take inspiration from OpenClaw's event and hook patterns, but do not depend on OpenClaw itself.

## Product Goals
- Provide a one-line bootstrap install such as `curl -fsSL ... | sh` that installs a signed `rocm` launcher into `~/.local/bin/`.
- Work on Linux and Windows systems with an AMD GPU; CPU-only systems remain
  useful for TUI, diagnostics, and provider-backed chat, but not local model
  serving.
- Work with an existing TheRock installation or install a self-contained TheRock runtime under a user-owned prefix.
- Detect updates every time `rocm` runs and offer to update the CLI, TheRock runtime, engines, or model recipes.
- Translate natural language requests into structured plans and explicit tool executions.
- Make local model serving easy:
  - `serve Qwen3.5 with vllm`
  - `serve a tiny local model on this AMD GPU`
  - `install the latest TheRock nightly for this GPU`
- Keep privileged operations explicit and auditable.

## Non-Goals
- `rocm-cli` is not a replacement for TheRock build-from-source workflows.
- `rocm-cli` is not a generic autonomous agent platform with arbitrary shell access.
- `rocm-cli` should not manage legacy ROCm release streams from `repo.radeon.com`.
- `rocm-cli` should not promise that every model can run on every host. Model fit must be resolved against CPU, VRAM, RAM, engine support, and artifact availability.

## Key Product Decisions
- TheRock-only:
  - Manage only TheRock `release` and `nightly` channels.
  - Legacy ROCm installs may be detected for diagnostics, but they are out of management scope.
- Runtime packaging:
  - Default to `pip` venv installs under a user-owned prefix.
  - Support tarball installs as an explicit alternative.
  - Prefer tarballs for system-style prefixes such as `/opt/rocm`.
- Chat-first:
  - `rocm` with no subcommand launches the TUI.
  - Scriptable subcommands remain for automation and CI.
- Local-first:
  - The default experience should work without any cloud API key.
  - Cloud providers are optional adapters, not the core product.
- Plugins over hard-coded engines:
  - Serving engines are process plugins discovered at runtime.
  - Provider integrations are adapters behind a common chat/runtime interface.
- Native automation subsystem:
  - Implement watchers, policies, sandboxing, and audit logs inside `rocm-cli`.
  - Do not make OpenClaw a required dependency.

## User Experience

### Startup Behavior
- Interactive TTY:
  - `rocm` launches the full-screen TUI.
- Non-interactive shell or explicit subcommand:
  - `rocm examine`
  - `rocm install sdk --channel release`
  - `rocm serve qwen3.5 --engine vllm`
- On startup, `rocm` should:
  - load local config and session state
  - perform a bounded update check using cached manifests and ETags
  - detect host capabilities
  - attach to or start the local supervisor if needed
- `rocmd` lifecycle:
  - default to on-demand startup for foreground CLI and TUI operations
  - remain long-lived only when automations, watchers, or managed background services are enabled

### TUI Goals
- Give the user a single interface for:
  - chat and planning
  - install and update flows
  - serving lifecycle
  - logs and health status
  - watcher events and queued proposals
- Show what the system is doing instead of hiding it behind a spinner.

### TUI Layout

The original transcript-and-status mockup is historical. Current TUI work should
use these surfaces instead:

- First-time setup: a dedicated setup screen before Home, with arrow-key folder
  choices, plain approval cards, live progress, and visible install output.
- Home and slash-command screens: stable left-side choices plus a plain-English
  detail pane. Commands such as `/examine`, `/runtimes`, `/engine`, `/services`,
  `/logs`, `/comfyui`, and `/serve` should open navigable screens, not append
  command dumps to a transcript.
- Focused work: approvals, install/update progress, log details, connection
  details, chat tool results, help, clear, quit, and short errors should use
  overlapping modal cards that own keyboard focus until closed.
- Chat: local assistant sessions can remain conversational, but ROCm command
  results and connection details should be reachable through action rows and
  modal cards rather than dumped inline.

### TUI Interaction Model
- The TUI should be tool-aware, not just conversational.
- Every actionable request becomes:
  1. intent parse
  2. structured plan
  3. approval decision if needed
  4. visible tool execution
  5. summarized result
- The active screen should show:
  - plans as navigable review screens
  - tool calls as approval cards when they change local state
  - live command output while the current operation is running
  - compact final outcomes with a modal/log path for details
  - artifacts such as config diffs, logs, or generated manifests behind
    explicit detail rows or log views

### TUI Modes
- `Ask`
  - explain state, compare engines, recommend models, answer ROCm/TheRock questions
- `Act`
  - run installs, updates, pulls, and serving actions
- `Serve`
  - focus on engine, model, endpoint, and request tests
- `Logs`
  - tail engine logs, install logs, and watcher output
- `Automations`
  - inspect events, policies, queued proposals, and audit history

### TUI Commands and Shortcuts
- Slash commands:
  - `/examine`
  - `/install sdk`
  - `/install driver`
  - `/update`
  - `/engine`
  - `/provider`
  - `/model`
  - `/logs`
  - `/automations`
  - `/plan`
- Keyboard model:
  - `Enter`: send or approve focused action
  - `Esc`: cancel focused modal or proposal
  - `Tab`: cycle panes
  - `Ctrl+L`: clear transient output
  - `Ctrl+R`: refresh status
  - `?`: open shortcuts help

### Approval Model
- No approval needed for:
  - read-only inspection
  - local health checks
  - dry-run planning
  - log viewing
- Explicit approval required for:
  - installing or upgrading TheRock runtime
  - installing Linux DKMS drivers
  - any future Windows driver management actions
  - switching `release` and `nightly`
  - binding public network ports
  - deleting caches or models
  - sending prompts to cloud providers when not already enabled

## CLI Surface

### Primary Commands
- `rocm`
- `rocm examine`
- `rocm install sdk --channel release|nightly [--format wheel|tarball] [--prefix PATH]`
- `rocm install driver --dkms`
- `rocm update`
- `rocm model`
- `rocm engines list`
- `rocm engines install <engine>`
- `rocm serve <model>`
- `rocm chat --provider local|anthropic|openai`
- `rocm dash [--demo | --replay <FILE> | --chat-mock]`
- `rocm automations list`
- `rocm automations enable <watcher>`
- `rocm logs`

### Natural Language Entry
- Input text should be accepted in both TUI and non-interactive form:
  - `rocm "serve qwen3.5 with vllm"`
  - `rocm "install the latest TheRock nightly for this GPU"`
  - `rocm "serve a tiny local model on this AMD GPU"`
- The planner should compile this into structured actions before execution.

## Architecture

### High-Level Components
- `rocm`
  - main CLI and TUI binary
  - request parsing
  - planning
  - approvals
  - local command execution
- `rocmd`
  - optional local supervisor
  - manages long-lived services, watcher execution, and background update checks
  - exposes a local Unix socket or named pipe API to the CLI
  - runs persistently only when background features are enabled; otherwise it may be started on demand and exited when idle
- metadata/index layer
  - release manifests
  - model recipes
  - engine compatibility data
  - provider defaults
- plugin runner
  - launches serving engine plugins in separate processes
  - handles health checks and lifecycle
- sandbox runner
  - executes watcher jobs in a contained environment with structured tools only

### Implementation Language
- Use Rust for the launcher, TUI, supervisor, plugin runtime, and sandbox policy engine.
- Use `ratatui` and `crossterm` for the TUI.
- Prefer subprocess-based plugin protocols over in-process dynamic libraries to avoid ABI problems.

### State Layout

```text
~/.config/rocm-cli/
  config.toml
  profiles/
  automations.toml

~/.local/share/rocm-cli/
  sessions/
  manifests/
  runtimes/
    therock/
      release/
      nightly/
  engines/
    pytorch/
    llama.cpp/
    vllm/
    sglang/
    atom/
  models/
  services/
  logs/
  cache/
  audit/

~/.local/bin/
  rocm
```

### Metadata Model
- TheRock release metadata should remain the source of truth for TheRock artifacts.
- `rocm-cli` should add its own signed metadata index for:
  - engine plugin versions and install rules
  - model aliases and recipes
  - hardware fit heuristics
  - default channel and policy settings
- This lets `rocm-cli` evolve without changing TheRock release assets for every product rule.

### Update Model
- Every `rocm` invocation should perform a quick manifest freshness check.
- Checks should be:
  - bounded by a short timeout
  - cached
  - tolerant of offline use
- Updates should be reported separately for:
  - `rocm-cli`
  - managed TheRock runtime
  - engine plugins
  - model recipes
- Updates should be offered on every run, but not auto-applied by default.

## Install and Runtime Management

### Bootstrap Install
- The one-line installer should:
  - detect host OS and architecture
  - download a signed `rocm` launcher
  - place it in `~/.local/bin/`
  - write a minimal config file if none exists
  - verify checksum or signature before activation
  - set PATH automatically for future terminals, and for the current Windows
    PowerShell process when the installer is invoked directly
  - avoid requiring ROCm, Python, Rust, or Cargo to be installed beforehand
- The bootstrap should not install drivers or large runtimes by default.

### Host Detection
- `rocm examine` should detect:
  - OS, kernel, distro, architecture
  - CPU features
  - AMD GPU presence and GPU family
  - existing TheRock runtime
  - legacy ROCm installs for compatibility reporting
  - existing engines
  - model cache state
  - permissions required for driver work

### TheRock Runtime Install
- Install modes:
  - adopt existing TheRock runtime
  - install a managed `pip` virtual environment under `~/.local/share/rocm-cli/runtimes/`
  - install a managed tarball under a user-selected prefix such as `/opt/rocm`
  - switch between managed `release` and `nightly` installs
- Default behavior:
  - use a `pip` venv install for user-scoped managed runtimes
  - keep runtime activation and Python package state isolated per installed version
  - use tarball installs only when explicitly requested or when targeting a system-style prefix
- The runtime manager should:
  - resolve the correct artifact by OS, architecture, and GPU family
  - prefer TheRock wheel sets for user-managed installs
  - support tarball installs for system-style layouts and non-venv deployment targets
  - maintain side-by-side versions for rollback
  - update symlinks for the active runtime atomically
- Windows V1 constraints:
  - support managed `pip` venv installs only
  - do not support tarball-based system installs in early phases
  - assume the system AMD driver is already installed
  - focus on TheRock ROCm and PyTorch wheel flows

### Driver Management
- Linux only.
- Driver installation should wrap official AMD install flows instead of reimplementing them.
- Responsibilities:
  - validate distro and kernel support
  - validate root or `sudo` availability
  - explain reboot requirements
  - execute the official DKMS install path
  - record driver state and last-known version
- Non-goals:
  - custom DKMS packaging logic
  - silent kernel changes
- Windows:
  - assume an AMD driver is already installed
  - detect and report driver presence and compatibility in `examine`
  - defer any Windows driver installation or upgrade flow to a later phase

### Existing Installations
- If an existing TheRock runtime is found, `rocm-cli` should allow:
  - adopt as external runtime
  - import into managed state
  - leave unmanaged and use read-only
- If only a legacy ROCm runtime is found, `rocm-cli` should:
  - report it in `examine`
  - warn that management and upgrades are out of scope
  - still allow CPU-only or provider-backed chat flows

## Serving and Chat Backend Architecture

### Engine Plugin Contract
- Every serving engine plugin should be a separate executable discovered from a plugin directory.
- Plugin protocol should be JSON-RPC over stdio or a local socket.
- Required plugin methods:
  - `detect`
  - `install`
  - `capabilities`
  - `resolve_model`
  - `launch`
  - `healthcheck`
  - `endpoint`
  - `stop`
  - `logs`
- Reported capabilities should include:
  - `cpu`
  - `rocm_gpu`
  - `multi_gpu`
  - `openai_compatible`
  - `tool_calling`
  - `quantized_models`
  - `distributed_serving`
  - `reasoning_parser`

### Built-In Engine Priorities
- `lemonade`
  - default local assistant/server engine
  - use ROCm GPU execution only; no CPU or Vulkan fallback
- `pytorch`
  - explicit managed Windows/Linux serving engine
  - built on TheRock PyTorch wheels with a `rocm-cli` managed serving wrapper
  - should expose the same normalized endpoint contract as the other engines
- `llama.cpp`
  - ROCm GPU path for quantized GGUF local models
  - fail loudly when ROCm GPU execution is unavailable
- `vllm`
  - high-throughput ROCm GPU serving path on supported Linux/WSL hosts
- `sglang`
  - reasoning- and router-friendly engine option
- `atom`
  - AMD-optimized path for select hardware and recipes
  - treat as experimental until packaging and compatibility mature
- Windows engine policy:
  - default to managed `lemonade` serving with ROCm GPU required
  - keep `pytorch` and `llama.cpp` available for explicit GPU-required workflows
  - use GPU execution through TheRock PyTorch on supported Windows systems when a usable AMD driver is present
  - do not make `vllm`, `sglang`, or `atom` native Windows GPU serving part of the V1 promise

### PyTorch Serving Engine
- The `pytorch` engine should be a first-party `rocm-cli` plugin, not a dependency on TorchServe.
- Detailed spec: `plans/rocm-cli-pytorch-engine-spec.md`
- Responsibilities:
  - create and manage a TheRock PyTorch virtual environment
  - install compatible model-serving dependencies such as `transformers`, tokenizers, and runtime helpers
  - serve selected models through the same local API contract used by the other engines
  - prefer GPU execution on Windows when a supported AMD driver and TheRock PyTorch wheel set are present
  - fail loudly when no usable ROCm GPU path is available
- Initial scope:
  - single-model local serving
  - chat and text-generation workloads
  - streaming responses
  - OpenAI-compatible local endpoint surface where practical
  - straightforward batching, not high-throughput distributed scheduling

### Provider Adapter Contract
- Provider adapters should implement:
  - `chat`
  - `stream_chat`
  - `models`
  - `tool_call_schema`
  - `auth_status`
- Providers:
  - `local`
  - `anthropic`
  - `openai`
- The chat planner should use the same tool schema no matter which provider is active.

### Model Registry
- Model recipes should map:
  - canonical model id
  - aliases such as `qwen3.5`
  - engine preferences
  - supported quantizations
  - minimum memory and VRAM expectations
  - required flags or parser settings
  - preferred endpoint settings
  - known unsupported combinations
- The registry should support:
  - family aliases
  - size variants
  - GPU fit explanations
  - explicit provider-backed alternatives when local GPU serving is unavailable

### Request Resolution Flow
- For a prompt like `serve Qwen3.5 with vllm`, the resolver should:
  1. parse action intent
  2. normalize model alias
  3. detect available engines and hardware
  4. check whether the requested combo is feasible
  5. produce a concrete plan
  6. ask for approval if changes are needed
  7. execute tools and stream results
- The resolver should not rely entirely on an LLM.
- Use a hybrid approach:
  - grammar and alias-based parsing for known actions
  - optional LLM planner for ambiguity resolution
  - final execution only from structured tool calls

## Automations and Watchers

### Design Goals
- Let `rocm-cli` observe state and react to events in a controlled way.
- Keep autonomy narrow, auditable, and revocable.
- Use structured tools, not arbitrary shell.

### Event Sources
- scheduler and cron-like timers
- managed service lifecycle events
- update checks
- health checks
- GPU metrics when available
- model endpoint health
- local webhook triggers

### Policy Levels
- `observe`
  - detect and notify only
- `propose`
  - build a plan and queue it for review in the TUI
- `contained`
  - auto-run only allowlisted low-risk actions inside the sandbox

### Examples
- `therock-update`
  - check for a newer release or nightly and notify
- `server-recover`
  - if a managed model server crashes, collect logs and restart it
- `gpu-thermal-protect`
  - if temperature or memory pressure is too high, stop or reduce serving load
- `cache-warm`
  - prefetch model weights or engine artifacts during idle periods

### Sandbox Design
- V1 Linux sandbox should use a rootless isolation tool such as `bubblewrap`.
- Worker constraints:
  - read-only root filesystem where possible
  - private temp and work directories
  - no arbitrary home directory access
  - no network unless the watcher explicitly needs it
  - no GPU device access unless required
  - process, CPU, RAM, and wall-clock limits
- Sandboxed jobs should not receive raw shell access.
- They should invoke a restricted internal tool API such as:
  - `check_updates`
  - `examine_snapshot`
  - `list_servers`
  - `restart_server`
  - `stop_server`
  - `prefetch_artifact`
  - `notify_user`

### Watcher Config Shape

```toml
[automation]
enabled = true
default_mode = "propose"

[[watchers]]
id = "therock-update"
on = "schedule:0 */6 * * *"
then = ["check_updates", "notify_if_newer"]
mode = "observe"

[[watchers]]
id = "server-recover"
on = "event:server.crashed"
then = ["collect_logs", "restart_server", "healthcheck"]
mode = "contained"

[[watchers]]
id = "driver-upgrade"
on = "event:update.available"
if = "component == 'driver'"
then = ["prepare_install_plan"]
mode = "propose"
```

### TUI Integration
- The TUI should expose an `Automations` view that shows:
  - active watchers
  - last event time
  - current policy
  - last outcome
  - queued proposals
  - audit history

## Security and Trust Model
- Separate read-only inspection from mutating actions.
- Require user approval for anything privileged or disruptive.
- Keep automation actions capability-scoped.
- Keep watcher execution in a sandbox with a narrow API.
- Keep secrets out of plugins unless explicitly granted.
- Store provider keys in OS keychain facilities when available, with file-based encrypted fallback if needed.
- Maintain an audit log for:
  - install actions
  - update actions
  - watcher-triggered actions
  - provider usage
  - service lifecycle changes

## Platform Support Matrix

| Host | Scope | Notes |
|------|-------|-------|
| Linux x86_64 with AMD GPU | Full V1 target | TUI, TheRock runtime, drivers, serving, automations |
| Linux x86_64 CPU-only | Supported non-serving host | TUI, provider chat, artifacts, no driver work, no local model serving fallback |
| Windows x86_64 with AMD GPU | Supported in early phases | TUI, chat, TheRock `pip` runtime, driver validation, Lemonade default local serving, explicit PyTorch/llama.cpp local serving, provider chat; assume driver already installed, no driver installer in V1 |
| Windows x86_64 CPU-only | Supported non-serving host | TUI, provider chat, TheRock wheel management where applicable, no local model serving fallback |
| macOS | Deferred | Provider-backed chat only unless future TheRock/engine artifacts make more possible |

## Initial Windows V1 Support Table

| Capability | Windows V1 status | Notes |
|------------|-------------------|-------|
| `rocm` CLI and TUI | Supported | Native Windows terminal experience is in scope |
| `rocm examine` | Supported | Detect host, GPU family, driver presence, and TheRock runtime state |
| TheRock runtime install | Supported | `pip` venv only |
| TheRock tarball install | Deferred | Do not target `Program Files`-style system installs in V1 |
| Windows driver install/upgrade | Deferred | Assume driver already installed; report compatibility only |
| Provider chat (`local`, `anthropic`, `openai`) | Supported | Same chat surface as Linux |
| Local serving via Lemonade | Supported | Default native local assistant/server engine with ROCm GPU required |
| Local serving via TheRock PyTorch wheels | Supported | Explicit managed PyTorch engine; ROCm GPU required |
| `llama.cpp` local serving | Supported GPU engine | Use for quantized GGUF workflows with ROCm GPU execution; no CPU fallback |
| Native Windows ROCm GPU serving via `vllm` | Deferred | vLLM upstream is Linux-only today |
| Native Windows ROCm GPU serving via `sglang` | Deferred | ROCm guidance is Linux and Docker oriented |
| Native Windows ROCm GPU serving via `atom` | Deferred | ATOM currently documents ROCm + Docker on Linux |
| WSL/remote Linux bridge | Later option | Not part of the native Windows V1 promise |

## Initial Windows GPU Target Priorities

| Windows GPU target | V1 priority | Rationale |
|--------------------|-------------|-----------|
| `gfx1151` | Primary | Marked release-ready on Windows in TheRock support tracking |
| `gfx1200`, `gfx1201` | Supported preview | Build passing and sanity-tested on Windows in TheRock support tracking |
| `gfx1150`, `gfx1030` | Experimental | Build passing or sanity-tested, but not release-ready |
| `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103` | Detect and allow manual use | Build passing or partial sanity state; not part of the first promised Windows matrix |
| `gfx1010`, `gfx1011`, `gfx1012`, `gfx906` | Detect only | Present in tracking, but not good V1 commitments for native Windows experience |

## Phased Implementation Plan

### Phase 0: Product Skeleton
- Create the repo layout and Rust workspace.
- Implement config loading, state directories, logging, and signed manifest fetch.
- Implement `rocm examine` with host detection and a plain CLI output path.
- Exit criteria:
  - `rocm examine` works on supported Linux and Windows hosts
  - bootstrap install downloads and runs the launcher

### Phase 1: Runtime Management
- Implement TheRock release and nightly manifest resolution.
- Implement self-contained runtime install, side-by-side versioning, and rollback.
- Add update checks on every invocation.
- Exit criteria:
  - user can install and activate a managed TheRock runtime
  - Windows uses managed `pip` venv installs only
  - update prompts appear consistently

### Phase 2: TUI Foundation
- Build the full-screen TUI shell.
- Add transcript pane, sidebar, log drawer, and composer.
- Implement slash commands and plan cards.
- Exit criteria:
  - `rocm` launches a usable TUI
  - plans and tool output stream correctly

### Phase 3: Engine Plugin MVP
- Define the engine plugin protocol.
- Ship built-in plugin adapters for:
  - `pytorch`
  - `llama.cpp`
  - `vllm`
- Add engine install and service lifecycle management.
- Exit criteria:
  - Windows local model serving works through `pytorch`
  - quantized GGUF serving works through `llama.cpp` with ROCm GPU execution
  - ROCm GPU serving works through `vllm` on Linux

### Phase 4: Chat Providers and NL Planning
- Implement provider adapter contracts for:
  - `local`
  - `anthropic`
  - `openai`
- Add hybrid parser and tool-call planner.
- Make TUI the default chat experience.
- Exit criteria:
  - `rocm` can answer questions and produce action plans
  - local and cloud providers share the same tool-execution path

### Phase 5: Model Registry and Recipe Engine
- Ship a signed model/engine recipe index.
- Add alias resolution, fit estimation, and explicit alternatives for
  unsupported or unfit model choices.
- Support natural language requests for common serve/install operations.
- Exit criteria:
  - model requests resolve deterministically
  - unfit model choices produce useful alternatives

### Phase 6: Driver Management
- Add Linux driver install wrapper around official AMD DKMS flows.
- Add Windows driver detection and compatibility reporting only.
- Add approval UI, reboot-required tracking, and health checks.
- Exit criteria:
  - driver install plans are explicit
  - `rocm-cli` can complete and verify a supported Linux DKMS install path
  - `rocm-cli` can verify that a usable Windows driver is present without trying to install it

### Phase 7: Expanded Engine Plugins
- Add `sglang` and `atom` plugin adapters.
- Add engine-specific recipes and health checks.
- Exit criteria:
  - all five engines share the same install, launch, and endpoint lifecycle on supported host platforms

### Phase 8: Automations and Watchers
- Implement event bus, watcher configs, policy engine, sandbox runner, and TUI audit panel.
- Ship a small starter set of watchers.
- Exit criteria:
  - watcher jobs are visible, policy-controlled, and auditable
  - contained auto-restart and update-notify flows work end-to-end

### Phase 9: Hardening and Release
- Add packaging, signatures, rollback validation, telemetry policy, and CI coverage.
- Test against supported Linux GPU families, supported Windows GPU families, and CPU-only non-serving hosts.
- Exit criteria:
  - bootstrap install, runtime install, serving, update, and watcher flows are stable
  - release process can ship signed CLI and metadata updates

## MVP Definition
- `rocm` launches a TUI on Linux and Windows.
- `rocm examine` detects host and runtime state.
- `rocm install sdk --channel release|nightly` installs a managed TheRock runtime.
- `rocm update` prompts for newer CLI/runtime versions.
- `rocm serve` can launch:
  - `pytorch` using TheRock PyTorch wheels on Windows
  - `llama.cpp` with ROCm GPU execution for quantized GGUF models
  - `vllm` on ROCm GPU on Linux
- `rocm` chat mode can use:
  - `local`
  - `anthropic`
  - `openai`
- `rocm automations` supports at least:
  - update notifications
  - managed server crash recovery

## Open Questions
- How much model metadata should live in TheRock-owned release assets versus a `rocm-cli` metadata index?
- What is the minimum supported Linux distro matrix for driver install support?
- What are the exact rules for switching between a managed `pip` venv runtime and a tarball-installed system prefix on the same machine?
- Should remote provider chat be enabled by explicit opt-in on first use, even when API keys are already present in the environment?

## Recommended Immediate Next Steps
- Keep the local-assistant service lifecycle on the regression list rather
  than treating it as pending implementation: Windows and WSL live acceptance
  passed with the verified `qwen`/Qwen2.5 PyTorch recipe and ROCm tool-call
  bridge. Re-run `scripts/local_assistant_therock_gpu_test.py` when changing
  provider, tool-call, chat handoff, or managed-server paths.
- Publish repository-owned production signing keys and hosted metadata/model
  recipe/source-policy indexes.
- Run privileged Linux driver acceptance on supported hardware with explicit
  root/sudo control.
- Expand GPU-family CI beyond the current local RDNA4/gfx1201 and existing
  self-hosted adapter smoke coverage.
