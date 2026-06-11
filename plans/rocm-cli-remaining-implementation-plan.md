# rocm-cli Remaining Implementation Plan

This tracker consolidates the current repo audit against:

- `plans/rocm-cli-implementation-plan.md`
- `plans/rocm-cli-pytorch-engine-spec.md`

It is intentionally scoped to plan-derived work. The only non-product choice in this file is keeping the remaining-work tracker separate from the canonical implementation plan so the original plan remains readable.

Current scope clarification: early references in older plan/spec text to CPU
serving or CPU fallback are superseded by the current no-fallback policy. They
are not open implementation gaps. Local serving paths must require ROCm GPU
execution and fail loudly when a managed TheRock ROCm runtime or supported GPU
path is unavailable.

## Status Legend

- `Implemented`: the repo appears to satisfy the plan item.
- `Partial`: a skeleton or useful subset exists, but exit criteria are not met.
- `Missing`: no first-party implementation was found.

## Phase Status

| Phase | Status | What Exists | Main Remaining Plan-Derived Gaps |
|---|---:|---|---|
| Phase 0: Product Skeleton | Partial | Rust workspace, app/crate layout, config/state dirs, deterministic first-time setup/bootstrap, bootstrap installers that verify signatures/checksums, seed a minimal config without overwriting user settings, require no preinstalled ROCm/Python/Rust/Cargo, and set PATH automatically for future terminals plus the current directly-invoked Windows PowerShell process, expanded bounded-latency `rocm doctor` host/driver/runtime/cache report, fast setup/Doctor GPU detection through native Windows registry/system APIs, native Linux KFD/sysfs/IP discovery, and one-shot WSL host display probing when needed, WSL managed-TheRock SDK gfx detection when sysfs/PATH tools are unavailable, active runtime state with explicit ambiguous-runtime reporting, adapter/plugin inventory, registered runtime inventory, CLI lifecycle audit records plus text lifecycle logs for update/install/runtime/engine/service actions, quiet bounded startup update check, optional signed metadata cache verification, and a workspace-local Rust/Cosmopolitan single-file rocm-cli artifact that runs directly on Windows and through the Linux path on WSL via `sh <artifact>` without sibling rocm-cli helper binaries | Production signed metadata publication, promotion of the Rust/Cosmopolitan build into the release pipeline, native-Linux validation, and direct WSL execution hardening if WSLInterop intercepts `./rocm` |
| Phase 1: Runtime Management | Partial | TheRock release/nightly resolution, pip venv installs using a single TheRock-index pinned `rocm[libraries,devel]` plus `torch`/`torchvision`/`torchaudio` transaction selected by exact ROCm wheel suffix for the current Python/platform, Windows pip-only enforcement, tarball flow on non-Windows, localized pip cache, cached TheRock index metadata with ETag, optional detached metadata signature verification, quiet freshness-gated startup update check, explicit runtime update apply/dry-run flow, versioned side-by-side runtime keys, active runtime activation marker, previous-runtime validation, read-only import/adopt, legacy ROCm migration guidance, explicit `rocm update` report with CLI/engine/recipe surface inventory | Production metadata public key and signed sidecar publication |
| Phase 2: TUI Foundation | Implemented | Ratatui shell, typed transcript/status/prompt/sidebar, first-view `/home` dashboard with arrow-key status/action cards, internal activity buffer seeded from persisted lifecycle logs, first-time setup flow with visible arrow-key/folder-change guidance, sidebar mode state (`ask`, `act`, `serve`, `logs`, `automations`), vertical slash completion with cycling/scrolling, plain command-shaped prompt inputs route to navigable command screens before natural-language planning, `/engine` singular surface, `/runtimes` ROCm install picker with list/activate/uninstall/import/adopt and legacy typed rollback guidance back to explicit install selection, `/update --apply` approval/preview flow, streamed command output, overlapping running-progress cards for contained CLI/proposal/service work, idle Ctrl-C quit confirmation, unknown slash-command help cards, overlapping approval cards with Enter/Y approval plus edit/cancel flow for mutating commands and queued review requests, `/reviews` review-request alias with hidden legacy `/proposals` compatibility, provider-key entry/clear screens with safe default cancel, `/logs` action-log listing, lifecycle tail, hidden-by-default file locations, detail scrolling, bounded query search, stateful arrow-key/Left-Right/PageUp/PageDown pagination, and `/logs follow [query]` live refreshed log following over recent lifecycle/action logs | None known within V1 plan constraints |
| Phase 3: Engine Plugin MVP | Implemented | Protocol types, PyTorch engine binary, `llama.cpp` adapter binary, Lemonade adapter for strict ROCm serving, `vllm` external-runtime adapter for Linux/WSL ROCm GPU serving, `detect/install/capabilities/resolve_model/launch/endpoint/healthcheck/stop/logs`, first-party stdio protocol routing coverage, foreground/managed launch, plugin-directory discovery helper, external plugin directory policy surfaced in CLI/TUI/docs, non-TUI `rocm services` list/logs/stop/restart parity with living services shown by default and failed/stopped history available through `--all`, active runtime wiring, shared recipe resolver before default engine selection, TheRock SDK env propagation for non-PyTorch HIP apps, explicit CPU-policy rejection for PyTorch/vLLM/Lemonade without implicit CPU fallback, Windows and WSL PyTorch GPU smoke coverage, Windows and WSL llama.cpp GPU smoke coverage, Windows and WSL Lemonade ROCm smoke coverage, WSL vLLM TheRock GPU smoke coverage, and in-process first-party engine/helper routes for the single-file universal binary | None known within V1 plan constraints |
| Phase 4: Chat Providers and NL Planning | Implemented | Provider adapter contract, local status/chat/live streaming, TUI `/chat [prompt]` streaming transcript integration, `/chat <message>` persistent local served-model chat when a ready managed local server exists, served-model TUI chat sessions that keep the selected local model pinned and keep ROCm tool results/approvals in the chat surface, expanded served-model ROCm bridge tools for full state snapshots, service logs, automations, natural-language plans, update checks, stop-server requests, and watcher controls, visible progress for tool-enabled chat checks, chat-originated approval cards that preserve the assistant's reason, approved chat-requested Install/Engine/Serve/Services/Automations results with inline recent output plus Logs pointers and one automatic assistant follow-up turn, real OpenAI/Anthropic HTTPS request/response adapters, real OpenAI/Anthropic streaming requests with SSE delta callbacks, explicit cloud-provider prompt opt-in, OS secure-store provider-key save/clear/status, in-process remote provider HTTP to avoid API-key argv leakage, unified tool schema id, hybrid deterministic parser/tool-call planner for common freeform serve/install/update/driver/uninstall/inspect requests, active model recipe registry aliases/preferences for context-aware freeform serve plans, optional provider-assisted ambiguity resolution through `config set-planner-provider`, TUI approval handoff for recognized mutating natural-language plans, non-interactive `rocm --yes <natural language request>` execution of deterministic final structured tool calls after flag-shaped/structured-looking request and placeholder validation | None known within V1 plan constraints |
| Phase 5: Model Registry and Recipe Engine | Partial | Shared built-in recipe skeleton with aliases, dtype/device policy, engine preferences, warnings, RAM/quantization/artifact guidance, manual alternatives, PyTorch resolver integration, non-TUI `rocm model`/`rocm models` registry listing, TUI `/model` recipe listing/completion, cached-VRAM fit, host system RAM fit, engine-support explanations, signed external recipe index schema/loading with detached signature verification, metadata-scoped artifact descriptors, explicit artifact `source_policy` records with validation/rendering/enforcement, engine-specific recipe records mapped into a versioned adapter hint during model resolution and launch, local artifact cache marker telemetry, approved direct HTTP(S) artifact prefetch with size/sha256 verification, and authenticated Hugging Face prefetch for gated recipe-registry artifacts with HTTPS plus token host-scope enforcement | Hosted production recipe index/key publication, hosted source-policy publication, and any additional source-specific authenticated download flows required by production indexes |
| Phase 6: Driver Management | Partial | `install driver` plan/preflight surface with explicit root/`sudo -v` availability checks for Linux DKMS plans, Windows validate-only driver detection/reporting, WSL ROCDXG preflight/setup helper, AMD-documented Ubuntu/Debian/RHEL/Oracle Linux/Rocky/SLES package-manager DKMS plans, TUI approval-to-`--yes` execution boundary, driver execution state/reboot marker, post-reboot reconciliation command, persisted passive post-check summary | Live privileged install acceptance coverage on supported Linux hosts |
| Phase 7: Expanded Engine Plugins | Partial | Packaged `atom`, `vllm`, and `sglang` adapters for Linux/WSL external runtimes; installer bundle validation requires `atom`, `vllm`, and `sglang`; inventory and model-runtime text for `atom`, `vllm`, and `sglang`; Windows runtime gating fails explicitly without CPU fallback; live WSL vLLM GPU acceptance against a rocm-cli managed TheRock runtime; ATOM adapter launches the upstream `atom.entrypoints.openai_server` module and propagates TheRock SDK env/library paths for non-PyTorch HIP apps; ATOM now has an opt-in GPU acceptance harness with offline exact-runtime selector self-tests; SGLang managed-runtime parity records managed TheRock env and now has an offline exact-runtime selector acceptance harness; live SGLang on this WSL `gfx1201` host is blocked by upstream SGLang ROCm kernel support (`gfx942`/`gfx950` only in `sgl-kernel/setup_rocm.py`) | Live ATOM GPU acceptance on supported Linux/WSL hardware/runtime, live SGLang GPU acceptance on supported SGLang ROCm hardware or after upstream adds host gfx support |
| Phase 8: Automations and Watchers | Partial | Built-in `therock-update`, `server-recover`, read-only `gpu-metrics`, reviewed `gpu-thermal-protect`, reviewed `cache-warm`, and reviewed `driver-upgrade`, enable/disable/list, daemon tick loop, automation trigger-event substrate for existing schedule, manifest, healthcheck, endpoint-health, bounded local GPU telemetry, exact `gpu.thermal_pressure` / `gpu.memory_pressure` reviewed stop proposals, exact `cache.warm` registry-verified prefetch proposal events with reviewed source-policy edit fields, exact `update.available` driver-plan proposal events or contained restricted dry-run driver plans, and loopback-only local webhook sources, automation events JSONL, audit mirroring, provider chat/stream usage audit records, proposal lifecycle JSONL, TUI proposal review/approve/reject/edit, sandbox runner, restricted internal tool API, contained read-only TheRock update checks with `notify_if_newer` notification audit records, recovery policy for failed/exited/unreachable and stale starting/recovering services, explicit watcher mode-to-policy routing for existing watchers | Broader event-source coverage beyond existing watchers and richer contained-mode execution for future mutating actions |
| Phase 9: Hardening and Release | Partial | Linux/Windows package scripts, release/nightly assets, SHA-256 verification, detached signature generation/verification, stable release CI signature enforcement, pre-upload release-readiness verification of stable and nightly bundle contents/checksum/signature sidecars plus installer-facing aliases and exact dist asset sets, staged nightly publication that preserves the previous public nightly until Linux and Windows assets are ready, Windows and Linux acceptance signature rejection before activation including missing sidecars and required-signature-without-public-key failures, generated private-key PEM packaging and public-key PEM installer verification, installer minimal-config seeding/no-overwrite acceptance, first-install PATH setup acceptance, previous-runtime validation tests, explicit local/off telemetry policy config, optional Rust signed metadata cache verification with generated-key tamper coverage, signed model recipe index verification with generated-key tamper coverage, cross-platform local no-fallback smoke coverage, Linux/Windows CI all-target test coverage, CI release-readiness and Python acceptance-harness self-test coverage, CI clippy with warnings denied, self-hosted GPU CI adapter detect/capabilities coverage for all first-party engines, and workspace-local Rust/Cosmopolitan universal-binary build plus release-gate scripts | Repository-owned signing and metadata key publication, broader GPU CI coverage, native-Linux single-exe validation, and wiring the universal-binary release gate into production publishing |

## Current Remaining Gates

Audited on 2026-06-06 after the Windows and WSL TheRock/PyTorch/Lemonade/
ComfyUI acceptance checkpoints, the Rust/Cosmopolitan universal-binary
checkpoint, the TUI slash-command navigability follow-ups, the
installer/driver/ATOM packaging follow-up, the CI hardening follow-up, the
focused runtime import/adopt plus sandbox/contained automation
re-verification, and the structured command-routing/local-server wording
checkpoint. A previous live service-log audit also found that the current
`Qwen/Qwen3.5-4B` PyTorch recipe is not a verified working local-assistant
path on the managed Transformers package line; the follow-up recipe/resolver
patch now routes `qwen` to a verified Qwen2.5 instruct model and gates explicit
Qwen3.5 PyTorch launches before service startup.

Open release caveats from the current working tree:

- The current single-file artifact is
  `.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe`; release
  packaging should expose it as `rocm.exe` on Windows and the same bytes as
  `rocm` for Linux/WSL.
- Windows smoke passes by running the artifact directly.
- WSL Linux-path smoke passes through `sh ./rocm`. Direct renamed `./rocm`
  inside WSL can still be intercepted by WSLInterop before rocm-cli starts.
- Native Linux validation and production release-pipeline promotion remain
  open.
- TUI first-view output must stay minimal. Only active foreground pip/install
  progress cards may be verbose; finished engine/service/assistant screens
  should not dump recent logs unless the user opens details.

Locally implemented and recently verified:

- Managed TheRock SDK installs on Windows and WSL using rocm-cli-owned Python
  environments, with pip cache inside the selected ROCm install folder. Fresh
  live validation after the `rocm[libraries,devel]` resolver change passed on
  Windows (`7.13.0a20260511`, `cp312-win_amd64`) and WSL
  (`7.13.0a20260513`, `cp312-linux_x86_64`).
- PyTorch engine install and GPU smoke tests on Windows and WSL using exact
  TheRock torch stack pins from the selected managed runtime.
- Lemonade is now the default local assistant/server engine. Windows and WSL
  Lemonade ROCm serving have live no-fallback acceptance through the universal
  binary. On WSL, rocm-cli uses Lemonade-packaged ROCm `llama-server` directly
  rather than Lemonade's router when the router cannot detect `/dev/dxg`.
- `llama.cpp` GPU serving on Windows and WSL with `gpu_required`, verifying
  HIP/BLAS libraries are loaded from the rocm-cli-managed TheRock runtime.
  The user-facing `llama.cpp` engine is backed by upstream `llama-server`.
- Local assistant tool-calling now includes a validated `rocm_command` bridge
  for read-only status/log checks and approval-routed mutating actions,
  including ComfyUI status/logs/install/start/stop and `llama.cpp`/`llama-server`
  managed serving.
- ComfyUI is now a managed app surface:
  `rocm comfyui status|logs|install|start|stop` and `/comfyui`
  status/logs/install/start routes download ComfyUI, install the packages it
  needs while preserving the managed ROCm GPU package stack, stream install
  output, render recent logs inline, show/open the local URL, report stale
  saved state as stopped instead of running, and have an opt-in live GPU
  acceptance harness with an offline CI self-test. The live harness now submits
  a real safetensors-backed cat image workflow through the ComfyUI HTTP API and
  passes on Windows and WSL.
- The single-file Rust/Cosmopolitan artifact
  `.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe` has passed
  Windows direct smoke, WSL Linux-path smoke through `sh ./rocm`, TUI smoke,
  PyTorch local-assistant E2E, Lemonade local-assistant E2E, and ComfyUI
  install/start/status/stop E2E using temporary state roots.
- WSL ROCDXG readiness, WSL doctor gfx/family detection, and WSL acceptance
  commands with isolated workspace-local state.
- TUI slash-command surfaces are navigable instead of transcript dumps; the
  latest focused pass covers arrow-key selection after accidental typing,
  provider-key safe clear defaults, two-step automation review decisions,
  manager detail scrolling, guided `/model` and `/install` completion labels,
  and `/logs`/service-log detail scrolling with file locations hidden until
  explicitly revealed.
- TUI focused-card pass now covers idle Ctrl-C as a Quit confirmation instead
  of immediate exit, unknown slash commands as an overlapping help card instead
  of a replacement screen, compact overlapping running-progress cards with a
  text progress bar and live output for contained CLI/proposal/service work,
  modal focus for those progress cards so hidden screen rows and completions do
  not move behind the card, and first-view command/chat tool results that point
  to Logs without exposing raw log paths.
- TUI prompt submission now routes command-shaped plain inputs such as
  `install sdk --channel release`, `update --apply --runtime ...`,
  `uninstall apply`, and `serve tiny.gguf --engine llama.cpp` to their
  navigable command screens before the natural-language planner sees them,
  while ambiguous plain-English requests such as `install ROCm` and
  `serve Qwen with llama.cpp` remain plan requests.
- Local server surfaces now distinguish live readiness from history:
  `rocm services` and the TUI status pane show living services only by default,
  while `rocm services list --all` exposes saved failed/stopped history.
- Bootstrap installers now require the packaged ATOM adapter before
  activation, seed a minimal config file when none exists, preserve existing
  config files across reinstall, require no preinstalled ROCm/Python/Rust/Cargo,
  and set PATH automatically. Windows persists the user PATH and updates the
  installer PowerShell process; Linux/WSL writes the shell profile and explains
  the new-terminal step after `curl | sh`. Windows acceptance and focused WSL
  installer tests verify those paths.
- `rocm doctor` and first-time setup now use fast host GPU detection on the
  normal path: Windows avoids PowerShell/CIM by reading registry/system APIs,
  native Linux uses sysfs/KFD/IP discovery before ROCm userland tools, and WSL
  uses a single host display bridge query only when Linux-side files cannot
  answer.
- Linux DKMS driver plans now surface explicit root/`sudo -v` and package
  manager preflight checks before any approval path, with documented
  package-manager plans for Ubuntu, Debian, RHEL, Oracle Linux, Rocky, and SLES.
- Linux and Windows CI now run `cargo test --workspace --all-targets`, and CI
  clippy now denies warnings with
  `cargo clippy --workspace --all-targets -- -D warnings`.
- Linux and Windows CI now run the release-readiness self-test, so verifier
  regressions for exact asset sets, stale assets, sidecars, and production
  trust input gates are caught before release/nightly publishing jobs.
- Linux and Windows CI now include the ComfyUI acceptance-harness offline
  self-test alongside the PyTorch, llama.cpp, local-assistant, vLLM, SGLang,
  ATOM, and WSL preflight harness self-tests.
- Release packaging now has a cross-platform pre-upload readiness verifier that
  checks archive structure, mandatory bundle files, checksum sidecar content,
  required signature sidecars, stable/nightly installer-facing aliases, and
  exact dist asset sets before publishing assets.
- Nightly publication now uses a staging release and re-verifies the combined
  Linux plus Windows asset set before replacing the public `nightly` release.
- Runtime import/adopt and sandbox/contained automation surfaces were
  re-verified on Windows and WSL with focused `rocm`, `rocmd`, and vLLM
  self-test filters on 2026-06-03.
- SGLang now has an opt-in Python GPU acceptance harness with offline
  exact-runtime selector self-tests; live SGLang remains gated on upstream
  ROCm kernel support for the host GPU target.
- PyTorch local-assistant recipe compatibility follow-up: the managed
  Transformers 4.57.6 line was verified to reject `Qwen/Qwen3.5-4B` with
  unknown architecture `qwen3_5`, while Qwen2.5-class models load through
  TheRock PyTorch on the AMD GPU. The built-in `qwen` alias now resolves to
  `Qwen/Qwen2.5-1.5B-Instruct` so the default helper is smarter than the 0.5B
  smoke path while still targeting <=8 GB APUs; `qwen-tiny` is the explicit
  0.5B smoke alias. The PyTorch adapter rejects explicit Qwen3.5 requests
  before launch with a clear no-fallback compatibility message.
- Live local-assistant acceptance now passes on Windows with the `qwen` alias:
  `scripts/local_assistant_therock_gpu_test.py --model qwen --require-tool-call`
  launched the managed PyTorch GPU-required service, reached the local
  provider, required a ROCm tool call, and observed the model use `rocm doctor`
  through the tool bridge before stopping the managed service.
- Live local-assistant acceptance now also passes on WSL with the `qwen` alias
  after path propagation fixes for managed child processes:
  `scripts/local_assistant_therock_gpu_test.py --rocm
  /home/jam/.cache/rocm-cli-target/debug/rocm --model qwen --require-tool-call`
  launched the managed PyTorch GPU-required service, wrote its service
  manifest under `/home/jam/.rocm/services`, required a ROCm tool call, and
  stopped the managed service cleanly.
- ComfyUI TUI logs/file-location polish is now covered by regression tests:
  `/comfyui logs` and `/comfyui log-files` keep the user on a navigable
  ComfyUI screen, file locations are shown only in an explicit overlapping
  detail card, Esc closes that card first, and ComfyUI start approval text uses
  the actual requested loopback address.
- WSL lightweight acceptance currently passes for the Python self-tests plus
  WSL `rocm doctor`, `rocm engines list`, `rocm comfyui status`, and
  `rocm services list`. The first WSL sidecar found a stale `/mnt/d` debug
  binary that rejected newer `comfyui`/`services` subcommands; a forced rebuild
  into `/home/jam/.cache/rocm-cli-target` fixed the artifact and verified those
  subcommands. Live WSL PyTorch local-assistant serving now passes, and live WSL
  ComfyUI now passes an opt-in install/start/status/logs/generate-cat/stop flow
  from the ext4 worktree.
- The hidden `rocmd mcp-call` helper now enforces the MCP read-only/mutating
  boundary itself: non-read-only tools require `--allow-mutation`, while
  user-facing TUI/chat flows continue to route mutating calls through approval
  cards. Focused tests tie this guard to the current MCP tool annotations.
- TheRock version strings with embedded dates now render with a readable build
  date in install/update output, non-TUI runtime lists, Doctor runtime state,
  the setup screen, and TUI ROCm install lists/details. The setup screen's
  `Show install log` row now opens a focused saved-log view on ready and failed
  setup screens, includes the backing file path or missing-file details, resets
  to latest output when opened, and is covered by arrow-key/Enter regression
  tests plus saved-log scrolling checks.
- TUI approval reviews are now a consistent overlapping modal surface across
  setup, ROCm installs, install/update, services, automations, ComfyUI,
  chat-tool, and serve flows. First-time setup approval uses plain English,
  keeps the selected folder and local pip-cache path visible, avoids raw
  command/wheel/torch/environment jargon, and uses the same Up/Down plus
  Enter/Y/N/Esc approval navigation as the rest of the TUI.

Remaining plan-derived work that is not a normal local code task:

- Production signing/publication: repository-owned public keys, matching
  release secrets, hosted signed metadata sidecars, hosted recipe index, and
  hosted source-policy metadata. Test-key paths are implemented; production
  trust roots must be supplied by the project owner.
- Privileged live driver acceptance: native Linux DKMS execution needs
  appropriate supported hosts and root/sudo control. Current code has the
  documented distro plan/preflight/execution/reconcile path but does not invent
  a privileged acceptance environment.
- Expanded-engine GPU acceptance gates: vLLM has WSL GPU acceptance on this
  host. SGLang live acceptance is blocked by upstream ROCm kernel support for
  this `gfx1201` host. ATOM live acceptance is also not claimed on this
  `gfx1201` Radeon host because upstream ATOM currently lists Instinct
  `gfx950`, `gfx942`, and `gfx90a` targets; the adapter, managed TheRock
  environment propagation, and opt-in live acceptance harness are implemented.
- Broader GPU-family CI: current coverage includes normal Linux/Windows CI,
  cross-platform local no-fallback smoke, a self-hosted MI300X smoke job that runs
  non-mutating detect/capabilities commands for all first-party engine
  adapters, and local RDNA4/gfx1201 Windows/WSL acceptance. Additional live
  serving coverage across GPU families requires CI hardware or lab machines.
- Automation event-source coverage beyond current local sources: the existing
  scheduler, managed-service, endpoint-health, healthcheck, GPU telemetry, and
  loopback webhook sources are implemented and tested. `driver-upgrade` is
  only a reviewed response to a local `update.available` driver signal today;
  rocm-cli does not detect a real AMD driver update feed yet. A production
  driver update feed or other non-webhook event producer needs a defined
  upstream source before rocm-cli can wire it without inventing behavior.

## Immediate Correctness Fixes

These are narrow plan-derived fixes that unblock truthful V1 status.

1. Enforce Windows V1 runtime constraints.
   - Plan source: Windows V1 runtime management is pip venv only.
   - Status: completed in `apps/rocm/src/therock.rs`.
   - Result: `rocm install sdk --format tarball` now returns a clear Windows V1 error before tarball resolution, with focused tests.

2. Make packaged Windows support complete.
   - Plan source: Windows support is part of the product matrix.
   - Status: completed for this branch.
   - Result: historical Windows package/install scripts verify the complete
     platform bundle lifecycle, including vendored Codex when that bundle shape
     is used. The current Rust/Cosmopolitan universal binary does not require a
     `rocm-codex.exe` sidecar.

3. Add `rocm logs --service <id>`.
   - Plan source: PyTorch engine spec expects service log access by service id.
   - Status: completed in `apps/rocm/src/main.rs`.
   - Result: no-arg `rocm logs` still prints the directory summary; `rocm logs --service <id>` validates the service id, reads the managed service manifest, and tails the service log.

4. Implement PyTorch protocol `healthcheck`, `stop`, and `logs`.
   - Plan source: all required engine methods are listed in the engine contract.
   - Status: completed in `engines/pytorch/src/main.rs` and
     `apps/rocmd/src/main.rs`.
   - Result: PyTorch now handles `healthcheck`, `stop`, and `logs` with state/log fallbacks and focused tests.
     `rocmd supervise` and `server-recover` prefer engine protocol health
     checks and keep port checks as compatibility probes for engines without
     protocol health state.

## Completed This Execution Round

- Windows PowerShell installer/package support from the prior round remains verified by the acceptance lifecycle.
- Historical Windows platform-bundle acceptance still covers vendored Codex
  packaging and uninstall matching; the universal binary path has no Codex
  sidecar requirement.
- Windows V1 rejects TheRock tarball runtime installs and points users to managed pip venv installs.
- `rocm logs --service <id>` tails managed service logs.
- PyTorch engine protocol now responds to `healthcheck`, `stop`, and `logs`.
- `rocmd supervise` now prefers engine protocol health checks for readiness and keeps port probing as a compatibility fallback.
- `server-recover` now uses protocol health status for running/ready managed services and can recover services whose endpoint is failed, unreachable, or exited.
- TUI now accepts `/engine`, `/model`, and `/plan` commands using the existing inventory/planning paths.
- `rocm doctor` now reports kernel, driver policy/status/detail, managed runtime count, managed service count, and model cache entry count.
- `rocm chat` now uses a provider status adapter surface for `local`, `openai`, and `anthropic`, including auth status, model list, and the shared ROCm tool schema id.
- `rocm chat --prompt` now sends a non-interactive prompt through the provider contract; the local provider posts to a ready managed OpenAI-compatible service.
- The local provider implements OpenAI-style SSE stream parsing and `stream: true` requests for managed local services.
- `rocm-core` now has a shared built-in model recipe registry skeleton with aliases, engine preferences, device policy, dtype, remote-code policy, and warnings.
- The PyTorch engine now resolves known recipes from the shared registry before falling back to local heuristics.
- The PyTorch engine now returns service-specific endpoint URLs, probes managed `torch.cuda` state during detect, and uses wheel-only pip installs for managed dependencies.
- `rocm serve --host` now requires `--allow-public-bind` for non-loopback binds, and the MCP launch tool forwards its existing `allow_public_bind` approval.
- Engine launch/request paths now check runtime plugin directories before packaged sibling binaries.
- `rocm engines list`, TUI `/engine`, and `docs/engine-plugins.md` now
  surface the external engine plugin directory policy: data-dir plugins are
  searched before packaged sibling binaries, installers/upgrades do not touch
  data-owned plugins, and `rocm uninstall --keep-data` is required to preserve
  them during uninstall.
- TUI command handling now includes `/logs <service-id>`, `?` help, `Ctrl+L` transcript clear, and `Ctrl+R` status refresh.
- TUI prompt submission now distinguishes structured command-shaped input from
  plain-English requests before planning, so non-slash `install sdk ...`,
  `update --apply ...`, `uninstall apply`, and `serve tiny.gguf --engine ...`
  open the same navigable screens as slash commands.
- `rocm services` is a first-class non-TUI command. Local server status
  surfaces show living services by default; failed/stopped saved manifests are
  available through explicit all/history views.
- Automation watcher actions now mirror into a general `audit/events.jsonl` log while preserving `automations/events.jsonl`.
- Watcher `propose` mode now writes pending proposal records to `automations/proposals.jsonl`, and `rocm automations` shows recent queued proposals.
- Windows acceptance now verifies checksum mismatch rejection before installer activation.
- Release package scripts now emit detached RSA/SHA-256 signatures when a signing private key is configured, and Linux/Windows installers can require matching signature verification before activation.
- Windows acceptance now verifies signature mismatch rejection before installer activation.
- Windows acceptance lifecycle passes after these changes.
- TheRock managed pip runtime installs now use one TheRock-index transaction
  with pinned `rocm[libraries,devel]`, `torch`, `torchvision`, and `torchaudio`
  versions selected by exact ROCm wheel suffix for the current
  Python/platform. The install is verified through `rocm_sdk` package roots,
  `rocm-sdk path --root`, and `rocm_sdk.find_libraries(...)` probes.
- OpenAI and Anthropic providers now issue real HTTPS chat-completion/message requests through the provider adapter contract when API keys and models are configured, using in-process HTTP so API keys are not passed through child-process arguments.
- OpenAI and Anthropic streaming now sends real streaming requests instead of wrapping blocking chat responses. OpenAI streams parse chat-completion `delta.content` SSE events; Anthropic streams parse `content_block_delta` text events and `message_stop`.
- OpenAI and Anthropic API keys can be saved and cleared from OS secure storage with `rocm config set-provider-key <provider>` and `rocm config clear-provider-key <provider>`; status/config surfaces show key state without printing the key.
- The packaged `llama.cpp` adapter binary (`rocm-engine-llama-cpp`) implements the first-party engine protocol around an external `llama-server`, while preserving `llama.cpp` as the user-facing engine id.
- The packaged `vllm` adapter binary (`rocm-engine-vllm`) implements the
  first-party protocol around an existing Linux/WSL vLLM command, keeps vLLM
  ROCm GPU-only inside rocm-cli, and reports native Windows as unsupported
  without CPU fallback.
- PyTorch and `llama.cpp` now have stdio protocol routing coverage for every
  engine method. Non-mutating methods use valid typed payloads; `install` and
  `launch` are covered through invalid-payload validation so tests prove the
  routes exist without installing runtimes or starting services.
- `rocmd sandbox-run` now exposes the plan-defined restricted internal tool API with Linux bubblewrap isolation when available and an explicit native restricted fallback for unsupported hosts.
- Automation proposals now carry stable ids, structured restricted-tool payloads, status transitions, and audit records.
- The TUI now focuses queued proposals, approves them with Enter or Y, rejects/cancels with Esc or N, supports structured proposal edits, and gates mutating install/engine/automation commands behind approval cards.
- Runtime activation and previous-runtime validation now use exact versioned runtime keys and an active runtime marker, so release/nightly or old/new installs can coexist.
- Existing rocm-cli runtime manifests can be imported read-only, and existing TheRock Python environments can be adopted read-only after probing `rocm_sdk`.
- TheRock pip and tarball metadata resolution now uses a rocm-cli metadata cache with ETag support.
- `rocm doctor` now uses bounded optional probes, Windows display inventory before ROCm tools, and reports dedicated gfx target/compatible TheRock family without requiring an existing ROCm install.
- The TUI `/runtimes` flow now covers list/activate/uninstall/import/adopt with option-aware adopt validation and flag completions. Normal users switch ROCm installs by selecting an installed item from the list; legacy typed rollback commands are guided back to that picker instead of being advertised as a primary action.
- WSL support now includes ROCDXG preflight/setup documentation and strict llama.cpp GPU E2E validation against managed TheRock libraries.
- Deferred/unsupported expanded engines now fail with explicit no-fallback messages when no external plugin is installed.
- CLI lifecycle audit records now cover update checks, SDK install/dry-run, runtime activation/import/adopt/rollback, engine installs, and managed service launches.
- `/model` in the TUI now lists the built-in recipe registry and completes model aliases instead of reporting the model registry as unimplemented.
- `rocm model` and alias `rocm models` now render the same model recipe
  registry in non-TUI mode instead of falling through to natural-language
  planning.
- Normal CLI startup now performs a quiet bounded TheRock update check when a managed runtime exists, using cached metadata, a freshness gate, and a hard metadata fetch timeout. First-run hosts with no managed runtimes are left untouched, and the localized pip wheel cache is not created.
- `rocm doctor` now appends active runtime state and non-invasive engine adapter/plugin inventory without invoking engine protocol detection, preserving bounded doctor latency.
- CLI lifecycle actions now write both JSONL audit records and plain-text logs: a global `logs/cli-lifecycle.log` and per-action logs under `logs/cli`. `/logs` shows these paths for TUI/CLI inspection.
- `/logs` now lists per-action CLI log files and tails recent lifecycle lines
  inline without creating first-run log directories.
- `server-recover` watcher policy now treats manifest `failed`, `exited`, and `unreachable` states as recoverable and also recovers managed services stuck in `starting` or `recovering` beyond a conservative stale grace window.
- `rocm update --apply` now applies a newer runtime through the same TheRock install path. It selects the active runtime by default, requires `--runtime <runtime-key>` when selection would be ambiguous, supports `--dry-run`, installs updates side-by-side, and only activates the new runtime when `--activate` is explicitly supplied.
- The TUI now mirrors the update apply flow: `/update` remains read-only, `/update --apply ...` validates arguments before approval, `/update --apply --preview` is treated as a non-mutating inline command, and completions include update flags plus ready runtime keys.
- TheRock metadata cache fetches can now verify detached sidecar signatures when `ROCM_CLI_METADATA_PUBLIC_KEY_PATH` or `ROCM_CLI_METADATA_PUBLIC_KEY_PEM` is configured, and fail loudly when `ROCM_CLI_REQUIRE_METADATA_SIGNATURE=1` is set without a usable key/signature.
- Engine-specific recipe metadata now has a versioned adapter contract: `rocm`
  maps the selected engine's recipe record into an engine protocol hint during
  model resolution and launch. PyTorch, llama.cpp, vLLM, SGLang, and ATOM
  validate the target engine plus contract version before echoing or applying
  the hint. Managed service manifests preserve the launch recipe so recovery
  restarts carry the same adapter recipe settings.
- PyTorch, llama.cpp, and vLLM GPU acceptance scripts now resolve the active
  exact runtime key instead of guessing from the newest runtime manifest; broad
  runtime ids must be unambiguous, PyTorch env manifests must match the chosen
  runtime, and all three scripts have offline `--self-test` selector coverage.

## Milestone 1: Stabilize The Existing Skeleton

Goal: make the currently implemented CLI/TUI/runtime/engine skeleton honest, testable, and aligned with V1 constraints.

- Expand `rocm doctor` with kernel, distro, CPU, runtime state, driver state, engine inventory, model cache, and legacy ROCm detection.
- Add basic logging paths and log writes for install/update/serve/automation lifecycle.
  - CLI audit events now cover update/install/runtime/engine/managed-service
    actions. TUI screen-owned commands now also persist full-output screen
    command logs under `logs/tui`, and `/logs` includes those files in the
    visible log browser/search surface alongside lifecycle and CLI action logs.
- Add bounded cached update checks on every `rocm` invocation.
  - Completed with a quiet, freshness-gated startup check that records status under the TheRock metadata cache and skips first-run/no-runtime hosts.
- Complete deeper engine inventory and active runtime reporting in `rocm doctor`.
  - Completed with `runtime_state` and `engine_inventory` sections that report active runtime keys/status and adapter/plugin availability without slow engine detect calls.
- Complete per-command lifecycle log writes for install/update/serve/automation operations.
  - Completed for CLI lifecycle audit callers with global and action-specific
    text logs surfaced by `/logs`.
- Expand watcher health policy beyond the initial failed/unreachable/exited service recovery states.
  - Completed for manifest terminal states and stale `starting`/`recovering` managed services. General event-bus/policy-engine work remains in the automations milestone.
- Add tests for Windows tarball rejection, service log lookup, protocol lifecycle responses, and no-network dry-run paths.

## Milestone 2: Runtime Management Completion

Goal: satisfy Phase 1 runtime exit criteria.

- Replace ad hoc release/nightly lookups with signed and cached metadata resolution.
  - ETag-backed caching exists, and optional detached metadata signature
    verification is implemented. Remaining work is production metadata key
    publication and signed sidecar metadata hosting.
- Add runtime update application flow.
  - Completed with `rocm update --apply [--runtime <runtime-key>] [--dry-run] [--activate]`.
  - `rocm update` now also reports non-runtime update surfaces explicitly:
    CLI feed not configured, first-party engines package-managed, external
    plugins user-managed, and model recipes built-in/signed-index/error.
    `--apply` still mutates runtime updates only.
- Add versioned side-by-side runtime installs.
  - Completed for TheRock runtime keys.
- Add active runtime activation via atomic pointer/symlink or platform-equivalent marker.
  - Completed with a platform-neutral active runtime marker and exact config key.
- Add previous-runtime rollback validation.
  - Completed with previous-key tracking and activation validation. The normal
    TUI does not present rollback as a primary concept; users switch versions
    by selecting an installed ROCm environment from the list.
- Add existing TheRock detection with adopt/import/read-only modes.
  - Completed for explicit import/adopt commands.
- Report legacy ROCm installs as unmanaged with migration guidance.
  - Completed in `rocm doctor`.

## Milestone 3: TUI, Chat, and Planning

Goal: satisfy Phase 2 and Phase 4 exit criteria.

- Add the planned TUI panes: transcript, sidebar, log view, composer, and mode state.
  - Sidebar mode state is implemented and updates for slash commands plus
    recognized natural-language plans.
  - Recent TUI/command/proposal activity is retained internally and seeds from
    the bounded tail of the persisted CLI lifecycle log without creating
    first-run log directories. The normal TUI does not show a persistent
    Activity pane; explicit history lives in `/logs`.
  - `/logs` provides a compact log navigation view with per-action CLI log file
    names, a recent lifecycle tail, and bounded query search over recent
    lifecycle/action-log lines.
  - Stateful TUI pagination is implemented with `/logs next`, `/logs prev`,
    and `/logs refresh` over the current browser query.
  - Live refreshed TUI log following is implemented with `/logs follow [query]`
    and `/logs stop`; manual next/previous paging exits follow mode.
- Represent plans, tool calls, approvals, logs, errors, and outcomes as typed transcript entries.
  - Completed for visible transcript rows with typed metadata for user input,
    block headers/bodies, stream lines, assistant lines, and blanks.
- Stream tool execution output into the TUI instead of collecting child output after exit.
  - Completed for TUI-launched `rocm` and `rocmd` jobs with burst caps,
    per-stream final-output collapse, CR progress handling, and scroll
    preservation.
- Add approval/edit/cancel flows for mutating actions.
- Wire local `stream_chat` events into the TUI transcript instead of collecting them behind the provider API only.
  - Completed for `/chat [prompt]`: local provider SSE deltas stream through
    callback events into assistant transcript lines, with tests for live
    delivery before connection close and chunked SSE decoding.
- Implement actual HTTPS chat calls for `openai` and `anthropic`.
  - Remote prompt sending now requires explicit rocm-cli opt-in with
    `rocm config enable-provider openai|anthropic`; ambient API keys alone do
    not send prompts. Provider keys can now be stored in the OS secure store,
    while environment variables remain session-only overrides.
- Replace keyword freeform planning with the planned hybrid parser/tool-call planner.
  - Completed for deterministic grammar/alias parsing of common
    serve/install/update/driver/uninstall/inspect requests, structured
    tool-call rendering, approval labeling, built-in model recipe alias
    normalization, and TUI plain-input plan routing.
  - Optional LLM ambiguity resolution is implemented behind
    `rocm config set-planner-provider local|openai|anthropic`. Provider output
    is reduced to a validated `rocm` tool call, public bind requests are
    rejected, and provider-assisted plans require interactive review instead of
    non-interactive `--yes` execution.
  - Recognized mutating natural-language requests in the TUI now render the
    structured plan and focus the normal approval card for the next executable
    action. Serve plans use `--managed` so approval launches a supervised
    endpoint. Non-interactive `rocm "..."` remains plan-only unless `--yes`
    grants explicit approval.
  - Non-interactive `rocm --yes <natural language request>` now renders the
    same structured plan and executes only the final structured `rocm` tool
    call after rejecting flag-shaped requests, unquoted structured command
    names, and placeholder values.

## Milestone 4: Engines and Model Registry

Goal: satisfy Phase 3 and Phase 5 before expanded engines.

- Add plugin-directory discovery while preserving sibling-binary discovery for packaged installs.
  - Completed with shared discovery order and user-facing installer/uninstall
    policy for external plugin directories.
- Add `llama.cpp` and `vllm` adapters with install, detect, resolve, launch, endpoint, healthcheck, stop, and logs.
  - `llama.cpp` is implemented for the full first-party protocol around an
    external `llama-server`.
  - `vllm` is implemented as a packaged external-runtime adapter for
    Linux/WSL ROCm GPU serving. It discovers `ROCM_CLI_VLLM_COMMAND`,
    `ROCM_CLI_VLLM_PYTHON`, active managed TheRock runtimes with a sibling
    `vllm` command, or `vllm` on `PATH`.
  - PyTorch and `llama.cpp` now have side-effect-free stdio routing tests for
    all protocol methods.
  - Live WSL vLLM ROCm GPU acceptance is complete against a rocm-cli managed
    TheRock runtime. Remaining expanded-engine live coverage is SGLang on
    supported SGLang ROCm hardware; the current `gfx1201` host is blocked by
    upstream SGLang kernel support.
  - PyTorch, llama.cpp, and vLLM GPU acceptance scripts now fail loudly when
    active or default runtime selection is ambiguous, instead of selecting the
    newest matching manifest.
- PyTorch serving now rejects `cpu_only` and treats `gpu_preferred` as
  `gpu_required`; there is no PyTorch CPU serving path in rocm-cli.
- Replace the built-in recipe registry skeleton with a signed model/engine recipe index.
  - Completed for schema validation, local signed-index loading, detached
    signature verification, no-fallback error reporting when a configured index
    is invalid, and PyTorch resolver consumption. Remaining work is hosted
    production index/key publication and update/cache policy.
- Add model aliases, engine preferences, quantization support, VRAM/RAM fit checks, unsupported-combination explanations, and manual alternative recommendations.
  - Built-in recipe aliases and engine preferences are implemented.
  - TUI `/model` now reports cached aggregate GPU VRAM fit as supported,
    unsupported, or unknown with a concrete reason and next action.
  - TUI `/model` now reports preferred-engine support per recipe and avoids
    implying automatic fallback behavior.
  - Built-in recipes now carry recommended system RAM, quantization, artifact
    expectations, and manual alternative recipe hints. `/model` renders these
    advisory checks and recommends alternatives only as explicit manual
    choices.
  - `rocm doctor` now reports host system RAM, and TUI `/model` compares that
    telemetry against each recipe's recommended system RAM.
  - Signed recipe index artifact descriptors can now render
    `metadata_available` or `blocked`.
  - Signed recipe indexes can carry `engine_recipes` with required flags,
    parser settings, preferred endpoint settings, unsupported combinations,
    and notes. `/model` renders these, and `rocm serve` now passes the selected
    engine recipe as a versioned protocol hint during adapter `resolve_model`
    and launch. First-party adapters validate the target engine and contract
    version before echoing or applying the hint; vLLM, SGLang, ATOM, and
    llama.cpp append selected-engine `required_flags` to the GPU server launch,
    while PyTorch maps only supported recipe flags to worker settings and
    rejects unsupported flags loudly.
  - Local artifact cache marker telemetry is implemented under
    `data/models/artifacts` with slug plus full-hex model/artifact key
    components to avoid collisions between similar signed-index refs, and
    restricted `prefetch_artifact` reports `source_policy_required` with
    `network_used: false` until source policy is explicitly approved.
  - Approved live artifact prefetch is implemented for direct HTTP(S)
    artifacts that declare `size_bytes` and `sha256`: the restricted tool
    requires explicit `--allow-artifact-download`, enforces an approved byte
    limit, verifies sha256, writes bytes atomically, records a cache marker,
    and keeps gated, hashless, or non-direct sources blocked.
  - Authenticated Hugging Face artifact prefetch is implemented for gated
    recipe-registry artifacts that declare HTTPS, `size_bytes`, and `sha256`.
    It requires `--allow-huggingface-download` plus a Hugging Face token from
    `ROCM_CLI_HUGGINGFACE_TOKEN`, `HF_TOKEN`, or `HUGGING_FACE_HUB_TOKEN`, and
    refuses to send that token to non-Hugging Face URLs or over plain HTTP.
    Signed recipe artifacts can now declare `source_policy` metadata with
    `direct_https_sha256`, `huggingface_public`, `huggingface_authenticated`, or
    `manual_only`. The signed-index loader validates host constraints, HTTPS
    and integrity metadata, `/model` renders the declared policy, and the
    restricted prefetch tool enforces manual-only and authenticated Hugging
    Face policy even after generic download approval.
    Remaining work is hosted source-policy publication for production recipe
    indexes and any additional source-specific authenticated flows required by
    those indexes.
- Make `rocm serve <model>` use the shared resolver before selecting an engine.
  - Completed for default engine selection: explicit `--engine` and configured
    defaults still win, otherwise a known shared recipe can select its first
    preferred engine and renders the selection source. No alternate engine is
    selected automatically.

## Milestone 5: Driver Management

Goal: satisfy Phase 6 without expanding beyond the plan.

- Linux: implement official AMD DKMS flow wrapper with distro/kernel/sudo preflight, visible plan, explicit approval, execution, reboot-required tracking, and post-install health checks.
  - `rocm install driver` now renders a visible plan by default.
  - AMD-documented Ubuntu 22.04/24.04, Debian 12/13, RHEL, Oracle Linux,
    Rocky, and SLES plans use package-manager DKMS commands and post-install
    checks.
  - Actual execution requires `--yes`; TUI approval appends `--yes` while
    preserving the visible command.
  - Execution state records pre/post driver summaries, boot id, command list,
    and reboot flags under `data/driver/state.json`.
  - `rocm install driver --reconcile` refreshes passive post-reboot checks,
    records reconciliation details back into `data/driver/state.json`, and
    reports whether the boot id changed after execution.
  - Passive post-check reconciliation now persists and renders a summary count
    for total/present/missing checks. This remains non-privileged and does not
    run DKMS commands during reconciliation.
  - AMD-documented package-manager plans now cover Ubuntu 22.04/24.04,
    Debian 12/13, RHEL 10.1/10.0/9.7/9.6/9.4/8.10, Oracle Linux
    10.1/9.7/8.10, Rocky 9.7, and SLES 15.7. Unsupported Linux IDs remain
    non-mutating instead of guessing commands.
  - Remaining work: privileged live acceptance coverage on supported hosts.
- Windows: add driver presence and compatibility reporting to `rocm doctor`; do not add Windows driver install or upgrade.

## Milestone 6: Automations, Sandbox, and Hardening

Goal: satisfy Phase 8 and Phase 9.

- Replace hard-coded watcher polling with an event bus for scheduler, service lifecycle, update checks, health checks, GPU metrics, endpoint health, and local webhooks.
  - Completed for the initial internal trigger-event substrate over the
    existing watchers: scheduler ticks for `therock-update`, recoverable
    managed-service manifest, engine healthcheck, and endpoint-health events
    for `server-recover`, bounded read-only local `amd-smi` availability
    events for `gpu-metrics`, and loopback-only local JSON webhook ingestion
    through `rocmd run --automations-enabled --local-webhook-port <port>`.
    Local webhooks validate the watcher id and exact event kind before
    dispatch: `therock-update` accepts `schedule.tick`, `server-recover`
    accepts the recoverable `service.*` event kinds, `gpu-metrics` accepts
    `gpu.metrics` and `gpu.metrics_unavailable`, `gpu-thermal-protect` accepts
    `gpu.thermal_pressure` and `gpu.memory_pressure`, `cache-warm` accepts
    `cache.warm`, and `driver-upgrade` accepts `update.available` with
    `payload.component=driver`. `cache-warm` proposals can carry reviewed
    source-policy fields for approved downloads, while webhook payloads still
    cannot inject those approvals. This is not a real AMD driver update check;
    it is the reviewed response path for a local driver-update signal.
    Remaining work is event producers not covered by the existing local
    sources, such as a production driver-update feed if/when one is defined.
- Add a real policy engine for `observe`, `propose`, and `contained`.
  - Existing watchers now route through an explicit mode-to-policy decision
    before applying behavior.
  - `therock-update` propose mode queues a reviewable read-only update-check
    proposal, while contained mode now consumes the restricted `check_updates`
    tool result, records the outcome without applying updates, and records a
    `notify_if_newer` notification/audit event only when the read-only report
    says a runtime update is available.
  - `server-recover` contained mode keeps the restricted restart path.
  - `driver-upgrade` propose mode queues a reviewed read-only driver plan
    proposal. Contained mode consumes the restricted `driver_plan` tool result
    directly, records the dry-run outcome, and never installs drivers.
  - Remaining work: additional non-webhook event producers and broader
    contained execution for future mutating actions with explicit plan backing.
- Add queued proposals and TUI review/history.
- Implement the Linux sandbox runner with rootless isolation and restricted internal tool API.
- Promote automation events into a general audit log covering installs, updates, watcher actions, provider usage, and service lifecycle.
  - Provider chat usage is now audited for both non-streaming and streaming
    provider paths. Streaming TUI chat records `provider` category events with
    action `stream_chat` after the stream completes.
- Add signing for CLI bundles and metadata.
- Publish a stable project signing public key and make release-channel signature verification mandatory by default after key distribution is finalized.
  - Bundle signing is now enforced for stable release packaging when
    `ROCM_CLI_REQUIRE_SIGNATURE=1` is set by CI.
  - Linux and Windows package scripts fail if signing is required and no
    private key is configured.
  - Linux and Windows acceptance cover bad-signature rejection before
    activation with generated test keys, and now also cover missing required
    signature sidecars before activation.
  - Linux and Windows acceptance cover required-signature mode without a public
    key, generated private-key PEM package signing, and generated public-key
    PEM installer verification before activation.
  - Rust-side signed metadata cache verification is implemented behind
    explicit metadata key/requirement settings, with generated-key coverage
    for valid signatures and tampered payload rejection.
  - Signed model recipe index loading has generated-key coverage for valid
    signatures and tampered payload rejection.
  - Remaining work: owner-published production public key, production metadata
    public key, matching release secret, signed metadata sidecars, and
    installer/default trust roots.
  - Telemetry policy/config is implemented for `local` and `off`; no external telemetry/reporting is implemented.
- Extend CI acceptance across serving, update, watcher flows, supported Linux/Windows GPU families, and local no-fallback smoke hosts.
  - Cross-platform local no-fallback smoke now runs in CI against isolated
    first-run config/data/cache roots. It verifies GPU-required planning,
    telemetry-off behavior, explicit CPU-policy rejection for adapters and
    serve commands, loud GPU-required failures, no first-run pip cache, and no
    runtime registry creation.
  - Remaining work: broader GPU-family CI coverage and any production release
    trust checks that depend on owner-published keys/secrets.

## Parallel Work Split

This section is historical. These worker lanes were useful while the remaining
plan was being implemented, but the first deliverables listed below are now
either implemented locally or gated by the owner/infrastructure/upstream items
called out in the phase table and `Current Remaining Gates`.

| Worker Lane | Owns | Current State |
|---|---|---|
| Runtime/Doctor | `apps/rocm/src/therock.rs`, doctor data in `crates/rocm-core` | Implemented locally; production metadata signing remains an owner publication gate |
| Engine Lifecycle | `crates/rocm-engine-protocol`, `engines/pytorch`, service log/status paths | Implemented locally for PyTorch/llama.cpp plus gated Linux/WSL adapters |
| TUI/Planner | `apps/rocm/src/tui.rs`, planner code in `apps/rocm/src/main.rs` or a new planner module | Implemented locally with navigable screens, approvals, progress cards, and structured routing |
| Providers | Provider modules plus config/key storage | Implemented locally for local/OpenAI/Anthropic providers and served-model ROCm tools |
| Automations/Security | `crates/rocm-core`, `apps/rocmd`, TUI automation view | Implemented locally for reviewed proposals, sandbox runner, audit logs, and existing watcher events |
| Release/Trust | installers, package scripts, workflows | Implemented locally for generated-key and signature/checksum acceptance; repository-owned production keys remain an owner gate |

## Non-Plan-Derived Items

No new product features are proposed here. The only non-plan-derived choice is this file's organization as a separate remaining-work tracker.
