# rocm-cli — post-merge ROADMAP

Tracks work remaining **after** the EAI-6871 merge that folded the standalone
`rocm-dash` telemetry dashboard into the `rocm` binary (`rocm dash`).

The merge shipped: the unified dashboard TUI (Overview / Hardware / Instances /
Bench / Chat), the no-key ChatGPT-OAuth chat backend, the unified `config.json`
(D6) with legacy `config.toml` migration, `pip`→`wheel` alignment, a
Windows-clean build, and `rocm dash --demo/--replay/--chat-mock`.

This file captures (1) the remaining work to reach full **feature parity** with
the two surfaces the merge replaces, and (2) the scoped **Per-process VRAM →
model** dashboard item.

---

## 1. Feature parity — retire `tui.rs` without regression

**Status: the only real parity gap.** Bare `rocm` and `rocm chat` still launch
the legacy chat-first assistant (`apps/rocm/src/tui.rs`); its retirement was
**deferred** because the new dashboard **Chat** tab is a read-only telemetry
chat and lacks the legacy assistant's agentic capabilities. Closing this gap is
the prerequisite for deleting `tui.rs` and routing bare `rocm`/`rocm chat` to
the unified TUI.

**What the dash Chat tab is missing vs `tui.rs`:**
- Mutating ROCm tool-calls with **in-chat approval** (`install_sdk`,
  `launch_server`, `stop_server`, `install_engine`, `watcher_enable/disable`).
- Natural-language **`/plan`** (Ask → Plan → Review → Run; `FreeformPlanAction`).
- Live machine inspection via subprocess (`rocm_command` read-only tools:
  engines / services / model / config / paths) — the dash chat only has GPU
  telemetry tools today.
- The 22 slash commands and the Anthropic provider.

Most of this logic already lives in the **bin** (`apps/rocm/main.rs`:
`run_chat_read_only_tool`, `validate_chat_install_sdk_tool_call`,
`natural_language_plan`, `ChatToolApprovalRequest`), so this is largely a
re-wiring job, not a rewrite.

**Build order (each stage independently testable):**
1. **Parity checklist** — turn the capability gap above into an explicit
   accept/retire gate.
2. **Execution seam** — define a `rocm-core`-free tool-call boundary: the dash
   chat emits tool-call *intents* (plain data); the bin (which owns `rocm-core`
   + the existing tool engine) executes them and feeds results back. Mirrors the
   existing `dash.rs` adapter pattern (`model_recipe_summaries`,
   `runtime_summaries`). **Load-bearing — do this first.**
3. **Read-only tools** through the seam (no approval needed) — fastest path to
   "the dash chat can inspect the machine like `tui.rs`."
4. **Mutating tools + in-chat approval** via the existing `ui/approval.rs`
   modal; reuse the bin's validators.
5. **Natural-language `/plan`** (depends on 2 + 4).
6. **Slash commands** — mostly glue over the tools/overlays.
7. **Provider decision** — keep the OAuth/APIM path, decide whether
   Anthropic/local-provider parity is required (product call; don't silently
   drop).
8. **Verify against the Stage-1 checklist → retire `tui.rs`** (delete
   `apps/rocm/src/tui.rs` + `mod tui;`, route bare `rocm`/`rocm chat` to the
   dash chat tab, prune now-dead helpers).

**Invariants to preserve throughout:** `rocm-dash-core` free of
tokio/ratatui/`rocm-core` at the boundary; reducer pure; chat `api_key`
env-only; `agent.rs` the sole `rig`-namer; `bollard` collectors-only;
ureq/reqwest partition intact.

### Related follow-ups (not strict parity)
- **Windows live dashboard.** `rocm dash` is unix-socket-only today (the daemon
  `tcp` listener is an unimplemented scaffold — `server.rs`: *"tcp listener not
  implemented"*); on Windows it builds but exits gracefully, and only
  `--demo`/`--replay` work. Implementing the TCP transport (+ platform-default
  `connect`/`listen`) would make the live dashboard work on Windows. (The
  standalone rocm-dash was Linux-only, so this is new scope, not a regression.)
- **Telemetry data migration.** D6 migrates the legacy `config.toml` →
  `config.json`, but **not** the standalone telemetry/data dirs
  (`~/.config/rocm-dash`, `~/.local/share/rocm-dash`) into the `rocm` app paths
  (`~/.rocm`, `~/rocm_venvs/default`). Migrate history or document the move.

---

## 2. Per-process VRAM → model attribution (rocm dash)

**Goal:** attribute each running model/serving instance's GPU VRAM by joining
`amd-smi` per-process VRAM to the owning model/container — so the dashboard
shows VRAM *per model*, not just per device. (Originally a `roadmap`/NEXT item
in the standalone rocm-dash.)

**Status: core implemented during the merge — wired end-to-end:**
- **Collector:** `crates/rocm-dash-collectors/src/amd_smi.rs` — `processes()` /
  `parse_processes()` parse `amd-smi process --json` into `GpuProcess`
  (`vram_used_mb`); a sysfs path exists in `sysfs.rs`.
- **Reducer:** `crates/rocm-dash-core/src/vram.rs` — `aggregate_process_vram()`
  (per-process → per-container via an injected PID→container resolver) with a
  **device-summed fallback** and graceful `(0,0)` for unmatched instances.
- **Daemon:** `crates/rocm-dash-daemon/src/runner.rs` (~L452) calls
  `processes()` then `aggregate_process_vram(...)` into `per_container_used`.
- **TUI:** `crates/rocm-dash-tui/src/ui/tabs/instances.rs` renders per-instance
  `vram_used/total`.

**Remaining work:**
- **Non-container / managed-service attribution.** The PID→owner resolver is
  cgroup/Docker-based, so containerized instances get per-process VRAM while
  non-container managed services (e.g. Lemonade, `rocm serve --managed`
  processes outside Docker) fall back to device-summed. Extend the resolver to
  match PIDs directly against the managed-services registry so those instances
  also get true per-model VRAM.
- **Kernel floor.** Per-process VRAM via `/proc/<pid>/fdinfo` needs amdgpu
  kernel ≥ 5.14; some MI300/MI355 cluster kernels may predate it. Detect and
  degrade clearly (don't show confidently-wrong numbers).
- **Real-hardware verification.** Validate the join on a multi-GPU box with live
  serving instances (the current tests are fixture-driven); confirm the numbers
  match `amd-smi`/`amdgpu_top`.
- **Per-GPU breakdown (optional).** Surface which GPU(s) a model's VRAM sits on
  for multi-GPU instances.

> Background and the broader standalone telemetry roadmap (disk/net I/O,
> throttle, UMC util, fan/PWM, `--json` one-shot, xGMI link-state) live in the
> `rocm-dash` repo wiki (`wiki/plans/`, `wiki/comparisons/competitive-tui-spike.md`).
