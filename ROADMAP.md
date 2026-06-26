# rocm-cli — post-merge ROADMAP

Tracks work remaining **after** the rocm-dash merge that folded the standalone
`rocm-dash` telemetry dashboard into the `rocm` binary (`rocm dash`).

The merge shipped: the unified dashboard TUI (Overview / Hardware / Instances /
Bench / Chat), the no-key ChatGPT-OAuth chat backend, the unified `config.json`
(D6) with legacy `config.toml` migration, `pip`→`wheel` alignment, a
Windows-clean build, and `rocm dash --demo/--replay/--chat-mock`.

This file captures (1) the remaining work to reach full **feature parity** with
the two surfaces the merge replaces, (2) the **background-helper (`rocm daemon`)
wiring** regression, and (3) the scoped **Per-process VRAM → model** dashboard
item.

> **What shipped (Supergoal §1 + §2).** Items (1) and (2) are now done. The dash
> Chat tab reached agentic parity (read-only ROCm tools, approval-gated mutating
> tools, `/plan`, `/provider` + Anthropic, a 30-command parity map) and bare
> `rocm`/`rocm chat` reroute to it; `tui.rs` is retained behind an explicit
> retire gate (deletion not performed). `rocm daemon` runs the real loop
> in-process via the `rocmd` lib with on-demand, double-spawn-guarded autostart.
> Item (3) (per-model VRAM) remains the open dashboard follow-up.

---

## 1. Feature parity — retire `tui.rs` without regression

**Status: parity REACHED; bare `rocm`/`rocm chat` REROUTED to the dash;
deletion GATED (not performed).** The dash **Chat** tab now matches the legacy
assistant's agentic capabilities, and bare `rocm` + interactive `rocm chat`
route to the unified dash chat (`dash::run_chat`). `apps/rocm/src/tui.rs` is
**retained** (anchored by `_RETAINED_TUI_ENTRY`) behind an explicit accept/retire
gate — the actual file deletion is intentionally not done in this supergoal.

**What shipped (Supergoal §1, Phases 3–9):**
- Read-only ROCm tools through a `rocm-core`-free execution seam
  (`tool_exec.rs` / `dash_seam.rs`): doctor, engines, services, logs, snapshots,
  automations, path/port checks, update-check, `rocm_command`.
- Mutating tools gated by the existing `ui/approval.rs` modal (`install_sdk`,
  `install_engine`, `launch_server`, `stop_server`, `watcher_enable/disable`)
  plus update/comfyui/uninstall/setup and automations review/approve/reject/edit.
- Natural-language **`/plan`** (Ask → Plan → Review → Run).
- **`/provider`** live backend switching + the **Anthropic** backend; the full
  slash-command set.
- A **30-command parity map** + accept/retire checklists:
  [`docs/dash-parity-map.md`](docs/dash-parity-map.md),
  [`docs/dash-parity-checklist.md`](docs/dash-parity-checklist.md),
  [`docs/tui-retirement-checklist.md`](docs/tui-retirement-checklist.md).

The original gap analysis and build order below are kept for historical context;
all stages are now done except the **gated** Stage-8 deletion.

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

## 2. Background-helper (`rocm daemon`) wiring — regression

**Status: DONE.** `rocm daemon` now runs the real foreground loop **in-process**
via the `rocmd` library (`rocm` → `rocmd` lib; acyclic), matching the "built into
rocm" policy. The helper is **autostarted on demand**, detached and
double-spawn-guarded (`ensure_background_helper_running` /
`background_helper_already_running`), from both `automations enable` and
`rocm serve --managed`. `render_daemon_text` is retained as the `--status` view
only. The original regression analysis below is kept for historical context.

**Status (historical): regression. The background helper is dormant.**
`rocm daemon` is documented as *"Start the background helper in the foreground"*
(`apps/rocm/src/main.rs:250`), but its handler only renders a status panel:

```rust
// apps/rocm/src/main.rs:1298
Some(Command::Daemon) => {
    print!("{}", render_daemon_text(&paths, &config));   // status only — no loop
    Ok(())
}
```

The real foreground loop, `run_daemon()`, exists **only in the separate `rocmd`
binary** (`apps/rocmd/src/lib.rs:2808`, reached via `rocmd run`). But:

- `apps/rocm` does **not** depend on `rocmd` (its `Cargo.toml` pulls
  `rocm-core`, `rocm-dash-daemon`, `rocm-dash-tui` — not `rocmd`).
- **Nothing in `apps/rocm` ever spawns `rocmd run` or `rocm daemon` as a running
  loop.** The only `"daemon"` references are status renderers and tests.
- `daemon_binary_path()` (`crates/rocm-core/src/lib.rs:5859`) resolves to the
  **`rocm`** binary itself, and `main.rs:10666` states *"policy: built into
  rocm; no separate rocmd binary is required"* — i.e. the intended design is
  that `rocm daemon` **is** the helper loop, re-executing itself. That loop
  logic is missing; it is stubbed to the status panel.

**Effect:** automation checks and on-demand local model servers have no helper
to run, regardless of the §1 chat-parity work. `run_daemon()` in `rocmd` is
orphaned code the product never invokes.

**Build order:**
1. **Decide the model** — either (a) fold `rocmd`'s `run_daemon()` into the
   `rocm` crate so `rocm daemon` runs the loop in-process (matches the
   "built into rocm" policy), or (b) have `rocm daemon` spawn `rocmd run`, with
   automations/managed-serve starting it on demand. (a) is the stated direction.
2. **Wire the chosen entry** so `rocm daemon` actually starts the loop in the
   foreground; keep `render_daemon_text` as the *status* view only.
3. **On-demand start** — have automation-enable and `rocm serve --managed`
   ensure the helper is running (spawn detached if not), then update
   `AutomationRuntimeState` so the status panel reflects reality.
4. **Reconcile `rocmd`** — once the loop lives in `rocm`, either remove the
   orphaned `rocmd` binary or make it a thin alias; don't ship two daemons.
5. **Verify** — `rocm daemon` runs and accepts work; automation checks fire;
   status panel shows `running`; clean shutdown.

**Invariants:** keep the user-owned unix socket (mode 0600) hardening from #17;
no second listener/socket competing with the `rocm dash` telemetry daemon
(`rocm-dash-daemon`) — these are distinct daemons and must stay distinct.

---

## 3. Per-process VRAM → model attribution (rocm dash)

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

> The broader telemetry roadmap (disk/net I/O, throttle, UMC util, fan/PWM,
> `--json` one-shot, xGMI link-state) will be tracked in future issues.
