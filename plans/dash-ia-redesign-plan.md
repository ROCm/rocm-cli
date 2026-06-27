# Dash IA redesign: ROCm / Serving tabs, inline Details, AI-oriented Observe

## Goal
Replace the single **Action** tab with two purpose-built tabs — **ROCm** (install/configure/manage the ROCm platform) and **Serving** (install/configure/manage serving engines + instances). Each is an **Actions list (left) + `Details: <active thing>` pane (right)** where the *current manager UI renders inline* instead of as a centered modal. Modals are retained only to **gate** input (confirmations, change-causing/destructive steps, the approval flow) and for deeper "drill into an object" layers. Reorient **Observe** around AI-serving metrics (throughput, tok/watt, TTFT, TPOT, power, queue) with node-efficiency + throughput hero panels, sourcing TTFT/TPOT from new **live** telemetry.

## Target IA
`Home(1) · ROCm(2) · Serving(3) · Observe(4) · Chat(5)` — 5 tabs. Home and Chat unchanged.

- **ROCm Actions**: Set up / Install ROCm · Check for updates · Diagnose (doctor/examine) · Runtimes · Command runner · (Uninstall, display-only)
- **Serving Actions**: Serve a model · Engines · Running instances/services · Providers & keys · Logs · (Optimize, soon)

Each Action's **Details** pane shows that manager's existing first screen inline (reuse current content); a deeper object or change-causing step opens a modal/approval.

## Navigation (global, applies to ROCm/Serving)
- **Tab / ⇧Tab** — switch tabs (always).
- **←/→** — move focus between the Actions list and the Details pane.
- **↑/↓ / j/k** — context-sensitive: traverse the focused pane (Actions list, or the list/fields inside Details).
- **Enter** — activate focused item (Actions → load it into Details/focus it; Details → advance / open gating modal).
- Retire the per-tab letter hotkeys (w/e/d/u/i/l/s/…); the Actions list replaces them. (`?` help, `t` theme, `:` palette, `Esc` menu, `q` stay.)

---

## Phase 1 — Tab IA (Home/ROCm/Serving/Observe/Chat)
**Files:** `app/mod.rs`, `ui/tabs/mod.rs`, `ui/mod.rs`, `ui/modal.rs`.
1. `ActiveTab`: replace `Action` with `Rocm`, `Serving` → variants `Home, Rocm, Serving, Observe, Chat`. Update `next`/`prev`/`from_digit` ('1'..='5'), `Default = Home`.
2. `selection_len` / `selected_index` / `set_selected` arms for `Rocm` & `Serving`.
3. `TAB_LABELS` → 5 entries; `tab_labels()`/`compute_chip_layout` → `[TabChip; 5]` (or `Vec`); `outlined_chip_spans` already label-driven.
4. Footer legend `1–4` → `1–5`; `handle_key` digit guard `'1'..='5'`; `modal.rs` help `1 .. 4` → `1 .. 5`.
5. Tab-bar width: 5 folders (~60 cols). `draw_tab_panel` already truncates cleanly when narrow; verify the wide-mode center panel min width and full-width fallback still show all 5 (consider compacting labels only if a real terminal clips).
**Verify:** `from_digit`/`next`/`prev` tests, `compute_chip_layout` geometry test (5 chips), tab click hit-test, footer render. Build + clippy + fmt.

## Phase 2 — ROCm & Serving tab modules (Actions + Details scaffold)
**Files:** new `ui/tabs/rocm.rs`, `ui/tabs/serving.rs`; retire `ui/tabs/action.rs` (port its list+detail+focus+hit_test). `ui/mod.rs` dispatch.
1. Generalize the Action focus model: `ActionFocus` → `PaneFocus { Actions, Details }` shared by both tabs; per-tab selection cursors (`rocm_sel`, `serving_sel`).
2. Each tab: left **Actions** bento (verbs for that domain) + right **`Details: <verb>`** bento. Initially Details shows the per-verb summary (today's inline detail), grounded in real manager flows.
3. Port `action::hit_test` → per-tab hit-test (verb row → select; Details → activate). Wire into `resolve_mouse` for both tabs.
4. Remove the old `Action` verb list / `verb_action`; map verbs to the existing `KeyAction::Open*` seam.
**Verify:** render tests for both tabs (verbs present, focus highlight, hit-test), `--test-threads=1` gate.

## Phase 3 — De-modal: render managers inline in the Details pane (largest)
**Files:** the ~9 operational managers (`serve_wizard, engine_manager, doctor_manager, update_manager, install_manager, config_manager, services_manager, runtime_manager, logs_view`, + `command_screen`, `onboarding`, `automations_manager` as applicable), `ui/mod.rs`, `app/mod.rs` event loop.
1. Refactor each manager `draw_X(f, area, …)` to **fill the given rect** (via `panel::bento`) instead of `centered_rect`+`draw_popup_frame`. Keep the rect parameter; the caller passes the Details pane rect.
2. ROCm/Serving Details pane renders whichever manager state is `Some` for that tab (activating an Action sets it). The manager's *first screen* is inline.
3. **Gating stays modal**: the approval overlay (`approval.rs`) is unchanged (it must own the screen); any confirm / destructive / "drill into object" step opens a centered modal/approval — i.e. `centered_rect` is retained only on those gating paths.
4. Key routing: when `PaneFocus::Details` and a manager is active, route keys to that manager's existing handler; `←`/`Esc` returns focus to Actions (closing the inline manager or backing out a layer). Preserve the existing manager key handlers; only the *open path* and *draw target* change.
5. Keep these as overlays (global, non-tab): `?` help, theme picker, Esc menu, command palette, approval.
**Approach:** fan out across managers with a workflow (one agent per manager group, disjoint files), each converting draw-target + open path and fixing its tests. Adversarial review pass after.
**Verify:** per-manager render-in-rect tests; event-loop routing tests; no double approval path; `--test-threads=1` gate; clippy `-D warnings`.

## Phase 4 — Live TTFT/TPOT telemetry (daemon/collector)
**Files:** `rocm-dash-collectors/src/vllm_prom.rs`, `rocm-dash-core/src/metrics.rs` (+ `traits.rs`, replay back-compat), collector tests.
1. `vllm_prom.rs`: add histogram keys `vllm:time_to_first_token_seconds_{sum,count}` and `vllm:time_per_output_token_seconds_{sum,count}`; derive average **ttft_ms / tpot_ms** (windowed delta like `gen_tps`, falling back to cumulative `sum/count`).
2. `metrics.rs`: `Instance.ttft_ms: Option<f64>`, `tpot_ms: Option<f64>` with serde defaults (NDJSON replay back-compat, mirroring `tokens_per_watt`). Update `Default`, `traits.rs` mapping, tests.
3. Honest fallback: `None` when unavailable; Observe shows `—` (never fabricated).
**Verify:** collector parse tests (sum/count → ms), replay back-compat test, schema round-trip. Requires daemon rebuild.

## Phase 5 — Observe redesign (AI-serving metrics)
**Files:** `ui/tabs/observe.rs`, `ui/tabs/instances.rs` (+ small `widgets.rs` helpers), reuse `efficiency`/`node_efficiency`.
1. Top band: **two hero panels** — *Node efficiency* (tok/watt, big number + trend) and *Node throughput* (Σ gen_tps, + total power W). Reuse `node_efficiency`/gradient/sparkline.
2. Per-instance table reoriented to AI metrics: model · throughput (tok/s) · **tok/watt** · TTFT · TPOT · power (W) · queue (running/waiting) · kv-cache%. tok/watt surfaced prominently (already in `Instance`).
3. Keep deep hardware/bench detail reachable (retain as a lower section or a drill-in) so nothing is lost; bench remains the source for historical TTFT/TPOT distribution.
4. Color is never the only signal (keep the a11y-of-color discipline); `—` for missing live latency until Phase 4 lands.
**Verify:** Observe render tests (tok/watt visible, heroes present, honest placeholders when empty); `--test-threads=1` gate.

---

## Cross-cutting
- **Verification gate every phase:** `cargo build`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, `cargo test -p rocm-dash-tui -- --test-threads=1` (deterministic; tui.rs flakes under parallelism), plus `-p rocm-dash-collectors`/`-p rocm-dash-core` for Phase 4.
- **Orchestration:** Phases 1, 2, 4, 5 are best done coherently in-hand; Phase 3 fans out across managers via a workflow (disjoint files) + adversarial review.
- **Commit per phase** on `dash-ui-polish-work` (or a fresh branch per your call); each phase is independently shippable.

## Key risks / decisions
- **5-tab width**: labels total ~60 cols; on narrow wide-mode center panels a tab could clip. Mitigation: full-width tab bar fallback already exists; compact labels only if needed.
- **Phase 3 is the heavy lift** (~12 managers, event-loop rewiring). Highest regression risk; gets the workflow + review treatment.
- **Phase 4 touches the daemon** (rebuild required); if vLLM histograms are absent for an engine, latency stays `—`.
- **Scope/order**: recommend executing in order 1→2→3→(4‖5). We can ship 1+2 first (visible IA), then 3, then telemetry+Observe.

## Suggested start
Phase 1 + 2 together (the visible IA + Actions/Details scaffold), reviewed, before the heavier Phase 3 de-modaling. Confirm the ROCm/Serving Action verb lists above and I'll proceed.
