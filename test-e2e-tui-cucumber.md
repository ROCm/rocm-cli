# WIP: Black-box E2E cucumber testing of the dash TUI

**Stage:** 9-done (PR #114 merged into main, squash `cfae8d3`)
**Pipeline:** standard
**Branch:** test-e2e-tui-cucumber
**Last Updated:** 2026-07-28
**Pre-PR-check:** passed — pre-pr-reviewer (opencode), 2026-07-23

**Token Usage:** in=81 out=19560 cache_create=1256181 cache_read=3594075 calls=45

---

## Problem

The E2E cucumber suite ([[test-add-e2e-robot-framework]]) drives the real `rocm`
binary black-box, but the interactive **dash TUI** is currently treated as
**untestable black-box** — it's only covered by in-crate journey tests
(`crates/rocm-dash-tui/tests/dash_journeys.rs`, commit `f315ba0`) that poke the
public `AppState` API + ratatui `TestBackend`. Two known bugs are documented as
untestable-black-box because of this: **EAI-7220** (TUI wrong-port, fixed via a
dash unit test, not e2e) and **EAI-7222** (privacy notice only shown in the TUI —
`chat_steps.rs:144`).

**The premise is wrong.** A TUI is NOT untestable black-box. A terminal app can
be driven exactly as a user would — spawn `rocm dash` under a **pseudo-terminal
(PTY)**, feed keystrokes, and assert on the rendered screen (the ANSI/character
grid). This is standard practice (tmux capture-pane, `expectrl`, `portable-pty`,
vhs, `insta` snapshots of terminal output). Closing this gap would:
- make the fourth north-star journey (`dash`) a TRUE black-box E2E, not a
  crate-internal seam test (goal #6 of the parent WIP);
- let EAI-7220 / EAI-7222 have real e2e scenarios instead of "documented gap";
- catch integration failures the `AppState` seam tests can't (actual terminal
  init, crossterm event wiring, real render, real key handling end to end).

## Solution (high-level — to be designed, not yet decided)

Add a PTY-backed driver to the cucumber harness so `.feature` scenarios can:
1. launch `rocm dash` (and `rocm chat`, which routes into the dash chat) inside a
   PTY with a fixed size (e.g. 80x24) and deterministic env;
2. send input (keystrokes, tab navigation, `/` commands, text + Enter);
3. read the emulated screen and assert on visible text / state transitions;
4. quit cleanly and assert exit behavior.

**Key design questions (open):**
- **Rust-native vs external.** Prefer Rust to honor EAI-7164 (one toolchain):
  `portable-pty` (wezterm) to spawn + `vt100`/`vte` to parse the screen grid, or
  `expectrl` for expect-style waits. Avoid a Python/tmux dependency if possible.
- **Determinism.** TUIs are timing- and size-sensitive. Need: fixed PTY size,
  `TERM` pinned, disable animations/spinners if any, a robust "wait until screen
  contains X" poll (not fixed sleeps), and the existing offline chat mock
  (`--dev-chat-mock`, main.rs:226) so chat scenarios don't need a live model.
- **Snapshot vs assertion.** State-transition assertions (like dash_journeys.rs)
  are the reliable signal; full-screen snapshots (`insta`) are brittle across
  ratatui/terminal versions. Likely: assert on presence of key strings + cursor/
  tab state, keep any snapshot as a no-panic/smoke check.
- **CI portability.** Must run on the mock platform (hosted ubuntu/win/mac) so
  it's a BLOCKING gate, not GPU-only. Windows PTY behavior (ConPTY) differs —
  verify `portable-pty` handles it, or gate `@requires-os` if not.
- **Isolation.** Reuse the per-scenario temp `ROCM_CLI_DATA_DIR` + planted
  service-record pattern already in the suite so dash reads deterministic state.

## Research findings (2026-07-14) — approach is viable, both halves confirmed

### Product side already has the determinism knobs we need (code-traced)
- **`interactive_terminal()` = `stdin().is_terminal() && stdout().is_terminal()`**
  (`crates/rocm-core/src/lib.rs:1221`). Bare `rocm` opens the TUI only when BOTH
  are a TTY. A **PTY satisfies both; a plain piped stdin does NOT** — this is
  exactly why PTY (not `echo … | rocm`) is the required approach.
- **`rocm dash --demo`** (`main.rs:398`, `dash::run` at `dash.rs:218`) writes a
  *deterministic synthetic session* and replays it → populated dashboard with
  **no GPU and no daemon**. Perfect for a mock-tier, cross-platform blocking gate.
  `--replay <ndjson>` does the same from a fixed file.
- **`--dev-chat-mock`** (`main.rs:226`) threads to `MockAgentClient::with_tool_call`
  (`crates/rocm-dash-tui/src/app/mod.rs:1542`) which returns a **fixed reply**
  string (`"GPU-2 is running hot: 87% util, 71°C, drawing 250 W …"`, tool
  `gpu_status`). So a chat exchange has a KNOWN assertable output — no live model.
- Launch surface (`apps/rocm/src/dash.rs`): bare `rocm` → `run_launcher`; `rocm
  dash` → `run` (full dash, bypasses launcher); `rocm chat` → `run_chat` (Chat
  tab). All good PTY targets.
- **These flags mean EAI-7222 (privacy notice) and a chat turn are both
  deterministically drivable today** — the "documented gap" was never a hard wall.

### Driver stack (Rust-native, honors EAI-7164 one-toolchain)
- **`portable-pty` 0.9.0 (wezterm, Feb 2025)** — spawn the real binary under a
  PTY; picks openpty on Unix / ConPTY on Windows at runtime. Standard choice.
- **`vt100` 0.16.2 (~mid-2025)** — parse the PTY byte stream into a screen grid
  (`screen.contents()`, `screen.cell(row,col)`, cursor). Forks `vt100-ctt` /
  `vt100-rust-patched` exist if we hit gaps. (`vte` = lower-level, only if needed.)
- **`expectrl` 0.9 / `rexpect` 0.7** — expect-style wait/match wrappers; pair
  with `vt100` for cell assertions. Optional convenience over hand-rolled polling.
- **ratatui `TestBackend`** = the WHITE-box in-process path the current
  `dash_journeys.rs` already uses. Keep it for unit-level; PTY is the black-box
  complement, not a replacement.

### Canonical pattern (sketch)
```rust
let pty = native_pty_system().openpty(PtySize{rows:24,cols:80,..})?;
let mut cmd = CommandBuilder::new("rocm"); cmd.arg("dash"); cmd.arg("--demo");
cmd.env("TERM","xterm-256color");
let mut child  = pty.slave.spawn_command(cmd)?;
let mut writer = pty.master.take_writer()?;
let mut reader = pty.master.try_clone_reader()?;
let mut parser = vt100::Parser::new(24,80,0);
writer.write_all(b"\t")?;      // Tab; arrows = \x1b[A/B/C/D; Enter = \r
// poll-until-contains with a deadline (NOT fixed sleep):
loop { let n = reader.read(&mut buf)?; parser.process(&buf[..n]);
       if parser.screen().contents().contains("Telemetry") { break; } }
```

### Determinism pitfalls (from research)
- **Poll-until-contains with a deadline, never `sleep(fixed)`** — the single
  biggest flakiness fix. Give each assertion a visible screen marker to wait on.
- **Pin `TERM` + PTY size (80x24)**; clear locale-dependent env for stable escapes.
- **Sanitize spinners/timers/live telemetry timestamps** before asserting (regex
  strip) — `--demo` already removes GPU/daemon variance, big help.
- **Avoid full-grid `insta` snapshots** (brittle to any layout tweak); prefer
  targeted `contains` / cell assertions; reserve snapshots for stable layouts.

### Windows / ConPTY caveat
- `portable-pty` DOES support ConPTY, so the same test *can* run on Windows CI —
  but crossterm under ConPTY can't use WinAPI for **input / cursor-position
  queries / resize**; outdated ConPTY on a runner has crashed TUIs on exit
  (wezterm #7774). **Likely call: gate dash PTY e2e `@requires-os:linux` (+ mac)
  and keep `TestBackend` for Windows coverage** — decide after a Windows spike.

## ✅ SPIKE PROVEN BY HAND (2026-07-14) — the "untestable" premise is refuted

Built a throwaway `portable-pty` 0.9 + `vt100` 0.16 driver
(`workspace/tui-spike/`, worktree `test-e2e-tui-cucumber`) and drove the REAL
release `rocm` binary black-box. Full drive→assert→input→redraw→quit loop worked:
- **Spawn** `rocm dash --demo` under an 80×24 PTY, `TERM=xterm-256color`,
  isolated `ROCM_CLI_DATA_DIR` → dash renders.
- **Assert** via `vt100` screen grid: `screen.contents()` contained `rocm.ai`
  (opened marker) — poll-until-contains, no fixed sleep.
- **Send** Tab (`\t`) → screen redrew to the connected dashboard: tab bar
  `1 Home / ● ROCm / 3 Serving / 4 Observe / 5 Chat`, ROCm-actions panel, footer
  keybindings (`Tab next  1–5 jump  j/k select  Enter open …`).
- **Quit** `q` → `child.try_wait()` reported clean exit; terminal restored.
- Determinism held: `--demo` (replay session, no GPU/daemon) made the render
  stable enough to assert on fixed strings. Result: `opened=true exited=true`.

Takeaways for the real harness:
- Stable markers observed: `rocm.ai`, tab labels `Home/Serving/Observe/Chat`,
  footer hints — all good assertion anchors. The header shows `replay:…` /
  `starting`→`connected`, so wait for `connected` (or a tab label) before asserting.
- portable-pty needs an ABSOLUTE binary path (relative `./target/...` failed to
  spawn) — harness must resolve `ROCM_CLI_BINARY`/build path to absolute.
- Spike lives OUTSIDE the cargo workspace (own `[workspace]` table) so it doesn't
  perturb the repo build. It's throwaway — real driver goes in the e2e crate.

## Scenarios (DRAFT — activate bdd-scenarios skill before finalizing)

```gherkin
Scenario: The dashboard opens to live telemetry
  Given a managed environment with GPU telemetry available
  When the user opens the dashboard
  Then the dashboard displays the telemetry view

Scenario: The user navigates between dashboard tabs
  Given the dashboard is open
  When the user moves to the next tab
  Then the newly selected tab is shown

Scenario: The chat privacy notice is shown before the first message (EAI-7222)
  Given the dashboard chat is open for the first time
  When the user views the chat tab
  Then a privacy notice is displayed before any message is sent

Scenario: A chat message receives a reply in the TUI
  Given the dashboard chat is open with the offline mock
  When the user submits a message
  Then a reply appears in the conversation

Scenario: The user quits the dashboard
  Given the dashboard is open
  When the user quits
  Then the process exits cleanly and the terminal is restored
```

## Implementation Steps

### Completed ✅
- ✅ **PTY spike proven by hand** (see section above): portable-pty + vt100 drives
  `rocm dash --demo` black-box — open, assert screen text, send Tab, see redraw,
  quit cleanly. Determinism via `--demo`. `workspace/tui-spike/` (throwaway).

### Landed on branch ✅ (HEAD `9adc56c`, 2026-07-20)
- ✅ Driver stack decided: **portable-pty + vt100** (no expectrl). Real driver in
  `tests/e2e-cucumber/tests/e2e/tui_driver.rs` (`TuiSession`) — supersedes the
  throwaway `workspace/tui-spike/`.
- ✅ Driver integrated into `E2eWorld` + dash/chat step defs
  (`tests/e2e/dash_steps.rs`, `tests/e2e/chat_steps.rs`).
- ✅ `features/dash.feature` + `features/chat.feature` written.
- ✅ Determinism knobs implemented: pinned 24×80 (40×120 detail) PTY, `TERM`/
  `COLUMNS`/`LINES`, poll-until-contains with deadline (`wait_for_screen`),
  bounded drain, robust Drop-based cleanup (commit `9adc56c` hardened this).
- Commits: `cdbe4d1` drive dash black-box over PTY, `4ba48c0` managed TUI
  journeys, `9adc56c` harden PTY session cleanup.

### Todo ✅ (all landed with PR #114)
- ✅ Wired into CI (6 follow-up commits on remote: isolate env, reap PTY setup failures, preserve diagnostics, address review feedback).
- ✅ EAI-7220 / EAI-7222: now real e2e scenarios in `dash.feature` / `chat.feature`.
- ✅ Deleted throwaway `workspace/tui-spike/` (2026-07-20) — fully absorbed into `tui_driver.rs`.
- ✅ Merged to main 2026-07-28.

## Next Steps

Task complete. The PTY-backed dash TUI harness (`tui_driver.rs`, dash/chat features) shipped in PR #114 (merged to main, 2026-07-28). Post-merge cleanup available: delete remote branch, remove worktree.

## Blockers / Open Questions

None. Task complete.

## Notes

- dash launches via `apps/rocm/src/main.rs` (`mod dash`, `dash::run_launcher`,
  `rocm dash`/`rocm chat` bypass the launcher). Offline chat mock flag exists
  (`--dev-chat-mock`, main.rs:226) — key enabler for deterministic chat scenarios.
- dash-tui crate uses **ratatui 0.30 + crossterm 0.28** (event-stream) — a real
  terminal app, so PTY driving is the correct black-box approach.
- Existing in-crate coverage to complement (not duplicate):
  `crates/rocm-dash-tui/tests/dash_journeys.rs` (5 tests, AppState + TestBackend)
  and `dash_characterization.rs` (render-only).
- Honors EAI-7164 (single Rust toolchain) → prefer a Rust-native PTY driver.

## Worktree Context

**Worktree directory**: `~/Developer/rocm-cli-wt/test-e2e-tui-cucumber` (active).
- Recreate with: `create_worktree.sh test-e2e-tui-cucumber`

## Work Log

### 2026-07-28 (continued)

- Checked branch state: local `cf913db` stale; remote has 6 review-feedback commits (isolate env, reap PTY setup, preserve diagnostics, address review, exit privacy chat cleanly).
- Confirmed PR #114 merged and approved (`cfae8d3` on main). Task complete.
- Rebased local onto origin/main: three commits hit `add/add` conflicts (tui_driver.rs, dash_steps.rs, dash.feature already on main in review-hardened form). Nothing local-only to keep.
- Reset `test-e2e-tui-cucumber` branch to `origin/main` (`d2662d5`). Worktree clean. Post-merge cleanup available when ready.

### 2026-07-20

- Pulled branch: fast-forward `74ee17d` → `9adc56c` (3 new commits). The real PTY
  harness the WIP had only *planned* is now on the branch — `tui_driver.rs`
  (`TuiSession`, 364 lines), `dash_steps.rs`, `dash.feature`, `chat.feature`.
- Stage moved 2 → 6-implementing. Todo list rewritten to reflect landed vs left.
- Compared the throwaway spike (`workspace/tui-spike/src/main.rs`) against the
  landed driver — spike's approach fully absorbed and hardened.
- (earlier) Session start: reviewed WIP status, oriented to spike-proven stage.

### 2026-07-14

- Created this WIP. User pushed back on the parent WIP's claim that the dash TUI
  is "untestable black-box" — correctly, since TUIs are PTY-testable.
- Captured the problem, a candidate PTY-driver solution, draft scenarios, and the
  open design questions (driver choice, Windows ConPTY, determinism).
- **Did a research pass (see Research findings section):**
  - Code-traced the product side: `interactive_terminal()` needs BOTH stdin+stdout
    TTY (→ PTY required, not pipes); `rocm dash --demo` gives a deterministic no-GPU/
    no-daemon session; `--dev-chat-mock` returns a FIXED reply string. So the dash is
    already very driveable deterministically, and EAI-7222 is not a real hard gap.
  - Web-researched the Rust PTY stack: `portable-pty` 0.9 + `vt100` 0.16 is the
    standard black-box combo; `expectrl`/`rexpect` optional; `TestBackend` is the
    white-box path the current journey tests use (complement, don't replace).
  - Windows/ConPTY is supported by portable-pty but crossterm input/cursor is
    flaky under it → likely gate dash PTY e2e to linux(+mac).
  - Determinism largely solved by `--demo` + `--dev-chat-mock` + poll-until-contains.
- Next: bdd-scenarios/e2e-testing skills to refine scenarios, then a minimal
  portable-pty spike (`rocm dash --demo` → see "Telemetry"/known string → send
  Tab → quit) before any harness plumbing.
