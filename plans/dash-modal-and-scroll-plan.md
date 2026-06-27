# Dash polish: re-modal the managers + real log scrolling

Two changes on `dash-ui-polish-work`, both in `crates/rocm-dash-tui`.

## Issue 1 — Bring the modal back

Today an open manager renders **inline** in the ROCm/Serving Details pane and only
falls back to a centered overlay on other tabs. That split is the bug: open a
manager on ROCm, click over to Observe, and it becomes an orphaned floating box
you must dismiss. Fix: managers always render as a **centered modal**, on every
tab. The Details bento (summary + steps + **Start**) stays exactly as-is; the
**Start** affordance (`PaneActivate` from Detail focus) keeps opening the manager,
which now appears in the modal window instead of in the bento.

- `ui/mod.rs` `draw()`: drop the `Rocm|Serving → detail_rect` branch — always
  `manager_rect = modal::centered_rect(82, 80, 130, 34, body)`. Update comments.
- No input changes needed: `should_pane_back_out` (Esc backs out at root),
  the click-through swallow (`has_open_overlay` in `resolve_mouse`), and the
  footer "Esc back out" hint all already behave correctly for a modal.

## Issue 2 — Mouse wheel / PgUp-PgDn scroll the hovered view, not the Actions list

Two defects: (a) scrolling anywhere on ROCm/Serving fires `Move(±3)`, jumping the
Actions selection and skipping rows; (b) the job console advertises "PgUp/PgDn
scroll" but every manager calls `draw_job_console(.., 0, ..)` — scroll is
**hard-coded to 0 and never implemented**. Scrolling over an open log console just
moves the obscured Actions list underneath.

### Console scroll state (centralized — kills 13 duplicated branches)
All 13 managers contain the byte-identical branch
`if active_job { draw_job_console(f, area, job, 0, theme); return; }`. Lift it into
one place.

- `app/mod.rs` `AppState`: add `console_scroll: u16`, `console_hscroll: u16`
  (init 0); reset both in `close_overlays()`.
- Add `active_job_id(&self) -> Option<&str>` (matches each manager's
  `active_job`) and `has_active_console(&self) -> bool`.
- Add `scroll_console(&mut self, dv: i16, dh: i16)`: clamp ≥ 0; clamp vertical
  against the active job's `output.len()` so you can't scroll into the void.
- `ui/job_console.rs`: `draw_job_console(.., scroll: (u16, u16), ..)` →
  `Paragraph::new(lines).scroll((v, h))`; footer hint mentions wheel + PgUp/PgDn.
- `ui/mod.rs` `draw_active_manager`: before dispatching, if
  `active_job_id()` resolves to a live job, draw the console there with
  `(console_scroll, console_hscroll)` and return.
- Remove the now-dead console branch + unused `draw_job_console` import from each
  of the 13 manager draw fns (input routing via `on_console_key` is untouched).

### Routing scroll to the right place
- New `KeyAction::ScrollConsole(i16, i16)` → `apply_action` calls
  `scroll_console`.
- Key path (event loop): a new arm **after** `should_pane_back_out`, **before**
  the per-manager arms — when `has_active_console()` and the key is
  PageUp/PageDown/Up/Down, consume it as a console scroll (these keys are
  otherwise ignored while a console shows; Ctrl+C/q/Esc/Enter still fall through
  to `on_console_key`).
- Mouse path (`resolve_mouse`, which has `&AppState`): handle
  ScrollUp/Down/Left/Right itself —
  - overlay open + active console → `ScrollConsole` (vertical ×3 lines,
    horizontal ×6 cols per notch; supports H wheel);
  - overlay open, no console (form screen) → `Nothing` (stop moving the hidden list);
  - no overlay, ROCm/Serving → `Move(±1)` **only when the pointer is over the
    Actions column** (`pane::actions_rect` hit-test), else `Nothing`;
  - otherwise delegate to `handle_mouse` (ThemePicker/Detail/Observe), whose
    selection-move notch drops `3 → 1` to stop the skip.
- `ui/tabs/pane.rs`: add `pub fn actions_rect(area) -> Rect` (left column of the
  existing `split_columns`).

## Tests
- Rewrite `managers_render_inline_in_the_details_pane` →
  `managers_render_as_centered_modal`: each manager title renders; it is centered
  (not the two-column inline layout).
- `demo_buffer_dump_*`: replace the "manager in-pane alongside ROCm/Serving
  actions" assertions with centered-modal assertions.
- Update `handle_mouse_routes_scroll_by_modal_and_tab` for the `3 → 1` notch.
- Add unit tests: console scroll via PgDn key and via wheel; wheel over the
  Details pane = `Nothing`; wheel while a form-screen overlay is open = `Nothing`;
  `scroll_console` clamps at 0.

## Gate
`cargo test -p rocm-dash-tui -- --test-threads=1` (the deterministic TUI gate),
plus `cargo clippy -p rocm-dash-tui`. The pre-existing no-GPU `apps/rocm` tui.rs
flake is out of scope.
