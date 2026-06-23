//! Characterization safety-net for the dash TUI (Supergoal Phase 0, updated P3).
//!
//! Freezes `ui::draw` behaviour for every tab in the current 4-tab IA
//! (Home / Action / Observe / Chat) as TestBackend buffer-text assertions, plus
//! a squeezed-height no-panic sweep. Phase 0 created this against the original
//! 5-tab model; Phase 3 folded the telemetry tabs into Observe and updated
//! these assertions in lockstep with the enum collapse.
//!
//! Ponytail: reuse the existing `TestBackend` → `Terminal` → `ui::draw` →
//! flatten-buffer pattern; no new test framework, no demo NDJSON.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Snapshot, SystemMetrics};
use rocm_dash_tui::app::{ActiveTab, AppState, ConnState};
use rocm_dash_tui::ui;

/// A synthetic single-GPU snapshot so each tab body has real content to paint.
fn synthetic_snapshot() -> Snapshot {
    Snapshot {
        host: SystemMetrics {
            cpu_overall_pct: 37.0,
            cpu_per_core_pct: vec![20.0, 40.0, 60.0, 80.0],
            memory_used_mb: 32_000,
            memory_total_mb: 128_000,
            disk_read_bps: 1_200_000,
            net_rx_bps: 2_500_000,
            ..Default::default()
        },
        gpus: vec![GpuMetrics {
            device_id: "GPU0".into(),
            vram_used_mb: 40_000,
            vram_total_mb: 192_000,
            gpu_utilization_pct: 72.0,
            temperature_c: 58.0,
            power_w: 420.0,
            clock_mhz: Some(2100.0),
        }],
        gpu_system_info: Some(GpuSystemInfo {
            gpu_model: "Instinct MI355X".into(),
            physical_gpu_count: 1,
            logical_gpu_count: 1,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build a connected `AppState` parked on `tab` with the synthetic snapshot.
fn state_on(tab: ActiveTab) -> AppState {
    let mut s = AppState::new("test-connect".into(), "default-dark".into());
    s.active_tab = tab;
    s.conn = ConnState::Connected {
        host: "localhost".into(),
        version: "1.0".into(),
    };
    s.latest = Some(synthetic_snapshot());
    s
}

/// Render the full `ui::draw` chrome to a flat buffer string at `cols`×`rows`.
fn render(state: &mut AppState, cols: u16, rows: u16) -> String {
    let backend = TestBackend::new(cols, rows);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| ui::draw(f, state)).unwrap();
    term.backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect()
}

/// The tab bar always paints every tab label; assert it is present so the
/// chrome itself is characterized once.
fn assert_tab_bar(out: &str) {
    for label in ["Home", "Action", "Observe", "Chat"] {
        assert!(out.contains(label), "tab bar missing {label:?}: {out:?}");
    }
}

#[test]
fn home_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Home), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("GPU UTILIZATION"),
        "home hero missing: {out:?}"
    );
}

#[test]
fn action_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Action), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("Serve a model"),
        "action verbs missing: {out:?}"
    );
}

#[test]
fn observe_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Observe), 160, 44);
    assert_tab_bar(&out);
    // Observe folds the former Overview/Hardware (host telemetry), Instances and
    // Bench surfaces into one tab.
    assert!(
        out.contains("CPU") && out.contains("Instances") && out.contains("Bench"),
        "observe folded surfaces missing: {out:?}"
    );
}

#[test]
fn chat_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Chat), 160, 44);
    assert_tab_bar(&out);
    assert!(out.contains("Chat"), "chat missing Chat block: {out:?}");
}

#[test]
fn wide_layout_shows_logs_or_context_dock_not_composer() {
    // Operational tab → LOGS dock; the dock must never be a chat composer.
    let observe = render(&mut state_on(ActiveTab::Observe), 200, 50);
    assert!(
        observe.contains("LOGS"),
        "wide Observe missing LOGS dock: {observe:?}"
    );
    // Home → CONTEXT rail (RUNNING SERVICES section).
    let home = render(&mut state_on(ActiveTab::Home), 200, 50);
    assert!(
        home.contains("CONTEXT") && home.contains("RUNNING SERVICES"),
        "wide Home missing CONTEXT rail: {home:?}"
    );
    // Neither dock leaks the chat composer.
    assert!(
        !observe.contains("press i to type"),
        "composer in Observe dock"
    );
    assert!(!home.contains("press i to type"), "composer in Home dock");
}

#[test]
fn narrow_layout_is_single_column_no_dock() {
    // Below the 180×45 threshold there is no LOGS/CONTEXT dock (fallback path).
    let out = render(&mut state_on(ActiveTab::Observe), 160, 44);
    assert!(
        !out.contains("LOGS"),
        "narrow layout must not show the dock: {out:?}"
    );
    assert!(
        !out.contains("CONTEXT"),
        "narrow layout must not show CONTEXT dock"
    );
}

// --- Phase 7: empty / loading / error states, honesty, a11y of color ---

#[test]
fn observe_empty_state_shows_placeholders() {
    // Connected but with no instances → honest empty placeholder, no banner.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connected {
        host: "h".into(),
        version: "1".into(),
    };
    s.latest = Some(synthetic_snapshot());
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("no instances"),
        "empty instances placeholder: {out:?}"
    );
}

#[test]
fn loading_state_connecting_banner() {
    // Connecting (no snapshot) → header shows the loading status + demo banner.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connecting;
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("connecting"),
        "loading status missing: {out:?}"
    );
    assert!(
        out.contains("demo data"),
        "demo banner expected while loading"
    );
}

#[test]
fn disconnected_banner_present() {
    // Disconnected → error status in header + demo banner on Observe.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Disconnected {
        reason: "daemon gone".into(),
    };
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("disconnected"),
        "error status missing: {out:?}"
    );
    assert!(
        out.contains("demo data"),
        "demo banner expected when disconnected"
    );
}

#[test]
fn honesty_demo_banner_absent_when_connected_with_telemetry() {
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connected {
        host: "h".into(),
        version: "1".into(),
    };
    s.latest = Some(synthetic_snapshot());
    let out = render(&mut s, 160, 44);
    assert!(
        !out.contains("demo data"),
        "banner must be hidden when live: {out:?}"
    );
}

#[test]
fn a11y_status_carries_text_label_not_color_only() {
    // Connection status is conveyed in words, not by color alone.
    let mut connected = state_on(ActiveTab::Home);
    let out = render(&mut connected, 160, 44);
    assert!(
        out.contains("connected"),
        "connected text label missing: {out:?}"
    );

    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Home;
    s.conn = ConnState::Disconnected { reason: "x".into() };
    let dis = render(&mut s, 160, 44);
    assert!(
        dis.contains("disconnected"),
        "disconnected text label missing: {dis:?}"
    );
}

#[test]
fn control_legend_is_on_the_bottom_row_not_the_top() {
    // The footer keyboard legend must render on the LAST row, never inside the
    // body near the top (regression: footer rect once collapsed onto the body).
    let cols = 160u16;
    let rows = 44u16;
    let mut s = state_on(ActiveTab::Home);
    let backend = TestBackend::new(cols, rows);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| ui::draw(f, &mut s)).unwrap();
    let buf = term.backend().buffer().clone();

    let row_text = |y: u16| -> String {
        (0..cols)
            .map(|x| {
                buf.cell((x, y))
                    .map_or(" ", ratatui::buffer::Cell::symbol)
                    .to_string()
            })
            .collect()
    };
    // "quit" (legend tail) is on the bottom row.
    assert!(
        row_text(rows - 1).contains("quit"),
        "legend must be on the bottom row: {:?}",
        row_text(rows - 1)
    );
    // The top rows (header band) must NOT carry the body-level legend chips like
    // "select"/"jump" — only the small chrome hint may live up top.
    let top = (0..6).map(row_text).collect::<String>();
    assert!(
        !top.contains(" jump "),
        "body legend leaked to the top: {top:?}"
    );
}

#[test]
fn every_tab_survives_squeezed_height() {
    // The body rect can collapse to 0–1 inner rows on a short terminal; assert
    // no tab panics when squeezed (the historical ActiveTab footgun).
    for tab in [
        ActiveTab::Home,
        ActiveTab::Action,
        ActiveTab::Observe,
        ActiveTab::Chat,
    ] {
        let mut s = state_on(tab);
        for h in [1u16, 2, 3, 5, 8] {
            let _ = render(&mut s, 80, h);
        }
    }
}
