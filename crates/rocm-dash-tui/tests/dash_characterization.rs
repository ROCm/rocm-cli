//! Characterization safety-net for the dash TUI (Supergoal Phase 0).
//!
//! Freezes the CURRENT behaviour of `ui::draw` for every existing tab as
//! TestBackend buffer-text assertions, so the later UX-migration phases (which
//! restructure the `ActiveTab` enum and its many literal couplings) are
//! regression-guarded. These tests assert today's reality; later phases update
//! them in lockstep with the enum edits.
//!
//! Ponytail: reuse the existing `TestBackend` → `Terminal` → `ui::draw` →
//! flatten-buffer pattern already used across `src/ui/**` unit tests; no new
//! test framework, no demo NDJSON (which is not committed).

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Snapshot, SystemMetrics};
use rocm_dash_tui::app::{ActiveTab, AppState};
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

/// Build an `AppState` parked on `tab` with the synthetic snapshot applied.
fn state_on(tab: ActiveTab) -> AppState {
    let mut s = AppState::new("test-connect".into(), "default-dark".into());
    s.active_tab = tab;
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
    for label in ["Overview", "Hardware", "Instances", "Bench", "Chat"] {
        assert!(out.contains(label), "tab bar missing {label:?}: {out:?}");
    }
}

#[test]
fn overview_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Overview), 160, 44);
    assert_tab_bar(&out);
    assert!(out.contains("Host"), "overview missing Host block: {out:?}");
}

#[test]
fn hardware_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Hardware), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("GPU0") && out.contains("util"),
        "hardware missing GPU detail: {out:?}"
    );
}

#[test]
fn instances_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Instances), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("Instances"),
        "instances missing Instances block: {out:?}"
    );
}

#[test]
fn bench_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Bench), 160, 44);
    assert_tab_bar(&out);
    assert!(out.contains("Bench"), "bench missing Bench block: {out:?}");
}

#[test]
fn chat_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Chat), 160, 44);
    assert_tab_bar(&out);
    assert!(out.contains("Chat"), "chat missing Chat block: {out:?}");
}

#[test]
fn every_tab_survives_squeezed_height() {
    // The body rect can collapse to 0–1 inner rows on a short terminal; assert
    // no tab panics when squeezed (the historical ActiveTab footgun).
    for tab in [
        ActiveTab::Overview,
        ActiveTab::Hardware,
        ActiveTab::Instances,
        ActiveTab::Bench,
        ActiveTab::Chat,
    ] {
        let mut s = state_on(tab);
        for h in [1u16, 2, 3, 5, 8] {
            let _ = render(&mut s, 80, h);
        }
    }
}
